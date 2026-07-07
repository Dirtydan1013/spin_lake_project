"""Unified per-rank chunked HDF5 storage for the production MPI drivers.

Layout (flat — one file per rank, no per-chunk subdirectories)::

    <run_dir>/rank{r}.h5
        ├── attrs                : shared run metadata
        ├── chunk0, chunk1, ...  : one group per flushed bin/block
        └── final_config         : warm-start configuration (spin state /
                                    operator strings + RNG), overwritten each flush

``checkpoint`` is the merged bin==flush size: every ``checkpoint`` samples the
driver computes one bin (profile / SSE: a mean over the samples; trajectory
engines: the raw per-trajectory arrays) and appends it as the next ``chunk{i}``
group in this rank's single file.  A crash loses at most the chunk being
written (the file is flushed after every chunk and after the final config).

This replaces the older ``rank{r}/chunk{c}.h5`` nested layout and the separate
``<config>_configs/rank{r}.h5`` warm-start files: chunks and the warm-start
configuration now live together in one file per rank.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np

_COMPRESS_MIN_BYTES = 1 << 20  # gzip chunk datasets larger than 1 MiB
# Warm-start op strings are always highly compressible (op_types is just a few
# distinct codes), so gzip them at a much lower threshold than bulk chunk data
# — this catches the work/string engines' ~0.8 MB configs that would otherwise
# sit under the 1 MiB bar.  Kept above tiny scalars (compression needs a
# chunked, non-scalar layout).
_CONFIG_COMPRESS_MIN_BYTES = 1 << 14  # 16 KiB


def rank_file(run_dir, rank: int) -> Path:
    """Path of the single HDF5 file owned by ``rank`` inside ``run_dir``."""
    return Path(run_dir) / f"rank{int(rank)}.h5"


def config_dir(run_dir) -> Path:
    """Warm-start config subdirectory: ``<run_dir>/configs``.

    Configs live in their own subdir (one ``rank{r}.h5`` per rank) rather than
    inside the data files, so the whole set can be reclaimed with a single
    ``rm -rf <run_dir>/configs`` once a run no longer needs to be warm-started
    — without touching the observable chunk files or needing an h5repack.
    """
    return Path(run_dir) / "configs"


class RankChunkWriter:
    """Owns one ``<run_dir>/rank{r}.h5`` and appends chunk / final-config groups.

    A fresh file is created per run (mode ``'w'``).  Use as a context manager or
    call :meth:`close` explicitly.
    """

    def __init__(self, run_dir, rank: int, run_attrs: dict | None = None):
        import h5py

        d = Path(run_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.path = rank_file(run_dir, rank)
        self.rank = int(rank)
        self._h5 = h5py.File(self.path, "w")
        self._h5.attrs["rank"] = int(rank)
        for key, value in (run_attrs or {}).items():
            self._h5.attrs[key] = value
        self._h5.flush()

    def write_chunk(self, idx: int, datasets: dict,
                    attrs: dict | None = None) -> None:
        """Append one bin/block as group ``chunk{idx}`` and flush to disk."""
        g = self._h5.create_group(f"chunk{int(idx)}")
        for key, value in datasets.items():
            arr = np.ascontiguousarray(value)
            if arr.nbytes > _COMPRESS_MIN_BYTES:
                g.create_dataset(key, data=arr, compression="gzip",
                                 compression_opts=4)
            else:
                g.create_dataset(key, data=arr)
        g.attrs["chunk"] = int(idx)
        for key, value in (attrs or {}).items():
            g.attrs[key] = value
        self._h5.flush()

    def write_final_config(self, datasets: dict,
                           attrs: dict | None = None) -> None:
        """(Re)write the ``final_config`` group with warm-start data.

        String/bytes values become group attributes (e.g. the serialised RNG
        state); everything else is stored as a dataset.
        """
        if "final_config" in self._h5:
            del self._h5["final_config"]
        g = self._h5.create_group("final_config")
        for key, value in datasets.items():
            if isinstance(value, (str, bytes)):
                g.attrs[key] = value
            else:
                # Warm-start op strings are long (M_total) but highly
                # compressible (op_types is just a few codes); gzip them — h5py
                # decompresses transparently on load, so warm start is
                # unaffected.  Non-scalar + above the small threshold only
                # (compression requires a chunked layout).
                arr = np.ascontiguousarray(value)
                if arr.ndim >= 1 and arr.nbytes > _CONFIG_COMPRESS_MIN_BYTES:
                    g.create_dataset(key, data=arr, compression="gzip",
                                     compression_opts=4)
                else:
                    g.create_dataset(key, data=arr)
        for key, value in (attrs or {}).items():
            g.attrs[key] = value
        self._h5.flush()

    def close(self) -> None:
        try:
            self._h5.close()
        except Exception:
            pass

    def __enter__(self) -> "RankChunkWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _saved_ranks(run_dir) -> list[int]:
    files = glob.glob(str(Path(run_dir) / "rank*.h5"))
    ranks = []
    for f in files:
        m = re.search(r"rank(\d+)\.h5$", f)
        if m:
            ranks.append(int(m.group(1)))
    return sorted(ranks)


def _resolve_config_dir(run_dir) -> Path:
    """Where a previous run's configs live.

    Prefers a ``configs/`` subdirectory (current layout); falls back to
    ``run_dir`` itself so a directory of ``rank{r}.h5`` config files — the
    work/string engines' ``config_out`` dir, or any legacy layout — still
    loads.
    """
    sub = Path(run_dir) / "configs"
    if _saved_ranks(sub):
        return sub
    return Path(run_dir)


def load_warm_config(run_dir, rank: int, verbose: bool = False) -> dict | None:
    """Load a rank's warm-start configuration for a later run.

    Resolves ``<run_dir>/configs/`` (current layout) or ``<run_dir>`` itself,
    then reads the ``final_config`` group of that dir's ``rank{r}.h5`` (falling
    back to legacy top-level datasets).  If the new run has more ranks than were
    saved, ranks wrap around (``rank % n_saved``) with a note; the per-rank
    seeds differ, so the chains decorrelate after a short while.

    Returns ``{dataset_name: ndarray, ..., 'attrs': dict}`` or ``None`` when no
    rank files are found.
    """
    import h5py

    base = _resolve_config_dir(run_dir)
    ranks = _saved_ranks(base)
    if not ranks:
        return None
    use = ranks[int(rank) % len(ranks)]
    if verbose and use != int(rank):
        print(f"[warm-start] rank {rank}: reusing saved rank{use}.h5 "
              f"({len(ranks)} saved ranks)", flush=True)
    out: dict = {}
    with h5py.File(rank_file(base, use), "r") as f:
        src = f["final_config"] if "final_config" in f else f
        for key in src.keys():
            out[key] = src[key][:]
        out["attrs"] = dict(src.attrs)
    return out


def check_config_compat(cfg: dict, expect: dict, label: str) -> None:
    """Raise with a clear message when saved attrs conflict with the run."""
    attrs = cfg.get("attrs", {})
    for key, want in expect.items():
        have = attrs.get(key)
        if have is not None and have != want:
            raise ValueError(
                f"[warm-start] {label}: saved config has {key}={have!r} but this "
                f"run uses {key}={want!r} — refusing to load an incompatible "
                f"configuration")


def iter_rank_chunks(run_dir, burn_in_fraction: float = 0.0):
    """Yield ``(rank, chunk_idx, group)`` for every chunk across all rank files.

    Groups are visited in ascending ``(rank, chunk_idx)`` order.  When
    ``burn_in_fraction > 0``, the first that fraction of each rank's chunks is
    skipped (per-rank equilibration burn-in).  The caller must copy any dataset
    it needs out of ``group`` before the next iteration (the file closes when
    the rank changes).
    """
    import h5py

    for r in _saved_ranks(run_dir):
        with h5py.File(rank_file(run_dir, r), "r") as f:
            idxs = sorted(int(m.group(1))
                          for name in f.keys()
                          for m in [re.fullmatch(r"chunk(\d+)", name)] if m)
            if burn_in_fraction > 0.0:
                idxs = idxs[int(len(idxs) * burn_in_fraction):]
            for i in idxs:
                yield r, i, f[f"chunk{i}"]
