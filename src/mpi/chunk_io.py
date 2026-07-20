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
import hashlib
import re
import socket
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


def _attr_values_equal(have, want) -> bool:
    """Compare scalar/array HDF5 attributes without ambiguous NumPy truth."""
    try:
        return bool(np.array_equal(np.asarray(have), np.asarray(want)))
    except (TypeError, ValueError):
        return have == want


def array_fingerprint(value) -> str:
    """Stable SHA-256 identity for a numeric model array (or ``None``).

    Shape and dtype are included so equal raw bytes with different geometry
    cannot be mistaken for the same checkpoint model.
    """
    if value is None:
        return "none"
    arr = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(arr.dtype.str.encode("ascii"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def compact_operator_checkpoint(types, sites) -> tuple[np.ndarray, np.ndarray]:
    """Losslessly narrow QAQMC operator arrays for continuation storage."""
    type_values = np.asarray(types)
    site_values = np.asarray(sites)
    if type_values.shape != site_values.shape:
        raise ValueError("operator type/site checkpoint shapes differ")
    if type_values.size:
        type_min = int(type_values.min())
        type_max = int(type_values.max())
        site_min = int(site_values.min())
        site_max = int(site_values.max())
    else:
        type_min = type_max = site_min = site_max = 0
    if type_min < np.iinfo(np.int8).min or type_max > np.iinfo(np.int8).max:
        raise ValueError("operator type checkpoint does not fit int8")
    if site_min < 0 or site_max > np.iinfo(np.uint32).max:
        raise ValueError("operator site checkpoint must fit uint32")
    if site_max <= np.iinfo(np.uint8).max:
        site_dtype = np.uint8
    elif site_max <= np.iinfo(np.uint16).max:
        site_dtype = np.uint16
    else:
        site_dtype = np.uint32
    return (
        np.ascontiguousarray(type_values, dtype=np.int8),
        np.ascontiguousarray(site_values, dtype=site_dtype),
    )


class RankChunkWriter:
    """Owns one ``<run_dir>/rank{r}.h5`` and appends chunk / final-config groups.

    A fresh file is created per run by default.  ``resume=True`` opens an
    existing file in append mode, validates immutable run attributes and
    removes only uncommitted ``_pending_chunk*`` transactions.
    """

    def __init__(self, run_dir, rank: int, run_attrs: dict | None = None,
                 resume: bool = False):
        import h5py

        d = Path(run_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.path = rank_file(run_dir, rank)
        self.rank = int(rank)
        existed = self.path.exists()
        self._h5 = h5py.File(self.path, "a" if resume else "w")
        if resume and existed:
            saved_rank = int(self._h5.attrs.get("rank", rank))
            if saved_rank != int(rank):
                self._h5.close()
                raise ValueError(
                    f"checkpoint rank mismatch: file has rank={saved_rank}, "
                    f"requested rank={rank}")
            for name in list(self._h5.keys()):
                if name.startswith("_pending_chunk"):
                    del self._h5[name]
            for key, value in (run_attrs or {}).items():
                if key in self._h5.attrs and not _attr_values_equal(
                        self._h5.attrs[key], value):
                    have = self._h5.attrs[key]
                    self._h5.close()
                    raise ValueError(
                        f"checkpoint attribute mismatch for {key}: "
                        f"saved={have!r}, requested={value!r}")
                self._h5.attrs[key] = value
        else:
            self._h5.attrs["rank"] = int(rank)
            for key, value in (run_attrs or {}).items():
                self._h5.attrs[key] = value
        self._h5.flush()

    def write_chunk(self, idx: int, datasets: dict,
                    attrs: dict | None = None,
                    checkpoint_datasets: dict | None = None,
                    checkpoint_attrs: dict | None = None,
                    prune_previous_checkpoints: bool = False) -> None:
        """Atomically commit samples and the state needed by the next chunk.

        Data are first flushed under ``_pending_chunk{idx}``; one HDF5 move
        publishes the completed transaction as ``chunk{idx}``.  A process
        killed before that move leaves a group ignored by readers and cleaned
        on the next ``resume=True`` open.
        """
        final_name = f"chunk{int(idx)}"
        pending_name = f"_pending_chunk{int(idx)}"
        if final_name in self._h5:
            raise ValueError(f"checkpoint already contains {final_name}")
        if pending_name in self._h5:
            del self._h5[pending_name]
        g = self._h5.create_group(pending_name)
        for key, value in datasets.items():
            arr = np.ascontiguousarray(value)
            if arr.nbytes > _COMPRESS_MIN_BYTES:
                g.create_dataset(key, data=arr, compression="gzip",
                                 compression_opts=4)
            else:
                g.create_dataset(key, data=arr)
        g.attrs["chunk"] = int(idx)
        # Provenance, not identity: per chunk (not file-level, which resume
        # validates for equality) — after a resume, later chunks of the same
        # rank file may come from a different node.
        g.attrs["hostname"] = socket.gethostname()
        for key, value in (attrs or {}).items():
            g.attrs[key] = value
        if checkpoint_datasets is not None or checkpoint_attrs is not None:
            state = g.create_group("checkpoint")
            for key, value in (checkpoint_datasets or {}).items():
                arr = np.ascontiguousarray(value)
                if arr.ndim >= 1 and arr.nbytes > _CONFIG_COMPRESS_MIN_BYTES:
                    state.create_dataset(key, data=arr, compression="gzip",
                                         compression_opts=4)
                else:
                    state.create_dataset(key, data=arr)
            for key, value in (checkpoint_attrs or {}).items():
                state.attrs[key] = value
        self._h5.flush()
        self._h5.move(pending_name, final_name)
        self._h5.flush()
        if prune_previous_checkpoints:
            for name in list(self._h5.keys()):
                match = re.fullmatch(r"chunk(\d+)", name)
                if (match and name != final_name
                        and "checkpoint" in self._h5[name]):
                    del self._h5[name]["checkpoint"]
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
        g.attrs["hostname"] = socket.gethostname()
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


def load_checkpointed_rank_chunks(run_dir, rank: int, dataset_names,
                                  expected_run_attrs: dict | None = None) -> dict:
    """Load committed raw chunks and the exact continuation checkpoint.

    Returns ``completed``, ``next_chunk``, concatenated datasets (``None``
    when no chunk exists), immutable file attributes and the nested checkpoint
    attached to the last committed chunk.  Chunk indices must be contiguous
    from zero and every dataset's leading dimension must match that chunk's
    ``n_trajectories`` attribute.
    """
    import h5py

    path = rank_file(run_dir, rank)
    empty = dict(
        completed=0,
        next_chunk=0,
        datasets={str(name): None for name in dataset_names},
        run_attrs={},
        checkpoint=None,
    )
    if not path.exists():
        return empty

    pieces = {str(name): [] for name in dataset_names}
    with h5py.File(path, "r") as f:
        for key, want in (expected_run_attrs or {}).items():
            if key not in f.attrs or not _attr_values_equal(f.attrs[key], want):
                have = f.attrs.get(key, None)
                raise ValueError(
                    f"checkpoint attribute mismatch for {key}: "
                    f"saved={have!r}, requested={want!r}")
        indices = sorted(
            int(match.group(1))
            for name in f.keys()
            for match in [re.fullmatch(r"chunk(\d+)", name)]
            if match
        )
        if indices != list(range(len(indices))):
            raise ValueError(
                f"non-contiguous committed chunks in {path}: {indices}")
        completed = 0
        for index in indices:
            group = f[f"chunk{index}"]
            count = int(group.attrs.get("n_trajectories", -1))
            if count < 0:
                raise ValueError(f"chunk{index} lacks n_trajectories")
            for name in pieces:
                if name not in group:
                    raise ValueError(f"chunk{index} lacks dataset {name!r}")
                value = np.asarray(group[name][:])
                if value.ndim == 0 or len(value) != count:
                    raise ValueError(
                        f"chunk{index}/{name} length does not match "
                        f"n_trajectories={count}")
                pieces[name].append(value)
            completed += count

        checkpoint = None
        if indices:
            last = f[f"chunk{indices[-1]}"]
            if "checkpoint" not in last:
                raise ValueError(
                    f"last committed chunk in {path} has no continuation checkpoint")
            state = last["checkpoint"]
            checkpoint = {
                "datasets": {key: np.asarray(state[key][:]) for key in state.keys()},
                "attrs": dict(state.attrs),
            }
        merged = {
            name: (np.concatenate(values) if values else None)
            for name, values in pieces.items()
        }
        return dict(
            completed=completed,
            next_chunk=len(indices),
            datasets=merged,
            run_attrs=dict(f.attrs),
            checkpoint=checkpoint,
        )


def checkpoint_tree_has_committed_chunks(run_dir) -> bool:
    """Whether any rank file below ``run_dir`` contains a published chunk."""
    import h5py

    root = Path(run_dir)
    if not root.exists():
        return False
    for path in root.rglob("rank*.h5"):
        with h5py.File(path, "r") as handle:
            if any(re.fullmatch(r"chunk\d+", name) for name in handle.keys()):
                return True
    return False


def collective_resume_decision(comm, *, rank: int, active: bool,
                               completed: int, allow_all_missing: bool,
                               label: str) -> bool:
    """Agree whether every active MPI rank resumes or every rank starts fresh.

    A mixture is unsafe: the ranks would contribute samples from different
    points in their chains while rank 0 still labels them as one protocol.
    """
    local = None if not active else bool(int(completed) > 0)
    states = comm.gather(local, root=0)
    if int(rank) == 0:
        active_states = [state for state in states if state is not None]
        any_saved = any(active_states)
        all_saved = all(active_states)
        if any_saved and not all_saved:
            decision = (
                "error",
                f"{label}: partial MPI checkpoint (some active ranks have "
                "committed chunks and others do not)",
            )
        elif all_saved:
            decision = ("resume", "")
        elif allow_all_missing:
            decision = ("fresh", "")
        else:
            decision = (
                "error",
                f"{label}: --resume found no committed CUDA checkpoint",
            )
    else:
        decision = None
    mode, message = comm.bcast(decision, root=0)
    if mode == "error":
        if "partial MPI checkpoint" in message:
            raise RuntimeError(message)
        raise FileNotFoundError(message)
    return mode == "resume"
