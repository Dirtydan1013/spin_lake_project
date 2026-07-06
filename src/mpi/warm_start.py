"""Warm-start configuration save/load shared by the MPI drivers.

Layout: <config_dir>/rank{r}.h5, one file per rank, each holding the final
Monte-Carlo configuration (operator strings, and where available the RNG
state) plus validation attrs.  A later run pointed at the same directory
loads its rank's file, installs the configuration, and skips thermalization
entirely.

If the new run has MORE ranks than the saved set, ranks wrap around
(rank % n_saved) with a warning — the per-rank seeds differ, so the chains
decorrelate after a short while.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np


def save_rank_config(config_dir, rank: int, datasets: dict,
                     attrs: dict | None = None) -> str:
    """Write <config_dir>/rank{rank}.h5 with the given datasets/attrs."""
    import h5py

    d = Path(config_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"rank{int(rank)}.h5"
    with h5py.File(path, "w") as f:
        for key, value in datasets.items():
            if isinstance(value, (str, bytes)):
                f.attrs[key] = value
            else:
                f.create_dataset(key, data=np.asarray(value))
        f.attrs["rank"] = int(rank)
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
    return str(path)


def load_rank_config(config_dir, rank: int, verbose: bool = False) -> dict | None:
    """Load this rank's config (wrapping modulo the saved rank count).

    Returns {dataset_name: ndarray, ..., 'attrs': dict} or None when the
    directory holds no rank files.
    """
    import h5py

    d = Path(config_dir)
    files = glob.glob(str(d / "rank*.h5"))
    if not files:
        return None
    ranks = sorted(int(re.search(r"rank(\d+)\.h5$", f).group(1)) for f in files)
    use = int(rank) % len(ranks)
    path = d / f"rank{ranks[use]}.h5"
    if verbose and ranks[use] != rank:
        print(f"[warm-start] rank {rank}: reusing saved rank{ranks[use]}.h5 "
              f"({len(ranks)} saved ranks)", flush=True)
    out = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            out[key] = f[key][:]
        out["attrs"] = dict(f.attrs)
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
