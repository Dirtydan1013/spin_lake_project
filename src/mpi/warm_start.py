"""Backward-compatibility shim for the warm-start API.

The per-rank chunk/config storage now lives in :mod:`src.mpi.chunk_io`, where
each rank owns a single ``rank{r}.h5`` holding ``chunk{i}`` groups plus a
``final_config`` group.  These wrappers keep the old ``save_rank_config`` /
``load_rank_config`` names working (writing/reading the new ``final_config``
group), so any external script pinned to this module still functions.

New code should import from :mod:`src.mpi.chunk_io` directly.
"""

from __future__ import annotations

from src.mpi.chunk_io import (  # noqa: F401  (re-exported for compatibility)
    RankChunkWriter,
    check_config_compat,
    load_warm_config,
    rank_file,
)


def save_rank_config(config_dir, rank: int, datasets: dict,
                     attrs: dict | None = None) -> str:
    """Write a config-only ``<config_dir>/rank{rank}.h5`` (final_config group)."""
    with RankChunkWriter(config_dir, rank) as w:
        w.write_final_config(datasets=datasets, attrs=attrs)
    return str(rank_file(config_dir, rank))


def load_rank_config(config_dir, rank: int, verbose: bool = False) -> dict | None:
    """Alias for :func:`src.mpi.chunk_io.load_warm_config`."""
    return load_warm_config(config_dir, rank, verbose=verbose)
