"""Helpers shared by the MPI production drivers.

These existed as per-driver copies; the values and formulas are part of the
reproducibility contract (rank seeds enter recorded warm-start configs and
checkpoint fingerprints), so centralising them must not change any output.
"""

from __future__ import annotations

import numpy as np

# Per-rank seed stride.  A large prime so independent-chain seeds do not
# correlate; the SAME constant is used by every driver and by the shared-model
# batch runner, which is what makes "4 ranks x 16 chains" and "64 ranks x 1
# chain" produce the same family of chains.  DO NOT change: recorded
# warm-start configs and CUDA checkpoint fingerprints embed seeds derived
# from it.
RANK_SEED_STRIDE = 9973


def rank_seed(seed: int, rank: int) -> int:
    """Seed of the independent chain owned by ``rank``."""
    return int(seed) + RANK_SEED_STRIDE * int(rank)


def cuda_device_for_rank(comm) -> int:
    """Map a node-local MPI rank to a visible CUDA device.

    Slurm ``--gpus-per-task=1`` exposes one device per process, in which case
    its local index is always zero.  Plain ``mpiexec`` commonly exposes every
    allocated device, so use the shared-memory communicator rank there.
    """
    import qaqmc_cuda

    from mpi4py import MPI

    visible = len(qaqmc_cuda.device_info())
    if visible <= 0:
        raise RuntimeError("CUDA backend selected but no GPU is visible")
    local = comm.Split_type(MPI.COMM_TYPE_SHARED)
    try:
        local_rank = local.Get_rank()
    finally:
        local.Free()
    return 0 if visible == 1 else int(local_rank % visible)


def permutation_checkpoint(site_perm, n_sites: int) -> np.ndarray:
    """Canonical int32 form of a site permutation for checkpoint fingerprints
    (identity when permutation is disabled)."""
    return np.asarray(
        np.arange(n_sites, dtype=np.int32) if site_perm is None else site_perm,
        dtype=np.int32,
    )
