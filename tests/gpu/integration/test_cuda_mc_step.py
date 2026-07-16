"""End-to-end structural regression for diagonal + cluster CUDA updates."""

from __future__ import annotations

import numpy as np
import pytest

qaqmc_cpp = pytest.importorskip("qaqmc_cpp")
qaqmc_cuda = pytest.importorskip("qaqmc_cuda")
if not qaqmc_cuda.is_available():
    pytest.skip("CUDA device is not available", allow_module_level=True)

from src.engines.qaqmc_cuda import CudaDiagonalBackend


def test_repeated_cuda_mc_steps_preserve_operator_and_worldline_invariants():
    n_sites = 6
    half_length = 2_000
    pos = np.arange(n_sites, dtype=np.float64).reshape(-1, 1)
    cpu = qaqmc_cpp.QAQMCEngine(
        n_sites, 1.0, -0.5, 1.5, 1.2, half_length, 0.01, 123, pos,
        neighbor_cutoff=1, delta_groups=40,
    )
    backend = CudaDiagonalBackend.from_cpu_engine(cpu)

    for sweep in range(40):
        diagonal = backend.engine.diagonal_update(seed=8001, sweep_id=sweep)
        cluster = backend.engine.cluster_update(seed=9001, sweep_id=sweep)
        assert diagonal["failed_slots"] == 0
        assert cluster["accepted_segments"] <= cluster["proposed_segments"]

        types, sites = backend.get_operator_string()
        assert np.all(np.isin(types, [-1, 1, 2]))
        assert np.all((sites[types != 2] >= 0) & (sites[types != 2] < n_sites))
        for site in range(n_sites):
            # Fixed |0...0> boundaries require even off-diagonal parity.
            assert np.count_nonzero((types == -1) & (sites == site)) % 2 == 0
