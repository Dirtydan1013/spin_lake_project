"""High-level CUDA work-protocol and endpoint semantics."""

from __future__ import annotations

import math

import numpy as np
import pytest

qaqmc_cuda = pytest.importorskip("qaqmc_cuda")
pytest.importorskip("qaqmc_cpp")
if not qaqmc_cuda.is_available():
    pytest.skip("CUDA device is not available", allow_module_level=True)

from src.engines.qaqmc_renyi_work_cuda import QAQMCRenyiWorkRydbergCUDA
from src.engines.qaqmc_string_work_cuda import QAQMCStringWorkRydbergCUDA


def test_renyi_work_accounting_is_exact_without_transition_sweeps():
    engine = QAQMCRenyiWorkRydbergCUDA(
        N=3, M=8, Omega=1.0, Rb=0.0, delta_min=0.0, delta_max=0.5,
        epsilon=0.02, seed=31, neighbor_cutoff=1, delta_groups=4,
        verbose=False,
    )
    start = np.zeros(3, dtype=np.uint8)
    end = np.array([1, 0, 1], dtype=np.uint8)
    engine.set_region_pair(start, end)
    engine.set_lambda_schedule(np.array([0.0, 0.25, 0.75, 1.0]))
    engine.set_sweeps_per_lambda(0, 0)
    engine.thermalize(0)

    result = engine.run_trajectory()
    expected_work = -2.0 * math.log1p(-0.75)
    assert result.work == pytest.approx(expected_work, abs=1e-15)
    assert result.exp_minus_work == pytest.approx(math.exp(-expected_work))
    assert result.final_swap_count == 0
    assert result.unjoined_at_end_count == 2
    assert result.topology_attempts == 0
    assert result.topology_accepts == 0

    batch = engine.run_trajectories(3, decorrelation_steps=0)
    np.testing.assert_allclose(batch.work_samples, expected_work, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(batch.final_swap_counts, np.zeros(3, np.int32))
    np.testing.assert_array_equal(
        batch.unjoined_counts_per_traj, np.full(3, 2, np.int32)
    )
    assert batch.delta_s2 == pytest.approx(expected_work, abs=1e-15)


def test_renyi_work_tracks_device_active_count_without_mask_download():
    engine = QAQMCRenyiWorkRydbergCUDA(
        N=3, M=8, Omega=1.0, Rb=0.0, delta_min=0.0, delta_max=0.5,
        epsilon=0.02, seed=37, neighbor_cutoff=1, delta_groups=4,
        verbose=False,
    )
    start = np.zeros(3, dtype=np.uint8)
    end = np.array([1, 0, 1], dtype=np.uint8)
    engine.set_region_pair(start, end)
    engine.set_lambda_schedule(np.array([0.0, 0.5, 1.0]))
    engine.set_sweeps_per_lambda(1, 0)
    engine.thermalize(0)

    result = engine.run_trajectory()
    # At lambda=1/2 every valid zero-bond proposal is accepted.  The endpoint
    # sweep is counted like the CPU reference but is deterministically skipped.
    assert result.final_swap_count == 2
    assert result.topology_attempts == 4
    assert result.topology_accepts == 2
    assert result.work == pytest.approx(0.0, abs=1e-15)
    assert engine.B_size == 2
    np.testing.assert_array_equal(engine.B_mask, end)


def test_string_work_endpoint_zero_weight_semantics_both_directions():
    engine = QAQMCStringWorkRydbergCUDA(
        N=3, M=8, Omega=1.0, Rb=0.0, delta_min=0.0, delta_max=0.5,
        epsilon=0.02, seed=73, neighbor_cutoff=1, delta_groups=4,
        verbose=False,
    )
    engine.set_string_sites([1])
    engine.set_lambda_schedule(np.array([0.0, 1.0]))

    engine.thermalize(0, direction="forward")
    assert engine._eng.has_checkpoint
    forward = engine.run_trajectory(direction="forward")
    assert forward.zero_weight
    assert forward.log_j == -math.inf
    assert forward.final_active_count == 0

    engine.thermalize(0, direction="reverse")
    reverse = engine.run_trajectory(direction="reverse")
    assert reverse.zero_weight
    assert reverse.log_j == -math.inf
    assert reverse.final_active_count == 1

    aggregate = engine.run_trajectories(
        3, decorrelation_steps=0, direction="forward"
    )
    assert engine._eng.has_checkpoint
    assert aggregate.o_c == 0.0
    assert aggregate.log_o_c == -math.inf
    assert aggregate.zero_weight_fraction == 1.0
