"""Host-side result semantics that do not require a CUDA device."""

from __future__ import annotations

import numpy as np

from src.engines.qaqmc_renyi_work_cuda import QAQMCRenyiWorkRydbergCUDA


def test_trivial_region_pair_returns_one_sample_per_requested_trajectory():
    engine = object.__new__(QAQMCRenyiWorkRydbergCUDA)
    engine._D_sites = np.empty(0, dtype=np.int32)
    result = engine.run_trajectories(4, decorrelation_steps=3)

    assert result.trajectory_count == 4
    assert result.delta_s2 == 0.0
    assert result.mean_exp_minus_work == 1.0
    for values in (
        result.work_samples,
        result.final_swap_counts,
        result.unjoined_counts_per_traj,
        result.topology_attempts_per_traj,
        result.topology_accepts_per_traj,
    ):
        np.testing.assert_array_equal(values, np.zeros(4, dtype=values.dtype))
