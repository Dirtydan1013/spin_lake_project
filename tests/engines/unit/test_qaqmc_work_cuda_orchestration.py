"""CPU-only tests for CUDA work-driver orchestration.

The fake backends exercise Python control flow on login/CI nodes where no CUDA
device is visible.  Kernel correctness remains covered by ``tests/gpu``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.engines.qaqmc_renyi_work_cuda import QAQMCRenyiWorkRydbergCUDA
from src.engines.qaqmc_string_work import StringWorkTrajectoryResult
from src.engines.qaqmc_string_work_cuda import QAQMCStringWorkRydbergCUDA


class _StringBackend:
    def __init__(self) -> None:
        self.has_checkpoint = True
        self.restores = 0
        self.saves = 0
        self.repairs = 0
        self.steps = 0

    def restore_device_checkpoint(self) -> None:
        self.restores += 1

    def save_device_checkpoint(self) -> None:
        self.saves += 1

    def set_seam_mask_consistent(self, _mask: int) -> None:
        self.repairs += 1

    def run_steps(self, count: int) -> None:
        self.steps += count


def test_string_trajectories_use_rolling_device_checkpoint():
    engine = object.__new__(QAQMCStringWorkRydbergCUDA)
    backend = _StringBackend()
    engine._eng = backend
    engine._length = 2
    engine._checkpoint_mask = 0
    engine.run_trajectory = lambda *_args, **_kwargs: StringWorkTrajectoryResult(
        log_j=math.log(2.0), zero_weight=False, final_active_count=2
    )

    result = engine.run_trajectories(3, decorrelation_steps=4)

    assert backend.restores == 3
    assert backend.saves == 3
    assert backend.repairs == 0
    assert backend.steps == 12
    assert result.o_c == pytest.approx(2.0)
    assert result.n_eff == pytest.approx(3.0)
    np.testing.assert_allclose(result.log_j_samples, math.log(2.0))


class _RenyiBackend:
    def __init__(self) -> None:
        self.topology_calls = 0
        self.mc_calls = 0
        self.mask_sets = 0
        self.restores = 0
        self.saves = 0
        self.steps = 0

    def topology_sweep(self, sites: np.ndarray, lambda_: float) -> dict:
        self.topology_calls += 1
        assert sites.tolist() == [0, 2]
        assert lambda_ == 0.5
        return {"attempts": 2, "accepts": 2, "active_count": 2}

    def mc_step(self) -> None:
        self.mc_calls += 1

    def set_mask(self, _mask: np.ndarray) -> None:
        self.mask_sets += 1

    def restore_checkpoint(self) -> None:
        self.restores += 1

    def save_checkpoint(self) -> None:
        self.saves += 1

    def run_steps(self, count: int) -> None:
        self.steps += count

    def get_mask(self) -> np.ndarray:
        raise AssertionError("production trajectory must not download the mask")


def test_renyi_trajectory_consumes_scalar_device_active_count():
    engine = object.__new__(QAQMCRenyiWorkRydbergCUDA)
    backend = _RenyiBackend()
    engine._backend = backend
    engine._schedule = np.array([0.0, 0.5, 1.0])
    engine._D_sites = np.array([0, 2], dtype=np.int32)
    engine._B_size = 0
    engine._n_topology = 1
    engine._n_qaqmc = 0

    result = engine.run_trajectory()

    assert backend.topology_calls == 1
    assert backend.mc_calls == 0
    assert result.final_swap_count == 2
    assert result.topology_attempts == 4
    assert result.topology_accepts == 2
    assert result.work == pytest.approx(0.0, abs=1e-15)


def test_renyi_trajectories_preallocate_typed_result_arrays():
    engine = object.__new__(QAQMCRenyiWorkRydbergCUDA)
    backend = _RenyiBackend()
    engine._backend = backend
    engine._A_start = np.zeros(3, dtype=np.uint8)
    engine._schedule = np.array([0.0, 0.5, 1.0])
    engine._D_sites = np.array([0, 2], dtype=np.int32)
    engine._B_size = 0
    engine._n_topology = 1
    engine._n_qaqmc = 0
    engine._checkpoint_valid = True

    result = engine.run_trajectories(3, decorrelation_steps=5)

    assert backend.mask_sets == 3
    assert backend.restores == 3
    assert backend.saves == 3
    assert backend.steps == 15
    assert backend.topology_calls == 3
    assert result.trajectory_count == 3
    assert result.total_topology_attempts == 12
    assert result.total_topology_accepts == 6
    assert result.total_unjoined_at_end == 0
    assert result.delta_s2 == pytest.approx(0.0, abs=1e-15)
    assert result.mean_exp_minus_work == pytest.approx(1.0)
    assert result.work_samples.dtype == np.float64
    assert result.final_swap_counts.dtype == np.int32
    assert result.unjoined_counts_per_traj.dtype == np.int32
    assert result.topology_attempts_per_traj.dtype == np.int64
    assert result.topology_accepts_per_traj.dtype == np.int64
    np.testing.assert_allclose(result.work_samples, 0.0, atol=1e-15)
    np.testing.assert_array_equal(result.final_swap_counts, 2)
