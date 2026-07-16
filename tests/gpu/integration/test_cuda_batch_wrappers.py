"""End-to-end gates for standard, string-work and Renyi-work CUDA batches."""

from __future__ import annotations

import numpy as np
import pytest

qaqmc_cuda = pytest.importorskip("qaqmc_cuda")
if not qaqmc_cuda.is_available():
    pytest.skip("CUDA device is not available", allow_module_level=True)

from src.engines.qaqmc_batch_cuda import QAQMC_Rydberg_CUDA_Batch
from src.engines.qaqmc_cuda import QAQMC_Rydberg_CUDA
from src.engines.qaqmc_renyi_work_batch_cuda import (
    QAQMCRenyiWorkRydbergCUDABatch,
)
from src.engines.qaqmc_renyi_work_cuda import QAQMCRenyiWorkRydbergCUDA
from src.engines.qaqmc_string_work_batch_cuda import (
    QAQMCStringWorkRydbergCUDABatch,
)
from src.engines.qaqmc_string_work_cuda import QAQMCStringWorkRydbergCUDA


def _common(seed: int = 71) -> dict:
    return {
        "N": 4,
        "M": 32,
        "Omega": 1.0,
        "Rb": 0.0,
        "delta_min": 0.2,
        "delta_max": 0.7,
        "epsilon": 0.01,
        "seed": seed,
        "pos": np.arange(4, dtype=np.float64).reshape(-1, 1),
        "neighbor_cutoff": 0,
        "delta_groups": 8,
        "verbose": False,
    }


def test_standard_batch_b1_matches_single_wrapper_exactly():
    single = QAQMC_Rydberg_CUDA(**_common())
    batch = QAQMC_Rydberg_CUDA_Batch(**_common(), batch_size=1)
    single.mc_step()
    batch.mc_step()
    single_types, single_sites = single.engine.get_operator_string()
    batch_types, batch_sites = batch.engine.get_operator_strings()
    np.testing.assert_array_equal(batch_types[0], single_types)
    np.testing.assert_array_equal(batch_sites[0], single_sites)
    np.testing.assert_array_equal(
        batch.midpoint_states()[0], single.engine.midpoint_state()
    )


def test_string_work_batch_runs_full_protocol_and_b1_matches_single():
    schedule = np.array([0.0, 0.5, 1.0])
    single = QAQMCStringWorkRydbergCUDA(**_common(seed=81))
    batch = QAQMCStringWorkRydbergCUDABatch(
        **_common(seed=81), batch_size=1
    )
    for engine in (single, batch):
        engine.set_string_sites([1])
        engine.set_lambda_schedule(schedule)
        engine.thermalize(0)
    expected = single.run_trajectories(
        2, decorrelation_steps=1,
        n_topology_sweeps_per_lambda=1,
        n_qaqmc_sweeps_per_lambda=1,
    )
    actual = batch.run_trajectories(
        2, decorrelation_steps=1,
        n_topology_sweeps_per_lambda=1,
        n_qaqmc_sweeps_per_lambda=1,
    )
    np.testing.assert_array_equal(actual.log_j_samples, expected.log_j_samples)


def test_string_work_batch_handles_partial_final_wave():
    engine = QAQMCStringWorkRydbergCUDABatch(
        **_common(seed=82), batch_size=2
    )
    engine.set_string_sites([1])
    engine.set_lambda_schedule(np.array([0.0, 0.5, 1.0]))
    engine.thermalize(0)
    result = engine.run_trajectories(
        3, decorrelation_steps=0,
        n_topology_sweeps_per_lambda=1,
        n_qaqmc_sweeps_per_lambda=0,
    )
    assert result.log_j_samples.shape == (3,)
    assert result.n_trajectories == 3


def test_renyi_work_batch_runs_full_protocol_and_b1_matches_single():
    schedule = np.array([0.0, 0.5, 1.0])
    single = QAQMCRenyiWorkRydbergCUDA(**_common(seed=91))
    batch = QAQMCRenyiWorkRydbergCUDABatch(
        **_common(seed=91), batch_size=1
    )
    for engine in (single, batch):
        engine.set_region(np.array([0, 1, 0, 0], dtype=np.uint8))
        engine.set_lambda_schedule(schedule)
        engine.set_sweeps_per_lambda(1, 1)
        engine.thermalize(0)
    expected = single.run_trajectories(2, decorrelation_steps=1)
    actual = batch.run_trajectories(2, decorrelation_steps=1)
    np.testing.assert_array_equal(actual.work_samples, expected.work_samples)
    np.testing.assert_array_equal(
        actual.final_swap_counts, expected.final_swap_counts
    )


def test_renyi_work_batch_handles_partial_final_wave():
    engine = QAQMCRenyiWorkRydbergCUDABatch(
        **_common(seed=92), batch_size=2
    )
    engine.set_region(np.array([0, 1, 0, 0], dtype=np.uint8))
    engine.set_lambda_schedule(np.array([0.0, 0.5, 1.0]))
    engine.set_sweeps_per_lambda(1, 0)
    engine.thermalize(0)
    result = engine.run_trajectories(3, decorrelation_steps=0)
    assert result.work_samples.shape == (3,)
    assert result.trajectory_count == 3
