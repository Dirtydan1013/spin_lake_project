"""Exact continuation gates for the real CUDA work engines.

These tests exercise the production HDF5 transaction format, not only the
device-to-device checkpoint primitive: a fresh engine restored from the last
committed chunk must generate the same Philox-driven continuation as the
original engine, sample for sample.
"""

from __future__ import annotations

import numpy as np
import pytest

qaqmc_cuda = pytest.importorskip("qaqmc_cuda")
pytest.importorskip("qaqmc_cpp")
if not qaqmc_cuda.is_available():
    pytest.skip("CUDA device is not available", allow_module_level=True)

from src.engines.qaqmc_renyi_work_cuda import QAQMCRenyiWorkRydbergCUDA
from src.engines.qaqmc_string_work_cuda import QAQMCStringWorkRydbergCUDA
from src.mpi.chunk_io import RankChunkWriter, load_checkpointed_rank_chunks
from src.mpi.qaqmc_renyi_work_mpi import (
    _renyi_cuda_checkpoint,
    _restore_renyi_cuda_checkpoint,
)
from src.mpi.qaqmc_string_work_mpi import (
    _restore_string_cuda_checkpoint,
    _string_cuda_checkpoint,
)


def _hdf_roundtrip(tmp_path, name: str, state_data: dict,
                   state_attrs: dict) -> dict:
    run_dir = tmp_path / name
    run_attrs = {"backend": "cuda", "protocol": name}
    with RankChunkWriter(run_dir, 0, run_attrs=run_attrs) as writer:
        writer.write_chunk(
            0,
            datasets={"samples": np.zeros(2, dtype=np.float64)},
            attrs={"n_trajectories": 2},
            checkpoint_datasets=state_data,
            checkpoint_attrs=state_attrs,
            prune_previous_checkpoints=True,
        )
    loaded = load_checkpointed_rank_chunks(
        run_dir, 0, ("samples",), expected_run_attrs=run_attrs
    )
    assert loaded["completed"] == 2
    return loaded["checkpoint"]


def _assert_checkpoint_equal(left: tuple[dict, dict],
                             right: tuple[dict, dict]) -> None:
    left_data, left_attrs = left
    right_data, right_attrs = right
    assert left_data.keys() == right_data.keys()
    assert left_attrs == right_attrs
    for key in left_data:
        np.testing.assert_array_equal(left_data[key], right_data[key])


def _string_engine(seed: int) -> QAQMCStringWorkRydbergCUDA:
    engine = QAQMCStringWorkRydbergCUDA(
        N=4, M=12, Omega=1.0, Rb=0.0,
        delta_min=0.0, delta_max=0.8, epsilon=0.04,
        seed=seed, neighbor_cutoff=1, delta_groups=6, verbose=False,
    )
    engine.set_string_sites([1, 2])
    engine.set_lambda_schedule(np.linspace(0.0, 1.0, 5))
    return engine


def test_string_cuda_hdf_resume_replays_exact_continuation(tmp_path):
    seed = 4101
    original = _string_engine(seed)
    original.thermalize(3, direction="forward")
    original.run_trajectories(
        2, decorrelation_steps=2,
        n_topology_sweeps_per_lambda=1,
        n_qaqmc_sweeps_per_lambda=1,
        direction="forward",
    )
    saved = _string_cuda_checkpoint(original, None, 4)
    checkpoint = _hdf_roundtrip(tmp_path, "string", *saved)

    expected = original.run_trajectories(
        3, decorrelation_steps=2,
        n_topology_sweeps_per_lambda=1,
        n_qaqmc_sweeps_per_lambda=1,
        direction="forward",
    )

    resumed = _string_engine(seed)
    _restore_string_cuda_checkpoint(
        resumed, checkpoint, None, 4, direction="forward"
    )
    actual = resumed.run_trajectories(
        3, decorrelation_steps=2,
        n_topology_sweeps_per_lambda=1,
        n_qaqmc_sweeps_per_lambda=1,
        direction="forward",
    )

    np.testing.assert_array_equal(actual.log_j_samples, expected.log_j_samples)
    _assert_checkpoint_equal(
        _string_cuda_checkpoint(resumed, None, 4),
        _string_cuda_checkpoint(original, None, 4),
    )


def _renyi_engine(seed: int) -> QAQMCRenyiWorkRydbergCUDA:
    engine = QAQMCRenyiWorkRydbergCUDA(
        N=4, M=12, Omega=1.0, Rb=0.0,
        delta_min=0.0, delta_max=0.8, epsilon=0.04,
        seed=seed, neighbor_cutoff=1, delta_groups=6, verbose=False,
    )
    engine.set_region(np.array([1, 0, 1, 0], dtype=np.uint8))
    engine.set_lambda_schedule(np.linspace(0.0, 1.0, 5))
    engine.set_sweeps_per_lambda(1, 1)
    return engine


def test_renyi_cuda_hdf_resume_replays_exact_continuation(tmp_path):
    seed = 4201
    original = _renyi_engine(seed)
    original.thermalize(3)
    original.run_trajectories(2, decorrelation_steps=2)
    saved = _renyi_cuda_checkpoint(original, None, 4)
    checkpoint = _hdf_roundtrip(tmp_path, "renyi", *saved)

    expected = original.run_trajectories(3, decorrelation_steps=2)

    resumed = _renyi_engine(seed)
    _restore_renyi_cuda_checkpoint(resumed, checkpoint, None, 4)
    actual = resumed.run_trajectories(3, decorrelation_steps=2)

    for field in (
        "work_samples",
        "final_swap_counts",
        "unjoined_counts_per_traj",
        "topology_attempts_per_traj",
        "topology_accepts_per_traj",
    ):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    _assert_checkpoint_equal(
        _renyi_cuda_checkpoint(resumed, None, 4),
        _renyi_cuda_checkpoint(original, None, 4),
    )
