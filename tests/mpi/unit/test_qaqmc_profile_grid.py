"""The MPI profile driver must not recreate the complete O(M) delta ramp."""

from __future__ import annotations

import numpy as np
import h5py

from src.engines.qaqmc import _write_qaqmc_delta_schedule
from src.mpi.qaqmc_mpi import _qaqmc_profile_grid


def test_profile_grid_materializes_only_measured_points_and_matches_formula():
    M = 1_000_003
    profile_step = 997
    delta_min, delta_max = -2.0, 6.0
    p, delta = _qaqmc_profile_grid(M, profile_step, delta_min, delta_max)

    assert len(p) == (2 * M) // profile_step
    assert len(delta) == len(p)
    assert len(p) < (2 * M) // 900  # explicitly not the full 2M schedule

    expected = np.empty_like(delta)
    fwd = p < M
    expected[fwd] = delta_min + (delta_max - delta_min) * (
        p[fwd].astype(np.float64) / M)
    expected[~fwd] = delta_max - (delta_max - delta_min) * (
        (p[~fwd] - M).astype(np.float64) / M)
    np.testing.assert_array_equal(delta.view(np.uint64), expected.view(np.uint64))


def test_profile_grid_handles_step_larger_than_operator_string():
    p, delta = _qaqmc_profile_grid(10, 21, -1.0, 1.0)
    assert p.shape == (0,)
    assert delta.shape == (0,)


def test_hdf_delta_schedule_keeps_schema_and_streams_in_bounded_chunks(tmp_path):
    path = tmp_path / "schedule.h5"
    M = 10_003
    with h5py.File(path, "w") as handle:
        dataset = _write_qaqmc_delta_schedule(
            handle, M, -2.0, 6.0, chunk_slots=257)
        assert dataset.shape == (2 * M,)
        assert dataset.dtype == np.dtype(np.float64)
        assert dataset.chunks == (257,)

    _, expected = _qaqmc_profile_grid(M, 1, -2.0, 6.0)
    with h5py.File(path, "r") as handle:
        actual = handle["delta_schedule"][:]
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
