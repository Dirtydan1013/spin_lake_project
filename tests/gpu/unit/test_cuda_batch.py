"""Correctness, independence and shared-memory tests for CUDA chain batches."""

from __future__ import annotations

import numpy as np
import pytest

qaqmc_cuda = pytest.importorskip("qaqmc_cuda")
if not qaqmc_cuda.is_available():
    pytest.skip("CUDA device is not available", allow_module_level=True)


def _site_only_model(n_sites: int) -> dict[str, np.ndarray]:
    return {
        "bond_sites": np.empty((0, 2), dtype=np.int32),
        "bond_vij": np.empty(0, dtype=np.float64),
        "inv_coord": np.ones(n_sites, dtype=np.float64),
        "alias_prob": np.ones((1, n_sites), dtype=np.float64),
        "alias_index": np.arange(n_sites, dtype=np.int32).reshape(1, -1),
        "alias_loc_kind": (
            np.arange(n_sites, dtype=np.int32) << 1
        ).reshape(1, -1),
        "bond_rmax": np.empty((1, 0), dtype=np.float64),
    }


def _diagonal_engine(types: np.ndarray, sites: np.ndarray):
    return qaqmc_cuda.DiagonalEngine(
        n_sites=4,
        half_length=types.size // 2,
        delta_min=-0.2,
        delta_max=0.8,
        epsilon=0.01,
        op_types=types,
        op_sites=sites,
        **_site_only_model(4),
    )


def _diagonal_batch(types: np.ndarray, sites: np.ndarray):
    return qaqmc_cuda.BatchedDiagonalEngine(
        batch_size=types.shape[0],
        n_sites=4,
        half_length=types.shape[1] // 2,
        delta_min=-0.2,
        delta_max=0.8,
        epsilon=0.01,
        op_types=types,
        op_sites=sites,
        **_site_only_model(4),
    )


def _renyi_batch(types: np.ndarray, sites: np.ndarray):
    return qaqmc_cuda.BatchedRenyiEngine(
        batch_size=types.shape[0],
        n_sites=4,
        half_length=types.shape[2] // 2,
        delta_min=-0.2,
        delta_max=0.8,
        epsilon=0.01,
        op_types=types,
        op_sites=sites,
        **_site_only_model(4),
    )


def test_batched_standard_matches_independent_single_chains_exactly():
    batch_size, length = 3, 514
    types = np.ones((batch_size, length), dtype=np.int32)
    sites = np.vstack([
        np.resize(np.array([0, 1, 2, 3], dtype=np.int32) + shift, length) % 4
        for shift in range(batch_size)
    ])
    original_types = types.copy()
    original_sites = sites.copy()
    batch = _diagonal_batch(types, sites)
    singles = [
        _diagonal_engine(original_types[k], original_sites[k])
        for k in range(batch_size)
    ]

    shared = int(batch.shared_model_bytes)
    assert shared > 0
    per_chain_state = int(singles[0].device_bytes) - shared
    assert int(batch.device_bytes) == shared + batch_size * per_chain_state

    seeds = np.array([101, 202, 303], dtype=np.uint64)
    sweeps = np.array([0, 7, 19], dtype=np.uint64)
    batch_diagonal = batch.diagonal_update(seeds, sweeps)
    for k, single in enumerate(singles):
        expected = single.diagonal_update(int(seeds[k]), int(sweeps[k]))
        assert batch_diagonal[k]["proposal_attempts"] == expected["proposal_attempts"]

    batch_cluster = batch.cluster_update(seeds ^ np.uint64(0x55AA), sweeps)
    for k, single in enumerate(singles):
        expected = single.cluster_update(
            int(seeds[k] ^ np.uint64(0x55AA)), int(sweeps[k])
        )
        assert batch_cluster[k]["accepted_segments"] == expected["accepted_segments"]

    got_types, got_sites = batch.get_operator_strings()
    for k, single in enumerate(singles):
        expected_types, expected_sites = single.get_operator_string()
        np.testing.assert_array_equal(got_types[k], expected_types)
        np.testing.assert_array_equal(got_sites[k], expected_sites)


def test_batched_standard_b1_is_api_and_trajectory_compatible():
    length = 258
    types = np.ones((1, length), dtype=np.int32)
    sites = np.resize(np.arange(4, dtype=np.int32), (1, length))
    batch = _diagonal_batch(types, sites)
    single = _diagonal_engine(types[0], sites[0])
    seeds = np.array([919], dtype=np.uint64)
    sweeps = np.array([11], dtype=np.uint64)
    batch.diagonal_update(seeds, sweeps)
    single.diagonal_update(919, 11)
    batch.cluster_update(seeds, sweeps)
    single.cluster_update(919, 11)
    got_types, got_sites = batch.get_operator_strings()
    expected_types, expected_sites = single.get_operator_string()
    np.testing.assert_array_equal(got_types[0], expected_types)
    np.testing.assert_array_equal(got_sites[0], expected_sites)


def test_batched_offdiagonal_masks_checkpoint_and_topology_are_per_chain():
    batch_size, length = 3, 32
    types = np.ones((batch_size, length), dtype=np.int32)
    sites = np.vstack([
        np.resize(np.array([0, 1, 2, 3], dtype=np.int32) + shift, length) % 4
        for shift in range(batch_size)
    ])
    batch = _diagonal_batch(types, sites)
    batch.set_string_sites(np.array([0, 2], dtype=np.int32), length // 2)
    masks = np.array([0, 3, 1], dtype=np.uint64)
    batch.set_seam_masks_consistent(masks)
    np.testing.assert_array_equal(batch.get_seam_masks(), masks)
    saved_types, saved_sites = batch.get_operator_strings()
    batch.save_checkpoint()
    assert batch.has_checkpoint

    stats = batch.topology_sweep(
        0.5,
        np.array([41, 42, 43], dtype=np.uint64),
        np.zeros(batch_size, dtype=np.uint64),
    )
    assert [row["accepts"] for row in stats] == [2, 2, 2]
    np.testing.assert_array_equal(batch.get_seam_masks(), masks ^ np.uint64(3))

    batch.restore_checkpoint()
    restored_types, restored_sites = batch.get_operator_strings()
    np.testing.assert_array_equal(restored_types, saved_types)
    np.testing.assert_array_equal(restored_sites, saved_sites)
    np.testing.assert_array_equal(batch.get_seam_masks(), masks)


def test_batched_renyi_masks_topology_and_checkpoint_are_independent():
    batch_size, length = 3, 32
    types = np.ones((batch_size, 2, length), dtype=np.int32)
    sites = np.empty_like(types)
    for chain in range(batch_size):
        sites[chain, 0] = np.resize(
            np.arange(4, dtype=np.int32) + chain, length
        ) % 4
        sites[chain, 1] = np.resize(
            np.arange(3, -1, -1, dtype=np.int32) + chain, length
        ) % 4
    batch = _renyi_batch(types, sites)
    masks = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 1],
    ], dtype=np.uint8)
    batch.set_masks(masks)
    np.testing.assert_array_equal(batch.get_masks(), masks)
    batch.save_checkpoint()
    saved_types, saved_sites = batch.get_operator_strings()

    stats = batch.topology_sweep(
        np.array([0, 2], dtype=np.int32),
        0.5,
        np.array([51, 52, 53], dtype=np.uint64),
        np.zeros(batch_size, dtype=np.uint64),
    )
    assert [row["accepts"] for row in stats] == [2, 2, 2]
    expected_masks = masks.copy()
    expected_masks[:, [0, 2]] ^= 1
    np.testing.assert_array_equal(batch.get_masks(), expected_masks)

    batch.cluster_update(
        np.array([61, 62, 63], dtype=np.uint64),
        np.ones(batch_size, dtype=np.uint64),
    )
    batch.restore_checkpoint()
    restored_types, restored_sites = batch.get_operator_strings()
    np.testing.assert_array_equal(restored_types, saved_types)
    np.testing.assert_array_equal(restored_sites, saved_sites)


def test_batched_profile_states_keep_batch_dimension():
    length = 18
    types = np.ones((2, length), dtype=np.int32)
    sites = np.zeros((2, length), dtype=np.int32)
    types[0, [1, 7]] = -1
    sites[0, [1, 7]] = [0, 1]
    types[1, [2, 11]] = -1
    sites[1, [2, 11]] = [2, 3]
    batch = _diagonal_batch(types, sites)
    packed = np.asarray(batch.profile_states(3))
    assert packed.shape == (2, length // 3, 1)
    for chain in range(2):
        single = _diagonal_engine(types[chain], sites[chain])
        np.testing.assert_array_equal(packed[chain], single.profile_states(3))
