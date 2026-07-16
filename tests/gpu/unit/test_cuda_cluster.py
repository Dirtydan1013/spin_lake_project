"""Deterministic open-boundary cluster tests for the CUDA backend."""

from __future__ import annotations

import numpy as np
import pytest

qaqmc_cuda = pytest.importorskip("qaqmc_cuda")
if not qaqmc_cuda.is_available():
    pytest.skip("CUDA device is not available", allow_module_level=True)


def _site_only_engine(n_sites: int, sites: np.ndarray):
    sites = np.asarray(sites, dtype=np.int32)
    assert len(sites) % 2 == 0
    n = n_sites
    return qaqmc_cuda.DiagonalEngine(
        n_sites=n_sites,
        half_length=len(sites) // 2,
        delta_min=0.0,
        delta_max=0.0,
        epsilon=0.01,
        bond_sites=np.empty((0, 2), dtype=np.int32),
        bond_vij=np.empty(0, dtype=np.float64),
        inv_coord=np.ones(n_sites, dtype=np.float64),
        alias_prob=np.ones((1, n), dtype=np.float64),
        alias_index=np.arange(n, dtype=np.int32).reshape(1, -1),
        alias_loc_kind=(np.arange(n, dtype=np.int32) << 1).reshape(1, -1),
        bond_rmax=np.empty((1, 0), dtype=np.float64),
        op_types=np.ones(len(sites), dtype=np.int32),
        op_sites=sites,
    )


def test_empty_bond_segments_accept_and_only_toggle_outer_site_ops():
    sites = np.array([0, 1, 0, 2, 0, 1, 2, 2, 1, 0, 2, 1], dtype=np.int32)
    engine = _site_only_engine(3, sites)
    stats = engine.cluster_update(seed=5, sweep_id=0)
    types, out_sites = engine.get_operator_string()

    expected = np.ones(len(sites), dtype=np.int32)
    proposed = 0
    for site in range(3):
        positions = np.flatnonzero(sites == site)
        proposed += max(len(positions) - 1, 0)
        if len(positions) >= 2:
            expected[positions[0]] = -1
            expected[positions[-1]] = -1

    np.testing.assert_array_equal(out_sites, sites)
    np.testing.assert_array_equal(types, expected)
    assert stats["proposed_segments"] == proposed
    assert stats["accepted_segments"] == proposed
    for site in range(3):
        assert np.count_nonzero((types == -1) & (out_sites == site)) % 2 == 0


def test_single_site_operator_has_only_frozen_boundary_segments():
    # Site 3 appears once; it must remain diagonal while sites 0/1 have one
    # accepted internal segment and toggle both endpoint operators.
    sites = np.array([0, 1, 3, 0, 1, 2], dtype=np.int32)
    engine = _site_only_engine(4, sites)
    stats = engine.cluster_update(seed=9, sweep_id=2)
    types, _ = engine.get_operator_string()
    assert types[2] == 1
    assert stats["proposed_segments"] == 2
    assert stats["accepted_segments"] == 2


def test_bond_segment_metropolis_frequency_matches_exact_weight_ratio():
    # At p=1 the worldline state is n0,n1=(1,0).  Flipping the internal
    # site-0 segment changes bond weight W10 -> W00, whose exact ratio is
    # 0.005/0.505 for the constants below.  Whether the first operator changes
    # from -1 to +1 directly records acceptance of that segment.
    types0 = np.array([-1, 2, 1, -1], dtype=np.int32)
    sites0 = np.array([0, 0, 0, 0], dtype=np.int32)
    engine = qaqmc_cuda.DiagonalEngine(
        n_sites=2,
        half_length=2,
        delta_min=0.5,
        delta_max=0.5,
        epsilon=0.01,
        bond_sites=np.array([[0, 1]], dtype=np.int32),
        bond_vij=np.array([0.2], dtype=np.float64),
        inv_coord=np.ones(2, dtype=np.float64),
        alias_prob=np.ones((1, 3), dtype=np.float64),
        alias_index=np.arange(3, dtype=np.int32).reshape(1, -1),
        alias_loc_kind=np.array([[0, 2, 1]], dtype=np.int32),
        bond_rmax=np.ones((1, 1), dtype=np.float64),
        op_types=types0,
        op_sites=sites0,
    )

    trials = 2_000
    accepted = 0
    for seed in range(trials):
        engine.set_operator_string(types0, sites0)
        engine.cluster_update(seed=100_000 + seed, sweep_id=0)
        types, _ = engine.get_operator_string()
        accepted += int(types[0] == 1)

    expected = 0.005 / 0.505
    observed = accepted / trials
    sigma = np.sqrt(expected * (1.0 - expected) / trials)
    assert abs(observed - expected) < 5.0 * sigma
