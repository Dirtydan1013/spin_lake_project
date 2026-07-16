"""Transition-kernel tests for the device-resident CUDA diagonal prototype."""

from __future__ import annotations

import numpy as np
import pytest

qaqmc_cuda = pytest.importorskip("qaqmc_cuda")
if not qaqmc_cuda.is_available():
    pytest.skip("CUDA device is not available", allow_module_level=True)


def _vose_alias(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic Vose builder matching the C++ table convention."""
    weights = np.asarray(weights, dtype=np.float64)
    n = len(weights)
    scaled = weights * n / weights.sum()
    prob = scaled.copy()
    alias = np.arange(n, dtype=np.int32)
    small = [i for i in range(n) if prob[i] < 1.0]
    large = [i for i in range(n) if prob[i] >= 1.0]
    while small and large:
        s = small.pop()
        large_i = large.pop()
        alias[s] = large_i
        prob[large_i] -= 1.0 - prob[s]
        (small if prob[large_i] < 1.0 else large).append(large_i)
    return prob, alias


def _two_site_engine(half_length: int, op_types=None, op_sites=None):
    # delta=0, vij=1, epsilon convention gives W=(1,1,1,0), Wmax=1.
    # Alias proposal weights are site0=1/2, site1=1/2, bond0=1.
    weights = np.array([0.5, 0.5, 1.0], dtype=np.float64)
    prob, alias = _vose_alias(weights)
    loc_kind = np.array([0 << 1, 1 << 1, (0 << 1) | 1], dtype=np.int32)
    length = 2 * half_length
    if op_types is None:
        op_types = np.ones(length, dtype=np.int32)
    if op_sites is None:
        op_sites = np.zeros(length, dtype=np.int32)
    return qaqmc_cuda.DiagonalEngine(
        n_sites=2,
        half_length=half_length,
        delta_min=0.0,
        delta_max=0.0,
        epsilon=0.01,
        bond_sites=np.array([[0, 1]], dtype=np.int32),
        bond_vij=np.array([1.0], dtype=np.float64),
        inv_coord=np.ones(2, dtype=np.float64),
        alias_prob=prob.reshape(1, -1),
        alias_index=alias.reshape(1, -1),
        alias_loc_kind=loc_kind.reshape(1, -1),
        bond_rmax=np.ones((1, 1), dtype=np.float64),
        op_types=np.asarray(op_types, dtype=np.int32),
        op_sites=np.asarray(op_sites, dtype=np.int32),
    )


def test_diagonal_distribution_matches_exact_conditional_weights():
    engine = _two_site_engine(half_length=200_000)
    stats = engine.diagonal_update(seed=712367, sweep_id=0)
    types, sites = engine.get_operator_string()

    assert stats["failed_slots"] == 0
    assert stats["updated_slots"] == len(types)
    assert np.all(np.isin(types, [1, 2]))
    p_site0 = np.mean((types == 1) & (sites == 0))
    p_site1 = np.mean((types == 1) & (sites == 1))
    p_bond = np.mean(types == 2)
    # Exact normalized weights are (0.25, 0.25, 0.50).
    np.testing.assert_allclose([p_site0, p_site1, p_bond], [0.25, 0.25, 0.50],
                               atol=0.004, rtol=0.0)


def test_diagonal_philox_replays_for_same_seed_sweep_and_input():
    half_length = 4097
    original_types = np.ones(2 * half_length, dtype=np.int32)
    original_sites = np.zeros(2 * half_length, dtype=np.int32)
    engine = _two_site_engine(half_length, original_types, original_sites)

    first_stats = engine.diagonal_update(seed=99, sweep_id=17)
    first_types, first_sites = engine.get_operator_string()
    engine.set_operator_string(original_types, original_sites)
    second_stats = engine.diagonal_update(seed=99, sweep_id=17)
    second_types, second_sites = engine.get_operator_string()

    np.testing.assert_array_equal(first_types, second_types)
    np.testing.assert_array_equal(first_sites, second_sites)
    assert first_stats["proposal_attempts"] == second_stats["proposal_attempts"]

    engine.set_operator_string(original_types, original_sites)
    engine.diagonal_update(seed=99, sweep_id=18)
    third_types, third_sites = engine.get_operator_string()
    assert not (np.array_equal(first_types, third_types)
                and np.array_equal(first_sites, third_sites))


def test_diagonal_preserves_offdiagonal_slots_exactly():
    half_length = 1025
    length = 2 * half_length
    types = np.ones(length, dtype=np.int32)
    sites = np.zeros(length, dtype=np.int32)
    offdiag = np.arange(0, length, 7)
    types[offdiag] = -1
    sites[offdiag] = np.arange(len(offdiag), dtype=np.int32) % 2
    expected_sites = sites[offdiag].copy()

    engine = _two_site_engine(half_length, types, sites)
    stats = engine.diagonal_update(seed=1234, sweep_id=3)
    out_types, out_sites = engine.get_operator_string()

    np.testing.assert_array_equal(out_types[offdiag], -np.ones(len(offdiag), np.int32))
    np.testing.assert_array_equal(out_sites[offdiag], expected_sites)
    assert stats["updated_slots"] == length - len(offdiag)
    assert stats["failed_slots"] == 0
    assert engine.device_bytes > 0
