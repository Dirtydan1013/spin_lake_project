"""Lossless compact-storage tests for the standard CPU QAQMC engine."""

from __future__ import annotations

import numpy as np
import pytest

qaqmc_cpp = pytest.importorskip("qaqmc_cpp")
from src.engines.qaqmc import QAQMC_Rydberg


def _positions(n_sites: int) -> np.ndarray:
    # Unique, mildly non-uniform distances avoid coincident sites and exercise
    # the full-bond builder without depending on a lattice helper.
    x = np.arange(n_sites, dtype=np.float64)
    return np.column_stack((x, 0.01 * (x % 7)))


def _engine(n_sites: int, half_length: int = 32, groups: int = 4):
    return qaqmc_cpp.QAQMCEngine(
        n_sites, 1.0, -2.0, 6.0, 1.2, half_length, 0.01, 12345,
        _positions(n_sites), neighbor_cutoff=-1, delta_groups=groups,
    )


def test_n216_uses_lossless_16bit_indices_and_no_materialized_delta_schedule():
    n_sites, groups, half_length = 216, 4, 32
    n_bonds = n_sites * (n_sites - 1) // 2
    max_alias = n_sites + n_bonds
    engine = _engine(n_sites, half_length=half_length, groups=groups)
    memory = engine.memory_breakdown

    assert engine.compact_op_sites
    assert engine.compact_alias_indices
    assert memory["delta_schedule_size_bytes"] == 0
    assert memory["delta_schedule_capacity_bytes"] == 0
    assert memory["op_types_size_bytes"] == 2 * half_length
    assert memory["op_sites16_size_bytes"] == 2 * (2 * half_length)
    assert memory["op_sites32_size_bytes"] == 0
    assert memory["alias_prob_size_bytes"] == groups * max_alias * 8
    assert memory["alias_idx16_size_bytes"] == groups * max_alias * 2
    assert memory["alias_idx32_size_bytes"] == 0
    assert memory["bond_W_rmax_size_bytes"] == groups * n_bonds * 8
    assert not any("bond_W_max" in key for key in memory)


def test_n384_full_bonds_falls_back_to_32bit_indices_without_overflow():
    n_sites, groups = 384, 2
    n_bonds = n_sites * (n_sites - 1) // 2
    max_alias = n_sites + n_bonds
    assert max_alias > np.iinfo(np.uint16).max

    engine = _engine(n_sites, half_length=8, groups=groups)
    memory = engine.memory_breakdown
    assert not engine.compact_op_sites
    assert not engine.compact_alias_indices
    assert memory["op_sites16_size_bytes"] == 0
    assert memory["op_sites32_size_bytes"] == 2 * 8 * 4
    assert memory["alias_idx16_size_bytes"] == 0
    assert memory["alias_idx32_size_bytes"] == groups * max_alias * 4
    engine.mc_step()
    types = engine.op_types
    sites = engine.op_sites
    assert np.all(sites[types == 2] < n_bonds)
    assert np.all(sites[types != 2] < n_sites)


def test_compact_operator_string_round_trips_through_compatible_int32_api():
    engine = _engine(216, half_length=16, groups=2)
    length = engine.M_total
    types = np.ones(length, dtype=np.int32)
    sites = (np.arange(length, dtype=np.int32) % engine.N).astype(np.int32)
    types[3] = 2
    sites[3] = 1234  # valid bond index, still larger than N
    types[7] = -1
    sites[7] = 19

    engine.set_op_string(types, sites)
    assert engine.op_types.dtype == np.int32
    assert engine.op_sites.dtype == np.int32
    np.testing.assert_array_equal(engine.op_types, types)
    np.testing.assert_array_equal(engine.op_sites, sites)


def test_on_demand_delta_schedule_matches_original_expression_bitwise():
    engine = _engine(216, half_length=257, groups=3)
    p = np.arange(engine.M_total, dtype=np.int64)
    expected = np.where(
        p < engine.M,
        -2.0 + (6.0 - (-2.0)) * (p.astype(np.float64) / engine.M),
        6.0 - (6.0 - (-2.0)) * ((p - engine.M).astype(np.float64) / engine.M),
    )
    actual = np.asarray(engine.delta_schedule)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_event_scratch_capacity_has_bounded_headroom_after_fluctuations():
    engine = _engine(216, half_length=4096, groups=16)
    for _ in range(12):
        engine.mc_step()
    memory = engine.memory_breakdown

    for prefix in ("site_op_list", "site_bond_list"):
        size = memory[f"{prefix}_size_bytes"]
        capacity = memory[f"{prefix}_capacity_bytes"]
        element_size = 4 if prefix == "site_op_list" else 8
        allowed_slack = max(4096 * element_size, size // 8)
        assert size <= capacity <= size + allowed_slack


def test_set_op_string_rejects_indices_that_would_narrow_or_corrupt_state():
    engine = _engine(216, half_length=4, groups=2)
    types = np.ones(engine.M_total, dtype=np.int32)
    sites = np.zeros(engine.M_total, dtype=np.int32)

    sites[0] = engine.N
    with pytest.raises(ValueError, match="location out of range"):
        engine.set_op_string(types, sites)

    sites[0] = 0
    types[0] = 7
    with pytest.raises(ValueError, match="invalid operator type"):
        engine.set_op_string(types, sites)


def test_python_wrapper_does_not_retain_full_int32_operator_mirrors():
    wrapper = QAQMC_Rydberg(
        N=32, M=64, Omega=1.0, Rb=1.2, delta_min=-2.0, delta_max=6.0,
        pos=_positions(32), epsilon=0.01, seed=9, verbose=False,
        use_cpp=True, neighbor_cutoff=-1, delta_groups=4,
    )
    assert wrapper._cpp_engine is not None
    assert not hasattr(wrapper, "_op_types")
    assert not hasattr(wrapper, "_op_sites")

    # Compatibility exports remain int32 and always reflect current C++ state.
    assert wrapper.op_types.dtype == np.int32
    assert wrapper.op_sites.dtype == np.int32
    wrapper.mc_step()
    np.testing.assert_array_equal(wrapper.op_types, wrapper._cpp_engine.op_types)
    np.testing.assert_array_equal(wrapper.op_sites, wrapper._cpp_engine.op_sites)


def test_compact_checkpoint_restore_replays_next_step_exactly():
    original = _engine(64, half_length=257, groups=8)
    for _ in range(7):
        original.mc_step()

    restored = _engine(64, half_length=257, groups=8)
    restored.set_op_string(original.op_types, original.op_sites)
    restored.set_rng_state(original.get_rng_state())

    original.mc_step()
    restored.mc_step()
    np.testing.assert_array_equal(restored.op_types, original.op_types)
    np.testing.assert_array_equal(restored.op_sites, original.op_sites)
    assert restored.get_rng_state() == original.get_rng_state()
