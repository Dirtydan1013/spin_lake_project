"""Lossless compact-storage tests for the standard CPU QAQMC engine."""

from __future__ import annotations

import numpy as np
import pytest

qaqmc_cpp = pytest.importorskip("qaqmc_cpp")
from src.engines.qaqmc import QAQMC_Rydberg
from src.engines.qaqmc_cpu_batch import QAQMCSharedModelBatch


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
    with pytest.raises(ValueError, match="at most 65535 bonds"):
        engine.bond_event_storage = "p_bond16"
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


def test_independent_engines_share_immutable_model_without_changing_trajectory():
    original = _engine(64, half_length=257, groups=8)
    model = original.model_data
    shared = qaqmc_cpp.QAQMCEngine(model, 12345)

    assert model.logical_bytes == original.model_memory_bytes
    assert original.model_use_count >= 3  # original, shared, Python model handle
    assert shared.model_memory_bytes == original.model_memory_bytes
    assert shared.compact_op_sites == original.compact_op_sites
    assert shared.compact_alias_indices == original.compact_alias_indices

    for _ in range(5):
        original.mc_step()
        shared.mc_step()
        np.testing.assert_array_equal(shared.op_types, original.op_types)
        np.testing.assert_array_equal(shared.op_sites, original.op_sites)
        assert shared.get_rng_state() == original.get_rng_state()


def test_python_wrapper_accepts_existing_model_data():
    first = QAQMC_Rydberg(
        N=32, M=64, Omega=1.0, Rb=1.2, delta_min=-2.0, delta_max=6.0,
        pos=_positions(32), epsilon=0.01, seed=9, verbose=False,
        use_cpp=True, neighbor_cutoff=-1, delta_groups=4,
    )
    second = QAQMC_Rydberg(
        N=32, M=64, Omega=1.0, Rb=1.2, delta_min=-2.0, delta_max=6.0,
        pos=_positions(32), epsilon=0.01, seed=10, verbose=False,
        use_cpp=True, neighbor_cutoff=-1, delta_groups=4,
        model_data=first._cpp_engine.model_data,
    )
    assert second._cpp_engine.model_memory_bytes == first._cpp_engine.model_memory_bytes
    assert first._cpp_engine.model_use_count >= 2


def test_position_only_bond_events_are_exact_and_halve_event_storage():
    balanced = _engine(64, half_length=4096, groups=8)
    aggressive = qaqmc_cpp.QAQMCEngine(balanced.model_data, 12345)
    aggressive.bond_event_storage = "p_only32"
    assert aggressive.bond_event_storage == "p_only32"

    for _ in range(8):
        balanced.mc_step()
        aggressive.mc_step()
        np.testing.assert_array_equal(aggressive.op_types, balanced.op_types)
        np.testing.assert_array_equal(aggressive.op_sites, balanced.op_sites)
        assert aggressive.get_rng_state() == balanced.get_rng_state()

    normal_memory = balanced.memory_breakdown
    compact_memory = aggressive.memory_breakdown
    assert normal_memory["site_bond_p_list_size_bytes"] == 0
    assert compact_memory["site_bond_list_size_bytes"] == 0
    assert compact_memory["site_bond_p_list_size_bytes"] * 2 == (
        normal_memory["site_bond_list_size_bytes"]
    )


def test_position_bond16_events_are_exact_and_use_six_bytes_per_event():
    balanced = _engine(64, half_length=4096, groups=8)
    compromise = qaqmc_cpp.QAQMCEngine(balanced.model_data, 12345)
    compromise.bond_event_storage = "p_bond16"
    assert compromise.bond_event_storage == "p_bond16"

    for _ in range(8):
        balanced.mc_step()
        compromise.mc_step()
        np.testing.assert_array_equal(compromise.op_types, balanced.op_types)
        np.testing.assert_array_equal(compromise.op_sites, balanced.op_sites)
        assert compromise.get_rng_state() == balanced.get_rng_state()

    normal_memory = balanced.memory_breakdown
    compact_memory = compromise.memory_breakdown
    compact_events = (
        compact_memory["site_bond_p_list_size_bytes"]
        + compact_memory["site_bond_b16_list_size_bytes"]
    )
    assert compact_events * 4 == normal_memory["site_bond_list_size_bytes"] * 3


def test_invalid_bond_event_storage_is_rejected():
    engine = _engine(8)
    with pytest.raises(ValueError, match="packed64.*p_only32"):
        engine.bond_event_storage = "compressed-ish"


def test_threaded_shared_model_batch_matches_independent_chains_exactly():
    kwargs = dict(
        N=16, M=257, Omega=1.0, Rb=1.2,
        delta_min=-2.0, delta_max=6.0, pos=_positions(16),
        epsilon=0.01, verbose=False, use_cpp=True,
        neighbor_cutoff=-1, delta_groups=8,
    )
    seeds = [700 + 9973 * lane for lane in range(4)]
    references = [QAQMC_Rydberg(seed=value, **kwargs) for value in seeds]
    with QAQMCSharedModelBatch(batch_size=4, seed=700, **kwargs) as batch:
        assert batch.chains[0]._cpp_engine.model_use_count >= 4
        assert batch.dominant_resident_bytes < sum(
            int(ref._cpp_engine.memory_breakdown["total_capacity_bytes"])
            for ref in references
        )
        for _ in range(5):
            batch.mc_step()
            for actual, expected in zip(batch.chains, references, strict=True):
                expected.mc_step()
                np.testing.assert_array_equal(actual.op_types, expected.op_types)
                np.testing.assert_array_equal(actual.op_sites, expected.op_sites)
                assert (actual._cpp_engine.get_rng_state()
                        == expected._cpp_engine.get_rng_state())


def test_threaded_shared_batch_profile_path_preserves_result_schema():
    kwargs = dict(
        N=8, M=32, Omega=1.0, Rb=1.2,
        delta_min=-2.0, delta_max=6.0, pos=_positions(8),
        epsilon=0.01, verbose=False, use_cpp=True,
        neighbor_cutoff=-1, delta_groups=4,
    )
    profile_args = dict(
        n_equil=1, n_samples=2, me_density=1, me_zl=1, me_cml=1,
        profile_step=8, batch_size=1, progress_callback=None,
        progress_every=1, n_snapshots=0, occ_nbatch=0,
    )
    references = [QAQMC_Rydberg(seed=90 + 9973 * lane, **kwargs)
                  for lane in range(2)]
    expected = [ref._cpp_engine.run_profile(**profile_args)
                for ref in references]
    with QAQMCSharedModelBatch(batch_size=2, seed=90, **kwargs) as batch:
        actual = batch.run_profiles(**profile_args)
    for got, want in zip(actual, expected, strict=True):
        assert got.keys() == want.keys()
        np.testing.assert_array_equal(got["density"], want["density"])
        np.testing.assert_array_equal(got["Z_l"], want["Z_l"])
        np.testing.assert_array_equal(got["C_m_l"], want["C_m_l"])
