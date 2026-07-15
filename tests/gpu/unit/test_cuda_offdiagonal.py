"""Exact seam and topology invariants for the CUDA string-work backend."""

from __future__ import annotations

import numpy as np
import pytest

qaqmc_cuda = pytest.importorskip("qaqmc_cuda")
qaqmc_cpp = pytest.importorskip("qaqmc_cpp")
if not qaqmc_cuda.is_available():
    pytest.skip("CUDA device is not available", allow_module_level=True)


def _engine(n_sites: int, types: np.ndarray, sites: np.ndarray):
    return qaqmc_cuda.DiagonalEngine(
        n_sites=n_sites,
        half_length=len(types) // 2,
        delta_min=0.0,
        delta_max=0.0,
        epsilon=0.01,
        bond_sites=np.empty((0, 2), dtype=np.int32),
        bond_vij=np.empty(0, dtype=np.float64),
        inv_coord=np.ones(n_sites, dtype=np.float64),
        alias_prob=np.ones((1, n_sites), dtype=np.float64),
        alias_index=np.arange(n_sites, dtype=np.int32).reshape(1, -1),
        alias_loc_kind=(np.arange(n_sites, dtype=np.int32) << 1).reshape(1, -1),
        bond_rmax=np.empty((1, 0), dtype=np.float64),
        op_types=np.asarray(types, dtype=np.int32),
        op_sites=np.asarray(sites, dtype=np.int32),
    )


def _unpack(packed: np.ndarray, n_sites: int) -> np.ndarray:
    shifts = np.arange(64, dtype=np.uint64)
    bits = ((packed[:, :, None] >> shifts) & np.uint64(1)).astype(np.uint8)
    return bits.reshape(len(packed), -1)[:, :n_sites]


def _assert_closure(engine, string_sites: list[int]) -> None:
    types, sites = engine.get_operator_string()
    for local, site in enumerate(string_sites):
        parity = int(np.count_nonzero((types == -1) & (sites == site)) & 1)
        assert parity == ((int(engine.seam_mask) >> local) & 1)


def test_seam_aware_profile_matches_serial_worldline_across_tile_boundary():
    n_sites = 70
    length = 1_030
    cut = 257
    types = np.ones(length, dtype=np.int32)
    sites = np.zeros(length, dtype=np.int32)
    for p, site in [(5, 0), (190, 12), (256, 64), (700, 12), (900, 13)]:
        types[p] = -1
        sites[p] = site
    # Sites 0 and 64 both have odd total parity, matching mask 0b11.
    engine = _engine(n_sites, types, sites)
    engine.set_string_sites(np.array([0, 64], dtype=np.int32), cut)
    engine.set_seam_mask_consistent(0b11)

    got = _unpack(np.asarray(engine.profile_states(1)), n_sites)
    state = np.zeros(n_sites, dtype=np.uint8)
    expected = []
    for p, (kind, site) in enumerate(zip(types, sites, strict=True)):
        if kind == -1:
            state[site] ^= 1
        if p + 1 == cut:
            state[0] ^= 1
            state[64] ^= 1
        expected.append(state.copy())
    np.testing.assert_array_equal(got, np.asarray(expected))


def test_device_closure_repair_can_repurpose_diagonal_slots():
    length = 32
    types = np.ones(length, dtype=np.int32)
    sites = np.zeros(length, dtype=np.int32)
    engine = _engine(5, types, sites)
    string_sites = [2, 3]
    engine.set_string_sites(np.asarray(string_sites, dtype=np.int32), length // 2)

    engine.set_seam_mask_consistent(0b11)
    _assert_closure(engine, string_sites)
    repaired_types, repaired_sites = engine.get_operator_string()
    assert np.count_nonzero((repaired_types == -1) & (repaired_sites == 2)) == 1
    assert np.count_nonzero((repaired_types == -1) & (repaired_sites == 3)) == 1

    engine.set_seam_mask_consistent(0)
    _assert_closure(engine, string_sites)


@pytest.mark.parametrize(
    "bad_type,bad_site,message",
    [(7, 0, "unsupported type"), (1, 3, "single-site operator"),
     (2, 0, "bond operator")],
)
def test_operator_upload_rejects_invalid_indices_before_kernel_launch(
    bad_type, bad_site, message
):
    types = np.ones(8, dtype=np.int32)
    sites = np.zeros(8, dtype=np.int32)
    engine = _engine(3, types, sites)
    types[0] = bad_type
    sites[0] = bad_site
    with pytest.raises(ValueError, match=message):
        engine.set_operator_string(types, sites)


def test_zero_bond_topology_sweeps_accept_and_preserve_closure():
    length = 16
    types = np.ones(length, dtype=np.int32)
    sites = np.array([1, 0, 2, 0, 1, 2, 0, 2,
                      1, 0, 2, 0, 1, 2, 0, 2], dtype=np.int32)
    engine = _engine(3, types, sites)
    string_sites = [0, 2]
    engine.set_string_sites(np.asarray(string_sites, dtype=np.int32), length // 2)
    engine.set_seam_mask_consistent(0)

    for sweep in range(20):
        stats = engine.topology_sweep(lambda_=0.5, seed=9001, sweep_id=sweep)
        assert stats["attempts"] == len(string_sites)
        assert stats["accepts"] == len(string_sites)
        assert stats["invalid"] == 0
        assert stats["active_count"] == int(engine.seam_mask).bit_count()
        _assert_closure(engine, string_sites)


def test_cluster_then_topology_keeps_event_cache_physically_consistent():
    length = 24
    types = np.ones(length, dtype=np.int32)
    sites = np.resize(np.array([0, 1, 2, 0, 2, 1], dtype=np.int32), length)
    engine = _engine(3, types, sites)
    string_sites = [0, 2]
    engine.set_string_sites(np.asarray(string_sites, dtype=np.int32), length // 2)
    engine.set_seam_mask_consistent(0)

    for sweep in range(12):
        engine.cluster_update(seed=100 + sweep, sweep_id=sweep)
        engine.topology_sweep(lambda_=0.5, seed=200 + sweep, sweep_id=sweep)
        _assert_closure(engine, string_sites)


def test_device_checkpoint_restores_operator_string_and_seam_exactly():
    length = 24
    types = np.ones(length, dtype=np.int32)
    sites = np.resize(np.array([0, 1, 2, 0, 2, 1], dtype=np.int32), length)
    engine = _engine(3, types, sites)
    string_sites = [0, 2]
    engine.set_string_sites(np.asarray(string_sites, dtype=np.int32), length // 2)
    engine.set_seam_mask_consistent(0b11)
    saved_types, saved_sites = engine.get_operator_string()
    engine.save_checkpoint()
    assert engine.has_checkpoint

    stats = engine.topology_sweep(lambda_=0.5, seed=331, sweep_id=0)
    assert stats["accepts"] == 2
    assert engine.seam_mask == 0
    engine.restore_checkpoint()

    restored_types, restored_sites = engine.get_operator_string()
    np.testing.assert_array_equal(restored_types, saved_types)
    np.testing.assert_array_equal(restored_sites, saved_sites)
    assert engine.seam_mask == 0b11
    _assert_closure(engine, string_sites)


def test_interacting_half_line_proposals_match_cpu_reference():
    n_sites, half_length = 5, 20
    length = 2 * half_length
    cut = 17
    pos = np.arange(n_sites, dtype=np.float64).reshape(-1, 1)
    cpu = qaqmc_cpp.QAQMCEngine(
        n_sites, 1.0, -0.4, 1.7, 1.1, half_length, 0.03, 81, pos,
        neighbor_cutoff=-1, delta_groups=20,
    )
    data = cpu.export_cuda_diagonal_data()
    string_sites = [1, 3]
    types = np.ones(length, dtype=np.int32)
    sites = np.resize(np.arange(n_sites, dtype=np.int32), length)
    for site, positions in {
        0: (2, 10), 1: (4, 24), 2: (7, 27),
        3: (14, 33), 4: (8, 38),
    }.items():
        for p in positions:
            types[p] = -1
            sites[p] = site
    bonds = np.asarray(data["bond_sites"], dtype=np.int32)
    common_bond = int(np.flatnonzero(
        np.all(np.isin(bonds, string_sites), axis=1)
    )[0])
    for p in (5, 6, 9, 11, 13, 15, 16, 18, 19, 21, 22, 23, 26, 29, 31):
        if types[p] == -1:
            continue
        types[p] = 2
        sites[p] = common_bond

    cpu.set_string_sites(string_sites, cut)
    cpu.set_op_string(types, sites)
    cpu.set_seam_mask_consistent(0)
    cpu.recompute_seam_snapshots()
    engine = qaqmc_cuda.DiagonalEngine(
        n_sites=n_sites, half_length=half_length,
        delta_min=-0.4, delta_max=1.7, epsilon=0.03,
        bond_sites=data["bond_sites"], bond_vij=data["bond_vij"],
        inv_coord=data["inv_coord"], alias_prob=data["alias_prob"],
        alias_index=data["alias_index"],
        alias_loc_kind=data["alias_loc_kind"], bond_rmax=data["bond_rmax"],
        op_types=types, op_sites=sites,
    )
    engine.set_string_sites(np.asarray(string_sites, dtype=np.int32), cut)
    engine.set_seam_mask_consistent(0)

    for local_index in range(len(string_sites)):
        for direction_right in (False, True):
            expected = cpu.build_half_line_proposal(local_index, direction_right)
            actual = engine.half_line_proposal(local_index, direction_right)
            assert bool(actual["valid"]) == bool(expected.valid)
            assert int(actual["terminal_p"]) == int(expected.terminal_p)
            if expected.valid:
                assert float(actual["log_physical_ratio"]) == pytest.approx(
                    float(expected.log_physical_ratio), rel=2e-11, abs=2e-11
                )
    # The read-only diagnostic must not mutate the chain.
    got_types, got_sites = engine.get_operator_string()
    np.testing.assert_array_equal(got_types, types)
    np.testing.assert_array_equal(got_sites, sites)
    assert engine.seam_mask == 0
