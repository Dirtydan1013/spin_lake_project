"""Exact channel-event and cluster tests for the two-replica CUDA backend."""

from __future__ import annotations

import numpy as np
import pytest

qaqmc_cuda = pytest.importorskip("qaqmc_cuda")
qaqmc_cpp = pytest.importorskip("qaqmc_cpp")
if not qaqmc_cuda.is_available():
    pytest.skip("CUDA device is not available", allow_module_level=True)


def _engine(n_sites, bond_sites, types, sites):
    bond_sites = np.asarray(bond_sites, dtype=np.int32)
    n_bonds = len(bond_sites)
    max_alias = n_sites + n_bonds
    loc_kind = np.empty(max_alias, dtype=np.int32)
    loc_kind[:n_sites] = np.arange(n_sites, dtype=np.int32) << 1
    loc_kind[n_sites:] = (np.arange(n_bonds, dtype=np.int32) << 1) | 1
    return qaqmc_cuda.RenyiEngine(
        n_sites=n_sites,
        half_length=types.shape[1] // 2,
        delta_min=0.2,
        delta_max=0.7,
        epsilon=0.01,
        bond_sites=bond_sites,
        bond_vij=np.ones(n_bonds, dtype=np.float64) * 0.4,
        inv_coord=np.ones(n_sites, dtype=np.float64),
        alias_prob=np.ones((1, max_alias), dtype=np.float64),
        alias_index=np.arange(max_alias, dtype=np.int32).reshape(1, -1),
        alias_loc_kind=loc_kind.reshape(1, -1),
        bond_rmax=np.ones((1, n_bonds), dtype=np.float64),
        op_types=np.asarray(types, dtype=np.int32),
        op_sites=np.asarray(sites, dtype=np.int32),
    )


def _cpu_events(n_sites, bond_sites, types, sites, mask, cut):
    length = types.shape[1]
    state = np.zeros((2, n_sites), dtype=np.int8)
    site_events = []
    bond_events = []
    bond_spin = np.zeros((2, length), dtype=np.int8)

    def channel(replica, site, p):
        return replica ^ (int(mask[site]) if p >= cut else 0)

    for p in range(length):
        for replica in range(2):
            kind = int(types[replica, p])
            loc = int(sites[replica, p])
            if kind in (-1, 1):
                ch = channel(replica, loc, p)
                group = ch * n_sites + loc
                site_events.append((group, p, (p << 1) | replica))
                if kind == -1:
                    state[ch, loc] ^= 1
            else:
                si, sj = bond_sites[loc]
                ci = channel(replica, int(si), p)
                cj = channel(replica, int(sj), p)
                bond_spin[replica, p] = 2 * state[ci, si] + state[cj, sj]
                packed = (p << 2) | (replica << 1)
                bond_events.append((ci * n_sites + int(si), p, packed))
                bond_events.append((cj * n_sites + int(sj), p, packed | 1))
    site_events.sort(key=lambda event: (event[0], event[1]))
    bond_events.sort(key=lambda event: (event[0], event[1]))
    return site_events, bond_events, bond_spin


@pytest.mark.parametrize("n_sites,length,cut", [(5, 514, 257), (65, 1_030, 511)])
def test_channel_events_and_bond_spins_match_cpu_exactly(n_sites, length, cut):
    rng = np.random.default_rng(7100 + n_sites)
    bonds = np.array([[0, n_sites - 1], [1, 2], [2, n_sites - 1]], dtype=np.int32)
    types = rng.choice(np.array([-1, 1, 2], dtype=np.int32),
                       size=(2, length), p=[0.28, 0.34, 0.38])
    sites = np.empty((2, length), dtype=np.int32)
    single = types != 2
    sites[single] = rng.integers(0, n_sites, size=int(single.sum()), dtype=np.int32)
    sites[~single] = rng.integers(0, len(bonds), size=int((~single).sum()),
                                  dtype=np.int32)
    mask = rng.integers(0, 2, size=n_sites, dtype=np.uint8)
    expected_site, expected_bond, expected_spin = _cpu_events(
        n_sites, bonds, types, sites, mask, cut
    )

    engine = _engine(n_sites, bonds, types, sites)
    engine.set_cut(cut)
    engine.set_mask(mask)
    result = engine.build_events(download=True)
    actual_site = [
        (int(key >> np.uint64(32)), int(key & np.uint64(0xFFFFFFFF)), int(value))
        for key, value in zip(result["site_keys"], result["site_values"], strict=True)
    ]
    actual_bond = [
        (int(key >> np.uint64(32)), int(key & np.uint64(0xFFFFFFFF)), int(value))
        for key, value in zip(result["bond_keys"], result["bond_values"], strict=True)
    ]
    assert actual_site == expected_site
    assert actual_bond == expected_bond
    type2 = types == 2
    got_spin = np.asarray(result["bond_spin"]).reshape(2, length)
    np.testing.assert_array_equal(got_spin[type2], expected_spin[type2])


def test_site_only_cluster_toggles_channel_path_boundaries_exactly():
    n_sites = 4
    length = 24
    types = np.ones((2, length), dtype=np.int32)
    sites = np.vstack([
        np.resize(np.array([0, 1, 2, 0, 3, 1], dtype=np.int32), length),
        np.resize(np.array([1, 0, 3, 2, 0, 2], dtype=np.int32), length),
    ])
    mask = np.array([1, 0, 1, 0], dtype=np.uint8)
    cut = 11
    engine = _engine(n_sites, np.empty((0, 2), dtype=np.int32), types, sites)
    engine.set_cut(cut)
    engine.set_mask(mask)
    stats = engine.cluster_update(seed=90210, sweep_id=3)
    got_types, got_sites = engine.get_operator_strings()
    np.testing.assert_array_equal(got_sites, sites)

    expected = types.copy()
    proposed = 0
    for site in range(n_sites):
        for channel in range(2):
            events = []
            for p in range(length):
                for replica in range(2):
                    if sites[replica, p] != site:
                        continue
                    mapped = replica ^ (int(mask[site]) if p >= cut else 0)
                    if mapped == channel:
                        events.append((p, replica))
            proposed += max(len(events) - 1, 0)
            if len(events) >= 2:
                p, replica = events[0]
                expected[replica, p] = -1
                p, replica = events[-1]
                expected[replica, p] = -1
    np.testing.assert_array_equal(got_types, expected)
    assert stats["proposed_segments"] == proposed
    assert stats["accepted_segments"] == proposed


def test_diagonal_update_preserves_both_replicas_offdiagonal_slots():
    n_sites = 5
    length = 258
    rng = np.random.default_rng(88)
    types = np.ones((2, length), dtype=np.int32)
    sites = rng.integers(0, n_sites, size=(2, length), dtype=np.int32)
    offdiag = rng.random((2, length)) < 0.22
    types[offdiag] = -1
    engine = _engine(n_sites, np.empty((0, 2), dtype=np.int32), types, sites)
    engine.set_mask(np.array([0, 1, 0, 1, 1], dtype=np.uint8))
    stats = engine.diagonal_update(seed=123, sweep_id=4)
    got_types, got_sites = engine.get_operator_strings()
    assert stats["updated_slots"] == int((~offdiag).sum())
    assert stats["failed_slots"] == 0
    np.testing.assert_array_equal(got_types[offdiag], -1)
    np.testing.assert_array_equal(got_sites[offdiag], sites[offdiag])


def test_two_replica_upload_rejects_corrupt_warm_config_before_kernel_launch():
    types = np.ones((2, 8), dtype=np.int32)
    sites = np.zeros((2, 8), dtype=np.int32)
    engine = _engine(2, np.empty((0, 2), dtype=np.int32), types, sites)
    sites[1, 7] = 2
    with pytest.raises(ValueError, match="single-site operator"):
        engine.set_operator_strings(types, sites)


def test_two_replica_constructor_rejects_corrupt_initial_config():
    types = np.ones((2, 8), dtype=np.int32)
    sites = np.zeros((2, 8), dtype=np.int32)
    types[0, 3] = 9
    with pytest.raises(ValueError, match="unsupported type"):
        _engine(2, np.empty((0, 2), dtype=np.int32), types, sites)


def test_zero_bond_topology_toggle_and_reprojection_are_exact():
    n_sites = 4
    length = 32
    types = np.ones((2, length), dtype=np.int32)
    sites = np.vstack([
        np.resize(np.arange(n_sites, dtype=np.int32), length),
        np.resize(np.arange(n_sites - 1, -1, -1, dtype=np.int32), length),
    ])
    engine = _engine(n_sites, np.empty((0, 2), dtype=np.int32), types, sites)
    engine.set_cut(13)
    engine.set_mask(np.zeros(n_sites, dtype=np.uint8))
    targets = np.array([0, 2], dtype=np.int32)

    first = engine.topology_sweep(
        topology_sites=targets, lambda_=0.5, seed=771, sweep_id=0
    )
    assert first["attempts"] == 2
    assert first["accepts"] == 2
    assert first["invalid"] == 0
    assert first["active_count"] == 2
    np.testing.assert_array_equal(engine.get_mask(), [1, 0, 1, 0])
    got_types, got_sites = engine.get_operator_strings()
    np.testing.assert_array_equal(got_types, types)
    np.testing.assert_array_equal(got_sites, sites)

    second = engine.topology_sweep(
        topology_sites=targets, lambda_=0.5, seed=772, sweep_id=1
    )
    assert second["accepts"] == 2
    assert second["active_count"] == 0
    np.testing.assert_array_equal(engine.get_mask(), np.zeros(n_sites, dtype=np.uint8))


def test_compact_topology_boundaries_reject_cross_channel_mismatch():
    length = 16
    types = np.ones((2, length), dtype=np.int32)
    sites = np.zeros((2, length), dtype=np.int32)
    # Replica 0 has one flip on each side of the cut: its own worldline closes,
    # as does replica 1's vacuum path, but exchanging their post-cut paths does
    # not close because the cut occupations differ.
    types[0, [2, 10]] = -1
    engine = _engine(1, np.empty((0, 2), dtype=np.int32), types, sites)
    engine.set_cut(8)
    engine.set_mask(np.zeros(1, dtype=np.uint8))

    ratio = engine.log_weight_ratio_for_toggle(0)
    assert ratio["current_valid"]
    assert not ratio["proposed_valid"]
    assert ratio["log_ratio"] <= -1e29
    stats = engine.topology_sweep(
        topology_sites=np.array([0], dtype=np.int32),
        lambda_=0.5, seed=10, sweep_id=0,
    )
    assert stats["attempts"] == 1
    assert stats["accepts"] == 0
    assert stats["invalid"] == 1
    np.testing.assert_array_equal(engine.get_mask(), [0])
    got_types, got_sites = engine.get_operator_strings()
    np.testing.assert_array_equal(got_types, types)
    np.testing.assert_array_equal(got_sites, sites)


@pytest.mark.parametrize("cut", [0, 16])
def test_compact_topology_supports_endpoint_cuts(cut):
    types = np.ones((2, 16), dtype=np.int32)
    sites = np.zeros((2, 16), dtype=np.int32)
    engine = _engine(1, np.empty((0, 2), dtype=np.int32), types, sites)
    engine.set_cut(cut)
    ratio = engine.log_weight_ratio_for_toggle(0)
    assert ratio["current_valid"] and ratio["proposed_valid"]
    assert ratio["log_ratio"] == 0.0
    stats = engine.topology_sweep(
        topology_sites=np.array([0], dtype=np.int32),
        lambda_=0.5, seed=11, sweep_id=cut,
    )
    assert stats["accepts"] == 1


def test_device_checkpoint_restore_is_exact_for_both_replicas():
    n_sites = 3
    length = 24
    types = np.ones((2, length), dtype=np.int32)
    sites = np.vstack([
        np.resize(np.array([0, 1, 2], dtype=np.int32), length),
        np.resize(np.array([2, 0, 1], dtype=np.int32), length),
    ])
    engine = _engine(n_sites, np.empty((0, 2), dtype=np.int32), types, sites)
    engine.save_checkpoint()
    assert engine.has_checkpoint
    engine.cluster_update(seed=99, sweep_id=2)
    changed_types, _ = engine.get_operator_strings()
    assert np.any(changed_types != types)
    engine.restore_checkpoint()
    restored_types, restored_sites = engine.get_operator_strings()
    np.testing.assert_array_equal(restored_types, types)
    np.testing.assert_array_equal(restored_sites, sites)


def test_interacting_topology_log_ratios_match_cpu_reference():
    """The most sensitive detailed-balance quantity agrees configuration-wise."""
    n_sites, half_length = 4, 12
    length = 2 * half_length
    pos = np.arange(n_sites, dtype=np.float64).reshape(-1, 1)
    params = dict(
        N=n_sites, Omega=1.0, delta_min=-0.7, delta_max=1.8,
        Rb=1.15, M=half_length, epsilon=0.03, seed=91, pos=pos,
        neighbor_cutoff=-1, delta_groups=12,
    )
    model = qaqmc_cpp.QAQMCEngine(**params)
    reference = qaqmc_cpp.QAQMCRenyiEngine(**params)
    data = model.export_cuda_diagonal_data()

    types = np.ones((2, length), dtype=np.int32)
    sites = np.vstack([
        np.resize(np.arange(n_sites, dtype=np.int32), length),
        np.resize(np.arange(n_sites - 1, -1, -1, dtype=np.int32), length),
    ])
    # Every actual-replica path has even parity both before and after the cut,
    # so current and proposed channel topologies are closed.  Intervals differ
    # between replicas and bonds touch the target, exercising the closure
    # theorem behind the compact GPU ratio path (valid toggle => log ratio 0).
    for replica, positions in enumerate(((13, 21), (15, 19))):
        for p in positions:
            types[replica, p] = -1
            sites[replica, p] = 0
    for replica, positions in enumerate(((2, 8), (4, 10))):
        for p in positions:
            types[replica, p] = -1
            sites[replica, p] = 1

    bonds = np.asarray(data["bond_sites"], dtype=np.int32)
    touching = int(np.flatnonzero(np.any(bonds == 0, axis=1))[0])
    for replica, positions in enumerate(((14, 16, 18, 20), (13, 17, 20, 22))):
        for p in positions:
            types[replica, p] = 2
            sites[replica, p] = touching

    mask = np.array([0, 0, 1, 0], dtype=np.uint8)
    reference.set_A_mask(mask)
    reference.set_mode(2)
    for replica in range(2):
        reference.set_replica_op_string(replica, types[replica], sites[replica])
    reference.recompute_midpoint_states()

    engine = qaqmc_cuda.RenyiEngine(
        n_sites=n_sites, half_length=half_length,
        delta_min=params["delta_min"], delta_max=params["delta_max"],
        epsilon=params["epsilon"], bond_sites=data["bond_sites"],
        bond_vij=data["bond_vij"], inv_coord=data["inv_coord"],
        alias_prob=data["alias_prob"], alias_index=data["alias_index"],
        alias_loc_kind=data["alias_loc_kind"], bond_rmax=data["bond_rmax"],
        op_types=types, op_sites=sites,
    )
    engine.set_mask(mask)

    for site in range(n_sites):
        cpu_ratio = float(reference.log_weight_ratio_for_toggle(site))
        gpu = engine.log_weight_ratio_for_toggle(site)
        assert gpu["current_valid"]
        assert gpu["proposed_valid"]
        assert cpu_ratio == pytest.approx(0.0, abs=2e-12)
        assert float(gpu["log_ratio"]) == pytest.approx(
            cpu_ratio, rel=2e-12, abs=2e-12
        )
    np.testing.assert_array_equal(engine.get_mask(), mask)
    got_types, got_sites = engine.get_operator_strings()
    np.testing.assert_array_equal(got_types, types)
    np.testing.assert_array_equal(got_sites, sites)
