"""Exact tests for GPU vertex-event construction and bond-spin propagation."""

from __future__ import annotations

import numpy as np
import pytest

qaqmc_cuda = pytest.importorskip("qaqmc_cuda")
if not qaqmc_cuda.is_available():
    pytest.skip("CUDA device is not available", allow_module_level=True)


def _make_event_engine(n_sites, bond_sites, types, sites):
    bond_sites = np.asarray(bond_sites, dtype=np.int32)
    n_bonds = len(bond_sites)
    max_alias = n_sites + n_bonds
    loc_kind = np.empty(max_alias, dtype=np.int32)
    loc_kind[:n_sites] = np.arange(n_sites, dtype=np.int32) << 1
    loc_kind[n_sites:] = (np.arange(n_bonds, dtype=np.int32) << 1) | 1
    # Event construction does not consume alias data; identity tables keep the
    # constructor contract valid and make accidental resampling well-defined.
    return qaqmc_cuda.DiagonalEngine(
        n_sites=n_sites,
        half_length=len(types) // 2,
        delta_min=0.0,
        delta_max=0.0,
        epsilon=0.01,
        bond_sites=bond_sites,
        bond_vij=np.ones(n_bonds, dtype=np.float64),
        inv_coord=np.ones(n_sites, dtype=np.float64),
        alias_prob=np.ones((1, max_alias), dtype=np.float64),
        alias_index=np.arange(max_alias, dtype=np.int32).reshape(1, -1),
        alias_loc_kind=loc_kind.reshape(1, -1),
        bond_rmax=np.ones((1, n_bonds), dtype=np.float64),
        op_types=np.asarray(types, dtype=np.int32),
        op_sites=np.asarray(sites, dtype=np.int32),
    )


def _cpu_events(n_sites, bond_sites, types, sites):
    state = np.zeros(n_sites, dtype=np.int8)
    site_events = []
    bond_events = []
    bond_spin = np.zeros(len(types), dtype=np.int8)
    for p, (kind, loc) in enumerate(zip(types, sites, strict=True)):
        if kind in (-1, 1):
            site_events.append((int(loc), p, p))
            if kind == -1:
                state[loc] ^= 1
        else:
            si, sj = bond_sites[loc]
            bond_spin[p] = 2 * state[si] + state[sj]
            bond_events.append((int(si), p, (p << 32) | (int(loc) << 1)))
            bond_events.append((int(sj), p, (p << 32) | (int(loc) << 1) | 1))
    site_events.sort(key=lambda event: (event[0], event[1]))
    bond_events.sort(key=lambda event: (event[0], event[1]))
    return site_events, bond_events, bond_spin


@pytest.mark.parametrize("length", [256, 258, 1026])
def test_event_streams_and_bond_spins_match_cpu_exactly(length):
    # Even length is required because the engine's schedule has length 2M.
    n_sites = 65
    bond_sites = np.array([[0, 64], [1, 2], [2, 64], [10, 11]], dtype=np.int32)
    rng = np.random.default_rng(5100 + length)
    types = rng.choice(np.array([-1, 1, 2], dtype=np.int32), size=length,
                       p=[0.30, 0.35, 0.35])
    sites = np.empty(length, dtype=np.int32)
    single = types != 2
    sites[single] = rng.integers(0, n_sites, size=int(single.sum()), dtype=np.int32)
    sites[~single] = rng.integers(0, len(bond_sites), size=int((~single).sum()),
                                  dtype=np.int32)

    expected_site, expected_bond, expected_spin = _cpu_events(
        n_sites, bond_sites, types, sites)
    engine = _make_event_engine(n_sites, bond_sites, types, sites)
    result = engine.build_events(download=True)

    assert result["site_events"] == len(expected_site)
    assert result["bond_events"] == len(expected_bond)
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
    np.testing.assert_array_equal(result["bond_spin"][type2], expected_spin[type2])


def test_rebuilding_events_is_repeatable_and_reuses_allocation():
    types = np.array([-1, 2, 1, 2, -1, 1] * 100, dtype=np.int32)
    sites = np.array([0, 0, 1, 1, 2, 2] * 100, dtype=np.int32)
    bonds = np.array([[0, 1], [1, 2]], dtype=np.int32)
    engine = _make_event_engine(3, bonds, types, sites)
    first = engine.build_events(download=True)
    bytes_after_first = engine.device_bytes
    second = engine.build_events(download=True)
    assert engine.device_bytes == bytes_after_first
    for key in ("site_keys", "site_values", "bond_keys", "bond_values", "bond_spin"):
        np.testing.assert_array_equal(first[key], second[key])
