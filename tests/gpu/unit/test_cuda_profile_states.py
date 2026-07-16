"""Exact tests for sparse device-side propagated-state snapshots."""

from __future__ import annotations

import numpy as np
import pytest

qaqmc_cuda = pytest.importorskip("qaqmc_cuda")
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


@pytest.mark.parametrize("n_sites", [1, 64, 65, 216, 384])
@pytest.mark.parametrize("profile_step", [1, 17, 256, 257, 515])
def test_profile_states_match_serial_worldline(n_sites: int, profile_step: int):
    length = 1_030
    rng = np.random.default_rng(1000 + n_sites + profile_step)
    types = np.where(rng.random(length) < 0.31, -1, 1).astype(np.int32)
    sites = rng.integers(0, n_sites, size=length, dtype=np.int32)
    engine = _engine(n_sites, types, sites)

    got = _unpack(np.asarray(engine.profile_states(profile_step)), n_sites)
    state = np.zeros(n_sites, dtype=np.uint8)
    expected = []
    for p, (kind, site) in enumerate(zip(types, sites)):
        if kind == -1:
            state[site] ^= 1
        if (p + 1) % profile_step == 0:
            expected.append(state.copy())
    np.testing.assert_array_equal(got, np.asarray(expected, dtype=np.uint8))


def test_half_length_profile_point_is_midpoint_after_slice_m_minus_one():
    n_sites = 70
    half_length = 600
    types = np.ones(2 * half_length, dtype=np.int32)
    sites = np.zeros(2 * half_length, dtype=np.int32)
    for p, site in [(0, 0), (255, 64), (599, 69), (600, 3)]:
        types[p] = -1
        sites[p] = site
    engine = _engine(n_sites, types, sites)
    states = _unpack(np.asarray(engine.profile_states(half_length)), n_sites)
    expected_midpoint = np.zeros(n_sites, dtype=np.uint8)
    expected_midpoint[[0, 64, 69]] = 1
    np.testing.assert_array_equal(states[0], expected_midpoint)

