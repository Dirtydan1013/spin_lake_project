"""Exhaustive CPU proof cases for the compact CUDA Renyi topology theorem."""

from __future__ import annotations

import numpy as np
import pytest

qaqmc_cpp = pytest.importorskip("qaqmc_cpp")


def _channel_terminal(
    before_cut: tuple[int, int], terminal: tuple[int, int], channel: int, mask: int
) -> int:
    source = channel ^ mask
    return before_cut[channel] ^ terminal[source] ^ before_cut[source]


def _topology_is_closed(
    before_cut: tuple[int, int], terminal: tuple[int, int], mask: int
) -> bool:
    return all(
        _channel_terminal(before_cut, terminal, channel, mask) == 0
        for channel in (0, 1)
    )


def _brute_channel_terminal(
    flips: np.ndarray, cut: int, mask: np.ndarray
) -> np.ndarray:
    channels = np.zeros((2, flips.shape[2]), dtype=np.uint8)
    for p in range(flips.shape[1]):
        for replica in (0, 1):
            active = np.flatnonzero(flips[replica, p])
            for site in active:
                channel = replica ^ (int(mask[site]) if p >= cut else 0)
                channels[channel, site] ^= 1
    return channels


def test_compact_boundary_formula_matches_bruteforce_channel_paths():
    """Randomly validate the exact formula used by the one-thread CUDA kernel."""
    rng = np.random.default_rng(91273)
    n_sites = 7
    length = 19
    for _ in range(500):
        flips = rng.integers(0, 2, size=(2, length, n_sites), dtype=np.uint8)
        cut = int(rng.integers(0, length + 1))
        mask = rng.integers(0, 2, size=n_sites, dtype=np.uint8)
        before = np.bitwise_xor.reduce(flips[:, :cut], axis=1, initial=0)
        terminal = np.bitwise_xor.reduce(flips, axis=1, initial=0)

        compact = np.empty((2, n_sites), dtype=np.uint8)
        for site in range(n_sites):
            boundaries = (
                (int(before[0, site]), int(before[1, site])),
                (int(terminal[0, site]), int(terminal[1, site])),
            )
            for channel in (0, 1):
                compact[channel, site] = _channel_terminal(
                    boundaries[0], boundaries[1], channel, int(mask[site])
                )
        np.testing.assert_array_equal(
            compact, _brute_channel_terminal(flips, cut, mask)
        )

        target = int(rng.integers(0, n_sites))
        toggled = mask.copy()
        toggled[target] ^= 1
        current_closed = not np.any(compact[:, target])
        proposed_closed = not np.any(
            _brute_channel_terminal(flips, cut, toggled)[:, target]
        )
        if current_closed and proposed_closed:
            assert before[0, target] == before[1, target]


def test_valid_single_site_toggle_has_zero_ratio_and_needs_no_reprojection():
    """Enumerate all two-replica flip strings for M=2 and both mask sectors.

    Simultaneous closure of the current and toggled topology forces equal cut
    occupations in both replicas.  Consequently changing the post-cut channel
    label cannot alter an actual-replica path or a diagonal bond weight.
    """
    half_length = 2
    length = 2 * half_length
    engine = qaqmc_cpp.QAQMCRenyiEngine(
        N=1,
        Omega=1.0,
        delta_min=-0.4,
        delta_max=0.8,
        Rb=0.0,
        M=half_length,
        epsilon=0.03,
        seed=17,
        pos=np.zeros((1, 1), dtype=np.float64),
        neighbor_cutoff=-1,
        delta_groups=4,
    )
    engine.set_mode(2)
    sites = np.zeros(length, dtype=np.int32)
    checked = 0

    for bits0 in range(1 << length):
        flips0 = tuple((bits0 >> p) & 1 for p in range(length))
        for bits1 in range(1 << length):
            flips1 = tuple((bits1 >> p) & 1 for p in range(length))
            before_cut = (
                sum(flips0[:half_length]) & 1,
                sum(flips1[:half_length]) & 1,
            )
            terminal = (sum(flips0) & 1, sum(flips1) & 1)
            types = [
                np.asarray([(-1 if bit else 1) for bit in flips0], dtype=np.int32),
                np.asarray([(-1 if bit else 1) for bit in flips1], dtype=np.int32),
            ]

            for mask in (0, 1):
                if not _topology_is_closed(before_cut, terminal, mask):
                    continue
                if not _topology_is_closed(before_cut, terminal, mask ^ 1):
                    continue

                engine.set_A_mask(np.asarray([mask], dtype=np.uint8))
                for replica in (0, 1):
                    engine.set_replica_op_string(replica, types[replica], sites)
                engine.recompute_midpoint_states()

                ratio = float(engine.log_weight_ratio_for_toggle(0))
                assert ratio == pytest.approx(0.0, abs=1e-15)
                before = [
                    np.asarray(engine.get_op_types(replica), dtype=np.int32).copy()
                    for replica in (0, 1)
                ]
                engine.apply_single_bit_toggle(0)
                for replica in (0, 1):
                    np.testing.assert_array_equal(engine.get_op_types(replica), before[replica])
                np.testing.assert_array_equal(engine.A_mask, [mask ^ 1])
                checked += 1

    assert checked == 64


def _segment_with_parity(rng: np.random.Generator, positions: tuple[int, int], parity: int):
    if parity:
        return {positions[int(rng.integers(0, 2))]}
    return set() if rng.integers(0, 2) == 0 else set(positions)


def test_compact_theorem_keeps_interacting_bond_weights_exactly_invariant():
    """Exercise the theorem with unequal replica paths and actual bond ops."""
    n_sites = 2
    half_length = 6
    length = 2 * half_length
    engine = qaqmc_cpp.QAQMCRenyiEngine(
        N=n_sites,
        Omega=1.0,
        delta_min=0.4,
        delta_max=1.6,
        Rb=1.1,
        M=half_length,
        epsilon=0.04,
        seed=29,
        pos=np.arange(n_sites, dtype=np.float64).reshape(-1, 1),
        neighbor_cutoff=-1,
        delta_groups=12,
    )
    engine.set_mode(2)
    bonds = np.asarray(engine.bond_sites, dtype=np.int32)
    matches = np.flatnonzero(np.all(bonds == np.array([0, 1]), axis=1))
    assert len(matches) == 1
    bond = int(matches[0])
    rng = np.random.default_rng(4401)

    # Disjoint slots let both physical sites independently satisfy: equal cut
    # occupations across replicas and even total actual-replica parity.
    slots = {
        0: ((0, 1), (6, 7)),
        1: ((2, 3), (8, 9)),
    }
    bond_slots = (4, 5, 10, 11)
    for _ in range(100):
        cut_parity = rng.integers(0, 2, size=n_sites)
        types = np.ones((2, length), dtype=np.int32)
        sites = np.vstack([
            np.arange(length, dtype=np.int32) % n_sites,
            (np.arange(length, dtype=np.int32) + 1) % n_sites,
        ])
        for replica in (0, 1):
            for site in range(n_sites):
                pre, post = slots[site]
                flips = _segment_with_parity(rng, pre, int(cut_parity[site]))
                flips |= _segment_with_parity(rng, post, int(cut_parity[site]))
                for p in flips:
                    types[replica, p] = -1
                    sites[replica, p] = site
            for p in bond_slots:
                types[replica, p] = 2
                sites[replica, p] = bond

        mask = rng.integers(0, 2, size=n_sites, dtype=np.uint8)
        engine.set_A_mask(mask)
        for replica in (0, 1):
            engine.set_replica_op_string(replica, types[replica], sites[replica])
        engine.recompute_midpoint_states()

        paths = engine.get_site_paths(0)
        assert paths["channel_0"][-1] == 0
        assert paths["channel_1"][-1] == 0
        ratio = float(engine.log_weight_ratio_for_toggle(0))
        assert ratio == pytest.approx(0.0, abs=2e-13)
        before = [
            np.asarray(engine.get_op_types(replica), dtype=np.int32).copy()
            for replica in (0, 1)
        ]
        engine.apply_single_bit_toggle(0)
        for replica in (0, 1):
            np.testing.assert_array_equal(engine.get_op_types(replica), before[replica])
