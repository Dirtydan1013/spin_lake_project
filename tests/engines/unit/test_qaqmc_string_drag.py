"""Seam-drag (cut-position) primitive: exactness and invariants.

`QAQMCEngine.seam_drag_to(m_new)` claims to return the exact
ln[W_{m_new}(sigma)/W_{m_star}(sigma)] of the current configuration while
moving the cut. The sharp check is a from-scratch full-weight recompute at
both cut positions (site-op weights are constant so only bond ops enter the
ratio) -- the same reference construction as the half-line Test 4
(tests/engines/unit/test_qaqmc_string_halfline.py).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import qaqmc_cpp
from src.rydberg.hamiltonian import build_rydberg_vij
from src.rydberg.lattices import generate_1d_chain


def _make_engine(N=6, M=20, seed=42, delta_groups=40, Omega=1.0,
                 delta_min=0.0, delta_max=1.5, Rb=1.2, epsilon=0.01,
                 neighbor_cutoff=1):
    pos = generate_1d_chain(N)
    eng = qaqmc_cpp.QAQMCEngine(
        N, Omega, delta_min, delta_max, Rb, M, epsilon, seed, pos,
        neighbor_cutoff=neighbor_cutoff, delta_groups=delta_groups,
    )
    return eng, pos


def _compute_bond_W(delta_i, delta_j, vij, epsilon):
    raw = np.array([0.0, delta_j, delta_i, -vij + delta_i + delta_j], dtype=np.float64)
    m_min = raw.min()
    m_abs = np.abs(raw[1:]).min()
    cij = (-m_min if m_min < 0.0 else 0.0) + epsilon * m_abs
    return raw + cij


def _total_bond_log_weight(op_types, op_sites, seam_mask, string_sites, m_star,
                           bond_sites, vij_list, coord_number, delta_schedule, epsilon):
    N = len(coord_number)
    state = np.zeros(N, dtype=np.int64)
    total = 0.0
    for p in range(len(op_types)):
        if p == m_star:
            for k, s in enumerate(string_sites):
                if (seam_mask >> k) & 1:
                    state[s] ^= 1
        ot = op_types[p]
        if ot == -1:
            state[op_sites[p]] ^= 1
        elif ot == 2:
            b = op_sites[p]
            si, sj = bond_sites[b]
            delta = delta_schedule[p]
            di = delta / coord_number[si] if coord_number[si] > 0 else 0.0
            dj = delta / coord_number[sj] if coord_number[sj] > 0 else 0.0
            W = _compute_bond_W(di, dj, vij_list[b], epsilon)
            w = W[state[si] * 2 + state[sj]]
            assert w > 0.0, "configuration must have strictly positive weight at this cut"
            total += np.log(w)
    return total


def _snapshots_match_recompute(eng):
    cm = np.array(eng.state_at_seam_minus).copy()
    cp = np.array(eng.state_at_seam_plus).copy()
    eng.recompute_seam_snapshots()
    return (np.array_equal(cm, np.array(eng.state_at_seam_minus))
            and np.array_equal(cp, np.array(eng.state_at_seam_plus)))


def _closure_ok(eng, sites):
    ot = np.array(eng.op_types)
    os_ = np.array(eng.op_sites)
    for k, s in enumerate(sites):
        n_flip = int(np.sum((ot == -1) & (os_ == s)))
        if (n_flip + ((eng.seam_mask >> k) & 1)) % 2 != 0:
            return False
    return True


def test_drag_matches_full_recompute():
    N, M = 6, 20
    Omega, delta_min, delta_max, Rb, epsilon = 1.0, 0.0, 1.5, 1.2, 0.01
    eng, pos = _make_engine(N=N, M=M, Omega=Omega, delta_min=delta_min,
                            delta_max=delta_max, Rb=Rb, epsilon=epsilon)
    sites = [0, 2, 4]
    eng.set_string_sites(sites, eng.M)
    eng.set_seam_mask_consistent(0b101)
    for _ in range(50):
        eng.mc_step()

    _, _, _, vij_list, bond_sites, coord_number = build_rydberg_vij(
        N, Omega, Rb, pos=pos, verbose=False, neighbor_cutoff=1,
    )
    delta_schedule = np.array(eng.delta_schedule)

    # Chain of drags over a grid spanning both ramps and both boundaries; the
    # op string is frozen between drags (no mc_step) so the brute-force totals
    # stay comparable across the whole chain.
    op_types = np.array(eng.op_types).copy()
    op_sites = np.array(eng.op_sites).copy()
    mask = eng.seam_mask
    for m_new in (27, 39, 13, 0, 5, eng.M):
        m_old = eng.m_star
        total_old = _total_bond_log_weight(
            op_types, op_sites, mask, sites, m_old,
            bond_sites, vij_list, coord_number, delta_schedule, epsilon)
        total_new = _total_bond_log_weight(
            op_types, op_sites, mask, sites, m_new,
            bond_sites, vij_list, coord_number, delta_schedule, epsilon)

        log_ratio = eng.seam_drag_to(m_new)

        assert np.isclose(log_ratio, total_new - total_old, atol=1e-9), (
            m_old, m_new, log_ratio, total_new - total_old)
        assert eng.m_star == m_new
        assert eng.seam_mask == mask
        assert _snapshots_match_recompute(eng), (m_old, m_new)
        # The drag must not touch the configuration itself.
        assert np.array_equal(np.array(eng.op_types), op_types)
        assert np.array_equal(np.array(eng.op_sites), op_sites)


def test_drag_zero_for_empty_mask_and_reversible():
    eng, _ = _make_engine(seed=7)
    sites = [1, 3]
    eng.set_string_sites(sites, eng.M)
    eng.set_seam_mask_consistent(0)
    for _ in range(30):
        eng.mc_step()

    # Empty mask: n^+ == n^- everywhere, every crossing is frame-blind.
    for m_new in (0, 3, 17, 39, eng.M):
        assert eng.seam_drag_to(m_new) == 0.0

    # Full mask: there-and-back on a frozen configuration cancels exactly.
    eng.set_seam_mask_consistent(0b11)
    for _ in range(30):
        eng.mc_step()
    for m_there in (2, 11, 31, 39):
        m_back = eng.m_star
        fwd = eng.seam_drag_to(m_there)
        bwd = eng.seam_drag_to(m_back)
        assert np.isfinite(fwd)
        assert np.isclose(fwd + bwd, 0.0, atol=1e-10), (m_there, fwd, bwd)


def test_drag_interleaved_with_mc_preserves_closure():
    eng, _ = _make_engine(N=6, M=16, seed=11)
    sites = [1, 3]
    eng.set_string_sites(sites, eng.M)
    eng.set_seam_mask_consistent(0b11)
    rng = np.random.default_rng(5)
    for it in range(200):
        m_new = int(rng.integers(0, 2 * 16))
        log_ratio = eng.seam_drag_to(m_new)
        assert np.isfinite(log_ratio), f"unexpected zero-weight crossing at iter {it}"
        assert _snapshots_match_recompute(eng), it
        assert _closure_ok(eng, sites), f"drag broke worldline closure (iter {it})"
        eng.mc_step()
        assert _closure_ok(eng, sites), f"mc_step at m_star={m_new} broke closure (iter {it})"


def test_seam_set_position_reanchors_without_work():
    eng, _ = _make_engine(seed=3)
    sites = [2, 4]
    eng.set_string_sites(sites, eng.M)
    eng.set_seam_mask_consistent(0b11)
    for _ in range(30):
        eng.mc_step()

    op_types = np.array(eng.op_types).copy()
    mask = eng.seam_mask
    for m_new in (0, 9, 39):
        eng.seam_set_position(m_new)
        assert eng.m_star == m_new
        assert eng.seam_mask == mask
        assert _snapshots_match_recompute(eng)
        assert np.array_equal(np.array(eng.op_types), op_types)

    # Out-of-range and unconfigured-string calls must raise, not corrupt.
    for bad in (-1, 2 * 20):
        try:
            eng.seam_set_position(bad)
            assert False, "expected ValueError for out-of-range m_new"
        except ValueError:
            pass
        try:
            eng.seam_drag_to(bad)
            assert False, "expected ValueError for out-of-range m_new"
        except ValueError:
            pass


def test_rung_rb_ratio_matches_bruteforce():
    # seam_rung_rb_ratio == Lambda_tgt/Lambda_cur with Lambda(s) = N*Omega/2
    # + sum_b W_b(s), the stationary conditional of diagonal_update's op menu
    # at the crossing slot; 1.0 exactly on flip slots.
    N, M = 6, 20
    Omega, Rb, epsilon = 1.0, 1.2, 0.01
    eng, pos = _make_engine(N=N, M=M, Omega=Omega, Rb=Rb, epsilon=epsilon, seed=17)
    sites = [1, 3]
    eng.set_string_sites(sites, eng.M)
    eng.set_seam_mask_consistent(0b11)
    for _ in range(50):
        eng.mc_step()

    _, _, _, vij_list, bond_sites, coord_number = build_rydberg_vij(
        N, Omega, Rb, pos=pos, verbose=False, neighbor_cutoff=1,
    )
    delta_schedule = np.array(eng.delta_schedule)

    def lam(state, p):
        total = N * Omega / 2.0
        for b, (si, sj) in enumerate(bond_sites):
            delta = delta_schedule[p]
            di = delta / coord_number[si] if coord_number[si] > 0 else 0.0
            dj = delta / coord_number[sj] if coord_number[sj] > 0 else 0.0
            W = _compute_bond_W(di, dj, vij_list[b], epsilon)
            total += W[state[si] * 2 + state[sj]]
        return total

    rng = np.random.default_rng(23)
    n_diag_checked = n_flip_checked = 0
    for it in range(120):
        eng.mc_step()
        eng.seam_set_position(int(rng.integers(1, 2 * M - 1)))
        op_types = np.array(eng.op_types)
        eng.recompute_seam_snapshots()
        s_minus = np.array(eng.state_at_seam_minus)
        s_plus = np.array(eng.state_at_seam_plus)
        for right in (True, False):
            p = eng.m_star if right else eng.m_star - 1
            got = eng.seam_rung_rb_ratio(right)
            if op_types[p] == -1:
                assert got == 1.0, (it, right, got)
                n_flip_checked += 1
            else:
                lam_minus, lam_plus = lam(s_minus, p), lam(s_plus, p)
                want = lam_minus / lam_plus if right else lam_plus / lam_minus
                assert np.isclose(got, want, rtol=1e-12), (it, right, got, want)
                n_diag_checked += 1
    assert n_diag_checked >= 50 and n_flip_checked >= 5


def test_block_rb_log_ratio_composes_and_is_readonly():
    # seam_rb_log_ratio_to(m') must equal the sum of per-slot RB log ratios
    # collected by stepping the cut one slot at a time (the factorization is
    # exact given the worldline), and must not move the cut or change the
    # configuration.
    eng, _ = _make_engine(N=6, M=20, seed=29)
    sites = [1, 3]
    eng.set_string_sites(sites, eng.M)
    eng.set_seam_mask_consistent(0b11)
    for _ in range(50):
        eng.mc_step()

    op_types = np.array(eng.op_types).copy()
    for m_target in (28, 39, 9, 0):
        m0 = eng.m_star
        block = eng.seam_rb_log_ratio_to(m_target)
        assert eng.m_star == m0
        assert np.array_equal(np.array(eng.op_types), op_types)
        assert _snapshots_match_recompute(eng)

        # per-slot composition on a frozen configuration
        step = 1 if m_target > m0 else -1
        total = 0.0
        while eng.m_star != m_target:
            total += np.log(eng.seam_rung_rb_ratio(step > 0))
            eng.seam_drag_to(eng.m_star + step)
        assert np.isclose(block, total, atol=1e-10), (m0, m_target, block, total)
    # zero-distance and empty-mask invariants
    assert eng.seam_rb_log_ratio_to(eng.m_star) == 0.0
    eng.set_seam_mask_consistent(0)
    for _ in range(10):
        eng.mc_step()
    for m_target in (3, 37):
        assert eng.seam_rb_log_ratio_to(m_target) == 0.0


def test_drag_orchestration_smoke():
    # Wrapper-level: shapes, grid validation, anchor override, diagnostics.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    from src.engines.qaqmc_string_work import QAQMCStringWorkRydberg

    N, M = 5, 16
    eng = QAQMCStringWorkRydberg(N=N, M=M, Omega=1.0, Rb=1.2, delta_min=0.0,
                                 delta_max=1.5, epsilon=0.01, seed=99,
                                 pos=generate_1d_chain(N), neighbor_cutoff=1,
                                 delta_groups=32)
    eng.set_string_sites([2])
    eng.thermalize(100, direction="reverse")  # full-mask sector at the anchor
    res = eng.run_drag_trajectories(np.array([20, 24]), n_trajectories=4,
                                    decorrelation_steps=10,
                                    n_qaqmc_sweeps_per_shift=1)
    assert res.m_anchor == M
    assert res.log_j_samples.shape == (4, 2)
    assert np.all(np.isfinite(res.log_r))
    assert np.all(res.n_eff > 0) and np.all(res.zero_weight_fraction == 0.0)

    # Anchor override re-anchors even when the cut is parked at the far end.
    res2 = eng.run_drag_trajectories(np.array([12, 8]), n_trajectories=2,
                                     decorrelation_steps=5, m_anchor=M)
    assert res2.m_anchor == M

    for bad in ([], [5, 5], [3, 7, 6], [2 * M]):
        try:
            eng.run_drag_trajectories(np.array(bad, dtype=np.int64), 1,
                                      decorrelation_steps=1)
            assert False, f"expected ValueError for grid {bad}"
        except ValueError:
            pass


def main():
    test_drag_matches_full_recompute()
    print("drag == from-scratch weight ratio (chained grid, both directions) passed")
    test_drag_zero_for_empty_mask_and_reversible()
    print("empty-mask zero + there-and-back reversibility passed")
    test_drag_interleaved_with_mc_preserves_closure()
    print("drag/mc_step interleave preserves closure + snapshot cache passed")
    test_seam_set_position_reanchors_without_work()
    print("seam_set_position re-anchor passed")
    test_rung_rb_ratio_matches_bruteforce()
    print("RB rung ratio == brute-force Lambda ratio passed")
    test_block_rb_log_ratio_composes_and_is_readonly()
    print("block RB == per-slot composition, read-only passed")
    test_drag_orchestration_smoke()
    print("drag orchestration smoke passed")


if __name__ == "__main__":
    main()
