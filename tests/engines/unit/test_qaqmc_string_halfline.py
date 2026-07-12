import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import qaqmc_cpp
from src.analysis.ed_core import qaqmc_exact_string_zratio
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


# ── Test 2: half-line involution (T_{i,s}^2 = 1) ────────────────────────────

def _test_half_line_involution():
    eng, _ = _make_engine()
    eng.set_string_sites([1, 3], eng.M)
    eng.set_seam_mask(0)
    for _ in range(30):
        eng.mc_step()

    orig_types = np.array(eng.op_types).copy()
    orig_sites = np.array(eng.op_sites).copy()
    orig_mask = eng.seam_mask

    # Not every (site, direction) combination is guaranteed to find a
    # terminal before the boundary (single-site vertices can be sparse when
    # bond weights dominate) -- that is a legitimate "invalid: reject"
    # outcome, not a bug. Only assert the involution property T_{i,s}^2=1 on
    # combinations that *are* valid, but require that at least some are.
    n_valid = 0
    for local_index in (0, 1):
        for direction_right in (True, False):
            prop1 = eng.build_half_line_proposal(local_index, direction_right)
            if not prop1.valid:
                continue
            n_valid += 1
            eng.commit_half_line_proposal(local_index, prop1)

            prop2 = eng.build_half_line_proposal(local_index, direction_right)
            assert prop2.valid
            assert prop2.terminal_p == prop1.terminal_p
            eng.commit_half_line_proposal(local_index, prop2)

            assert np.array_equal(np.array(eng.op_types), orig_types)
            assert np.array_equal(np.array(eng.op_sites), orig_sites)
            assert eng.seam_mask == orig_mask

    assert n_valid >= 2, "expected at least two valid (site, direction) half-lines to test involution on"


# ── Test 4: incremental log-ratio vs from-scratch full weight recompute ─────

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
            assert w > 0.0, "sampled configuration must have strictly positive weight"
            total += np.log(w)
    return total


def _test_incremental_ratio_matches_full_recompute():
    N, M = 6, 20
    Omega, delta_min, delta_max, Rb, epsilon = 1.0, 0.0, 1.5, 1.2, 0.01
    eng, pos = _make_engine(N=N, M=M, Omega=Omega, delta_min=delta_min,
                            delta_max=delta_max, Rb=Rb, epsilon=epsilon)
    eng.set_string_sites([0, 2, 4], eng.M)
    eng.set_seam_mask(0b101)  # start with a non-trivial sector, not just empty
    for _ in range(30):
        eng.mc_step()

    _, _, _, vij_list, bond_sites, coord_number = build_rydberg_vij(
        N, Omega, Rb, pos=pos, verbose=False, neighbor_cutoff=1,
    )
    delta_schedule = np.array(eng.delta_schedule)

    n_checked = 0
    for local_index in (0, 1, 2):
        for direction_right in (True, False):
            op_types_before = np.array(eng.op_types).copy()
            op_sites_before = np.array(eng.op_sites).copy()
            seam_before = eng.seam_mask
            string_sites = list(eng.string_sites)
            m_star = eng.m_star

            prop = eng.build_half_line_proposal(local_index, direction_right)
            if not prop.valid:
                continue  # ran into the boundary; nothing to cross-check here

            total_before = _total_bond_log_weight(
                op_types_before, op_sites_before, seam_before, string_sites, m_star,
                bond_sites, vij_list, coord_number, delta_schedule, epsilon)

            eng.commit_half_line_proposal(local_index, prop)

            op_types_after = np.array(eng.op_types).copy()
            seam_after = eng.seam_mask
            total_after = _total_bond_log_weight(
                op_types_after, op_sites_before, seam_after, string_sites, m_star,
                bond_sites, vij_list, coord_number, delta_schedule, epsilon)

            assert np.isclose(total_after - total_before, prop.log_physical_ratio, atol=1e-9), (
                local_index, direction_right, total_after - total_before, prop.log_physical_ratio)
            n_checked += 1

            # Restore the pre-toggle configuration so the next (local_index,
            # direction) check starts from a clean, independently-verified
            # state rather than compounding on top of this one.
            eng.set_op_string(op_types_before.astype(np.int32), op_sites_before.astype(np.int32))
            eng.set_seam_mask(int(seam_before))
            eng.recompute_seam_snapshots()  # sync caches with the restored op string

    assert n_checked >= 3, "expected at least a few valid half-line proposals to be checked"


# ── Seam-cache consistency: cached snapshots == definitional recompute ──────

def _test_seam_cache_stays_consistent():
    # Two historical bugs lived here (fixed 2026-07-12): commit always flipped
    # state_at_seam_plus_ (a left-walk terminal changes n^-, not n^+), and
    # topology_sweep consumed snapshots staled by the preceding
    # cluster_update. Invariants: (a) after any commit, cached snapshots must
    # equal a fresh recompute; (b) after topology_sweep (entered right after
    # mc_step, i.e. with a stale cache), same.
    eng, _ = _make_engine(N=6, M=16, seed=7)
    eng.set_string_sites([1, 3], eng.M)
    for _ in range(100):
        eng.mc_step()

    def cache_matches_recompute():
        cm = np.array(eng.state_at_seam_minus).copy()
        cp = np.array(eng.state_at_seam_plus).copy()
        eng.recompute_seam_snapshots()
        return (np.array_equal(cm, np.array(eng.state_at_seam_minus))
                and np.array_equal(cp, np.array(eng.state_at_seam_plus)))

    n_commits = 0
    for it in range(600):
        eng.mc_step()
        eng.recompute_seam_snapshots()
        prop = eng.build_half_line_proposal(it % 2, bool(it % 4 < 2))
        if prop.valid:
            eng.commit_half_line_proposal(it % 2, prop)
            n_commits += 1
            assert cache_matches_recompute(), f"commit corrupted seam cache (iter {it})"
        eng.topology_sweep(0.5)
        assert cache_matches_recompute(), f"topology_sweep left stale seam cache (iter {it})"
    assert n_commits >= 100, "expected plenty of committed toggles to exercise both walk directions"


# ── Worldline closure: parity(sigma^x ops) == seam bit through mask resets ──

def _test_mask_reset_preserves_worldline_closure():
    # Fixed |0...0> boundaries at both tau ends impose, per string site,
    # parity(# type -1 ops on site) == seam bit. Every kernel preserves it,
    # so a raw mask write that changes a bit strands the engine in an
    # unphysical sector forever (the third 2026-07-12 bug: trajectory resets
    # used the raw setter, poisoning alternating trajectories).
    # set_seam_mask_consistent must repair it.
    eng, _ = _make_engine(N=6, M=16, seed=11)
    sites = [1, 3]
    eng.set_string_sites(sites, eng.M)
    eng.set_seam_mask_consistent(0)

    def closure_ok():
        ot = np.array(eng.op_types)
        os_ = np.array(eng.op_sites)
        for k, s in enumerate(sites):
            n_flip = int(np.sum((ot == -1) & (os_ == s)))
            if (n_flip + ((eng.seam_mask >> k) & 1)) % 2 != 0:
                return False
        return True

    for masks in ((0b11, 0b00), (0b01, 0b10), (0b11, 0b01), (0b00, 0b11)):
        for mask in masks:
            eng.set_seam_mask_consistent(mask)
            assert closure_ok(), f"closure broken right after reset to {mask:#04b}"
            for _ in range(20):
                eng.mc_step()
                eng.topology_sweep(0.5)
            assert closure_ok(), f"closure broken after evolution at reset {mask:#04b}"


# ── Phase B capstone: fixed lambda=1/2 sector residence probability vs ED ───

def _test_sector_residence_probability_matches_ed():
    N, M = 5, 16
    Omega, delta_min, delta_max, Rb, epsilon = 1.0, 0.0, 1.5, 1.2, 0.01
    site = 2  # middle site of a 5-site chain
    eng, pos = _make_engine(N=N, M=M, Omega=Omega, delta_min=delta_min,
                            delta_max=delta_max, Rb=Rb, epsilon=epsilon,
                            neighbor_cutoff=1, delta_groups=32, seed=123)
    eng.set_string_sites([site], eng.M)
    eng.set_seam_mask(0)

    n_therm = 2000
    for _ in range(n_therm):
        eng.mc_step()

    n_samples = 40000
    n_active = 0
    for _ in range(n_samples):
        eng.topology_sweep(0.5)
        eng.mc_step()
        n_active += eng.seam_mask & 1

    p_active = n_active / n_samples
    p_inactive = 1.0 - p_active
    assert 0.0 < p_active < 1.0, "need both sectors visited to estimate a ratio"
    empirical_ratio = p_active / p_inactive

    ed = qaqmc_exact_string_zratio(
        N=N, Omega=Omega, delta_min=delta_min, delta_max=delta_max, Rb=Rb, M=M,
        B_sets=[[site]], m_star=M, pos=pos, epsilon=epsilon, neighbor_cutoff=1,
    )
    expected_ratio = ed["O_B"][frozenset([site])]

    rel_err = abs(empirical_ratio - expected_ratio) / abs(expected_ratio)
    assert rel_err < 0.15, (
        f"empirical P(active)/P(inactive)={empirical_ratio:.4f} vs "
        f"ED O_site={expected_ratio:.4f}, rel_err={rel_err:.3f}"
    )
    return empirical_ratio, expected_ratio, rel_err


def main():
    _test_half_line_involution()
    print("Test 2 (half-line involution) passed")
    _test_incremental_ratio_matches_full_recompute()
    print("Test 4 (incremental vs from-scratch weight ratio) passed")
    _test_seam_cache_stays_consistent()
    print("Test 5 (seam cache == recompute through commits/sweeps) passed")
    _test_mask_reset_preserves_worldline_closure()
    print("Test 6 (worldline closure through consistent mask resets) passed")
    empirical_ratio, expected_ratio, rel_err = _test_sector_residence_probability_matches_ed()
    print(f"Phase B capstone (sector residence vs ED) passed: "
          f"empirical={empirical_ratio:.4f} ED={expected_ratio:.4f} rel_err={rel_err:.3f}")


if __name__ == "__main__":
    main()
