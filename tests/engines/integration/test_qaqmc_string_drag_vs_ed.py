"""Seam-drag curve vs ED on the 1D chain (docs/design/seam_drag_curve.md, M3 gate).

Gate A (sharp, drag-only): the Rao-Blackwellized drag ladder walks the cut
slot-by-slot from the full-mask equilibrium at m_anchor = M and estimates
Z_X(m)/Z_X(M) at every record point in one pass; compare each point against
the exact ratio O_B(m)/O_B(M) from ``qaqmc_exact_string_zratio`` (Z_empty is
m-independent, so the O_B ratio IS the Z_X ratio).

Gate B (end-to-end): compose the ladder curve with the existing
lambda-Jarzynski anchor O_C(M) to get the full curve O_C(m) and compare
against ED at every m.

Rate-independence: a correct kernel is unbiased at ANY sampling rate
(E14/seam-parity lesson: a single-point ED match can hide compensating
biases, but a rate-DEPENDENT systematic cannot) -- so Gate A runs at two
decorrelation rates and both must hit ED.

Trajectory pin: the plain Jarzynski drag (run_drag_trajectories) is exact in
expectation but heavy-tailed (the raw per-config crossing ratio has
~1/epsilon whales), so single-direction realizations swing far off ED in
either direction. The geometric mean of forward and inverted-reverse
(BAR-lite) cancels most of that and must agree with ED, which pins the
trajectory machinery without requiring converged tails.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analysis.ed_core import qaqmc_exact_string_zratio
from src.engines.qaqmc_string_work import QAQMCStringWorkRydberg, cosine_schedule
from src.rydberg.lattices import generate_1d_chain

N, M = 5, 16
M_TOTAL = 2 * M
OMEGA, DELTA_MIN, DELTA_MAX, RB, EPSILON = 1.0, 0.0, 1.5, 1.2, 0.01
SITE = 2
GRID_RIGHT = [20, 24, 28, 30]
GRID_LEFT = [12, 8, 4, 2]


def _make_engine(seed, m_star=None):
    pos = generate_1d_chain(N)
    eng = QAQMCStringWorkRydberg(
        N=N, M=M, Omega=OMEGA, Rb=RB, delta_min=DELTA_MIN, delta_max=DELTA_MAX,
        epsilon=EPSILON, seed=seed, pos=pos, neighbor_cutoff=1, delta_groups=32,
    )
    eng.set_string_sites([SITE], m_star=m_star)  # default anchor m_star = M
    return eng, pos


def _ed_o_b(pos, m_values):
    out = {}
    for m in m_values:
        ed = qaqmc_exact_string_zratio(
            N=N, Omega=OMEGA, delta_min=DELTA_MIN, delta_max=DELTA_MAX, Rb=RB,
            M=M, B_sets=[[SITE]], m_star=int(m), pos=pos, epsilon=EPSILON,
            neighbor_cutoff=1,
        )
        out[int(m)] = ed["O_B"][frozenset([SITE])]
    return out


def _ladder_families(eng, n_samples_per_rung, n_sweeps_between_samples,
                     slots_per_rung=1, bidirectional=False):
    eng.thermalize(1000, direction="reverse")  # full-mask sector at the anchor
    res_r = eng.run_drag_ladder(
        np.array(GRID_RIGHT), n_samples_per_rung=n_samples_per_rung,
        n_sweeps_between_samples=n_sweeps_between_samples,
        n_equil_at_anchor=100, slots_per_rung=slots_per_rung, m_anchor=M,
        bidirectional=bidirectional)
    res_l = eng.run_drag_ladder(
        np.array(GRID_LEFT), n_samples_per_rung=n_samples_per_rung,
        n_sweeps_between_samples=n_sweeps_between_samples,
        n_equil_at_anchor=100, slots_per_rung=slots_per_rung, m_anchor=M,
        bidirectional=bidirectional)
    return res_r, res_l


def _check_family_vs_ed(res, ed, label, tol):
    for j, m in enumerate(res.m_grid):
        exact = ed[int(m)] / ed[M]
        rel_err = abs(res.r[j] - exact) / abs(exact)
        print(f"  [{label}] m={int(m):2d}  r_ladder={res.r[j]:.4f}"
              f"(sem {res.log_r_sem[j]:.4f})  r_ED={exact:.4f}  rel_err={rel_err:.3f}")
        assert rel_err < tol, (label, int(m), res.r[j], exact, rel_err)


def test_drag_ladder_curve_matches_ed():
    eng, pos = _make_engine(seed=20260730)
    ed = _ed_o_b(pos, GRID_RIGHT + GRID_LEFT + [M])
    res_r, res_l = _ladder_families(eng, n_samples_per_rung=1200,
                                    n_sweeps_between_samples=2)
    _check_family_vs_ed(res_r, ed, "right", tol=0.08)
    _check_family_vs_ed(res_l, ed, "left", tol=0.08)


def test_drag_ladder_bidirectional_matches_ed_and_spr_consistent():
    # Symmetric fwd/rev rung estimator (the large-M mode): must hit ED like
    # the one-sided ladder, AND agree with itself across rung block sizes.
    # The spr-consistency clause is the gate that would have caught the
    # M=3e6 one-sided Jensen bias (probes 27140/27141: -540 vs -1210 for a
    # O(-1) truth, per-rung bias ~ spr^2) -- a biased rung estimator gives
    # spr-DEPENDENT curves, an unbiased one cannot.
    results = {}
    for spr, seed in ((1, 41), (4, 42)):
        eng, pos = _make_engine(seed=seed)
        ed = _ed_o_b(pos, GRID_RIGHT + GRID_LEFT + [M])
        res_r, res_l = _ladder_families(eng, n_samples_per_rung=800,
                                        n_sweeps_between_samples=2,
                                        slots_per_rung=spr,
                                        bidirectional=True)
        assert res_r.bidirectional and res_l.bidirectional
        _check_family_vs_ed(res_r, ed, f"right-bidir spr={spr}", tol=0.08)
        _check_family_vs_ed(res_l, ed, f"left-bidir spr={spr}", tol=0.08)
        results[spr] = (res_r, res_l)
    for fam, name in ((0, "right"), (1, "left")):
        a, b = results[1][fam], results[4][fam]
        diff = abs(float(a.log_r[-1] - b.log_r[-1]))
        sem = float(np.sqrt(a.log_r_sem[-1] ** 2 + b.log_r_sem[-1] ** 2))
        print(f"  [spr-consistency {name}] |dlog r|={diff:.4f} sem={sem:.4f}")
        assert diff < max(3.0 * sem, 0.08), (
            f"bidirectional ladder is spr-dependent on the {name} family: "
            f"|dlog r|={diff:.4f} (sem {sem:.4f})")


def test_drag_ladder_rate_independence():
    # Same gate at two decorrelation rates and at a multi-slot rung size; a
    # biased kernel, a dishonest equilibrium (insufficient decorrelation
    # between rung samples), or a broken block-RB factorization shows up as
    # a knob-dependent shift, so ALL variants must hit ED.
    for sweeps, spr, seed in ((1, 1, 31), (4, 1, 32), (2, 4, 33)):
        eng, pos = _make_engine(seed=seed)
        ed = _ed_o_b(pos, [GRID_RIGHT[-1], GRID_LEFT[-1], M])
        res_r, res_l = _ladder_families(eng, n_samples_per_rung=800,
                                        n_sweeps_between_samples=sweeps,
                                        slots_per_rung=spr)
        for res, m_far in ((res_r, GRID_RIGHT[-1]), (res_l, GRID_LEFT[-1])):
            j = list(res.m_grid).index(m_far)
            exact = ed[m_far] / ed[M]
            rel_err = abs(res.r[j] - exact) / abs(exact)
            print(f"  [rate sweeps={sweeps} spr={spr}] m={m_far}  r={res.r[j]:.4f}  "
                  f"r_ED={exact:.4f}  rel_err={rel_err:.3f}")
            assert rel_err < 0.10, (sweeps, spr, m_far, res.r[j], exact, rel_err)


def test_drag_jarzynski_geomean_matches_ed():
    # Trajectory (non-RB) drag: each direction's REALIZED value swings either
    # way depending on whether a whale trajectory landed in the sample (the
    # estimator is unbiased but heavy-tailed), so no per-side or ordering
    # assertion is sound. The geometric mean of forward and inverted-reverse
    # (BAR-lite) cancels most of the one-sided error and must sit near ED --
    # this pins the trajectory machinery without requiring converged tails.
    _, pos = _make_engine(seed=0)
    ed = _ed_o_b(pos, [16, 20])
    exact = ed[20] / ed[16]

    eng_f, _ = _make_engine(seed=41)
    eng_f.thermalize(1000, direction="reverse")
    fwd = eng_f.run_drag_trajectories(np.array([20]), n_trajectories=1200,
                                      decorrelation_steps=50,
                                      n_qaqmc_sweeps_per_shift=2, m_anchor=M)
    eng_r, _ = _make_engine(seed=42, m_star=20)
    eng_r.thermalize(1000, direction="reverse")
    rev = eng_r.run_drag_trajectories(np.array([16]), n_trajectories=1200,
                                      decorrelation_steps=50,
                                      n_qaqmc_sweeps_per_shift=2, m_anchor=20)
    bar_lite = float(np.sqrt(fwd.r[0] / rev.r[0]))  # geo-mean of fwd and 1/rev
    rel_err = abs(bar_lite - exact) / exact
    print(f"  fwd={fwd.r[0]:.4f}  1/rev={1.0 / rev.r[0]:.4f}  "
          f"geo-mean={bar_lite:.4f}  ED={exact:.4f}  rel_err={rel_err:.3f}")
    assert rel_err < 0.15, (bar_lite, exact, rel_err)


def test_drag_mirrored_curve_matches_ed():
    # Mirror-averaged curve (M4-1): geo-mean of the two branches at the same
    # delta must match the ED geo-mean sqrt(O(m) O(2M-m))/O(M). This gates
    # the run_drag_curve_mirrored plumbing; the convergence systematics
    # (odd-part removal, clean ~1/M^2 tail) are established exactly in the
    # ED M-scaling study (docs/design/seam_drag_curve.md SS6).
    eng, pos = _make_engine(seed=555)
    grid_fwd = np.array(GRID_LEFT)  # 12, 8, 4, 2
    mirror = [2 * M - m for m in grid_fwd]
    ed = _ed_o_b(pos, list(grid_fwd) + mirror + [M])
    eng.thermalize(1000, direction="reverse")
    res = eng.run_drag_curve_mirrored(grid_fwd, n_samples_per_rung=1200,
                                      n_sweeps_between_samples=2)
    for j, m in enumerate(res.m_forward):
        exact = np.sqrt(ed[int(m)] * ed[int(res.m_mirror[j])]) / ed[M]
        rel_err = abs(res.r_mirror[j] - exact) / exact
        print(f"  [mirror] m={int(m):2d}/{int(res.m_mirror[j]):2d}  "
              f"r_geo={res.r_mirror[j]:.4f}(sem {res.log_r_sem[j]:.4f})  "
              f"r_ED={exact:.4f}  rel_err={rel_err:.3f}")
        assert rel_err < 0.08, (int(m), res.r_mirror[j], exact, rel_err)


def test_drag_two_site_ladder_matches_ed():
    # Multi-site string (M4-3): a crossed bond op can now touch TWO active
    # seam sites at once (both endpoints flip between frames) -- exercised
    # here on C = {1, 3} with bonds (1,2)/(2,3)/(3,4) each touching one site
    # and the state-XOR bookkeeping touching both.
    sites = [1, 3]
    pos = generate_1d_chain(N)
    ed = {}
    for m in GRID_RIGHT + GRID_LEFT + [M]:
        e = qaqmc_exact_string_zratio(
            N=N, Omega=OMEGA, delta_min=DELTA_MIN, delta_max=DELTA_MAX, Rb=RB,
            M=M, B_sets=[sites], m_star=int(m), pos=pos, epsilon=EPSILON,
            neighbor_cutoff=1)
        ed[int(m)] = e["O_B"][frozenset(sites)]

    eng = QAQMCStringWorkRydberg(
        N=N, M=M, Omega=OMEGA, Rb=RB, delta_min=DELTA_MIN, delta_max=DELTA_MAX,
        epsilon=EPSILON, seed=606, pos=pos, neighbor_cutoff=1, delta_groups=32)
    eng.set_string_sites(sites)
    res_r, res_l = _ladder_families(eng, n_samples_per_rung=1200,
                                    n_sweeps_between_samples=2)
    _check_family_vs_ed(res_r, ed, "2site right", tol=0.10)
    _check_family_vs_ed(res_l, ed, "2site left", tol=0.10)


def test_growth_anchor_matches_ed():
    # Sector-growth residence ladder (anchor v2): every partial product
    # Z_{bits 0..k}/Z_empty must match ED's O_B of the partial string --
    # a per-stage gate, not just the endpoint (compensating-bias discipline).
    sites = [1, 2, 3]
    pos = generate_1d_chain(N)
    ed = qaqmc_exact_string_zratio(
        N=N, Omega=OMEGA, delta_min=DELTA_MIN, delta_max=DELTA_MAX, Rb=RB,
        M=M, B_sets=[[1], [1, 2], [1, 2, 3]], m_star=M, pos=pos,
        epsilon=EPSILON, neighbor_cutoff=1)
    partials = [frozenset(sites[:k + 1]) for k in range(len(sites))]

    eng = QAQMCStringWorkRydberg(
        N=N, M=M, Omega=OMEGA, Rb=RB, delta_min=DELTA_MIN, delta_max=DELTA_MAX,
        epsilon=EPSILON, seed=909, pos=pos, neighbor_cutoff=1, delta_groups=32)
    eng.set_string_sites(sites)
    eng.thermalize(1000)
    res = eng.run_growth_residence_ladder(n_samples_per_stage=6000,
                                          n_sweeps_between_samples=1,
                                          n_equil_per_stage=200,
                                          n_tune_samples=300)
    cum = np.cumsum(res.log_r)
    for k, fs in enumerate(partials):
        got = float(np.exp(cum[k]))
        exact = ed["O_B"][fs]
        rel_err = abs(got - exact) / exact
        print(f"  [growth] stage {k} (C={sorted(fs)}): O={got:.4f} "
              f"(lam={res.lambdas[k]:.3f}, p_on={res.p_on[k]:.3f}, "
              f"flips={res.n_flips[k]})  O_ED={exact:.4f}  rel_err={rel_err:.3f}")
        assert res.n_flips[k] > 50, f"stage {k} under-mixed"
        assert rel_err < 0.10, (k, got, exact, rel_err)
    # ladder must hand the engine back in the full-mask sector (drag segue)
    assert eng._eng.seam_mask == 0b111


def test_drag_composed_curve_matches_ed():
    # End-to-end: O_C(m) = O_C(M) [lambda-Jarzynski anchor] x r_ladder(m).
    eng, pos = _make_engine(seed=777)
    ed = _ed_o_b(pos, GRID_RIGHT + GRID_LEFT + [M])

    eng.set_lambda_schedule(cosine_schedule(150))
    eng.thermalize(2000)  # empty sector for the forward lambda protocol
    anchor = eng.run_trajectories(n_trajectories=3000, decorrelation_steps=200,
                                  n_topology_sweeps_per_lambda=4,
                                  n_qaqmc_sweeps_per_lambda=4)
    rel_anchor = abs(anchor.o_c - ed[M]) / abs(ed[M])
    print(f"  anchor O_C(M): jarzynski={anchor.o_c:.4f}  ED={ed[M]:.4f}  "
          f"rel_err={rel_anchor:.3f}  N_eff={anchor.n_eff:.0f}/{anchor.n_trajectories}")
    assert rel_anchor < 0.10, (anchor.o_c, ed[M], rel_anchor)

    res_r, res_l = _ladder_families(eng, n_samples_per_rung=1200,
                                    n_sweeps_between_samples=2)
    for res in (res_r, res_l):
        for j, m in enumerate(res.m_grid):
            o_c_m = anchor.o_c * res.r[j]
            rel_err = abs(o_c_m - ed[int(m)]) / abs(ed[int(m)])
            print(f"  [composed] m={int(m):2d}  O_C={o_c_m:.4f}  "
                  f"O_C(ED)={ed[int(m)]:.4f}  rel_err={rel_err:.3f}")
            assert rel_err < 0.12, (int(m), o_c_m, ed[int(m)], rel_err)


def main():
    print("Gate A: RB drag-ladder Z_X(m)/Z_X(M) curve vs ED:")
    test_drag_ladder_curve_matches_ed()
    print("Rate-independence (sweeps=1 vs 4):")
    test_drag_ladder_rate_independence()
    print("Jarzynski trajectory geo-mean (fwd x 1/rev, BAR-lite):")
    test_drag_jarzynski_geomean_matches_ed()
    print("Mirror-averaged curve vs ED:")
    test_drag_mirrored_curve_matches_ed()
    print("Two-site string ladder vs ED:")
    test_drag_two_site_ladder_matches_ed()
    print("Gate B: composed O_C(m) curve vs ED:")
    test_drag_composed_curve_matches_ed()
    print("Seam-drag vs ED gates passed")


if __name__ == "__main__":
    main()
