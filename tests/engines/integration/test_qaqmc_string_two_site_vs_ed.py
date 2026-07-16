import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analysis.ed_core import qaqmc_exact_string_zratio
from src.engines.qaqmc_string_work import QAQMCStringWorkRydberg, cosine_schedule
from src.rydberg.lattices import generate_1d_chain


N, M = 6, 20
OMEGA, DELTA_MIN, DELTA_MAX, RB, EPSILON = 1.0, 0.0, 1.5, 1.2, 0.01
SITES = [1, 3]  # a genuine two-site string, not just nearest neighbors


def _run(K, n_traj, n_relax=4, decorrelation_steps=200, seed=321):
    pos = generate_1d_chain(N)
    eng = QAQMCStringWorkRydberg(
        N=N, M=M, Omega=OMEGA, Rb=RB, delta_min=DELTA_MIN, delta_max=DELTA_MAX,
        epsilon=EPSILON, seed=seed, pos=pos, neighbor_cutoff=1, delta_groups=32,
    )
    eng.set_string_sites(SITES)
    eng.set_lambda_schedule(cosine_schedule(K))
    eng.thermalize(2000)
    return eng.run_trajectories(n_trajectories=n_traj, decorrelation_steps=decorrelation_steps,
                                n_topology_sweeps_per_lambda=n_relax,
                                n_qaqmc_sweeps_per_lambda=n_relax)


def _ed_o_c():
    pos = generate_1d_chain(N)
    ed = qaqmc_exact_string_zratio(
        N=N, Omega=OMEGA, delta_min=DELTA_MIN, delta_max=DELTA_MAX, Rb=RB, M=M,
        B_sets=[SITES], m_star=M, pos=pos, epsilon=EPSILON, neighbor_cutoff=1,
    )
    return ed["O_B"][frozenset(SITES)]


def _test_two_site_string_matches_ed_at_slow_quench():
    expected = _ed_o_c()
    result = _run(K=400, n_traj=4000)
    rel_err = abs(result.o_c - expected) / abs(expected)
    print(f"  K=400: O_C(Jarzynski)={result.o_c:.4f}  O_C(ED)={expected:.4f}  "
          f"rel_err={rel_err:.3f}  N_eff={result.n_eff:.0f}/{result.n_trajectories}  "
          f"zero_weight_frac={result.zero_weight_fraction:.3f}")
    assert rel_err < 0.08, (result.o_c, expected, rel_err)
    assert result.n_eff > 0.05 * result.n_trajectories


def _test_estimator_unbiased_at_every_quench_speed():
    # Jarzynski/AIS is exact in expectation at ANY schedule length K provided
    # each lambda-kernel has the correct invariant measure -- quench speed
    # trades variance (n_eff), never bias. So the sharp check is that every K
    # agrees with ED within its statistical resolution, while n_eff grows
    # with K. (A systematic K-trend in O_C is the fingerprint of a biased
    # topology kernel: the pre-2026-07-12 seam-cache/worldline-closure bugs
    # showed up exactly that way, as a fake "finite-K convergence" that a
    # monotonicity assert here then rewarded.)
    expected = _ed_o_c()
    n_effs = {}
    for K in (50, 150, 400):
        result = _run(K=K, n_traj=4000)
        rel_err = abs(result.o_c - expected) / abs(expected)
        n_effs[K] = result.n_eff
        print(f"  K={K}: O_C={result.o_c:.4f}  ED={expected:.4f}  rel_err={rel_err:.3f}  "
              f"n_eff={result.n_eff:.0f}  zero_weight_frac={result.zero_weight_fraction:.3f}")
        # ~4-sigma gate at the worst n_eff seen at K=50 (sigma ~ O_C/sqrt(n_eff))
        assert rel_err < 0.05, (K, result.o_c, expected, rel_err)
    assert n_effs[400] > n_effs[50], n_effs


def main():
    print("Two-site string O_C vs ED at a slow quench (K=400):")
    _test_two_site_string_matches_ed_at_slow_quench()
    print("Two-site string: quench-speed scan (K=50,150,400) vs ED:")
    _test_estimator_unbiased_at_every_quench_speed()
    print("Phase D two-site string vs ED checks passed")


if __name__ == "__main__":
    main()
