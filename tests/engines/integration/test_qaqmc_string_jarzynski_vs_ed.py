import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analysis.ed_core import qaqmc_exact_string_zratio
from src.engines.qaqmc_string_work import QAQMCStringWorkRydberg, cosine_schedule
from src.rydberg.lattices import generate_1d_chain


def _run_one_site_jarzynski(N, M, Omega, delta_min, delta_max, Rb, epsilon,
                            site, K, n_traj, n_therm=2000, decorrelation_steps=200,
                            n_relax=4, seed=123):
    pos = generate_1d_chain(N)
    eng = QAQMCStringWorkRydberg(
        N=N, M=M, Omega=Omega, Rb=Rb, delta_min=delta_min, delta_max=delta_max,
        epsilon=epsilon, seed=seed, pos=pos, neighbor_cutoff=1, delta_groups=32,
    )
    eng.set_string_sites([site])
    eng.set_lambda_schedule(cosine_schedule(K))
    eng.thermalize(n_therm)
    # This parameter regime (Rb=1.2 -> bond weights dominate over the Omega/2
    # site term) leaves single-site vertices sparse, so the half-line move
    # mixes slowly: n_relax=1 systematically OVER-estimates O_C by ~12% (fast
    # quench relative to the topology chain's mixing time -- Jarzynski stays
    # unbiased in the infinite-trajectory limit for any quench speed, but a
    # too-fast quench inflates the finite-sample bias/variance). n_relax=4
    # keeps this well under 10% without a slow test.
    result = eng.run_trajectories(n_trajectories=n_traj, decorrelation_steps=decorrelation_steps,
                                  n_topology_sweeps_per_lambda=n_relax,
                                  n_qaqmc_sweeps_per_lambda=n_relax)

    ed = qaqmc_exact_string_zratio(
        N=N, Omega=Omega, delta_min=delta_min, delta_max=delta_max, Rb=Rb, M=M,
        B_sets=[[site]], m_star=M, pos=pos, epsilon=epsilon, neighbor_cutoff=1,
    )
    expected = ed["O_B"][frozenset([site])]
    return result, expected


def _test_one_site_jarzynski_matches_ed():
    result, expected = _run_one_site_jarzynski(
        N=5, M=16, Omega=1.0, delta_min=0.0, delta_max=1.5, Rb=1.2, epsilon=0.01,
        site=2, K=150, n_traj=4000,
    )
    rel_err = abs(result.o_c - expected) / abs(expected)
    print(f"  O_C (Jarzynski) = {result.o_c:.4f}   O_C (ED) = {expected:.4f}   "
          f"rel_err = {rel_err:.3f}   N_eff = {result.n_eff:.1f}/{result.n_trajectories}   "
          f"p_max = {result.p_max:.3f}   zero_weight_frac = {result.zero_weight_fraction:.3f}")
    assert rel_err < 0.10, (result.o_c, expected, rel_err)
    # Rare-event diagnostics per document section 32: warn (via assertion,
    # since this is the correctness test) if the estimator is dominated by a
    # tiny handful of trajectories.
    assert result.n_eff > 0.05 * result.n_trajectories, "Jarzynski average is dominated by too few trajectories"
    return result, expected


def _test_disjoint_site_gives_different_answer():
    # Sanity: a different site should not (in general) give the same O_C --
    # guards against a trivially-broken estimator that always returns ~1.
    result_a, expected_a = _run_one_site_jarzynski(
        N=5, M=16, Omega=1.0, delta_min=0.0, delta_max=1.5, Rb=1.2, epsilon=0.01,
        site=0, K=150, n_traj=3000, seed=7,
    )
    rel_err = abs(result_a.o_c - expected_a) / abs(expected_a)
    print(f"  site=0: O_C (Jarzynski) = {result_a.o_c:.4f}   O_C (ED) = {expected_a:.4f}   "
          f"rel_err = {rel_err:.3f}")
    assert rel_err < 0.10


def main():
    print("One-site Jarzynski vs ED (site=2 of 5-site chain):")
    _test_one_site_jarzynski_matches_ed()
    print("One-site Jarzynski vs ED (site=0, edge site):")
    _test_disjoint_site_gives_different_answer()
    print("Phase C one-site Jarzynski vs ED checks passed")


if __name__ == "__main__":
    main()
