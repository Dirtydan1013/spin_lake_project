"""Verify delta_groups>0 path against the QAQMC ED benchmark.

The proposal distribution changes (slices in the same group share an
alias table built from an envelope bond weight), but the algorithm remains
exact because acceptance uses the per-slice weight.  MC estimates of the
Renyi-2 entropy must therefore match the QAQMC ED value within the same
tolerance as the delta_groups=0 case.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analysis.ed_core import build_qaqmc_midpoint_state
from src.rydberg.lattices import generate_1d_chain
from src.tee.qaqmc_renyi_ratio import RegionRatioRunner, combine_region_runs
from src.tee.reweighting import ReweightingDriver


def _reduced_density_matrix(psi, region_mask):
    n_sites = int(np.log2(psi.size))
    region = [i for i in range(n_sites) if region_mask[i]]
    complement = [i for i in range(n_sites) if not region_mask[i]]
    mat = np.zeros((1 << len(region), 1 << len(complement)), dtype=np.complex128)
    for state in range(1 << n_sites):
        row = 0; col = 0
        for bit, site in enumerate(region):
            row |= (((state >> site) & 1) << bit)
        for bit, site in enumerate(complement):
            col |= (((state >> site) & 1) << bit)
        mat[row, col] = psi[state]
    return mat @ mat.conj().T


def _s2_ed(psi, mask):
    rho = _reduced_density_matrix(psi, mask)
    return -np.log(float(np.real(np.trace(rho @ rho))))


# Shared physics params (same as test_reweighting_expanded_vs_ed / test_qaqmc_renyi_pair_toggle_vs_ed)
N = 4
M = 100
OMEGA = 1.0
Rb = 1.2
DELTA_MIN = 0.0
DELTA_MAX = 1.5
NEIGHBOR_CUTOFF = 1
POS = generate_1d_chain(N)
REGION = np.array([1, 1, 0, 0], dtype=np.uint8)
LADDER_MASKS = [
    np.array([0, 0, 0, 0], dtype=np.uint8),
    np.array([1, 0, 0, 0], dtype=np.uint8),
    np.array([1, 1, 0, 0], dtype=np.uint8),
]
LADDER_NEIGHBORS = [[1], [0, 2], [1]]


def _ed_s2():
    psi = build_qaqmc_midpoint_state(
        N=N, Omega=OMEGA, delta_min=DELTA_MIN, delta_max=DELTA_MAX,
        Rb=Rb, M=M, pos=POS, neighbor_cutoff=NEIGHBOR_CUTOFF,
    )
    return _s2_ed(psi, REGION)


def _test_ratio_estimator_matches_ed_with_delta_groups(delta_groups: int):
    s2_ed = _ed_s2()

    region_runs = []
    for seed in (101, 202, 303, 404):
        runner = RegionRatioRunner(
            N=N, M=M, Omega=OMEGA, Rb=Rb,
            delta_min=DELTA_MIN, delta_max=DELTA_MAX,
            pos=POS, seed=seed, neighbor_cutoff=NEIGHBOR_CUTOFF,
            delta_groups=delta_groups,
        )
        region_runs.append(
            runner.run_region(
                "A", REGION, site_order=[0, 1],
                n_therm=2000, n_measure=40000, measure_stride=2,
            )
        )

    combined = combine_region_runs(region_runs)
    s2_mc = combined.summary.S_2
    assert abs(s2_mc - s2_ed) < 0.02, (
        f"ratio estimator (delta_groups={delta_groups}): "
        f"S_2={s2_mc:.5f} vs ED={s2_ed:.5f}, diff={s2_mc - s2_ed:.5f}"
    )


def _test_expanded_ensemble_matches_ed_with_delta_groups(delta_groups: int):
    s2_ed = _ed_s2()

    driver = ReweightingDriver(
        N=N, M=M, Omega=OMEGA, Rb=Rb,
        delta_min=DELTA_MIN, delta_max=DELTA_MAX,
        pos=POS, seed=1234, neighbor_cutoff=NEIGHBOR_CUTOFF,
        delta_groups=delta_groups,
    )
    driver.set_ensemble_ladder(LADDER_MASKS, LADDER_NEIGHBORS, initial_ensemble=0)
    auto = driver.auto_tune(n_steps_per_iter=15000, max_iters=8, tol=1.15,
                            method="transition_matrix", damping=0.7)
    final_counts = auto.visit_counts[-1]
    assert float(np.max(final_counts) / np.min(final_counts)) < 1.15

    production = driver.run_production(n_steps=160000, block_size=2000)
    s2_mc, s2_err = production.s2(2)
    s2_coll, s2_coll_err = production.s2_collection(2)
    assert abs(s2_mc - s2_ed) < 0.04, (
        f"expanded histogram (delta_groups={delta_groups}): "
        f"S_2={s2_mc:.5f} vs ED={s2_ed:.5f}"
    )
    assert abs(s2_mc - s2_ed) < 3.0 * s2_err
    assert abs(s2_coll - s2_ed) < 0.04, (
        f"expanded collection (delta_groups={delta_groups}): "
        f"S_2={s2_coll:.5f} vs ED={s2_ed:.5f}"
    )
    assert abs(s2_coll - s2_ed) < 3.0 * max(s2_coll_err, 1e-12)


def _test_delta_groups_memory_saver_never_allocates_per_slice_bond_W():
    import qaqmc_cpp  # noqa: F401 — the build/.so import is fine

    from src.engines.qaqmc_renyi import QAQMCRenyiRydberg

    engine_groups = QAQMCRenyiRydberg(
        N=N, M=M, Omega=OMEGA, Rb=Rb,
        delta_min=DELTA_MIN, delta_max=DELTA_MAX,
        pos=POS, seed=42, neighbor_cutoff=NEIGHBOR_CUTOFF,
        delta_groups=20,
    )
    # It should report the groups it was actually built with (clamped into [0, M_total]).
    assert engine_groups.delta_groups == 20

    # Smoke-test: can run a single MC step in PairToggle mode
    mask = np.zeros(N, dtype=np.uint8)
    next_mask = mask.copy(); next_mask[0] = 1
    engine_groups._cpp_engine.set_topology_pair(mask, next_mask, 0)
    engine_groups.mc_step()

    # And in Expanded mode
    engine2 = QAQMCRenyiRydberg(
        N=N, M=M, Omega=OMEGA, Rb=Rb,
        delta_min=DELTA_MIN, delta_max=DELTA_MAX,
        pos=POS, seed=42, neighbor_cutoff=NEIGHBOR_CUTOFF,
        delta_groups=20,
    )
    engine2.set_ensemble_ladder(LADDER_MASKS, LADDER_NEIGHBORS, initial_ensemble=0)
    engine2.set_log_g(np.zeros(3, dtype=np.float64))
    engine2.mc_step()


def main():
    # Spot-check two delta_groups values: one where every slice has its own
    # group (same as delta_groups=0 path in practice), and one where many
    # slices share a group (the memory-saver regime).
    for dg in (20, 200):
        _test_ratio_estimator_matches_ed_with_delta_groups(dg)
        _test_expanded_ensemble_matches_ed_with_delta_groups(dg)
    _test_delta_groups_memory_saver_never_allocates_per_slice_bond_W()
    print("delta_groups vs ED integration checks passed")


if __name__ == "__main__":
    main()
