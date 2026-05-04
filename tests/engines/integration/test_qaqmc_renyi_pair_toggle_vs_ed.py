import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analysis.ed_core import build_qaqmc_midpoint_state
from src.rydberg.lattices import generate_1d_chain
from src.tee.qaqmc_renyi_ratio import RegionRatioRunner, combine_region_runs


def _reduced_density_matrix(psi, region_mask):
    N = int(np.log2(psi.size))
    region = [i for i in range(N) if region_mask[i]]
    complement = [i for i in range(N) if not region_mask[i]]
    mat = np.zeros((1 << len(region), 1 << len(complement)), dtype=np.complex128)
    for state in range(1 << N):
        row = 0
        col = 0
        for bit, site in enumerate(region):
            row |= (((state >> site) & 1) << bit)
        for bit, site in enumerate(complement):
            col |= (((state >> site) & 1) << bit)
        mat[row, col] = psi[state]
    return mat @ mat.conj().T


def _purity(psi, region_mask):
    rho = _reduced_density_matrix(psi, region_mask)
    return float(np.real(np.trace(rho @ rho)))


def _test_pair_toggle_matches_ed_within_sampling_error():
    N = 4
    Omega = 1.0
    Rb = 1.2
    delta_min = 0.0
    delta_max = 1.5
    M = 100
    pos = generate_1d_chain(N)
    region_mask = np.array([1, 1, 0, 0], dtype=np.uint8)

    psi = build_qaqmc_midpoint_state(
        N=N,
        Omega=Omega,
        delta_min=delta_min,
        delta_max=delta_max,
        Rb=Rb,
        M=M,
        pos=pos,
        neighbor_cutoff=1,
    )
    s2_ed = -np.log(_purity(psi, region_mask))

    runner = RegionRatioRunner(
        N=N,
        M=M,
        Omega=Omega,
        Rb=Rb,
        delta_min=delta_min,
        delta_max=delta_max,
        pos=pos,
        seed=1234,
        neighbor_cutoff=1,
    )
    masks = [
        np.array([0, 0, 0, 0], dtype=np.uint8),
        np.array([1, 0, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0, 0], dtype=np.uint8),
    ]
    purities = [_purity(psi, mask) for mask in masks]
    exact_ratios = np.array(
        [purities[1] / purities[0], purities[2] / purities[1]],
        dtype=np.float64,
    )

    runner = RegionRatioRunner(
        N=N,
        M=M,
        Omega=Omega,
        Rb=Rb,
        delta_min=delta_min,
        delta_max=delta_max,
        pos=pos,
        seed=1234,
        neighbor_cutoff=1,
    )
    result = runner.run_region(
        "A",
        np.array([1, 1, 0, 0], dtype=np.uint8),
        site_order=[0, 1],
        n_therm=2000,
        n_measure=100000,
        measure_stride=2,
    )

    mc_ratios = np.array([item.ratio for item in result.ratio_results], dtype=np.float64)
    mc_ratio_errs = np.array([item.ratio_err for item in result.ratio_results], dtype=np.float64)
    s2_mc = result.summary.S_2
    assert abs(s2_mc - s2_ed) < 0.03
    assert abs(s2_mc - s2_ed) < 3.5 * result.summary.S_2_err
    assert np.all(np.abs(mc_ratios - exact_ratios) < 0.025)
    assert np.all(np.abs(mc_ratios - exact_ratios) < 4.0 * mc_ratio_errs)


def _test_combined_independent_runs_match_ed_more_tightly():
    N = 4
    Omega = 1.0
    Rb = 1.2
    delta_min = 0.0
    delta_max = 1.5
    M = 100
    pos = generate_1d_chain(N)
    region_mask = np.array([1, 1, 0, 0], dtype=np.uint8)

    psi = build_qaqmc_midpoint_state(
        N=N,
        Omega=Omega,
        delta_min=delta_min,
        delta_max=delta_max,
        Rb=Rb,
        M=M,
        pos=pos,
        neighbor_cutoff=1,
    )
    s2_ed = -np.log(_purity(psi, region_mask))

    region_runs = []
    for seed in (101, 202, 303, 404):
        runner = RegionRatioRunner(
            N=N,
            M=M,
            Omega=Omega,
            Rb=Rb,
            delta_min=delta_min,
            delta_max=delta_max,
            pos=pos,
            seed=seed,
            neighbor_cutoff=1,
        )
        region_runs.append(
            runner.run_region(
                "A",
                region_mask,
                site_order=[0, 1],
                n_therm=2000,
                n_measure=40000,
                measure_stride=2,
            )
        )

    combined = combine_region_runs(region_runs)
    assert abs(combined.summary.S_2 - s2_ed) < 0.015


def _test_single_chain_blocked_error_bar_is_consistent_with_ed():
    N = 4
    Omega = 1.0
    Rb = 1.2
    delta_min = 0.0
    delta_max = 1.5
    M = 100
    pos = generate_1d_chain(N)
    region_mask = np.array([1, 1, 0, 0], dtype=np.uint8)

    psi = build_qaqmc_midpoint_state(
        N=N,
        Omega=Omega,
        delta_min=delta_min,
        delta_max=delta_max,
        Rb=Rb,
        M=M,
        pos=pos,
        neighbor_cutoff=1,
    )
    s2_ed = -np.log(_purity(psi, region_mask))

    runner = RegionRatioRunner(
        N=N,
        M=M,
        Omega=Omega,
        Rb=Rb,
        delta_min=delta_min,
        delta_max=delta_max,
        pos=pos,
        seed=101,
        neighbor_cutoff=1,
    )
    result = runner.run_region(
        "A",
        region_mask,
        site_order=[0, 1],
        n_therm=2000,
        n_measure=40000,
        measure_stride=2,
        block_size=500,
    )

    assert abs(result.summary.S_2 - s2_ed) < 2.0 * result.summary.S_2_err


def main():
    _test_pair_toggle_matches_ed_within_sampling_error()
    _test_combined_independent_runs_match_ed_more_tightly()
    _test_single_chain_blocked_error_bar_is_consistent_with_ed()
    print("QAQMC Renyi pair-toggle vs ED integration checks passed")


if __name__ == "__main__":
    main()
