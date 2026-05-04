"""
Probe: compare MC ratio estimator (QAQMCRenyi) on 1D chain to ED of the
non-adiabatic state |psi_M> produced by a symmetric QAQMC schedule
delta_min -> delta_max -> delta_min at the symmetric point t=M.

Prints both Tr[rho^2] (full quantum purity) and diagonal-basis purity
so we can see which one MC matches.
"""

import os
import sys
import time

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analysis.ed_core import build_qaqmc_midpoint_state
from src.rydberg.lattices import generate_1d_chain
from src.tee.qaqmc_renyi_ratio import RegionRatioRunner


def build_nonadiabatic_state(N, Omega, Rb, pos, delta_min, delta_max, M, neighbor_cutoff):
    return build_qaqmc_midpoint_state(
        N=N,
        Omega=Omega,
        delta_min=delta_min,
        delta_max=delta_max,
        Rb=Rb,
        M=M,
        pos=pos,
        neighbor_cutoff=neighbor_cutoff,
    )


def reduced_density_matrix(psi, region_mask):
    N = int(np.log2(psi.size))
    A = [i for i in range(N) if region_mask[i]]
    B = [i for i in range(N) if not region_mask[i]]
    nA = len(A); nB = len(B)
    T = np.zeros((1 << nA, 1 << nB), dtype=np.complex128)
    for s in range(1 << N):
        a_idx = 0
        for k, i in enumerate(A):
            a_idx |= (((s >> i) & 1) << k)
        b_idx = 0
        for k, i in enumerate(B):
            b_idx |= (((s >> i) & 1) << k)
        T[a_idx, b_idx] = psi[s]
    return T @ T.conj().T


def renyi2_full(rho_A):
    return float(np.real(np.trace(rho_A @ rho_A)))


def renyi2_diag(rho_A):
    diag = np.real(np.diag(rho_A))
    return float(np.sum(diag * diag))


def main():
    N = 4
    Omega = 1.0
    Rb = 1.2
    delta_min = 0.0
    delta_max = 1.5
    M = 100
    pos = generate_1d_chain(N)
    region_mask = np.array([1, 1, 0, 0], dtype=np.uint8)

    neighbor_cutoff = 1
    psi = build_nonadiabatic_state(N, Omega, Rb, pos, delta_min, delta_max, M, neighbor_cutoff)
    print(f"Norm of non-adiabatic state: {np.linalg.norm(psi):.6f}")

    # Full region {0,1}
    rho_full = reduced_density_matrix(psi, region_mask)
    purity_full = renyi2_full(rho_full)
    diag_purity_full = renyi2_diag(rho_full)
    S2_full = -np.log(purity_full)
    S2_diag_full = -np.log(diag_purity_full)
    print(f"\nRegion {{0,1}} on non-adiabatic |psi_M>:")
    print(f"  Tr[rho^2] = {purity_full:.4f}    S_2 = {S2_full:.4f}")
    print(f"  diag purity = {diag_purity_full:.4f}  S_2_diag = {S2_diag_full:.4f}")

    # Single-site {0}
    mask0 = np.zeros(N, dtype=np.uint8); mask0[0] = 1
    rho0 = reduced_density_matrix(psi, mask0)
    purity0 = renyi2_full(rho0)
    diag_purity0 = renyi2_diag(rho0)
    print(f"\nSingle-site {{0}}:")
    print(f"  Tr[rho^2] = {purity0:.4f}   diag purity = {diag_purity0:.4f}")

    # MC
    runner = RegionRatioRunner(
        N=N, M=M, Omega=Omega, Rb=Rb,
        delta_min=delta_min, delta_max=delta_max, pos=pos, seed=1234, neighbor_cutoff=neighbor_cutoff,
    )
    site_order = [0, 1]
    t0 = time.time()
    region_result = runner.run_region(
        "A", region_mask, site_order=site_order,
        n_therm=500, n_measure=20000,
    )
    dt = time.time() - t0
    mc_s2 = region_result.summary.S_2
    mc_err = region_result.summary.S_2_err
    print(f"\nMC ratio estimator: S_2 = {mc_s2:.4f} +/- {mc_err:.4f}  ({dt:.1f}s)")
    print(f"  step-0 ratio = {region_result.summary.ratios[0]:.4f}  err={region_result.summary.ratio_errs[0]:.4f}")
    print(f"  step-1 ratio = {region_result.summary.ratios[1]:.4f}  err={region_result.summary.ratio_errs[1]:.4f}")

    # Expected step-0 under the two hypotheses:
    print(f"\nExpected step-0 ratio:")
    print(f"  if Tr[rho^2]:   {purity0:.4f}")
    print(f"  if diag purity: {diag_purity0:.4f}")

    # Expected step-1 given ratio chain walks {0}->{0,1}
    # under Tr[rho^2]:  step_1 = Tr[rho_{01}^2] / Tr[rho_0^2]
    # under diag      : step_1 = diag_purity_{01} / diag_purity_0
    print(f"Expected step-1 ratio:")
    print(f"  if Tr[rho^2]:   {purity_full/purity0:.4f}")
    print(f"  if diag purity: {diag_purity_full/diag_purity0:.4f}")

    print(f"\nExpected S_2 total:")
    print(f"  if Tr[rho^2]:   {S2_full:.4f}   |MC - S2_full|  = {abs(mc_s2 - S2_full):.4f}")
    print(f"  if diag purity: {S2_diag_full:.4f}   |MC - S2_diag|  = {abs(mc_s2 - S2_diag_full):.4f}")


if __name__ == "__main__":
    main()
