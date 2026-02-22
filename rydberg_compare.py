"""
SSE QMC vs Exact Diagonalization benchmark for Rydberg atom arrays.

Runs a comparison of finite-temperature energies and densities between
ED (exact) and SSE QMC for a 1D chain with open boundary conditions.

Usage:
    python rydberg_compare.py
"""

import time
import numpy as np

from rydberg_ed import build_rydberg_hamiltonian, ed_thermal_energy, ed_thermal_density
from rydberg_sse import SSE_Rydberg


def run_comparison(
    N: int = 4,
    Omega: float = 1.0,
    Rb: float = 1.2,
    betas: list = None,
    delta_ratios: list = None,
    n_equil: int = 20000,
    n_measure: int = 100000,
    epsilon: float = 0.01,
):
    """Compare SSE QMC against ED for the specified parameters.

    Args:
        N:             Number of sites.
        Omega:         Rabi frequency.
        Rb:            Blockade radius.
        betas:         List of inverse temperatures to sweep.
        delta_ratios:  List of δ/Ω values to sweep.
        n_equil:       Equilibration steps per simulation.
        n_measure:     Measurement steps per simulation.
        epsilon:       Regularisation for bond weights.
    """
    if betas is None:
        betas = [1.0, 2.0, 5.0]
    if delta_ratios is None:
        delta_ratios = [-2.0, -1.0, 0.0, 1.0, 2.0]

    print("=" * 90)
    print(f"  SSE QMC vs ED  |  N={N}, Ω={Omega}, Rb={Rb}")
    print("=" * 90)

    # Trigger Numba JIT compilation on a small dummy run
    print("  [Numba JIT compiling...]", end="", flush=True)
    t0 = time.time()
    _w = SSE_Rydberg(2, 1.0, 0.0, 1.2, 1.0, seed=0)
    _w.mc_step()
    print(f" done ({time.time() - t0:.1f}s)")

    for beta in betas:
        print(f"\n{'─' * 90}")
        print(f"  β = {beta:.1f}  (T/Ω = {1.0 / beta:.2f})")
        print(f"{'─' * 90}")
        print(f"  {'δ/Ω':>6s}  │ {'E_ED/ΩN':>12s}  {'E_SSE/ΩN':>12s}  {'±err':>10s}  "
              f"{'rel%':>8s}  │ {'n_ED':>7s}  {'n_SSE':>7s}  │ {'M':>5s}")
        print(f"  {'─' * 6}──┼─{'─' * 12}──{'─' * 12}──{'─' * 10}──{'─' * 8}──┼─"
              f"{'─' * 7}──{'─' * 7}──┼─{'─' * 5}")

        for dr in delta_ratios:
            delta = dr * Omega

            # Exact diagonalization
            H     = build_rydberg_hamiltonian(N, Omega, delta, Rb)
            e_ed  = ed_thermal_energy(H, beta) / (Omega * N)
            n_ed  = ed_thermal_density(H, N, beta)

            # SSE QMC
            seed  = abs(int(42 + dr * 1000 + beta * 100))
            sse   = SSE_Rydberg(N, Omega, delta, Rb, beta,
                                epsilon=epsilon, seed=seed)
            res   = sse.run(n_equil=n_equil, n_measure=n_measure)

            e_sse = res['energy_mean'] / (Omega * N)
            e_err = res['energy_err']  / (Omega * N)
            n_sse = res['density_mean']
            rel   = abs(e_sse - e_ed) / max(abs(e_ed), 1e-10) * 100

            print(f"  {dr:6.1f}  │ {e_ed:12.6f}  {e_sse:12.6f}  {e_err:10.6f}  "
                  f"{rel:7.2f}%  │ {n_ed:7.4f}  {n_sse:7.4f}  │ {res['M']:5d}")

    print(f"\n{'=' * 90}")
    print("  Done!")


if __name__ == '__main__':
    run_comparison()
