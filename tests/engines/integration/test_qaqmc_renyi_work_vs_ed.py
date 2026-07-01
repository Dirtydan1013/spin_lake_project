"""
Integration test: QAQMCRenyiWorkEngine ΔS_2 vs exact-diagonalization S_2(A_end) - S_2(A_start).

Verifies:
  1. ∅ → A case: delta_s2 matches S_2(A) from ED midpoint state.
  2. Nested pair (A → AB) case: delta_s2 matches S_2(AB) − S_2(A).
  3. Sum across a chain of nested rungs matches end-to-end S_2 (KP workflow integration).

Uses a small 1D Rydberg chain (N=4) where ED is feasible and a coarse λ schedule
(K=4) where the Jarzynski estimator has manageable variance. Tolerances account
for finite-trajectory Jarzynski noise + "multiply by 1" prescription bias.
"""
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO_ROOT)

from src.analysis.ed_core import build_qaqmc_midpoint_state
from src.engines.qaqmc_renyi_work import QAQMCRenyiWorkRydberg
from src.rydberg.lattices import generate_1d_chain


def _ed_s2(psi: np.ndarray, A_mask: np.ndarray) -> float:
    """S_2(A) = -log Tr(ρ_A^2) for state |psi> on N qubits."""
    N = int(np.log2(psi.size))
    A_sites  = [i for i in range(N) if A_mask[i] == 1]
    AC_sites = [i for i in range(N) if A_mask[i] == 0]
    nA, nAC = len(A_sites), len(AC_sites)
    M = np.zeros((1 << nA, 1 << nAC), dtype=np.float64)
    for s in range(psi.size):
        bits = [(s >> i) & 1 for i in range(N)]
        a  = sum(bits[A_sites[j]]  << j for j in range(nA))
        ac = sum(bits[AC_sites[j]] << j for j in range(nAC))
        M[a, ac] = psi[s]
    rho_A = M @ M.T
    tr_rho2 = float(np.trace(rho_A @ rho_A))
    return -float(np.log(max(tr_rho2, 1e-300)))


def _build_setup():
    N, M = 4, 100
    Omega, Rb = 1.0, 1.2
    delta_min, delta_max = -1.0, 2.0
    epsilon = 0.05
    pos = generate_1d_chain(N, 1.0)
    psi = build_qaqmc_midpoint_state(
        N=N, Omega=Omega, delta_min=delta_min, delta_max=delta_max,
        Rb=Rb, M=M, pos=pos, epsilon=epsilon, neighbor_cutoff=None,
    )
    return dict(N=N, M=M, Omega=Omega, Rb=Rb,
                delta_min=delta_min, delta_max=delta_max,
                epsilon=epsilon, pos=pos, psi=psi)


def test_empty_to_A_matches_ed():
    setup = _build_setup()
    N, M = setup["N"], setup["M"]
    A = np.array([1, 1, 0, 0], dtype=np.uint8)
    s2_ed = _ed_s2(setup["psi"], A)

    eng = QAQMCRenyiWorkRydberg(
        N=N, M=M, Omega=setup["Omega"], Rb=setup["Rb"],
        delta_min=setup["delta_min"], delta_max=setup["delta_max"],
        epsilon=setup["epsilon"], seed=7, pos=setup["pos"],
        neighbor_cutoff=-1, delta_groups=200,
    )
    eng.set_region(A)
    # K = 50 is a compromise: large enough that Jarzynski "multiply by 1"
    # dissipation bias stays under ~0.05 for this N=4 system, small enough
    # that 2000 trajectories run in well under a minute.  For rigorous
    # convergence validation use scripts/slurm_scripts/probe_renyi_work_ed.sh.
    eng.set_lambda_schedule(np.linspace(0.0, 1.0, 51))
    eng.thermalize(3000)
    res = eng.run_trajectories(n_trajectories=2000, decorrelation_steps=500)

    # Bootstrap std error of delta_s2
    exp_minus_w = np.exp(-res.work_samples)
    rng = np.random.default_rng(0)
    boot = [-np.log(exp_minus_w[rng.integers(0, len(exp_minus_w), len(exp_minus_w))].mean())
            for _ in range(200)]
    sd = float(np.std(boot, ddof=1))

    diff = res.delta_s2 - s2_ed
    # Tolerance accounts for K=50 Jarzynski dissipation bias (~0.04 typical)
    # plus statistical fluctuation; rigorous convergence is tested via MPI.
    assert abs(diff) < max(5 * sd, 0.15), (
        f"∅ → A delta_s2 mismatch: QMC={res.delta_s2:+.4f}, ED={s2_ed:+.4f}, "
        f"diff={diff:+.4f}, bootstrap sd={sd:.4f}"
    )


def test_nested_pair_matches_ed():
    setup = _build_setup()
    N, M = setup["N"], setup["M"]
    A  = np.array([1, 0, 0, 0], dtype=np.uint8)
    AB = np.array([1, 1, 0, 0], dtype=np.uint8)
    s2_A  = _ed_s2(setup["psi"], A)
    s2_AB = _ed_s2(setup["psi"], AB)
    delta_ed = s2_AB - s2_A

    eng = QAQMCRenyiWorkRydberg(
        N=N, M=M, Omega=setup["Omega"], Rb=setup["Rb"],
        delta_min=setup["delta_min"], delta_max=setup["delta_max"],
        epsilon=setup["epsilon"], seed=11, pos=setup["pos"],
        neighbor_cutoff=-1, delta_groups=200,
    )
    eng.set_region_pair(A, AB)
    eng.set_lambda_schedule(np.linspace(0.0, 1.0, 5))
    eng.thermalize(3000)
    res = eng.run_trajectories(n_trajectories=2000, decorrelation_steps=200)

    exp_minus_w = np.exp(-res.work_samples)
    rng = np.random.default_rng(0)
    boot = [-np.log(exp_minus_w[rng.integers(0, len(exp_minus_w), len(exp_minus_w))].mean())
            for _ in range(200)]
    sd = float(np.std(boot, ddof=1))

    diff = res.delta_s2 - delta_ed
    # Tolerance accounts for K=50 Jarzynski dissipation bias (~0.04 typical)
    # plus statistical fluctuation; rigorous convergence is tested via MPI.
    assert abs(diff) < max(5 * sd, 0.15), (
        f"A → AB delta_s2 mismatch: QMC={res.delta_s2:+.4f}, ED={delta_ed:+.4f}, "
        f"diff={diff:+.4f}, bootstrap sd={sd:.4f}"
    )


def test_ladder_sum_matches_end_to_end():
    """Sum across nested rungs ∅→A→AB→ABC equals ED S_2(ABC) end-to-end."""
    setup = _build_setup()
    N, M = setup["N"], setup["M"]
    zero = np.zeros(N, dtype=np.uint8)
    A   = np.array([1, 0, 0, 0], dtype=np.uint8)
    AB  = np.array([1, 1, 0, 0], dtype=np.uint8)
    ABC = np.array([1, 1, 1, 0], dtype=np.uint8)
    s2_ABC_ed = _ed_s2(setup["psi"], ABC)

    eng = QAQMCRenyiWorkRydberg(
        N=N, M=M, Omega=setup["Omega"], Rb=setup["Rb"],
        delta_min=setup["delta_min"], delta_max=setup["delta_max"],
        epsilon=setup["epsilon"], seed=17, pos=setup["pos"],
        neighbor_cutoff=-1, delta_groups=200,
    )

    total = 0.0
    for start, end in [(zero, A), (A, AB), (AB, ABC)]:
        eng.set_region_pair(start, end)
        eng.set_lambda_schedule(np.linspace(0.0, 1.0, 51))   # K=50 per rung
        eng.thermalize(3000)
        res = eng.run_trajectories(n_trajectories=2000, decorrelation_steps=500)
        total += res.delta_s2

    diff = total - s2_ABC_ed
    # Tolerance: 3 rungs × ~0.05 bias each = ~0.15 typical at K=50
    assert abs(diff) < 0.30, (
        f"Ladder sum mismatch: QMC sum={total:+.4f}, ED S_2(ABC)={s2_ABC_ed:+.4f}, "
        f"diff={diff:+.4f}"
    )


def main():
    test_empty_to_A_matches_ed()
    print("  PASS  ∅ → A vs ED")
    test_nested_pair_matches_ed()
    print("  PASS  nested A → AB vs ED")
    test_ladder_sum_matches_end_to_end()
    print("  PASS  ladder ∅ → A → AB → ABC sum vs end-to-end ED")
    print("QAQMCRenyiWorkEngine ED integration tests passed")


if __name__ == "__main__":
    main()
