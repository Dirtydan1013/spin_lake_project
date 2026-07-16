"""Thermal SSE off-diagonal string work vs exact diagonalization.

O_C(beta) = Tr[X_C e^{-beta H}] / Tr[e^{-beta H}],  X_C = prod_{i in C} sx_i,
computed exactly from the spectrum and compared with the Jarzynski estimate
from SSEStringWorkRydberg (forward and reverse), on a small Rydberg chain.
Also checks that the seam machinery leaves DIAGONAL physics intact: with the
seam mask held empty, energy/density must match a seam-free engine's values.
"""

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _ROOT)

qaqmc_cpp = pytest.importorskip("qaqmc_cpp")
if not hasattr(qaqmc_cpp.SSEEngine, "topology_sweep"):
    pytest.skip("qaqmc_cpp .so predates the SSE string seam — rebuild csrc",
                allow_module_level=True)

from src.analysis.ed_core import build_rydberg_hamiltonian
from src.engines.sse_string_work import SSEStringWorkRydberg, cosine_schedule
from src.rydberg.lattices import generate_1d_chain

N, OMEGA, RB = 6, 1.0, 1.4
POS = generate_1d_chain(N)


def ed_o_c(delta, beta, sites):
    H = build_rydberg_hamiltonian(N, OMEGA, delta, RB, pos=POS)
    evals, evecs = np.linalg.eigh(H)
    w = np.exp(-beta * (evals - evals[0]))
    dim = 1 << N
    mask = 0
    for s in sites:
        mask |= (1 << s)
    perm = np.arange(dim) ^ mask
    # Tr[X_C e^{-bH}] = sum_k w_k <k|X_C|k>,  <k|X_C|k> = sum_s v_k(s^C) v_k(s)
    xdiag = np.einsum("sk,sk->k", evecs[perm, :], evecs)
    return float((w * xdiag).sum() / w.sum())


def run_engine(delta, beta, sites, direction, seed=17, n_traj=1500,
               k_lambda=120):
    eng = SSEStringWorkRydberg(N=N, beta=beta, Omega=OMEGA, Rb=RB,
                               delta=delta, seed=seed, pos=POS)
    eng.set_string_sites(sites)
    eng.set_lambda_schedule(cosine_schedule(k_lambda))
    eng.thermalize(2000, direction=direction)
    res = eng.run_trajectories(n_trajectories=n_traj, decorrelation_steps=20,
                               n_topology_sweeps_per_lambda=1,
                               n_qaqmc_sweeps_per_lambda=1,
                               direction=direction)
    return res


@pytest.mark.parametrize("sites", [[2], [2, 3]])
def test_o_c_matches_ed_forward_and_reverse(sites):
    delta, beta = 1.0, 3.0
    exact = ed_o_c(delta, beta, sites)
    assert exact > 1e-4          # sanity: measurable signal at these params
    for direction in ("forward", "reverse"):
        res = run_engine(delta, beta, sites, direction)
        # jackknife over trajectory blocks for a defensible error bar
        lj = res.log_j_samples
        blocks = np.array_split(lj, 10)
        sign = 1.0 if direction == "forward" else -1.0
        vals = []
        for b in blocks:
            fin = np.isfinite(b)
            vals.append(np.exp(sign * (np.log(np.mean(
                np.exp(b[fin] - b[fin].max()))) + b[fin].max()))
                if fin.any() else 0.0)
        vals = np.asarray(vals)
        est, sem = vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))
        assert abs(est - exact) < max(6.0 * sem, 0.05 * exact), \
            (f"{direction} {sites}: O_C = {est:.5f}±{sem:.5f} "
             f"vs ED {exact:.5f} (n_eff={res.n_eff:.0f})")


def test_empty_seam_preserves_diagonal_physics():
    # With the string configured but the mask held at B = empty, the engine
    # samples the ORIGINAL ensemble: E and <n> must match a seam-free run
    # within errors (the seam hooks must be exact no-ops for b = 0).
    delta, beta, seed = 1.2, 4.0, 23

    def measure(with_string):
        eng = qaqmc_cpp.SSEEngine(N=N, Omega=OMEGA, delta=delta, Rb=RB,
                                  beta=beta, epsilon=0.01, seed=seed,
                                  pos=np.ascontiguousarray(POS, np.float64))
        if with_string:
            eng.set_string_sites([1, 4], m_star=0)
        r = eng.run(n_equil=2000, n_samples=100_000)
        e = np.asarray(r["energies"])
        d = np.asarray(r["densities"])
        return (e.mean(), e.std() / np.sqrt(len(e) / 50),
                d.mean(), d.std() / np.sqrt(len(d) / 50))

    e0, se0, d0, sd0 = measure(False)
    e1, se1, d1, sd1 = measure(True)
    assert abs(e0 - e1) < 5.0 * np.hypot(se0, se1)
    assert abs(d0 - d1) < 5.0 * np.hypot(sd0, sd1)


def test_fixed_lambda_sector_balance():
    # At fixed lambda the extended ensemble weights sector B by
    # lambda^|B| (1-lambda)^{L-|B|} Z_B; for L = 1 the active fraction must be
    # lambda*O_C / (1-lambda + lambda*O_C) with O_C from ED.
    delta, beta, lam = 1.0, 3.0, 0.5
    exact = ed_o_c(delta, beta, [2])
    expect = lam * exact / (1.0 - lam + lam * exact)

    eng = SSEStringWorkRydberg(N=N, beta=beta, Omega=OMEGA, Rb=RB,
                               delta=delta, seed=5, pos=POS)
    eng.set_string_sites([2])
    for _ in range(2000):
        eng._eng.mc_step()
    hits, n_meas = 0, 6000
    for _ in range(n_meas):
        eng._eng.topology_sweep(lam)
        eng._eng.mc_step()
        hits += int(eng._eng.seam_mask & 1)
    frac = hits / n_meas
    sem = np.sqrt(expect * (1 - expect) / (n_meas / 20))  # crude tau ~ 10
    assert abs(frac - expect) < max(6.0 * sem, 0.02), \
        f"sector fraction {frac:.4f} vs expected {expect:.4f}"
