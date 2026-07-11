"""SSE fidelity-susceptibility estimator vs exact diagonalization.

The engine's measure_chi_f_terms() returns (G_L, G_R) = sums of
d(ln W_p)/d(delta) over the bond ops in the two halves of the operator
string.  The Wang-Liu-Troyer estimator (PRX 5, 031007) is

    chi_F(beta) = ( <G_L G_R> - <G_L><G_R> ) / 2
                = int_0^{beta/2} dtau tau [ <V(tau)V(0)> - <V>^2 ],

with V = dH/d(delta) = -sum_i n_i.  The right-hand side is evaluated here
exactly from the spectrum; agreement pins both the per-vertex d ln W/d delta
table (raw weights AND the cij offset branches) and the normalisation.
beta -> infinity recovers the ground-state chi_F = sum_n |<n|V|0>|^2/(E_n-E_0)^2.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from src.analysis.ed_core import build_rydberg_hamiltonian
from src.rydberg.lattices import generate_1d_chain

qaqmc_cpp = pytest.importorskip("qaqmc_cpp")
if not hasattr(qaqmc_cpp.SSEEngine, "measure_chi_f_terms"):
    pytest.skip("qaqmc_cpp .so predates measure_chi_f_terms — rebuild csrc",
                allow_module_level=True)


def ed_chi_f(N, Omega, delta, Rb, pos, beta):
    """chi_F(beta) = int_0^{beta/2} tau [<V(tau)V(0)> - <V>^2] dtau, exact."""
    H = build_rydberg_hamiltonian(N, Omega, delta, Rb, pos=pos)
    evals, evecs = np.linalg.eigh(H)
    e = evals - evals[0]
    dim = 1 << N
    n_tot = np.array([bin(s).count("1") for s in range(dim)], dtype=np.float64)
    V = evecs.T @ (-(n_tot[:, None]) * evecs)      # V = -sum_i n_i, eigenbasis
    w = np.exp(-beta * e)
    Z = w.sum()
    v_mean = (w * np.diag(V)).sum() / Z

    T = beta / 2.0
    a = e[:, None] - e[None, :]                    # E_m - E_n
    # int_0^T tau e^{a tau} dtau, weighted by e^{-beta E_m}:
    #   a != 0: [e^{-beta(E_m+E_n)/2} (aT - 1) + e^{-beta E_m}] / a^2
    #   a == 0:  e^{-beta E_m} T^2/2
    with np.errstate(divide="ignore", invalid="ignore"):
        mid = np.exp(-0.5 * beta * (e[:, None] + e[None, :]))
        I = (mid * (a * T - 1.0) + w[:, None]) / a**2
    np.fill_diagonal(I, 0.0)
    I[np.abs(a) < 1e-12] = 0.0
    same = (np.abs(a) < 1e-12)
    corr = (np.abs(V) ** 2 * I).sum()
    corr += (np.abs(V) ** 2 * same * w[:, None]).sum() * (T**2 / 2.0)
    return corr / Z - v_mean**2 * (T**2 / 2.0)


def sse_chi_f(N, Omega, delta, Rb, pos, beta, seed=11,
              n_equil=4000, n_samples=400_000, n_bins=20):
    eng = qaqmc_cpp.SSEEngine(N=N, Omega=Omega, delta=delta, Rb=Rb,
                              beta=beta, epsilon=0.01, seed=seed,
                              pos=np.ascontiguousarray(pos, dtype=np.float64))
    r = eng.run(n_equil=n_equil, n_samples=n_samples, measure_chi_f=True)
    gl = np.asarray(r["chi_gl"], dtype=np.float64)
    gr = np.asarray(r["chi_gr"], dtype=np.float64)
    # covariance per bin (bin-local means) -> mean, SEM over bins
    vals = []
    for b in np.array_split(np.arange(n_samples), n_bins):
        vals.append(0.5 * (np.mean(gl[b] * gr[b]) - gl[b].mean() * gr[b].mean()))
    vals = np.asarray(vals)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(n_bins), gl, gr


@pytest.mark.parametrize("delta,beta", [(1.2, 4.0), (0.5, 6.0)])
def test_chi_f_matches_ed(delta, beta):
    N, Omega, Rb = 6, 1.0, 1.4
    pos = generate_1d_chain(N)
    exact = ed_chi_f(N, Omega, delta, Rb, pos, beta)
    est, sem, _gl, _gr = sse_chi_f(N, Omega, delta, Rb, pos, beta)
    assert exact > 0.0
    assert abs(est - exact) < max(5.0 * sem, 0.02 * exact), \
        f"chi_F SSE {est:.5f}±{sem:.5f} vs ED {exact:.5f}"


def test_half_string_symmetry():
    # <G_L> == <G_R> by tau-translation invariance of the trace.
    N, Omega, Rb = 6, 1.0, 1.4
    pos = generate_1d_chain(N)
    _est, _sem, gl, gr = sse_chi_f(N, Omega, 1.2, Rb, pos, 4.0,
                                   n_samples=200_000)
    sem = np.std(gl - gr) / np.sqrt(len(gl))
    assert abs(gl.mean() - gr.mean()) < 6.0 * sem
