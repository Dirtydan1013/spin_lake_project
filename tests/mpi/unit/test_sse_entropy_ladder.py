"""SSE entropy ladder vs exact diagonalization on a small chain.

Validates the full Method-A pipeline (Wang-Pollet Eq. 7): the analytic
high-T anchor (E_inf, varH) against ED traces, and the ladder-integrated
S(beta) against the exact S = ln Z + beta E from the spectrum.  Runs the
driver in-process (COMM_WORLD size 1 — no mpiexec needed).
"""

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "plots", "plot_diagonal"))
sys.path.insert(0, os.path.join(_ROOT, "plots", "plot_sse"))

pytest.importorskip("qaqmc_cpp")

from src.analysis.ed_core import build_rydberg_hamiltonian
from src.rydberg.lattices import generate_1d_chain
from src.mpi.sse_entropy_mpi import _high_t_anchor, _min_image_vij, run_ladder

N, OMEGA, DELTA, RB = 6, 1.0, 1.2, 1.4
POS = generate_1d_chain(N)


def test_high_t_anchor_matches_ed_traces():
    vij = _min_image_vij(POS, OMEGA, RB, None)
    e_inf, var_h = _high_t_anchor(vij, DELTA, OMEGA, N)
    H = build_rydberg_hamiltonian(N, OMEGA, DELTA, RB, pos=POS)
    dim = H.shape[0]
    tr1 = np.trace(H) / dim
    tr2 = np.trace(H @ H) / dim
    assert abs(e_inf - tr1) < 1e-9 * max(1.0, abs(tr1))
    assert abs(var_h - (tr2 - tr1**2)) < 1e-9 * max(1.0, tr2 - tr1**2)


def test_entropy_curve_matches_ed(tmp_path):
    run_dir = str(tmp_path / "ladder")
    run_ladder(lattice="1d_chain", N=N, nx=0, ny=0, a=1.0, Omega=OMEGA,
               delta=DELTA, Rb=RB, seed=7, boundary="open",
               beta_min=1e-3, beta_max=8.0, n_beta=40,
               n_equil0=500, n_equil_warm=200,
               n_samples=40000, checkpoint=2000,
               run_dir=run_dir, verbose=False)

    from plot_entropy import entropy_curve, load_ladder
    meta, betas, e_mean, e_sem = load_ladder(run_dir, burn_frac=0.25)
    S, S_err = entropy_curve(meta, betas, e_mean, e_sem)

    H = build_rydberg_hamiltonian(N, OMEGA, DELTA, RB, pos=POS)
    evals = np.linalg.eigvalsh(H)

    def s_exact(beta):
        w = np.exp(-beta * (evals - evals[0]))
        Z = w.sum()
        E = (w * evals).sum() / Z
        return np.log(Z) - beta * evals[0] + beta * E

    checked = 0
    for m, b in enumerate(betas):
        if b < 0.5:
            continue
        exact = s_exact(b)
        tol = max(5.0 * S_err[m], 0.02 * N * np.log(2))
        assert abs(S[m] - exact) < tol, \
            f"beta={b:g}: S_sse={S[m]:.4f}±{S_err[m]:.4f} vs ED {exact:.4f}"
        checked += 1
    assert checked >= 5
