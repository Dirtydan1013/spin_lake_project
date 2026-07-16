"""QAQMC symmetric-point density lag == 2 * v_eff * chi_F  (ED only).

Adiabatic perturbation theory for the imaginary-time product evolution
(Liu-Polkovnikov-Sandvik, PRB 87, 174302): at the SYMMETRIC point t = M the
evolved state lags the instantaneous ground state and a diagonal observable
V = dH/d(delta) = -sum_i n_i deviates linearly in the velocity,

    <V>_M - <V>_gs = +2 * v_eff * chi_F,
    v_eff = Dlam_per_op * Etil,   Etil = offset(delta) - E_gs(delta),

(Etil is the ground-state eigenvalue of the applied operator -H + offset;
1/Etil is the effective imaginary time per operator).  At ASYMMETRIC points
the palindromic operator sequence is transpose-symmetric under t <-> 2M-t, so
the O(v) term cancels and the lag is O(v^2) — the linear chi_F extraction is
only valid at the sweep endpoint.  Both facts are asserted here.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from src.analysis.ed_core import (build_rydberg_hamiltonian,
                                  qaqmc_exact_asymmetric_observables,
                                  _qaqmc_slice_offset)
from src.rydberg.hamiltonian import build_rydberg_vij
from src.rydberg.lattices import generate_1d_chain

N, OMEGA, RB, EPS = 6, 1.0, 1.4, 0.01
POS = generate_1d_chain(N)


def _gs_props(delta):
    H = build_rydberg_hamiltonian(N, OMEGA, delta, RB, pos=POS)
    evals, evecs = np.linalg.eigh(H)
    dim = 1 << N
    n_tot = np.array([bin(s).count("1") for s in range(dim)], dtype=np.float64)
    v0 = evecs[:, 0]
    dens = float(v0 @ (n_tot * v0)) / N
    Vn0 = evecs.T @ (-(n_tot) * v0)
    de = evals - evals[0]
    de[0] = np.inf
    chi = float(np.sum(np.abs(Vn0) ** 2 / de**2))
    _, _, _, vij_list, bond_sites, coord = build_rydberg_vij(
        N, OMEGA, RB, POS, verbose=False)
    off = _qaqmc_slice_offset(delta, vij_list, bond_sites, coord, EPS)
    return dens, chi, off - evals[0]


def test_symmetric_point_lag_is_2_v_chi_f():
    dmax = 1.2
    dens_gs, chi, etil = _gs_props(dmax)
    c_prev = None
    for M in (400, 800, 1600):
        r = qaqmc_exact_asymmetric_observables(
            N, OMEGA, 0.0, dmax, RB, M, pos=POS, epsilon=EPS)
        dev = r["density_symmetric"] - dens_gs
        c = -dev * N / ((dmax / M) * etil * chi)
        assert dev < 0.0                      # density lags below gs while δ rises
        if c_prev is not None:                # converging toward 2 from above
            assert abs(c - 2.0) < abs(c_prev - 2.0)
        c_prev = c
    assert abs(c_prev - 2.0) < 0.25, f"velocity constant c={c_prev:.3f} != 2"


def test_asymmetric_points_have_no_linear_term():
    # Palindrome transpose symmetry: away from t = M the lag is O(v^2), i.e.
    # dev * M^2 stays put while dev * M (a linear term) would not.
    dmax = 1.5
    devs = {}
    for M in (400, 800):
        r = qaqmc_exact_asymmetric_observables(
            N, OMEGA, 0.0, dmax, RB, M, pos=POS, epsilon=EPS)
        t = int(0.6 * M)
        dens_gs, _chi, _etil = _gs_props(r["deltas"][t])
        devs[M] = r["density_mean"][t] - dens_gs
    ratio = devs[400] / devs[800]
    assert 3.0 < ratio < 6.0, \
        f"asymmetric lag ratio {ratio:.2f}: expected ~4 (O(v^2)), ~2 would be O(v)"
