"""
Exact Diagonalization (ED) for Rydberg atom arrays.
Used for benchmarking QMC on small system sizes.
Includes real-time dynamics with time-dependent detuning.
"""

import numpy as np
from numba import njit
from scipy.linalg import eigh
from src.hamiltonian import build_rydberg_vij


def build_rydberg_hamiltonian(N: int, Omega: float, delta: float, Rb: float, pos: np.ndarray = None) -> np.ndarray:
    """Build the full 2^N x 2^N Hamiltonian matrix."""
    dim = 1 << N
    H = np.zeros((dim, dim), dtype=np.float64)

    V, _, _, _, _, _ = build_rydberg_vij(N, Omega, Rb, pos)

    for s in range(dim):
        diag = 0.0
        for i in range(N):
            ni = (s >> i) & 1
            if ni:
                diag -= delta
            for j in range(i + 1, N):
                nj = (s >> j) & 1
                if ni and nj:
                    diag += V[i, j]
        H[s, s] = diag
        for i in range(N):
            t = s ^ (1 << i)
            H[s, t] -= Omega / 2.0

    return H


@njit(cache=True, nogil=True)
def _build_diag_terms_numba(N: int, dim: int, V: np.ndarray):
    n_tot = np.zeros(dim, dtype=np.float64)
    mz_val = np.zeros(dim, dtype=np.float64)
    dens_val = np.zeros(dim, dtype=np.float64)
    v_diag = np.zeros(dim, dtype=np.float64)

    for s in range(dim):
        nt = 0.0
        mz = 0.0
        for i in range(N):
            ni = (s >> i) & 1
            nt += ni
            phase = 1.0 if (i % 2 == 0) else -1.0
            mz += phase * (ni - 0.5)
        n_tot[s] = nt
        dens_val[s] = nt / N
        mz_val[s] = mz / N

        e = 0.0
        for i in range(N):
            ni = (s >> i) & 1
            if ni == 0:
                continue
            for j in range(i + 1, N):
                nj = (s >> j) & 1
                if nj == 1:
                    e += V[i, j]
        v_diag[s] = e

    return n_tot, dens_val, mz_val, v_diag


@njit(cache=True, nogil=True)
def _apply_minus_h_inplace_numba(inp: np.ndarray, out: np.ndarray,
                                 delta: float, Omega: float, N: int,
                                 n_tot: np.ndarray, v_diag: np.ndarray, const_offset: float):
    dim = inp.shape[0]
    half_omega = 0.5 * Omega
    shift = const_offset + N * half_omega

    for s in range(dim):
        diag_e = v_diag[s] - delta * n_tot[s]
        acc = (shift - diag_e) * inp[s]
        for i in range(N):
            t = s ^ (1 << i)
            acc += half_omega * inp[t]
        out[s] = acc


@njit(cache=True, nogil=True)
def _qaqmc_slice_offset(delta: float, N: int, vij_list: np.ndarray, epsilon: float) -> float:
    """
    Total per-slice constant offset C for the QAQMC propagator (-H + C).
    Matches alias table construction:
      m1   = min(0, delta_b, 2*delta_b - V_ij)
      m2   = min(delta_b, 2*delta_b - V_ij)
      c_ij = |m1| + epsilon * |m2|
    and delta_b = delta / (N - 1).
    """
    if N <= 1:
        return 0.0
    delta_b = delta / (N - 1)
    c_total = 0.0
    for vij in vij_list:
        two_db_vij = 2.0 * delta_b - vij
        m1 = min(0.0, delta_b, two_db_vij)
        m2 = min(delta_b, two_db_vij)
        c_total += abs(m1) + epsilon * abs(m2)
    return c_total


def qaqmc_exact_asymmetric_observables(
    N: int,
    Omega: float,
    delta_min: float,
    delta_max: float,
    Rb: float,
    M: int,
    pos: np.ndarray = None,
    psi0: np.ndarray = None,
    epsilon: float = 0.01,
    normalize_each_step: bool = True,
):
    if M <= 0:
        raise ValueError("M must be positive.")

    dim = 1 << N
    if psi0 is None:
        psi = np.zeros(dim, dtype=np.float64)
        psi[0] = 1.0
    else:
        psi = np.asarray(psi0, dtype=np.float64).copy()
        n0 = np.linalg.norm(psi)
        if n0 == 0.0:
            raise ValueError("psi0 must have non-zero norm.")
        psi /= n0

    V, _, _, vij_list, _, _ = build_rydberg_vij(N, Omega, Rb, pos)
    n_tot, dens_val, _mz_val, v_diag = _build_diag_terms_numba(N, dim, V)

    M_total = 2 * M
    d_lambda = (delta_max - delta_min) / M
    lambdas = np.empty(M_total, dtype=np.float64)
    for p in range(M):
        lambdas[p] = delta_min + p * d_lambda
    for p in range(M, M_total):
        lambdas[p] = delta_max - (p - M) * d_lambda

    offsets = np.empty(M_total, dtype=np.float64)
    for t in range(M_total):
        offsets[t] = _qaqmc_slice_offset(lambdas[t], N, vij_list, epsilon)

    # Forward states R_t: right_states[t] == state after t operators
    right_states = np.empty((M_total + 1, dim), dtype=np.float64)
    right_states[0, :] = psi
    cur_r = psi.copy()
    nxt_r = np.empty(dim, dtype=np.float64)
    for t in range(M_total):
        _apply_minus_h_inplace_numba(cur_r, nxt_r, lambdas[t], Omega, N, n_tot, v_diag, offsets[t])
        cur_r, nxt_r = nxt_r, cur_r
        if normalize_each_step:
            nr = np.linalg.norm(cur_r)
            if nr > 0:
                cur_r /= nr
        right_states[t + 1, :] = cur_r

    deltas = lambdas[:M]
    density_mean = np.empty(M, dtype=np.float64)

    # Backward sweep computes L_t on the fly, avoiding a full left_states buffer
    cur_l = psi.copy()
    nxt_l = np.empty(dim, dtype=np.float64)
    l_sym = None
    for t in range(M_total - 1, -1, -1):
        _apply_minus_h_inplace_numba(cur_l, nxt_l, lambdas[t], Omega, N, n_tot, v_diag, offsets[t])
        cur_l, nxt_l = nxt_l, cur_l
        if normalize_each_step:
            nl = np.linalg.norm(cur_l)
            if nl > 0:
                cur_l /= nl

        if t == M:
            l_sym = cur_l.copy()

        if t < M:
            r = right_states[t]
            l = cur_l
            weight = l * r
            denom = np.sum(weight)
            if abs(denom) < 1e-300:
                raise RuntimeError(f"Asymmetric denominator nearly zero at t={t}.")
            density_mean[t] = np.sum(weight * dens_val) / denom

    if l_sym is None:
        raise RuntimeError("Failed to build symmetric left state at t=M.")

    r_sym = right_states[M]
    weight_sym = l_sym * r_sym
    denom_sym = np.sum(weight_sym)
    dens_sym = float(np.sum(weight_sym * dens_val) / denom_sym)

    return {
        "deltas": deltas,
        "density_mean": density_mean,
        "density_err": np.zeros(M, dtype=np.float64),
        "density_symmetric": dens_sym,
    }


# ── Real-time dynamics ────────────────────────────────────────────────────────

@njit(cache=True, nogil=True)
def _build_H_offdiag(N, dim, half_omega):
    """Build the off-diagonal part of H (sigma_x terms). Called once."""
    H = np.zeros((dim, dim), dtype=np.float64)
    for s in range(dim):
        for i in range(N):
            t = s ^ (1 << i)
            H[s, t] = -half_omega
    return H


@njit(cache=True, nogil=True)
def _set_H_diagonal(H_diag, v_diag, n_tot, delta, dim):
    """Update H diagonal in-place for a given delta. Called every step."""
    for s in range(dim):
        H_diag[s] = v_diag[s] - delta * n_tot[s]


@njit(cache=True, nogil=True)
def _propagate_eigh(psi_re, psi_im, evals, evecs, dt, dim):
    """
    Apply exp(-i H dt) to psi using precomputed eigendecomposition.
    psi_new = U @ diag(exp(-i E dt)) @ U^T @ psi
    All done in real arithmetic where possible.
    """
    n = evals.shape[0]
    # Project onto eigenbasis: c = U^T @ psi  (complex)
    c_re = np.zeros(n, dtype=np.float64)
    c_im = np.zeros(n, dtype=np.float64)
    for k in range(n):
        r = 0.0
        m = 0.0
        for s in range(dim):
            r += evecs[s, k] * psi_re[s]
            m += evecs[s, k] * psi_im[s]
        c_re[k] = r
        c_im[k] = m

    # Apply phase: c_k *= exp(-i E_k dt)
    for k in range(n):
        phase = -evals[k] * dt
        cos_p = np.cos(phase)
        sin_p = np.sin(phase)
        re = c_re[k] * cos_p - c_im[k] * sin_p
        im = c_re[k] * sin_p + c_im[k] * cos_p
        c_re[k] = re
        c_im[k] = im

    # Transform back: psi = U @ c
    for s in range(dim):
        r = 0.0
        m = 0.0
        for k in range(n):
            r += evecs[s, k] * c_re[k]
            m += evecs[s, k] * c_im[k]
        psi_re[s] = r
        psi_im[s] = m


@njit(cache=True, nogil=True)
def _measure_diag(psi_re, psi_im, obs_diag, dim):
    """Compute <psi|O|psi> for diagonal observable O."""
    val = 0.0
    for s in range(dim):
        prob = psi_re[s] * psi_re[s] + psi_im[s] * psi_im[s]
        val += prob * obs_diag[s]
    return val


def ramp_real_time(N: int, Omega: float, Rb: float, pos: np.ndarray,
                   delta_i: float, delta_f: float, v: float,
                   dt: float = 0.01, psi0: np.ndarray = None):
    """
    Simulate real-time linear ramp of detuning via exact diagonalization.

    Solves i d|psi>/dt = H(t)|psi> with delta(t) = delta_i + v*t.
    Uses eigh decomposition at each step (faster than expm for dim < ~1024).

    Parameters
    ----------
    N        : number of atoms
    Omega    : Rabi frequency
    Rb       : blockade radius
    pos      : (N, d) atom positions
    delta_i  : initial detuning
    delta_f  : final detuning
    v        : sweep speed  (delta_f - delta_i) / T_total
    dt       : time step for piecewise-constant propagation
    psi0     : initial state vector (default: ground state of H(delta_i))

    Returns
    -------
    dict with keys:
        times      : (n_steps+1,) time array
        deltas     : (n_steps+1,) delta(t) at each step
        density    : (n_steps+1,) mean Rydberg density <n>(t)
        mz         : (n_steps+1,) staggered magnetization
        psi_final  : final state vector (complex128)
    """
    T_total = abs(delta_f - delta_i) / v
    n_steps = int(np.ceil(T_total / dt))
    dt_actual = T_total / n_steps
    dim = 1 << N

    # Precompute delta-independent parts (single build_rydberg_vij call)
    V, _, _, _, _, _ = build_rydberg_vij(N, Omega, Rb, pos)
    n_tot, dens_diag, mz_diag, v_diag = _build_diag_terms_numba(N, dim, V)

    # Off-diagonal H (sigma_x), built once
    half_omega = Omega / 2.0
    H_offdiag = _build_H_offdiag(N, dim, half_omega)

    # Working H matrix (updated each step)
    H = H_offdiag.copy()

    # Initial state
    if psi0 is None:
        _set_H_diagonal(H.ravel()[::dim + 1], v_diag, n_tot, delta_i, dim)
        evals, evecs = eigh(H)
        psi0_c = evecs[:, 0].astype(np.complex128)
    else:
        psi0_c = np.asarray(psi0, dtype=np.complex128)

    psi_re = psi0_c.real.copy()
    psi_im = psi0_c.imag.copy()

    # Output arrays
    times = np.empty(n_steps + 1)
    deltas_out = np.empty(n_steps + 1)
    density = np.empty(n_steps + 1)
    mz = np.empty(n_steps + 1)

    # Measure initial state
    times[0] = 0.0
    deltas_out[0] = delta_i
    density[0] = _measure_diag(psi_re, psi_im, dens_diag, dim)
    mz[0] = _measure_diag(psi_re, psi_im, mz_diag, dim)

    # Time evolution
    H_diag_view = np.diagonal(H).copy()  # writable diagonal buffer
    for step in range(n_steps):
        t_mid = (step + 0.5) * dt_actual
        delta_t = delta_i + v * t_mid

        # Update H diagonal and diagonalize
        _set_H_diagonal(H_diag_view, v_diag, n_tot, delta_t, dim)
        np.fill_diagonal(H, H_diag_view)
        evals, evecs = eigh(H)

        # Propagate: |psi> = exp(-i H dt) |psi>
        _propagate_eigh(psi_re, psi_im, evals, evecs, dt_actual, dim)

        t_now = (step + 1) * dt_actual
        deltas_out[step + 1] = delta_i + v * t_now
        times[step + 1] = t_now
        density[step + 1] = _measure_diag(psi_re, psi_im, dens_diag, dim)
        mz[step + 1] = _measure_diag(psi_re, psi_im, mz_diag, dim)

    psi_final = psi_re + 1j * psi_im
    return {
        'times': times,
        'deltas': deltas_out,
        'density': density,
        'mz': mz,
        'psi_final': psi_final,
    }
