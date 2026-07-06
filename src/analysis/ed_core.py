"""
Exact Diagonalization (ED) for Rydberg atom arrays.
Used for benchmarking QMC on small system sizes.
Includes real-time dynamics with time-dependent detuning.
"""

import numpy as np
from numba import njit
from scipy.linalg import eigh
from src.rydberg.hamiltonian import build_rydberg_vij


def build_rydberg_hamiltonian(
    N: int,
    Omega: float,
    delta: float,
    Rb: float,
    pos: np.ndarray = None,
    neighbor_cutoff: int | None = None,
) -> np.ndarray:
    """Build the full 2^N x 2^N Hamiltonian matrix."""
    dim = 1 << N
    H = np.zeros((dim, dim), dtype=np.float64)

    V, _, _, _, _, _ = build_rydberg_vij(N, Omega, Rb, pos, verbose=False, neighbor_cutoff=neighbor_cutoff)

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
def _qaqmc_slice_offset(
    delta: float,
    vij_list: np.ndarray,
    bond_sites: np.ndarray,
    coord_number: np.ndarray,
    epsilon: float,
) -> float:
    """
    Total per-slice constant offset C for the QAQMC propagator (-H + C).
    Matches alias table construction:
      m1   = min(0, delta_b, 2*delta_b - V_ij)
      m2   = min(delta_b, 2*delta_b - V_ij)
      c_ij = |m1| + epsilon * |m2|
    and the per-bond detunings are split asymmetrically as
    delta_i = delta / coord_number[i].
    """
    c_total = 0.0
    for b, vij in enumerate(vij_list):
        si = bond_sites[b, 0]
        sj = bond_sites[b, 1]
        delta_i = delta / coord_number[si] if coord_number[si] > 0 else 0.0
        delta_j = delta / coord_number[sj] if coord_number[sj] > 0 else 0.0
        m1 = min(0.0, delta_j, delta_i, delta_i + delta_j - vij)
        m2 = min(abs(0.0), abs(delta_j), abs(delta_i), abs(delta_i + delta_j - vij))
        c_total += abs(m1) + epsilon * abs(m2)
    return c_total


def build_qaqmc_midpoint_state(
    N: int,
    Omega: float,
    delta_min: float,
    delta_max: float,
    Rb: float,
    M: int,
    pos: np.ndarray = None,
    epsilon: float = 0.01,
    neighbor_cutoff: int | None = None,
):
    if M <= 0:
        raise ValueError("M must be positive.")

    dim = 1 << N
    psi = np.zeros(dim, dtype=np.float64)
    psi[0] = 1.0

    V, _, _, vij_list, bond_sites, coord_number = build_rydberg_vij(
        N,
        Omega,
        Rb,
        pos,
        verbose=False,
        neighbor_cutoff=neighbor_cutoff,
    )
    n_tot, _dens_val, _mz_val, v_diag = _build_diag_terms_numba(N, dim, V)

    for p in range(M):
        delta = delta_min + (delta_max - delta_min) * (p / M)
        c_shift = _qaqmc_slice_offset(delta, vij_list, bond_sites, coord_number, epsilon) + N * (Omega / 2.0)
        out = np.empty(dim, dtype=np.float64)
        _apply_minus_h_inplace_numba(psi, out, delta, Omega, N, n_tot, v_diag, c_shift)
        psi = out
        norm = np.linalg.norm(psi)
        if norm > 0.0:
            psi /= norm

    return psi


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

    V, _, _, vij_list, bond_sites, coord_number = build_rydberg_vij(
        N,
        Omega,
        Rb,
        pos,
        verbose=False,
    )
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
        offsets[t] = _qaqmc_slice_offset(lambdas[t], vij_list, bond_sites, coord_number, epsilon)

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


def qaqmc_exact_renyi2_at_cut(
    N: int,
    Omega: float,
    delta_min: float,
    delta_max: float,
    Rb: float,
    M: int,
    A_mask,
    m_star: int | None = None,
    pos: np.ndarray = None,
    psi0: np.ndarray = None,
    epsilon: float = 0.01,
    neighbor_cutoff: int | None = None,
    normalize_each_step: bool = True,
):
    """
    Exact Renyi-2 swap estimator at an arbitrary QAQMC imaginary-time cut.

    The cut sits before operator index ``m_star``.  The right state contains
    operators ``0 .. m_star-1`` and the left state contains operators
    ``m_star .. 2M-1``.  For an off-center cut these two states differ, so the
    quantity matched by the two-replica swap estimator is

        rho_A = Tr_Ac |R><L| / <L|R>,
        S2(A) = -log Tr_A rho_A^2.

    Near the turning point this approaches the usual midpoint pure-state
    Renyi-2 entropy as the QAQMC discretization is refined, but at finite M the
    exact swap estimator is this left/right transition-density quantity.
    """
    if M <= 0:
        raise ValueError("M must be positive.")

    A_mask = np.asarray(A_mask, dtype=np.uint8)
    if A_mask.shape != (N,):
        raise ValueError(f"A_mask must have shape ({N},), got {A_mask.shape}.")

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

    M_total = 2 * M
    if m_star is None:
        m_star = M
    if not (0 <= m_star <= M_total):
        raise ValueError(f"m_star must be in [0, {M_total}], got {m_star}.")

    V, _, _, vij_list, bond_sites, coord_number = build_rydberg_vij(
        N, Omega, Rb, pos, verbose=False, neighbor_cutoff=neighbor_cutoff,
    )
    n_tot, _dens_val, _mz_val, v_diag = _build_diag_terms_numba(N, dim, V)

    d_lambda = (delta_max - delta_min) / M
    lambdas = np.empty(M_total, dtype=np.float64)
    for p in range(M):
        lambdas[p] = delta_min + p * d_lambda
    for p in range(M, M_total):
        lambdas[p] = delta_max - (p - M) * d_lambda

    offsets = np.empty(M_total, dtype=np.float64)
    for t in range(M_total):
        offsets[t] = _qaqmc_slice_offset(lambdas[t], vij_list, bond_sites, coord_number, epsilon)

    r_state = psi.copy()
    tmp = np.empty(dim, dtype=np.float64)
    for t in range(m_star):
        _apply_minus_h_inplace_numba(r_state, tmp, lambdas[t], Omega, N, n_tot, v_diag, offsets[t])
        r_state, tmp = tmp, r_state
        if normalize_each_step:
            nr = np.linalg.norm(r_state)
            if nr > 0.0:
                r_state /= nr

    l_state = psi.copy()
    for t in range(M_total - 1, m_star - 1, -1):
        _apply_minus_h_inplace_numba(l_state, tmp, lambdas[t], Omega, N, n_tot, v_diag, offsets[t])
        l_state, tmp = tmp, l_state
        if normalize_each_step:
            nl = np.linalg.norm(l_state)
            if nl > 0.0:
                l_state /= nl

    overlap = float(np.dot(l_state, r_state))
    if abs(overlap) < 1e-300:
        raise RuntimeError(f"Asymmetric Renyi denominator nearly zero at m_star={m_star}.")

    A_sites = [i for i in range(N) if A_mask[i] == 1]
    Ac_sites = [i for i in range(N) if A_mask[i] == 0]
    nA, nAc = len(A_sites), len(Ac_sites)
    r_mat = np.zeros((1 << nA, 1 << nAc), dtype=np.float64)
    l_mat = np.zeros_like(r_mat)
    for s in range(dim):
        bits = [(s >> i) & 1 for i in range(N)]
        a = sum(bits[A_sites[j]] << j for j in range(nA))
        ac = sum(bits[Ac_sites[j]] << j for j in range(nAc))
        r_mat[a, ac] = r_state[s]
        l_mat[a, ac] = l_state[s]

    rho_A = (r_mat @ l_mat.T) / overlap
    purity = float(np.trace(rho_A @ rho_A))
    if purity <= 0.0:
        raise RuntimeError(f"Asymmetric Renyi purity is non-positive: {purity}.")

    if m_star < M_total:
        delta_at_cut = float(lambdas[m_star])
    else:
        delta_at_cut = float(delta_min)

    return {
        "m_star": int(m_star),
        "delta_at_cut": delta_at_cut,
        "overlap": overlap,
        "purity": purity,
        "S2": -float(np.log(purity)),
        "rho_A": rho_A,
        "right_state": r_state,
        "left_state": l_state,
    }


def qaqmc_exact_string_zratio(
    N: int,
    Omega: float,
    delta_min: float,
    delta_max: float,
    Rb: float,
    M: int,
    B_sets,
    m_star: int | None = None,
    pos: np.ndarray = None,
    psi0: np.ndarray = None,
    epsilon: float = 0.01,
    neighbor_cutoff: int | None = None,
    normalize_each_step: bool = True,
):
    """
    Exact ground truth for the QAQMC off-diagonal string matrix elements

        Z_B = <psi_L| P_L  X_B  P_R |psi_R>,      X_B = prod_{i in B} sigma_i^x

    inserted at imaginary-time slice ``m_star`` (default: the ramp turning
    point p=M, matching ``state_at_M_`` in the C++ engine). Generalizes
    ``qaqmc_exact_asymmetric_observables``'s symmetric-cut construction to an
    arbitrary cut position and to an X_B insertion instead of a plain overlap.

    Convention (must match the C++ seam implementation): the cut sits *before*
    operator index ``m_star`` in the 0-indexed operator string, i.e.
    ``r_state`` includes operators ``0 .. m_star-1`` and ``l_state`` includes
    operators ``m_star .. M_total-1``. Occupation index bit ``i`` = site ``i``
    (matches ``build_rydberg_hamiltonian`` / ``_apply_minus_h_inplace_numba``).

    Parameters
    ----------
    B_sets : iterable of iterables of int
        Each element is a subset of sites (a candidate ``B``) to evaluate.
        The empty set (``Z_empty``) is always computed as the normalization
        baseline and does not need to be included explicitly.

    Returns
    -------
    dict with:
        'm_star' : int
        'Z_empty': float                          -- Z at B = empty set
        'Z_B'    : dict[frozenset[int], float]     -- Z_B for each requested B
        'O_B'    : dict[frozenset[int], float]     -- Z_B / Z_empty
    """
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

    M_total = 2 * M
    if m_star is None:
        m_star = M
    if not (0 <= m_star <= M_total):
        raise ValueError(f"m_star must be in [0, {M_total}], got {m_star}.")

    V, _, _, vij_list, bond_sites, coord_number = build_rydberg_vij(
        N, Omega, Rb, pos, verbose=False, neighbor_cutoff=neighbor_cutoff,
    )
    n_tot, _dens_val, _mz_val, v_diag = _build_diag_terms_numba(N, dim, V)

    d_lambda = (delta_max - delta_min) / M
    lambdas = np.empty(M_total, dtype=np.float64)
    for p in range(M):
        lambdas[p] = delta_min + p * d_lambda
    for p in range(M, M_total):
        lambdas[p] = delta_max - (p - M) * d_lambda

    offsets = np.empty(M_total, dtype=np.float64)
    for t in range(M_total):
        offsets[t] = _qaqmc_slice_offset(lambdas[t], vij_list, bond_sites, coord_number, epsilon)

    # r_state = H(m_star-1) ... H(0) psi   (operators strictly before the cut)
    r_state = psi.copy()
    tmp = np.empty(dim, dtype=np.float64)
    for t in range(m_star):
        _apply_minus_h_inplace_numba(r_state, tmp, lambdas[t], Omega, N, n_tot, v_diag, offsets[t])
        r_state, tmp = tmp, r_state
        if normalize_each_step:
            nr = np.linalg.norm(r_state)
            if nr > 0:
                r_state /= nr

    # l_state = H(m_star) ... H(M_total-1) psi   (operators at/after the cut,
    # accumulated backward from the same boundary state)
    l_state = psi.copy()
    for t in range(M_total - 1, m_star - 1, -1):
        _apply_minus_h_inplace_numba(l_state, tmp, lambdas[t], Omega, N, n_tot, v_diag, offsets[t])
        l_state, tmp = tmp, l_state
        if normalize_each_step:
            nl = np.linalg.norm(l_state)
            if nl > 0:
                l_state /= nl

    def _z_for_mask(mask: int) -> float:
        if mask == 0:
            return float(np.sum(l_state * r_state))
        idx = np.arange(dim) ^ mask
        return float(np.sum(l_state * r_state[idx]))

    z_empty = _z_for_mask(0)

    z_b = {}
    o_b = {}
    for sites in B_sets:
        sites = frozenset(sites)
        mask = 0
        for i in sites:
            mask |= (1 << i)
        z = _z_for_mask(mask)
        z_b[sites] = z
        o_b[sites] = z / z_empty if z_empty != 0.0 else float("nan")

    return {
        "m_star": m_star,
        "Z_empty": z_empty,
        "Z_B": z_b,
        "O_B": o_b,
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
