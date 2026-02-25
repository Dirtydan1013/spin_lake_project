"""
Measurement logics and observable formulas for Exact Diagonalization and QMC algorithms.
"""

import numpy as np
from numba import njit

# --- Helper functions for array measurements ---

@njit(cache=True, nogil=True)
def calc_density_numba(state: np.ndarray) -> float:
    total = 0.0
    N = len(state)
    for i in range(N):
        total += state[i]
    return total / N if N > 0 else 0.0

@njit(cache=True, nogil=True)
def calc_staggered_magnetization_numba(state: np.ndarray) -> float:
    N = len(state)
    if N == 0:
        return 0.0
    m_z = 0.0
    for i in range(N):
        sz = state[i] - 0.5
        phase = 1.0 if i % 2 == 0 else -1.0
        m_z += phase * sz
    return m_z / N

@njit(cache=True, nogil=True)
def measure_symmetric_from_ops_numba(op_types: np.ndarray, op_sites: np.ndarray, N: int, M: int):
    cur = np.zeros(N, dtype=np.int32)
    for p in range(M):
        if op_types[p] == -1:
            cur[op_sites[p]] ^= 1
    return calc_density_numba(cur), calc_staggered_magnetization_numba(cur)

@njit(cache=True, nogil=True)
def measure_asymmetric_all_from_ops_numba(op_types: np.ndarray, op_sites: np.ndarray, N: int, M: int):
    cur = np.zeros(N, dtype=np.int32)
    densities = np.empty(M)
    m_zs = np.empty(M)
    for p in range(M):
        densities[p] = calc_density_numba(cur)
        m_zs[p] = calc_staggered_magnetization_numba(cur)
        if op_types[p] == -1:
            cur[op_sites[p]] ^= 1
    return densities, m_zs

def calc_density(state: np.ndarray) -> float:
    """Average site density (n_i)."""
    return float(calc_density_numba(state))

def calc_staggered_magnetization(state: np.ndarray) -> float:
    """Staggered magnetization for A-B bipartite sublattices."""
    return float(calc_staggered_magnetization_numba(state))

# --- Macro Observable Analysis ---

def calc_chi(N: int, m_z_sq_mean: float, m_z_abs_mean: float) -> float:
    """
    Computes staggered susceptibility Chi.
    chi = N * (<m_z^2> - <|m_z|>^2)
    """
    return N * (m_z_sq_mean - m_z_abs_mean**2)

def calc_binder_cumulant(m_z_sq_mean: float, m_z_quad_mean: float) -> float:
    """
    Computes Binder Cumulant U.
    U = 1.5 * (1 - <m_z^4> / (3 * <m_z^2>^2))
    """
    if m_z_sq_mean == 0:
        return 0.0
    return 1.5 * (1.0 - m_z_quad_mean / (3.0 * (m_z_sq_mean ** 2)))
