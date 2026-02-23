"""
Measurement logics and observable formulas for Exact Diagonalization and QMC algorithms.
"""

import numpy as np

# --- Helper functions for array measurements ---

def calc_density(state: np.ndarray) -> float:
    """Average site density (n_i)."""
    return float(np.mean(state))

def calc_staggered_magnetization(state: np.ndarray) -> float:
    """Staggered magnetization for A-B bipartite sublattices."""
    N = len(state)
    m_z = 0.0
    for i in range(N):
        # A simple staggered phase (-1)^i for 1D. 
        # (This can be generalized using bipartite neighbor coordinates later)
        sz = state[i] - 0.5
        phase = 1.0 if i % 2 == 0 else -1.0
        m_z += phase * sz
    return m_z / N

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
