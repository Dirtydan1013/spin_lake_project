"""
Measurement logics and observable formulas for Exact Diagonalization and QMC algorithms.
"""

import numpy as np

# --- Helper functions for array measurements ---

def calc_density(state: np.ndarray) -> float:
    """Mean Rydberg occupancy: (1/N) sum_i n_i."""
    return float(np.mean(state))


def calc_staggered_magnetization(state: np.ndarray) -> float:
    """Staggered magnetization: (1/N) sum_i (-1)^i (n_i - 0.5)."""
    N = len(state)
    phases = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    return float(np.sum(phases * (state - 0.5)) / N)
