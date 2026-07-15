"""
Python wrapper for the thermal SSE off-diagonal string estimator.

Measures  O_C(beta) = Tr[X_C e^{-beta H}] / Tr[e^{-beta H}],
X_C = prod_{i in C} sigma_i^x, via the same interpolating-ensemble /
Jarzynski-work protocol as the QAQMC version (src/engines/qaqmc_string_work.py)
— the trajectory logic is inherited unchanged; only the engine differs
(finite-temperature SSEEngine with a periodic-tau seam, see
csrc/cpu/detail/sse_off_diagonal_core.hpp).  The seam defaults to m_star = 0 (tau = 0):
the trace is tau-translation invariant and slot 0 stays valid as M grows.
"""

from __future__ import annotations

import numpy as np

from src.engines.qaqmc_string_work import (QAQMCStringWorkRydberg,
                                           cosine_schedule)
from src.engines.qaqmc_string_work import qaqmc_cpp

__all__ = ["SSEStringWorkRydberg", "cosine_schedule"]


class SSEStringWorkRydberg(QAQMCStringWorkRydberg):
    """Thermal string-work driver around ``qaqmc_cpp.SSEEngine``.

    Typical usage::

        eng = SSEStringWorkRydberg(N=6, beta=4.0, Omega=1.0, Rb=1.4, delta=1.2,
                                   pos=pos)
        eng.set_string_sites([2, 3])            # m_star = 0 by default
        eng.set_lambda_schedule(cosine_schedule(200))
        eng.thermalize(2000)
        res = eng.run_trajectories(n_trajectories=2000)
        print(res.o_c)                          # <X_C>_beta
    """

    def __init__(self, N: int, beta: float, Omega: float = 1.0,
                 Rb: float = 1.2, delta: float = 1.0,
                 epsilon: float = 0.01, seed: int = 42,
                 pos: np.ndarray | None = None,
                 neighbor_cutoff: int | None = None,
                 box_vectors: np.ndarray | None = None):
        if pos is None:
            pos = np.arange(N).reshape(-1, 1).astype(np.float64)
        pos_arr = np.ascontiguousarray(pos, dtype=np.float64)
        nc = neighbor_cutoff if neighbor_cutoff is not None else -1
        box = (np.ascontiguousarray(box_vectors, dtype=np.float64)
               if box_vectors is not None else None)
        # note: deliberately NOT calling super().__init__ (different engine)
        self._eng = qaqmc_cpp.SSEEngine(
            N=N, Omega=Omega, delta=delta, Rb=Rb, beta=float(beta),
            epsilon=epsilon, seed=seed, pos=pos_arr,
            neighbor_cutoff=nc, box_vectors=box)
        self.N = N
        self.beta = float(beta)
        self._length = 0
        self._lambda_schedule: np.ndarray | None = None

    def set_string_sites(self, sites, m_star: int | None = None) -> None:
        sites = list(sites)
        if m_star is None:
            m_star = 0                    # tau = 0 seam (see module docstring)
        self._eng.set_string_sites(sites, int(m_star))
        self._length = len(sites)
