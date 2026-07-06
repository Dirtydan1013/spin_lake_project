"""
Python wrapper for the C++ two-replica QAQMC nonequilibrium-work Renyi engine.

This wraps `qaqmc_cpp.QAQMCRenyiWorkEngine` and exposes a Pythonic interface
for driving Jarzynski-style trajectories that estimate
    ΔS_2 = S_2(A_end) − S_2(A_start) = − log ⟨exp(−w)⟩
for any nested mask pair (A_start ⊆ A_end). When A_start = ∅ this reduces to
the paper's S_2(A_end) estimator; for KP ladder rungs the same call returns
the rung increment S_2(M_{k+1}) − S_2(M_k).
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass

import numpy as np

try:
    _gpp = shutil.which("g++")
    if _gpp and os.name == "nt":
        _mingw_bin = os.path.dirname(os.path.realpath(_gpp))
        os.add_dll_directory(_mingw_bin)
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _build_dir = os.path.join(_repo_root, "build")
    if _build_dir not in sys.path:
        sys.path.insert(0, _build_dir)
    import qaqmc_cpp
except (ImportError, OSError) as exc:
    raise ImportError(
        "qaqmc_cpp with QAQMCRenyiWorkEngine is required for src.engines.qaqmc_renyi_work"
    ) from exc


@dataclass
class WorkRunResult:
    """Aggregated result of `run_trajectories`."""
    delta_s2: float                 # = S_2(A_end) - S_2(A_start)
    mean_exp_minus_work: float      # = Z_{A_end} / Z_{A_start}
    work_mean: float
    work_var: float
    trajectory_count: int
    total_topology_attempts: int
    total_topology_accepts: int
    total_unjoined_at_end: int
    work_samples: np.ndarray        # (n_traj,) raw w per trajectory
    final_swap_counts: np.ndarray   # (n_traj,) |B| at end of each trajectory
    unjoined_counts_per_traj: np.ndarray   # (n_traj,) |D|-|B| per traj
    topology_attempts_per_traj: np.ndarray # (n_traj,)
    topology_accepts_per_traj: np.ndarray  # (n_traj,)


@dataclass
class WorkTrajectoryResult:
    """Single-trajectory diagnostics returned by `run_trajectory`."""
    work: float
    exp_minus_work: float
    final_swap_count: int
    unjoined_at_end_count: int
    topology_attempts: int
    topology_accepts: int


class QAQMCRenyiWorkRydberg:
    """Driver around `qaqmc_cpp.QAQMCRenyiWorkEngine`.

    Typical usage::

        eng = QAQMCRenyiWorkRydberg(N=N, M=M, Omega=..., Rb=..., delta_min=..., delta_max=..., pos=pos)
        eng.set_region(A_mask)                          # or set_region_pair(A_start, A_end)
        eng.set_lambda_schedule(np.linspace(0, 1, 101))
        eng.thermalize(2000)
        res = eng.run_trajectories(n_trajectories=500, decorrelation_steps=100)
        print(res.delta_s2, res.work_var)
    """

    def __init__(
        self,
        N: int,
        M: int,
        Omega: float = 1.0,
        Rb: float = 1.2,
        delta_min: float = 0.0,
        delta_max: float = 1.0,
        epsilon: float = 0.01,
        seed: int = 42,
        pos: np.ndarray | None = None,
        neighbor_cutoff: int | None = None,
        delta_groups: int = 600,
    ):
        if pos is None:
            pos = np.arange(N).reshape(-1, 1).astype(np.float64)
        pos_arr = np.ascontiguousarray(pos, dtype=np.float64)
        nc = neighbor_cutoff if neighbor_cutoff is not None else -1

        self._cpp_engine = qaqmc_cpp.QAQMCRenyiWorkEngine(
            N=N, Omega=Omega, delta_min=delta_min, delta_max=delta_max,
            Rb=Rb, M=M, epsilon=epsilon, seed=seed, pos=pos_arr,
            neighbor_cutoff=nc, delta_groups=int(delta_groups),
        )
        self.N = N
        self.M = M
        self.M_total = 2 * M
        self.pos = pos_arr

    # ── Configuration ───────────────────────────────────────────────────────

    def set_region(self, A_mask: np.ndarray) -> None:
        """Set ∅→A pair (paper main-text case). Result is S_2(A)."""
        mask = np.ascontiguousarray(A_mask, dtype=np.uint8)
        if mask.shape != (self.N,):
            raise ValueError(f"A_mask must have shape ({self.N},), got {mask.shape}")
        self._cpp_engine.set_region(mask)

    def set_region_pair(self, A_start_mask: np.ndarray, A_end_mask: np.ndarray) -> None:
        """Set nested pair (A_start ⊆ A_end). Result is S_2(A_end) − S_2(A_start)."""
        s = np.ascontiguousarray(A_start_mask, dtype=np.uint8)
        e = np.ascontiguousarray(A_end_mask,   dtype=np.uint8)
        if s.shape != (self.N,) or e.shape != (self.N,):
            raise ValueError(f"masks must have shape ({self.N},)")
        self._cpp_engine.set_region_pair(s, e)

    def set_cut(self, m_star: int) -> None:
        """Move the entanglement cut (swap boundary) to slice m_star in
        [0, M_total]; default is M (the ramp turning point).  Vacuum-resets
        the operator strings, so call BEFORE thermalize()."""
        self._cpp_engine.set_cut(int(m_star))

    def get_cut(self) -> int:
        return int(self._cpp_engine.get_cut())

    def set_lambda_schedule(self, lambdas: np.ndarray) -> None:
        """Forward λ schedule; must start at 0, end at 1, monotonic non-decreasing."""
        arr = np.ascontiguousarray(lambdas, dtype=np.float64)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError("lambdas must be 1D array with at least two points")
        self._cpp_engine.set_lambda_schedule(arr)

    def set_sweeps_per_lambda(self, n_topology_sweeps: int = 1, n_qaqmc_sweeps: int = 1) -> None:
        self._cpp_engine.set_sweeps_per_lambda(int(n_topology_sweeps), int(n_qaqmc_sweeps))

    # ── Execution ───────────────────────────────────────────────────────────

    def thermalize(self, n_steps: int) -> None:
        """Thermalise in the start sector (backend A_mask = A_start, B = ∅)."""
        self._cpp_engine.thermalize(int(n_steps))

    def run_trajectory(self) -> WorkTrajectoryResult:
        r = self._cpp_engine.run_trajectory()
        return WorkTrajectoryResult(
            work=float(r.work),
            exp_minus_work=float(r.exp_minus_work),
            final_swap_count=int(r.final_swap_count),
            unjoined_at_end_count=int(r.unjoined_at_end_count),
            topology_attempts=int(r.topology_attempts),
            topology_accepts=int(r.topology_accepts),
        )

    def run_trajectories(self, n_trajectories: int, decorrelation_steps: int) -> WorkRunResult:
        r = self._cpp_engine.run_trajectories(int(n_trajectories), int(decorrelation_steps))
        return WorkRunResult(
            delta_s2=float(r.delta_s2),
            mean_exp_minus_work=float(r.mean_exp_minus_work),
            work_mean=float(r.work_mean),
            work_var=float(r.work_var),
            trajectory_count=int(r.trajectory_count),
            total_topology_attempts=int(r.total_topology_attempts),
            total_topology_accepts=int(r.total_topology_accepts),
            total_unjoined_at_end=int(r.total_unjoined_at_end),
            work_samples=np.asarray(r.work_samples, dtype=np.float64),
            final_swap_counts=np.asarray(r.final_swap_counts, dtype=np.int32),
            unjoined_counts_per_traj=np.asarray(r.unjoined_counts_per_traj, dtype=np.int32),
            topology_attempts_per_traj=np.asarray(r.topology_attempts_per_traj, dtype=np.int64),
            topology_accepts_per_traj=np.asarray(r.topology_accepts_per_traj, dtype=np.int64),
        )

    # ── Accessors ──────────────────────────────────────────────────────────

    @property
    def A_start_mask(self) -> np.ndarray:
        return np.asarray(self._cpp_engine.A_start_mask, dtype=np.uint8)

    @property
    def A_end_mask(self) -> np.ndarray:
        return np.asarray(self._cpp_engine.A_end_mask, dtype=np.uint8)

    @property
    def D_mask(self) -> np.ndarray:
        return np.asarray(self._cpp_engine.D_mask, dtype=np.uint8)

    @property
    def B_mask(self) -> np.ndarray:
        return np.asarray(self._cpp_engine.B_mask, dtype=np.uint8)

    @property
    def D_size(self) -> int:
        return int(self._cpp_engine.D_size)

    @property
    def B_size(self) -> int:
        return int(self._cpp_engine.B_size)

    @property
    def lambda_schedule(self) -> np.ndarray:
        return np.asarray(self._cpp_engine.lambda_schedule, dtype=np.float64)
