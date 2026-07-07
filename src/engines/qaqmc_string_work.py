"""
Python wrapper for the single-replica QAQMC off-diagonal string estimator.

Implements the nonequilibrium-work / Jarzynski method from
document/QAQMC_LineCluster_Jarzynski_String_Implementation.md: measures

    O_C = Z_C / Z_empty = <psi_L| P_L X_C P_R |psi_R> / <psi_L| P_L P_R |psi_R>

for X_C = prod_{i in C} sigma_i^x inserted at imaginary-time slice m_star, via
an interpolating ensemble g_lambda(B) = lambda^|B| (1-lambda)^(L_C-|B|) over
subsets B of C. Topology sampling uses the engine's half-line move
(`QAQMCEngine.attempt_string_toggle`/`topology_sweep`, Phase B); ordinary
relaxation uses the seam-aware `diagonal_update`/`cluster_update` (Phase A).

This module is Python-level orchestration around C++ primitives -- the hot
per-step work (topology_sweep, mc_step) runs in C++, matching the design of
`QAQMCRenyiWorkRydberg` (src/engines/qaqmc_renyi_work.py), the analogous
two-replica Renyi-entropy estimator this mirrors.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np

try:
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _candidate in (_repo_root, os.path.join(_repo_root, "build")):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
    import qaqmc_cpp
except ImportError as exc:
    raise ImportError(
        "qaqmc_cpp with QAQMCEngine string-seam support is required for "
        "src.engines.qaqmc_string_work"
    ) from exc


def _log_g(lam: float, active: int, length: int) -> float:
    """Endpoint-safe log g_lambda(B) = |B| log(lambda) + (L-|B|) log(1-lambda)."""
    if lam <= 0.0:
        return 0.0 if active == 0 else -math.inf
    if lam >= 1.0:
        return 0.0 if active == length else -math.inf
    return active * math.log(lam) + (length - active) * math.log1p(-lam)


@dataclass
class StringWorkTrajectoryResult:
    log_j: float
    zero_weight: bool
    final_active_count: int


@dataclass
class StringWorkRunResult:
    o_c: float                  # estimate of Z_C / Z_empty
    log_o_c: float
    n_trajectories: int
    n_eff: float                # Jarzynski effective sample size
    p_max: float                # largest normalized trajectory weight
    zero_weight_fraction: float # fraction of trajectories that ended with J=0
    log_j_samples: np.ndarray


class QAQMCStringWorkRydberg:
    """Driver around `qaqmc_cpp.QAQMCEngine`'s Phase A/B string-seam primitives.

    Typical usage::

        eng = QAQMCStringWorkRydberg(N=5, M=16, Omega=1.0, Rb=1.2,
                                     delta_min=0.0, delta_max=1.5, pos=pos)
        eng.set_string_sites([2])                       # L_C = 1, m_star = M
        eng.set_lambda_schedule(cosine_schedule(200))
        eng.thermalize(2000)
        res = eng.run_trajectories(n_trajectories=3000, decorrelation_steps=100)
        print(res.o_c, res.n_eff)
    """

    def __init__(self, N: int, M: int, Omega: float = 1.0, Rb: float = 1.2,
                 delta_min: float = 0.0, delta_max: float = 1.0,
                 epsilon: float = 0.01, seed: int = 42,
                 pos: np.ndarray | None = None,
                 neighbor_cutoff: int | None = None,
                 delta_groups: int = 600,
                 box_vectors: np.ndarray | None = None):
        if pos is None:
            pos = np.arange(N).reshape(-1, 1).astype(np.float64)
        pos_arr = np.ascontiguousarray(pos, dtype=np.float64)
        nc = neighbor_cutoff if neighbor_cutoff is not None else -1
        box = (np.ascontiguousarray(box_vectors, dtype=np.float64)
               if box_vectors is not None else None)
        self._eng = qaqmc_cpp.QAQMCEngine(
            N, Omega, delta_min, delta_max, Rb, M, epsilon, seed, pos_arr,
            neighbor_cutoff=nc, delta_groups=int(delta_groups), box_vectors=box,
        )
        self.N = N
        self.M = M
        self.M_total = 2 * M
        self._length = 0
        self._lambda_schedule: np.ndarray | None = None

    # ── Configuration ────────────────────────────────────────────────────

    def set_string_sites(self, sites, m_star: int | None = None) -> None:
        sites = list(sites)
        if m_star is None:
            m_star = self.M
        self._eng.set_string_sites(sites, int(m_star))
        self._length = len(sites)

    def set_lambda_schedule(self, lambdas: np.ndarray) -> None:
        lambdas = np.asarray(lambdas, dtype=np.float64)
        if lambdas.ndim != 1 or lambdas.size < 2:
            raise ValueError("lambda_schedule must be a 1D array of length >= 2")
        if lambdas[0] != 0.0 or lambdas[-1] != 1.0:
            raise ValueError("lambda_schedule must run from 0.0 to 1.0")
        if np.any(np.diff(lambdas) < 0.0):
            raise ValueError("lambda_schedule must be monotonically non-decreasing")
        self._lambda_schedule = lambdas

    def _full_mask(self) -> int:
        return (1 << self._length) - 1

    def thermalize(self, n_steps: int, direction: str = "forward") -> None:
        """Equilibrate at the trajectory's starting sector: B=empty for a
        forward (lambda: 0->1) run, B=C (all bits active) for a reverse
        (lambda: 1->0) run -- document section 29/33."""
        if direction == "forward":
            self._eng.set_seam_mask(0)
        elif direction == "reverse":
            self._eng.set_seam_mask(self._full_mask())
        else:
            raise ValueError(f"direction must be 'forward' or 'reverse', got {direction!r}")
        for _ in range(n_steps):
            self._eng.mc_step()

    # ── Trajectories ─────────────────────────────────────────────────────

    def run_trajectory(self, n_topology_sweeps_per_lambda: int = 1,
                       n_qaqmc_sweeps_per_lambda: int = 1,
                       direction: str = "forward") -> StringWorkTrajectoryResult:
        if self._lambda_schedule is None:
            raise RuntimeError("call set_lambda_schedule() first")

        if direction == "forward":
            schedule = self._lambda_schedule
            if self._eng.seam_mask != 0:
                raise RuntimeError("forward trajectory must start in the empty sector (seam_mask=0)")
        elif direction == "reverse":
            schedule = self._lambda_schedule[::-1]
            if self._eng.seam_mask != self._full_mask():
                raise RuntimeError("reverse trajectory must start in the full sector (seam_mask=all-ones)")
        else:
            raise ValueError(f"direction must be 'forward' or 'reverse', got {direction!r}")

        length = self._length
        log_j = 0.0
        K = len(schedule) - 1
        for m in range(K):
            lam_now = schedule[m]
            lam_next = schedule[m + 1]
            active = bin(self._eng.seam_mask).count("1")
            new_log_g = _log_g(lam_next, active, length)
            old_log_g = _log_g(lam_now, active, length)
            if not math.isfinite(new_log_g):
                # B_m != C at lambda -> 1 (or B_m != empty at lambda -> 0):
                # exact zero, per document section 19 -- not a "multiply by
                # one" prescription, this is a different method than the
                # Renyi TEE work engine. Symmetric for the reverse direction
                # (schedule ends at 0, requiring B_m -> empty).
                return StringWorkTrajectoryResult(
                    log_j=-math.inf, zero_weight=True, final_active_count=active)
            log_j += new_log_g - old_log_g
            if m + 1 < K:
                for _ in range(n_topology_sweeps_per_lambda):
                    self._eng.topology_sweep(float(lam_next))
                for _ in range(n_qaqmc_sweeps_per_lambda):
                    self._eng.mc_step()

        final_active = bin(self._eng.seam_mask).count("1")
        return StringWorkTrajectoryResult(log_j=log_j, zero_weight=False,
                                          final_active_count=final_active)

    def run_trajectories(self, n_trajectories: int, decorrelation_steps: int = 100,
                         n_topology_sweeps_per_lambda: int = 1,
                         n_qaqmc_sweeps_per_lambda: int = 1,
                         direction: str = "forward") -> StringWorkRunResult:
        if direction not in ("forward", "reverse"):
            raise ValueError(f"direction must be 'forward' or 'reverse', got {direction!r}")
        reset_mask = 0 if direction == "forward" else self._full_mask()

        log_j_samples = np.empty(n_trajectories, dtype=np.float64)
        zero_count = 0
        for r in range(n_trajectories):
            # Trajectories always start at the sector matching lambda=0 (B=
            # empty, forward) or lambda=1 (B=C, reverse); resetting the mask
            # directly is safe here since site-operator weight (Omega/2)
            # doesn't depend on type 1 vs -1, so any leftover terminal-op
            # type from a previous trajectory is still a valid (positive-
            # weight) configuration.
            self._eng.set_seam_mask(reset_mask)
            for _ in range(decorrelation_steps):
                self._eng.mc_step()
            result = self.run_trajectory(n_topology_sweeps_per_lambda, n_qaqmc_sweeps_per_lambda,
                                         direction=direction)
            log_j_samples[r] = result.log_j
            if result.zero_weight:
                zero_count += 1

        finite = np.isfinite(log_j_samples)
        if not np.any(finite):
            return StringWorkRunResult(
                o_c=0.0, log_o_c=-math.inf, n_trajectories=n_trajectories,
                n_eff=0.0, p_max=0.0, zero_weight_fraction=1.0,
                log_j_samples=log_j_samples)

        max_log = log_j_samples[finite].max()
        weights = np.zeros(n_trajectories, dtype=np.float64)
        weights[finite] = np.exp(log_j_samples[finite] - max_log)
        sum_w = weights.sum()
        log_mean_j = max_log + math.log(sum_w / n_trajectories)

        # Forward: mean(J_fwd) = Z_C/Z_empty = O_C directly.
        # Reverse: mean(J_rev) = Z_empty/Z_C = 1/O_C (document section 33),
        # so invert to report both directions on the same O_C scale.
        if direction == "forward":
            log_o_c = log_mean_j
        else:
            log_o_c = -log_mean_j
        o_c = math.exp(log_o_c)

        p = weights / sum_w
        n_eff = 1.0 / float(np.sum(p ** 2))
        p_max = float(p.max())

        return StringWorkRunResult(
            o_c=o_c, log_o_c=log_o_c, n_trajectories=n_trajectories,
            n_eff=n_eff, p_max=p_max,
            zero_weight_fraction=zero_count / n_trajectories,
            log_j_samples=log_j_samples,
        )


def cosine_schedule(k_steps: int) -> np.ndarray:
    """lambda(t) = 0.5*(1 - cos(pi*t)), t = m/K -- slower near the endpoints
    than a linear schedule (document section 20)."""
    t = np.linspace(0.0, 1.0, k_steps + 1)
    return 0.5 * (1.0 - np.cos(np.pi * t))
