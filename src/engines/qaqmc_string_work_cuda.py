"""Device-resident CUDA backend for QAQMC off-diagonal string work.

The Jarzynski orchestration and result types are shared with the trusted CPU
wrapper.  Only the transition backend changes: operator strings, seam-aware
worldline scans, cluster events and half-line topology moves stay on one GPU.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import qaqmc_cpp
except (ImportError, OSError):
    qaqmc_cpp = None

from src.engines.qaqmc_cuda import CudaDiagonalBackend, cuda_available
from src.engines.qaqmc_string_work import (
    QAQMCStringWorkRydberg,
    StringWorkRunResult,
)


class QAQMCStringWorkRydbergCUDA(QAQMCStringWorkRydberg):
    """CUDA implementation of :class:`QAQMCStringWorkRydberg`.

    A short-lived CPU engine constructs the geometry and grouped alias
    envelopes.  The Markov-chain state is then copied once to CUDA and the CPU
    object is released; trajectory steps do not transfer operator strings.
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
        box_vectors: np.ndarray | None = None,
        device: int = 0,
        verbose: bool = True,
    ) -> None:
        if qaqmc_cpp is None:
            raise RuntimeError("qaqmc_cpp is required to construct CUDA model tables")
        if not cuda_available():
            raise RuntimeError("qaqmc_cuda is unavailable or no GPU is visible")
        positions = (
            np.arange(N, dtype=np.float64).reshape(-1, 1)
            if pos is None
            else np.ascontiguousarray(pos, dtype=np.float64)
        )
        if positions.ndim != 2 or positions.shape[0] != N:
            raise ValueError("pos must have shape (N, dimension)")
        box = (
            None
            if box_vectors is None
            else np.ascontiguousarray(box_vectors, dtype=np.float64)
        )
        cpu = qaqmc_cpp.QAQMCEngine(
            N,
            Omega,
            delta_min,
            delta_max,
            Rb,
            M,
            epsilon,
            seed,
            positions,
            neighbor_cutoff=(-1 if neighbor_cutoff is None else neighbor_cutoff),
            delta_groups=int(delta_groups),
            box_vectors=box,
        )
        self._eng = CudaDiagonalBackend.from_cpu_engine(
            cpu, device=int(device), seed=int(seed)
        )
        self.N = int(N)
        self.M = int(M)
        self.M_total = 2 * int(M)
        self._length = 0
        self._lambda_schedule: np.ndarray | None = None
        self._checkpoint_mask: int | None = None
        self.device = int(device)
        if verbose:
            import qaqmc_cuda

            info = qaqmc_cuda.device_info()[self.device]
            print(
                f"[QAQMC-STRING-CUDA] device={self.device} {info['name']} "
                f"N={N} M={M} resident={self._eng.device_bytes / 2**20:.1f} MiB"
            )

    def set_string_sites(self, sites, m_star: int | None = None) -> None:
        super().set_string_sites(sites, m_star)
        self._checkpoint_mask = None

    def thermalize(self, n_steps: int, direction: str = "forward") -> None:
        """Equilibrate and seed a device-to-device start-sector checkpoint."""
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if direction == "forward":
            reset_mask = 0
        elif direction == "reverse":
            reset_mask = self._full_mask()
        else:
            raise ValueError(
                f"direction must be 'forward' or 'reverse', got {direction!r}"
            )
        if self._eng.has_checkpoint and self._checkpoint_mask == reset_mask:
            self._eng.restore_device_checkpoint()
        else:
            self._eng.set_seam_mask_consistent(reset_mask)
        self._eng.run_steps(int(n_steps))
        self._eng.save_device_checkpoint()
        self._checkpoint_mask = reset_mask

    def run_trajectories(
        self,
        n_trajectories: int,
        decorrelation_steps: int = 100,
        n_topology_sweeps_per_lambda: int = 1,
        n_qaqmc_sweeps_per_lambda: int = 1,
        direction: str = "forward",
    ) -> StringWorkRunResult:
        """Run trajectories from a rolling D2D start-sector checkpoint.

        Restoring the checkpoint replaces the CPU wrapper's per-trajectory
        closure repair.  This changes neither the Markov transition nor RNG
        counters; it only avoids up to ``|C|`` full operator-string scans.
        """
        if n_trajectories < 0 or decorrelation_steps < 0:
            raise ValueError("trajectory and decorrelation counts must be non-negative")
        if direction not in ("forward", "reverse"):
            raise ValueError(
                f"direction must be 'forward' or 'reverse', got {direction!r}"
            )
        reset_mask = 0 if direction == "forward" else self._full_mask()

        log_j_samples = np.empty(int(n_trajectories), dtype=np.float64)
        zero_count = 0
        for trajectory in range(int(n_trajectories)):
            if self._eng.has_checkpoint and self._checkpoint_mask == reset_mask:
                self._eng.restore_device_checkpoint()
            else:
                self._eng.set_seam_mask_consistent(reset_mask)
            self._eng.run_steps(int(decorrelation_steps))
            self._eng.save_device_checkpoint()
            self._checkpoint_mask = reset_mask
            result = self.run_trajectory(
                n_topology_sweeps_per_lambda,
                n_qaqmc_sweeps_per_lambda,
                direction=direction,
            )
            log_j_samples[trajectory] = result.log_j
            zero_count += int(result.zero_weight)

        finite = np.isfinite(log_j_samples)
        if not np.any(finite):
            return StringWorkRunResult(
                o_c=0.0,
                log_o_c=-math.inf,
                n_trajectories=int(n_trajectories),
                n_eff=0.0,
                p_max=0.0,
                zero_weight_fraction=1.0,
                log_j_samples=log_j_samples,
            )

        max_log = float(log_j_samples[finite].max())
        weights = np.zeros(int(n_trajectories), dtype=np.float64)
        weights[finite] = np.exp(log_j_samples[finite] - max_log)
        sum_w = float(weights.sum())
        log_mean_j = max_log + math.log(sum_w / int(n_trajectories))
        log_o_c = log_mean_j if direction == "forward" else -log_mean_j
        probabilities = weights / sum_w
        return StringWorkRunResult(
            o_c=math.exp(log_o_c),
            log_o_c=log_o_c,
            n_trajectories=int(n_trajectories),
            n_eff=1.0 / float(np.sum(probabilities**2)),
            p_max=float(probabilities.max()),
            zero_weight_fraction=zero_count / int(n_trajectories),
            log_j_samples=log_j_samples,
        )

    @property
    def device_bytes(self) -> int:
        return self._eng.device_bytes
