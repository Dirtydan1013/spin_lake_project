"""Batched off-diagonal string-work trajectories on one CUDA device."""

from __future__ import annotations

import math

import numpy as np

try:
    import qaqmc_cpp
except (ImportError, OSError):
    qaqmc_cpp = None

try:
    import qaqmc_cuda
except (ImportError, OSError):
    qaqmc_cuda = None

from src.engines.qaqmc_batch_cuda import CudaDiagonalBatchBackend
from src.engines.qaqmc_cuda import cuda_available
from src.engines.qaqmc_string_work import StringWorkRunResult, _log_g


class QAQMCStringWorkRydbergCUDABatch:
    """Run B independent string-work trajectories concurrently on one GPU."""

    def __init__(
        self,
        N: int,
        M: int,
        *,
        batch_size: int,
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
        if qaqmc_cpp is None or not cuda_available():
            raise RuntimeError("qaqmc_cpp plus a usable qaqmc_cuda build are required")
        positions = (
            np.arange(N, dtype=np.float64).reshape(-1, 1)
            if pos is None else np.ascontiguousarray(pos, dtype=np.float64)
        )
        box = (None if box_vectors is None
               else np.ascontiguousarray(box_vectors, dtype=np.float64))
        cpu = qaqmc_cpp.QAQMCEngine(
            N, Omega, delta_min, delta_max, Rb, M, epsilon, seed, positions,
            neighbor_cutoff=(-1 if neighbor_cutoff is None else neighbor_cutoff),
            delta_groups=int(delta_groups), box_vectors=box,
        )
        self._backend = CudaDiagonalBatchBackend.from_cpu_engine(
            cpu, batch_size=int(batch_size), device=int(device), seed=int(seed)
        )
        self.N = int(N)
        self.M = int(M)
        self.M_total = 2 * int(M)
        self.batch_size = int(batch_size)
        self.device = int(device)
        self._length = 0
        self._schedule: np.ndarray | None = None
        self._checkpoint_mask: int | None = None
        if verbose:
            info = qaqmc_cuda.device_info()[self.device]
            print(
                f"[QAQMC-STRING-CUDA-BATCH] device={device} {info['name']} "
                f"B={batch_size} N={N} M={M} "
                f"resident={self.device_bytes / 2**20:.1f} MiB"
            )

    @property
    def device_bytes(self) -> int:
        return self._backend.device_bytes

    @property
    def shared_model_bytes(self) -> int:
        return self._backend.shared_model_bytes

    def set_string_sites(self, sites, m_star: int | None = None) -> None:
        values = np.ascontiguousarray(sites, dtype=np.int32)
        self._backend.set_string_sites(values, m_star)
        self._length = len(values)
        self._checkpoint_mask = None

    def set_lambda_schedule(self, lambdas: np.ndarray) -> None:
        values = np.ascontiguousarray(lambdas, dtype=np.float64)
        if (values.ndim != 1 or len(values) < 2 or values[0] != 0.0
                or values[-1] != 1.0 or np.any(np.diff(values) < 0)):
            raise ValueError("lambda schedule must increase from exactly zero to one")
        self._schedule = values

    def _full_mask(self) -> int:
        return (1 << self._length) - 1

    def _reset_masks(self, direction: str) -> np.ndarray:
        if direction == "forward":
            value = 0
        elif direction == "reverse":
            value = self._full_mask()
        else:
            raise ValueError("direction must be 'forward' or 'reverse'")
        masks = np.full(self.batch_size, value, dtype=np.uint64)
        self._backend.set_seam_masks_consistent(masks)
        return masks

    def thermalize(self, n_steps: int, direction: str = "forward") -> None:
        masks = self._reset_masks(direction)
        self._backend.run_steps(int(n_steps))
        self._backend.save_device_checkpoint()
        self._checkpoint_mask = int(masks[0])

    def _run_batch_trajectory(
        self,
        n_topology_sweeps: int,
        n_qaqmc_sweeps: int,
        direction: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._schedule is None:
            raise RuntimeError("call set_lambda_schedule first")
        schedule = self._schedule if direction == "forward" else self._schedule[::-1]
        log_j = np.zeros(self.batch_size, dtype=np.float64)
        zero = np.zeros(self.batch_size, dtype=bool)
        steps = len(schedule) - 1
        for index in range(steps):
            now = float(schedule[index])
            following = float(schedule[index + 1])
            masks = self._backend.seam_masks
            active = np.fromiter(
                (int(value).bit_count() for value in masks),
                dtype=np.int32,
                count=self.batch_size,
            )
            for chain in range(self.batch_size):
                if zero[chain]:
                    continue
                new_log = _log_g(following, int(active[chain]), self._length)
                old_log = _log_g(now, int(active[chain]), self._length)
                if not math.isfinite(new_log):
                    log_j[chain] = -math.inf
                    zero[chain] = True
                else:
                    log_j[chain] += new_log - old_log
            if index + 1 < steps:
                for _ in range(int(n_topology_sweeps)):
                    self._backend.topology_sweep(following)
                for _ in range(int(n_qaqmc_sweeps)):
                    self._backend.mc_step()
        return log_j, zero

    def run_trajectories(
        self,
        n_trajectories: int,
        decorrelation_steps: int = 100,
        n_topology_sweeps_per_lambda: int = 1,
        n_qaqmc_sweeps_per_lambda: int = 1,
        direction: str = "forward",
    ) -> StringWorkRunResult:
        count = int(n_trajectories)
        if count < 0 or decorrelation_steps < 0:
            raise ValueError("trajectory/decorrelation counts must be non-negative")
        samples = np.empty(count, dtype=np.float64)
        zero_count = 0
        reset = 0 if direction == "forward" else self._full_mask()
        for begin in range(0, count, self.batch_size):
            active = min(self.batch_size, count - begin)
            if self._backend.has_checkpoint and self._checkpoint_mask == reset:
                self._backend.restore_device_checkpoint()
            else:
                self._backend.set_seam_masks_consistent(
                    np.full(self.batch_size, reset, dtype=np.uint64)
                )
            self._backend.run_steps(int(decorrelation_steps))
            self._backend.save_device_checkpoint()
            self._checkpoint_mask = reset
            values, zero = self._run_batch_trajectory(
                n_topology_sweeps_per_lambda,
                n_qaqmc_sweeps_per_lambda,
                direction,
            )
            stop = begin + active
            samples[begin:stop] = values[:active]
            zero_count += int(zero[:active].sum())
        finite = np.isfinite(samples)
        if not np.any(finite):
            return StringWorkRunResult(
                0.0, -math.inf, count, 0.0, 0.0, 1.0, samples
            )
        maximum = float(samples[finite].max())
        weights = np.zeros(count, dtype=np.float64)
        weights[finite] = np.exp(samples[finite] - maximum)
        sum_weights = float(weights.sum())
        log_mean = maximum + math.log(sum_weights / count)
        log_o_c = log_mean if direction == "forward" else -log_mean
        probabilities = weights / sum_weights
        return StringWorkRunResult(
            o_c=math.exp(log_o_c),
            log_o_c=log_o_c,
            n_trajectories=count,
            n_eff=1.0 / float(np.sum(probabilities**2)),
            p_max=float(probabilities.max()),
            zero_weight_fraction=zero_count / count,
            log_j_samples=samples,
        )
