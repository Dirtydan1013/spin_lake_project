"""Batched two-replica CUDA nonequilibrium-work engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import qaqmc_cpp
except (ImportError, OSError):
    qaqmc_cpp = None

try:
    import qaqmc_cuda
except (ImportError, OSError):
    qaqmc_cuda = None

from src.engines.qaqmc_batch_cuda import independent_chain_seeds
from src.engines.qaqmc_cuda import (
    _CLUSTER_STREAM,
    _DIAGONAL_STREAM,
    _U64_MASK,
    cuda_available,
)
from src.engines.qaqmc_renyi_work import WorkRunResult
from src.engines.qaqmc_renyi_work_cuda import _TOPOLOGY_STREAM


@dataclass
class CudaRenyiBatchBackend:
    engine: Any
    seeds: np.ndarray
    sweep_ids: np.ndarray
    topology_ids: np.ndarray

    @classmethod
    def from_cpu_model(
        cls, cpu_model: Any, *, batch_size: int, device: int = 0, seed: int = 0
    ) -> "CudaRenyiBatchBackend":
        if not cuda_available():
            raise RuntimeError("qaqmc_cuda is unavailable or no GPU is visible")
        batch_size = int(batch_size)
        length = 2 * int(cpu_model.M)
        types = np.ones((batch_size, 2, length), dtype=np.int32)
        sites = np.zeros_like(types)
        data = cpu_model.export_cuda_diagonal_data()
        engine = qaqmc_cuda.BatchedRenyiEngine(
            batch_size=batch_size,
            n_sites=int(cpu_model.N),
            half_length=int(cpu_model.M),
            delta_min=float(cpu_model.delta_min),
            delta_max=float(cpu_model.delta_max),
            epsilon=float(cpu_model.epsilon),
            bond_sites=data["bond_sites"],
            bond_vij=data["bond_vij"],
            inv_coord=data["inv_coord"],
            alias_prob=data["alias_prob"],
            alias_index=data["alias_index"],
            alias_loc_kind=data["alias_loc_kind"],
            bond_rmax=data["bond_rmax"],
            op_types=types,
            op_sites=sites,
            device=int(device),
        )
        return cls(
            engine=engine,
            seeds=independent_chain_seeds(seed, batch_size),
            sweep_ids=np.zeros(batch_size, dtype=np.uint64),
            topology_ids=np.zeros(batch_size, dtype=np.uint64),
        )

    @property
    def batch_size(self) -> int:
        return int(self.engine.batch_size)

    @property
    def N(self) -> int:
        return int(self.engine.n_sites)

    @property
    def M_total(self) -> int:
        return int(self.engine.length)

    @property
    def device_bytes(self) -> int:
        return int(self.engine.device_bytes)

    @property
    def shared_model_bytes(self) -> int:
        return int(self.engine.shared_model_bytes)

    def set_cut(self, cut: int) -> None:
        self.engine.set_cut(int(cut))

    def set_masks(self, masks: np.ndarray) -> None:
        values = np.ascontiguousarray(masks, dtype=np.uint8)
        expected = (self.batch_size, self.N)
        if values.shape != expected:
            raise ValueError(f"masks must have shape {expected}")
        self.engine.set_masks(values)

    def get_masks(self) -> np.ndarray:
        return np.asarray(self.engine.get_masks(), dtype=np.uint8)

    def mc_step(self) -> list[dict[str, Any]]:
        sweeps = self.sweep_ids.copy()
        diagonal = self.engine.diagonal_update(
            self.seeds ^ np.uint64(_DIAGONAL_STREAM), sweeps
        )
        for chain, stats in enumerate(diagonal):
            if stats["failed_slots"]:
                raise RuntimeError(
                    f"CUDA Renyi batch chain {chain} proposal limit reached: {stats}"
                )
        cluster = self.engine.cluster_update(
            self.seeds ^ np.uint64(_CLUSTER_STREAM), sweeps
        )
        self.sweep_ids += np.uint64(1)
        return [
            {"sweep_id": int(sweeps[k]),
             "diagonal": dict(diagonal[k]), "cluster": dict(cluster[k])}
            for k in range(self.batch_size)
        ]

    def run_steps(self, count: int) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        for _ in range(int(count)):
            self.mc_step()

    def topology_sweep(
        self, topology_sites: np.ndarray, lambda_: float
    ) -> list[dict[str, int | float]]:
        sites = np.ascontiguousarray(topology_sites, dtype=np.int32)
        ids = self.topology_ids.copy()
        stats = self.engine.topology_sweep(
            sites, float(lambda_), self.seeds ^ np.uint64(_TOPOLOGY_STREAM), ids
        )
        self.topology_ids += np.uint64(1)
        return [dict(row) for row in stats]

    def get_operator_strings(self) -> tuple[np.ndarray, np.ndarray]:
        types, sites = self.engine.get_operator_strings()
        return np.asarray(types, dtype=np.int32), np.asarray(sites, dtype=np.int32)

    def set_operator_strings(self, types: np.ndarray, sites: np.ndarray) -> None:
        expected = (self.batch_size, 2, self.M_total)
        types = np.ascontiguousarray(types, dtype=np.int32)
        sites = np.ascontiguousarray(sites, dtype=np.int32)
        if types.shape != expected or sites.shape != expected:
            raise ValueError(f"operator strings must have shape {expected}")
        self.engine.set_operator_strings(types, sites)

    def save_checkpoint(self) -> None:
        self.engine.save_checkpoint()

    def restore_checkpoint(self) -> None:
        self.engine.restore_checkpoint()

    @property
    def has_checkpoint(self) -> bool:
        return bool(self.engine.has_checkpoint)


class QAQMCRenyiWorkRydbergCUDABatch:
    """Run B independent Rényi-work trajectories concurrently on one GPU."""

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
        self._backend = CudaRenyiBatchBackend.from_cpu_model(
            cpu, batch_size=int(batch_size), device=int(device), seed=int(seed)
        )
        self.N = int(N)
        self.M = int(M)
        self.M_total = 2 * int(M)
        self.batch_size = int(batch_size)
        self.device = int(device)
        self._A_start = np.zeros(self.N, dtype=np.uint8)
        self._A_end = np.zeros(self.N, dtype=np.uint8)
        self._D_sites = np.empty(0, dtype=np.int32)
        self._B_size = np.zeros(self.batch_size, dtype=np.int32)
        self._schedule: np.ndarray | None = None
        self._n_topology = 1
        self._n_qaqmc = 1
        if verbose:
            info = qaqmc_cuda.device_info()[self.device]
            print(
                f"[QAQMC-RENYI-CUDA-BATCH] device={device} {info['name']} "
                f"B={batch_size} N={N} M={M} "
                f"resident={self.device_bytes / 2**20:.1f} MiB"
            )

    @property
    def device_bytes(self) -> int:
        return self._backend.device_bytes

    @property
    def shared_model_bytes(self) -> int:
        return self._backend.shared_model_bytes

    def _repeated_mask(self, mask: np.ndarray) -> np.ndarray:
        return np.repeat(mask[None, :], self.batch_size, axis=0)

    def _vacuum(self) -> None:
        types = np.ones((self.batch_size, 2, self.M_total), dtype=np.int32)
        sites = np.zeros_like(types)
        self._backend.set_operator_strings(types, sites)

    def set_region(self, mask: np.ndarray) -> None:
        self.set_region_pair(np.zeros(self.N, dtype=np.uint8), mask)

    def set_region_pair(self, start: np.ndarray, end: np.ndarray) -> None:
        start = np.ascontiguousarray(start, dtype=np.uint8)
        end = np.ascontiguousarray(end, dtype=np.uint8)
        if start.shape != (self.N,) or end.shape != (self.N,):
            raise ValueError(f"masks must have shape ({self.N},)")
        if np.any((start > 1) | (end > 1)) or np.any(start & ~end):
            raise ValueError("masks must be binary and start must be a subset of end")
        self._A_start = start.copy()
        self._A_end = end.copy()
        self._D_sites = np.flatnonzero(end & (1 - start)).astype(np.int32)
        self._B_size.fill(0)
        self._backend.set_masks(self._repeated_mask(start))
        self._vacuum()

    def set_cut(self, cut: int) -> None:
        self._backend.set_cut(int(cut))
        self._backend.set_masks(self._repeated_mask(self._A_start))
        self._B_size.fill(0)
        self._vacuum()

    def set_lambda_schedule(self, lambdas: np.ndarray) -> None:
        values = np.ascontiguousarray(lambdas, dtype=np.float64)
        if (values.ndim != 1 or len(values) < 2 or values[0] != 0.0
                or values[-1] != 1.0 or np.any(np.diff(values) < 0)):
            raise ValueError("lambda schedule must increase from exactly zero to one")
        self._schedule = values

    def set_sweeps_per_lambda(
        self, n_topology_sweeps: int = 1, n_qaqmc_sweeps: int = 1
    ) -> None:
        if n_topology_sweeps < 0 or n_qaqmc_sweeps < 0:
            raise ValueError("sweep counts must be non-negative")
        self._n_topology = int(n_topology_sweeps)
        self._n_qaqmc = int(n_qaqmc_sweeps)

    def _reset_start(self) -> None:
        self._B_size.fill(0)
        self._backend.set_masks(self._repeated_mask(self._A_start))
        if self._backend.has_checkpoint:
            self._backend.restore_checkpoint()
        else:
            self._vacuum()

    def thermalize(self, n_steps: int) -> None:
        self._reset_start()
        self._backend.run_steps(int(n_steps))
        self._backend.save_checkpoint()

    def _run_batch_trajectory(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._schedule is None:
            raise RuntimeError("call set_lambda_schedule first")
        work = np.zeros(self.batch_size, dtype=np.float64)
        final = np.zeros(self.batch_size, dtype=np.int32)
        unjoined = np.zeros(self.batch_size, dtype=np.int32)
        attempts = np.zeros(self.batch_size, dtype=np.int64)
        accepts = np.zeros(self.batch_size, dtype=np.int64)
        d_size = len(self._D_sites)
        for old, new in zip(self._schedule[:-1], self._schedule[1:], strict=True):
            joined = self._B_size
            remaining = d_size - joined
            if np.any(joined):
                work -= joined * (math.log(float(new)) - math.log(float(old)))
            if new >= 1.0:
                unjoined += remaining
            elif np.any(remaining):
                work -= remaining * (
                    math.log1p(-float(new)) - math.log1p(-float(old))
                )
            if 0.0 < new < 1.0:
                for _ in range(self._n_topology):
                    stats = self._backend.topology_sweep(self._D_sites, float(new))
                    for chain, row in enumerate(stats):
                        attempts[chain] += int(row["attempts"])
                        accepts[chain] += int(row["accepts"])
                        self._B_size[chain] = int(row["active_count"])
            else:
                attempts += self._n_topology * d_size
            for _ in range(self._n_qaqmc):
                self._backend.mc_step()
        final[:] = self._B_size
        return work, final, unjoined, attempts, accepts

    def run_trajectories(
        self, n_trajectories: int, decorrelation_steps: int
    ) -> WorkRunResult:
        count = int(n_trajectories)
        if count < 0 or decorrelation_steps < 0:
            raise ValueError("trajectory/decorrelation counts must be non-negative")
        work = np.empty(count, dtype=np.float64)
        final = np.empty(count, dtype=np.int32)
        unjoined = np.empty(count, dtype=np.int32)
        attempts = np.empty(count, dtype=np.int64)
        accepts = np.empty(count, dtype=np.int64)
        for begin in range(0, count, self.batch_size):
            active = min(self.batch_size, count - begin)
            self._reset_start()
            self._backend.run_steps(int(decorrelation_steps))
            self._backend.save_checkpoint()
            rows = self._run_batch_trajectory()
            stop = begin + active
            work[begin:stop] = rows[0][:active]
            final[begin:stop] = rows[1][:active]
            unjoined[begin:stop] = rows[2][:active]
            attempts[begin:stop] = rows[3][:active]
            accepts[begin:stop] = rows[4][:active]
        if count:
            values = -work
            maximum = float(values.max())
            log_mean = maximum + math.log(float(np.exp(values - maximum).mean()))
            mean_exp = math.exp(log_mean)
            delta_s2 = -log_mean
            work_mean = float(work.mean())
            work_var = float(work.var(ddof=1)) if count > 1 else 0.0
        else:
            mean_exp, delta_s2, work_mean, work_var = 1.0, 0.0, 0.0, 0.0
        return WorkRunResult(
            delta_s2, mean_exp, work_mean, work_var, count,
            int(attempts.sum()), int(accepts.sum()), int(unjoined.sum()),
            work, final, unjoined, attempts, accepts,
        )
