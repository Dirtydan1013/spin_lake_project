"""Device-resident CUDA QAQMC Rényi nonequilibrium-work engine."""

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

from src.engines.qaqmc_renyi_work import WorkRunResult, WorkTrajectoryResult


_U64_MASK = (1 << 64) - 1
_DIAGONAL_STREAM = 0xD1A60A1C5EED1234
_CLUSTER_STREAM = 0xC1057E2A5EED5678
_TOPOLOGY_STREAM = 0x52E4F19A5EED9ABC


@dataclass
class CudaRenyiBackend:
    engine: Any
    seed: int
    sweep_id: int = 0
    topology_id: int = 0

    @classmethod
    def from_cpu_model(
        cls, cpu_model: Any, *, device: int = 0, seed: int = 0
    ) -> "CudaRenyiBackend":
        data = cpu_model.export_cuda_diagonal_data()
        length = 2 * int(cpu_model.M)
        types = np.ones((2, length), dtype=np.int32)
        sites = np.zeros((2, length), dtype=np.int32)
        engine = qaqmc_cuda.RenyiEngine(
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
        return cls(engine=engine, seed=int(seed) & _U64_MASK)

    @property
    def N(self) -> int:
        return int(self.engine.n_sites)

    @property
    def M(self) -> int:
        return int(self.engine.half_length)

    @property
    def M_total(self) -> int:
        return int(self.engine.length)

    @property
    def device_bytes(self) -> int:
        return int(self.engine.device_bytes)

    def set_cut(self, cut: int) -> None:
        self.engine.set_cut(int(cut))

    def set_mask(self, mask: np.ndarray) -> None:
        self.engine.set_mask(np.ascontiguousarray(mask, dtype=np.uint8))

    def get_mask(self) -> np.ndarray:
        return np.asarray(self.engine.get_mask(), dtype=np.uint8)

    def mc_step(self) -> dict[str, Any]:
        sweep = int(self.sweep_id)
        diagonal = dict(
            self.engine.diagonal_update(
                seed=(self.seed ^ _DIAGONAL_STREAM) & _U64_MASK,
                sweep_id=sweep,
            )
        )
        if diagonal["failed_slots"]:
            raise RuntimeError(f"CUDA Renyi diagonal proposal limit reached: {diagonal}")
        cluster = dict(
            self.engine.cluster_update(
                seed=(self.seed ^ _CLUSTER_STREAM) & _U64_MASK,
                sweep_id=sweep,
            )
        )
        self.sweep_id += 1
        return {"sweep_id": sweep, "diagonal": diagonal, "cluster": cluster}

    def run_steps(self, count: int) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        for _ in range(count):
            self.mc_step()

    def topology_sweep(
        self, topology_sites: np.ndarray, lambda_: float
    ) -> dict[str, int | float]:
        topology = int(self.topology_id)
        result = dict(
            self.engine.topology_sweep(
                topology_sites=np.ascontiguousarray(topology_sites, dtype=np.int32),
                lambda_=float(lambda_),
                seed=(self.seed ^ _TOPOLOGY_STREAM) & _U64_MASK,
                sweep_id=topology,
            )
        )
        self.topology_id += 1
        return result

    def get_operator_strings(self) -> tuple[np.ndarray, np.ndarray]:
        types, sites = self.engine.get_operator_strings()
        return np.asarray(types, dtype=np.int32), np.asarray(sites, dtype=np.int32)

    def set_operator_strings(self, types: np.ndarray, sites: np.ndarray) -> None:
        expected = (2, self.M_total)
        types = np.ascontiguousarray(types, dtype=np.int32)
        sites = np.ascontiguousarray(sites, dtype=np.int32)
        if types.shape != expected or sites.shape != expected:
            raise ValueError(f"operator strings must have shape {expected}")
        self.engine.set_operator_strings(types, sites)

    def save_checkpoint(self) -> None:
        self.engine.save_checkpoint()

    def restore_checkpoint(self) -> None:
        self.engine.restore_checkpoint()


class QAQMCRenyiWorkRydbergCUDA:
    """CUDA counterpart of ``QAQMCRenyiWorkRydberg``.

    The outer Jarzynski protocol remains explicit Python for auditability;
    every O(M) transition, topology ratio and checkpoint copy executes on the
    GPU without an operator-string PCIe round trip.
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
        if qaqmc_cpp is None or qaqmc_cuda is None or not qaqmc_cuda.is_available():
            raise RuntimeError("qaqmc_cpp plus a usable qaqmc_cuda build are required")
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
        cpu_model = qaqmc_cpp.QAQMCEngine(
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
        self._backend = CudaRenyiBackend.from_cpu_model(
            cpu_model, device=int(device), seed=int(seed)
        )
        # Compatibility facade for the existing MPI warm-start path.
        self._cpp_engine = self
        self.N = int(N)
        self.M = int(M)
        self.M_total = 2 * int(M)
        self.pos = positions
        self._A_start = np.zeros(self.N, dtype=np.uint8)
        self._A_end = np.zeros(self.N, dtype=np.uint8)
        self._D_mask = np.zeros(self.N, dtype=np.uint8)
        self._D_sites = np.empty(0, dtype=np.int32)
        self._B_size = 0
        self._schedule: np.ndarray | None = None
        self._n_topology = 1
        self._n_qaqmc = 1
        self._checkpoint_valid = False
        if verbose:
            info = qaqmc_cuda.device_info()[int(device)]
            print(
                f"[QAQMC-RENYI-CUDA] device={device} {info['name']} N={N} M={M} "
                f"resident={self._backend.device_bytes / 2**20:.1f} MiB"
            )

    def _vacuum(self) -> None:
        types = np.ones((2, self.M_total), dtype=np.int32)
        sites = np.zeros((2, self.M_total), dtype=np.int32)
        self._backend.set_operator_strings(types, sites)

    def set_region(self, mask: np.ndarray) -> None:
        self.set_region_pair(np.zeros(self.N, dtype=np.uint8), mask)

    def set_region_pair(self, start: np.ndarray, end: np.ndarray) -> None:
        start = np.ascontiguousarray(start, dtype=np.uint8)
        end = np.ascontiguousarray(end, dtype=np.uint8)
        if start.shape != (self.N,) or end.shape != (self.N,):
            raise ValueError(f"masks must have shape ({self.N},)")
        if np.any((start > 1) | (end > 1)) or np.any(start & ~end):
            raise ValueError("masks must be binary and A_start must be a subset of A_end")
        self._A_start = start.copy()
        self._A_end = end.copy()
        self._D_mask = (end & (1 - start)).astype(np.uint8)
        self._D_sites = np.flatnonzero(self._D_mask).astype(np.int32)
        self._B_size = 0
        self._backend.set_mask(self._A_start)
        self._vacuum()
        self._checkpoint_valid = False

    def set_cut(self, cut: int) -> None:
        self._backend.set_cut(int(cut))
        self._B_size = 0
        self._backend.set_mask(self._A_start)
        self._vacuum()
        self._checkpoint_valid = False

    def get_cut(self) -> int:
        return int(self._backend.engine.cut)

    def set_lambda_schedule(self, lambdas: np.ndarray) -> None:
        values = np.ascontiguousarray(lambdas, dtype=np.float64)
        if values.ndim != 1 or len(values) < 2:
            raise ValueError("lambda schedule must be a one-dimensional array")
        if values[0] != 0.0 or values[-1] != 1.0 or np.any(np.diff(values) < 0):
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
        self._B_size = 0
        self._backend.set_mask(self._A_start)
        if self._checkpoint_valid:
            self._backend.restore_checkpoint()
        else:
            self._vacuum()

    def thermalize(self, n_steps: int) -> None:
        self._reset_start()
        self._backend.run_steps(int(n_steps))
        self._backend.save_checkpoint()
        self._checkpoint_valid = True

    def export_start_config(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Download the current A_start checkpoint for warm-start files."""
        self._backend.set_mask(self._A_start)
        if self._checkpoint_valid:
            self._backend.restore_checkpoint()
        types, sites = self._backend.get_operator_strings()
        return types[0].copy(), sites[0].copy(), types[1].copy(), sites[1].copy()

    def import_start_config(
        self,
        types0: np.ndarray,
        sites0: np.ndarray,
        types1: np.ndarray,
        sites1: np.ndarray,
    ) -> None:
        types = np.ascontiguousarray(np.stack([types0, types1]), dtype=np.int32)
        sites = np.ascontiguousarray(np.stack([sites0, sites1]), dtype=np.int32)
        self._B_size = 0
        self._backend.set_mask(self._A_start)
        self._backend.set_operator_strings(types, sites)
        self._backend.save_checkpoint()
        self._checkpoint_valid = True

    def _accumulate_work(self, old: float, new: float) -> tuple[float, int]:
        b = int(self._B_size)
        d = len(self._D_sites)
        join = b * (math.log(new) - math.log(old)) if b else 0.0
        unjoined = d - b
        if not unjoined:
            split = 0.0
            end_unjoined = 0
        elif new >= 1.0:
            split = 0.0
            end_unjoined = unjoined
        else:
            split = unjoined * (math.log1p(-new) - math.log1p(-old))
            end_unjoined = 0
        return -(join + split), end_unjoined

    def run_trajectory(self) -> WorkTrajectoryResult:
        if self._schedule is None:
            raise RuntimeError("call set_lambda_schedule first")
        if not len(self._D_sites):
            return WorkTrajectoryResult(0.0, 1.0, 0, 0, 0, 0)
        work = 0.0
        attempts = accepts = unjoined_at_end = 0
        for old, new in zip(self._schedule[:-1], self._schedule[1:], strict=True):
            dw, unjoined = self._accumulate_work(float(old), float(new))
            work += dw
            unjoined_at_end += unjoined
            if 0.0 < new < 1.0:
                for _ in range(self._n_topology):
                    stats = self._backend.topology_sweep(self._D_sites, float(new))
                    attempts += int(stats["attempts"])
                    accepts += int(stats["accepts"])
                    self._B_size = int(stats["active_count"])
            else:
                # CPU reference still visits every proposal at λ endpoints;
                # they are counted as attempts and deterministically rejected.
                attempts += self._n_topology * len(self._D_sites)
            for _ in range(self._n_qaqmc):
                self._backend.mc_step()
        return WorkTrajectoryResult(
            work=work,
            exp_minus_work=math.exp(-work),
            final_swap_count=int(self._B_size),
            unjoined_at_end_count=unjoined_at_end,
            topology_attempts=attempts,
            topology_accepts=accepts,
        )

    def run_trajectories(
        self, n_trajectories: int, decorrelation_steps: int
    ) -> WorkRunResult:
        if n_trajectories < 0 or decorrelation_steps < 0:
            raise ValueError("trajectory/decorrelation counts must be non-negative")
        if not len(self._D_sites):
            count = int(n_trajectories)
            return WorkRunResult(
                0.0, 1.0, 0.0, 0.0, count, 0, 0, 0,
                np.zeros(count, dtype=np.float64),
                np.zeros(count, dtype=np.int32),
                np.zeros(count, dtype=np.int32),
                np.zeros(count, dtype=np.int64),
                np.zeros(count, dtype=np.int64),
            )
        count = int(n_trajectories)
        work = np.empty(count, dtype=np.float64)
        final = np.empty(count, dtype=np.int32)
        unjoined = np.empty(count, dtype=np.int32)
        attempts = np.empty(count, dtype=np.int64)
        accepts = np.empty(count, dtype=np.int64)
        for trajectory in range(count):
            self._reset_start()
            self._backend.run_steps(int(decorrelation_steps))
            self._backend.save_checkpoint()
            self._checkpoint_valid = True
            row = self.run_trajectory()
            work[trajectory] = row.work
            final[trajectory] = row.final_swap_count
            unjoined[trajectory] = row.unjoined_at_end_count
            attempts[trajectory] = row.topology_attempts
            accepts[trajectory] = row.topology_accepts
        if len(work):
            x = -work
            maximum = float(x.max())
            log_mean = maximum + math.log(float(np.exp(x - maximum).mean()))
            mean_exp = math.exp(log_mean)
            delta_s2 = -log_mean
            work_mean = float(work.mean())
            work_var = float(work.var(ddof=1)) if len(work) > 1 else 0.0
        else:
            mean_exp, delta_s2, work_mean, work_var = 1.0, 0.0, 0.0, 0.0
        return WorkRunResult(
            delta_s2, mean_exp, work_mean, work_var, len(work),
            int(attempts.sum()), int(accepts.sum()), int(unjoined.sum()),
            work, final, unjoined, attempts, accepts,
        )

    @property
    def A_start_mask(self) -> np.ndarray:
        return self._A_start.copy()

    @property
    def A_end_mask(self) -> np.ndarray:
        return self._A_end.copy()

    @property
    def D_mask(self) -> np.ndarray:
        return self._D_mask.copy()

    @property
    def B_mask(self) -> np.ndarray:
        current = self._backend.get_mask()
        return ((current ^ self._A_start) & self._D_mask).astype(np.uint8)

    @property
    def D_size(self) -> int:
        return len(self._D_sites)

    @property
    def B_size(self) -> int:
        return int(self._B_size)

    @property
    def lambda_schedule(self) -> np.ndarray:
        return np.empty(0) if self._schedule is None else self._schedule.copy()

    @property
    def device_bytes(self) -> int:
        return self._backend.device_bytes
