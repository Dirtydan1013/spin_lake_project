"""True in-process batched CUDA backend for independent QAQMC chains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:
    import qaqmc_cpp
except (ImportError, OSError):
    qaqmc_cpp = None

try:
    import qaqmc_cuda
except (ImportError, OSError):
    qaqmc_cuda = None

from src.engines.qaqmc_cuda import (
    _CLUSTER_STREAM,
    _DIAGONAL_STREAM,
    _TOPOLOGY_STREAM,
    _U64_MASK,
    cuda_available,
)


_CHAIN_STRIDE = 0x9E3779B97F4A7C15


def independent_chain_seeds(seed: int, batch_size: int) -> np.ndarray:
    """Deterministic non-overlapping Philox keys, preserving B=1 compatibility."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    base = int(seed) & _U64_MASK
    return np.asarray(
        [(base + chain * _CHAIN_STRIDE) & _U64_MASK
         for chain in range(batch_size)],
        dtype=np.uint64,
    )


@dataclass
class CudaDiagonalBatchBackend:
    """One CUDA context owning B standard/off-diagonal Markov chains."""

    engine: Any
    seeds: np.ndarray
    sweep_ids: np.ndarray
    topology_ids: np.ndarray

    @classmethod
    def from_cpu_engine(
        cls,
        cpu_engine: Any,
        *,
        batch_size: int,
        device: int = 0,
        seed: int = 0,
        op_types: np.ndarray | None = None,
        op_sites: np.ndarray | None = None,
    ) -> "CudaDiagonalBatchBackend":
        if not cuda_available():
            raise RuntimeError("qaqmc_cuda is unavailable or no GPU is visible")
        batch_size = int(batch_size)
        length = 2 * int(cpu_engine.M)
        if op_types is None:
            types = np.repeat(
                np.asarray(cpu_engine.op_types, dtype=np.int32)[None, :],
                batch_size,
                axis=0,
            )
        else:
            types = np.ascontiguousarray(op_types, dtype=np.int32)
        if op_sites is None:
            sites = np.repeat(
                np.asarray(cpu_engine.op_sites, dtype=np.int32)[None, :],
                batch_size,
                axis=0,
            )
        else:
            sites = np.ascontiguousarray(op_sites, dtype=np.int32)
        expected = (batch_size, length)
        if types.shape != expected or sites.shape != expected:
            raise ValueError(f"operator arrays must have shape {expected}")
        data = cpu_engine.export_cuda_diagonal_data()
        engine = qaqmc_cuda.BatchedDiagonalEngine(
            batch_size=batch_size,
            n_sites=int(cpu_engine.N),
            half_length=int(cpu_engine.M),
            delta_min=float(cpu_engine.delta_min),
            delta_max=float(cpu_engine.delta_max),
            epsilon=float(cpu_engine.epsilon),
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
    def M(self) -> int:
        return self.M_total // 2

    @property
    def device_bytes(self) -> int:
        return int(self.engine.device_bytes)

    @property
    def shared_model_bytes(self) -> int:
        return int(self.engine.shared_model_bytes)

    def mc_step(self) -> list[dict[str, Any]]:
        sweeps = self.sweep_ids.copy()
        diagonal = self.engine.diagonal_update(
            self.seeds ^ np.uint64(_DIAGONAL_STREAM), sweeps
        )
        for chain, stats in enumerate(diagonal):
            if stats["failed_slots"]:
                raise RuntimeError(
                    f"CUDA batch chain {chain} proposal limit reached: {stats}"
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

    def set_string_sites(
        self, sites: Sequence[int], m_star: int | None = None
    ) -> None:
        values = np.ascontiguousarray(sites, dtype=np.int32)
        if values.ndim != 1 or len(values) > 64:
            raise ValueError("string_sites must be 1D with at most 64 sites")
        if np.any((values < 0) | (values >= self.N)):
            raise ValueError("string site is outside the lattice")
        if len(np.unique(values)) != len(values):
            raise ValueError("string sites must be unique")
        self.engine.set_string_sites(values, self.M if m_star is None else int(m_star))
        self.topology_ids.fill(0)

    def set_seam_masks_consistent(self, masks: np.ndarray) -> None:
        values = np.ascontiguousarray(masks, dtype=np.uint64)
        if values.shape != (self.batch_size,):
            raise ValueError(f"masks must have shape ({self.batch_size},)")
        self.engine.set_seam_masks_consistent(values)

    @property
    def seam_masks(self) -> np.ndarray:
        return np.asarray(self.engine.get_seam_masks(), dtype=np.uint64)

    def topology_sweep(self, lambda_: float) -> list[dict[str, int | float]]:
        ids = self.topology_ids.copy()
        stats = self.engine.topology_sweep(
            float(lambda_), self.seeds ^ np.uint64(_TOPOLOGY_STREAM), ids
        )
        self.topology_ids += np.uint64(1)
        return [dict(row) for row in stats]

    def save_device_checkpoint(self) -> None:
        self.engine.save_checkpoint()

    def restore_device_checkpoint(self) -> None:
        self.engine.restore_checkpoint()

    @property
    def has_checkpoint(self) -> bool:
        return bool(self.engine.has_checkpoint)

    def get_operator_strings(self) -> tuple[np.ndarray, np.ndarray]:
        types, sites = self.engine.get_operator_strings()
        return np.asarray(types, dtype=np.int32), np.asarray(sites, dtype=np.int32)

    def set_operator_strings(self, types: np.ndarray, sites: np.ndarray) -> None:
        expected = (self.batch_size, self.M_total)
        types = np.ascontiguousarray(types, dtype=np.int32)
        sites = np.ascontiguousarray(sites, dtype=np.int32)
        if types.shape != expected or sites.shape != expected:
            raise ValueError(f"operator arrays must have shape {expected}")
        self.engine.set_operator_strings(types, sites)

    def profile_states(self, profile_step: int) -> np.ndarray:
        packed = np.asarray(self.engine.profile_states(int(profile_step)), dtype=np.uint64)
        shifts = np.arange(64, dtype=np.uint64)
        bits = ((packed[..., None] >> shifts) & np.uint64(1)).astype(np.uint8)
        return bits.reshape(self.batch_size, packed.shape[1], -1)[..., : self.N]

    def midpoint_states(self) -> np.ndarray:
        return self.profile_states(self.M)[:, 0, :]


class QAQMC_Rydberg_CUDA_Batch:
    """Geometry-level constructor for B standard QAQMC chains on one GPU."""

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
        pos: np.ndarray | None = None,
        epsilon: float = 0.01,
        seed: int = 42,
        neighbor_cutoff: int | None = None,
        delta_groups: int = 600,
        box_vectors: np.ndarray | None = None,
        device: int = 0,
        verbose: bool = True,
    ) -> None:
        if qaqmc_cpp is None:
            raise RuntimeError("qaqmc_cpp is required to construct CUDA model tables")
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
        self.batch_size = int(batch_size)
        self.device = int(device)
        if verbose:
            info = qaqmc_cuda.device_info()[self.device]
            print(
                f"[QAQMC-CUDA-BATCH] device={device} {info['name']} "
                f"B={batch_size} N={N} M={M} "
                f"resident={self.device_bytes / 2**20:.1f} MiB "
                f"shared={self.shared_model_bytes / 2**20:.1f} MiB"
            )

    @property
    def engine(self) -> CudaDiagonalBatchBackend:
        return self._backend

    @property
    def device_bytes(self) -> int:
        return self._backend.device_bytes

    @property
    def shared_model_bytes(self) -> int:
        return self._backend.shared_model_bytes

    def mc_step(self) -> list[dict[str, Any]]:
        return self._backend.mc_step()

    def run_steps(self, count: int) -> None:
        self._backend.run_steps(count)

    def midpoint_states(self) -> np.ndarray:
        return self._backend.midpoint_states()
