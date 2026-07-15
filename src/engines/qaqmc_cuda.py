"""High-level device-resident CUDA backend for the standard QAQMC chain.

The trusted C++ engine is used once during construction to build the Rydberg
interaction graph and grouped alias envelopes.  The operator string, Markov
updates, event streams and cluster state then remain on the selected GPU.
Only compact sampled worldline states or explicit checkpoints cross PCIe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import qaqmc_cuda
except (ImportError, OSError):
    qaqmc_cuda = None

try:
    import qaqmc_cpp
except (ImportError, OSError):
    qaqmc_cpp = None


_U64_MASK = (1 << 64) - 1
_DIAGONAL_STREAM = 0xD1A60A1C5EED1234
_CLUSTER_STREAM = 0xC1057E2A5EED5678
_TOPOLOGY_STREAM = 0x5701A65E5EED9ABC


def cuda_available() -> bool:
    """Return whether the optional extension and a CUDA device are available."""
    return qaqmc_cuda is not None and bool(qaqmc_cuda.is_available())


def _normalise_site_sets(
    sets: Iterable[Sequence[int]], n_sites: int, name: str
) -> tuple[list[np.ndarray], np.ndarray, int]:
    arrays: list[np.ndarray] = []
    lengths: list[int] = []
    for values in sets:
        arr = np.ascontiguousarray(values, dtype=np.int32)
        if arr.ndim != 1:
            raise ValueError(f"each {name} entry must be one-dimensional")
        if np.any((arr < 0) | (arr >= n_sites)):
            raise ValueError(f"{name} contains a site outside [0, {n_sites})")
        arrays.append(arr)
        lengths.append(len(arr))
    unique_lengths: list[int] = []
    group_of = np.empty(len(arrays), dtype=np.int32)
    for index, length in enumerate(lengths):
        if length not in unique_lengths:
            unique_lengths.append(length)
        group_of[index] = unique_lengths.index(length)
    return arrays, group_of, len(unique_lengths)


@dataclass
class CudaDiagonalBackend:
    """Owner of one independent QAQMC Markov chain on one CUDA device."""

    engine: Any
    seed: int = 0
    sweep_id: int = 0
    topology_id: int = 0
    _bulk_sites: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    _loop_sets: list[np.ndarray] = field(default_factory=list)
    _loop_group: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    _n_loop_groups: int = 0
    _string_sets: list[np.ndarray] = field(default_factory=list)
    _string_group: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    _n_string_groups: int = 0
    _vbs_corners: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.int32))
    _vbs_parity: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    _vbs_sign: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))
    _ss_sign: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))
    _vbs_ref00: int = 0
    _vbs_ref10: int = 0

    @classmethod
    def from_cpu_engine(
        cls, cpu_engine: Any, device: int = 0, seed: int = 0
    ) -> "CudaDiagonalBackend":
        if qaqmc_cuda is None:
            raise RuntimeError("qaqmc_cuda extension is not built")
        if not qaqmc_cuda.is_available():
            raise RuntimeError("CUDA runtime does not see a usable GPU")
        data = cpu_engine.export_cuda_diagonal_data()
        engine = qaqmc_cuda.DiagonalEngine(
            n_sites=cpu_engine.N,
            half_length=cpu_engine.M,
            delta_min=cpu_engine.delta_min,
            delta_max=cpu_engine.delta_max,
            epsilon=cpu_engine.epsilon,
            bond_sites=data["bond_sites"],
            bond_vij=data["bond_vij"],
            inv_coord=data["inv_coord"],
            alias_prob=data["alias_prob"],
            alias_index=data["alias_index"],
            alias_loc_kind=data["alias_loc_kind"],
            bond_rmax=data["bond_rmax"],
            op_types=np.asarray(cpu_engine.op_types, dtype=np.int32),
            op_sites=np.asarray(cpu_engine.op_sites, dtype=np.int32),
            device=device,
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

    def diagonal_update(self, seed: int, sweep_id: int) -> dict[str, int | float]:
        return dict(self.engine.diagonal_update(seed=seed, sweep_id=sweep_id))

    def mc_step(self) -> dict[str, Any]:
        """Run one diagonal+cluster update and advance the replay counter."""
        sweep = int(self.sweep_id)
        diagonal = dict(
            self.engine.diagonal_update(
                seed=(self.seed ^ _DIAGONAL_STREAM) & _U64_MASK,
                sweep_id=sweep,
            )
        )
        if diagonal["failed_slots"]:
            raise RuntimeError(f"CUDA diagonal proposal limit reached: {diagonal}")
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

    @property
    def seam_mask(self) -> int:
        return int(self.engine.seam_mask)

    def set_string_sites(
        self, sites: Sequence[int], m_star: int | None = None
    ) -> None:
        arr = np.ascontiguousarray(sites, dtype=np.int32)
        if arr.ndim != 1 or len(arr) > 64:
            raise ValueError("string_sites must be one-dimensional with at most 64 sites")
        if np.any((arr < 0) | (arr >= self.N)) or len(np.unique(arr)) != len(arr):
            raise ValueError("string_sites must contain unique valid physical sites")
        cut = self.M if m_star is None else int(m_star)
        self.engine.set_string_sites(arr, cut)
        self.topology_id = 0

    def set_seam_mask_consistent(self, mask: int) -> None:
        self.engine.set_seam_mask_consistent(int(mask))

    @property
    def has_checkpoint(self) -> bool:
        return bool(self.engine.has_checkpoint)

    def save_device_checkpoint(self) -> None:
        self.engine.save_checkpoint()

    def restore_device_checkpoint(self) -> None:
        self.engine.restore_checkpoint()

    def topology_sweep(self, lambda_: float) -> dict[str, int | float]:
        topology = int(self.topology_id)
        result = dict(
            self.engine.topology_sweep(
                lambda_=float(lambda_),
                seed=(self.seed ^ _TOPOLOGY_STREAM) & _U64_MASK,
                sweep_id=topology,
            )
        )
        self.topology_id += 1
        return result

    def get_operator_string(self) -> tuple[np.ndarray, np.ndarray]:
        types, sites = self.engine.get_operator_string()
        return np.asarray(types), np.asarray(sites)

    def set_operator_string(self, types: np.ndarray, sites: np.ndarray) -> None:
        types = np.ascontiguousarray(types, dtype=np.int32)
        sites = np.ascontiguousarray(sites, dtype=np.int32)
        if types.shape != (self.M_total,) or sites.shape != (self.M_total,):
            raise ValueError(f"operator arrays must have shape ({self.M_total},)")
        self.engine.set_operator_string(types, sites)

    @property
    def op_types(self) -> np.ndarray:
        return self.get_operator_string()[0]

    @property
    def op_sites(self) -> np.ndarray:
        return self.get_operator_string()[1]

    def set_op_string(self, types: np.ndarray, sites: np.ndarray) -> None:
        self.set_operator_string(types, sites)

    def profile_states(self, profile_step: int) -> np.ndarray:
        """Return uint8 states after every ``profile_step`` operator slices."""
        packed = np.asarray(self.engine.profile_states(int(profile_step)), dtype=np.uint64)
        shifts = np.arange(64, dtype=np.uint64)
        bits = ((packed[:, :, None] >> shifts) & np.uint64(1)).astype(np.uint8)
        return bits.reshape(len(packed), -1)[:, : self.N]

    def midpoint_state(self) -> np.ndarray:
        return self.profile_states(self.M)[0]

    def set_bulk_sites(self, sites: Sequence[int]) -> None:
        arr = np.ascontiguousarray(sites, dtype=np.int32)
        if arr.ndim != 1 or np.any((arr < 0) | (arr >= self.N)):
            raise ValueError(f"bulk sites must be one-dimensional and in [0, {self.N})")
        self._bulk_sites = arr

    def set_observable_sites(
        self,
        loop_sets: Iterable[Sequence[int]],
        string_sets: Iterable[Sequence[int]],
    ) -> None:
        (self._loop_sets, self._loop_group, self._n_loop_groups) = (
            _normalise_site_sets(loop_sets, self.N, "loop_sets")
        )
        (self._string_sets, self._string_group, self._n_string_groups) = (
            _normalise_site_sets(string_sets, self.N, "string_sets")
        )

    def set_vbs_triangles(
        self,
        corners: np.ndarray,
        n1_parity: Sequence[int],
        vbs_sign: Sequence[float],
        ss_sign: Sequence[float],
        ref00: int,
        ref10: int,
    ) -> None:
        corners = np.ascontiguousarray(corners, dtype=np.int32).reshape(-1, 3)
        n_triangles = len(corners)
        parity = np.ascontiguousarray(n1_parity, dtype=np.int32)
        vbs = np.ascontiguousarray(vbs_sign, dtype=np.float64)
        ss = np.ascontiguousarray(ss_sign, dtype=np.float64)
        if any(arr.shape != (n_triangles,) for arr in (parity, vbs, ss)):
            raise ValueError("VBS parity/sign arrays must have one entry per triangle")
        if np.any((corners < 0) | (corners >= self.N)):
            raise ValueError("VBS triangle corner is outside the lattice")
        if n_triangles and not (0 <= ref00 < n_triangles and 0 <= ref10 < n_triangles):
            raise ValueError("VBS reference triangle is out of range")
        self._vbs_corners = corners
        self._vbs_parity = parity
        self._vbs_sign = vbs
        self._ss_sign = ss
        self._vbs_ref00 = int(ref00)
        self._vbs_ref10 = int(ref10)

    @staticmethod
    def _measure_products(
        states: np.ndarray,
        sets: list[np.ndarray],
        group_of: np.ndarray,
        n_groups: int,
    ) -> np.ndarray:
        output = np.zeros((len(states), n_groups), dtype=np.float64)
        counts = np.zeros(n_groups, dtype=np.int32)
        signed = 1 - 2 * states.astype(np.int8, copy=False)
        for index, sites in enumerate(sets):
            group = int(group_of[index])
            output[:, group] += np.prod(signed[:, sites], axis=1, dtype=np.int64)
            counts[group] += 1
        if n_groups:
            output /= counts[None, :]
        return output

    def measure_states(self, states: np.ndarray) -> dict[str, np.ndarray]:
        states = np.asarray(states, dtype=np.uint8)
        if states.ndim == 1:
            states = states[None, :]
        if states.ndim != 2 or states.shape[1] != self.N:
            raise ValueError(f"states must have shape (n_points, {self.N})")
        density_sites = self._bulk_sites if len(self._bulk_sites) else np.arange(self.N)
        result = {
            "density": np.mean(states[:, density_sites], axis=1),
            "Z_l": self._measure_products(
                states, self._loop_sets, self._loop_group, self._n_loop_groups
            ),
            "C_m_l": self._measure_products(
                states, self._string_sets, self._string_group,
                self._n_string_groups,
            ),
        }
        if len(self._vbs_corners):
            triangle_state = (
                4 * states[:, self._vbs_corners[:, 0]]
                + 2 * states[:, self._vbs_corners[:, 1]]
                + states[:, self._vbs_corners[:, 2]]
            )
            state00 = triangle_state[:, self._vbs_ref00]
            state10 = triangle_state[:, self._vbs_ref10]
            gauge = np.where(state10 == state00, 1.0, -1.0)
            even_u = np.where(triangle_state == state00[:, None], 1.0, -1.0)
            odd_u = (
                np.where(triangle_state == state10[:, None], 1.0, -1.0)
                * gauge[:, None]
            )
            u = np.where(self._vbs_parity[None, :] == 0, even_u, odd_u)
            result["M_vbs"] = np.mean(u * self._vbs_sign[None, :], axis=1)
            result["M_ss"] = np.mean(u * self._ss_sign[None, :], axis=1)
        else:
            result["M_vbs"] = np.empty((len(states), 0), dtype=np.float64)
            result["M_ss"] = np.empty((len(states), 0), dtype=np.float64)
        return result

    def measure_at_midpoint(self) -> dict[str, Any]:
        measured = self.measure_states(self.midpoint_state())
        return {
            "density": float(measured["density"][0]),
            "Z_l": measured["Z_l"][0],
            "C_m_l": measured["C_m_l"][0],
            "M_vbs": (float(measured["M_vbs"][0])
                      if measured["M_vbs"].ndim == 1 else None),
            "M_ss": (float(measured["M_ss"][0])
                     if measured["M_ss"].ndim == 1 else None),
        }

    def measure_profile(self, profile_step: int) -> dict[str, np.ndarray]:
        return self.measure_states(self.profile_states(profile_step))

    @staticmethod
    def occupation_sf_matrices(
        states: np.ndarray,
        site_cell_R: np.ndarray,
        site_basis: Sequence[int],
        q_points: np.ndarray,
        site_in_bulk: Sequence[int] | None = None,
        n_basis: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Sublattice occupation-SF matrices from compact profile states.

        This host reducer is intended for a small selected set of delta points;
        the O(2M) propagation remains on the GPU and only ``points * N`` bits
        are downloaded.  Matrices use ``s_a * conj(s_b)`` like the CPU profile
        backend.
        """
        states = np.asarray(states, dtype=np.float64)
        cell_R = np.asarray(site_cell_R, dtype=np.float64)
        basis = np.asarray(site_basis, dtype=np.int32)
        q_points = np.atleast_2d(np.asarray(q_points, dtype=np.float64))
        if states.ndim == 1:
            states = states[None, :]
        if cell_R.ndim != 2 or states.shape[1] != len(cell_R) or len(basis) != len(cell_R):
            raise ValueError("states, site_cell_R and site_basis site dimensions differ")
        if q_points.shape[1] != cell_R.shape[1]:
            raise ValueError("q-point and site-cell coordinate dimensions differ")
        if n_basis is None:
            valid = basis[basis >= 0]
            n_basis = int(valid.max()) + 1 if len(valid) else 0
        if n_basis < 0:
            raise ValueError("n_basis must be non-negative")
        bulk = (np.ones(len(basis), dtype=bool) if site_in_bulk is None
                else np.asarray(site_in_bulk, dtype=bool))
        if bulk.shape != basis.shape:
            raise ValueError("site_in_bulk must have one entry per site")

        phase = np.exp(1j * (q_points @ cell_R.T))
        shape = (len(states), len(q_points), n_basis)
        s_full = np.zeros(shape, dtype=np.complex128)
        s_bulk = np.zeros(shape, dtype=np.complex128)
        for sublattice in range(n_basis):
            selected = basis == sublattice
            if np.any(selected):
                s_full[:, :, sublattice] = states[:, selected] @ phase[:, selected].T
            selected_bulk = selected & bulk
            if np.any(selected_bulk):
                s_bulk[:, :, sublattice] = (
                    states[:, selected_bulk] @ phase[:, selected_bulk].T
                )
        full_matrix = s_full[..., :, None] * np.conj(s_full[..., None, :])
        bulk_matrix = s_bulk[..., :, None] * np.conj(s_bulk[..., None, :])
        return {
            "S_full": full_matrix,
            "S_bulk": bulk_matrix,
            "s_full": s_full,
            "s_bulk": s_bulk,
            "n_profile": states,
        }

    def run_onthefly(
        self,
        n_equil: int,
        n_samples: int,
        me_density: int = 1,
        me_zl: int = 1,
        me_cml: int = 1,
    ) -> dict[str, np.ndarray]:
        """Midpoint sampler compatible with the standard CPU result layout."""
        for value, name in ((n_equil, "n_equil"), (n_samples, "n_samples")):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        intervals = [max(1, int(me_density)), max(1, int(me_zl)), max(1, int(me_cml))]
        self.run_steps(n_equil)
        total_steps = n_samples * min(intervals)
        density: list[float] = []
        loops: list[np.ndarray] = []
        strings: list[np.ndarray] = []
        for step in range(1, total_steps + 1):
            self.mc_step()
            if any(step % interval == 0 for interval in intervals):
                obs = self.measure_at_midpoint()
                if step % intervals[0] == 0:
                    density.append(obs["density"])
                if step % intervals[1] == 0:
                    loops.append(obs["Z_l"])
                if step % intervals[2] == 0:
                    strings.append(obs["C_m_l"])
        return {
            "density": np.asarray(density, dtype=np.float64),
            "Z_l": np.asarray(loops, dtype=np.float64).reshape(-1, self._n_loop_groups),
            "C_m_l": np.asarray(strings, dtype=np.float64).reshape(
                -1, self._n_string_groups
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        types, sites = self.get_operator_string()
        return {
            "format_version": 1,
            "n_sites": self.N,
            "half_length": self.M,
            "seed": int(self.seed),
            "sweep_id": int(self.sweep_id),
            "topology_id": int(self.topology_id),
            "op_types": types.astype(np.int8, copy=False),
            "op_sites": sites.astype(np.int32, copy=False),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("format_version", 0)) != 1:
            raise ValueError("unsupported CUDA checkpoint format")
        if int(state["n_sites"]) != self.N or int(state["half_length"]) != self.M:
            raise ValueError("checkpoint geometry/operator length does not match engine")
        self.set_operator_string(state["op_types"], state["op_sites"])
        self.seed = int(state["seed"]) & _U64_MASK
        self.sweep_id = int(state["sweep_id"])
        self.topology_id = int(state.get("topology_id", 0))

    def save_checkpoint(self, path: str | os.PathLike[str]) -> None:
        """Atomically save operator string and stateless Philox replay counter."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
            ) as handle:
                temporary_name = handle.name
                np.savez(handle, **state)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def load_checkpoint(self, path: str | os.PathLike[str]) -> None:
        with np.load(path, allow_pickle=False) as archive:
            self.load_state_dict({key: archive[key] for key in archive.files})


class QAQMC_Rydberg_CUDA:
    """Geometry-level constructor mirroring ``QAQMC_Rydberg`` for one GPU."""

    def __init__(
        self,
        N: int,
        M: int,
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
            raise RuntimeError("qaqmc_cpp is required once to build CUDA alias tables")
        if not cuda_available():
            raise RuntimeError("qaqmc_cuda is unavailable or no GPU is visible")
        positions = (
            np.arange(N, dtype=np.float64).reshape(-1, 1)
            if pos is None else np.ascontiguousarray(pos, dtype=np.float64)
        )
        if positions.ndim != 2 or positions.shape[0] != N:
            raise ValueError("pos must have shape (N, dimension)")
        box = (None if box_vectors is None
               else np.ascontiguousarray(box_vectors, dtype=np.float64))
        cpu = qaqmc_cpp.QAQMCEngine(
            N, Omega, delta_min, delta_max, Rb, M, epsilon, seed, positions,
            neighbor_cutoff=(-1 if neighbor_cutoff is None else neighbor_cutoff),
            delta_groups=delta_groups,
            box_vectors=box,
        )
        self._backend = CudaDiagonalBackend.from_cpu_engine(
            cpu, device=device, seed=seed
        )
        self.N = int(N)
        self.M = int(M)
        self.M_total = 2 * int(M)
        self.pos = positions
        self.seed = int(seed)
        if verbose:
            info = qaqmc_cuda.device_info()[device]
            print(
                f"[QAQMC-CUDA] device={device} {info['name']} N={N} M={M} "
                f"resident={self._backend.device_bytes / 2**20:.1f} MiB"
            )

    @property
    def engine(self) -> CudaDiagonalBackend:
        return self._backend

    def mc_step(self) -> dict[str, Any]:
        return self._backend.mc_step()

    def run_steps(self, count: int) -> None:
        self._backend.run_steps(count)

    def set_bulk_sites(self, sites: Sequence[int]) -> None:
        self._backend.set_bulk_sites(sites)

    def set_observable_sites(
        self,
        loop_sets: Iterable[Sequence[int]],
        string_sets: Iterable[Sequence[int]],
    ) -> None:
        self._backend.set_observable_sites(loop_sets, string_sets)

    def set_vbs_triangles(self, *args: Any, **kwargs: Any) -> None:
        self._backend.set_vbs_triangles(*args, **kwargs)

    def measure_at_midpoint(self) -> dict[str, Any]:
        return self._backend.measure_at_midpoint()

    def measure_profile(self, profile_step: int) -> dict[str, np.ndarray]:
        return self._backend.measure_profile(profile_step)

    def run_onthefly(self, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        return self._backend.run_onthefly(*args, **kwargs)

    def save_checkpoint(self, path: str | os.PathLike[str]) -> None:
        self._backend.save_checkpoint(path)

    def load_checkpoint(self, path: str | os.PathLike[str]) -> None:
        self._backend.load_checkpoint(path)
