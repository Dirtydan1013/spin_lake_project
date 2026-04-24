"""
Two-replica QAQMC wrapper for Renyi-2 / ratio-estimator development.

This wrapper intentionally uses a dedicated C++ core so the original QAQMC
engine remains untouched.
"""

from __future__ import annotations

import numpy as np

try:
    import os
    import shutil
    import sys

    _gpp = shutil.which("g++")
    if _gpp and os.name == "nt":
        _mingw_bin = os.path.dirname(os.path.realpath(_gpp))
        os.add_dll_directory(_mingw_bin)
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _build_dir = os.path.join(_repo_root, "build")
    if _build_dir not in sys.path:
        sys.path.insert(0, _build_dir)
    import qaqmc_cpp
except (ImportError, OSError) as exc:
    raise ImportError("qaqmc_cpp with QAQMCRenyiEngine is required for src.qaqmc_renyi") from exc


class QAQMCRenyiRydberg:
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
    ):
        if pos is None:
            pos = np.arange(N).reshape(-1, 1).astype(np.float64)
        pos_arr = np.ascontiguousarray(pos, dtype=np.float64)
        nc = neighbor_cutoff if neighbor_cutoff is not None else -1

        self._cpp_engine = qaqmc_cpp.QAQMCRenyiEngine(
            N=N,
            Omega=Omega,
            delta_min=delta_min,
            delta_max=delta_max,
            Rb=Rb,
            M=M,
            epsilon=epsilon,
            seed=seed,
            pos=pos_arr,
            neighbor_cutoff=nc,
        )
        self.N = N
        self.M = M
        self.M_total = 2 * M
        self.pos = pos_arr

    def mc_step(self):
        self._cpp_engine.mc_step()

    def run_steps(self, n_steps: int):
        self._cpp_engine.run_steps(int(n_steps))

    def set_A_mask(self, mask):
        mask_arr = np.ascontiguousarray(mask, dtype=np.uint8)
        self._cpp_engine.set_A_mask(mask_arr)

    def set_topology_pair(self, A_k, A_kp1, diff_site: int):
        lower = np.ascontiguousarray(A_k, dtype=np.uint8)
        upper = np.ascontiguousarray(A_kp1, dtype=np.uint8)
        self._cpp_engine.set_topology_pair(lower, upper, int(diff_site))

    def reset_visit_counts(self):
        self._cpp_engine.reset_visit_counts()

    def get_visit_counts(self) -> np.ndarray:
        return np.array(self._cpp_engine.get_visit_counts(), dtype=np.int64)

    def set_ensemble_ladder(self, masks, neighbors, initial_ensemble: int = 0):
        mask_list = [np.ascontiguousarray(mask, dtype=np.uint8) for mask in masks]
        neighbor_list = [list(map(int, row)) for row in neighbors]
        self._cpp_engine.set_ensemble_ladder(mask_list, neighbor_list, int(initial_ensemble))

    def set_log_g(self, log_g):
        self._cpp_engine.set_log_g(np.ascontiguousarray(log_g, dtype=np.float64))

    def reset_visit_counts_ext(self):
        self._cpp_engine.reset_visit_counts_ext()

    def get_visit_counts_ext(self) -> np.ndarray:
        return np.array(self._cpp_engine.get_visit_counts_ext(), dtype=np.int64)

    def reset_transition_counts(self):
        self._cpp_engine.reset_transition_counts()

    def get_transition_counts(self) -> np.ndarray:
        return np.array(self._cpp_engine.get_transition_counts(), dtype=np.int64)

    def reset_collection_counts(self):
        self._cpp_engine.reset_collection_counts()

    def get_collection_counts(self) -> np.ndarray:
        return np.array(self._cpp_engine.get_collection_counts(), dtype=np.float64)

    def topology_toggle(self):
        self._cpp_engine.topology_toggle()

    def ensemble_switch(self):
        self._cpp_engine.ensemble_switch()

    def log_weight_ratio_for_site(self, site: int, from_topology: int, to_topology: int) -> float:
        return float(self._cpp_engine.log_weight_ratio_for_site(int(site), int(from_topology), int(to_topology)))

    def set_indicator_site(self, site: int):
        self._cpp_engine.set_indicator_site(int(site))

    def reset_indicator(self):
        self._cpp_engine.reset_indicator()

    def get_indicator_avg(self) -> float:
        return float(self._cpp_engine.get_indicator_avg())

    def current_indicator(self) -> int:
        return int(self._cpp_engine.current_indicator())

    def recompute_midpoint_states(self):
        self._cpp_engine.recompute_midpoint_states()

    def set_replica_op_string(self, replica: int, types, sites):
        types_arr = np.ascontiguousarray(types, dtype=np.int32)
        sites_arr = np.ascontiguousarray(sites, dtype=np.int32)
        self._cpp_engine.set_replica_op_string(int(replica), types_arr, sites_arr)

    def get_state_at_M(self, replica: int) -> np.ndarray:
        return np.array(self._cpp_engine.get_state_at_M(int(replica)), dtype=np.int32)

    def get_op_types(self, replica: int) -> np.ndarray:
        return np.array(self._cpp_engine.get_op_types(int(replica)), dtype=np.int32)

    def get_op_sites(self, replica: int) -> np.ndarray:
        return np.array(self._cpp_engine.get_op_sites(int(replica)), dtype=np.int32)

    def get_site_paths(self, site: int) -> dict[str, np.ndarray]:
        raw = self._cpp_engine.get_site_paths(int(site))
        return {k: np.array(v, dtype=np.int32) for k, v in raw.items()}

    @property
    def bond_sites(self) -> np.ndarray:
        return np.array(self._cpp_engine.bond_sites, dtype=np.int32)

    @property
    def delta_schedule(self) -> np.ndarray:
        return np.array(self._cpp_engine.delta_schedule, dtype=np.float64)

    @property
    def A_mask(self) -> np.ndarray:
        return np.array(self._cpp_engine.A_mask, dtype=np.uint8)

    def get_topology_mask(self, topology: int) -> np.ndarray:
        return np.array(self._cpp_engine.get_topology_mask(int(topology)), dtype=np.uint8)

    @property
    def current_topology(self) -> int:
        return int(self._cpp_engine.current_topology)

    @property
    def current_ensemble(self) -> int:
        return int(self._cpp_engine.current_ensemble)

    @property
    def ensemble_count(self) -> int:
        return int(self._cpp_engine.ensemble_count)

    @property
    def mode(self) -> int:
        return int(self._cpp_engine.mode)

    @property
    def diff_site(self) -> int:
        return int(self._cpp_engine.diff_site)
