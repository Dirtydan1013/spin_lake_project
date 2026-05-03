"""
Expanded-ensemble reweighting helpers for QAQMC Renyi development.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from src.engines.qaqmc_renyi import QAQMCRenyiRydberg


@dataclass
class AutoTuneResult:
    log_g: np.ndarray
    visit_counts: list[np.ndarray]
    collection_counts: list[np.ndarray] | None = None


@dataclass
class ExpandedProductionResult:
    log_g: np.ndarray
    visit_counts: np.ndarray
    block_visit_counts: np.ndarray | None
    transition_counts: np.ndarray
    collection_counts: np.ndarray | None
    log_z: np.ndarray
    log_z_err: np.ndarray
    block_collection_counts: np.ndarray | None = None
    collection_log_z: np.ndarray | None = None
    collection_log_z_err: np.ndarray | None = None

    def s2(self, target_ensemble: int, reference_ensemble: int = 0) -> tuple[float, float]:
        target = int(target_ensemble)
        reference = int(reference_ensemble)
        return (
            float(-(self.log_z[target] - self.log_z[reference])),
            float(np.sqrt(self.log_z_err[target] ** 2 + self.log_z_err[reference] ** 2)),
        )

    def s2_collection(self, target_ensemble: int, reference_ensemble: int = 0) -> tuple[float, float]:
        if self.collection_log_z is None or self.collection_log_z_err is None:
            raise ValueError("collection-based log Z is not available")
        target = int(target_ensemble)
        reference = int(reference_ensemble)
        return (
            float(-(self.collection_log_z[target] - self.collection_log_z[reference])),
            float(np.sqrt(
                self.collection_log_z_err[target] ** 2 +
                self.collection_log_z_err[reference] ** 2
            )),
        )


def _as_mask_list(masks) -> list[np.ndarray]:
    return [np.ascontiguousarray(mask, dtype=np.uint8).reshape(-1) for mask in masks]


def _normalized_log_z(counts, log_g, *, reference_ensemble: int = 0) -> np.ndarray:
    count_arr = np.asarray(counts, dtype=np.float64).reshape(-1)
    log_g_arr = np.asarray(log_g, dtype=np.float64).reshape(-1)
    if count_arr.size != log_g_arr.size:
        raise ValueError("counts and log_g must have the same length")
    if np.any(count_arr <= 0.0):
        raise ValueError("all ensemble counts must be positive to form log Z")

    log_z = np.log(count_arr) - log_g_arr
    log_z -= log_z[int(reference_ensemble)]
    return log_z


def _jackknife_log_z(block_counts, log_g, *, reference_ensemble: int = 0) -> tuple[np.ndarray, np.ndarray]:
    block_arr = np.asarray(block_counts, dtype=np.int64)
    if block_arr.ndim != 2:
        raise ValueError("block_counts must be a 2D array")
    if block_arr.shape[0] <= 1:
        total = np.sum(block_arr, axis=0, dtype=np.int64)
        log_z = _normalized_log_z(total, log_g, reference_ensemble=reference_ensemble)
        return log_z, np.zeros_like(log_z)

    total = np.sum(block_arr, axis=0, dtype=np.int64)
    log_z = _normalized_log_z(total, log_g, reference_ensemble=reference_ensemble)

    loo = []
    for idx in range(block_arr.shape[0]):
        counts = total - block_arr[idx]
        if np.any(counts <= 0):
            raise ValueError("each leave-one-out block count must remain positive")
        loo.append(_normalized_log_z(counts, log_g, reference_ensemble=reference_ensemble))
    loo_arr = np.asarray(loo, dtype=np.float64)
    loo_mean = np.mean(loo_arr, axis=0)
    prefactor = (block_arr.shape[0] - 1) / float(block_arr.shape[0])
    log_z_err = np.sqrt(prefactor * np.sum((loo_arr - loo_mean) ** 2, axis=0))
    return log_z, log_z_err


def _is_zero_count_log_z_error(exc: ValueError) -> bool:
    msg = str(exc)
    return (
        "ensemble counts must be positive" in msg or
        "leave-one-out block count must remain positive" in msg
    )


def _nan_log_z_pair(log_g) -> tuple[np.ndarray, np.ndarray]:
    shape = np.asarray(log_g, dtype=np.float64).reshape(-1).shape
    return np.full(shape, np.nan, dtype=np.float64), np.full(shape, np.nan, dtype=np.float64)


def _jackknife_log_z_or_nan(block_counts, log_g, *, reference_ensemble: int = 0,
                            require_histogram: bool = True) -> tuple[np.ndarray, np.ndarray]:
    try:
        return _jackknife_log_z(
            block_counts,
            log_g,
            reference_ensemble=reference_ensemble,
        )
    except ValueError as exc:
        if require_histogram or not _is_zero_count_log_z_error(exc):
            raise
        return _nan_log_z_pair(log_g)


def _row_stochastic_from_collection(collection_counts) -> np.ndarray:
    matrix = np.asarray(collection_counts, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("collection_counts must be a square 2D array")
    row_sums = np.sum(matrix, axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError("each ensemble must contribute positive collection weight")
    return matrix / row_sums[:, None]


def _stationary_distribution_from_transition_matrix(transition_matrix, *, tol: float = 1e-12,
                                                    max_iters: int = 100000) -> np.ndarray:
    matrix = np.asarray(transition_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition_matrix must be a square 2D array")
    n_ens = matrix.shape[0]
    prob = np.full(n_ens, 1.0 / n_ens, dtype=np.float64)
    for _ in range(int(max_iters)):
        next_prob = prob @ matrix
        next_prob /= np.sum(next_prob)
        if np.max(np.abs(next_prob - prob)) < float(tol):
            return next_prob
        prob = next_prob
    return prob


def _normalized_log_z_from_collection(collection_counts, log_g, *,
                                      reference_ensemble: int = 0) -> np.ndarray:
    log_g_arr = np.asarray(log_g, dtype=np.float64).reshape(-1)
    trans = _row_stochastic_from_collection(collection_counts)
    stationary = _stationary_distribution_from_transition_matrix(trans)
    if stationary.size != log_g_arr.size:
        raise ValueError("collection_counts and log_g must have compatible sizes")
    if np.any(stationary <= 0.0):
        raise ValueError("stationary distribution must stay positive")
    log_z = np.log(stationary) - log_g_arr
    log_z -= log_z[int(reference_ensemble)]
    return log_z


def _jackknife_log_z_from_collection(block_collection_counts, log_g, *,
                                     reference_ensemble: int = 0) -> tuple[np.ndarray, np.ndarray]:
    block_arr = np.asarray(block_collection_counts, dtype=np.float64)
    if block_arr.ndim != 3 or block_arr.shape[1] != block_arr.shape[2]:
        raise ValueError("block_collection_counts must be a 3D square-array stack")
    if block_arr.shape[0] <= 1:
        total = np.sum(block_arr, axis=0, dtype=np.float64)
        log_z = _normalized_log_z_from_collection(total, log_g, reference_ensemble=reference_ensemble)
        return log_z, np.zeros_like(log_z)

    total = np.sum(block_arr, axis=0, dtype=np.float64)
    log_z = _normalized_log_z_from_collection(total, log_g, reference_ensemble=reference_ensemble)

    loo = []
    for idx in range(block_arr.shape[0]):
        collection = total - block_arr[idx]
        loo.append(_normalized_log_z_from_collection(
            collection,
            log_g,
            reference_ensemble=reference_ensemble,
        ))
    loo_arr = np.asarray(loo, dtype=np.float64)
    loo_mean = np.mean(loo_arr, axis=0)
    prefactor = (block_arr.shape[0] - 1) / float(block_arr.shape[0])
    log_z_err = np.sqrt(prefactor * np.sum((loo_arr - loo_mean) ** 2, axis=0))
    return log_z, log_z_err


def _is_zero_collection_log_z_error(exc: ValueError) -> bool:
    msg = str(exc)
    return (
        "ensemble must contribute positive collection weight" in msg or
        "stationary distribution must stay positive" in msg
    )


def _jackknife_log_z_from_collection_or_nan(block_collection_counts, log_g, *,
                                            reference_ensemble: int = 0,
                                            require_collection: bool = True
                                            ) -> tuple[np.ndarray, np.ndarray]:
    try:
        return _jackknife_log_z_from_collection(
            block_collection_counts,
            log_g,
            reference_ensemble=reference_ensemble,
        )
    except ValueError as exc:
        if require_collection or not _is_zero_collection_log_z_error(exc):
            raise
        return _nan_log_z_pair(log_g)


class ReweightingDriver:
    def __init__(self, engine: QAQMCRenyiRydberg | None = None, **engine_kwargs):
        if engine is None:
            engine = QAQMCRenyiRydberg(**engine_kwargs)
        self.engine = engine
        self._masks: list[np.ndarray] = []
        self._neighbors: list[list[int]] = []
        self._log_g: np.ndarray | None = None

    def set_ensemble_ladder(self, masks, neighbors, *, initial_ensemble: int = 0):
        mask_list = _as_mask_list(masks)
        self.engine.set_ensemble_ladder(mask_list, neighbors, initial_ensemble=initial_ensemble)
        self._masks = mask_list
        self._neighbors = [list(map(int, row)) for row in neighbors]
        self._log_g = np.zeros(len(mask_list), dtype=np.float64)
        self.engine.set_log_g(self._log_g)

    def set_log_g(self, log_g):
        log_g_arr = np.ascontiguousarray(log_g, dtype=np.float64).reshape(-1)
        if not self._masks:
            raise ValueError("set_ensemble_ladder must be called before set_log_g")
        if log_g_arr.size != len(self._masks):
            raise ValueError("log_g length mismatch")
        self._log_g = log_g_arr.copy()
        self.engine.set_log_g(self._log_g)

    @property
    def log_g(self) -> np.ndarray:
        if self._log_g is None:
            raise ValueError("log_g is not initialized")
        return self._log_g.copy()

    def warm_up(self, n_steps: int) -> None:
        """Thermalise the MC chain at the current ``log_g`` without recording stats.

        Use after ``set_log_g`` (e.g. after loading saved weights) to give the
        chain a chance to leave the engine's fresh-constructor state before
        autotune or production starts measuring.  All counters are reset both
        before and after so any subsequent autotune / production run sees a
        clean slate.  ``log_g`` is preserved.
        """
        if not self._masks:
            raise ValueError("set_ensemble_ladder must be called before warm_up")
        self.engine.reset_visit_counts_ext()
        self.engine.reset_transition_counts()
        self.engine.reset_collection_counts()
        n = int(n_steps)
        if n > 0:
            self.engine.run_steps(n)
        self.engine.reset_visit_counts_ext()
        self.engine.reset_transition_counts()
        self.engine.reset_collection_counts()

    def auto_tune(self, *, n_steps_per_iter: int, max_iters: int = 10, tol: float = 0.3,
                  method: str = "visit_count", damping: float = 1.0) -> AutoTuneResult:
        if not self._masks:
            raise ValueError("set_ensemble_ladder must be called before auto_tune")
        if n_steps_per_iter <= 0:
            raise ValueError("n_steps_per_iter must be positive")
        if max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if not (0.0 < float(damping) <= 1.0):
            raise ValueError("damping must be in (0, 1]")
        if self._log_g is None:
            self._log_g = np.zeros(len(self._masks), dtype=np.float64)
            self.engine.set_log_g(self._log_g)
        method_name = str(method).strip().lower()
        if method_name not in {"visit_count", "transition_matrix"}:
            raise ValueError("method must be 'visit_count' or 'transition_matrix'")

        history: list[np.ndarray] = []
        collection_history: list[np.ndarray] = []
        for _ in range(int(max_iters)):
            self.engine.reset_visit_counts_ext()
            if method_name == "transition_matrix":
                self.engine.reset_collection_counts()
            self.engine.run_steps(int(n_steps_per_iter))
            counts = self.engine.get_visit_counts_ext()
            history.append(counts.copy())

            safe_counts = np.maximum(counts.astype(np.float64), 1.0)
            if method_name == "visit_count":
                target_log_g = self._log_g - np.log(safe_counts / safe_counts.mean())
            else:
                collection = self.engine.get_collection_counts()
                collection_history.append(collection.copy())
                log_z_est = _normalized_log_z_from_collection(
                    collection,
                    self._log_g,
                    reference_ensemble=0,
                )
                target_log_g = -log_z_est
            target_log_g -= target_log_g[0]
            self._log_g = (1.0 - float(damping)) * self._log_g + float(damping) * target_log_g
            self._log_g -= self._log_g[0]
            self.engine.set_log_g(self._log_g)

            if float(np.max(safe_counts) / np.min(safe_counts)) < float(tol):
                break

        return AutoTuneResult(
            log_g=self.log_g,
            visit_counts=history,
            collection_counts=collection_history if method_name == "transition_matrix" else None,
        )

    def run_production(self, *, n_steps: int, block_size: int | None = None,
                       reference_ensemble: int = 0,
                       require_histogram: bool = True,
                       require_collection: bool = True) -> ExpandedProductionResult:
        if not self._masks:
            raise ValueError("set_ensemble_ladder must be called before run_production")
        if self._log_g is None:
            raise ValueError("log_g must be initialized before run_production")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")

        self.engine.reset_transition_counts()
        self.engine.reset_collection_counts()
        if block_size is None:
            self.engine.reset_visit_counts_ext()
            self.engine.run_steps(int(n_steps))
            counts = self.engine.get_visit_counts_ext()
            transitions = self.engine.get_transition_counts()
            collection = self.engine.get_collection_counts()
            log_z, log_z_err = _jackknife_log_z_or_nan(
                counts.reshape(1, -1),
                self._log_g,
                reference_ensemble=reference_ensemble,
                require_histogram=bool(require_histogram),
            )
            collection_log_z, collection_log_z_err = _jackknife_log_z_from_collection_or_nan(
                collection.reshape(1, collection.shape[0], collection.shape[1]),
                self._log_g,
                reference_ensemble=reference_ensemble,
                require_collection=bool(require_collection),
            )
            return ExpandedProductionResult(
                log_g=self.log_g,
                visit_counts=counts.copy(),
                block_visit_counts=None,
                transition_counts=transitions.copy(),
                collection_counts=collection.copy(),
                log_z=log_z,
                log_z_err=log_z_err,
                block_collection_counts=None,
                collection_log_z=collection_log_z,
                collection_log_z_err=collection_log_z_err,
            )

        if block_size <= 0:
            raise ValueError("block_size must be positive")

        blocks = []
        total_transition = np.zeros((len(self._masks), len(self._masks)), dtype=np.int64)
        total_collection = np.zeros((len(self._masks), len(self._masks)), dtype=np.float64)
        collection_blocks = []
        remaining = int(n_steps)
        while remaining > 0:
            current_block = min(int(block_size), remaining)
            self.engine.reset_visit_counts_ext()
            self.engine.reset_transition_counts()
            self.engine.reset_collection_counts()
            self.engine.run_steps(current_block)
            blocks.append(self.engine.get_visit_counts_ext())
            total_transition += self.engine.get_transition_counts()
            block_collection = self.engine.get_collection_counts()
            total_collection += block_collection
            collection_blocks.append(block_collection)
            remaining -= current_block

        block_arr = np.asarray(blocks, dtype=np.int64)
        collection_block_arr = np.asarray(collection_blocks, dtype=np.float64)
        counts = np.sum(block_arr, axis=0, dtype=np.int64)
        log_z, log_z_err = _jackknife_log_z_or_nan(
            block_arr,
            self._log_g,
            reference_ensemble=reference_ensemble,
            require_histogram=bool(require_histogram),
        )
        collection_log_z, collection_log_z_err = _jackknife_log_z_from_collection_or_nan(
            collection_block_arr,
            self._log_g,
            reference_ensemble=reference_ensemble,
            require_collection=bool(require_collection),
        )
        return ExpandedProductionResult(
            log_g=self.log_g,
            visit_counts=counts,
            block_visit_counts=block_arr,
            transition_counts=total_transition,
            collection_counts=total_collection,
            log_z=log_z,
            log_z_err=log_z_err,
            block_collection_counts=collection_block_arr,
            collection_log_z=collection_log_z,
            collection_log_z_err=collection_log_z_err,
        )

    def run_until_target_error(self, *, target_ensemble: int, target_s2_err: float,
                               batch_steps: int, block_size: int,
                               max_steps: int, reference_ensemble: int = 0,
                               min_steps: int = 0,
                               estimator: str = "histogram") -> ExpandedProductionResult:
        if target_s2_err <= 0.0:
            raise ValueError("target_s2_err must be positive")
        if batch_steps <= 0:
            raise ValueError("batch_steps must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if batch_steps % block_size != 0:
            raise ValueError("batch_steps must be divisible by block_size")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if min_steps < 0:
            raise ValueError("min_steps must be non-negative")
        estimator_name = str(estimator).strip().lower()
        if estimator_name not in {"histogram", "collection"}:
            raise ValueError("estimator must be 'histogram' or 'collection'")

        if not self._masks:
            raise ValueError("set_ensemble_ladder must be called before run_until_target_error")
        if self._log_g is None:
            raise ValueError("log_g must be initialized before run_until_target_error")

        target_idx = int(target_ensemble)
        ref_idx = int(reference_ensemble)
        block_records: list[np.ndarray] = []
        total_transition = np.zeros((len(self._masks), len(self._masks)), dtype=np.int64)
        total_collection = np.zeros((len(self._masks), len(self._masks)), dtype=np.float64)
        collection_blocks: list[np.ndarray] = []
        steps_done = 0

        while steps_done < int(max_steps):
            current_batch = min(int(batch_steps), int(max_steps) - steps_done)
            remaining = current_batch
            while remaining > 0:
                current_block = min(int(block_size), remaining)
                self.engine.reset_visit_counts_ext()
                self.engine.reset_transition_counts()
                self.engine.reset_collection_counts()
                self.engine.run_steps(current_block)
                block_records.append(self.engine.get_visit_counts_ext())
                total_transition += self.engine.get_transition_counts()
                block_collection = self.engine.get_collection_counts()
                total_collection += block_collection
                collection_blocks.append(block_collection)
                remaining -= current_block
                steps_done += current_block

            block_arr = np.asarray(block_records, dtype=np.int64)
            collection_block_arr = np.asarray(collection_blocks, dtype=np.float64)
            counts = np.sum(block_arr, axis=0, dtype=np.int64)
            log_z, log_z_err = _jackknife_log_z(block_arr, self._log_g, reference_ensemble=ref_idx)
            collection_log_z, collection_log_z_err = _jackknife_log_z_from_collection(
                collection_block_arr,
                self._log_g,
                reference_ensemble=ref_idx,
            )
            if estimator_name == "histogram":
                current_err = float(np.sqrt(log_z_err[target_idx] ** 2 + log_z_err[ref_idx] ** 2))
            else:
                current_err = float(np.sqrt(
                    collection_log_z_err[target_idx] ** 2 +
                    collection_log_z_err[ref_idx] ** 2
                ))
            if steps_done >= int(min_steps) and current_err <= float(target_s2_err):
                return ExpandedProductionResult(
                    log_g=self.log_g,
                    visit_counts=counts,
                    block_visit_counts=block_arr,
                    transition_counts=total_transition,
                    collection_counts=total_collection,
                    log_z=log_z,
                    log_z_err=log_z_err,
                    block_collection_counts=collection_block_arr,
                    collection_log_z=collection_log_z,
                    collection_log_z_err=collection_log_z_err,
                )

        block_arr = np.asarray(block_records, dtype=np.int64)
        collection_block_arr = np.asarray(collection_blocks, dtype=np.float64)
        counts = np.sum(block_arr, axis=0, dtype=np.int64)
        log_z, log_z_err = _jackknife_log_z(block_arr, self._log_g, reference_ensemble=ref_idx)
        collection_log_z, collection_log_z_err = _jackknife_log_z_from_collection(
            collection_block_arr,
            self._log_g,
            reference_ensemble=ref_idx,
        )
        return ExpandedProductionResult(
            log_g=self.log_g,
            visit_counts=counts,
            block_visit_counts=block_arr,
            transition_counts=total_transition,
            collection_counts=total_collection,
            log_z=log_z,
            log_z_err=log_z_err,
            block_collection_counts=collection_block_arr,
            collection_log_z=collection_log_z,
            collection_log_z_err=collection_log_z_err,
        )


def combine_expanded_production_results(results, *, reference_ensemble: int = 0,
                                        require_histogram: bool = True,
                                        require_collection: bool = True) -> ExpandedProductionResult:
    result_list = list(results)
    if not result_list:
        raise ValueError("results must be non-empty")

    first = result_list[0]
    log_g = np.asarray(first.log_g, dtype=np.float64).reshape(-1)
    n_ens = log_g.size

    def _same_optional_shape(value, *, square: bool = False) -> bool:
        if value is None:
            return True
        arr = np.asarray(value)
        if square:
            return arr.shape == (n_ens, n_ens)
        return arr.shape == (n_ens,)

    for item in result_list[1:]:
        item_log_g = np.asarray(item.log_g, dtype=np.float64).reshape(-1)
        if item_log_g.size != n_ens or not np.allclose(item_log_g, log_g):
            raise ValueError("all results must share the same log_g")
        if not _same_optional_shape(item.visit_counts):
            raise ValueError("visit_counts shape mismatch")
        if not _same_optional_shape(item.collection_counts, square=True):
            raise ValueError("collection_counts shape mismatch")
        if not _same_optional_shape(item.transition_counts, square=True):
            raise ValueError("transition_counts shape mismatch")

    total_visit = np.sum(
        [np.asarray(item.visit_counts, dtype=np.int64).reshape(-1) for item in result_list],
        axis=0,
        dtype=np.int64,
    )
    total_transition = np.sum(
        [np.asarray(item.transition_counts, dtype=np.int64).reshape(n_ens, n_ens) for item in result_list],
        axis=0,
        dtype=np.int64,
    )

    total_collection = None
    if all(item.collection_counts is not None for item in result_list):
        total_collection = np.sum(
            [np.asarray(item.collection_counts, dtype=np.float64).reshape(n_ens, n_ens) for item in result_list],
            axis=0,
            dtype=np.float64,
        )

    block_visit = None
    if all(item.block_visit_counts is not None for item in result_list):
        block_visit = np.concatenate(
            [np.asarray(item.block_visit_counts, dtype=np.int64).reshape(-1, n_ens) for item in result_list],
            axis=0,
        )
    elif any(item.block_visit_counts is not None for item in result_list):
        raise ValueError("either all or none of the results must carry block_visit_counts")

    block_collection = None
    if all(item.block_collection_counts is not None for item in result_list):
        block_collection = np.concatenate(
            [np.asarray(item.block_collection_counts, dtype=np.float64).reshape(-1, n_ens, n_ens)
             for item in result_list],
            axis=0,
        )
    elif any(item.block_collection_counts is not None for item in result_list):
        raise ValueError("either all or none of the results must carry block_collection_counts")

    if block_visit is not None:
        log_z, log_z_err = _jackknife_log_z_or_nan(
            block_visit,
            log_g,
            reference_ensemble=reference_ensemble,
            require_histogram=bool(require_histogram),
        )
    else:
        log_z, log_z_err = _jackknife_log_z_or_nan(
            total_visit.reshape(1, -1),
            log_g,
            reference_ensemble=reference_ensemble,
            require_histogram=bool(require_histogram),
        )

    collection_log_z = None
    collection_log_z_err = None
    if total_collection is not None:
        if block_collection is not None:
            collection_log_z, collection_log_z_err = _jackknife_log_z_from_collection_or_nan(
                block_collection,
                log_g,
                reference_ensemble=reference_ensemble,
                require_collection=bool(require_collection),
            )
        else:
            collection_log_z, collection_log_z_err = _jackknife_log_z_from_collection_or_nan(
                total_collection.reshape(1, n_ens, n_ens),
                log_g,
                reference_ensemble=reference_ensemble,
                require_collection=bool(require_collection),
            )

    return ExpandedProductionResult(
        log_g=log_g.copy(),
        visit_counts=total_visit,
        block_visit_counts=block_visit,
        transition_counts=total_transition,
        collection_counts=total_collection,
        log_z=log_z,
        log_z_err=log_z_err,
        block_collection_counts=block_collection,
        collection_log_z=collection_log_z,
        collection_log_z_err=collection_log_z_err,
    )


def _write_optional_dataset(group, name: str, value):
    if value is None:
        return
    group.create_dataset(name, data=np.asarray(value))


def _read_optional_dataset(group, name: str, *, dtype=None):
    if name not in group:
        return None
    data = np.asarray(group[name])
    if dtype is not None:
        data = data.astype(dtype)
    return data


def save_expanded_result_hdf5(filepath, *, masks, neighbors, result: ExpandedProductionResult,
                              auto_tune: AutoTuneResult | None = None,
                              params: dict | None = None,
                              pos=None):
    path = Path(filepath)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    mask_arr = np.asarray(masks, dtype=np.uint8)
    neighbor_counts = np.asarray([len(row) for row in neighbors], dtype=np.int32)
    neighbor_flat = np.asarray([item for row in neighbors for item in row], dtype=np.int32)

    with h5py.File(path, "w") as h5:
        manifest = h5.create_group("manifest")
        manifest.create_dataset("masks", data=mask_arr)
        manifest.create_dataset("neighbor_counts", data=neighbor_counts)
        manifest.create_dataset("neighbor_flat", data=neighbor_flat)
        if params:
            for key, value in params.items():
                manifest.attrs[str(key)] = value
        if pos is not None:
            manifest.create_dataset("pos", data=np.asarray(pos, dtype=np.float64))

        result_group = h5.create_group("result")
        result_group.create_dataset("log_g", data=np.asarray(result.log_g, dtype=np.float64))
        result_group.create_dataset("visit_counts", data=np.asarray(result.visit_counts, dtype=np.int64))
        result_group.create_dataset("transition_counts", data=np.asarray(result.transition_counts, dtype=np.int64))
        result_group.create_dataset("log_z", data=np.asarray(result.log_z, dtype=np.float64))
        result_group.create_dataset("log_z_err", data=np.asarray(result.log_z_err, dtype=np.float64))
        _write_optional_dataset(result_group, "block_visit_counts", result.block_visit_counts)
        _write_optional_dataset(result_group, "collection_counts", result.collection_counts)
        _write_optional_dataset(result_group, "block_collection_counts", result.block_collection_counts)
        _write_optional_dataset(result_group, "collection_log_z", result.collection_log_z)
        _write_optional_dataset(result_group, "collection_log_z_err", result.collection_log_z_err)

        if auto_tune is not None:
            auto_group = h5.create_group("auto_tune")
            auto_group.create_dataset("log_g", data=np.asarray(auto_tune.log_g, dtype=np.float64))
            if auto_tune.visit_counts:
                auto_group.create_dataset(
                    "visit_counts",
                    data=np.asarray(auto_tune.visit_counts, dtype=np.int64),
                )
            if auto_tune.collection_counts:
                auto_group.create_dataset(
                    "collection_counts",
                    data=np.asarray(auto_tune.collection_counts, dtype=np.float64),
                )


def load_expanded_result_hdf5(filepath) -> dict:
    with h5py.File(filepath, "r") as h5:
        manifest = h5["manifest"]
        masks = np.asarray(manifest["masks"], dtype=np.uint8)
        neighbor_counts = np.asarray(manifest["neighbor_counts"], dtype=np.int32)
        neighbor_flat = np.asarray(manifest["neighbor_flat"], dtype=np.int32)
        neighbors = []
        offset = 0
        for count in neighbor_counts:
            count_int = int(count)
            neighbors.append(neighbor_flat[offset:offset + count_int].astype(np.int32).tolist())
            offset += count_int

        result_group = h5["result"]
        result = ExpandedProductionResult(
            log_g=np.asarray(result_group["log_g"], dtype=np.float64),
            visit_counts=np.asarray(result_group["visit_counts"], dtype=np.int64),
            block_visit_counts=_read_optional_dataset(result_group, "block_visit_counts", dtype=np.int64),
            transition_counts=np.asarray(result_group["transition_counts"], dtype=np.int64),
            collection_counts=_read_optional_dataset(result_group, "collection_counts", dtype=np.float64),
            log_z=np.asarray(result_group["log_z"], dtype=np.float64),
            log_z_err=np.asarray(result_group["log_z_err"], dtype=np.float64),
            block_collection_counts=_read_optional_dataset(result_group, "block_collection_counts", dtype=np.float64),
            collection_log_z=_read_optional_dataset(result_group, "collection_log_z", dtype=np.float64),
            collection_log_z_err=_read_optional_dataset(result_group, "collection_log_z_err", dtype=np.float64),
        )

        auto = None
        if "auto_tune" in h5:
            auto_group = h5["auto_tune"]
            visit_history = _read_optional_dataset(auto_group, "visit_counts", dtype=np.int64)
            collection_history = _read_optional_dataset(auto_group, "collection_counts", dtype=np.float64)
            auto = AutoTuneResult(
                log_g=np.asarray(auto_group["log_g"], dtype=np.float64),
                visit_counts=[] if visit_history is None else [np.asarray(item, dtype=np.int64) for item in visit_history],
                collection_counts=None if collection_history is None else [
                    np.asarray(item, dtype=np.float64) for item in collection_history
                ],
            )

        return {
            "masks": masks,
            "neighbors": neighbors,
            "params": dict(manifest.attrs),
            "pos": _read_optional_dataset(manifest, "pos", dtype=np.float64),
            "result": result,
            "auto_tune": auto,
        }
