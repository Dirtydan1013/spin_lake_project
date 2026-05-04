import os
import sys

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.mpi.reweighting_mpi import run_expanded_mpi


class _FakeExpandedMPIEngine:
    def __init__(self):
        self._log_g = np.zeros(3, dtype=np.float64)
        self._counts = np.zeros(3, dtype=np.int64)
        self._transitions = np.zeros((3, 3), dtype=np.int64)
        self._collection = np.zeros((3, 3), dtype=np.float64)
        self._call_index = 0

    def set_ensemble_ladder(self, masks, neighbors, initial_ensemble=0):
        self.masks = [np.array(mask, dtype=np.uint8) for mask in masks]
        self.neighbors = [list(row) for row in neighbors]
        self.initial_ensemble = int(initial_ensemble)

    def set_log_g(self, log_g):
        self._log_g = np.array(log_g, dtype=np.float64)

    def reset_visit_counts_ext(self):
        self._counts = np.zeros(3, dtype=np.int64)

    def reset_transition_counts(self):
        self._transitions = np.zeros((3, 3), dtype=np.int64)

    def reset_collection_counts(self):
        self._collection = np.zeros((3, 3), dtype=np.float64)

    def run_steps(self, n_steps):
        self.n_steps = int(n_steps)
        self._call_index += 1
        if self._call_index == 1:
            self._counts = np.array([40, 30, 20], dtype=np.int64)
            self._collection = np.array(
                [[70.0, 30.0, 0.0], [50.0, 10.0, 40.0], [0.0, 60.0, 40.0]],
                dtype=np.float64,
            )
        else:
            self._counts = np.array([25, 30, 35], dtype=np.int64)
            self._transitions = np.array([[0, 4, 0], [3, 0, 2], [0, 1, 0]], dtype=np.int64)
            self._collection = np.array(
                [[35.0, 15.0, 0.0], [20.0, 6.0, 24.0], [0.0, 18.0, 12.0]],
                dtype=np.float64,
            )

    def get_visit_counts_ext(self):
        return self._counts.copy()

    def get_transition_counts(self):
        return self._transitions.copy()

    def get_collection_counts(self):
        return self._collection.copy()


def _test_single_rank_expanded_mpi_driver_with_fake_engine():
    masks = [
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([1, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0, 2], [1]]
    result = run_expanded_mpi(
        N=3,
        M=4,
        masks=masks,
        neighbors=neighbors,
        target_ensemble=2,
        autotune_steps_per_iter=100,
        autotune_max_iters=1,
        autotune_tol=1.01,
        autotune_method="transition_matrix",
        n_steps=90,
        block_size=30,
        comm=MPI.COMM_SELF,
        engine_factory=lambda rank: _FakeExpandedMPIEngine(),
    )

    assert result.auto_tune.collection_counts is not None
    expected_target = np.array([0.0, -np.log(0.6), -np.log(0.4)], dtype=np.float64)
    expected_log_g = 0.7 * expected_target
    assert np.allclose(result.auto_tune.log_g, expected_log_g, atol=1e-8)
    assert result.production.visit_counts.tolist() == [75, 90, 105]
    assert result.production.transition_counts.tolist() == [[0, 12, 0], [9, 0, 6], [0, 3, 0]]
    assert result.production.collection_counts.shape == (3, 3)
    s2_hist, err_hist = result.production.s2(2)
    s2_coll, err_coll = result.production.s2_collection(2)
    assert np.isfinite(s2_hist)
    assert np.isfinite(s2_coll)
    assert err_hist >= 0.0
    assert err_coll >= 0.0


def main():
    _test_single_rank_expanded_mpi_driver_with_fake_engine()
    print("Expanded reweighting MPI unit checks passed")


if __name__ == "__main__":
    main()
