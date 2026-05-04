import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tee.reweighting import (
    ReweightingDriver,
    _normalized_log_z_from_collection,
    combine_expanded_production_results,
    load_expanded_result_hdf5,
    save_expanded_result_hdf5,
)


class _FakeExpandedEngine:
    def __init__(self, count_sequence, transition_sequence=None, collection_sequence=None):
        self.count_sequence = [np.asarray(item, dtype=np.int64) for item in count_sequence]
        if transition_sequence is None:
            transition_sequence = [
                np.zeros((self.count_sequence[0].size, self.count_sequence[0].size), dtype=np.int64)
                for _ in self.count_sequence
            ]
        self.transition_sequence = [np.asarray(item, dtype=np.int64) for item in transition_sequence]
        if collection_sequence is None:
            collection_sequence = [
                np.zeros((self.count_sequence[0].size, self.count_sequence[0].size), dtype=np.float64)
                for _ in self.count_sequence
            ]
        self.collection_sequence = [np.asarray(item, dtype=np.float64) for item in collection_sequence]
        self._call_index = 0
        self._counts = np.zeros_like(self.count_sequence[0])
        self._transitions = np.zeros_like(self.transition_sequence[0])
        self._collection = np.zeros_like(self.collection_sequence[0])

    def set_ensemble_ladder(self, masks, neighbors, initial_ensemble=0):
        self.masks = [np.array(mask, dtype=np.uint8) for mask in masks]
        self.neighbors = [list(row) for row in neighbors]
        self.initial_ensemble = int(initial_ensemble)

    def set_log_g(self, log_g):
        self.log_g = np.array(log_g, dtype=np.float64)

    def reset_visit_counts_ext(self):
        self._counts = np.zeros_like(self._counts)

    def reset_transition_counts(self):
        self._transitions = np.zeros_like(self._transitions)

    def reset_collection_counts(self):
        self._collection = np.zeros_like(self._collection)

    def run_steps(self, n_steps):
        self.n_steps = int(n_steps)
        self._counts = self.count_sequence[self._call_index].copy()
        self._transitions = self.transition_sequence[self._call_index].copy()
        self._collection = self.collection_sequence[self._call_index].copy()
        self._call_index += 1

    def get_visit_counts_ext(self):
        return self._counts.copy()

    def get_transition_counts(self):
        return self._transitions.copy()

    def get_collection_counts(self):
        return self._collection.copy()


def _test_auto_tune_updates_log_g():
    fake = _FakeExpandedEngine([
        [20, 10, 5],
        [11, 10, 9],
    ])
    driver = ReweightingDriver(engine=fake)
    masks = [
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([1, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0, 2], [1]]
    driver.set_ensemble_ladder(masks, neighbors)
    result = driver.auto_tune(n_steps_per_iter=100, max_iters=3, tol=1.25)

    assert len(result.visit_counts) == 2
    assert np.allclose(result.log_g[0], 0.0)
    assert result.log_g[2] > result.log_g[1] > result.log_g[0]


def _test_run_production_block_jackknife():
    fake = _FakeExpandedEngine(
        count_sequence=[
            [30, 10, 5],
            [20, 20, 10],
            [25, 15, 10],
        ],
        transition_sequence=[
            np.array([[0, 2, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int64),
            np.array([[0, 1, 0], [1, 0, 2], [0, 0, 0]], dtype=np.int64),
            np.array([[0, 3, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int64),
        ],
        collection_sequence=[
            np.array([[7.0, 3.0, 0.0], [5.0, 1.0, 4.0], [0.0, 6.0, 4.0]], dtype=np.float64),
            np.array([[7.0, 3.0, 0.0], [5.0, 1.0, 4.0], [0.0, 6.0, 4.0]], dtype=np.float64),
            np.array([[7.0, 3.0, 0.0], [5.0, 1.0, 4.0], [0.0, 6.0, 4.0]], dtype=np.float64),
        ],
    )
    driver = ReweightingDriver(engine=fake)
    masks = [
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([1, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0, 2], [1]]
    driver.set_ensemble_ladder(masks, neighbors)
    driver.set_log_g(np.zeros(3, dtype=np.float64))
    result = driver.run_production(n_steps=30, block_size=10)

    assert result.visit_counts.tolist() == [75, 45, 25]
    assert result.block_visit_counts.shape == (3, 3)
    assert result.block_collection_counts.shape == (3, 3, 3)
    assert np.isclose(result.log_z[0], 0.0)
    assert result.log_z_err[1] > 0.0
    assert result.log_z_err[2] > 0.0
    assert result.transition_counts.tolist() == [[0, 6, 0], [1, 0, 4], [0, 0, 0]]
    assert result.collection_counts.shape == (3, 3)
    s2_collection, s2_collection_err = result.s2_collection(2)
    assert np.isfinite(s2_collection)
    assert s2_collection_err >= 0.0


def _test_collection_matrix_recovers_log_z():
    collection = np.array(
        [
            [70.0, 30.0, 0.0],
            [50.0, 10.0, 40.0],
            [0.0, 60.0, 40.0],
        ],
        dtype=np.float64,
    )
    log_z = _normalized_log_z_from_collection(collection, np.zeros(3, dtype=np.float64))
    expected = np.array([0.0, np.log(0.6), np.log(0.4)], dtype=np.float64)
    assert np.allclose(log_z, expected, atol=1e-8)


def _test_transition_matrix_auto_tune_updates_log_g():
    fake = _FakeExpandedEngine(
        count_sequence=[[50, 30, 20]],
        collection_sequence=[
            np.array(
                [
                    [70.0, 30.0, 0.0],
                    [50.0, 10.0, 40.0],
                    [0.0, 60.0, 40.0],
                ],
                dtype=np.float64,
            )
        ],
    )
    driver = ReweightingDriver(engine=fake)
    masks = [
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([1, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0, 2], [1]]
    driver.set_ensemble_ladder(masks, neighbors)
    result = driver.auto_tune(
        n_steps_per_iter=100,
        max_iters=1,
        tol=1.01,
        method="transition_matrix",
    )

    assert len(result.visit_counts) == 1
    assert result.collection_counts is not None
    assert len(result.collection_counts) == 1
    expected = np.array([0.0, -np.log(0.6), -np.log(0.4)], dtype=np.float64)
    assert np.allclose(result.log_g, expected, atol=1e-8)


def _test_run_until_target_error_stops_after_enough_blocks():
    fake = _FakeExpandedEngine(
        count_sequence=[
            [20, 20, 20],
            [20, 20, 20],
            [20, 20, 20],
            [20, 20, 20],
        ],
        transition_sequence=[
            np.zeros((3, 3), dtype=np.int64)
            for _ in range(4)
        ],
        collection_sequence=[
            np.array([[7.0, 3.0, 0.0], [5.0, 1.0, 4.0], [0.0, 6.0, 4.0]], dtype=np.float64)
            for _ in range(4)
        ],
    )
    driver = ReweightingDriver(engine=fake)
    masks = [
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([1, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0, 2], [1]]
    driver.set_ensemble_ladder(masks, neighbors)
    driver.set_log_g(np.zeros(3, dtype=np.float64))
    result = driver.run_until_target_error(
        target_ensemble=2,
        target_s2_err=1e-6,
        batch_steps=20,
        block_size=10,
        max_steps=40,
        min_steps=20,
    )

    assert result.block_visit_counts.shape[0] >= 2
    s2, s2_err = result.s2(2)
    assert np.isfinite(s2)
    assert s2_err <= 1e-6


def _test_run_until_target_error_supports_collection_estimator():
    fake = _FakeExpandedEngine(
        count_sequence=[
            [20, 20, 20],
            [20, 20, 20],
            [20, 20, 20],
            [20, 20, 20],
        ],
        transition_sequence=[
            np.zeros((3, 3), dtype=np.int64)
            for _ in range(4)
        ],
        collection_sequence=[
            np.array([[7.0, 3.0, 0.0], [5.0, 1.0, 4.0], [0.0, 6.0, 4.0]], dtype=np.float64)
            for _ in range(4)
        ],
    )
    driver = ReweightingDriver(engine=fake)
    masks = [
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([1, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0, 2], [1]]
    driver.set_ensemble_ladder(masks, neighbors)
    driver.set_log_g(np.zeros(3, dtype=np.float64))
    result = driver.run_until_target_error(
        target_ensemble=2,
        target_s2_err=1e-6,
        batch_steps=20,
        block_size=10,
        max_steps=40,
        min_steps=20,
        estimator="collection",
    )

    assert result.block_collection_counts.shape[0] >= 2
    s2, s2_err = result.s2_collection(2)
    assert np.isfinite(s2)
    assert s2_err <= 1e-6


def _test_combine_expanded_production_results():
    fake0 = _FakeExpandedEngine(
        count_sequence=[[30, 20, 10]],
        transition_sequence=[np.array([[0, 2, 0], [1, 0, 1], [0, 0, 0]], dtype=np.int64)],
        collection_sequence=[np.array([[7.0, 3.0, 0.0], [5.0, 1.0, 4.0], [0.0, 6.0, 4.0]], dtype=np.float64)],
    )
    fake1 = _FakeExpandedEngine(
        count_sequence=[[15, 15, 20]],
        transition_sequence=[np.array([[0, 1, 0], [1, 0, 2], [0, 0, 0]], dtype=np.int64)],
        collection_sequence=[np.array([[3.5, 1.5, 0.0], [2.5, 0.5, 2.0], [0.0, 3.0, 2.0]], dtype=np.float64)],
    )
    masks = [
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([1, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0, 2], [1]]

    driver0 = ReweightingDriver(engine=fake0)
    driver0.set_ensemble_ladder(masks, neighbors)
    driver0.set_log_g(np.zeros(3, dtype=np.float64))
    result0 = driver0.run_production(n_steps=10, block_size=10)

    driver1 = ReweightingDriver(engine=fake1)
    driver1.set_ensemble_ladder(masks, neighbors)
    driver1.set_log_g(np.zeros(3, dtype=np.float64))
    result1 = driver1.run_production(n_steps=10, block_size=10)

    combined = combine_expanded_production_results([result0, result1])
    assert combined.visit_counts.tolist() == [45, 35, 30]
    assert combined.transition_counts.tolist() == [[0, 3, 0], [2, 0, 3], [0, 0, 0]]
    assert combined.collection_counts.shape == (3, 3)
    assert combined.block_visit_counts.shape == (2, 3)
    assert combined.block_collection_counts.shape == (2, 3, 3)
    s2_hist, s2_hist_err = combined.s2(2)
    s2_coll, s2_coll_err = combined.s2_collection(2)
    assert np.isfinite(s2_hist)
    assert np.isfinite(s2_coll)
    assert s2_hist_err >= 0.0
    assert s2_coll_err >= 0.0


def _test_expanded_result_hdf5_roundtrip():
    fake = _FakeExpandedEngine(
        count_sequence=[[30, 20, 10], [20, 20, 20]],
        transition_sequence=[
            np.array([[0, 2, 0], [1, 0, 1], [0, 0, 0]], dtype=np.int64),
            np.zeros((3, 3), dtype=np.int64),
        ],
        collection_sequence=[
            np.array([[7.0, 3.0, 0.0], [5.0, 1.0, 4.0], [0.0, 6.0, 4.0]], dtype=np.float64),
            np.array([[7.0, 3.0, 0.0], [5.0, 1.0, 4.0], [0.0, 6.0, 4.0]], dtype=np.float64),
        ],
    )
    driver = ReweightingDriver(engine=fake)
    masks = [
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([1, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0, 2], [1]]
    driver.set_ensemble_ladder(masks, neighbors)
    driver.set_log_g(np.zeros(3, dtype=np.float64))
    result = driver.run_production(n_steps=10, block_size=10)
    auto = driver.auto_tune(n_steps_per_iter=10, max_iters=1, tol=2.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "expanded.h5")
        save_expanded_result_hdf5(
            path,
            masks=masks,
            neighbors=neighbors,
            result=result,
            auto_tune=auto,
            params={"tag": "roundtrip"},
            pos=np.arange(3, dtype=np.float64).reshape(-1, 1),
        )
        loaded = load_expanded_result_hdf5(path)
        assert loaded["masks"].tolist() == [mask.tolist() for mask in masks]
        assert loaded["neighbors"] == neighbors
        assert loaded["params"]["tag"] == "roundtrip"
        assert loaded["pos"].shape == (3, 1)
        assert loaded["result"].visit_counts.tolist() == result.visit_counts.tolist()
        assert loaded["result"].collection_counts.shape == (3, 3)
        assert loaded["auto_tune"] is not None
        assert loaded["auto_tune"].visit_counts


def main():
    _test_auto_tune_updates_log_g()
    _test_run_production_block_jackknife()
    _test_collection_matrix_recovers_log_z()
    _test_transition_matrix_auto_tune_updates_log_g()
    _test_run_until_target_error_stops_after_enough_blocks()
    _test_run_until_target_error_supports_collection_estimator()
    _test_combine_expanded_production_results()
    _test_expanded_result_hdf5_roundtrip()
    print("Reweighting unit checks passed")


if __name__ == "__main__":
    main()
