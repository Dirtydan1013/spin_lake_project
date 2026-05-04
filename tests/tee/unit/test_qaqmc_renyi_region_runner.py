import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tee.qaqmc_renyi_ratio import (
    RatioResult,
    RegionRatioRunner,
    load_ratio_result_hdf5,
    save_ratio_result_hdf5,
)


class _FakeEngine:
    def __init__(self, ratio):
        self.ratio = float(ratio)
        self.calls = []

    def set_topology_pair(self, lower_mask, upper_mask, diff_site):
        self.calls.append((
            "pair",
            np.array(lower_mask, dtype=np.uint8),
            np.array(upper_mask, dtype=np.uint8),
            int(diff_site),
        ))

    def get_visit_counts(self):
        low = 100
        high = int(round(self.ratio * low))
        self.calls.append(("counts", low, high))
        return np.array([low, high], dtype=np.int64)

    def run_steps(self, n_steps):
        self.calls.append(("run", int(n_steps)))

    def reset_visit_counts(self):
        self.calls.append(("reset", None))


def _test_region_runner_builds_nested_masks():
    ratios = [0.5, 0.25, 0.8]
    engines = []

    def factory(step_index):
        engine = _FakeEngine(ratios[step_index])
        engines.append(engine)
        return engine

    runner = RegionRatioRunner(engine_factory=factory)
    result = runner.run_region(
        "A",
        region_mask=np.array([0, 1, 1, 1, 0], dtype=np.uint8),
        site_order=[1, 2, 3],
        n_therm=2,
        n_measure=4,
    )

    assert np.isclose(result.summary.S_2, -(np.log(0.5) + np.log(0.25) + np.log(0.8)))
    seen_pairs = [(engine.calls[0][1].tolist(), engine.calls[0][2].tolist(), engine.calls[0][3]) for engine in engines]
    assert seen_pairs == [
        ([0, 0, 0, 0, 0], [0, 1, 0, 0, 0], 1),
        ([0, 1, 0, 0, 0], [0, 1, 1, 0, 0], 2),
        ([0, 1, 1, 0, 0], [0, 1, 1, 1, 0], 3),
    ]
    seen_lower_masks = [pair[0] for pair in seen_pairs]
    assert seen_lower_masks == [
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
    ]


def _test_ratio_hdf5_roundtrip():
    ratio_result = RatioResult(
        ratio=0.4,
        ratio_err=0.03,
        visit_count_low=100,
        visit_count_high=40,
        block_visit_count_low=np.array([40, 60], dtype=np.int64),
        block_visit_count_high=np.array([10, 30], dtype=np.int64),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ratio.h5")
        save_ratio_result_hdf5(
            path,
            region_name="AB",
            step_index=2,
            A_mask=np.array([1, 1, 0, 0], dtype=np.uint8),
            next_site=3,
            result=ratio_result,
            params={"n_measure": 100},
        )
        loaded = load_ratio_result_hdf5(path)
        assert loaded["params"]["region_name"] == "AB"
        assert int(loaded["params"]["step_index"]) == 2
        assert loaded["A_mask"].tolist() == [1, 1, 0, 0]
        assert loaded["next_site"] == 3
        assert np.isclose(loaded["result"].ratio, 0.4)
        assert loaded["result"].visit_count_low == 100
        assert loaded["result"].visit_count_high == 40
        assert loaded["result"].block_visit_count_low.tolist() == [40, 60]
        assert loaded["result"].block_visit_count_high.tolist() == [10, 30]


def main():
    _test_region_runner_builds_nested_masks()
    _test_ratio_hdf5_roundtrip()
    print("QAQMC Renyi region runner unit checks passed")


if __name__ == "__main__":
    main()
