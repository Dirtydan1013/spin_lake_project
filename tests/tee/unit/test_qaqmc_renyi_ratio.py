import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tee.qaqmc_renyi_ratio import (
    RatioRunner,
    estimate_pair_ratio_error,
    estimate_pair_ratio_error_from_blocks,
)


class _FakeEngine:
    def __init__(self):
        self.calls = []
        self._visit_counts = np.array([12, 8], dtype=np.int64)

    def set_topology_pair(self, lower_mask, upper_mask, diff_site):
        self.calls.append(("pair", lower_mask.copy(), upper_mask.copy(), int(diff_site)))

    def get_visit_counts(self):
        self.calls.append(("counts", None))
        return self._visit_counts.copy()

    def run_steps(self, n_steps):
        self.calls.append(("run", int(n_steps)))

    def reset_visit_counts(self):
        self.calls.append(("reset", None))


class _BlockFakeEngine:
    def __init__(self, block_counts):
        self.block_counts = [tuple(map(int, pair)) for pair in block_counts]
        self.calls = []
        self._block_index = 0
        self._current = np.array([0, 0], dtype=np.int64)
        self._collecting = False

    def set_topology_pair(self, lower_mask, upper_mask, diff_site):
        self.calls.append(("pair", lower_mask.copy(), upper_mask.copy(), int(diff_site)))

    def get_visit_counts(self):
        self.calls.append(("counts", int(self._current[0]), int(self._current[1])))
        return self._current.copy()

    def run_steps(self, n_steps):
        self.calls.append(("run", int(n_steps)))
        if not self._collecting:
            return
        low, high = self.block_counts[self._block_index]
        self._current = np.array([low, high], dtype=np.int64)
        self._block_index += 1

    def reset_visit_counts(self):
        self.calls.append(("reset", None))
        self._current = np.array([0, 0], dtype=np.int64)
        self._collecting = True


def _test_pair_ratio_error_formula():
    err = estimate_pair_ratio_error(12, 8)
    expected = (8.0 / 12.0) * np.sqrt((1.0 / 12.0) + (1.0 / 8.0))
    assert np.isclose(err, expected)


def _test_ratio_runner_call_order():
    fake = _FakeEngine()
    runner = RatioRunner(engine=fake)
    mask = np.array([0, 1, 0], dtype=np.uint8)
    result = runner.run_single_ratio(mask, next_site=2, n_therm=5, n_measure=20, measure_stride=3)
    assert np.isclose(result.ratio, 8.0 / 12.0)
    assert result.visit_count_low == 12
    assert result.visit_count_high == 8
    assert fake.calls[0][0] == "pair"
    assert fake.calls[0][1].tolist() == [0, 1, 0]
    assert fake.calls[0][2].tolist() == [0, 1, 1]
    assert fake.calls[0][3] == 2
    assert [call[0] for call in fake.calls] == ["pair", "run", "reset", "run", "counts"]
    assert fake.calls[1] == ("run", 5)
    assert fake.calls[3] == ("run", 60)


def _test_block_ratio_error_exceeds_binomial_when_blocks_fluctuate():
    block_lows = [10, 10, 10, 10]
    block_highs = [2, 8, 2, 8]
    block_err = estimate_pair_ratio_error_from_blocks(block_lows, block_highs)
    naive_err = estimate_pair_ratio_error(sum(block_lows), sum(block_highs))
    assert block_err > naive_err


def _test_ratio_runner_collects_block_counts():
    fake = _BlockFakeEngine([(8, 2), (8, 6), (8, 2)])
    runner = RatioRunner(engine=fake)
    result = runner.run_single_ratio(
        np.array([0, 0, 0], dtype=np.uint8),
        next_site=1,
        n_therm=4,
        n_measure=9,
        measure_stride=2,
        block_size=3,
    )
    assert result.visit_count_low == 24
    assert result.visit_count_high == 10
    assert result.block_visit_count_low.tolist() == [8, 8, 8]
    assert result.block_visit_count_high.tolist() == [2, 6, 2]
    assert [call[0] for call in fake.calls] == [
        "pair", "run", "reset", "run", "counts", "reset", "run", "counts", "reset", "run", "counts"
    ]
    assert fake.calls[1] == ("run", 4)
    assert fake.calls[3] == ("run", 6)
    assert fake.calls[6] == ("run", 6)
    assert fake.calls[9] == ("run", 6)


def main():
    _test_pair_ratio_error_formula()
    _test_ratio_runner_call_order()
    _test_block_ratio_error_exceeds_binomial_when_blocks_fluctuate()
    _test_ratio_runner_collects_block_counts()
    print("QAQMC Renyi ratio unit checks passed")


if __name__ == "__main__":
    main()
