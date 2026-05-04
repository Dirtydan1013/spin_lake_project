import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.engines.qaqmc_renyi import QAQMCRenyiRydberg
from src.tee.qaqmc_renyi_ratio import RatioRunner


def _test_runner_matches_manual_sequence():
    mask = np.zeros(6, dtype=np.uint8)
    mask[[1, 3]] = 1
    next_site = 2
    n_therm = 5
    n_measure = 12
    measure_stride = 3

    manual_engine = QAQMCRenyiRydberg(N=6, M=10, seed=19)
    next_mask = mask.copy()
    next_mask[next_site] = 1
    manual_engine.set_topology_pair(mask, next_mask, next_site)
    manual_engine.run_steps(n_therm)
    manual_engine.reset_visit_counts()
    manual_engine.run_steps(n_measure * measure_stride)
    manual_counts = manual_engine.get_visit_counts()
    manual_ratio = manual_counts[1] / manual_counts[0]

    runner_engine = QAQMCRenyiRydberg(N=6, M=10, seed=19)
    runner = RatioRunner(engine=runner_engine)
    result = runner.run_single_ratio(
        mask,
        next_site=next_site,
        n_therm=n_therm,
        n_measure=n_measure,
        measure_stride=measure_stride,
    )

    assert np.isclose(result.ratio, manual_ratio)
    assert result.visit_count_low + result.visit_count_high == n_measure * measure_stride
    assert result.ratio > 0.0


def _test_runner_blocking_matches_manual_blocks():
    mask = np.zeros(6, dtype=np.uint8)
    next_site = 2
    n_therm = 5
    n_measure = 12
    measure_stride = 2
    block_size = 4

    manual_engine = QAQMCRenyiRydberg(N=6, M=10, seed=23)
    next_mask = mask.copy()
    next_mask[next_site] = 1
    manual_engine.set_topology_pair(mask, next_mask, next_site)
    manual_engine.run_steps(n_therm)
    manual_lows = []
    manual_highs = []
    for _ in range(n_measure // block_size):
        manual_engine.reset_visit_counts()
        manual_engine.run_steps(block_size * measure_stride)
        block_counts = manual_engine.get_visit_counts()
        manual_lows.append(int(block_counts[0]))
        manual_highs.append(int(block_counts[1]))

    runner_engine = QAQMCRenyiRydberg(N=6, M=10, seed=23)
    runner = RatioRunner(engine=runner_engine)
    result = runner.run_single_ratio(
        mask,
        next_site=next_site,
        n_therm=n_therm,
        n_measure=n_measure,
        measure_stride=measure_stride,
        block_size=block_size,
    )

    assert result.block_visit_count_low.tolist() == manual_lows
    assert result.block_visit_count_high.tolist() == manual_highs


def main():
    _test_runner_matches_manual_sequence()
    _test_runner_blocking_matches_manual_blocks()
    print("QAQMC Renyi ratio integration checks passed")


if __name__ == "__main__":
    main()
