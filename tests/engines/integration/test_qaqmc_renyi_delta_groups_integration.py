import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.engines.qaqmc_renyi import QAQMCRenyiRydberg
from src.tee.qaqmc_renyi_ratio import RatioRunner
from src.tee.reweighting import ReweightingDriver


def _test_ratio_runner_with_delta_groups():
    mask = np.zeros(6, dtype=np.uint8)
    mask[1] = 1
    next_site = 2

    engine = QAQMCRenyiRydberg(N=6, M=12, seed=31, delta_groups=4)
    assert engine.delta_groups == 4
    runner = RatioRunner(engine=engine)
    result = runner.run_single_ratio(
        mask,
        next_site=next_site,
        n_therm=8,
        n_measure=18,
        measure_stride=2,
        block_size=6,
    )

    assert result.visit_count_low + result.visit_count_high == 36
    assert result.block_visit_count_low.size == 3
    assert result.block_visit_count_high.size == 3
    assert np.isfinite(result.ratio)
    assert result.ratio >= 0.0


def _test_expanded_reweighting_with_delta_groups():
    masks = [
        np.array([0, 0, 0, 0, 0, 0], dtype=np.uint8),
        np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8),
        np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0, 2], [1]]

    engine = QAQMCRenyiRydberg(N=6, M=10, seed=37, delta_groups=5)
    driver = ReweightingDriver(engine=engine)
    driver.set_ensemble_ladder(masks, neighbors, initial_ensemble=0)
    driver.set_log_g(np.zeros(len(masks), dtype=np.float64))

    result = driver.run_production(n_steps=30, block_size=10)
    assert result.visit_counts.sum() == 30
    assert result.block_visit_counts.shape == (3, 3)
    assert result.transition_counts.shape == (3, 3)
    assert np.all(result.visit_counts >= 0)


def main():
    _test_ratio_runner_with_delta_groups()
    _test_expanded_reweighting_with_delta_groups()
    print("QAQMC Renyi delta-groups integration checks passed")


if __name__ == "__main__":
    main()
