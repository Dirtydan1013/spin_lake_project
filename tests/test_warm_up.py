"""Tests for ``ReweightingDriver.warm_up``.

Contract:

- ``warm_up(n_steps)`` thermalises the underlying MC chain by calling
  ``engine.run_steps(n_steps)``.
- It MUST NOT change ``self._log_g``.
- It MUST leave ``visit_counts_ext``, ``transition_counts``, and
  ``collection_counts`` at zero (resets before AND after the run).
- It is allowed only after ``set_ensemble_ladder``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.tee.reweighting import ReweightingDriver


def _make_driver_with_mock_engine():
    engine = MagicMock()
    # stub the methods ReweightingDriver actually invokes during set_ladder
    engine.set_ensemble_ladder = MagicMock()
    engine.set_log_g = MagicMock()
    engine.reset_visit_counts_ext = MagicMock()
    engine.reset_transition_counts = MagicMock()
    engine.reset_collection_counts = MagicMock()
    engine.run_steps = MagicMock()
    driver = ReweightingDriver(engine=engine)
    masks = [np.array([0, 0], dtype=np.uint8), np.array([1, 0], dtype=np.uint8)]
    neighbors = [[1], [0]]
    driver.set_ensemble_ladder(masks, neighbors)
    # Reset call counts AFTER set_ensemble_ladder so test assertions only
    # see what warm_up itself does.
    engine.reset_visit_counts_ext.reset_mock()
    engine.reset_transition_counts.reset_mock()
    engine.reset_collection_counts.reset_mock()
    engine.run_steps.reset_mock()
    return driver, engine


class TestWarmUpExists:
    def test_warm_up_method_present(self):
        driver, _ = _make_driver_with_mock_engine()
        assert hasattr(driver, "warm_up") and callable(driver.warm_up)


class TestWarmUpBehaviour:
    def test_warm_up_calls_run_steps_with_argument(self):
        driver, engine = _make_driver_with_mock_engine()
        driver.warm_up(1234)
        engine.run_steps.assert_called_once_with(1234)

    def test_warm_up_does_not_modify_log_g(self):
        driver, _ = _make_driver_with_mock_engine()
        driver.set_log_g(np.array([0.0, 1.5]))
        before = driver.log_g.copy()
        driver.warm_up(100)
        np.testing.assert_array_equal(driver.log_g, before)

    def test_warm_up_resets_counters_before_and_after(self):
        """Counters must be zeroed both pre- and post- the chain run, so the
        production run that follows starts from a clean slate."""
        driver, engine = _make_driver_with_mock_engine()
        driver.warm_up(50)
        # exact policy: at least called once before and once after
        assert engine.reset_visit_counts_ext.call_count >= 2
        assert engine.reset_transition_counts.call_count >= 2
        assert engine.reset_collection_counts.call_count >= 2

    def test_warm_up_zero_steps_is_noop_run_but_still_resets(self):
        driver, engine = _make_driver_with_mock_engine()
        driver.warm_up(0)
        # We still want counters reset so the API is unconditional.
        # run_steps may be called with 0 or skipped — both acceptable.
        assert engine.reset_visit_counts_ext.call_count >= 1


class TestWarmUpRequiresLadder:
    def test_warm_up_before_set_ladder_raises(self):
        engine = MagicMock()
        driver = ReweightingDriver(engine=engine)
        with pytest.raises(ValueError, match="set_ensemble_ladder"):
            driver.warm_up(100)
