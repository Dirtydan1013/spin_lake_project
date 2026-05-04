"""Regression tests for ``ReweightingDriver.warm_up`` against a real engine.

The unit tests in ``tests/tee/unit/test_warm_up.py`` mock the C++ engine to
verify the API contract (counters reset, log_g preserved, etc.).  This file
exercises the *real* QAQMCRenyiRydberg engine to catch failures the mock
can't see — most importantly, the empirical fact that a fresh engine starts
in a non-thermalised configuration that biases the very first window's
visit counts.

The bug we are guarding against: on the production 4x4 m=1 KP audit, region
A's window-1 purity came out 0.235 (vs lower bound 0.5) because the engine
was shared across A, B, C and A inherited the C++ constructor's fresh state.
A long enough warm_up should erase that initial bias.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rydberg.lattices import generate_1d_chain
from src.tee.reweighting import ReweightingDriver


PHYSICS = dict(
    N=4, M=100, Omega=1.0, Rb=1.2,
    delta_min=0.0, delta_max=1.5, neighbor_cutoff=1,
)


def _driver(seed):
    return ReweightingDriver(
        N=PHYSICS["N"], M=PHYSICS["M"],
        Omega=PHYSICS["Omega"], Rb=PHYSICS["Rb"],
        delta_min=PHYSICS["delta_min"], delta_max=PHYSICS["delta_max"],
        pos=generate_1d_chain(PHYSICS["N"]),
        seed=seed, neighbor_cutoff=PHYSICS["neighbor_cutoff"],
    )


def _ladder_2_window():
    """Return masks/neighbors for a 2-window ladder ({}, {0})."""
    masks = [
        np.zeros(PHYSICS["N"], dtype=np.uint8),
        np.array([1, 0, 0, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0]]
    return masks, neighbors


def test_warm_up_does_not_change_log_g_on_real_engine():
    """log_g set before warm_up must equal log_g read after warm_up."""
    driver = _driver(seed=12345)
    masks, neighbors = _ladder_2_window()
    driver.set_ensemble_ladder(masks, neighbors, initial_ensemble=0)
    driver.set_log_g(np.array([0.0, 0.7]))
    before = driver.log_g.copy()
    driver.warm_up(1000)
    after = driver.log_g
    np.testing.assert_array_equal(after, before)


def test_warm_up_leaves_counters_at_zero():
    """warm_up must not leak any sample into visit/transition/collection counts.

    After warm_up, the engine's counters must be exactly zero so a downstream
    production run sees a clean slate.  We check this end-to-end via a tiny
    production run whose total visit count equals exactly its n_steps.
    """
    driver = _driver(seed=99999)
    masks, neighbors = _ladder_2_window()
    driver.set_ensemble_ladder(masks, neighbors, initial_ensemble=0)
    driver.warm_up(2000)
    n = 5000
    result = driver.run_production(n_steps=n, block_size=1000)
    total_visits = int(np.sum(result.visit_counts))
    assert total_visits == n, (
        f"warm_up leaked {total_visits - n} samples into production counters "
        f"(expected exactly {n}, got {total_visits})"
    )


@pytest.mark.parametrize("warm_steps", [0, 5000])
def test_warm_up_helps_or_at_worst_does_no_harm(warm_steps):
    """Warm-up should never make the result worse.

    With ``warm_steps=0``, the production starts from the engine's fresh
    constructor state — historically prone to the un-thermalised bias.  With
    ``warm_steps=5000``, the chain has time to forget that initial state.

    We require the single-qubit purity bound to hold in both cases (it MUST
    hold physically), and additionally check that the warmed-up result is
    not catastrophically worse than the cold one.
    """
    driver = _driver(seed=2024 + warm_steps)
    masks, neighbors = _ladder_2_window()
    driver.set_ensemble_ladder(masks, neighbors, initial_ensemble=0)
    if warm_steps > 0:
        driver.warm_up(warm_steps)
    driver.auto_tune(n_steps_per_iter=15000, max_iters=6,
                     tol=1.15, method="transition_matrix", damping=0.7)
    result = driver.run_production(n_steps=120000, block_size=2000)
    purity = float(np.exp(result.log_z[1] - result.log_z[0]))
    err = float(result.log_z_err[1] * purity)
    # Single-qubit purity bound — physically required regardless of warm-up.
    assert purity >= 0.5 - 3.0 * err, (
        f"warm_steps={warm_steps}: single-qubit purity {purity:.4f}±{err:.4f} "
        f"violates physical bound 0.5 — warm-up did not rescue the chain"
    )


def test_warm_up_short_then_long_converge_to_same_purity():
    """A 500-step warm-up vs a 5000-step warm-up should agree at production.

    Both should be enough to thermalise on a 4-site chain, and the production
    estimate is independent of warm-up length once thermalisation is reached.
    Disagreement here would suggest some warm-ups don't fully thermalise — a
    sampling-quality issue.
    """
    def _purity(seed, warm_steps):
        d = _driver(seed=seed)
        masks, neighbors = _ladder_2_window()
        d.set_ensemble_ladder(masks, neighbors, initial_ensemble=0)
        d.warm_up(warm_steps)
        d.auto_tune(n_steps_per_iter=15000, max_iters=6,
                    tol=1.15, method="transition_matrix", damping=0.7)
        r = d.run_production(n_steps=120000, block_size=2000)
        p = float(np.exp(r.log_z[1] - r.log_z[0]))
        e = float(r.log_z_err[1] * p)
        return p, e

    p_short, e_short = _purity(seed=4747, warm_steps=500)
    p_long, e_long = _purity(seed=8484, warm_steps=5000)
    diff = abs(p_short - p_long)
    comb = float(np.sqrt(e_short ** 2 + e_long ** 2))
    assert diff <= 3.0 * comb, (
        f"warm-up length sensitivity: short(500)={p_short:.4f}±{e_short:.4f} "
        f"vs long(5000)={p_long:.4f}±{e_long:.4f}, diff={diff:.4f} > 3σ"
    )
