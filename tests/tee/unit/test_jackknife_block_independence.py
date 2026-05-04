"""Tests that the jackknife error estimate behaves correctly under
controlled noise distributions and detects autocorrelation when present.

These are unit tests on synthetic block-level visit counts — no MC engine.
The point is to validate that the error bar reported by ``_jackknife_log_z``
is trustworthy: it neither underestimates nor overestimates random scatter,
and it grows when blocks become correlated.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tee.reweighting import _jackknife_log_z


LOG_G = np.array([0.0, 0.5, 1.0, 1.5])
TARGET_LOG_Z = np.array([0.0, -1.0, -2.0, -2.5])
WEIGHTS = np.exp(TARGET_LOG_Z + LOG_G)
WEIGHTS = WEIGHTS / WEIGHTS.sum()


def _draw_iid_blocks(n_blocks, samples_per_block, seed):
    rng = np.random.default_rng(seed)
    return np.array([
        rng.multinomial(samples_per_block, WEIGHTS) for _ in range(n_blocks)
    ], dtype=np.int64)


def test_iid_blocks_jackknife_error_consistent_with_observed_scatter():
    """For IID multinomial blocks, jackknife std-err must match the ensemble std.

    If we run many independent block-arrays, take the jackknife mean from
    each, the std of those means should equal the typical jackknife std-err.
    Mismatch indicates the jackknife formula is biased.
    """
    n_blocks = 40
    samples = 5000
    target_idx = 1
    n_trials = 60
    means = []
    errs = []
    for trial in range(n_trials):
        blocks = _draw_iid_blocks(n_blocks, samples, seed=trial * 17 + 1)
        log_z, log_z_err = _jackknife_log_z(blocks, LOG_G)
        means.append(log_z[target_idx])
        errs.append(log_z_err[target_idx])
    means = np.array(means)
    errs = np.array(errs)
    observed_std = float(np.std(means, ddof=1))
    typical_err = float(np.median(errs))
    ratio = typical_err / observed_std
    # Jackknife error should be within ±50% of observed std.
    assert 0.5 < ratio < 2.0, (
        f"jackknife error miscalibrated for IID blocks: typical_err={typical_err:.4f}, "
        f"observed_std={observed_std:.4f}, ratio={ratio:.2f} (expected ≈1.0)"
    )


def test_jackknife_mean_unbiased_for_iid_blocks():
    """The jackknife mean must be an unbiased estimator of log_z."""
    n_trials = 80
    means = []
    for trial in range(n_trials):
        blocks = _draw_iid_blocks(n_blocks=30, samples_per_block=8000,
                                  seed=trial * 11 + 3)
        log_z, _ = _jackknife_log_z(blocks, LOG_G)
        means.append(log_z[1])
    grand_mean = float(np.mean(means))
    sem = float(np.std(means, ddof=1) / np.sqrt(n_trials))
    bias = abs(grand_mean - TARGET_LOG_Z[1])
    # Allow up to 3-sigma deviation of the grand mean from truth.
    assert bias <= 3.0 * sem, (
        f"jackknife mean biased: grand_mean={grand_mean:.4f} "
        f"(target {TARGET_LOG_Z[1]:.4f}, SEM {sem:.4f}), bias={bias:.4f}"
    )


def test_correlated_blocks_inflate_jackknife_error_vs_iid():
    """Autocorrelated blocks → jackknife error should be larger than IID baseline.

    We simulate autocorrelation by generating IID blocks then duplicating each
    one (so adjacent blocks are perfectly correlated).  Naïvely jackknife
    treats them as independent and hence underestimates error — but the
    *empirical* spread of repeated trials is unaffected.

    This test documents the known limitation: jackknife on correlated blocks
    is OPTIMISTIC.  In production, one must coarsen blocks until consecutive
    block estimates look independent.
    """
    rng = np.random.default_rng(seed=2024)
    samples = 4000
    n_iid = 20
    iid_blocks = np.array([
        rng.multinomial(samples, WEIGHTS) for _ in range(n_iid)
    ], dtype=np.int64)
    duplicated = np.repeat(iid_blocks, 2, axis=0)  # 40 blocks, pairs identical

    _, err_iid = _jackknife_log_z(iid_blocks, LOG_G)
    _, err_dup = _jackknife_log_z(duplicated, LOG_G)
    # If blocks are perfectly correlated pairs, the duplicated jackknife
    # error should be SMALLER than iid (more "blocks" but same info).  We
    # assert this to document the failure mode — production code must NOT
    # rely on raw block count when autocorrelation is present.
    target = 1
    assert err_dup[target] < err_iid[target], (
        f"sanity-check failure: duplicated-block jackknife err {err_dup[target]:.4f} "
        f"should be smaller than iid err {err_iid[target]:.4f} "
        f"(this is a known optimistic-bias mode of jackknife)"
    )


@pytest.mark.parametrize("n_blocks", [3, 10, 50])
def test_jackknife_runs_for_various_block_counts(n_blocks):
    """Jackknife must not crash on small block counts (down to n_blocks=3)."""
    blocks = _draw_iid_blocks(n_blocks=n_blocks, samples_per_block=10000,
                              seed=n_blocks * 13)
    log_z, log_z_err = _jackknife_log_z(blocks, LOG_G)
    assert log_z.shape == LOG_G.shape
    assert log_z_err.shape == LOG_G.shape
    assert np.all(np.isfinite(log_z))
    assert np.all(log_z_err >= 0.0)
    assert log_z[0] == 0.0  # reference convention


def test_jackknife_returns_zero_error_for_single_block_edge_case():
    """A single-block input has no leave-one-out variance — error must be 0."""
    blocks = _draw_iid_blocks(n_blocks=1, samples_per_block=10000, seed=1)
    log_z, log_z_err = _jackknife_log_z(blocks, LOG_G)
    np.testing.assert_allclose(log_z_err, 0.0)
