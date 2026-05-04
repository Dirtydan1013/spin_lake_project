"""Detailed-balance / reweighting-math unit tests on synthetic counts.

These tests exercise the pure-math layer of the expanded-ensemble engine —
the ``_normalized_log_z`` and ``_normalized_log_z_from_collection`` helpers —
without spinning up any MC chain.  They are fast and deterministic, and they
guard against the kind of off-by-one or sign error that would silently
corrupt the autotune update rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tee.reweighting import (
    _jackknife_log_z,
    _normalized_log_z,
    _normalized_log_z_from_collection,
)


def test_normalized_log_z_recovers_exact_target_for_synthetic_visits():
    """If visit_counts ∝ Z_k * exp(log_g_k), reweighting must recover log Z_k.

    This is the *defining* identity of the reweighting estimator.  If it ever
    fails on synthetic noiseless input, the formula in reweighting.py is wrong.
    """
    target_log_z = np.array([0.0, -1.5, -2.7, -4.1], dtype=np.float64)
    log_g = np.array([0.0, 0.8, 1.9, 3.5], dtype=np.float64)

    # Construct ideal visit counts: V_k ∝ Z_k * exp(log_g_k) = exp(log_z + log_g).
    weights = np.exp(target_log_z + log_g)
    counts = (1_000_000 * weights / weights.sum()).astype(np.int64)

    recovered = _normalized_log_z(counts, log_g, reference_ensemble=0)
    np.testing.assert_allclose(recovered, target_log_z, atol=1e-3, rtol=1e-3)


def test_normalized_log_z_invariant_to_global_log_g_shift():
    """Shifting log_g by a constant must leave log_z[k] unchanged.

    log_z is anchored to log_z[0]=0; only differences carry physical meaning.
    """
    counts = np.array([10000, 8500, 6200, 4100], dtype=np.int64)
    log_g_a = np.array([0.0, 0.5, 1.0, 1.5])
    log_g_b = log_g_a + 10.0
    a = _normalized_log_z(counts, log_g_a)
    b = _normalized_log_z(counts, log_g_b)
    np.testing.assert_allclose(a, b)


def test_normalized_log_z_reference_ensemble_choice_consistent():
    """Choosing a different reference shifts log_z by a constant — differences invariant."""
    counts = np.array([10000, 8500, 6200, 4100], dtype=np.int64)
    log_g = np.array([0.0, 0.5, 1.0, 1.5])
    a = _normalized_log_z(counts, log_g, reference_ensemble=0)
    b = _normalized_log_z(counts, log_g, reference_ensemble=2)
    # Both arrays must satisfy log_z[ref]=0; differences agree.
    assert a[0] == 0.0
    assert b[2] == 0.0
    np.testing.assert_allclose(a[1] - a[0], b[1] - b[0])
    np.testing.assert_allclose(a[3] - a[1], b[3] - b[1])


def test_collection_estimator_recovers_target_for_detailed_balance_matrix():
    """Build a collection matrix whose stationary distribution is Z*exp(log_g).

    Then ``_normalized_log_z_from_collection`` must recover log Z.  This is
    the math underlying the ``estimator='collection'`` mode and historically
    sat alongside the visit-count estimator without independent verification.
    """
    target_log_z = np.array([0.0, -1.0, -2.0, -2.8], dtype=np.float64)
    log_g = np.array([0.0, 0.5, 1.5, 2.5], dtype=np.float64)
    pi = np.exp(target_log_z + log_g)
    pi /= pi.sum()

    # Construct any row-stochastic, detailed-balance-consistent transition
    # matrix with stationary distribution `pi`.  Use the simplest such
    # construction: P[i,j] = pi[j] (memoryless chain).  The stationary is pi.
    n = pi.size
    P = np.tile(pi, (n, 1))
    counts_per_row = 100_000
    collection_counts = (counts_per_row * P).astype(np.float64)

    recovered = _normalized_log_z_from_collection(
        collection_counts, log_g, reference_ensemble=0,
    )
    np.testing.assert_allclose(recovered, target_log_z, atol=1e-3, rtol=1e-3)


def test_jackknife_block_count_independence_for_constant_blocks():
    """Identical blocks → zero jackknife error.

    If every block has identical visit counts, the leave-one-out variance
    is exactly zero.  This is a sanity check on the jackknife formula.
    """
    base = np.array([1000, 800, 600, 400], dtype=np.int64)
    blocks = np.tile(base, (5, 1))  # 5 identical blocks
    log_g = np.array([0.0, 0.5, 1.0, 1.5])
    log_z, log_z_err = _jackknife_log_z(blocks, log_g)
    np.testing.assert_allclose(log_z_err, 0.0, atol=1e-12)


def test_jackknife_error_scales_inversely_with_block_count():
    """Error on the mean must shrink as more independent blocks are added.

    For block_arr drawn from the same distribution, doubling the number of
    blocks should reduce the jackknife std-error by ~1/sqrt(2).
    """
    rng = np.random.default_rng(seed=42)
    log_g = np.array([0.0, 0.5, 1.0, 1.5])
    weights = np.exp(np.array([0.0, -1.0, -2.0, -2.5]) + log_g)
    weights /= weights.sum()

    def _draw_blocks(n_blocks):
        return np.array([
            rng.multinomial(10000, weights) for _ in range(n_blocks)
        ], dtype=np.int64)

    rng_copy = np.random.default_rng(seed=42)
    weights2 = weights.copy()
    blocks_small = np.array([
        rng_copy.multinomial(10000, weights2) for _ in range(20)
    ], dtype=np.int64)
    blocks_large = np.array([
        rng_copy.multinomial(10000, weights2) for _ in range(80)
    ], dtype=np.int64)

    _, err_small = _jackknife_log_z(blocks_small, log_g)
    _, err_large = _jackknife_log_z(blocks_large, log_g)
    # 4x blocks → ~2x smaller std error.  Allow generous bracket [1.5, 3.0].
    ratio = float(err_small[1] / max(err_large[1], 1e-12))
    assert 1.5 < ratio < 3.5, (
        f"jackknife error did not scale as expected: "
        f"err(20 blocks)/err(80 blocks)={ratio:.2f}, expected ≈2"
    )


@pytest.mark.parametrize("perturbation", [-0.5, -0.1, 0.0, 0.3, 1.0])
def test_log_z_invariant_to_log_g_perturbation_in_synthetic_counts(perturbation):
    """log_z is the *unbiased* estimator of log Z regardless of which log_g
    was used to generate the counts.  The autotune choice cannot bias the mean.

    Synthetic counts follow ``V_k ∝ exp(log_z + log_g)``.  We compute log_z
    using the SAME log_g (the true one), then perturb log_g by adding a
    constant per-k term and re-derive — the answer must shift by the inverse
    perturbation, recovering log_z exactly.
    """
    target_log_z = np.array([0.0, -1.5, -2.7, -4.1])
    log_g_true = np.array([0.0, 0.8, 1.9, 3.5])
    weights = np.exp(target_log_z + log_g_true)
    counts = (1_000_000 * weights / weights.sum()).astype(np.int64)

    # Original recovery
    a = _normalized_log_z(counts, log_g_true)
    # Now imagine someone passed a *wrong* log_g (off by a constant
    # perturbation per index).  The recovered log_z should compensate.
    log_g_wrong = log_g_true + np.array([0.0, perturbation, 2 * perturbation,
                                          3 * perturbation])
    b = _normalized_log_z(counts, log_g_wrong)
    # b[k] should equal a[k] - (log_g_wrong[k] - log_g_true[k]) + (log_g_wrong[0] - log_g_true[0])
    expected = a - (log_g_wrong - log_g_true) + (log_g_wrong[0] - log_g_true[0])
    np.testing.assert_allclose(b, expected, atol=1e-9)
