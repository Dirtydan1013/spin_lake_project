"""Flyvbjerg–Petersen blocking analysis helpers for single-point estimates.

The error of a Monte-Carlo mean is estimated from block means at increasing
block size b; the curve err(b) rises until blocks are longer than the
autocorrelation time and then plateaus.  A still-rising curve at the largest b
means the naive error is an underestimate.

Two flavours:
- :func:`blocking_curve_linear`   — estimator is a plain mean (energy, ⟨J⟩).
- :func:`blocking_curve_jackknife` — arbitrary nonlinear estimator of the
  pooled samples (e.g. ΔS₂ = −log⟨e^{−W}⟩); leave-one-block-out jackknife.

Blocks never cross chain boundaries (chains = independent MPI ranks).
"""

from __future__ import annotations

import numpy as np


def default_bin_sizes(n_min_chain, min_blocks=16, base=1):
    """Powers of 2 (times ``base`` samples) keeping ≥ min_blocks total blocks."""
    sizes = []
    b = 1
    while n_min_chain // b >= 2 and b <= n_min_chain:
        sizes.append(b * base)
        b *= 2
    # keep sizes that leave at least min_blocks blocks in the shortest chain
    # times one chain; the caller pools blocks over chains so this is loose.
    return [s for s in sizes if (n_min_chain // (s // base)) >= 2] or [base]


def _blocks(chain, b):
    """Consecutive length-b block means of one chain (tail trimmed)."""
    n = (len(chain) // b) * b
    if n == 0:
        return np.empty(0)
    return np.asarray(chain[:n], dtype=np.float64).reshape(-1, b).mean(axis=1)


def _raw_blocks(chain, b):
    """Consecutive length-b sample blocks (list of arrays, tail trimmed)."""
    n = (len(chain) // b) * b
    return [np.asarray(chain[i:i + b], dtype=np.float64) for i in range(0, n, b)]


def blocking_curve_linear(chains, bin_sizes=None, min_blocks=8):
    """(bin_sizes, sem, n_blocks) for the pooled mean of ``chains``.

    ``chains`` is a list of 1D arrays (one per independent chain/rank).  For
    each b, block means are pooled over chains and the SEM of the pooled
    blocks is reported.
    """
    chains = [np.asarray(c, dtype=np.float64).reshape(-1) for c in chains
              if len(c) > 0]
    n_min = min(len(c) for c in chains)
    if bin_sizes is None:
        bin_sizes = default_bin_sizes(n_min)
    bs, sems, counts = [], [], []
    for b in bin_sizes:
        blocks = np.concatenate([_blocks(c, b) for c in chains])
        if len(blocks) < min_blocks:
            break
        bs.append(b)
        sems.append(blocks.std(ddof=1) / np.sqrt(len(blocks)))
        counts.append(len(blocks))
    return np.array(bs), np.array(sems), np.array(counts)


def blocking_curve_jackknife(chains, estimator, bin_sizes=None, min_blocks=8):
    """(bin_sizes, err, n_blocks) for a nonlinear ``estimator`` of the pooled samples.

    ``estimator(samples)`` maps a 1D array of raw samples to a float.  For each
    b the samples are cut into per-chain blocks of b, and the leave-one-block-out
    jackknife error of the estimator is reported.
    """
    chains = [np.asarray(c, dtype=np.float64).reshape(-1) for c in chains
              if len(c) > 0]
    n_min = min(len(c) for c in chains)
    if bin_sizes is None:
        bin_sizes = default_bin_sizes(n_min)
    bs, errs, counts = [], [], []
    for b in bin_sizes:
        blocks = [blk for c in chains for blk in _raw_blocks(c, b)]
        n = len(blocks)
        if n < min_blocks:
            break
        pooled = np.concatenate(blocks)
        sizes = np.array([len(blk) for blk in blocks])
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        loo = np.empty(n)
        for i in range(n):
            loo[i] = estimator(np.concatenate(
                [pooled[:offsets[i]], pooled[offsets[i + 1]:]]))
        loo_mean = loo.mean()
        errs.append(np.sqrt((n - 1) / n * np.sum((loo - loo_mean) ** 2)))
        bs.append(b)
        counts.append(n)
    return np.array(bs), np.array(errs), np.array(counts)
