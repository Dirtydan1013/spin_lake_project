"""Cross-ladder consistency: same mask via different ladders must agree.

The expanded-ensemble engine computes ``log_z[k] = log(Z_k / Z_0)`` where
``Z_k`` is the partition function of the k-th window's mask.  This value is a
function of the *mask* alone — the path the ladder took to reach that mask
must not affect the equilibrium answer.

In production we observed a 12-sigma mismatch on the 4x4 m=1 KP lattice:
``Tr(rho^2_{site 33, 34})`` was 0.97 ± 0.025 when reached via region C's
ladder and 0.36 ± 0.04 when reached via region CA's longer ladder.  That is
not a physics question, it's a numerical consistency violation.

This test runs two independent drivers (different seeds) with two ladders
that share an intermediate window.  The shared-window purity must agree
within 3 sigma combined error.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rydberg.lattices import generate_1d_chain
from src.tee.reweighting import ReweightingDriver


PHYSICS = dict(
    Omega=1.0, Rb=1.2, delta_min=0.0, delta_max=1.5,
    M=100, neighbor_cutoff=1, N=4,
)


def _driver(seed):
    return ReweightingDriver(
        N=PHYSICS["N"], M=PHYSICS["M"],
        Omega=PHYSICS["Omega"], Rb=PHYSICS["Rb"],
        delta_min=PHYSICS["delta_min"], delta_max=PHYSICS["delta_max"],
        pos=generate_1d_chain(PHYSICS["N"]),
        seed=seed, neighbor_cutoff=PHYSICS["neighbor_cutoff"],
    )


def _ladder(masks_2d):
    """Wrap a list of bit-masks into masks + neighbors (linear ladder)."""
    masks = [np.asarray(m, dtype=np.uint8) for m in masks_2d]
    neighbors = []
    for idx in range(len(masks)):
        row = []
        if idx - 1 >= 0:
            row.append(idx - 1)
        if idx + 1 < len(masks):
            row.append(idx + 1)
        neighbors.append(row)
    return masks, neighbors


def _run(driver, masks_2d, *, autotune_steps=15000, prod_steps=160000):
    masks, neighbors = _ladder(masks_2d)
    driver.set_ensemble_ladder(masks, neighbors, initial_ensemble=0)
    driver.auto_tune(n_steps_per_iter=autotune_steps, max_iters=6,
                     tol=1.15, method="transition_matrix", damping=0.7)
    return driver.run_production(n_steps=prod_steps, block_size=2000)


# Shared mask whose Tr(rho^2) we will probe via two ladders.
SHARED_MASK = [1, 1, 0, 0]
# Short ladder: empty -> {0} -> {0,1}.  Target is the shared mask itself.
LADDER_SHORT = [
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 0, 0],
]
# Long ladder: empty -> {0} -> {0,1} -> {0,1,2} -> {0,1,2,3}.  Window 2 IS
# the shared mask, so log_z[2] must match log_z[2] of the short ladder.
LADDER_LONG = [
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
]


def test_shared_window_purity_matches_across_ladders():
    """Tr(rho^2_{0,1}) at window 2 of the short ladder == window 2 of the long."""
    short = _run(_driver(seed=1234), LADDER_SHORT)
    long_ = _run(_driver(seed=5678), LADDER_LONG)

    # Sanity: window 2 of both ladders is the shared mask
    short_purity = float(np.exp(short.log_z[2] - short.log_z[0]))
    short_err = float(short.log_z_err[2] * short_purity)
    long_purity = float(np.exp(long_.log_z[2] - long_.log_z[0]))
    long_err = float(long_.log_z_err[2] * long_purity)

    diff = abs(short_purity - long_purity)
    combined_err = float(np.sqrt(short_err ** 2 + long_err ** 2))

    assert diff <= 3.0 * combined_err, (
        f"cross-ladder mismatch on the SAME physical mask {SHARED_MASK}: "
        f"short ladder gives {short_purity:.4f} ± {short_err:.4f}, "
        f"long ladder gives {long_purity:.4f} ± {long_err:.4f}, "
        f"diff={diff:.4f} > 3*combined_err={3.0 * combined_err:.4f}"
    )


def test_shared_window_purity_matches_under_different_intermediate_paths():
    """Two short ladders with different first-site choices must agree at the target.

    Ladder L0:  empty -> {0} -> {0,1}
    Ladder L1:  empty -> {1} -> {0,1}

    Both end at {0,1}; only the order in which sites 0/1 are added differs.
    The final-window purity is independent of that choice.
    """
    L0 = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]]
    L1 = [[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]]

    a = _run(_driver(seed=1001), L0)
    b = _run(_driver(seed=2002), L1)

    pa = float(np.exp(a.log_z[2] - a.log_z[0]))
    ea = float(a.log_z_err[2] * pa)
    pb = float(np.exp(b.log_z[2] - b.log_z[0]))
    eb = float(b.log_z_err[2] * pb)

    diff = abs(pa - pb)
    comb = float(np.sqrt(ea ** 2 + eb ** 2))
    assert diff <= 3.0 * comb, (
        f"site-order-dependent target purity: L0={pa:.4f} ± {ea:.4f}, "
        f"L1={pb:.4f} ± {eb:.4f}, diff={diff:.4f} > 3*comb={3.0 * comb:.4f}"
    )


@pytest.mark.parametrize("seed_pair", [(11, 12), (33, 44), (55, 66)])
def test_shared_window_purity_robust_across_seed_pairs(seed_pair):
    """Cross-ladder agreement must hold across independent seed combinations."""
    s1, s2 = seed_pair
    short = _run(_driver(seed=s1), LADDER_SHORT, prod_steps=100000)
    long_ = _run(_driver(seed=s2), LADDER_LONG, prod_steps=100000)
    pa = float(np.exp(short.log_z[2] - short.log_z[0]))
    ea = float(short.log_z_err[2] * pa)
    pb = float(np.exp(long_.log_z[2] - long_.log_z[0]))
    eb = float(long_.log_z_err[2] * pb)
    diff = abs(pa - pb)
    comb = float(np.sqrt(ea ** 2 + eb ** 2))
    assert diff <= 3.0 * comb, (
        f"seeds={seed_pair}: cross-ladder mismatch "
        f"short={pa:.4f}±{ea:.4f}, long={pb:.4f}±{eb:.4f}, "
        f"diff={diff:.4f} > 3*comb={3.0 * comb:.4f}"
    )
