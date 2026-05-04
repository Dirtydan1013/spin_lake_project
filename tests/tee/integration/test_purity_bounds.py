"""Physical-bound regression tests for the expanded-ensemble engine.

For any window k whose mask covers ``n_k`` qubits, the path-integral identity
``Tr(rho^2_X) = Z_k / Z_0`` (with the engine's normalisation, ``log_z[0]=0``)
constrains the result to::

    1 / 2**n_k  <=  exp(log_z[k])  <=  1.

The window-1 lower bound (``Tr(rho^2_{single qubit}) >= 1/2``) is the tightest
in practice and was historically violated (~0.235 on the 4x4 m=1 KP A region)
by un-thermalised shared-engine MC runs.  This test guards against that
specific failure mode at integration scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rydberg.lattices import generate_1d_chain
from src.tee.reweighting import ReweightingDriver


def _make_driver(*, N=4, M=100, seed=1234):
    pos = generate_1d_chain(N)
    return ReweightingDriver(
        N=N, M=M, Omega=1.0, Rb=1.2,
        delta_min=0.0, delta_max=1.5, pos=pos,
        seed=seed, neighbor_cutoff=1,
    )


def _ladder_first_k_sites(N, k):
    """Return masks/neighbors for windows {empty, {0}, {0,1}, ..., {0,...,k-1}}."""
    masks = [np.zeros(N, dtype=np.uint8)]
    cur = np.zeros(N, dtype=np.uint8)
    for i in range(k):
        cur = cur.copy()
        cur[i] = 1
        masks.append(cur)
    neighbors = []
    for idx in range(len(masks)):
        row = []
        if idx - 1 >= 0:
            row.append(idx - 1)
        if idx + 1 < len(masks):
            row.append(idx + 1)
        neighbors.append(row)
    return masks, neighbors


def _run(driver, masks, neighbors, *, autotune_steps=15000, prod_steps=120000,
         block_size=2000):
    driver.set_ensemble_ladder(masks, neighbors, initial_ensemble=0)
    driver.auto_tune(n_steps_per_iter=autotune_steps, max_iters=6,
                     tol=1.15, method="transition_matrix", damping=0.7)
    return driver.run_production(n_steps=prod_steps, block_size=block_size)


def test_window1_purity_above_single_qubit_lower_bound():
    """Tr(rho^2) for a single qubit is in [0.5, 1].  Window 1 must respect this."""
    driver = _make_driver()
    masks, neighbors = _ladder_first_k_sites(N=4, k=2)  # windows {}, {0}, {0,1}
    result = _run(driver, masks, neighbors)
    purity_1 = float(np.exp(result.log_z[1] - result.log_z[0]))
    err_1 = float(result.log_z_err[1] * purity_1)  # propagated to linear scale
    # Allow up to 3-sigma slack for stat noise; the bug had purity_1 ≈ 0.24
    # which is ~10sigma below 0.5, so this margin is safe but discriminating.
    assert purity_1 >= 0.5 - 3.0 * err_1, (
        f"window-1 purity {purity_1:.3f} ± {err_1:.3f} violates single-qubit "
        f"lower bound 0.5 — likely un-thermalised initial state contamination"
    )
    assert purity_1 <= 1.0 + 3.0 * err_1, (
        f"window-1 purity {purity_1:.3f} ± {err_1:.3f} exceeds 1.0 upper bound"
    )


def test_all_windows_within_universal_purity_bounds():
    """For every window k, 1/2^n_k <= Tr(rho^2_X) <= 1.

    n_k counts qubits in mask_k.  This is a theorem regardless of state, lattice
    or sampling quality — any violation is a bug in either the engine or the
    reweighting math, not a physics question.
    """
    driver = _make_driver()
    masks, neighbors = _ladder_first_k_sites(N=4, k=3)  # 4 windows
    result = _run(driver, masks, neighbors, prod_steps=160000)

    for k, mask in enumerate(masks):
        n_k = int(np.sum(mask))
        purity = float(np.exp(result.log_z[k] - result.log_z[0]))
        err = float(result.log_z_err[k] * purity) if k > 0 else 0.0
        lower = 0.5 ** n_k
        # +/- 3 sigma slack so the test isn't flaky on edge windows
        assert purity >= lower - 3.0 * err, (
            f"window {k} (mask sum={n_k}): Tr(rho^2)={purity:.4f} ± {err:.4f} "
            f"below universal bound 1/2^{n_k}={lower:.4f}"
        )
        assert purity <= 1.0 + 3.0 * max(err, 1e-12), (
            f"window {k} (mask sum={n_k}): Tr(rho^2)={purity:.4f} > 1.0"
        )


@pytest.mark.parametrize("seed", [1234, 5678, 91011])
def test_window1_bound_robust_across_seeds(seed):
    """The single-qubit bound must hold across independent MC streams.

    Multi-seed coverage rules out a single accidentally-OK seed masking a
    systematic bias.
    """
    driver = _make_driver(seed=seed)
    masks, neighbors = _ladder_first_k_sites(N=4, k=1)  # just {empty, {0}}
    result = _run(driver, masks, neighbors, prod_steps=80000)
    purity = float(np.exp(result.log_z[1] - result.log_z[0]))
    err = float(result.log_z_err[1] * purity)
    assert purity >= 0.5 - 3.0 * err, (
        f"seed={seed}: purity {purity:.3f} ± {err:.3f} violates single-qubit "
        f"lower bound 0.5"
    )
