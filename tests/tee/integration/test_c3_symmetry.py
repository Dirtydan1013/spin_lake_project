"""C3 symmetry test on a 3-site equilateral triangle.

A Rydberg ground state on a fully C3-symmetric 3-site triangle has
``S_2(site_0) = S_2(site_1) = S_2(site_2)`` by symmetry.  Any disagreement
beyond stat error indicates the engine itself is breaking the geometric
symmetry — for example, biased site initialisation, asymmetric MC proposal
weights, or shared-engine state contamination across regions.

This test is the "control" complement to the production 4x4 KP setup, which
*does* break C3 (cropping artefact in ``kagome_bond_triangle``).  Here we
remove the cropping and confirm the engine itself is symmetric.
"""

from __future__ import annotations

import numpy as np

from src.tee.reweighting import ReweightingDriver


# Equilateral triangle, side length 1.0; all three sites are within the
# Rydberg blockade radius Rb=1.2 of one another.
TRIANGLE_POS = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, np.sqrt(3.0) / 2.0],
], dtype=np.float64)


PHYSICS = dict(
    N=3, M=100, Omega=1.0, Rb=1.2,
    delta_min=0.0, delta_max=1.5, neighbor_cutoff=1,
)


def _driver(seed):
    return ReweightingDriver(
        N=PHYSICS["N"], M=PHYSICS["M"],
        Omega=PHYSICS["Omega"], Rb=PHYSICS["Rb"],
        delta_min=PHYSICS["delta_min"], delta_max=PHYSICS["delta_max"],
        pos=TRIANGLE_POS,
        seed=seed, neighbor_cutoff=PHYSICS["neighbor_cutoff"],
    )


def _single_site_purity(driver, site_idx):
    """Compute Tr(rho^2_{single site}) using a 2-window ladder."""
    masks = [
        np.zeros(PHYSICS["N"], dtype=np.uint8),
        np.zeros(PHYSICS["N"], dtype=np.uint8),
    ]
    masks[1][site_idx] = 1
    neighbors = [[1], [0]]
    driver.set_ensemble_ladder(masks, neighbors, initial_ensemble=0)
    driver.auto_tune(n_steps_per_iter=15000, max_iters=6,
                     tol=1.15, method="transition_matrix", damping=0.7)
    result = driver.run_production(n_steps=120000, block_size=2000)
    p = float(np.exp(result.log_z[1] - result.log_z[0]))
    e = float(result.log_z_err[1] * p)  # propagated to linear scale
    return p, e


def test_three_single_qubit_purities_agree_under_c3():
    """S2 on each of the three triangle vertices must agree within 3 sigma."""
    purities = []
    errs = []
    # Independent seed per region so the three estimates are statistically
    # independent — agreement then is non-trivial.
    for site_idx, seed in enumerate([1001, 2002, 3003]):
        p, e = _single_site_purity(_driver(seed=seed), site_idx)
        purities.append(p)
        errs.append(e)

    # Pairwise check: every pair must agree within 3-sigma combined error.
    labels = ["site_0", "site_1", "site_2"]
    for i in range(3):
        for j in range(i + 1, 3):
            diff = abs(purities[i] - purities[j])
            comb = float(np.sqrt(errs[i] ** 2 + errs[j] ** 2))
            assert diff <= 3.0 * comb, (
                f"C3 symmetry broken: {labels[i]}={purities[i]:.4f}±{errs[i]:.4f} "
                f"vs {labels[j]}={purities[j]:.4f}±{errs[j]:.4f}, "
                f"diff={diff:.4f} > 3*comb={3.0 * comb:.4f}"
            )

    # All three must also respect the single-qubit lower bound, just to be
    # sure none of them is silently wrong but they happen to agree.
    for i, p in enumerate(purities):
        assert p >= 0.5 - 3.0 * errs[i], (
            f"site_{i} purity {p:.4f} ± {errs[i]:.4f} below single-qubit "
            f"bound 0.5 — C3 symmetric or not, this is unphysical"
        )
