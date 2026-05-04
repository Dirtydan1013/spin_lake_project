"""Cross-method consistency: ratio method and expanded ensemble must agree.

Both methods compute the same observable ``Tr(rho^2_X)`` for a given region X
of the QAQMC midpoint state.  Their internals differ — the ratio method runs
one MC chain per site addition and multiplies per-site Z ratios; the expanded
ensemble traverses a single chain through a ladder of ensembles.  The final
S2 must agree within combined statistical error.

The conversation that prompted this audit found a 12-sigma mismatch on the
4x4 m=1 KP lattice (CA's ladder window 2 vs C's ladder window 2).  This kind
of cross-method comparison is the most direct guard against such failures.
"""

from __future__ import annotations

import numpy as np

from src.engines.qaqmc_renyi import QAQMCRenyiRydberg
from src.rydberg.lattices import generate_1d_chain
from src.tee.qaqmc_renyi_ratio import RatioRunner
from src.tee.reweighting import ReweightingDriver


PHYSICS = dict(
    N=4, M=100, Omega=1.0, Rb=1.2,
    delta_min=0.0, delta_max=1.5, neighbor_cutoff=1,
)


def _new_engine(seed):
    return QAQMCRenyiRydberg(
        N=PHYSICS["N"], M=PHYSICS["M"],
        Omega=PHYSICS["Omega"], Rb=PHYSICS["Rb"],
        delta_min=PHYSICS["delta_min"], delta_max=PHYSICS["delta_max"],
        pos=generate_1d_chain(PHYSICS["N"]),
        seed=seed, neighbor_cutoff=PHYSICS["neighbor_cutoff"],
    )


def _new_driver(seed):
    return ReweightingDriver(
        N=PHYSICS["N"], M=PHYSICS["M"],
        Omega=PHYSICS["Omega"], Rb=PHYSICS["Rb"],
        delta_min=PHYSICS["delta_min"], delta_max=PHYSICS["delta_max"],
        pos=generate_1d_chain(PHYSICS["N"]),
        seed=seed, neighbor_cutoff=PHYSICS["neighbor_cutoff"],
    )


def _s2_via_ratio_method(seed, *, n_therm=2000, n_measure=20000, block_size=500):
    """Compute S2(region={0,1}) by chaining single-site ratios."""
    runner = RatioRunner(engine=_new_engine(seed))
    site_order = [0, 1]
    log_ratio = 0.0
    log_ratio_var = 0.0
    cur_mask = np.zeros(PHYSICS["N"], dtype=np.uint8)
    for site in site_order:
        result = runner.run_single_ratio(
            cur_mask, next_site=site,
            n_therm=n_therm, n_measure=n_measure, block_size=block_size,
        )
        # Log-ratio with propagated error
        r = float(result.ratio)
        re = float(result.ratio_err)
        log_ratio += float(np.log(r))
        log_ratio_var += (re / r) ** 2
        cur_mask = cur_mask.copy()
        cur_mask[site] = 1
    s2 = float(-log_ratio)
    s2_err = float(np.sqrt(log_ratio_var))
    return s2, s2_err


def _s2_via_expanded(seed, *, autotune_steps=15000, prod_steps=160000):
    masks = [
        np.array([0, 0, 0, 0], dtype=np.uint8),
        np.array([1, 0, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0, 2], [1]]
    driver = _new_driver(seed)
    driver.set_ensemble_ladder(masks, neighbors, initial_ensemble=0)
    driver.auto_tune(n_steps_per_iter=autotune_steps, max_iters=6,
                     tol=1.15, method="transition_matrix", damping=0.7)
    result = driver.run_production(n_steps=prod_steps, block_size=2000)
    s2, s2_err = result.s2(target_ensemble=2, reference_ensemble=0)
    return float(s2), float(s2_err)


def test_ratio_method_matches_expanded_ensemble():
    """S2 from ratio method == S2 from expanded ensemble within 3 sigma."""
    s2_r, err_r = _s2_via_ratio_method(seed=4321)
    s2_e, err_e = _s2_via_expanded(seed=8765)
    diff = abs(s2_r - s2_e)
    combined = float(np.sqrt(err_r ** 2 + err_e ** 2))
    assert diff <= 3.0 * combined, (
        f"method mismatch: ratio S2={s2_r:.4f} ± {err_r:.4f}, "
        f"expanded S2={s2_e:.4f} ± {err_e:.4f}, "
        f"diff={diff:.4f} > 3*comb={3.0 * combined:.4f}"
    )


def test_expanded_ensemble_collection_matches_visit_estimator():
    """Within one expanded run, the two estimators must agree.

    ``s2`` (from visit-count histogram) and ``s2_collection`` (from the
    transition matrix's stationary distribution) are independent ways to
    extract log_z from the same MC stream.  Disagreement signals an autotune
    or detailed-balance bug.
    """
    masks = [
        np.array([0, 0, 0, 0], dtype=np.uint8),
        np.array([1, 0, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0, 2], [1]]
    driver = _new_driver(seed=2468)
    driver.set_ensemble_ladder(masks, neighbors, initial_ensemble=0)
    driver.auto_tune(n_steps_per_iter=15000, max_iters=6,
                     tol=1.15, method="transition_matrix", damping=0.7)
    result = driver.run_production(n_steps=160000, block_size=2000)
    s2_v, err_v = result.s2(target_ensemble=2)
    s2_c, err_c = result.s2_collection(target_ensemble=2)
    diff = abs(s2_v - s2_c)
    comb = float(np.sqrt(err_v ** 2 + err_c ** 2))
    assert diff <= 3.0 * comb, (
        f"intra-run estimator mismatch: visit S2={s2_v:.4f} ± {err_v:.4f}, "
        f"collection S2={s2_c:.4f} ± {err_c:.4f}, "
        f"diff={diff:.4f} > 3*comb={3.0 * comb:.4f}"
    )
