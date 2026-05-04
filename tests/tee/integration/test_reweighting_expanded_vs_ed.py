import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analysis.ed_core import build_qaqmc_midpoint_state
from src.rydberg.lattices import generate_1d_chain
from src.tee.reweighting import ReweightingDriver


def _reduced_density_matrix(psi, region_mask):
    n_sites = int(np.log2(psi.size))
    region = [i for i in range(n_sites) if region_mask[i]]
    complement = [i for i in range(n_sites) if not region_mask[i]]
    mat = np.zeros((1 << len(region), 1 << len(complement)), dtype=np.complex128)
    for state in range(1 << n_sites):
        row = 0
        col = 0
        for bit, site in enumerate(region):
            row |= (((state >> site) & 1) << bit)
        for bit, site in enumerate(complement):
            col |= (((state >> site) & 1) << bit)
        mat[row, col] = psi[state]
    return mat @ mat.conj().T


def _test_expanded_ensemble_matches_midpoint_ed():
    N = 4
    Omega = 1.0
    Rb = 1.2
    delta_min = 0.0
    delta_max = 1.5
    M = 100
    pos = generate_1d_chain(N)
    region_mask = np.array([1, 1, 0, 0], dtype=np.uint8)

    psi = build_qaqmc_midpoint_state(
        N=N,
        Omega=Omega,
        delta_min=delta_min,
        delta_max=delta_max,
        Rb=Rb,
        M=M,
        pos=pos,
        neighbor_cutoff=1,
    )
    rho = _reduced_density_matrix(psi, region_mask)
    s2_ed = -np.log(float(np.real(np.trace(rho @ rho))))

    masks = [
        np.array([0, 0, 0, 0], dtype=np.uint8),
        np.array([1, 0, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0, 2], [1]]

    driver = ReweightingDriver(
        N=N,
        M=M,
        Omega=Omega,
        Rb=Rb,
        delta_min=delta_min,
        delta_max=delta_max,
        pos=pos,
        seed=1234,
        neighbor_cutoff=1,
    )
    driver.set_ensemble_ladder(masks, neighbors, initial_ensemble=0)
    auto = driver.auto_tune(n_steps_per_iter=10000, max_iters=4, tol=1.2)
    final_counts = auto.visit_counts[-1]
    assert float(np.max(final_counts) / np.min(final_counts)) < 1.2

    production = driver.run_production(n_steps=120000, block_size=2000)
    s2_mc, s2_err = production.s2(2)
    s2_coll, s2_coll_err = production.s2_collection(2)
    # Tightened from absolute 0.04 to 0.015 (≈2% relative on s2≈0.7).
    # 120k production steps yields ~1% jackknife stat error; the old 4% bound
    # could paper over real systematic biases (eg the un-thermalised initial
    # state contamination found in the 4x4 KP audit).
    assert abs(s2_mc - s2_ed) < 0.015, (
        f"visit-estimator: |MC-ED|={abs(s2_mc - s2_ed):.4f} > 0.015 "
        f"(MC={s2_mc:.4f}±{s2_err:.4f}, ED={s2_ed:.4f})"
    )
    assert abs(s2_mc - s2_ed) < 3.0 * s2_err
    assert abs(s2_coll - s2_ed) < 0.015, (
        f"collection-estimator: |MC-ED|={abs(s2_coll - s2_ed):.4f} > 0.015"
    )
    assert abs(s2_coll - s2_ed) < 3.0 * max(s2_coll_err, 1e-12)


def _test_transition_matrix_reweighting_matches_midpoint_ed():
    N = 4
    Omega = 1.0
    Rb = 1.2
    delta_min = 0.0
    delta_max = 1.5
    M = 100
    pos = generate_1d_chain(N)
    region_mask = np.array([1, 1, 0, 0], dtype=np.uint8)

    psi = build_qaqmc_midpoint_state(
        N=N,
        Omega=Omega,
        delta_min=delta_min,
        delta_max=delta_max,
        Rb=Rb,
        M=M,
        pos=pos,
        neighbor_cutoff=1,
    )
    rho = _reduced_density_matrix(psi, region_mask)
    s2_ed = -np.log(float(np.real(np.trace(rho @ rho))))

    masks = [
        np.array([0, 0, 0, 0], dtype=np.uint8),
        np.array([1, 0, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0, 2], [1]]

    driver = ReweightingDriver(
        N=N,
        M=M,
        Omega=Omega,
        Rb=Rb,
        delta_min=delta_min,
        delta_max=delta_max,
        pos=pos,
        seed=1234,
        neighbor_cutoff=1,
    )
    driver.set_ensemble_ladder(masks, neighbors, initial_ensemble=0)
    auto = driver.auto_tune(
        n_steps_per_iter=15000,
        max_iters=8,
        tol=1.15,
        method="transition_matrix",
        damping=0.7,
    )
    assert auto.collection_counts is not None
    final_counts = auto.visit_counts[-1]
    assert float(np.max(final_counts) / np.min(final_counts)) < 1.15

    production = driver.run_production(n_steps=160000, block_size=2000)
    s2_mc, s2_err = production.s2(2)
    s2_coll, s2_coll_err = production.s2_collection(2)
    assert production.collection_counts is not None
    assert abs(s2_mc - s2_ed) < 0.015, (
        f"visit-estimator (transition_matrix autotune): |MC-ED|={abs(s2_mc - s2_ed):.4f} > 0.015"
    )
    assert abs(s2_mc - s2_ed) < 3.0 * s2_err
    assert abs(s2_coll - s2_ed) < 0.015, (
        f"collection-estimator (transition_matrix autotune): |MC-ED|={abs(s2_coll - s2_ed):.4f} > 0.015"
    )
    assert abs(s2_coll - s2_ed) < 3.0 * s2_coll_err


# -----------------------------------------------------------------------------
# Pytest-discoverable multi-seed variant.  The two _test_* functions above run
# only as scripts (python file.py) — pytest skips them.  This function adds an
# auto-collected check that exercises the same code path across 5 seeds so a
# single-seed lucky pass can't mask a systematic bias.
# -----------------------------------------------------------------------------

import pytest  # noqa: E402  (kept here so the script-style runs above don't import pytest)


def _run_one_seed(seed):
    N = 4
    Omega = 1.0
    Rb = 1.2
    delta_min = 0.0
    delta_max = 1.5
    M = 100
    pos = generate_1d_chain(N)
    region_mask = np.array([1, 1, 0, 0], dtype=np.uint8)

    psi = build_qaqmc_midpoint_state(
        N=N, Omega=Omega, delta_min=delta_min, delta_max=delta_max,
        Rb=Rb, M=M, pos=pos, neighbor_cutoff=1,
    )
    rho = _reduced_density_matrix(psi, region_mask)
    s2_ed = -np.log(float(np.real(np.trace(rho @ rho))))

    masks = [
        np.array([0, 0, 0, 0], dtype=np.uint8),
        np.array([1, 0, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0, 2], [1]]
    driver = ReweightingDriver(
        N=N, M=M, Omega=Omega, Rb=Rb,
        delta_min=delta_min, delta_max=delta_max,
        pos=pos, seed=seed, neighbor_cutoff=1,
    )
    driver.set_ensemble_ladder(masks, neighbors, initial_ensemble=0)
    driver.auto_tune(n_steps_per_iter=15000, max_iters=6,
                     tol=1.15, method="transition_matrix", damping=0.7)
    production = driver.run_production(n_steps=160000, block_size=2000)
    s2_mc, s2_err = production.s2(2)
    return float(s2_ed), float(s2_mc), float(s2_err)


SEEDS_FOR_AGGREGATE = [101, 202, 303, 404, 505]


@pytest.mark.parametrize("seed", SEEDS_FOR_AGGREGATE)
def test_expanded_vs_ed_per_seed_within_3sigma(seed):
    """Each seed must agree with ED within 3 sigma of its own jackknife error.

    Per-seed absolute tolerance would falsely assume small-s2 cases (s2 ~ 0.08)
    can be pinned down to 1% — but at 160k steps the stat error itself is ~20%
    of s2 there.  Use the 3-sigma stat bound here; the aggregate-mean test
    below provides the systematic-bias floor.
    """
    s2_ed, s2_mc, s2_err = _run_one_seed(seed)
    diff = abs(s2_mc - s2_ed)
    assert diff < 3.0 * s2_err, (
        f"seed={seed}: |MC-ED|={diff:.4f} > 3*{s2_err:.4f} = {3 * s2_err:.4f} "
        f"(MC={s2_mc:.4f}±{s2_err:.4f}, ED={s2_ed:.4f})"
    )


def test_expanded_vs_ed_seed_averaged_within_3sem():
    """Seed-averaged MC must hit ED within 3 standard errors of the mean.

    Per-seed jackknife error is dominated by MC noise; averaging across
    independent seeds reduces SEM by sqrt(N) and so naturally surfaces any
    systematic bias that exceeds the achievable stat precision.

    We deliberately do NOT add a small absolute floor (e.g. < 0.01) here —
    such a floor below 3*SEM would create a flaky test (a "1.7-sigma seed"
    fails purely from MC noise).  The 3*SEM bound itself adapts to whatever
    stat precision the chosen step count delivers; a real systematic bias
    of ~3% on s2 ~ 0.7 (~5*SEM here) would still surface clearly.
    """
    triples = [_run_one_seed(s) for s in SEEDS_FOR_AGGREGATE]
    s2_eds = np.array([t[0] for t in triples])
    s2_mcs = np.array([t[1] for t in triples])
    s2_errs = np.array([t[2] for t in triples])
    np.testing.assert_allclose(s2_eds, s2_eds[0], atol=1e-9,
                               err_msg="ED must be seed-independent")
    s2_ed = float(s2_eds[0])
    mean_mc = float(np.mean(s2_mcs))
    sem = float(np.sqrt(np.sum(s2_errs ** 2)) / len(s2_errs))
    diff = abs(mean_mc - s2_ed)
    assert diff < 3.0 * sem, (
        f"seed-averaged |<MC>-ED|={diff:.4f} > 3*SEM={3 * sem:.4f} "
        f"(<MC>={mean_mc:.4f}, ED={s2_ed:.4f}); "
        f"per-seed values: {s2_mcs.tolist()}"
    )


def main():
    _test_expanded_ensemble_matches_midpoint_ed()
    _test_transition_matrix_reweighting_matches_midpoint_ed()
    print("Expanded-ensemble reweighting vs ED integration checks passed")


if __name__ == "__main__":
    main()
