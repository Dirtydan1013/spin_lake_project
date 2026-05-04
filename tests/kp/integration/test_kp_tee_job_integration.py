from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.tee.compose_tee import KPResult, compose_kp, load_kp_result_hdf5, summarize_region
from src.kp.kp_geometry import build_kagome_bond_ordering_bonds, build_kp_ladders, build_kp_region_masks
from src.kp.kp_tee_job import run_expanded_job, run_ratio_job
from src.tee.reweighting import ExpandedProductionResult, load_expanded_result_hdf5


@dataclass(frozen=True)
class _FakeRatioPayload:
    kp_result: KPResult


@dataclass(frozen=True)
class _FakeRatioRun:
    spec: object
    result: _FakeRatioPayload


@dataclass(frozen=True)
class _FakeExpandedRun:
    spec: object
    ladder: object
    autotune: object
    production: ExpandedProductionResult


class _FakeRatioWorkflow:
    def __init__(self, spec, kp_result):
        self._spec = spec
        self._kp_result = kp_result

    def run(self, **kwargs):
        return _FakeRatioRun(spec=self._spec, result=_FakeRatioPayload(kp_result=self._kp_result))


class _FakeExpandedWorkflow:
    def __init__(self, spec, ladders):
        self._spec = spec
        self._ladders = ladders

    def run_region(self, region_name, **kwargs):
        ladder = self._ladders[str(region_name)]
        return _FakeExpandedRun(
            spec=self._spec,
            ladder=ladder,
            autotune=None,
            production=_build_fake_expanded_result(len(ladder.masks)),
        )


def _build_fake_kp_result() -> KPResult:
    summaries = {}
    for name in ("A", "B", "C", "AB", "BC", "CA", "ABC"):
        summaries[name] = summarize_region(
            name,
            sites_ordered=np.array([0], dtype=np.int32),
            ratios=np.array([0.9], dtype=np.float64),
            ratio_errs=np.array([0.01], dtype=np.float64),
        )
    gamma, gamma_err = compose_kp(summaries)
    return KPResult(region_summaries=summaries, gamma=gamma, gamma_err=gamma_err)


def _build_fake_expanded_result(n_ens: int) -> ExpandedProductionResult:
    visit_counts = np.arange(100, 100 + n_ens, dtype=np.int64)
    block_visit_counts = np.vstack([visit_counts, visit_counts]).astype(np.int64)
    transition_counts = np.eye(n_ens, dtype=np.int64) * 10
    collection_counts = np.eye(n_ens, dtype=np.float64) * 10.0
    log_g = np.linspace(0.0, 0.1 * max(n_ens - 1, 0), n_ens, dtype=np.float64)
    log_z = np.linspace(0.0, 0.18 * max(n_ens - 1, 0), n_ens, dtype=np.float64)
    log_z_err = np.linspace(0.0, 0.02 * max(n_ens - 1, 0), n_ens, dtype=np.float64)
    collection_log_z = np.linspace(0.0, 0.17 * max(n_ens - 1, 0), n_ens, dtype=np.float64)
    collection_log_z_err = np.linspace(0.0, 0.015 * max(n_ens - 1, 0), n_ens, dtype=np.float64)
    block_collection_counts = np.stack([collection_counts, collection_counts], axis=0)
    return ExpandedProductionResult(
        log_g=log_g,
        visit_counts=visit_counts,
        block_visit_counts=block_visit_counts,
        transition_counts=transition_counts,
        collection_counts=collection_counts,
        log_z=log_z,
        log_z_err=log_z_err,
        block_collection_counts=block_collection_counts,
        collection_log_z=collection_log_z,
        collection_log_z_err=collection_log_z_err,
    )


def _test_run_ratio_job_writes_geometry_and_kp_result():
    spec = build_kp_region_masks(8, 8, m=1)
    fake_workflow = _FakeRatioWorkflow(spec, _build_fake_kp_result())
    with tempfile.TemporaryDirectory() as tmpdir:
        payload = run_ratio_job(
            nx=8,
            ny=8,
            m=1,
            M=16,
            Omega=1.0,
            Rb=1.2,
            delta_min=0.0,
            delta_max=1.0,
            epsilon=0.01,
            seed=7,
            a=1.0,
            neighbor_cutoff=3,
            n_therm=10,
            n_measure=20,
            measure_stride=1,
            block_size=5,
            preferred_center_label="C35",
            output_dir=tmpdir,
            workflow=fake_workflow,
        )
        assert Path(payload["geometry_path"]).exists()
        assert Path(payload["kp_result_path"]).exists()
        loaded = load_kp_result_hdf5(payload["kp_result_path"])
        assert set(loaded.region_summaries.keys()) == {"A", "B", "C", "AB", "BC", "CA", "ABC"}


def _test_run_expanded_job_writes_per_region_results():
    spec = build_kp_region_masks(8, 8, m=1)
    ordering_bonds = build_kagome_bond_ordering_bonds(8, 8)
    ladders = build_kp_ladders(spec, bond_sites=ordering_bonds)
    fake_workflow = _FakeExpandedWorkflow(spec, ladders)
    with tempfile.TemporaryDirectory() as tmpdir:
        payload = run_expanded_job(
            nx=8,
            ny=8,
            m=1,
            M=16,
            Omega=1.0,
            Rb=1.2,
            delta_min=0.0,
            delta_max=1.0,
            epsilon=0.01,
            seed=7,
            a=1.0,
            neighbor_cutoff=3,
            regions=["A", "BC"],
            preferred_center_label="C35",
            output_dir=tmpdir,
            autotune_steps_per_iter=10,
            autotune_max_iters=2,
            autotune_tol=1.1,
            autotune_method="transition_matrix",
            autotune_damping=0.7,
            n_steps=100,
            block_size=10,
            target_s2_err=None,
            batch_steps=None,
            max_steps=None,
            min_steps=0,
            estimator="collection",
            workflow=fake_workflow,
        )
        assert Path(payload["geometry_path"]).exists()
        for region_name in ("A", "BC"):
            path = Path(payload[region_name]["path"])
            assert path.exists()
            loaded = load_expanded_result_hdf5(path)
            assert loaded["params"]["region_name"] == region_name


def main():
    _test_run_ratio_job_writes_geometry_and_kp_result()
    _test_run_expanded_job_writes_per_region_results()
    print("KP TEE job integration checks passed")


if __name__ == "__main__":
    main()
