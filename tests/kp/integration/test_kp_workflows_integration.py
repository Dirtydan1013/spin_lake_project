from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tee.compose_tee import summarize_region
from src.kp.kp_workflows import KagomeKPExpandedWorkflow, KagomeKPRatioWorkflow
from src.tee.qaqmc_renyi_ratio import KPRatioRunner, RegionRunResult


class _FakeRegionRunner:
    def __init__(self):
        self.calls = []

    def run_region(self, region_name, region_mask, *, bond_sites=None, site_order=None,
                   n_therm, n_measure, measure_stride=1, block_size=None):
        self.calls.append({
            "region_name": str(region_name),
            "region_mask": np.asarray(region_mask, dtype=np.uint8).copy(),
            "site_order": None if site_order is None else np.asarray(site_order, dtype=np.int32).copy(),
        })
        if site_order is None or len(site_order) == 0:
            ratio = np.array([], dtype=np.float64)
            ratio_err = np.array([], dtype=np.float64)
            sites = np.array([], dtype=np.int32)
        else:
            sites = np.asarray(site_order, dtype=np.int32)
            ratio = np.full(sites.size, np.exp(-0.1), dtype=np.float64)
            ratio_err = np.full(sites.size, 0.01, dtype=np.float64)
        summary = summarize_region(str(region_name), sites, ratio, ratio_err)
        return RegionRunResult(
            region_name=str(region_name),
            site_order=sites,
            ratio_results=[],
            summary=summary,
        )


class _FakeExpandedDriver:
    def __init__(self):
        self.masks = None
        self.neighbors = None
        self.autotune_calls = []
        self.production_calls = []
        self.target_error_calls = []

    def set_ensemble_ladder(self, masks, neighbors, *, initial_ensemble=0):
        self.masks = [np.asarray(mask, dtype=np.uint8).copy() for mask in masks]
        self.neighbors = [list(map(int, row)) for row in neighbors]
        self.initial_ensemble = int(initial_ensemble)

    def auto_tune(self, **kwargs):
        self.autotune_calls.append(dict(kwargs))
        return {"log_g": np.zeros(len(self.masks), dtype=np.float64)}

    def run_production(self, *, n_steps, block_size=None):
        self.production_calls.append({"n_steps": int(n_steps), "block_size": block_size})
        return {"mode": "production", "n_steps": int(n_steps), "n_ensembles": len(self.masks)}

    def run_until_target_error(self, **kwargs):
        self.target_error_calls.append(dict(kwargs))
        return {"mode": "target_error", "kwargs": dict(kwargs), "n_ensembles": len(self.masks)}


def _test_ratio_workflow_builds_all_kp_regions_and_site_orders():
    fake_region_runner = _FakeRegionRunner()
    workflow = KagomeKPRatioWorkflow(kp_runner=KPRatioRunner(region_runner=fake_region_runner))
    bond_sites = np.array([[0, 1], [1, 2]], dtype=np.int32)
    result = workflow.run(
        nx=8,
        ny=8,
        m=2,
        bond_sites=bond_sites,
        n_therm=1,
        n_measure=2,
    )
    assert result.spec.center_label == "C27"  # auto-picked geometric centre of 8x8 (i=j=3)
    assert set(result.spec.region_masks) == {"A", "B", "C", "AB", "BC", "CA", "ABC"}
    assert len(fake_region_runner.calls) == 7
    for call in fake_region_runner.calls:
        assert call["site_order"] is not None
        assert int(np.sum(call["region_mask"])) == call["site_order"].size


def _test_expanded_workflow_builds_incremental_ladder():
    drivers = {}

    def factory(region_name):
        driver = _FakeExpandedDriver()
        drivers[str(region_name)] = driver
        return driver

    workflow = KagomeKPExpandedWorkflow(driver_factory=factory)
    bond_sites = np.array([[0, 1], [1, 2]], dtype=np.int32)
    result = workflow.run_region(
        "A",
        nx=8,
        ny=8,
        m=2,
        bond_sites=bond_sites,
        autotune_steps_per_iter=10,
        n_steps=100,
        block_size=20,
    )
    driver = drivers["A"]
    assert result.spec.center_label == "C27"  # auto-picked geometric centre of 8x8 (i=j=3)
    assert result.ladder.target_ensemble == int(np.sum(result.spec.region_masks["A"]))
    assert len(driver.masks) == result.ladder.target_ensemble + 1
    assert driver.neighbors[0] == [1]
    assert driver.neighbors[-1] == [len(driver.neighbors) - 2]
    assert len(driver.autotune_calls) == 1
    assert len(driver.production_calls) == 1


def main():
    _test_ratio_workflow_builds_all_kp_regions_and_site_orders()
    _test_expanded_workflow_builds_incremental_ladder()
    print("KP workflow integration checks passed")


if __name__ == "__main__":
    main()
