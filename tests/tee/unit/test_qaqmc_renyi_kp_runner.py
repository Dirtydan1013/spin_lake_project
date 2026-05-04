import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tee.compose_tee import (
    KPResult,
    load_kp_result_hdf5,
    load_region_summary_hdf5,
    save_kp_result_hdf5,
    save_region_summary_hdf5,
    summarize_region,
)
from src.tee.qaqmc_renyi_ratio import KPRatioRunner, RegionRunResult


class _FakeRegionRunner:
    def __init__(self, summaries):
        self.summaries = summaries
        self.calls = []

    def run_region(self, region_name, region_mask, *, bond_sites=None, site_order=None,
                   n_therm, n_measure, measure_stride=1, block_size=None):
        self.calls.append((region_name, np.array(region_mask, dtype=np.uint8), None if site_order is None else list(site_order)))
        summary = self.summaries[region_name]
        return RegionRunResult(
            region_name=region_name,
            site_order=summary.sites_ordered,
            ratio_results=[],
            summary=summary,
        )


def _make_summary(region_name, value):
    ratio = np.exp(-value)
    return summarize_region(region_name, [0], [ratio], [0.1 * ratio])


def _test_kp_runner_composes_regions():
    summaries = {
        "A": _make_summary("A", 1.0),
        "B": _make_summary("B", 1.1),
        "C": _make_summary("C", 1.2),
        "AB": _make_summary("AB", 2.0),
        "BC": _make_summary("BC", 2.1),
        "CA": _make_summary("CA", 2.2),
        "ABC": _make_summary("ABC", 3.0),
    }
    fake_region_runner = _FakeRegionRunner(summaries)
    kp_runner = KPRatioRunner(region_runner=fake_region_runner)
    masks = {name: np.array([1], dtype=np.uint8) for name in summaries}
    result = kp_runner.run_kp(masks, n_therm=1, n_measure=2)
    expected_gamma = 1.0 + 1.1 + 1.2 - 2.0 - 2.1 - 2.2 + 3.0
    assert np.isclose(result.kp_result.gamma, expected_gamma)
    assert set(result.region_runs) == set(summaries)


def _test_region_and_kp_hdf5_roundtrip():
    summary = summarize_region("A", [1, 2], [0.5, 0.25], [0.05, 0.025])
    kp_result = KPResult(
        region_summaries={
            name: summarize_region(name, [0], [0.5], [0.05])
            for name in ("A", "B", "C", "AB", "BC", "CA", "ABC")
        },
        gamma=0.7,
        gamma_err=0.12,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        region_path = os.path.join(tmpdir, "region.h5")
        kp_path = os.path.join(tmpdir, "kp.h5")
        save_region_summary_hdf5(region_path, summary)
        loaded_region = load_region_summary_hdf5(region_path)
        assert loaded_region.region_name == "A"
        assert loaded_region.sites_ordered.tolist() == [1, 2]
        save_kp_result_hdf5(kp_path, kp_result)
        loaded_kp = load_kp_result_hdf5(kp_path)
        assert np.isclose(loaded_kp.gamma, 0.7)
        assert np.isclose(loaded_kp.gamma_err, 0.12)
        assert set(loaded_kp.region_summaries) == set(kp_result.region_summaries)


def main():
    _test_kp_runner_composes_regions()
    _test_region_and_kp_hdf5_roundtrip()
    print("QAQMC Renyi KP runner unit checks passed")


if __name__ == "__main__":
    main()
