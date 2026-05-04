import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tee.qaqmc_renyi_ratio import (
    RatioResult,
    RegionRunResult,
    build_ratio_job_specs,
    combine_region_runs,
    combine_visit_counts,
    coarsen_ratio_result_blocks,
    load_ratio_manifest_hdf5,
    scan_region_blocking,
    save_ratio_manifest_hdf5,
)
from src.tee.compose_tee import summarize_region


def _test_combine_visit_counts():
    result = combine_visit_counts([3, 5, 4], [10, 10, 10])
    assert np.isclose(result.ratio, 30.0 / 12.0)
    assert result.visit_count_low == 12
    assert result.visit_count_high == 30
    assert result.ratio_err > 0.0


def _test_build_ratio_jobs_and_manifest_roundtrip():
    region_mask = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
    jobs = build_ratio_job_specs(
        "ABC",
        region_mask,
        site_order=[1, 2, 3],
        output_dir="data/ratios",
    )
    assert [job.next_site for job in jobs] == [1, 2, 3]
    assert jobs[0].A_mask.tolist() == [0, 0, 0, 0, 0]
    assert jobs[1].A_mask.tolist() == [0, 1, 0, 0, 0]
    assert jobs[2].A_mask.tolist() == [0, 1, 1, 0, 0]
    assert jobs[1].output_path.endswith("ABC_step001.h5")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "manifest.h5")
        save_ratio_manifest_hdf5(path, jobs, params={"region_count": 1})
        loaded = load_ratio_manifest_hdf5(path)
        assert int(loaded["params"]["region_count"]) == 1
        loaded_jobs = loaded["jobs"]
        assert [job.region_name for job in loaded_jobs] == ["ABC", "ABC", "ABC"]
        assert [job.next_site for job in loaded_jobs] == [1, 2, 3]
        assert loaded_jobs[2].A_mask.tolist() == [0, 1, 1, 0, 0]


def _test_combine_region_runs():
    site_order = np.array([0, 1], dtype=np.int32)
    run0_results = [
        RatioResult(ratio=0.4, ratio_err=0.1, visit_count_low=10, visit_count_high=4),
        RatioResult(ratio=1.5, ratio_err=0.5, visit_count_low=6, visit_count_high=9),
    ]
    run1_results = [
        RatioResult(ratio=0.5, ratio_err=0.1, visit_count_low=8, visit_count_high=4),
        RatioResult(ratio=1.4, ratio_err=0.4, visit_count_low=10, visit_count_high=14),
    ]
    run0 = RegionRunResult(
        region_name="A",
        site_order=site_order,
        ratio_results=run0_results,
        summary=summarize_region(
            "A",
            site_order,
            [item.ratio for item in run0_results],
            [item.ratio_err for item in run0_results],
        ),
    )
    run1 = RegionRunResult(
        region_name="A",
        site_order=site_order,
        ratio_results=run1_results,
        summary=summarize_region(
            "A",
            site_order,
            [item.ratio for item in run1_results],
            [item.ratio_err for item in run1_results],
        ),
    )

    combined = combine_region_runs([run0, run1])
    assert combined.region_name == "A"
    assert combined.site_order.tolist() == [0, 1]
    assert combined.ratio_results[0].visit_count_low == 18
    assert combined.ratio_results[0].visit_count_high == 8
    assert combined.ratio_results[1].visit_count_low == 16
    assert combined.ratio_results[1].visit_count_high == 23
    assert np.isclose(combined.ratio_results[0].ratio, 8.0 / 18.0)
    assert np.isclose(combined.ratio_results[1].ratio, 23.0 / 16.0)
    assert np.isclose(combined.summary.S_2, -np.log((8.0 / 18.0) * (23.0 / 16.0)))


def _test_coarsen_ratio_blocks_and_scan_region():
    site_order = np.array([0, 1], dtype=np.int32)
    ratio0 = RatioResult(
        ratio=0.5,
        ratio_err=0.1,
        visit_count_low=24,
        visit_count_high=12,
        block_visit_count_low=np.array([6, 6, 6, 6], dtype=np.int64),
        block_visit_count_high=np.array([2, 4, 2, 4], dtype=np.int64),
    )
    ratio1 = RatioResult(
        ratio=1.2,
        ratio_err=0.1,
        visit_count_low=20,
        visit_count_high=24,
        block_visit_count_low=np.array([5, 5, 5, 5], dtype=np.int64),
        block_visit_count_high=np.array([4, 8, 4, 8], dtype=np.int64),
    )
    coarsened = coarsen_ratio_result_blocks(ratio0, 2)
    assert coarsened.block_visit_count_low.tolist() == [12, 12]
    assert coarsened.block_visit_count_high.tolist() == [6, 6]
    assert np.isclose(coarsened.ratio, 0.5)

    region = RegionRunResult(
        region_name="A",
        site_order=site_order,
        ratio_results=[ratio0, ratio1],
        summary=summarize_region(
            "A",
            site_order,
            [ratio0.ratio, ratio1.ratio],
            [ratio0.ratio_err, ratio1.ratio_err],
        ),
    )
    scan = scan_region_blocking(region, coarse_factors=[1, 2, 4])
    assert [entry.coarse_factor for entry in scan] == [1, 2]
    assert [entry.n_blocks for entry in scan] == [4, 2]
    assert all(np.isclose(entry.s2, -np.log(0.5 * 1.2)) for entry in scan)


def main():
    _test_combine_visit_counts()
    _test_build_ratio_jobs_and_manifest_roundtrip()
    _test_combine_region_runs()
    _test_coarsen_ratio_blocks_and_scan_region()
    print("QAQMC Renyi ratio manifest unit checks passed")


if __name__ == "__main__":
    main()
