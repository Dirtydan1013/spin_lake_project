from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.tee.compose_tee import load_kp_result_hdf5
from src.kp.kp_ratio_cluster import build_kp_ratio_manifest, collect_manifest_results
from src.tee.qaqmc_renyi_ratio import load_ratio_manifest_hdf5, save_ratio_result_hdf5


def _test_collect_manifest_results_from_fake_ratio_outputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        payload = build_kp_ratio_manifest(nx=8, ny=8, m=1, output_dir=tmpdir)
        manifest = load_ratio_manifest_hdf5(payload["manifest_path"])
        for job in manifest["jobs"]:
            save_ratio_result_hdf5(
                job.output_path,
                region_name=job.region_name,
                step_index=job.step_index,
                A_mask=job.A_mask,
                next_site=job.next_site,
                result=type("R", (), {
                    "visit_count_low": 100,
                    "visit_count_high": 90,
                    "ratio": 0.9,
                    "ratio_err": 0.01,
                    "block_visit_count_low": np.array([50, 50], dtype=np.int64),
                    "block_visit_count_high": np.array([45, 45], dtype=np.int64),
                })(),
            )
        collected = collect_manifest_results(
            manifest_path=payload["manifest_path"],
            output_dir=tmpdir,
        )
        assert Path(collected["kp_result_path"]).exists()
        kp = load_kp_result_hdf5(collected["kp_result_path"])
        assert set(kp.region_summaries.keys()) == {"A", "B", "C", "AB", "BC", "CA", "ABC"}


def main():
    _test_collect_manifest_results_from_fake_ratio_outputs()
    print("KP ratio cluster integration checks passed")


if __name__ == "__main__":
    main()
