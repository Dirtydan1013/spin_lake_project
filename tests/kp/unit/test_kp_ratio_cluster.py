from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.kp.kp_ratio_cluster import build_kp_ratio_manifest, count_manifest_jobs


def _test_build_manifest_has_expected_job_count_for_m1():
    with tempfile.TemporaryDirectory() as tmpdir:
        payload = build_kp_ratio_manifest(nx=8, ny=8, m=1, output_dir=tmpdir)
        assert payload["n_jobs"] == 72
        assert count_manifest_jobs(payload["manifest_path"]) == 72
        summary = json.loads(Path(payload["summary_path"]).read_text(encoding="utf-8"))
        assert summary["region_job_counts"]["A"] == 6
        assert summary["region_job_counts"]["ABC"] == 18


def main():
    _test_build_manifest_has_expected_job_count_for_m1()
    print("KP ratio cluster unit checks passed")


if __name__ == "__main__":
    main()
