"""Slurm-facing CUDA production runner smoke and resume test."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

qaqmc_cuda = pytest.importorskip("qaqmc_cuda")
pytest.importorskip("qaqmc_cpp")
if not qaqmc_cuda.is_available():
    pytest.skip("CUDA device is not available", allow_module_level=True)


def test_rank_local_profile_output_and_checkpoint_resume(tmp_path):
    root = Path(__file__).resolve().parents[3]
    script = root / "main_scripts/python_scripts/run_qaqmc_cuda.py"
    run_dir = tmp_path / "run"
    base = [
        sys.executable, str(script),
        "--lattice", "kagome_bond", "--boundary", "periodic",
        "--nx", "4", "--ny", "4", "--a", "4.0",
        "--M", "200", "--Rb", "2.4",
        "--delta-min", "-2", "--delta-max", "6",
        "--neighbor-cutoff", "1", "--delta-groups", "20",
        "--profile-step", "40", "--n-equil", "2",
        "--checkpoint", "2", "--seed", "91", "--rank", "0",
        "--occ-sf-grid-n", "2", "--occ-sf-deltas", "0.0",
        "--run-dir", str(run_dir),
    ]
    environment = dict(os.environ)
    subprocess.run(base + ["--n-samples", "3"], cwd=tmp_path,
                   env=environment, check=True, timeout=120)
    subprocess.run(base + ["--n-samples", "4"], cwd=tmp_path,
                   env=environment, check=True, timeout=120)

    assert (run_dir / "rank0.checkpoint.npz").is_file()
    with h5py.File(run_dir / "rank0.h5", "r") as handle:
        assert int(np.asarray(handle["bins/sample_count"]).sum()) == 4
        assert handle["bins/density"].shape[1] == 10
        assert handle["bins/Z_l"].shape[0] == 3
        assert handle["bins/M_vbs"].shape[1] == 10
        assert handle["bins/occ_S_full_re"].shape[-2:] == (6, 6)
        assert handle["bins/occ_nprof"].shape[-1] == 96

