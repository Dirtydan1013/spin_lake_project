"""Backwards-compat shim tests for ``src.mpi.kp_tee_job_mpi``.

The shim must continue to:
- Accept ``--method {ratio,expanded}`` and route correctly.
- Re-export ``run_ratio_job_mpi`` and ``run_expanded_job_mpi`` so any external
  Python caller importing them keeps working.
- Reject calls without ``--method`` (existing slurm scripts always pass it).
"""

from __future__ import annotations

import pytest

import src.mpi.kp_tee_job_mpi as shim
from src.mpi.kp_tee_ratio_mpi import run_ratio_job_mpi as ratio_canonical
from src.mpi.kp_tee_expanded_mpi import run_expanded_job_mpi as expanded_canonical


REQUIRED = ["--nx", "4", "--ny", "4", "--m", "1", "--M", "1000",
            "--output_dir", "/tmp/x"]


class TestShimReExports:
    def test_ratio_reexport_is_canonical(self):
        assert shim.run_ratio_job_mpi is ratio_canonical

    def test_expanded_reexport_is_canonical(self):
        assert shim.run_expanded_job_mpi is expanded_canonical


class TestShimParser:
    def test_method_required(self):
        parser = shim.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(REQUIRED)  # missing --method

    def test_method_choices(self):
        parser = shim.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--method", "bogus"] + REQUIRED)

    def test_method_ratio_accepts_ratio_flags(self):
        parser = shim.build_parser()
        args = parser.parse_args(["--method", "ratio"] + REQUIRED + [
            "--n_therm", "5000", "--n_measure", "10000"
        ])
        assert args.method == "ratio"
        assert args.n_therm == 5000
        assert args.n_measure == 10000

    def test_method_expanded_accepts_expanded_flags(self):
        parser = shim.build_parser()
        args = parser.parse_args(["--method", "expanded"] + REQUIRED + [
            "--regions", "A,B,C", "--autotune_max_iters", "12"
        ])
        assert args.method == "expanded"
        assert args.regions == "A,B,C"
        assert args.autotune_max_iters == 12

    def test_shim_accepts_union_of_flags(self):
        """Legacy slurm scripts may pass any combination; shim must tolerate."""
        parser = shim.build_parser()
        # cross-method flags coexist (the unused side is just ignored)
        args = parser.parse_args(["--method", "ratio"] + REQUIRED + [
            "--regions", "A,B,C",  # expanded flag, harmless under ratio
        ])
        assert args.method == "ratio"
        assert args.regions == "A,B,C"
