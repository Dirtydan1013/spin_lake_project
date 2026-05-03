"""CLI / argparse tests for ``src.mpi.kp_tee_expanded_mpi``."""

from __future__ import annotations

import pytest

from src.mpi.kp_tee_expanded_mpi import build_parser, run_expanded_job_mpi


REQUIRED = ["--nx", "4", "--ny", "4", "--m", "1", "--M", "1000",
            "--output_dir", "/tmp/x"]


class TestExpandedParserSurface:
    def test_no_method_flag(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(REQUIRED + ["--method", "expanded"])

    def test_no_ratio_only_flags(self):
        """Ratio-specific flags must not appear on the expanded entry point."""
        parser = build_parser()
        for flag in ["--n_therm", "--n_measure", "--n_measure_total",
                     "--measure_stride"]:
            with pytest.raises(SystemExit):
                parser.parse_args(REQUIRED + [flag, "100"])

    def test_minimum_args_parse(self):
        parser = build_parser()
        args = parser.parse_args(REQUIRED)
        assert args.nx == 4 and args.M == 1000

    def test_expanded_defaults(self):
        parser = build_parser()
        args = parser.parse_args(REQUIRED)
        assert args.regions == ""
        assert args.autotune_steps_per_iter == 15000
        assert args.autotune_max_iters == 8
        assert args.autotune_tol == 1.15
        assert args.autotune_method == "transition_matrix"
        assert args.autotune_damping == 0.7
        assert args.n_steps == -1
        assert args.n_steps_total == -1
        assert args.target_s2_err == -1.0
        assert args.batch_steps == -1
        assert args.max_steps == -1
        assert args.min_steps == 0
        assert args.estimator == "collection"

    def test_regions_pass_through(self):
        parser = build_parser()
        args = parser.parse_args(REQUIRED + ["--regions", "A,B,C"])
        assert args.regions == "A,B,C"

    def test_autotune_overrides(self):
        parser = build_parser()
        args = parser.parse_args(REQUIRED + [
            "--autotune_max_iters", "20",
            "--autotune_tol", "1.05",
            "--autotune_damping", "0.5",
        ])
        assert args.autotune_max_iters == 20
        assert args.autotune_tol == 1.05
        assert args.autotune_damping == 0.5


class TestExpandedReExports:
    def test_run_expanded_job_mpi_callable(self):
        assert callable(run_expanded_job_mpi)
