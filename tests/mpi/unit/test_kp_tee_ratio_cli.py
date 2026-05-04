"""CLI / argparse tests for ``src.mpi.kp_tee_ratio_mpi``.

We exercise ``build_parser`` directly so MPI is never initialised here.
``run_ratio_job_mpi`` is re-exported from this module — confirm that too.
"""

from __future__ import annotations

import pytest

from src.mpi.kp_tee_ratio_mpi import build_parser, run_ratio_job_mpi


REQUIRED = ["--nx", "4", "--ny", "4", "--m", "1", "--M", "1000",
            "--output_dir", "/tmp/x"]


class TestRatioParserSurface:
    def test_no_method_flag(self):
        # The whole point of splitting: ratio entry point must NOT carry --method
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(REQUIRED + ["--method", "ratio"])

    def test_no_expanded_only_flags(self):
        """The ratio parser must not accept expanded-specific flags.

        Even after adding the Day-3 modes (--log_g_init, --skip_autotune),
        these must remain expanded-only.  This is a regression guard.
        """
        parser = build_parser()
        for flag in [
            "--regions", "--autotune_steps_per_iter", "--autotune_max_iters",
            "--autotune_tol", "--autotune_method", "--autotune_damping",
            "--n_steps", "--n_steps_total", "--target_s2_err",
            "--batch_steps", "--max_steps", "--min_steps", "--estimator",
            "--log_g_init", "--skip_autotune", "--warm_up_steps",
        ]:
            with pytest.raises(SystemExit):
                parser.parse_args(REQUIRED + [flag, "0"])

    def test_minimum_args_parse(self):
        parser = build_parser()
        args = parser.parse_args(REQUIRED)
        assert args.nx == 4 and args.ny == 4 and args.m == 1 and args.M == 1000

    def test_ratio_specific_defaults(self):
        parser = build_parser()
        args = parser.parse_args(REQUIRED)
        assert args.n_therm == 2000
        assert args.n_measure == 50000
        assert args.n_measure_total == -1
        assert args.measure_stride == 1

    def test_n_measure_overrides(self):
        parser = build_parser()
        args = parser.parse_args(REQUIRED + ["--n_measure", "1234"])
        assert args.n_measure == 1234

    def test_n_measure_total_overrides(self):
        parser = build_parser()
        args = parser.parse_args(REQUIRED + ["--n_measure_total", "8000"])
        assert args.n_measure_total == 8000


class TestRatioReExports:
    def test_run_ratio_job_mpi_callable(self):
        # We don't call it (would need MPI + engine), just ensure it's
        # importable from the entry-point module so external scripts that
        # import it continue to work.
        assert callable(run_ratio_job_mpi)
