"""CLI tests for the three modes on the expanded entry point.

- Cold tune (default): no new flags
- Resume tune: ``--log_g_init <dir>``
- Frozen production: ``--log_g_init <dir> --skip_autotune``

Plus ``--warm_up_steps <n>`` for MC-chain thermalisation before measurement.
``validate_mode_args`` enforces that ``--skip_autotune`` requires
``--log_g_init``.
"""

from __future__ import annotations

import pytest

from src.mpi.kp_tee_expanded_mpi import build_parser


REQUIRED = ["--nx", "4", "--ny", "4", "--m", "1", "--M", "1000",
            "--output_dir", "/tmp/x"]


class TestLogGInitFlag:
    def test_log_g_init_attribute_present(self):
        """After Day 3, parsing without --log_g_init still creates the attr."""
        parser = build_parser()
        args = parser.parse_args(REQUIRED)
        # Direct access (not getattr-with-default) so this raises today.
        _ = args.log_g_init  # AttributeError today

    def test_log_g_init_default_is_none(self):
        parser = build_parser()
        args = parser.parse_args(REQUIRED)
        assert args.log_g_init is None

    def test_log_g_init_accepts_path(self, tmp_path):
        parser = build_parser()
        args = parser.parse_args(REQUIRED + ["--log_g_init", str(tmp_path)])
        assert args.log_g_init == str(tmp_path)


class TestSkipAutotuneFlag:
    def test_skip_autotune_attribute_present(self):
        parser = build_parser()
        args = parser.parse_args(REQUIRED)
        _ = args.skip_autotune  # AttributeError today

    def test_skip_autotune_default_false(self):
        parser = build_parser()
        args = parser.parse_args(REQUIRED)
        assert args.skip_autotune is False

    def test_skip_autotune_accepted_with_log_g_init(self, tmp_path):
        parser = build_parser()
        args = parser.parse_args(REQUIRED + [
            "--log_g_init", str(tmp_path), "--skip_autotune"
        ])
        assert args.skip_autotune is True


class TestWarmUpStepsFlag:
    def test_warm_up_steps_attribute_present(self):
        parser = build_parser()
        args = parser.parse_args(REQUIRED)
        _ = args.warm_up_steps  # AttributeError today

    def test_warm_up_steps_default_zero_cold(self):
        parser = build_parser()
        args = parser.parse_args(REQUIRED)
        assert args.warm_up_steps == 0

    def test_warm_up_steps_explicit_overrides(self, tmp_path):
        parser = build_parser()
        args = parser.parse_args(REQUIRED + [
            "--log_g_init", str(tmp_path),
            "--warm_up_steps", "0",
        ])
        # Explicit --warm_up_steps must always win over the load-time default.
        assert args.warm_up_steps == 0

    def test_warm_up_steps_positive_accepted(self):
        parser = build_parser()
        args = parser.parse_args(REQUIRED + ["--warm_up_steps", "12345"])
        assert args.warm_up_steps == 12345


class TestSkipAutotuneRequiresLogGInit:
    """``--skip_autotune`` without ``--log_g_init`` is a user error.

    Day-3 implementation must expose ``validate_mode_args(args)`` that raises
    ``ValueError`` when ``args.skip_autotune`` is set without ``log_g_init``.
    Today the import itself fails (function doesn't exist), so the test fails
    cleanly under xfail.
    """
    def test_validator_exists_and_rejects_skip_without_init(self):
        from src.mpi.kp_tee_expanded_mpi import validate_mode_args

        parser = build_parser()
        args = parser.parse_args(REQUIRED + [
            "--log_g_init", "/tmp/nonexistent",
            "--skip_autotune",
        ])
        # With log_g_init set, validation should pass.
        validate_mode_args(args)

        # Without log_g_init, --skip_autotune is invalid.
        args.log_g_init = None
        with pytest.raises(ValueError, match="log_g_init"):
            validate_mode_args(args)
