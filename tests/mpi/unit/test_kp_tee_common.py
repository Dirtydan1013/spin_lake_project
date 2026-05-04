"""Unit tests for ``src.mpi.kp_tee_common`` helpers.

These tests do not invoke MPI or any MC engine; they exercise pure helpers
that both the ratio and expanded entry points depend on.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.mpi.kp_tee_common import (
    _geometry_payload,
    _resolve_total_per_rank,
    _write_geometry_json,
    add_common_args,
    normalize_common_args,
)


# ----------------------------------------------------------------------------
# _resolve_total_per_rank
# ----------------------------------------------------------------------------

class TestResolveTotalPerRank:
    def test_only_per_rank_returns_per_rank(self):
        assert _resolve_total_per_rank(100, -1, 8, name="x") == 100

    def test_only_total_divides_ceil(self):
        # 100 / 8 = 12.5 -> ceil = 13
        assert _resolve_total_per_rank(-1, 100, 8, name="x") == 13

    def test_only_total_exact_division(self):
        assert _resolve_total_per_rank(-1, 80, 8, name="x") == 10

    def test_neither_returns_sentinel(self):
        assert _resolve_total_per_rank(-1, -1, 8, name="x") == -1

    def test_neither_returns_zero_with_zero_sentinel(self):
        assert _resolve_total_per_rank(0, -1, 8, name="x", sentinel=0) == 0

    def test_min_steps_default_zero_sentinel_uses_zero(self):
        # min_steps semantics: sentinel=0 means "0 means unset"
        assert _resolve_total_per_rank(0, -1, 8, name="min_steps", sentinel=0) == 0

    def test_min_steps_per_rank_above_sentinel(self):
        assert _resolve_total_per_rank(50, -1, 8, name="min_steps", sentinel=0) == 50

    def test_both_set_raises(self):
        with pytest.raises(ValueError, match="not both"):
            _resolve_total_per_rank(50, 200, 8, name="n_steps")

    def test_zero_n_ranks_with_total_raises(self):
        with pytest.raises(ValueError, match="n_ranks"):
            _resolve_total_per_rank(-1, 100, 0, name="x")


# ----------------------------------------------------------------------------
# add_common_args / normalize_common_args
# ----------------------------------------------------------------------------

def _make_parser_with_common():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    return parser


REQUIRED_FLAGS = ["--nx", "1", "--ny", "1", "--m", "1", "--M", "100",
                  "--output_dir", "/tmp/x"]


class TestAddCommonArgs:
    def test_required_flags_enforced(self):
        parser = _make_parser_with_common()
        with pytest.raises(SystemExit):
            parser.parse_args([])  # missing all required

    def test_minimal_args_parse(self):
        parser = _make_parser_with_common()
        args = parser.parse_args(REQUIRED_FLAGS)
        assert args.nx == 1
        assert args.ny == 1
        assert args.m == 1
        assert args.M == 100
        assert args.output_dir == "/tmp/x"

    def test_default_values(self):
        parser = _make_parser_with_common()
        args = parser.parse_args(REQUIRED_FLAGS)
        # Defaults that downstream code depends on
        assert args.a == 1.0
        assert args.Omega == 1.0
        assert args.Rb == 1.2
        assert args.delta_min == 0.0
        assert args.delta_max == 1.0
        assert args.epsilon == 0.01
        assert args.seed == 42
        assert args.neighbor_cutoff == -1
        assert args.delta_groups == 0
        assert args.preferred_center_label == "auto"
        assert args.block_size == -1

    def test_lattice_choice_constraint(self):
        parser = _make_parser_with_common()
        with pytest.raises(SystemExit):
            parser.parse_args(REQUIRED_FLAGS + ["--lattice", "bogus"])

    def test_lattice_default(self):
        parser = _make_parser_with_common()
        args = parser.parse_args(REQUIRED_FLAGS)
        # default is DEFAULT_LATTICE; just check it's a known choice
        assert args.lattice in {"kagome_bond", "kagome_bond_triangle"}


class TestNormalizeCommonArgs:
    def test_neighbor_cutoff_negative_becomes_none(self):
        parser = _make_parser_with_common()
        args = parser.parse_args(REQUIRED_FLAGS + ["--neighbor_cutoff", "-1"])
        out = normalize_common_args(args)
        assert out["neighbor_cutoff"] is None

    def test_neighbor_cutoff_zero_or_positive_kept(self):
        parser = _make_parser_with_common()
        args = parser.parse_args(REQUIRED_FLAGS + ["--neighbor_cutoff", "3"])
        out = normalize_common_args(args)
        assert out["neighbor_cutoff"] == 3

    def test_block_size_negative_becomes_none(self):
        parser = _make_parser_with_common()
        args = parser.parse_args(REQUIRED_FLAGS + ["--block_size", "-1"])
        out = normalize_common_args(args)
        assert out["block_size"] is None

    def test_block_size_positive_kept(self):
        parser = _make_parser_with_common()
        args = parser.parse_args(REQUIRED_FLAGS + ["--block_size", "500"])
        out = normalize_common_args(args)
        assert out["block_size"] == 500

    @pytest.mark.parametrize("token", ["", "auto", "AUTO", "Auto", "none", "None", "  "])
    def test_preferred_center_label_sentinels_become_none(self, token):
        parser = _make_parser_with_common()
        args = parser.parse_args(REQUIRED_FLAGS + ["--preferred_center_label", token])
        out = normalize_common_args(args)
        assert out["preferred_center_label"] is None

    def test_preferred_center_label_concrete_kept(self):
        parser = _make_parser_with_common()
        args = parser.parse_args(REQUIRED_FLAGS + ["--preferred_center_label", "C12"])
        out = normalize_common_args(args)
        assert out["preferred_center_label"] == "C12"


# ----------------------------------------------------------------------------
# _geometry_payload + _write_geometry_json
# ----------------------------------------------------------------------------

def _fake_spec():
    """Minimal duck-typed spec object covering the fields _geometry_payload reads."""
    masks = {
        "A": np.array([1, 0, 1, 0], dtype=np.uint8),
        "B": np.array([0, 1, 0, 1], dtype=np.uint8),
    }
    indices = {"A": [0, 2], "B": [1, 3]}
    site_orders = {"A": [0, 2], "B": [1, 3]}
    return SimpleNamespace(
        center_label="C0",
        region_indices=indices,
        region_masks=masks,
        site_orders=site_orders,
        outer_paths=[("C0", "K1", "C1"), ("C1", "K2", "C2")],
        branch_paths=[("C0", "K3", "C3")],
    )


class TestGeometryPayload:
    def test_basic_shape(self):
        spec = _fake_spec()
        payload = _geometry_payload(spec)
        assert payload["center_label"] == "C0"
        assert payload["region_indices"] == {"A": [0, 2], "B": [1, 3]}
        assert payload["region_sizes"] == {"A": 2, "B": 2}
        assert payload["site_orders"] == {"A": [0, 2], "B": [1, 3]}
        assert payload["outer_paths"] == [["C0", "K1", "C1"], ["C1", "K2", "C2"]]
        assert payload["branch_paths"] == [["C0", "K3", "C3"]]

    def test_site_orders_none(self):
        spec = _fake_spec()
        spec.site_orders = None
        assert _geometry_payload(spec)["site_orders"] is None

    def test_region_indices_handle_numpy_arrays(self):
        spec = _fake_spec()
        spec.region_indices = {"A": np.array([0, 2], dtype=np.int64),
                               "B": np.array([1, 3], dtype=np.int64)}
        out = _geometry_payload(spec)
        # Must serialize as plain lists of ints, not numpy
        assert out["region_indices"]["A"] == [0, 2]
        assert all(isinstance(x, int) for x in out["region_indices"]["A"])


class TestWriteGeometryJson:
    def test_creates_parent_dir_and_writes_valid_json(self, tmp_path):
        target = tmp_path / "nested" / "dir" / "geo.json"
        out = _write_geometry_json(target, spec=_fake_spec(),
                                   params={"method": "ratio", "M": 100})
        assert out == target
        assert target.exists()
        loaded = json.loads(target.read_text())
        assert loaded["params"]["method"] == "ratio"
        assert loaded["params"]["M"] == 100
        assert loaded["geometry"]["center_label"] == "C0"

    def test_overwrites_existing_file(self, tmp_path):
        target = tmp_path / "geo.json"
        target.write_text('{"stale": true}')
        _write_geometry_json(target, spec=_fake_spec(), params={"v": 1})
        loaded = json.loads(target.read_text())
        assert "stale" not in loaded
        assert loaded["params"]["v"] == 1
