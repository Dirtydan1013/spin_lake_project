from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from src.kp.kp_geometry import build_kp_region_masks
from src.kp.kp_tee_job import _parse_regions, _write_geometry_json


def _test_parse_regions_default_and_dedup():
    assert _parse_regions("") == ["A", "B", "C", "AB", "BC", "CA", "ABC"]
    assert _parse_regions("a,b,a,abc") == ["A", "B", "ABC"]


def _test_parse_regions_rejects_unknown():
    try:
        _parse_regions("A,Z")
    except ValueError as exc:
        assert "unknown KP region" in str(exc)
    else:
        raise AssertionError("expected unknown region to raise ValueError")


def _test_write_geometry_json_roundtrip():
    spec = build_kp_region_masks(8, 8, m=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = _write_geometry_json(
            Path(tmpdir) / "geom.json",
            spec=spec,
            params={"nx": 8, "ny": 8, "m": 1},
        )
        payload = json.loads(Path(out).read_text(encoding="utf-8"))
        assert payload["params"]["nx"] == 8
        assert payload["geometry"]["center_label"] == spec.center_label
        assert "A" in payload["geometry"]["region_indices"]
        assert isinstance(payload["geometry"]["outer_paths"], list)
        assert isinstance(payload["geometry"]["branch_paths"], list)


def main():
    _test_parse_regions_default_and_dedup()
    _test_parse_regions_rejects_unknown()
    _test_write_geometry_json_roundtrip()
    print("KP TEE job unit checks passed")


if __name__ == "__main__":
    main()
