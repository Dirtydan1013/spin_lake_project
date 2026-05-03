"""Skeleton tests for ``src.tee.log_g_io`` (Day-3 module).

The contract:

- ``load_region_log_g(h5_path, expected_masks)`` reads ``result/log_g`` from a
  region HDF5 file produced by an earlier expanded-ensemble run.
- It validates length matches ``len(expected_masks)``.
- It validates masks (when present in the h5 manifest) byte-equal
  ``expected_masks`` — log_g indices are positionally tied to mask order.
- It rejects NaN values.
- Returns a writable ``np.ndarray`` of dtype float64.

These are ``xfail(strict=True)`` until Day 3.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test fixtures: a tiny h5 in the schema produced by save_expanded_result_hdf5
# ---------------------------------------------------------------------------

def _write_region_h5(path, *, log_g, masks=None, params=None):
    """Write the minimum schema load_region_log_g cares about."""
    masks = np.asarray(masks if masks is not None
                       else np.zeros((len(log_g), 4), dtype=np.uint8))
    with h5py.File(path, "w") as h5:
        manifest = h5.create_group("manifest")
        manifest.create_dataset("masks", data=masks.astype(np.uint8))
        if params:
            for k, v in params.items():
                manifest.attrs[k] = v
        result = h5.create_group("result")
        result.create_dataset("log_g", data=np.asarray(log_g, dtype=np.float64))


@pytest.fixture
def good_h5(tmp_path):
    path = tmp_path / "kp_expanded_A.h5"
    masks = np.array(
        [[0, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]], dtype=np.uint8,
    )
    _write_region_h5(path, log_g=[0.0, 0.5, 1.2, 2.0, 2.7], masks=masks,
                     params={"M": 1000, "lattice": "kagome_bond"})
    return path, masks


@pytest.mark.xfail(strict=True, reason="log_g_io implemented in Day 3")
class TestLogGIoModuleExists:
    def test_can_import_load_region_log_g(self):
        from src.tee.log_g_io import load_region_log_g  # noqa: F401


@pytest.mark.xfail(strict=True, reason="log_g_io implemented in Day 3")
class TestLoadRegionLogG:
    def test_returns_float64_array(self, good_h5):
        from src.tee.log_g_io import load_region_log_g
        path, masks = good_h5
        out = load_region_log_g(path, expected_masks=masks)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert out.shape == (5,)

    def test_values_match_what_was_written(self, good_h5):
        from src.tee.log_g_io import load_region_log_g
        path, masks = good_h5
        out = load_region_log_g(path, expected_masks=masks)
        np.testing.assert_allclose(out, [0.0, 0.5, 1.2, 2.0, 2.7])

    def test_size_mismatch_raises(self, good_h5):
        from src.tee.log_g_io import load_region_log_g
        path, masks = good_h5
        wrong = masks[:3]  # only 3 windows
        with pytest.raises(ValueError, match="length"):
            load_region_log_g(path, expected_masks=wrong)

    def test_mask_mismatch_raises(self, good_h5):
        from src.tee.log_g_io import load_region_log_g
        path, masks = good_h5
        bad_masks = masks.copy()
        bad_masks[1, 0] = 0  # window 1 should have site 0 set; flip it
        with pytest.raises(ValueError, match="mask"):
            load_region_log_g(path, expected_masks=bad_masks)

    def test_nan_in_log_g_raises(self, tmp_path):
        from src.tee.log_g_io import load_region_log_g
        path = tmp_path / "bad.h5"
        masks = np.zeros((3, 4), dtype=np.uint8)
        _write_region_h5(path, log_g=[0.0, np.nan, 1.0], masks=masks)
        with pytest.raises(ValueError, match="NaN"):
            load_region_log_g(path, expected_masks=masks)

    def test_missing_file_raises(self, tmp_path):
        from src.tee.log_g_io import load_region_log_g
        with pytest.raises(FileNotFoundError):
            load_region_log_g(tmp_path / "nope.h5",
                              expected_masks=np.zeros((3, 4), dtype=np.uint8))

    def test_metadata_mismatch_warns_but_loads(self, good_h5):
        from src.tee.log_g_io import load_region_log_g
        path, masks = good_h5
        # Same masks/length but different physics params: should warn, not raise.
        with pytest.warns(UserWarning, match="(M|lattice)"):
            out = load_region_log_g(path, expected_masks=masks,
                                    expected_params={"M": 9999})
        assert out.shape == (5,)

    def test_returned_array_is_writable(self, good_h5):
        """Caller often wants to pass log_g to set_log_g, which mutates."""
        from src.tee.log_g_io import load_region_log_g
        path, masks = good_h5
        out = load_region_log_g(path, expected_masks=masks)
        out[0] = 999.0  # must not raise
