"""Load saved ``log_g`` weights from expanded-ensemble HDF5 outputs.

Used by the frozen-production / resume-tune modes of the expanded entry
point (see :mod:`src.mpi.kp_tee_expanded_mpi`).  The contract:

- ``log_g`` indices are positionally tied to the mask order in the new
  ladder.  We refuse to load a file whose stored masks differ from the
  caller's expected masks.
- Length must match.  No NaN.  Returned array is a writable copy
  (``set_log_g`` mutates).
- Physical-parameter mismatches (M, lattice, ...) are surfaced as
  ``UserWarning`` so the caller is told but not blocked.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np


def load_region_log_g(
    h5_path,
    expected_masks,
    *,
    expected_params: dict | None = None,
) -> np.ndarray:
    """Read ``result/log_g`` from a region h5, validating positional consistency.

    Parameters
    ----------
    h5_path : path-like
        Path to a region HDF5 produced by ``save_expanded_result_hdf5``.
    expected_masks : array-like, shape ``(n_windows, n_sites)``
        The mask ladder of the *new* run.  log_g[k] applies to the k-th mask;
        if the stored masks differ from these, the weights are not safe to
        reuse and we refuse to load.
    expected_params : dict, optional
        Physical params (e.g. ``{"M": 1000, "lattice": "kagome_bond"}``) the
        caller believes the saved tune used.  Mismatches warn but do not raise.

    Returns
    -------
    log_g : np.ndarray, shape ``(n_windows,)``, dtype ``float64``, writable.

    Raises
    ------
    FileNotFoundError
        If ``h5_path`` does not exist.
    ValueError
        If sizes don't match, masks don't match, or any log_g entry is NaN.
    """
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"log_g h5 not found: {path}")

    expected_masks_arr = np.ascontiguousarray(expected_masks, dtype=np.uint8)
    if expected_masks_arr.ndim != 2:
        raise ValueError("expected_masks must be a 2D array (n_windows, n_sites)")
    n_windows = expected_masks_arr.shape[0]

    with h5py.File(path, "r") as h5:
        if "result/log_g" not in h5:
            raise ValueError(f"{path} has no result/log_g dataset")
        log_g = np.asarray(h5["result/log_g"][:], dtype=np.float64)

        if log_g.size != n_windows:
            raise ValueError(
                f"log_g length {log_g.size} does not match expected ladder "
                f"length {n_windows}; are you loading the correct region?"
            )

        if "manifest/masks" in h5:
            stored_masks = np.asarray(h5["manifest/masks"][:], dtype=np.uint8)
            if stored_masks.shape != expected_masks_arr.shape:
                raise ValueError(
                    f"saved mask shape {stored_masks.shape} does not match "
                    f"expected ladder mask shape {expected_masks_arr.shape}"
                )
            if not np.array_equal(stored_masks, expected_masks_arr):
                raise ValueError(
                    f"saved masks in {path} differ from the expected ladder; "
                    "log_g indices are tied to mask order so reuse is unsafe"
                )

        if expected_params and "manifest" in h5:
            attrs = dict(h5["manifest"].attrs)
            for key, expected_val in expected_params.items():
                if key not in attrs:
                    continue
                stored = attrs[key]
                # Decode bytes (h5 string attrs may come back as bytes)
                if isinstance(stored, bytes):
                    stored = stored.decode("utf-8", errors="replace")
                if stored != expected_val:
                    warnings.warn(
                        f"saved log_g param mismatch in {path.name}: "
                        f"{key}={stored!r} (saved) vs {expected_val!r} (expected); "
                        "loading anyway",
                        UserWarning,
                        stacklevel=2,
                    )

    if np.any(np.isnan(log_g)):
        raise ValueError(f"NaN found in log_g loaded from {path}")
    if np.any(np.isinf(log_g)):
        raise ValueError(f"Inf found in log_g loaded from {path}")

    return np.ascontiguousarray(log_g, dtype=np.float64)
