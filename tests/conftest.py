"""Pytest configuration: ensure ``src.`` is importable from tests."""

from __future__ import annotations

import os
import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Also expose the build directory so ``import qaqmc_cpp`` succeeds for tests
# that exercise the C++ engine.  QAQMC_TEST_BUILD_DIR lets A/B and sanitizer
# jobs select an isolated extension even when a stale module exists at repo
# root.  Move the selected build to index 0 rather than merely appending it.
_BUILD_OVERRIDE = os.environ.get("QAQMC_TEST_BUILD_DIR")
_BUILD_DIR = Path(_BUILD_OVERRIDE or (_REPO_ROOT / "build")).resolve()
if _BUILD_DIR.exists():
    _build_str = str(_BUILD_DIR)
    while _build_str in sys.path:
        sys.path.remove(_build_str)
    sys.path.insert(0, _build_str)
    # Several legacy test modules prepend the repository root during
    # collection.  Pin an explicitly selected extension now, before those
    # imports can silently load a stale repo-root .so into sys.modules.
    if _BUILD_OVERRIDE:
        _qaqmc_cpp = importlib.import_module("qaqmc_cpp")
        _loaded_dir = Path(_qaqmc_cpp.__file__).resolve().parent
        if _loaded_dir != _BUILD_DIR:
            raise RuntimeError(
                f"QAQMC_TEST_BUILD_DIR selected {_BUILD_DIR}, but pytest loaded "
                f"qaqmc_cpp from {_loaded_dir}")

# Optional CUDA build is kept separate so CPU-only configure/build remains
# untouched.  GPU tests import qaqmc_cuda from here when the module has been
# built.  Appended AFTER the selected CPU build so a qaqmc_cpp compiled inside
# build_cuda can never shadow the explicitly pinned CPU extension above.
_CUDA_BUILD_DIR = _REPO_ROOT / "build_cuda"
if _CUDA_BUILD_DIR.exists() and str(_CUDA_BUILD_DIR) not in sys.path:
    sys.path.append(str(_CUDA_BUILD_DIR))
