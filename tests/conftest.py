"""Pytest configuration: ensure ``src.`` is importable from tests."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Also expose the build directory so ``import qaqmc_cpp`` succeeds for tests
# that exercise the C++ engine.  Tests that don't need it remain unaffected.
_BUILD_DIR = _REPO_ROOT / "build"
if _BUILD_DIR.exists() and str(_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_DIR))

# Optional CUDA build is kept separate so CPU-only configure/build remains
# untouched.  GPU tests import from here when the module has been built.
_CUDA_BUILD_DIR = _REPO_ROOT / "build_cuda"
if _CUDA_BUILD_DIR.exists() and str(_CUDA_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_CUDA_BUILD_DIR))
