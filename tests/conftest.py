"""Pytest configuration: ensure ``src.`` is importable from tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Also expose the build directory so ``import qaqmc_cpp`` succeeds for tests
# that exercise the C++ engine.  QAQMC_TEST_BUILD_DIR lets A/B and sanitizer
# jobs select an isolated extension even when a stale module exists at repo
# root.  Move the selected build to index 0 rather than merely appending it.
_BUILD_DIR = Path(os.environ.get("QAQMC_TEST_BUILD_DIR", _REPO_ROOT / "build"))
if _BUILD_DIR.exists():
    _build_str = str(_BUILD_DIR)
    while _build_str in sys.path:
        sys.path.remove(_build_str)
    sys.path.insert(0, _build_str)
