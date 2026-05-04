from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_plot_module():
    # tests/rydberg/integration/<this file>  → repo root is parents[3].
    # (Prior layout was TEST/INTEGRATION_TEST, hence the original parents[2].)
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "python script" / "plot_kagome_dual_indices.py"
    spec = importlib.util.spec_from_file_location("plot_kagome_dual_indices", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_plot_kagome_dual_indices_writes_output(tmp_path):
    module = _load_plot_module()
    out = tmp_path / "dual_indices.png"
    path = module.plot_kagome_dual_indices(4, 4, 1.0, out=str(out))
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0
