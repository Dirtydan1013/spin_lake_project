"""Public CPU extension API gates for the modular binding layout."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


qaqmc_cpp = pytest.importorskip("qaqmc_cpp")


def test_cpu_extension_exports_all_engine_registration_units():
    expected = {
        "HalfLineProposal",
        "QAQMCModelData",
        "QAQMCEngine",
        "QAQMCRenyiEngine",
        "SSEEngine",
        "WorkTrajectoryResult",
        "WorkRunResult",
        "QAQMCRenyiWorkEngine",
        "has_openmp",
        "omp_max_threads",
    }
    missing = sorted(name for name in expected if not hasattr(qaqmc_cpp, name))
    assert not missing, f"missing modular CPU bindings: {missing}"


def test_cpu_to_cuda_model_export_keeps_gpu_wrapper_schema():
    positions = np.column_stack((np.arange(8, dtype=np.float64),
                                 np.zeros(8, dtype=np.float64)))
    engine = qaqmc_cpp.QAQMCEngine(
        8, 1.0, -2.0, 6.0, 1.2, 32, 0.01, 17, positions,
        neighbor_cutoff=-1, delta_groups=4,
    )
    data = engine.export_cuda_diagonal_data()
    expected_keys = {
        "alias_prob", "alias_index", "alias_loc_kind", "bond_rmax",
        "bond_sites", "bond_vij", "inv_coord",
    }
    assert set(data) == expected_keys
    n_bonds = 8 * 7 // 2
    max_alias = 8 + n_bonds
    assert data["alias_prob"].shape == (4, max_alias)
    assert data["alias_index"].shape == (4, max_alias)
    assert data["alias_index"].dtype == np.int32
    assert data["alias_loc_kind"].shape == (4, max_alias)
    assert data["bond_rmax"].shape == (4, n_bonds)
    assert data["bond_sites"].shape == (n_bonds, 2)
    assert data["bond_vij"].shape == (n_bonds,)
    assert data["inv_coord"].shape == (8,)
    expected_loc = np.concatenate((
        2 * np.arange(8, dtype=np.int32),
        2 * np.arange(n_bonds, dtype=np.int32) + 1,
    ))
    np.testing.assert_array_equal(data["alias_loc_kind"][0], expected_loc)
    np.testing.assert_array_equal(
        data["alias_loc_kind"], np.tile(expected_loc, (4, 1)))
    assert np.all((0 <= data["alias_index"])
                  & (data["alias_index"] < max_alias))
    assert engine.delta_min == -2.0
    assert engine.delta_max == 6.0
    assert engine.epsilon == 0.01

    one_site = qaqmc_cpp.QAQMCEngine(
        1, 1.0, -2.0, 6.0, 1.2, 8, 0.01, 19,
        np.zeros((1, 2), dtype=np.float64),
        neighbor_cutoff=-1, delta_groups=2,
    ).export_cuda_diagonal_data()
    assert one_site["bond_sites"].shape == (0, 2)
    assert one_site["bond_rmax"].shape == (2, 0)


@pytest.mark.parametrize(
    "module_name",
    [
        "src.engines.qaqmc",
        "src.engines.qaqmc_cpu_batch",
        "src.engines.qaqmc_renyi",
        "src.engines.qaqmc_renyi_work",
        "src.engines.qaqmc_string_work",
        "src.mpi.qaqmc_mpi",
        "src.mpi.qaqmc_renyi_work_mpi",
        "src.mpi.qaqmc_string_work_mpi",
    ],
)
def test_existing_python_entry_points_import_without_path_changes(module_name):
    importlib.import_module(module_name)
