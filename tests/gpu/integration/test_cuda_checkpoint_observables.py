"""High-level CUDA engine replay and diagonal-observable integration tests."""

from __future__ import annotations

import numpy as np
import pytest

qaqmc_cpp = pytest.importorskip("qaqmc_cpp")
qaqmc_cuda = pytest.importorskip("qaqmc_cuda")
if not qaqmc_cuda.is_available():
    pytest.skip("CUDA device is not available", allow_module_level=True)

from src.engines.qaqmc_cuda import CudaDiagonalBackend


def _backend(seed: int = 777) -> CudaDiagonalBackend:
    n_sites = 8
    pos = np.arange(n_sites, dtype=np.float64).reshape(-1, 1)
    cpu = qaqmc_cpp.QAQMCEngine(
        n_sites, 1.0, -0.5, 1.5, 1.2, 1_500, 0.01, seed, pos,
        neighbor_cutoff=2, delta_groups=40,
    )
    return CudaDiagonalBackend.from_cpu_engine(cpu, seed=seed)


def _serial_states(types: np.ndarray, sites: np.ndarray, n_sites: int, step: int):
    state = np.zeros(n_sites, dtype=np.uint8)
    output = []
    for p, (kind, site) in enumerate(zip(types, sites)):
        if kind == -1:
            state[site] ^= 1
        if (p + 1) % step == 0:
            output.append(state.copy())
    return np.asarray(output)


def test_checkpoint_restarts_exact_philox_trajectory(tmp_path):
    original = _backend()
    original.run_steps(4)
    checkpoint = tmp_path / "cuda_chain.npz"
    original.save_checkpoint(checkpoint)

    restored = _backend(seed=12)
    restored.load_checkpoint(checkpoint)
    assert restored.seed == original.seed
    assert restored.sweep_id == original.sweep_id

    for _ in range(6):
        original.mc_step()
        restored.mc_step()
        np.testing.assert_array_equal(
            original.get_operator_string()[0], restored.get_operator_string()[0]
        )
        np.testing.assert_array_equal(
            original.get_operator_string()[1], restored.get_operator_string()[1]
        )


def test_profile_and_midpoint_observables_match_downloaded_operator_string():
    backend = _backend()
    backend.run_steps(3)
    backend.set_bulk_sites([1, 2, 3, 4, 5, 6])
    loops = [[0, 1, 2, 3], [2, 3, 4, 5], [0, 2, 4, 6, 7, 5]]
    strings = [[0, 2, 4], [1, 3, 5], [0, 1, 2, 3, 4]]
    backend.set_observable_sites(loops, strings)
    backend.set_vbs_triangles(
        np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
        [0, 1], [1, -1], [1, 1], 0, 1,
    )

    profile_step = 137
    got_states = backend.profile_states(profile_step)
    types, sites = backend.get_operator_string()
    expected_states = _serial_states(types, sites, backend.N, profile_step)
    np.testing.assert_array_equal(got_states, expected_states)

    got = backend.measure_profile(profile_step)
    np.testing.assert_allclose(got["density"], expected_states[:, 1:7].mean(axis=1))
    midpoint = backend.measure_at_midpoint()
    midpoint_state = _serial_states(types, sites, backend.N, backend.M)[0]
    assert midpoint["density"] == pytest.approx(midpoint_state[1:7].mean())
    assert got["Z_l"].shape == (len(expected_states), 2)
    assert got["C_m_l"].shape == (len(expected_states), 2)
    assert got["M_vbs"].shape == (len(expected_states),)
    assert got["M_ss"].shape == (len(expected_states),)

    cell_r = np.column_stack((np.arange(backend.N), np.zeros(backend.N)))
    basis = np.arange(backend.N) % 2
    q_points = np.array([[0.0, 0.0], [np.pi / 4, 0.0]])
    sf = backend.occupation_sf_matrices(
        expected_states[:3], cell_r, basis, q_points,
        site_in_bulk=np.arange(backend.N) >= 2, n_basis=2,
    )
    np.testing.assert_allclose(
        sf["S_full"],
        sf["s_full"][..., :, None] * np.conj(sf["s_full"][..., None, :]),
    )
