"""Small statistical end-to-end smoke gates for both CUDA work protocols."""

from __future__ import annotations

import numpy as np
import pytest

qaqmc_cuda = pytest.importorskip("qaqmc_cuda")
pytest.importorskip("qaqmc_cpp")
if not qaqmc_cuda.is_available():
    pytest.skip("CUDA device is not available", allow_module_level=True)

from src.analysis.ed_core import build_qaqmc_midpoint_state, qaqmc_exact_string_zratio
from src.engines.qaqmc_renyi_work_cuda import QAQMCRenyiWorkRydbergCUDA
from src.engines.qaqmc_string_work import cosine_schedule
from src.engines.qaqmc_string_work_cuda import QAQMCStringWorkRydbergCUDA
from src.rydberg.lattices import generate_1d_chain


def _ed_s2(psi: np.ndarray, mask: np.ndarray) -> float:
    n_sites = int(np.log2(psi.size))
    region = np.flatnonzero(mask).tolist()
    complement = np.flatnonzero(1 - mask).tolist()
    matrix = np.zeros((1 << len(region), 1 << len(complement)), dtype=np.float64)
    for state, amplitude in enumerate(psi):
        a = sum(((state >> site) & 1) << j for j, site in enumerate(region))
        b = sum(((state >> site) & 1) << j
                for j, site in enumerate(complement))
        matrix[a, b] = amplitude
    rho = matrix @ matrix.T
    return -float(np.log(max(float(np.trace(rho @ rho)), 1e-300)))


def test_cuda_one_site_string_work_smoke_matches_ed():
    n_sites, half_length = 3, 8
    pos = generate_1d_chain(n_sites, 1.0)
    params = dict(
        N=n_sites, M=half_length, Omega=1.0, Rb=0.7,
        delta_min=0.0, delta_max=0.8, epsilon=0.08,
        pos=pos, neighbor_cutoff=1, delta_groups=8,
    )
    engine = QAQMCStringWorkRydbergCUDA(
        **params, seed=1201, verbose=False
    )
    engine.set_string_sites([1])
    engine.set_lambda_schedule(cosine_schedule(20))
    engine.thermalize(100)
    result = engine.run_trajectories(
        300, decorrelation_steps=20,
        n_topology_sweeps_per_lambda=1,
        n_qaqmc_sweeps_per_lambda=1,
    )
    exact = qaqmc_exact_string_zratio(
        N=n_sites, Omega=1.0, delta_min=0.0, delta_max=0.8,
        Rb=0.7, M=half_length, B_sets=[[1]], m_star=half_length,
        pos=pos, epsilon=0.08, neighbor_cutoff=1,
    )["O_B"][frozenset([1])]
    relative_error = abs(result.o_c - exact) / max(abs(exact), 1e-12)
    assert relative_error < 0.35, (
        result.o_c, exact, relative_error, result.n_eff,
        result.zero_weight_fraction,
    )


def test_cuda_renyi_work_smoke_matches_midpoint_ed():
    n_sites, half_length = 3, 20
    pos = generate_1d_chain(n_sites, 1.0)
    mask = np.array([1, 0, 0], dtype=np.uint8)
    psi = build_qaqmc_midpoint_state(
        N=n_sites, Omega=1.0, delta_min=-0.5, delta_max=1.2,
        Rb=0.8, M=half_length, pos=pos, epsilon=0.08,
        neighbor_cutoff=1,
    )
    exact = _ed_s2(psi, mask)
    engine = QAQMCRenyiWorkRydbergCUDA(
        N=n_sites, M=half_length, Omega=1.0, Rb=0.8,
        delta_min=-0.5, delta_max=1.2, epsilon=0.08,
        seed=1301, pos=pos, neighbor_cutoff=1, delta_groups=20,
        verbose=False,
    )
    engine.set_region(mask)
    engine.set_lambda_schedule(np.linspace(0.0, 1.0, 21))
    engine.thermalize(100)
    result = engine.run_trajectories(300, decorrelation_steps=20)
    assert abs(result.delta_s2 - exact) < 0.35, (
        result.delta_s2, exact, result.work_var,
        result.total_unjoined_at_end,
    )
