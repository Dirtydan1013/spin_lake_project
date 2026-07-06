import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO_ROOT)

from src.analysis.ed_core import qaqmc_exact_renyi2_at_cut
from src.engines.qaqmc_renyi_work import QAQMCRenyiWorkRydberg
from src.rydberg.lattices import generate_1d_chain


def _jarzynski_s2_and_bootstrap(work_samples: np.ndarray, n_boot: int = 200) -> tuple[float, float]:
    x = -np.asarray(work_samples, dtype=np.float64)
    x_max = float(x.max())
    s2 = -(x_max + np.log(np.exp(x - x_max).mean()))

    rng = np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        xb = x[idx]
        xb_max = float(xb.max())
        boot.append(-(xb_max + np.log(np.exp(xb - xb_max).mean())))
    return s2, float(np.std(boot, ddof=1))


def test_n6_forward_ramp_delta4_half_chain_matches_ed():
    """N=6 one-dimensional chain, forward-ramp cut at delta=4 on a 0->8 ramp."""
    N = 6
    M = 100
    Omega = 1.0
    Rb = 1.2
    delta_min = 0.0
    delta_max = 8.0
    epsilon = 0.01
    m_star = M // 2

    pos = generate_1d_chain(N, 1.0)
    A = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)

    ed = qaqmc_exact_renyi2_at_cut(
        N=N,
        Omega=Omega,
        delta_min=delta_min,
        delta_max=delta_max,
        Rb=Rb,
        M=M,
        A_mask=A,
        m_star=m_star,
        pos=pos,
        epsilon=epsilon,
        neighbor_cutoff=1,
    )
    assert ed["delta_at_cut"] == 4.0

    eng = QAQMCRenyiWorkRydberg(
        N=N,
        M=M,
        Omega=Omega,
        Rb=Rb,
        delta_min=delta_min,
        delta_max=delta_max,
        epsilon=epsilon,
        seed=1034,
        pos=pos,
        neighbor_cutoff=1,
        delta_groups=200,
    )
    eng.set_cut(m_star)
    eng.set_region(A)
    eng.set_lambda_schedule(np.linspace(0.0, 1.0, 101))
    eng.thermalize(2000)
    result = eng.run_trajectories(n_trajectories=1000, decorrelation_steps=200)

    s2, boot_sd = _jarzynski_s2_and_bootstrap(result.work_samples)
    diff = s2 - ed["S2"]
    assert abs(diff) < max(4.0 * boot_sd, 0.12), (
        f"asymmetric cut mismatch: QMC={s2:+.5f}, ED={ed['S2']:+.5f}, "
        f"diff={diff:+.5f}, bootstrap sd={boot_sd:.5f}"
    )
