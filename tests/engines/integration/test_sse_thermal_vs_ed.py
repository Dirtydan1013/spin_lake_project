"""SSE finite-temperature engine vs exact thermal ED.

Checks the C++ SSE engine's thermal averages against Tr(O e^{-βH})/Tr(e^{-βH})
on a small 1D-chain Rydberg system, and separately verifies the warm-start
round-trip (set_config restores a chain that keeps sampling the same ensemble).

Heavy (long MC chains); run as a script:
    python tests/engines/integration/test_sse_thermal_vs_ed.py
"""

import sys
from pathlib import Path

import numpy as np

# tests/engines/integration/<file> → parents[3] == repo root (so `import
# qaqmc_cpp` + `src.` work when run standalone, not only under conftest).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import qaqmc_cpp
from src.analysis.ed_core import build_rydberg_hamiltonian
from src.rydberg.lattices import generate_1d_chain


N = 6
OMEGA, DELTA, RB, BETA, EPSILON = 1.0, 0.6, 1.4, 2.0, 0.01
POS = np.ascontiguousarray(generate_1d_chain(N, 1.0), dtype=np.float64)


def _ed_thermal(delta=DELTA, beta=BETA):
    """Exact thermal <energy> and <density> for the Rydberg chain."""
    H = build_rydberg_hamiltonian(N, OMEGA, delta, RB, pos=POS, neighbor_cutoff=None)
    evals, evecs = np.linalg.eigh(H)
    w = np.exp(-beta * (evals - evals.min()))
    w /= w.sum()
    dim = 1 << N
    n_mean_diag = np.array([bin(s).count("1") / N for s in range(dim)])
    dens_per_eig = (evecs**2 * n_mean_diag[:, None]).sum(axis=0)
    return float((w * evals).sum()), float((w * dens_per_eig).sum())


def _make_engine(seed):
    return qaqmc_cpp.SSEEngine(N=N, Omega=OMEGA, delta=DELTA, Rb=RB, beta=BETA,
                               epsilon=EPSILON, seed=seed, pos=POS, neighbor_cutoff=-1)


def _blocked_mean_err(x, n_blocks=200):
    x = np.asarray(x, dtype=np.float64)
    m = (len(x) // n_blocks) * n_blocks
    b = x[:m].reshape(n_blocks, -1).mean(axis=1)
    return float(b.mean()), float(b.std(ddof=1) / np.sqrt(n_blocks))


def _test_thermal_matches_ed_within_sampling_error():
    ed_energy, ed_density = _ed_thermal()
    eng = _make_engine(seed=12345)
    res = eng.run(n_equil=20000, n_samples=400000)

    d_mean, d_err = _blocked_mean_err(res["densities"])
    e_mean, e_err = _blocked_mean_err(res["energies"])
    d_sigma = (d_mean - ed_density) / max(d_err, 1e-12)
    e_sigma = (e_mean - ed_energy) / max(e_err, 1e-12)
    print(f"  density: SSE={d_mean:.5f}±{d_err:.5f} ED={ed_density:.5f} ({d_sigma:+.2f}σ)")
    print(f"  energy : SSE={e_mean:.5f}±{e_err:.5f} ED={ed_energy:.5f} ({e_sigma:+.2f}σ)")
    assert abs(d_sigma) < 4.0, f"density off by {d_sigma:.2f}σ"
    assert abs(e_sigma) < 4.0, f"energy off by {e_sigma:.2f}σ"


def _test_warm_start_roundtrip_preserves_ensemble():
    """set_config from a thermalized chain must resume the SAME ensemble."""
    _, ed_density = _ed_thermal()

    src = _make_engine(seed=2024)
    src.run(n_equil=20000, n_samples=0)  # thermalize
    state = np.asarray(src.state, dtype=np.int32)
    op_types = np.asarray(src.op_types, dtype=np.int32)
    op_sites = np.asarray(src.op_sites, dtype=np.int32)
    rng_state = src.get_rng_state()

    # Fresh engine, install the saved config + RNG, then sample with NO equil.
    dst = _make_engine(seed=999)  # different seed, overwritten by set_rng_state
    dst.set_config(state, op_types, op_sites)
    dst.set_rng_state(rng_state)
    res = dst.run(n_equil=0, n_samples=200000)

    d_mean, d_err = _blocked_mean_err(res["densities"])
    d_sigma = (d_mean - ed_density) / max(d_err, 1e-12)
    print(f"  warm-start density: {d_mean:.5f}±{d_err:.5f} ED={ed_density:.5f} ({d_sigma:+.2f}σ)")
    assert abs(d_sigma) < 4.0, f"warm-started chain off by {d_sigma:.2f}σ"


def main():
    print("SSE thermal <energy>/<density> vs exact ED (N=6 chain, β=2):")
    _test_thermal_matches_ed_within_sampling_error()
    print("SSE warm-start (set_config) round-trip vs ED:")
    _test_warm_start_roundtrip_preserves_ensemble()
    print("SSE thermal vs ED integration checks passed")


if __name__ == "__main__":
    main()
