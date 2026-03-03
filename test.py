"""
Verify QAQMC sampler by comparing post-processed Rydberg density
against exact QAQMC propagation on a 1×1 Ruby (Kagome-bond) lattice.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.qaqmc import QAQMC_Rydberg
from src.postprocess import QAQMCArchive, obs_density_asym
from src.ed_core import qaqmc_exact_asymmetric_observables
from src.lattices import generate_kagome_bond_lattice

# ── Shared parameters ─────────────────────────────────────────────────────────
N_CELLS = 2          # 1×1 unit cell → 6 atoms
A       = 4        # lattice constant
OMEGA   = 1.0
DELTA_MIN = 0.0
DELTA_MAX = 8.0
Rb      = 2.4
M       = 20        # imaginary-time slices (half sweep)
EPSILON = 0.01
SEED    = 7
N_EQUIL   = 4_000
N_SAMPLES = 30_000

H5_PATH = 'data/test_ruby.h5'

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    # ── 1. Geometry ───────────────────────────────────────────────────────────
    pos = generate_kagome_bond_lattice(nx=N_CELLS, ny=N_CELLS, a=A)
    N   = len(pos)
    print(f"Lattice: {N} atoms")

    # ── 2. Run QAQMC and save raw operator sequences ──────────────────────────
    print("\n=== QAQMC run_and_save ===")
    q = QAQMC_Rydberg(N=N, Omega=OMEGA, delta_min=DELTA_MIN, delta_max=DELTA_MAX,
                      Rb=Rb, M=M, epsilon=EPSILON, seed=SEED, omp_threads=1,pos=pos
                      )
    q.run_and_save(H5_PATH, n_equil=N_EQUIL, n_samples=N_SAMPLES,
                   n_jobs=4, backend="process")

    # ── 3. Post-process: ⟨n(δ)⟩ via asymmetric estimator ─────────────────────
    print("\n=== Post-processing: obs_density_asym ===")
    arc = QAQMCArchive(H5_PATH)
    print(arc)

    res_asym = arc.compute(obs_density_asym, n_jobs=4)
    deltas_qmc = arc.deltas           # (M,) forward-sweep δ values
    density_qmc = res_asym['mean']     # (M,) mean ⟨n⟩
    density_err = res_asym['err']      # (M,) binned standard error

    print(f"  δ range: [{deltas_qmc[0]:.2f}, {deltas_qmc[-1]:.2f}]")
    print(f"  ⟨n⟩ range: [{density_qmc.min():.4f}, {density_qmc.max():.4f}]")

    # ── 4. Exact QAQMC propagation ────────────────────────────────────────────
    print("\n=== Exact QAQMC propagation ===")
    exact = qaqmc_exact_asymmetric_observables(
        N=N, Omega=OMEGA, delta_min=DELTA_MIN, delta_max=DELTA_MAX,
        Rb=Rb, M=M, pos=pos, epsilon=EPSILON,
        normalize_each_step=True
    )

    deltas_ex  = exact['deltas']          # (M,)
    density_ex = exact['density_mean']    # (M,)

    print(f"  δ range: [{deltas_ex[0]:.2f}, {deltas_ex[-1]:.2f}]")
    print(f"  ⟨n⟩ range: [{density_ex.min():.4f}, {density_ex.max():.4f}]")

    # ── 5. Agreement check ────────────────────────────────────────────────────
    # Compare at every 10th slice to print a table
    print("\n  δ       ⟨n⟩_exact   ⟨n⟩_QMC   diff")
    print("  " + "-"*45)
    step = max(1, M // 16)
    for i in range(0, M, step):
        diff = density_qmc[i] - density_ex[i]
        print(f"  {deltas_ex[i]:5.2f}   {density_ex[i]:.5f}    "
              f"{density_qmc[i]:.5f}   {diff:+.5f}  "
              f"({'✓' if abs(diff) < 5*density_err[i]+1e-4 else '✗'})")

    # Max absolute deviation
    max_dev = np.max(np.abs(density_qmc - density_ex))
    print(f"\n  Max |ΔQMC - exact|: {max_dev:.5f}")

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(deltas_qmc, density_qmc - density_err, density_qmc + density_err,
                    alpha=0.25, color='steelblue', label='QMC ±1σ')
    ax.plot(deltas_qmc, density_qmc, 'o-', ms=3, lw=1.5,
            color='steelblue', label=f'QAQMC (N={N_SAMPLES:,} samples)')
    ax.plot(deltas_ex, density_ex, '--', lw=2,
            color='tomato', label='Exact QAQMC propagation')

    ax.set_xlabel(r'Detuning $\delta/\Omega$', fontsize=13)
    ax.set_ylabel(r'Rydberg density $\langle n \rangle$', fontsize=13)
    ax.set_title(f'QAQMC vs Exact  —  Ruby lattice ({N}atoms)\n'
                 r'$\Omega=1,\ R_b/a=2.4,\ M=160$', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = 'data/qaqmc_density_vs_exact.png'
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved → {plot_path}")
    plt.close(fig)
