import numpy as np
import matplotlib.pyplot as plt
from src.qaqmc import QAQMC_Rydberg
from src.ed_core import build_rydberg_hamiltonian, ed_ground_state_density

def main():
    N = 8
    Omega = 1.0
    Rb = 1.25
    delta_min = -5.0
    delta_max = 1.0
    
    # 1. Exact Diagonalization Ground State
    print("Computing ED Ground State...")
    H = build_rydberg_hamiltonian(N, Omega, delta_max, Rb)
    ed_gs_density = ed_ground_state_density(H, N)
    print(f"ED Ground State Density: {ed_gs_density:.6f}")
    
    # 2. QAQMC Convergence
    M_values = [10, 20, 40, 80, 160, 320, 640,1280,2560,5120,10240]
    qaqmc_densities = []
    qaqmc_errors = []
    
    for M in M_values:
        print(f"Running QAQMC for M = {M}...")
        qaqmc = QAQMC_Rydberg(N=N, Omega=Omega, delta_min=delta_min, delta_max=delta_max, Rb=Rb, M=M, seed=42)
        res = qaqmc.run(n_equil=20000, n_measure=20000, n_jobs=4)
        qaqmc_densities.append(res['density_mean'])
        qaqmc_errors.append(res['density_err'])
        print(f"  Result: {res['density_mean']:.6f} +/- {res['density_err']:.6f}")
        
    # 3. Plotting
    plt.figure(figsize=(10, 6))
    plt.errorbar(M_values, qaqmc_densities, yerr=qaqmc_errors, fmt='o-', capsize=5, label='QAQMC Density', color='blue', markeredgecolor='black')
    plt.axhline(y=ed_gs_density, color='red', linestyle='--', label=f'ED GS Density ({ed_gs_density:.4f})')
    
    plt.xscale('log')
    plt.xticks(M_values, [str(m) for m in M_values])
    plt.xlabel('Projection Length $M$', fontsize=12)
    plt.ylabel(r'Excitation Density $\langle n \rangle$', fontsize=12)
    plt.title('QAQMC Convergence to Ground State Expectation Value', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('qaqmc_convergence.png', dpi=300)
    print("Plot saved to qaqmc_convergence.png")

if __name__ == '__main__':
    main()
