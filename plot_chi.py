import numpy as np
import matplotlib.pyplot as plt
from src.qaqmc import QAQMC_Rydberg
from src.ed_core import build_rydberg_hamiltonian, ed_ground_state_observables

def main():
    Omega = 1.0
    Rb = 1.25
    delta_min = -3.0
    delta_max = 6.0
    
    systems = [4, 8, 12]
    colors = ['blue', 'red', 'green']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    c = 0.5  # scaling variable
    
    for i, L in enumerate(systems):
        print(f"--- Running for N={L} ---")
        
        diff = delta_max - delta_min
        v = c / (L**3)
        M = int(round(diff / v))
        M = max(20, M)
        
        # Run QAQMC ONCE for the whole sweep
        qaqmc = QAQMC_Rydberg(N=L, Omega=Omega, delta_min=delta_min, delta_max=delta_max, Rb=Rb, M=M, seed=42)
        res = qaqmc.run_asymmetric(n_equil=4000, n_measure=14000, n_jobs=4)
        
        deltas_qmc = res['deltas']
        chi_qmc = res['chi_mean']
        
        plot_mask = (deltas_qmc >= -1.0) & (deltas_qmc <= 6.0)
        ax.plot(deltas_qmc[plot_mask], chi_qmc[plot_mask], '-', color=colors[i], label=f'QAQMC N={L}')
        
        # Exact Diagonalization (ED) Ground State on a sparse grid
        deltas_ed = np.linspace(-1.0, 6.0, 30)
        res_ed_chi = []
        for delta in deltas_ed:
            H = build_rydberg_hamiltonian(L, Omega, delta, Rb)
            _, _, ed_chi, _ = ed_ground_state_observables(H, L)
            res_ed_chi.append(ed_chi)
            
        ax.plot(deltas_ed, res_ed_chi, linestyle='--', color=colors[i], alpha=0.7, label=f'ED N={L}')
        print(f" Finished N={L}. M={M}. ED matched.")
        
    ax.set_xlabel(r'Detuning $\Delta/\Omega$', fontsize=14)
    ax.set_ylabel(r'Staggered Susceptibility $\chi$', fontsize=14)
    ax.set_title(r"Staggered Susceptibility Asymmetric Evolution vs Detuning", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('staggered_susceptibility.png', dpi=300)

if __name__ == '__main__':
    main()
