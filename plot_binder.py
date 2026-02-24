import numpy as np
import matplotlib.pyplot as plt
from src.qaqmc import QAQMC_Rydberg
from src.ed_core import build_rydberg_hamiltonian, ed_ground_state_observables

def main():
    Omega = 1.0
    Rb = 1.25
    delta_min = -3.0
    delta_max = 4.0
    
    # systems = [4, 8, 12]
    # colors = ['blue', 'red', 'green']
    systems = [4, 8]
    colors = ['blue', 'red']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    c = 0.5  # scaling variable c = 1/2
    
    for i, L in enumerate(systems):
        print(f"--- Running for N={L} ---")
        
        diff = delta_max - delta_min
        # Adiabatic scaling v = c / L^3
        v = c / (L**3)
        M = int(round(diff / v))
        M = max(20, M)
        
        # Run QAQMC ONCE for the whole sweep
        qaqmc = QAQMC_Rydberg(N=L, Omega=Omega, delta_min=delta_min, delta_max=delta_max, Rb=Rb, M=M, seed=42)
        res = qaqmc.run_asymmetric(n_equil=4000, n_measure=16000, n_jobs=4)
        
        deltas_qmc = res['deltas']
        u_qmc = res['binder_mean']
        
        # Filter slightly to plotting window
        plot_mask = (deltas_qmc >= -1.0) & (deltas_qmc <= 4.0)
        ax.plot(deltas_qmc[plot_mask], u_qmc[plot_mask], '-', color=colors[i], label=f'QAQMC N={L}')
        
        # Exact Diagonalization (ED) Ground State on a sparse grid
        deltas_ed = np.linspace(-1.0, 4.0, 30)
        res_ed_u = []
        for delta in deltas_ed:
            H = build_rydberg_hamiltonian(L, Omega, delta, Rb)
            _, _, _, ed_u = ed_ground_state_observables(H, L)
            res_ed_u.append(ed_u)
            
        ax.plot(deltas_ed, res_ed_u, linestyle='--', color=colors[i], alpha=0.7, label=f'ED N={L}')
        
        print(f" Finished N={L}. M={M}. ED matched.")
        
    ax.set_xlabel(r'Detuning $\Delta/\Omega$', fontsize=14)
    ax.set_ylabel(r'Binder Cumulant $U$ (Asymmetric)', fontsize=14)
    ax.set_title(r"Binder Cumulant Asymmetric Evolution vs Detuning", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('binder_cumulant1.png', dpi=300)
    print("Plot saved to binder_cumulant.png")

if __name__ == '__main__':
    main()
