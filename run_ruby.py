import time
import numpy as np

from rydberg_ed import build_rydberg_hamiltonian, ed_thermal_energy, ed_thermal_density
from rydberg_sse import SSE_Rydberg
from lattices import generate_ruby_lattice

def test_ruby_lattice_qmc():
    print("=== Testing 1x1 Ruby Lattice (6 sites) ===")
    
    # Parameters
    nx, ny = 1, 1
    Omega = 1.0
    delta = 0.5
    Rb = 1.5
    beta = 2.0
    
    # Generate coordinates
    pos = generate_ruby_lattice(nx, ny, a=3.0)
    N = pos.shape[0]
    print(f"Number of sites: {N}")
    print("Coordinates:")
    for p in pos:
        print(f"  ({p[0]:.3f}, {p[1]:.3f})")
    
    # Exact Diagonalization
    print("\n--- Exact Diagonalization ---")
    t0 = time.time()
    H = build_rydberg_hamiltonian(N, Omega=Omega, delta=delta, Rb=Rb, pos=pos)
    ed_E = ed_thermal_energy(H, beta)
    ed_n = ed_thermal_density(H, N, beta)
    print(f"ED Energy:  {ed_E:.6f}")
    print(f"ED Density: {ed_n:.6f}")
    print(f"ED Time:    {time.time() - t0:.3f} s")
    
    # SSE QMC
    print("\n--- SSE QMC ---")
    n_equil = 50000
    n_measure = 50000
    print(f"Equilibration steps: {n_equil}")
    print(f"Measurement steps:   {n_measure}")
    
    t0 = time.time()
    sse = SSE_Rydberg(N, Omega=Omega, delta=delta, Rb=Rb, beta=beta, pos=pos)
    res = sse.run(n_equil=n_equil, n_measure=n_measure)
    print(f"SSE Time:   {time.time() - t0:.3f} s")
    
    print("\n--- Results ---")
    print(f"Energy:   ED = {ed_E:8.5f}, SSE = {res['energy_mean']:8.5f} ± {res['energy_err']:.5f}  | Diff = {abs(ed_E - res['energy_mean']):.5f}")
    print(f"Density:  ED = {ed_n:8.5f}, SSE = {res['density_mean']:8.5f} ± {res['density_err']:.5f}  | Diff = {abs(ed_n - res['density_mean']):.5f}")

if __name__ == "__main__":
    test_ruby_lattice_qmc()
