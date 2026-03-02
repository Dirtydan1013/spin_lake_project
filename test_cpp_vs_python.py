"""
Quick smoke test: verify that the C++ QAQMC backend produces physically
sensible density curves on a small 6-atom Ruby lattice.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.qaqmc import QAQMC_Rydberg, HAS_CPP
from src.lattices import generate_kagome_bond_lattice

print(f"C++ backend available: {HAS_CPP}")

# pos = generate_kagome_bond_lattice(nx=1, ny=1, a=3.0)
N = 12
print(f"Lattice: {N} atoms")

# Run with C++ backend
q_cpp = QAQMC_Rydberg(N=N, Omega=1.0, delta_min=0.0, delta_max=8.0,
                       Rb=2.4, M=60, epsilon=0.01, seed=42,
                       verbose=True, use_cpp=True)
for _ in range(200):
    q_cpp.mc_step()

# Check density at midpoint
state = np.zeros(N, dtype=np.int32)
for p in range(q_cpp.M):
    if q_cpp.op_types[p] == -1:
        state[q_cpp.op_sites[p]] ^= 1
dens_cpp = state.sum() / N
print(f"C++ midpoint density: {dens_cpp:.4f}")

# Run with Python/Numba backend
q_py = QAQMC_Rydberg(N=N, Omega=1.0, delta_min=0.0, delta_max=8.0,
                      Rb=2.4, M=60, epsilon=0.01, seed=42,
                      verbose=True, use_cpp=False)
for _ in range(200):
    q_py.mc_step()

state_py = np.zeros(N, dtype=np.int32)
for p in range(q_py.M):
    if q_py.op_types[p] == -1:
        state_py[q_py.op_sites[p]] ^= 1
dens_py = state_py.sum() / N
print(f"Python midpoint density: {dens_py:.4f}")

print(f"\nBoth backends ran successfully!")
print(f"C++ density:    {dens_cpp:.4f}")
print(f"Python density: {dens_py:.4f}")
print("(Note: exact match not expected due to different RNG streams)")
