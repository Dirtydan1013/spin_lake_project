"""
Exact Diagonalization (ED) for Rydberg atom arrays.
Used for benchmarking QMC on small system sizes.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from src.hamiltonian import build_rydberg_vij

def build_rydberg_hamiltonian(N: int, Omega: float, delta: float, Rb: float, pos: np.ndarray = None) -> np.ndarray:
    """Build the full 2^N × 2^N Hamiltonian matrix."""
    dim = 1 << N
    H = np.zeros((dim, dim))

    V, _, _, _, _ = build_rydberg_vij(N, Omega, Rb, pos)

    for s in range(dim):
        diag = 0.0
        for i in range(N):
            ni = (s >> i) & 1
            if ni:
                diag -= delta
            for j in range(i + 1, N):
                nj = (s >> j) & 1
                if ni and nj:
                    diag += V[i, j]
        H[s, s] = diag
        for i in range(N):
            t = s ^ (1 << i)
            H[s, t] -= Omega / 2.0

    return H

def ed_thermal_energy(H: np.ndarray, beta: float) -> float:
    eigvals, _ = eigh(H)
    E0 = eigvals[0]
    boltz = np.exp(-beta * (eigvals - E0))
    Z = np.sum(boltz)
    return float(np.sum(eigvals * boltz) / Z)

def ed_thermal_density(H: np.ndarray, N: int, beta: float) -> float:
    dim = 1 << N
    eigvals, eigvecs = eigh(H)
    E0 = eigvals[0]
    boltz = np.exp(-beta * (eigvals - E0))
    Z = np.sum(boltz)
    rho_diag = boltz / Z

    density = 0.0
    for i in range(N):
        n_i = np.array([(s >> i) & 1 for s in range(dim)], dtype=float)
        for k in range(dim):
            density += rho_diag[k] * np.dot(eigvecs[:, k] * n_i, eigvecs[:, k])
    return float(density / N)

def ed_ground_state_energy(H: np.ndarray) -> float:
    eigvals, _ = eigh(H)
    return float(eigvals[0])

def ed_ground_state_density(H: np.ndarray, N: int) -> float:
    dim = 1 << N
    eigvals, eigvecs = eigh(H)
    v0 = eigvecs[:, 0]

    density = 0.0
    for i in range(N):
        n_i = np.array([(s >> i) & 1 for s in range(dim)], dtype=float)
        density += np.dot(v0 * n_i, v0)
    return float(density / N)

def ed_ground_state_observables(H: np.ndarray, N: int):
    dim = 1 << N
    if dim <= 1024:
        eigvals, eigvecs = eigh(H)
        v0 = eigvecs[:, 0]
    else:
        v0 = eigsh(H, k=1, which='SA')[1][:, 0]

    density = 0.0
    staggered_abs = 0.0
    staggered_sq = 0.0
    staggered_qd = 0.0
    
    for s in range(dim):
        prob = v0[s]**2
        if prob < 1e-12: continue
        dens = 0.0
        stag = 0.0
        for i in range(N):
            n_i = (s >> i) & 1
            dens += n_i
            stag += n_i * (1 if i % 2 == 0 else -1)
        dens /= N
        stag /= N
        
        density += prob * dens
        staggered_abs += prob * abs(stag)
        staggered_sq += prob * (stag**2)
        staggered_qd += prob * (stag**4)
        
    chi = N * (staggered_sq - staggered_abs**2)
    binder = 1.5 * (1.0 - (staggered_qd / (3.0 * staggered_sq**2 + 1e-12))) if staggered_sq > 0 else 0.0
        
    return density, staggered_sq, chi, binder
