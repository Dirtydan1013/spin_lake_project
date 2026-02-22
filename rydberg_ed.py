"""
Exact Diagonalization (ED) for Rydberg atom arrays.
Used for benchmarking SSE QMC on small system sizes.

Hamiltonian:
  H = -(Ω/2) Σ_i σ^x_i  -  δ Σ_i n_i  +  Σ_{i<j} V_ij n_i n_j
  V_ij = Ω (Rb / |i-j|)^6
"""

import numpy as np
from scipy.linalg import eigh


def build_rydberg_hamiltonian(N: int, Omega: float, delta: float, Rb: float, pos: np.ndarray = None) -> np.ndarray:
    """Build the full 2^N × 2^N Hamiltonian matrix.

    Args:
        N:     Number of sites.
        Omega: Rabi frequency (sets energy scale).
        delta: Detuning.
        Rb:    Blockade radius (in units of lattice spacing).
        pos:   Optional (N, d) array of atom coordinates. If None, assumes 1D chain with lattice spacing 1.

    Returns:
        H: (2^N, 2^N) Hamiltonian matrix.
    """
    dim = 1 << N
    H = np.zeros((dim, dim))

    # Precompute interaction strengths
    V = np.zeros((N, N))
    if pos is None:
        pos = np.arange(N).reshape(-1, 1)
        
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(pos[i] - pos[j])
            V[i, j] = Omega * (Rb / dist) ** 6

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
        # Off-diagonal: flip spin i
        for i in range(N):
            t = s ^ (1 << i)
            H[s, t] -= Omega / 2.0

    return H


def ed_thermal_energy(H: np.ndarray, beta: float) -> float:
    """Compute finite-temperature energy ⟨H⟩ via exact diagonalization.

    Args:
        H:    Hamiltonian matrix.
        beta: Inverse temperature β = 1/T.

    Returns:
        Thermal expectation value ⟨H⟩.
    """
    eigvals, _ = eigh(H)
    E0 = eigvals[0]
    boltz = np.exp(-beta * (eigvals - E0))
    Z = np.sum(boltz)
    return np.sum(eigvals * boltz) / Z


def ed_thermal_density(H: np.ndarray, N: int, beta: float) -> float:
    """Compute average Rydberg excitation density ⟨n⟩ = (1/N) Σ_i ⟨n_i⟩.

    Args:
        H:    Hamiltonian matrix.
        N:    Number of sites.
        beta: Inverse temperature.

    Returns:
        Average site occupation density.
    """
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
    return density / N
