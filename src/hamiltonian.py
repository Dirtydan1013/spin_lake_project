"""
Physical parameters, matrices, and interaction calculations for the Rydberg Hamiltonian.
"""
import numpy as np

def build_rydberg_vij(N: int, Omega: float, Rb: float, pos: np.ndarray = None):
    """
    Computes the interaction matrix V_ij and flattened bond elements 
    for the Rydberg Hamiltonian.

    Args:
        N: Number of sites.
        Omega: Rabi frequency (sets energy scale).
        Rb: Blockade radius (in units of lattice spacing).
        pos: Optional (N, d) array of atom coordinates. Assumes 1D chain if None.

    Returns:
        V: (N, N) Interaction matrix V[i, j].
        bonds_i, bonds_j: Flat arrays indexing interacting sites.
        vij_list: Flattened interaction values corresponding to bonds.
        bond_sites: (n_bonds, 2) shaped layout array.
    """
    if pos is None:
        pos = np.arange(N).reshape(-1, 1)

    V = np.zeros((N, N))
    bonds_i, bonds_j, vij_list = [], [], []
    
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(pos[i] - pos[j])
            vij = Omega * (Rb / dist) ** 6
            V[i, j] = vij
            V[j, i] = vij  # Symmetric element
            
            bonds_i.append(i)
            bonds_j.append(j)
            vij_list.append(vij)
            
    n_bonds = len(bonds_i)
    bond_sites = np.zeros((max(n_bonds, 1), 2), dtype=np.int32)
    for b in range(n_bonds):
        bond_sites[b, 0] = bonds_i[b]
        bond_sites[b, 1] = bonds_j[b]
        
    return V, np.array(bonds_i), np.array(bonds_j), np.array(vij_list), bond_sites
