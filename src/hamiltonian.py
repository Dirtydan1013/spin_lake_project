"""
Physical parameters, matrices, and interaction calculations for the Rydberg Hamiltonian.
"""
import numpy as np
import concurrent.futures
from numba import njit

try:
    from tqdm import trange
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _compute_shell_cutoff_dist(pos, neighbor_cutoff, tol=1e-6):
    """Compute the distance threshold for the given neighbor shell.

    Groups all unique pairwise distances into shells using relative tolerance,
    then returns the max distance of the `neighbor_cutoff`-th shell (1-indexed).

    Args:
        pos: (N, d) array of atom positions.
        neighbor_cutoff: Shell number (1 = nearest, 2 = next-nearest, ...).
        tol: Relative tolerance for grouping distances into the same shell.

    Returns:
        cutoff_dist: Maximum distance for the requested shell (with tolerance).
    """
    N = len(pos)
    dists = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            dists.append(d)
    dists = np.array(dists)
    dists.sort()

    # Group into shells: consecutive distances within relative tolerance
    shells = [dists[0]]
    for d in dists[1:]:
        if abs(d - shells[-1]) / max(shells[-1], 1e-15) > tol:
            shells.append(d)
    shells = np.array(shells)

    if neighbor_cutoff > len(shells):
        # Requested more shells than exist → keep all
        return shells[-1] * (1.0 + tol)

    return shells[neighbor_cutoff - 1] * (1.0 + tol)


@njit
def _compute_vij_worker_numba(i, N, pos, Omega, Rb):
    """Numba-jitted inner loop for calculating interaction distances for atom i."""
    bonds_j = []
    vij_list = []
    dist_list = []
    
    for j in range(i + 1, N):
        # Calculate Euclidean distance manually for numba compatibility
        dist = 0.0
        for d in range(pos.shape[1]):
            dist += (pos[i, d] - pos[j, d]) ** 2
        dist = np.sqrt(dist)
        vij = Omega * (Rb / dist) ** 6
        bonds_j.append(j)
        vij_list.append(vij)
        dist_list.append(dist)
        
    return bonds_j, vij_list, dist_list

def _vij_worker(args):
    """Wrapper function for ProcessPoolExecutor"""
    i_start, i_end, N, pos, Omega, Rb = args
    results = []
    for i in range(i_start, i_end):
        bonds_j, vij_list, dist_list = _compute_vij_worker_numba(i, N, pos, Omega, Rb)
        results.append((i, bonds_j, vij_list, dist_list))
    return results

def build_rydberg_vij(N: int, Omega: float, Rb: float, pos: np.ndarray = None, 
                      verbose: bool = True, n_jobs: int = 1, backend: str = "process",
                      neighbor_cutoff: int = None):
    """
    Computes the interaction matrix V_ij and flattened bond elements 
    for the Rydberg Hamiltonian.

    Args:
        N: Number of sites.
        Omega: Rabi frequency (sets energy scale).
        Rb: Blockade radius (in units of lattice spacing).
        pos: Optional (N, d) array of atom coordinates. Assumes 1D chain if None.
        verbose: Show progress bar if True.
        n_jobs: Number of parallel workers.
        backend: "thread" or "process" backend for multiprocessing.
        neighbor_cutoff: Keep only bonds within the first N neighbor shells.
            1 = nearest neighbors only, 2 = nearest + next-nearest, etc.
            None = keep all bonds (no cutoff).

    Returns:
        V: (N, N) Interaction matrix V[i, j].
        bonds_i, bonds_j: Flat arrays indexing interacting sites.
        vij_list: Flattened interaction values corresponding to bonds.
        bond_sites: (n_bonds, 2) shaped layout array.
    """
    if pos is None:
        pos = np.arange(N).reshape(-1, 1).astype(np.float64)
    else:
        pos = pos.astype(np.float64)

    # Compute shell cutoff distance if requested
    cutoff_dist = None
    if neighbor_cutoff is not None and neighbor_cutoff > 0:
        cutoff_dist = _compute_shell_cutoff_dist(pos, neighbor_cutoff)
        if verbose:
            print(f"  Neighbor cutoff = {neighbor_cutoff} → max dist = {cutoff_dist:.6f}")

    V = np.zeros((N, N), dtype=np.float64)
    bonds_i = []
    bonds_j_all = []
    vij_all = []

    if n_jobs > 1 and N > 100:  # Only parallelize for sufficiently large N
        tasks = []
        chunk_size = max(1, N // n_jobs)
        for i in range(0, N, chunk_size):
            tasks.append((i, min(i + chunk_size, N), N, pos, Omega, Rb))
            
        executor_cls = concurrent.futures.ProcessPoolExecutor if backend == "process" else concurrent.futures.ThreadPoolExecutor
        
        with executor_cls(max_workers=n_jobs) as executor:
            if HAS_TQDM and verbose:
                results_iter = []
                for res in executor.map(_vij_worker, tasks):
                    results_iter.append(res)
            else:
                results_iter = list(executor.map(_vij_worker, tasks))

        for chunk_results in results_iter:
            for i, b_j, v_list, d_list in chunk_results:
                for idx, j in enumerate(b_j):
                    if cutoff_dist is not None and d_list[idx] > cutoff_dist:
                        continue
                    vij = v_list[idx]
                    V[i, j] = vij
                    V[j, i] = vij
                    bonds_i.append(i)
                    bonds_j_all.append(j)
                    vij_all.append(vij)

    else:
        # Sequential execution
        loop_iter = trange(N, desc="Calc V_ij") if (HAS_TQDM and verbose) else range(N)
        for i in loop_iter:
            b_j, v_list, d_list = _compute_vij_worker_numba(i, N, pos, Omega, Rb)
            for idx, j in enumerate(b_j):
                if cutoff_dist is not None and d_list[idx] > cutoff_dist:
                    continue
                vij = v_list[idx]
                V[i, j] = vij
                V[j, i] = vij
                bonds_i.append(i)
                bonds_j_all.append(j)
                vij_all.append(vij)

    n_bonds = len(bonds_i)
    if verbose:
        n_total = N * (N - 1) // 2
        print(f"  Bonds: {n_bonds} / {n_total} total"
              f" ({100*n_bonds/max(n_total,1):.1f}% kept)")

    bond_sites = np.zeros((max(n_bonds, 1), 2), dtype=np.int32)
    coord_number = np.zeros(N, dtype=np.int32)
    for b in range(n_bonds):
        si, sj = bonds_i[b], bonds_j_all[b]
        bond_sites[b, 0] = si
        bond_sites[b, 1] = sj
        coord_number[si] += 1
        coord_number[sj] += 1
        
    return V, np.array(bonds_i, dtype=np.int64), np.array(bonds_j_all, dtype=np.int64), np.array(vij_all, dtype=np.float64), bond_sites, coord_number
