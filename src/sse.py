"""
Stochastic Series Expansion (SSE) QMC Python wrapper.
"""
import numpy as np
from numba import int32

from src.hamiltonian import build_rydberg_vij
from src.qmc_updates import build_alias_table, sse_diagonal_update, sse_cluster_update
from src.measurement import calc_density, calc_staggered_magnetization

class SSE_Rydberg:
    def __init__(self, N: int, Omega: float, delta: float, Rb: float,
                 beta: float, epsilon: float = 0.01, seed: int = 42,
                 pos: np.ndarray = None):
        self.N = N
        self.Omega = Omega
        self.delta = delta
        self.Rb = Rb
        self.beta = beta
        self.epsilon = epsilon
        self.pos = pos

        np.random.seed(seed)

        V, bonds_i, bonds_j, vij_list, self.bond_sites = build_rydberg_vij(N, Omega, Rb, pos)
        
        n_bonds = len(bonds_i)
        
        max_alias = N + n_bonds
        self.alias_prob = np.zeros(max_alias, dtype=np.float64)
        self.alias_idx  = np.zeros(max_alias, dtype=np.int64)
        self.op_map_kind = np.zeros(max_alias, dtype=np.int32)
        self.op_map_loc  = np.zeros(max_alias, dtype=np.int32)
        
        weights = []
        op_map_kind = []
        op_map_loc  = []

        site_W = Omega / 2.0
        for i in range(N):
            weights.append(site_W)
            op_map_kind.append(0)
            op_map_loc.append(i)

        self.bond_W = np.zeros((max(n_bonds, 1), 4), dtype=np.float64)
        self.bond_W_max = np.zeros(max(n_bonds, 1), dtype=np.float64)
        
        for b in range(n_bonds):
            vij = vij_list[b]
            delta_b = delta / (N - 1) if N > 1 else delta
            
            m1 = min(0.0, delta_b, 2 * delta_b - vij)
            m2 = min(delta_b, 2 * delta_b - vij)
            cij = abs(m1) + epsilon * abs(m2)
            
            self.bond_W[b, 0] = cij
            self.bond_W[b, 1] = delta_b + cij
            self.bond_W[b, 2] = delta_b + cij
            self.bond_W[b, 3] = -vij + 2 * delta_b + cij
            
            bmax = np.max(self.bond_W[b])
            self.bond_W_max[b] = bmax
            
            weights.append(bmax)
            op_map_kind.append(1)
            op_map_loc.append(b)

        self.n_alias = len(weights)
        if self.n_alias > 0:
            p_arr, i_arr = build_alias_table(weights)
            self.alias_prob[:self.n_alias] = p_arr
            self.alias_idx[:self.n_alias] = i_arr
            self.op_map_kind[:self.n_alias] = op_map_kind
            self.op_map_loc[:self.n_alias] = op_map_loc

        # Total normalization constant 𝒩
        self.norm_N = sum(weights)

        # Initial configuration
        self.state = np.random.randint(0, 2, size=N, dtype=np.int32)
        self.M = 20
        self.op_types = np.zeros(self.M, dtype=np.int32)
        self.op_sites = np.full(self.M, -1, dtype=np.int32)
        self.n_ops = 0

    def mc_step(self):
        self.n_ops = sse_diagonal_update(
            self.op_types, self.op_sites, self.state, self.M, self.n_ops,
            self.beta, self.norm_N,
            self.bond_sites, self.bond_W, self.bond_W_max,
            self.alias_prob, self.alias_idx, self.n_alias,
            self.op_map_kind, self.op_map_loc, self.N
        )
        sse_cluster_update(
            self.op_types, self.op_sites, self.state, self.M, self.N,
            self.bond_sites, self.bond_W
        )
        self.adjust_M()

    def adjust_M(self):
        new_M = int(self.n_ops * 1.33)
        if new_M > self.M:
            old_M = self.M
            self.M = new_M
            new_types = np.zeros(self.M, dtype=np.int32)
            new_sites = np.full(self.M, -1, dtype=np.int32)
            new_types[:old_M] = self.op_types
            new_sites[:old_M] = self.op_sites
            self.op_types = new_types
            self.op_sites = new_sites

    def measure_energy(self) -> float:
        # E = -⟨n⟩/β + shift
        shift = 0.0
        for b in range(len(self.bond_W)):
            shift += (self.bond_W[b, 0])  # cij
        return - self.n_ops / self.beta + shift

    def measure_observables(self) -> dict:
        return {
            'energy': self.measure_energy(),
            'density': calc_density(self.state),
            'm_z': calc_staggered_magnetization(self.state)
        }

    def run(self, n_equil: int = 10000, n_measure: int = 50000):
        for _ in range(n_equil):
            self.mc_step()

        energies = np.empty(n_measure)
        densities = np.empty(n_measure)
        m_z = np.empty(n_measure)
        
        for step in range(n_measure):
            self.mc_step()
            obs = self.measure_observables()
            energies[step] = obs['energy']
            densities[step] = obs['density']
            m_z[step] = obs['m_z']

        n_bins = 50
        bs = n_measure // n_bins
        e_bins = np.array([np.mean(energies[i*bs:(i+1)*bs]) for i in range(n_bins)])
        d_bins = np.array([np.mean(densities[i*bs:(i+1)*bs]) for i in range(n_bins)])
        
        # Susceptibility requires calculating fluctuations directly from samples
        m_z_sq = m_z ** 2
        m_z_abs = np.abs(m_z)
        
        chi_bins = np.array([
            self.N * (np.mean(m_z_sq[i*bs:(i+1)*bs]) - np.mean(m_z_abs[i*bs:(i+1)*bs])**2)
            for i in range(n_bins)
        ])
        
        binder_bins = np.array([
            1.5 * (1.0 - np.mean(m_z_sq[i*bs:(i+1)*bs]**2) / (3.0 * np.mean(m_z_sq[i*bs:(i+1)*bs])**2 + 1e-12))
            for i in range(n_bins)
        ])

        return {
            'energy_mean': float(np.mean(e_bins)),
            'energy_err':  float(np.std(e_bins) / np.sqrt(n_bins)),
            'density_mean': float(np.mean(d_bins)),
            'density_err':  float(np.std(d_bins) / np.sqrt(n_bins)),
            'chi_mean': float(np.mean(chi_bins)),
            'chi_err': float(np.std(chi_bins) / np.sqrt(n_bins)),
            'binder_mean': float(np.mean(binder_bins)),
            'binder_err': float(np.std(binder_bins) / np.sqrt(n_bins)),
            'm_z_sq_mean': float(np.mean(m_z_sq)),
            'M': self.M
        }
