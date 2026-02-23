"""
Quasi-Adiabatic Quantum Monte Carlo (QAQMC) Python wrapper.
"""
import numpy as np

from src.hamiltonian import build_rydberg_vij
from src.qmc_updates import build_alias_table, qaqmc_diagonal_update, qaqmc_cluster_update
from src.measurement import calc_density, calc_staggered_magnetization

class QAQMC_Rydberg:
    def __init__(self, N: int, Omega: float, delta_min: float, delta_max: float, Rb: float,
                 M: int, epsilon: float = 0.01, seed: int = 42, pos: np.ndarray = None):
        self.N = N
        self.Omega = Omega
        self.Rb = Rb
        
        self.M = M
        self.M_total = 2 * M
        
        self.pos = pos
        if self.pos is None:
            self.pos = np.arange(N).reshape(-1, 1)

        np.random.seed(seed)
        
        # We only need the distance-dependent V_ij without the time-dependent delta yet
        _, bonds_i, bonds_j, vij_list, self.bond_sites = build_rydberg_vij(N, Omega, Rb, pos)
        
        n_bonds = len(bonds_i)
        
        # The evolution sweep delta_min -> delta_max -> delta_min
        delta_sched = np.empty(self.M_total, dtype=np.float64)
        for p in range(self.M):
            delta_sched[p] = delta_min + (delta_max - delta_min) * (p / self.M)
        for p in range(self.M, self.M_total):
            delta_sched[p] = delta_max - (delta_max - delta_min) * ((p - self.M) / self.M)
            
        max_alias = N + n_bonds
        
        self.bond_W_all = np.zeros((self.M_total, max(n_bonds, 1), 4), dtype=np.float64)
        self.bond_W_max_all = np.zeros((self.M_total, max(n_bonds, 1)), dtype=np.float64)
        
        self.n_alias_all = np.zeros(self.M_total, dtype=np.int32)
        self.alias_prob_all = np.zeros((self.M_total, max_alias), dtype=np.float64)
        self.alias_idx_all = np.zeros((self.M_total, max_alias), dtype=np.int64)
        self.op_map_kind_all = np.zeros((self.M_total, max_alias), dtype=np.int32)
        self.op_map_loc_all = np.zeros((self.M_total, max_alias), dtype=np.int32)
        
        for p in range(self.M_total):
            delta = delta_sched[p]
            delta_b = delta / (N - 1) if N > 1 else delta
            
            weights = []
            op_map_kind = []
            op_map_loc = []
            
            for i in range(N):
                weights.append(Omega / 2.0)
                op_map_kind.append(0)
                op_map_loc.append(i)
                
            for b in range(n_bonds):
                vij = vij_list[b]
                m1 = min(0.0, delta_b, 2 * delta_b - vij)
                m2 = min(delta_b, 2 * delta_b - vij)
                cij = abs(m1) + epsilon * abs(m2)
                
                self.bond_W_all[p, b, 0] = cij
                self.bond_W_all[p, b, 1] = delta_b + cij
                self.bond_W_all[p, b, 2] = delta_b + cij
                self.bond_W_all[p, b, 3] = -vij + 2 * delta_b + cij
                bmax = np.max(self.bond_W_all[p, b])
                self.bond_W_max_all[p, b] = bmax
                
                weights.append(bmax)
                op_map_kind.append(1)
                op_map_loc.append(b)
                
            n_a = len(weights)
            self.n_alias_all[p] = n_a
            self.op_map_kind_all[p, :n_a] = op_map_kind
            self.op_map_loc_all[p, :n_a] = op_map_loc
            
            p_arr, i_arr = build_alias_table(weights)
            self.alias_prob_all[p, :n_a] = p_arr
            self.alias_idx_all[p, :n_a] = i_arr
            
        self.site_W = Omega / 2.0
        self.site_W_max = Omega / 2.0
        
        self.op_types = np.ones(self.M_total, dtype=np.int32) 
        self.op_sites = np.zeros(self.M_total, dtype=np.int32)
        self.state = np.zeros(N, dtype=np.int32)

    def mc_step(self):
        boundary_state = np.zeros(self.N, dtype=np.int32)
        
        qaqmc_diagonal_update(
            self.op_types, self.op_sites, boundary_state,
            self.M_total,
            self.bond_sites, self.bond_W_all, self.bond_W_max_all,
            self.n_alias_all, self.alias_prob_all, self.alias_idx_all,
            self.op_map_kind_all, self.op_map_loc_all,
            self.site_W, self.site_W_max, self.N)
            
        boundary_state = np.zeros(self.N, dtype=np.int32)
            
        qaqmc_cluster_update(
            self.op_types, self.op_sites, boundary_state, 
            self.M_total, self.N,
            self.bond_sites, self.bond_W_all)
            
    def measure_symmetric(self) -> dict:
        cur = np.zeros(self.N, dtype=np.int32)
        for p in range(self.M):
            if self.op_types[p] == -1:
                cur[self.op_sites[p]] ^= 1
                
        return {
            'density': calc_density(cur),
            'm_z': calc_staggered_magnetization(cur)
        }
        
    def run(self, n_equil=5000, n_measure=10000):
        for _ in range(n_equil):
            self.mc_step()
            
        densities = np.empty(n_measure)
        m_z = np.empty(n_measure)
        
        for step in range(n_measure):
            self.mc_step()
            obs = self.measure_symmetric()
            densities[step] = obs['density']
            m_z[step] = obs['m_z']
            
        n_bins = 50
        bs = n_measure // n_bins
        d_bins = np.array([np.mean(densities[i*bs:(i+1)*bs]) for i in range(n_bins)])
        
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
            'density_mean': float(np.mean(d_bins)),
            'density_err':  float(np.std(d_bins) / np.sqrt(n_bins)),
            'chi_mean': float(np.mean(chi_bins)),
            'chi_err': float(np.std(chi_bins) / np.sqrt(n_bins)),
            'binder_mean': float(np.mean(binder_bins)),
            'binder_err': float(np.std(binder_bins) / np.sqrt(n_bins)),
            'm_z_sq_mean': float(np.mean(m_z_sq)),
            'M': self.M
        }
