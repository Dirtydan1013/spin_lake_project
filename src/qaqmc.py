"""
Quasi-Adiabatic Quantum Monte Carlo (QAQMC) Python wrapper.
"""
import numpy as np

try:
    from tqdm import trange
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import concurrent.futures

from src.hamiltonian import build_rydberg_vij
from src.qmc_updates import build_alias_table, qaqmc_diagonal_update, qaqmc_cluster_update
from src.measurement import (
    measure_symmetric_from_ops_numba,
    measure_asymmetric_all_from_ops_numba,
)

def _split_work(n_total, n_jobs):
    base = n_total // n_jobs
    rem = n_total % n_jobs
    return [base + (1 if i < rem else 0) for i in range(n_jobs)]

def _get_executor_class(backend):
    if backend == "thread":
        return concurrent.futures.ThreadPoolExecutor
    if backend == "process":
        return concurrent.futures.ProcessPoolExecutor
    raise ValueError(f"Unsupported backend={backend!r}. Use 'thread' or 'process'.")

def _run_asymmetric_worker(kwargs, seed, n_equil, n_measure, worker_id, verbose):
    np.random.seed(seed)
    instance = QAQMC_Rydberg(**kwargs)
    
    use_tqdm = HAS_TQDM and verbose and worker_id == 0
    equil_iter = trange(n_equil, desc="Equil (W0)", leave=False) if use_tqdm else range(n_equil)
    for _ in equil_iter:
        instance.mc_step()
        
    density_sum = np.zeros(instance.M, dtype=np.float64)
    density_sumsq = np.zeros(instance.M, dtype=np.float64)
    mz_abs_sum = np.zeros(instance.M, dtype=np.float64)
    mz_sq_sum = np.zeros(instance.M, dtype=np.float64)
    mz_quad_sum = np.zeros(instance.M, dtype=np.float64)
    
    measure_iter = trange(n_measure, desc="Meas  (W0)", leave=False) if use_tqdm else range(n_measure)
    for step in measure_iter:
        instance.mc_step()
        d_arr, mz_arr = instance.measure_asymmetric_all()
        mz_sq = mz_arr * mz_arr
        density_sum += d_arr
        density_sumsq += d_arr * d_arr
        mz_abs_sum += np.abs(mz_arr)
        mz_sq_sum += mz_sq
        mz_quad_sum += mz_sq * mz_sq

    return n_measure, density_sum, density_sumsq, mz_abs_sum, mz_sq_sum, mz_quad_sum

def _run_symmetric_worker(kwargs, seed, n_equil, n_measure, worker_id, verbose):
    np.random.seed(seed)
    instance = QAQMC_Rydberg(**kwargs)
    
    use_tqdm = HAS_TQDM and verbose and worker_id == 0
    equil_iter = trange(n_equil, desc="Equil (W0)", leave=False) if use_tqdm else range(n_equil)
    for _ in equil_iter:
        instance.mc_step()
        
    densities = np.empty(n_measure)
    m_z = np.empty(n_measure)
    
    measure_iter = trange(n_measure, desc="Meas  (W0)", leave=False) if use_tqdm else range(n_measure)
    for step in measure_iter:
        instance.mc_step()
        obs = instance.measure_symmetric()
        densities[step] = obs['density']
        m_z[step] = obs['m_z']
        
    return densities, m_z

class QAQMC_Rydberg:
    def __init__(self, N: int, Omega: float, delta_min: float, delta_max: float, Rb: float,
                 M: int, epsilon: float = 0.01, seed: int = 42, pos: np.ndarray = None):
        self.init_kwargs = {
            'N': N, 'Omega': Omega, 'delta_min': delta_min, 'delta_max': delta_max,
            'Rb': Rb, 'M': M, 'epsilon': epsilon, 'seed': seed, 'pos': pos
        }
        self.N = N
        self.Omega = Omega
        self.Rb = Rb
        self.delta_min = delta_min
        self.delta_max = delta_max
        
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
        density, m_z = measure_symmetric_from_ops_numba(
            self.op_types, self.op_sites, self.N, self.M
        )
        return {
            'density': float(density),
            'm_z': float(m_z)
        }
        
    def measure_asymmetric_all(self):
        """Measures observables at every time slice p from 0 to M."""
        return measure_asymmetric_all_from_ops_numba(
            self.op_types, self.op_sites, self.N, self.M
        )
        
    def run_asymmetric(self, n_equil=5000, n_measure=10000, verbose=True, n_jobs=1, backend="thread"):
        """Runs QAQMC and collects asymmetric expectation values across the parameter sweep."""
        if n_jobs > 1:
            futures = []
            counts = _split_work(n_measure, n_jobs)
            executor_cls = _get_executor_class(backend)
            with executor_cls(max_workers=n_jobs) as executor:
                for i, count in enumerate(counts):
                    if count <= 0:
                        continue
                    seed_i = self.init_kwargs['seed'] + (i + 1) * 1234
                    futures.append(executor.submit(_run_asymmetric_worker, self.init_kwargs, seed_i, n_equil, count, i, verbose))

            actual_measure = 0
            density_sum = np.zeros(self.M, dtype=np.float64)
            density_sumsq = np.zeros(self.M, dtype=np.float64)
            mz_abs_sum = np.zeros(self.M, dtype=np.float64)
            mz_sq_sum = np.zeros(self.M, dtype=np.float64)
            mz_quad_sum = np.zeros(self.M, dtype=np.float64)
            for f in futures:
                c, d_sum, d_sumsq, mz_abs_s, mz_sq_s, mz_quad_s = f.result()
                actual_measure += c
                density_sum += d_sum
                density_sumsq += d_sumsq
                mz_abs_sum += mz_abs_s
                mz_sq_sum += mz_sq_s
                mz_quad_sum += mz_quad_s
        else:
            actual_measure = n_measure
            equil_iter = trange(n_equil, desc="Equil  ", leave=False) if (HAS_TQDM and verbose) else range(n_equil)
            for _ in equil_iter:
                self.mc_step()

            density_sum = np.zeros(self.M, dtype=np.float64)
            density_sumsq = np.zeros(self.M, dtype=np.float64)
            mz_abs_sum = np.zeros(self.M, dtype=np.float64)
            mz_sq_sum = np.zeros(self.M, dtype=np.float64)
            mz_quad_sum = np.zeros(self.M, dtype=np.float64)
            
            measure_iter = trange(n_measure, desc="Measure", leave=False) if (HAS_TQDM and verbose) else range(n_measure)
            for step in measure_iter:
                self.mc_step()
                d_arr, mz_arr = self.measure_asymmetric_all()
                mz_sq = mz_arr * mz_arr
                density_sum += d_arr
                density_sumsq += d_arr * d_arr
                mz_abs_sum += np.abs(mz_arr)
                mz_sq_sum += mz_sq
                mz_quad_sum += mz_sq * mz_sq

        inv_n = 1.0 / actual_measure
        densities_mean = density_sum * inv_n
        density_var = np.maximum(density_sumsq * inv_n - densities_mean**2, 0.0)
        densities_err = np.sqrt(density_var * inv_n)

        m_z_sq_mean = mz_sq_sum * inv_n
        m_z_abs_mean = mz_abs_sum * inv_n
        m_z_quad_mean = mz_quad_sum * inv_n
        m_z_sq_var = np.maximum(m_z_quad_mean - m_z_sq_mean**2, 0.0)
        m_z_sq_err = np.sqrt(m_z_sq_var * inv_n)
        
        chi_mean = self.N * (m_z_sq_mean - m_z_abs_mean**2)
        binder_mean = 1.5 * (1.0 - m_z_quad_mean / (3.0 * m_z_sq_mean**2 + 1e-12))
        
        deltas = np.linspace(self.delta_min, self.delta_max, self.M, endpoint=False)
        
        return {
            'deltas': deltas,
            'density_mean': densities_mean,
            'density_err': densities_err,
            'chi_mean': chi_mean,
            'binder_mean': binder_mean,
            'm_z_sq_mean': m_z_sq_mean,
            'm_z_sq_err': m_z_sq_err
        }
        
    def run(self, n_equil=5000, n_measure=10000, verbose=True, n_jobs=1, backend="thread"):
        if n_jobs > 1:
            futures = []
            counts = _split_work(n_measure, n_jobs)
            executor_cls = _get_executor_class(backend)
            with executor_cls(max_workers=n_jobs) as executor:
                for i, count in enumerate(counts):
                    if count <= 0:
                        continue
                    seed_i = self.init_kwargs['seed'] + (i + 1) * 1234
                    futures.append(executor.submit(_run_symmetric_worker, self.init_kwargs, seed_i, n_equil, count, i, verbose))
                    
            densities_list, mz_list = [], []
            for f in futures:
                d, mz = f.result()
                densities_list.append(d)
                mz_list.append(mz)
                
            densities = np.concatenate(densities_list)
            m_z = np.concatenate(mz_list)
            actual_measure = densities.shape[0]
        else:
            actual_measure = n_measure
            equil_iter = trange(n_equil, desc="Equil  ", leave=False) if (HAS_TQDM and verbose) else range(n_equil)
            for _ in equil_iter:
                self.mc_step()
                
            densities = np.empty(n_measure)
            m_z = np.empty(n_measure)
            
            measure_iter = trange(n_measure, desc="Measure", leave=False) if (HAS_TQDM and verbose) else range(n_measure)
            for step in measure_iter:
                self.mc_step()
                obs = self.measure_symmetric()
                densities[step] = obs['density']
                m_z[step] = obs['m_z']
            
        n_bins = min(50, actual_measure)
        bs = actual_measure // n_bins
        if n_bins == 0 or bs == 0:
            bs = 1
            n_bins = actual_measure
            
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
            'density_err':  float(np.std(d_bins) / np.sqrt(n_bins)) if n_bins > 0 else 0.0,
            'chi_mean': float(np.mean(chi_bins)),
            'chi_err': float(np.std(chi_bins) / np.sqrt(n_bins)) if n_bins > 0 else 0.0,
            'binder_mean': float(np.mean(binder_bins)),
            'binder_err': float(np.std(binder_bins) / np.sqrt(n_bins)) if n_bins > 0 else 0.0,
            'm_z_sq_mean': float(np.mean(m_z_sq)),
            'M': self.M
        }
