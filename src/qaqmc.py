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
import h5py
import datetime
import time

from src.hamiltonian import build_rydberg_vij
from src.qaqmc_updates import build_qaqmc_alias_tables, qaqmc_diagonal_update, qaqmc_cluster_update


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


def _run_and_save_worker(kwargs, seed, n_equil, n_samples, worker_id, verbose):
    np.random.seed(seed)
    instance = QAQMC_Rydberg(**kwargs)
    
    use_tqdm = HAS_TQDM and verbose and worker_id == 0
    t0_equil = time.perf_counter()
    equil_iter = trange(n_equil, desc="Equil (W0)", leave=False) if use_tqdm else range(n_equil)
    for _ in equil_iter:
        instance.mc_step()
    t_equil = time.perf_counter() - t0_equil
        
    M2 = instance.M_total
    types_arr = np.empty((n_samples, M2), dtype=np.int8)
    sites_arr = np.empty((n_samples, M2), dtype=np.int16)
    
    t0_sample = time.perf_counter()
    measure_iter = trange(n_samples, desc="Samp  (W0)", leave=False) if use_tqdm else range(n_samples)
    for i in measure_iter:
        instance.mc_step()
        types_arr[i] = instance.op_types[:M2].astype(np.int8)
        sites_arr[i] = instance.op_sites[:M2].astype(np.int16)
    t_sample = time.perf_counter() - t0_sample
        
    return types_arr, sites_arr, t_equil, t_sample




class QAQMC_Rydberg:
    def __init__(self, N: int, Omega: float, delta_min: float, delta_max: float, Rb: float,
                 M: int, epsilon: float = 0.01, seed: int = 42, pos: np.ndarray = None,
                 verbose: bool = True, n_jobs: int = 1, backend: str = "process"):
        self.init_kwargs = {
            'N': N, 'Omega': Omega, 'delta_min': delta_min, 'delta_max': delta_max,
            'Rb': Rb, 'M': M, 'epsilon': epsilon, 'seed': seed, 'pos': pos,
            'verbose': False, 'n_jobs': 1, 'backend': "thread" # Workers shouldn't spawn more workers
        }
        self.N = N
        self.Omega = Omega
        self.Rb = Rb
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.backend = backend
        
        self.M = M
        self.M_total = 2 * M
        
        self.pos = pos
        if self.pos is None:
            self.pos = np.arange(N).reshape(-1, 1).astype(np.float64)

        np.random.seed(seed)
        
        # We only need the distance-dependent V_ij without the time-dependent delta yet
        _, bonds_i, bonds_j, vij_list, self.bond_sites = build_rydberg_vij(
            N, Omega, Rb, pos, verbose=verbose, n_jobs=n_jobs, backend=backend
        )
        
        n_bonds = len(bonds_i)
        
        # The evolution sweep delta_min -> delta_max -> delta_min
        delta_sched = np.empty(self.M_total, dtype=np.float64)
        for p in range(self.M):
            delta_sched[p] = delta_min + (delta_max - delta_min) * (p / self.M)
        for p in range(self.M, self.M_total):
            delta_sched[p] = delta_max - (delta_max - delta_min) * ((p - self.M) / self.M)
            
        res = build_qaqmc_alias_tables(self.M_total, N, n_bonds, Omega, delta_sched, vij_list, epsilon)
        self.bond_W_all = res[0]
        self.bond_W_max_all = res[1]
        self.n_alias_all = res[2]
        self.alias_prob_all = res[3]
        self.alias_idx_all = res[4]
        self.op_map_kind_all = res[5]
        self.op_map_loc_all = res[6]
            
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
            
    def run_and_save(self, filepath: str, n_equil: int = 5000,
                     n_samples: int = 10000, verbose: bool = True,
                     compression: str = 'gzip', compression_opts: int = 4,
                     n_jobs: int = 1, backend: str = "thread"):
        """
        Run QAQMC and save every operator-sequence snapshot to an HDF5 file.

        No measurement is performed here—the raw operator sequences are stored
        so that *any* observable can be computed offline via postprocess.py.

        File layout
        -----------
        params/           (HDF5 group attributes)
            N, Omega, Rb, delta_min, delta_max, M, epsilon, seed, timestamp
        geometry/
            pos            (N, d) float64 – atom coordinates
        schedule/
            delta_schedule (2M,) float64 – δ(p) for each imaginary-time slice
        samples/
            op_types       (n_samples, 2M) int8  – operator type per slice
            op_sites       (n_samples, 2M) int16 – site/bond index per slice

        Parameters
        ----------
        filepath       : Output path, e.g. 'data/run_N8_M512.h5'
        n_equil        : Equilibration steps (discarded, not saved)
        n_samples      : Number of MCMC snapshots to save
        verbose        : Show tqdm progress bar
        compression    : HDF5 compression filter ('gzip', 'lzf', or None)
        compression_opts: gzip compression level (1=fast … 9=small)
        n_jobs         : Number of parallel workers
        backend        : "thread" or "process" backend for concurrent.futures
        """
        # Warm-up JIT in main thread
        self.mc_step()

        M2 = self.M_total  # 2M
        t0_overall = time.perf_counter()

        if n_jobs > 1:
            futures = []
            counts = _split_work(n_samples, n_jobs)
            executor_cls = _get_executor_class(backend)
            with executor_cls(max_workers=n_jobs) as executor:
                for i, count in enumerate(counts):
                    if count <= 0:
                        continue
                    seed_i = self.init_kwargs['seed'] + (i + 1) * 1234
                    futures.append(executor.submit(_run_and_save_worker, self.init_kwargs, seed_i, n_equil, count, i, verbose))
                    
            types_list, sites_list = [], []
            t_equil_max, t_sample_max = 0.0, 0.0
            for f in futures:
                t_arr, s_arr, t_eq, t_sa = f.result()
                types_list.append(t_arr)
                sites_list.append(s_arr)
                t_equil_max = max(t_equil_max, t_eq)
                t_sample_max = max(t_sample_max, t_sa)
                
            all_types = np.vstack(types_list)
            all_sites = np.vstack(sites_list)
            t_equil = t_equil_max
            t_sample = t_sample_max
        else:
            all_types, all_sites, t_equil, t_sample = _run_and_save_worker(
                self.init_kwargs, self.init_kwargs['seed'], n_equil, n_samples, 0, verbose
            )

        kw = dict(compression=compression, compression_opts=compression_opts) \
             if compression == 'gzip' else dict(compression=compression) \
             if compression else {}

        kw_i8  = dict(**kw, dtype='int8')
        kw_i16 = dict(**kw, dtype='int16')

        with h5py.File(filepath, 'w') as f:

            # ── metadata ─────────────────────────────────────────────────────
            pg = f.create_group('params')
            for k, v in self.init_kwargs.items():
                if k != 'pos' and v is not None:
                    pg.attrs[k] = v
            pg.attrs['n_equil']        = n_equil
            pg.attrs['n_samples']       = n_samples
            pg.attrs['timestamp']       = datetime.datetime.utcnow().isoformat()
            pg.attrs['equil_time_s']    = t_equil   # max wall time for equilibration

            # ── geometry ─────────────────────────────────────────────────────
            gg = f.create_group('geometry')
            pos_stored = self.pos if self.pos is not None \
                         else np.arange(self.N).reshape(-1, 1).astype(np.float64)
            gg.create_dataset('pos', data=pos_stored.astype(np.float64))

            # ── δ schedule ───────────────────────────────────────────────────
            sg = f.create_group('schedule')
            delta_sched = np.empty(M2, dtype=np.float64)
            for p in range(self.M):
                delta_sched[p] = (self.delta_min
                                  + (self.delta_max - self.delta_min) * (p / self.M))
            for p in range(self.M, M2):
                delta_sched[p] = (self.delta_max
                                  - (self.delta_max - self.delta_min) * ((p - self.M) / self.M))
            sg.create_dataset('delta_schedule', data=delta_sched)

            # ── sample datasets ──────────────────────────────────────────────
            smg = f.create_group('samples')
            smg.create_dataset('op_types', data=all_types, **kw_i8)
            smg.create_dataset('op_sites', data=all_sites, **kw_i16)

            pg.attrs['sample_time_s'] = t_sample
            pg.attrs['total_time_s']  = t_equil + t_sample

        if verbose:
            total = time.perf_counter() - t0_overall
            print(f"Saved {n_samples} samples → {filepath}  "
                  f"(workers max equil {t_equil:.1f}s + sample {t_sample:.1f}s, overall {total:.1f}s)")
