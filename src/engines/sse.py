"""
Stochastic Series Expansion (SSE) QMC Python wrapper.

Prefers the C++ backend (SSEEngine from qaqmc_cpp) when available;
falls back to the Numba-accelerated Python path automatically.
"""
import numpy as np
import h5py
from pathlib import Path
import concurrent.futures
import multiprocessing

try:
    from tqdm import trange
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import time

# ── Linux: force 'spawn' to avoid fork-after-OpenMP-init deadlock ─────────────
if multiprocessing.get_start_method(allow_none=True) is None:
    import platform
    if platform.system() == "Linux":
        multiprocessing.set_start_method("spawn", force=True)

from src.rydberg.hamiltonian import build_rydberg_vij
from src.engines.sse_updates import build_alias_table, sse_diagonal_update, sse_cluster_update
from src.analysis.measurement import calc_density, calc_staggered_magnetization

try:
    import os, shutil
    _gpp = shutil.which('g++')
    if _gpp and os.name == 'nt':
        _mingw_bin = os.path.dirname(os.path.realpath(_gpp))
        os.add_dll_directory(_mingw_bin)
    import qaqmc_cpp
    HAS_CPP = True
except (ImportError, OSError):
    HAS_CPP = False


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


def _sse_worker(init_kwargs, seed, n_equil, n_samples, worker_id, verbose):
    """Run an independent SSE Markov chain and return scalar observable arrays."""
    kw = dict(init_kwargs, seed=seed, verbose=False)
    engine = SSE_Rydberg(**kw)

    use_tqdm = HAS_TQDM and verbose and worker_id == 0

    # Equilibrate
    eq_iter = trange(n_equil, desc="Equil (W0)", leave=False) if use_tqdm else range(n_equil)
    for _ in eq_iter:
        engine.mc_step()

    # Sample
    if engine._cpp_engine is not None:
        raw = engine._cpp_engine.run(
            n_equil=0, n_samples=n_samples,
            progress_callback=None, progress_every=1
        )
        energies  = np.asarray(raw['energies'],  dtype=np.float64)
        densities = np.asarray(raw['densities'], dtype=np.float64)
        mz_arr    = np.asarray(raw['mz'],        dtype=np.float64)
        n_ops_arr = np.asarray(raw['n_ops'],     dtype=np.int32)
        M_final   = int(engine._cpp_engine.M)
    else:
        energies  = np.empty(n_samples, dtype=np.float64)
        densities = np.empty(n_samples, dtype=np.float64)
        mz_arr    = np.empty(n_samples, dtype=np.float64)
        n_ops_arr = np.empty(n_samples, dtype=np.int32)
        meas_iter = trange(n_samples, desc="Samp  (W0)", leave=False) if use_tqdm else range(n_samples)
        for step in meas_iter:
            engine.mc_step()
            obs = engine.measure_observables()
            energies[step]  = obs['energy']
            densities[step] = obs['density']
            mz_arr[step]    = obs['m_z']
            n_ops_arr[step] = engine.n_ops
        M_final = engine.M

    return energies, densities, mz_arr, n_ops_arr, M_final


class SSE_Rydberg:
    def __init__(self, N: int, Omega: float, delta: float, Rb: float,
                 beta: float, epsilon: float = 0.01, seed: int = 42,
                 pos: np.ndarray = None,
                 use_cpp: bool = True,
                 neighbor_cutoff: int = None,
                 verbose: bool = True):
        # Store init kwargs for multi-worker spawning (exclude seed/verbose)
        self._init_kwargs = {
            'N': N, 'Omega': Omega, 'delta': delta, 'Rb': Rb,
            'beta': beta, 'epsilon': epsilon, 'pos': pos,
            'use_cpp': use_cpp, 'neighbor_cutoff': neighbor_cutoff,
            'verbose': False,
        }
        self.N = N
        self.Omega = Omega
        self.delta = delta
        self.Rb = Rb
        self.beta = beta
        self.epsilon = epsilon
        self.pos = pos
        self.verbose = verbose
        self.neighbor_cutoff = neighbor_cutoff

        if self.pos is None:
            self.pos = np.arange(N).reshape(-1, 1).astype(np.float64)

        # ── Try C++ backend ──────────────────────────────────────────────────
        self._cpp_engine = None
        nc = neighbor_cutoff if neighbor_cutoff is not None else -1
        if use_cpp and HAS_CPP:
            pos_arr = np.ascontiguousarray(self.pos, dtype=np.float64)
            self._cpp_engine = qaqmc_cpp.SSEEngine(
                N=N, Omega=Omega, delta=delta, Rb=Rb,
                beta=beta, epsilon=epsilon, seed=seed,
                pos=pos_arr, neighbor_cutoff=nc
            )
            if verbose:
                n_bonds = len(self._cpp_engine.bond_sites)
                print(f"[SSE] Using C++ backend (N={N}, beta={beta}, bonds={n_bonds})")
            return

        # ── Fallback: Python/Numba path ──────────────────────────────────────
        np.random.seed(seed)
        if verbose:
            print("[SSE] Using Python/Numba fallback...")

        V, bonds_i, bonds_j, vij_list, self.bond_sites, _ = build_rydberg_vij(
            N, Omega, Rb, pos, neighbor_cutoff=neighbor_cutoff
        )

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

        self.bond_W     = np.zeros((max(n_bonds, 1), 4), dtype=np.float64)
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

        self.norm_N = sum(weights)

        # Initial configuration
        self.state    = np.random.randint(0, 2, size=N, dtype=np.int32)
        self.M        = 20
        self.op_types = np.zeros(self.M, dtype=np.int32)
        self.op_sites = np.full(self.M, -1, dtype=np.int32)
        self.n_ops    = 0

    # ── Single MCMC step ──────────────────────────────────────────────────────

    def mc_step(self):
        if self._cpp_engine is not None:
            self._cpp_engine.mc_step()
            return

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

    # ── Observables ───────────────────────────────────────────────────────────

    def measure_energy(self) -> float:
        if self._cpp_engine is not None:
            return self._cpp_engine.measure_energy()
        shift = sum(self.bond_W[b, 0] for b in range(len(self.bond_W)))
        # Correct for type-1 (diagonal) site ops that inflate n_ops:
        # sigma_x has zero diagonal, but diagonal update inserts site ops
        # as type-1, contributing N*Omega/2 false energy.
        return -self.n_ops / self.beta + shift + self.N * self.Omega / 2.0

    def measure_observables(self) -> dict:
        if self._cpp_engine is not None:
            return {
                'energy':  self._cpp_engine.measure_energy(),
                'density': self._cpp_engine.measure_density(),
                'm_z':     self._cpp_engine.measure_mz(),
            }
        return {
            'energy':  self.measure_energy(),
            'density': calc_density(self.state),
            'm_z':     calc_staggered_magnetization(self.state),
        }

    # ── Full run with binned statistics ──────────────────────────────────────

    def run(self, n_equil: int = 10000, n_measure: int = 50000,
            verbose: bool = None) -> dict:
        """
        Equilibrate and collect measurements, returning binned statistics.

        Parameters
        ----------
        n_equil   : Equilibration steps (discarded).
        n_measure : Measurement steps.
        verbose   : Show tqdm progress bar (default: inherits self.verbose).

        Returns
        -------
        dict with keys:
            energy_mean/err, density_mean/err,
            chi_mean/err (staggered susceptibility N*(<mz^2> - <|mz|>^2)),
            binder_mean/err (Binder cumulant),
            m_z_sq_mean, M (final operator string length)
        """
        if verbose is None:
            verbose = self.verbose

        use_tqdm = HAS_TQDM and verbose

        # ── C++ fast path ────────────────────────────────────────────────────
        if self._cpp_engine is not None:
            eq_iter = trange(n_equil, desc="Equil", leave=False) if use_tqdm else range(n_equil)
            for _ in eq_iter:
                self._cpp_engine.mc_step()

            result = self._cpp_engine.run(
                n_equil=0, n_samples=n_measure,
                progress_callback=None, progress_every=1
            )
            energies  = result['energies']
            densities = result['densities']
            m_z       = result['mz']
            return self._compute_stats(energies, densities, m_z,
                                       self._cpp_engine.M)

        # ── Python/Numba path ────────────────────────────────────────────────
        eq_iter = trange(n_equil, desc="Equil", leave=False) if use_tqdm else range(n_equil)
        for _ in eq_iter:
            self.mc_step()

        energies  = np.empty(n_measure)
        densities = np.empty(n_measure)
        m_z       = np.empty(n_measure)

        meas_iter = trange(n_measure, desc="Measure", leave=False) if use_tqdm else range(n_measure)
        for step in meas_iter:
            self.mc_step()
            obs = self.measure_observables()
            energies[step]  = obs['energy']
            densities[step] = obs['density']
            m_z[step]       = obs['m_z']

        return self._compute_stats(energies, densities, m_z, self.M)

    # ── Run and save raw observables to HDF5 ─────────────────────────────────

    def run_and_save(self, filepath: str, n_equil: int = 10000,
                     n_samples: int = 50000, verbose: bool = None,
                     compression: str = 'gzip', compression_opts: int = 4,
                     chunk_samples: int = 0,
                     n_jobs: int = 1, backend: str = "thread"):
        """
        Equilibrate, collect per-step measurements, and save to HDF5.

        No binning is performed here — raw scalar time series are stored so
        that any observable or autocorrelation analysis can be done offline
        via SSEArchive (postprocess.py).

        File layout
        -----------
        params/       (HDF5 group attributes)
            N, Omega, delta, Rb, beta, epsilon, n_equil, n_samples,
            M_final, backend
        geometry/
            pos          (N, d) float64 – atom positions
        samples/
            energies     (n_samples,) float64
            densities    (n_samples,) float64
            mz           (n_samples,) float64
            n_ops        (n_samples,) int32

        Parameters
        ----------
        filepath         : Output path, e.g. 'data/sse_run.h5'
        n_equil          : Equilibration steps (discarded)
        n_samples        : Number of measurement steps to save
        verbose          : Show tqdm progress bar (default: inherits self.verbose)
        compression      : HDF5 compression filter ('gzip', 'lzf', or None)
        compression_opts : gzip level (1=fast … 9=small)
        chunk_samples    : Stream to disk in chunks of this size (0 = all at once)
        n_jobs           : Number of parallel workers (independent chains)
        backend          : 'thread' or 'process' for concurrent.futures
        """
        if verbose is None:
            verbose = self.verbose
        use_tqdm = HAS_TQDM and verbose

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # ── Multi-worker path ─────────────────────────────────────────────────
        if n_jobs > 1:
            return self._run_and_save_multiworker(
                filepath, n_equil, n_samples, verbose,
                compression, compression_opts, n_jobs, backend)

        # ── Equilibrate ──────────────────────────────────────────────────────
        eq_iter = trange(n_equil, desc="Equil", leave=False) if use_tqdm else range(n_equil)
        for _ in eq_iter:
            if self._cpp_engine is not None:
                self._cpp_engine.mc_step()
            else:
                self.mc_step()

        # ── Sample ───────────────────────────────────────────────────────────
        streaming = chunk_samples > 0
        cargs = {'compression': compression, 'compression_opts': compression_opts} \
                if compression == 'gzip' else \
                {'compression': compression} if compression else {}

        if not streaming:
            # ── Collect all at once ──────────────────────────────────────────
            if self._cpp_engine is not None:
                raw = self._cpp_engine.run(
                    n_equil=0, n_samples=n_samples,
                    progress_callback=None, progress_every=1
                )
                energies  = np.asarray(raw['energies'],  dtype=np.float64)
                densities = np.asarray(raw['densities'], dtype=np.float64)
                mz_arr    = np.asarray(raw['mz'],        dtype=np.float64)
                n_ops_arr = np.asarray(raw['n_ops'],     dtype=np.int32)
                M_final   = int(self._cpp_engine.M)
                backend   = 'cpp'
            else:
                energies  = np.empty(n_samples, dtype=np.float64)
                densities = np.empty(n_samples, dtype=np.float64)
                mz_arr    = np.empty(n_samples, dtype=np.float64)
                n_ops_arr = np.empty(n_samples, dtype=np.int32)

                meas_iter = trange(n_samples, desc="Measure", leave=False) if use_tqdm else range(n_samples)
                for step in meas_iter:
                    self.mc_step()
                    obs = self.measure_observables()
                    energies[step]  = obs['energy']
                    densities[step] = obs['density']
                    mz_arr[step]    = obs['m_z']
                    n_ops_arr[step] = self.n_ops
                M_final = self.M
                backend = 'python'

            with h5py.File(filepath, 'w') as f:
                p = f.create_group('params')
                p.attrs['N']         = self.N
                p.attrs['Omega']     = self.Omega
                p.attrs['delta']     = self.delta
                p.attrs['Rb']        = self.Rb
                p.attrs['beta']      = self.beta
                p.attrs['epsilon']   = self.epsilon
                p.attrs['n_equil']   = n_equil
                p.attrs['n_samples'] = n_samples
                p.attrs['M_final']   = M_final
                p.attrs['backend']   = backend

                g = f.create_group('geometry')
                g.create_dataset('pos', data=self.pos, **cargs)

                s = f.create_group('samples')
                s.create_dataset('energies',  data=energies,  **cargs)
                s.create_dataset('densities', data=densities, **cargs)
                s.create_dataset('mz',        data=mz_arr,    **cargs)
                s.create_dataset('n_ops',     data=n_ops_arr, **cargs)

        else:
            # ── Chunked streaming: write incrementally to HDF5 ───────────
            chunk_samples = max(1, int(chunk_samples))
            backend = 'cpp' if self._cpp_engine is not None else 'python'

            with h5py.File(filepath, 'w') as f:
                p = f.create_group('params')
                p.attrs['N']         = self.N
                p.attrs['Omega']     = self.Omega
                p.attrs['delta']     = self.delta
                p.attrs['Rb']        = self.Rb
                p.attrs['beta']      = self.beta
                p.attrs['epsilon']   = self.epsilon
                p.attrs['n_equil']   = n_equil
                p.attrs['n_samples'] = n_samples
                p.attrs['M_final']   = 0
                p.attrs['backend']   = backend

                g = f.create_group('geometry')
                g.create_dataset('pos', data=self.pos, **cargs)

                sg = f.create_group('samples')
                ds_e = sg.create_dataset('energies',  shape=(n_samples,), dtype='float64', **cargs)
                ds_d = sg.create_dataset('densities', shape=(n_samples,), dtype='float64', **cargs)
                ds_m = sg.create_dataset('mz',        shape=(n_samples,), dtype='float64', **cargs)
                ds_n = sg.create_dataset('n_ops',     shape=(n_samples,), dtype='int32',   **cargs)

                samp_bar = trange(n_samples, desc="Measure", leave=False) if use_tqdm else None
                written = 0

                if self._cpp_engine is not None:
                    while written < n_samples:
                        cur = min(chunk_samples, n_samples - written)
                        raw = self._cpp_engine.run(
                            n_equil=0, n_samples=cur,
                            progress_callback=None, progress_every=1
                        )
                        ds_e[written:written + cur] = raw['energies']
                        ds_d[written:written + cur] = raw['densities']
                        ds_m[written:written + cur] = raw['mz']
                        ds_n[written:written + cur] = raw['n_ops']
                        written += cur
                        if samp_bar is not None:
                            samp_bar.update(cur)
                    M_final = int(self._cpp_engine.M)
                else:
                    while written < n_samples:
                        cur = min(chunk_samples, n_samples - written)
                        e_buf = np.empty(cur, dtype=np.float64)
                        d_buf = np.empty(cur, dtype=np.float64)
                        m_buf = np.empty(cur, dtype=np.float64)
                        n_buf = np.empty(cur, dtype=np.int32)
                        for i in range(cur):
                            self.mc_step()
                            obs = self.measure_observables()
                            e_buf[i] = obs['energy']
                            d_buf[i] = obs['density']
                            m_buf[i] = obs['m_z']
                            n_buf[i] = self.n_ops
                        ds_e[written:written + cur] = e_buf
                        ds_d[written:written + cur] = d_buf
                        ds_m[written:written + cur] = m_buf
                        ds_n[written:written + cur] = n_buf
                        written += cur
                        if samp_bar is not None:
                            samp_bar.update(cur)
                    M_final = self.M

                if samp_bar is not None:
                    samp_bar.close()
                p.attrs['M_final'] = M_final

        if verbose:
            print(f"[SSE] Saved {n_samples} samples -> {filepath}")

    def _run_and_save_multiworker(self, filepath, n_equil, n_samples,
                                   verbose, compression, compression_opts,
                                   n_jobs, backend):
        """Run n_jobs independent SSE chains and merge results into one HDF5."""
        counts = _split_work(n_samples, n_jobs)
        base_seed = hash((self._init_kwargs.get('N'), id(self))) & 0x7FFFFFFF

        executor_cls = _get_executor_class(backend)
        futures = []
        with executor_cls(max_workers=n_jobs) as executor:
            for i, count in enumerate(counts):
                if count <= 0:
                    continue
                seed_i = base_seed + (i + 1) * 9973
                futures.append(executor.submit(
                    _sse_worker, self._init_kwargs, seed_i,
                    n_equil, count, i, verbose))

        # Gather results
        all_e, all_d, all_m, all_n = [], [], [], []
        M_final = 0
        for fut in futures:
            e, d, m, n, mf = fut.result()
            all_e.append(e)
            all_d.append(d)
            all_m.append(m)
            all_n.append(n)
            M_final = max(M_final, mf)

        energies  = np.concatenate(all_e)
        densities = np.concatenate(all_d)
        mz_arr    = np.concatenate(all_m)
        n_ops_arr = np.concatenate(all_n)
        backend_str = 'cpp' if self._cpp_engine is not None else 'python'

        cargs = {'compression': compression, 'compression_opts': compression_opts} \
                if compression == 'gzip' else \
                {'compression': compression} if compression else {}

        with h5py.File(filepath, 'w') as f:
            p = f.create_group('params')
            p.attrs['N']         = self.N
            p.attrs['Omega']     = self.Omega
            p.attrs['delta']     = self.delta
            p.attrs['Rb']        = self.Rb
            p.attrs['beta']      = self.beta
            p.attrs['epsilon']   = self.epsilon
            p.attrs['n_equil']   = n_equil
            p.attrs['n_samples'] = n_samples
            p.attrs['n_jobs']    = n_jobs
            p.attrs['M_final']   = M_final
            p.attrs['backend']   = backend_str

            g = f.create_group('geometry')
            g.create_dataset('pos', data=self.pos, **cargs)

            s = f.create_group('samples')
            s.create_dataset('energies',  data=energies,  **cargs)
            s.create_dataset('densities', data=densities, **cargs)
            s.create_dataset('mz',        data=mz_arr,    **cargs)
            s.create_dataset('n_ops',     data=n_ops_arr, **cargs)

        if verbose:
            print(f"[SSE] Saved {n_samples} samples ({n_jobs} workers) -> {filepath}")

    @staticmethod
    def _compute_stats(energies: np.ndarray, densities: np.ndarray,
                       m_z: np.ndarray, M: int,
                       n_bins: int = 50) -> dict:
        n_measure = len(energies)
        bs = max(1, n_measure // n_bins)

        m_z_sq  = m_z ** 2
        m_z_abs = np.abs(m_z)

        e_bins = np.array([np.mean(energies[i*bs:(i+1)*bs])  for i in range(n_bins)])
        d_bins = np.array([np.mean(densities[i*bs:(i+1)*bs]) for i in range(n_bins)])

        chi_bins = np.array([
            np.mean(m_z_sq[i*bs:(i+1)*bs]) - np.mean(m_z_abs[i*bs:(i+1)*bs]) ** 2
            for i in range(n_bins)
        ]) * len(m_z) / n_bins  # approximate N scaling

        binder_bins = np.array([
            1.5 * (1.0 - np.mean(m_z_sq[i*bs:(i+1)*bs] ** 2)
                   / (3.0 * np.mean(m_z_sq[i*bs:(i+1)*bs]) ** 2 + 1e-12))
            for i in range(n_bins)
        ])

        return {
            'energy_mean':  float(np.mean(e_bins)),
            'energy_err':   float(np.std(e_bins)  / np.sqrt(n_bins)),
            'density_mean': float(np.mean(d_bins)),
            'density_err':  float(np.std(d_bins)  / np.sqrt(n_bins)),
            'chi_mean':     float(np.mean(chi_bins)),
            'chi_err':      float(np.std(chi_bins) / np.sqrt(n_bins)),
            'binder_mean':  float(np.mean(binder_bins)),
            'binder_err':   float(np.std(binder_bins) / np.sqrt(n_bins)),
            'm_z_sq_mean':  float(np.mean(m_z_sq)),
            'M':            M,
        }
