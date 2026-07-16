"""
Stochastic Series Expansion (SSE) QMC Python wrapper.

Thin API layer over the C++ engine (qaqmc_cpp.SSEEngine).  The historical
pure-Python/Numba fallback engine and the in-process multi-worker mode were
removed in 2026-07: no callers, physics had diverged from the C++ engine
(periodic spatial boundary, seam, chi_F), and silently degrading on a missing
.so produced wrong-physics data instead of a loud failure.  Parallelism
happens at the MPI-rank level (src/mpi/sse_mpi.py).  The old implementation
remains in git history (src/engines/sse_updates.py before this commit).
"""
import numpy as np
import h5py
from pathlib import Path

try:
    from tqdm import trange
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import os, shutil
    _gpp = shutil.which('g++')
    if _gpp and os.name == 'nt':
        _mingw_bin = os.path.dirname(os.path.realpath(_gpp))
        os.add_dll_directory(_mingw_bin)
    import qaqmc_cpp
    HAS_CPP = True
    _CPP_IMPORT_ERROR = None
except (ImportError, OSError) as _e:
    HAS_CPP = False
    _CPP_IMPORT_ERROR = _e


def _require_cpp():
    if not HAS_CPP:
        raise ImportError(
            "qaqmc_cpp extension is not importable — build it first "
            "(cmake -S . -B build -G Ninja && cmake --build build, see README "
            "部署步驟) and ensure the build dir or a deployed .so is on "
            f"PYTHONPATH.  Original error: {_CPP_IMPORT_ERROR!r}"
        )


class SSE_Rydberg:
    def __init__(self, N: int, Omega: float, delta: float, Rb: float,
                 beta: float, epsilon: float = 0.01, seed: int = 42,
                 pos: np.ndarray = None,
                 use_cpp: bool = True,
                 neighbor_cutoff: int = None,
                 box_vectors: np.ndarray = None,
                 verbose: bool = True):
        if not use_cpp:
            raise ValueError(
                "use_cpp=False (the Python/Numba fallback engine) was removed "
                "2026-07 — the C++ engine is the only implementation.")
        _require_cpp()

        self._init_kwargs = {
            'N': N, 'Omega': Omega, 'delta': delta, 'Rb': Rb,
            'beta': beta, 'epsilon': epsilon, 'pos': pos,
            'use_cpp': True, 'neighbor_cutoff': neighbor_cutoff,
            'box_vectors': box_vectors,
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

        nc = neighbor_cutoff if neighbor_cutoff is not None else -1
        box = (np.ascontiguousarray(box_vectors, dtype=np.float64)
               if box_vectors is not None else None)
        pos_arr = np.ascontiguousarray(self.pos, dtype=np.float64)
        self._cpp_engine = qaqmc_cpp.SSEEngine(
            N=N, Omega=Omega, delta=delta, Rb=Rb,
            beta=beta, epsilon=epsilon, seed=seed,
            pos=pos_arr, neighbor_cutoff=nc, box_vectors=box
        )
        if verbose:
            n_bonds = len(self._cpp_engine.bond_sites)
            print(f"[SSE] Using C++ backend (N={N}, beta={beta}, bonds={n_bonds})")

    # ── Single MCMC step ──────────────────────────────────────────────────────

    def mc_step(self):
        self._cpp_engine.mc_step()

    # ── Observables ───────────────────────────────────────────────────────────

    def measure_energy(self) -> float:
        return self._cpp_engine.measure_energy()

    def measure_observables(self) -> dict:
        return {
            'energy':  self._cpp_engine.measure_energy(),
            'density': self._cpp_engine.measure_density(),
            'm_z':     self._cpp_engine.measure_mz(),
        }

    @property
    def n_ops(self) -> int:
        return int(self._cpp_engine.n_ops)

    @property
    def M(self) -> int:
        return int(self._cpp_engine.M)

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

    # ── Run and save raw observables to HDF5 ─────────────────────────────────

    def run_and_save(self, filepath: str, n_equil: int = 10000,
                     n_samples: int = 50000, verbose: bool = None,
                     compression: str = 'gzip', compression_opts: int = 4,
                     chunk_samples: int = 0):
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
        """
        if verbose is None:
            verbose = self.verbose
        use_tqdm = HAS_TQDM and verbose

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # ── Equilibrate ──────────────────────────────────────────────────────
        eq_iter = trange(n_equil, desc="Equil", leave=False) if use_tqdm else range(n_equil)
        for _ in eq_iter:
            self._cpp_engine.mc_step()

        # ── Sample ───────────────────────────────────────────────────────────
        streaming = chunk_samples > 0
        cargs = {'compression': compression, 'compression_opts': compression_opts} \
                if compression == 'gzip' else \
                {'compression': compression} if compression else {}

        if not streaming:
            raw = self._cpp_engine.run(
                n_equil=0, n_samples=n_samples,
                progress_callback=None, progress_every=1
            )
            energies  = np.asarray(raw['energies'],  dtype=np.float64)
            densities = np.asarray(raw['densities'], dtype=np.float64)
            mz_arr    = np.asarray(raw['mz'],        dtype=np.float64)
            n_ops_arr = np.asarray(raw['n_ops'],     dtype=np.int32)
            M_final   = int(self._cpp_engine.M)

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
                p.attrs['backend']   = 'cpp'

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
                p.attrs['backend']   = 'cpp'

                g = f.create_group('geometry')
                g.create_dataset('pos', data=self.pos, **cargs)

                sg = f.create_group('samples')
                ds_e = sg.create_dataset('energies',  shape=(n_samples,), dtype='float64', **cargs)
                ds_d = sg.create_dataset('densities', shape=(n_samples,), dtype='float64', **cargs)
                ds_m = sg.create_dataset('mz',        shape=(n_samples,), dtype='float64', **cargs)
                ds_n = sg.create_dataset('n_ops',     shape=(n_samples,), dtype='int32',   **cargs)

                samp_bar = trange(n_samples, desc="Measure", leave=False) if use_tqdm else None
                written = 0
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

                if samp_bar is not None:
                    samp_bar.close()
                p.attrs['M_final'] = int(self._cpp_engine.M)

        if verbose:
            print(f"[SSE] Saved {n_samples} samples -> {filepath}")

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
