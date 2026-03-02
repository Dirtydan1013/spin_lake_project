"""
postprocess.py — Offline analysis of QAQMC operator-sequence archives.

Workflow
--------
1. Run a simulation and save raw data:
       qaqmc = QAQMC_Rydberg(...)
       qaqmc.run_and_save('data/run.h5', n_equil=5000, n_samples=30000)

2. Load the archive and compute any observable you like:
       from src.postprocess import QAQMCArchive
       arc = QAQMCArchive('data/run.h5')
       density  = arc.compute(obs_density_asym)
       ms2_mean = arc.compute(obs_ms2_sym)
"""

import numpy as np
import h5py
from pathlib import Path
import concurrent.futures
from copy import copy


# ── Worker function ───────────────────────────────────────────────────────────
def _postprocess_worker_chunk(arc, observable_fn, chunk_start, chunk_stop):
    chunk_results = []
    for ot, os in arc.iter_samples(chunk_start, chunk_stop):
        chunk_results.append(observable_fn(ot, os, arc))
    return chunk_results


# ── Archive reader ────────────────────────────────────────────────────────────

class QAQMCArchive:
    """
    Read-only view of a QAQMC HDF5 archive produced by `run_and_save`.

    Attributes
    ----------
    params         : dict   – all simulation parameters (N, M, Omega, …)
    pos            : ndarray (N, d) – atom coordinates
    delta_schedule : ndarray (2M,) – δ at each imaginary-time slice
    N, M, M_total  : int
    n_samples      : int
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(filepath)

        with h5py.File(filepath, 'r') as f:
            # Parameters
            self.params = dict(f['params'].attrs)
            self.N          = int(self.params['N'])
            self.M          = int(self.params['M'])
            self.M_total    = 2 * self.M
            self.n_samples  = int(self.params['n_samples'])
            self.Omega      = float(self.params['Omega'])
            self.Rb         = float(self.params['Rb'])
            self.delta_min  = float(self.params['delta_min'])
            self.delta_max  = float(self.params['delta_max'])

            # Geometry & schedule (small → load immediately)
            self.pos            = f['geometry/pos'][:]
            self.delta_schedule = f['schedule/delta_schedule'][:]

        # Only the first M entries are the forward sweep (δ_min → δ_max)
        self.deltas = self.delta_schedule[:self.M]

    # ── Iterate over snapshots ────────────────────────────────────────────────

    def iter_samples(self, start: int = 0, stop: int = None, step: int = 1):
        """
        Lazy iterator yielding (op_types, op_sites) one snapshot at a time.

        Parameters
        ----------
        start, stop, step : slice parameters (default: all samples)

        Yields
        ------
        op_types : (2M,) int8 ndarray
        op_sites : (2M,) int16 ndarray
        """
        stop = stop or self.n_samples
        with h5py.File(self.filepath, 'r') as f:
            ds_t = f['samples/op_types']
            ds_s = f['samples/op_sites']
            for i in range(start, stop, step):
                yield ds_t[i].astype(np.int32), ds_s[i].astype(np.int32)

    def load_samples(self, start: int = 0, stop: int = None):
        """
        Load a slice of samples into memory as two arrays.

        Returns
        -------
        op_types : (n, 2M) int32
        op_sites : (n, 2M) int32
        """
        stop = stop or self.n_samples
        with h5py.File(self.filepath, 'r') as f:
            op_types = f['samples/op_types'][start:stop].astype(np.int32)
            op_sites = f['samples/op_sites'][start:stop].astype(np.int32)
        return op_types, op_sites

    # ── Generic compute engine ────────────────────────────────────────────────

    def compute(self, observable_fn, start: int = 0, stop: int = None,
                n_bins: int = 50, n_jobs: int = 1):
        """
        Apply `observable_fn` to every snapshot and return binned statistics.

        Parameters
        ----------
        observable_fn : callable
            Signature: fn(op_types, op_sites, arc) → scalar or ndarray
            Receives one sample's operator arrays and this archive instance.
        start, stop   : sample range (default: all)
        n_bins        : number of bins for error estimation
        n_jobs        : number of parallel processes to use (default 1)

        Returns
        -------
        dict with keys 'mean', 'err', 'bins'
            mean : float or (M,) ndarray   – overall estimator
            err  : float or (M,) ndarray   – binned standard error
            bins : list of per-bin means
        """
        stop = stop or self.n_samples
        n_total = (stop - start)
        bin_size = max(1, n_total // n_bins)
        
        # When multiprocessing is enabled, we need to pass a lightweight
        # copy of the archive (without open file handles) to the workers
        arc_copy = copy(self)

        all_results = []
        if n_jobs > 1:
            chunk_size = max(1, n_total // n_jobs)
            chunks = []
            for i in range(start, stop, chunk_size):
                end = min(i + chunk_size, stop)
                chunks.append((i, end))

            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(_postprocess_worker_chunk, arc_copy, observable_fn, c[0], c[1]) for c in chunks]
                for future in concurrent.futures.as_completed(futures):
                    all_results.extend(future.result())
        else:
            all_results = _postprocess_worker_chunk(arc_copy, observable_fn, start, stop)

        bins = []
        for i in range(0, len(all_results), bin_size):
            chunk = all_results[i : i + bin_size]
            bins.append(np.mean(chunk, axis=0))

        bins_arr = np.array(bins)
        return {
            'mean': np.mean(bins_arr, axis=0),
            'err' : np.std(bins_arr,  axis=0) / max(1, np.sqrt(len(bins_arr))),
            'bins': bins_arr,
        }

    def __repr__(self):
        return (f"QAQMCArchive('{self.filepath.name}', "
                f"N={self.N}, M={self.M}, n_samples={self.n_samples})")


# ── Helper: rebuild spin state at slice p ────────────────────────────────────

def rebuild_state_at(op_types: np.ndarray, op_sites: np.ndarray,
                     N: int, p: int) -> np.ndarray:
    """
    Reconstruct the spin configuration at imaginary-time slice p
    by walking forward from |00...0⟩.

    Parameters
    ----------
    op_types, op_sites : one sample's operator arrays (length 2M)
    N                  : number of sites
    p                  : target slice index ∈ [0, 2M]

    Returns
    -------
    state : (N,) int32 array, values in {0, 1}
    """
    state = np.zeros(N, dtype=np.int32)
    for t in range(p):
        if op_types[t] == -1:          # off-diagonal (flip) operator
            state[op_sites[t]] ^= 1
    return state


# ── Built-in observable functions ─────────────────────────────────────────────
# Each has signature: fn(op_types, op_sites, arc) → value
# Pass them directly to arc.compute(...)

def obs_density_asym(op_types, op_sites, arc):
    """Rydberg density ⟨n⟩ at each asymmetric time slice p ∈ [0, M)."""
    M, N = arc.M, arc.N
    out = np.empty(M, dtype=np.float64)
    state = np.zeros(N, dtype=np.int32)
    for p in range(M):
        out[p] = state.sum() / N
        if op_types[p] == -1:
            state[op_sites[p]] ^= 1
    return out


def obs_mz_asym(op_types, op_sites, arc):
    """Staggered magnetization m_z at each asymmetric slice."""
    M, N = arc.M, arc.N
    out = np.empty(M, dtype=np.float64)
    state = np.zeros(N, dtype=np.int32)
    phases = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    for p in range(M):
        out[p] = np.dot(phases, state - 0.5) / N
        if op_types[p] == -1:
            state[op_sites[p]] ^= 1
    return out


def obs_density_sym(op_types, op_sites, arc):
    """Scalar Rydberg density at the symmetric midpoint slice p = M."""
    state = rebuild_state_at(op_types, op_sites, arc.N, arc.M)
    return float(state.sum() / arc.N)


def obs_mz_sym(op_types, op_sites, arc):
    """Staggered magnetization at the symmetric midpoint."""
    state = rebuild_state_at(op_types, op_sites, arc.N, arc.M)
    N = arc.N
    m = sum(((1 if i % 2 == 0 else -1) * (state[i] - 0.5)) for i in range(N))
    return float(m / N)


def obs_nn_corr_asym(i: int, j: int):
    """
    Factory: returns an observable function for ⟨n_i n_j⟩ along the sweep.
    Usage: arc.compute(obs_nn_corr_asym(0, 2))
    """
    def _fn(op_types, op_sites, arc):
        M, N = arc.M, arc.N
        out = np.empty(M, dtype=np.float64)
        state = np.zeros(N, dtype=np.int32)
        for p in range(M):
            out[p] = float(state[i] * state[j])
            if op_types[p] == -1:
                state[op_sites[p]] ^= 1
        return out
    return _fn


def obs_string_op_asym(i: int, j: int):
    """
    Factory: string operator ∏_{k=i}^{j} (1 - 2n_k) along the sweep.
    Equivalent to (-1)^{number of excitations between i and j}.
    """
    def _fn(op_types, op_sites, arc):
        M, N = arc.M, arc.N
        out = np.empty(M, dtype=np.float64)
        state = np.zeros(N, dtype=np.int32)
        for p in range(M):
            prod = 1.0
            for k in range(i, j + 1):
                prod *= (1 - 2 * state[k])
            out[p] = prod
            if op_types[p] == -1:
                state[op_sites[p]] ^= 1
        return out
    return _fn


def obs_loop_string_op(loop_sites):
    """
    Factory: closed-loop string operator along an arbitrary ordered list of
    site indices.

        W_loop = ∏_{k ∈ loop_sites} (1 - 2 n_k)

    This is the natural observable for detecting Z₂ topological order or
    Rydberg blockade order on a Ruby / Kagome lattice.  The sites do **not**
    need to form a geometrically straight path—pass any ordered list of site
    indices that trace your closed contour.

    Parameters
    ----------
    loop_sites : list[int]
        Ordered site indices forming the loop, e.g. a hexagonal plaquette on
        the Ruby lattice.  The product is over exactly these sites.

    Returns
    -------
    callable
        Observable function with signature fn(op_types, op_sites, arc) → (M,)

    Example — hexagonal plaquette on a 1×1 Ruby lattice (6 atoms, sites 0-5)
    -------------------------------------------------------------------------
        from src.lattices import generate_ruby_lattice
        pos = generate_ruby_lattice(nx=1, ny=1)
        # The six sites of the unit cell already form the elementary hexagon
        hex_loop = [0, 1, 2, 3, 4, 5]
        arc.compute(obs_loop_string_op(hex_loop))

    Example — custom plaquette by spatial proximity
    ------------------------------------------------
        import numpy as np
        center = pos.mean(axis=0)
        dists  = np.linalg.norm(pos - center, axis=1)
        hex_loop = list(np.argsort(dists)[:6])   # 6 nearest to centre
        arc.compute(obs_loop_string_op(hex_loop))
    """
    sites_arr = list(loop_sites)   # freeze at factory call time

    def _fn(op_types, op_sites, arc):
        M, N = arc.M, arc.N
        out = np.empty(M, dtype=np.float64)
        state = np.zeros(N, dtype=np.int32)
        for p in range(M):
            prod = 1.0
            for k in sites_arr:
                prod *= (1 - 2 * int(state[k]))
            out[p] = prod
            if op_types[p] == -1:
                state[op_sites[p]] ^= 1
        return out

    _fn.__doc__ = f"Loop string op over sites {sites_arr}"
    return _fn
