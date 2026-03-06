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
from numba import njit, prange


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
        """
        stop = stop or self.n_samples
        with h5py.File(self.filepath, 'r') as f:
            ds_t = f['samples/op_types']
            ds_s = f['samples/op_sites']
            for i in range(start, stop, step):
                yield ds_t[i].astype(np.int32), ds_s[i].astype(np.int32)

    def load_samples(self, start: int = 0, stop: int = None):
        """
        Load a slice of samples into memory as two contiguous arrays.

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

        Samples are batch-loaded into memory, then observable_fn receives
        (all_op_types, all_op_sites, arc) for vectorized/Numba processing.

        Parameters
        ----------
        observable_fn : callable
            Signature: fn(op_types_2d, op_sites_2d, arc) → (n_samples, ...) array
        start, stop   : sample range (default: all)
        n_bins        : number of bins for error estimation

        Returns
        -------
        dict with keys 'mean', 'err', 'bins'
        """
        stop = stop or self.n_samples
        n_total = stop - start

        # Batch load all samples at once (much faster than one-by-one HDF5)
        all_ot, all_os = self.load_samples(start, stop)

        # Call observable function on the full batch
        all_results = observable_fn(all_ot, all_os, self)

        # Binning for error estimation
        bin_size = max(1, n_total // n_bins)
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


# ── Numba-accelerated kernels ─────────────────────────────────────────────────

@njit(cache=True, parallel=True)
def _density_asym_kernel(op_types, op_sites, M, N):
    """Compute ⟨n⟩ at each forward-sweep slice for all samples (batch)."""
    n_samples = op_types.shape[0]
    out = np.empty((n_samples, M), dtype=np.float64)
    inv_N = 1.0 / N
    for s in prange(n_samples):
        state = np.zeros(N, dtype=np.int32)
        for p in range(M):
            total = 0
            for i in range(N):
                total += state[i]
            out[s, p] = total * inv_N
            if op_types[s, p] == -1:
                state[op_sites[s, p]] ^= 1
    return out


@njit(cache=True, parallel=True)
def _mz_asym_kernel(op_types, op_sites, M, N):
    """Staggered magnetization at each forward-sweep slice (batch)."""
    n_samples = op_types.shape[0]
    out = np.empty((n_samples, M), dtype=np.float64)
    inv_N = 1.0 / N
    for s in prange(n_samples):
        state = np.zeros(N, dtype=np.int32)
        for p in range(M):
            mz = 0.0
            for i in range(N):
                phase = 1.0 if (i % 2 == 0) else -1.0
                mz += phase * (state[i] - 0.5)
            out[s, p] = mz * inv_N
            if op_types[s, p] == -1:
                state[op_sites[s, p]] ^= 1
    return out


@njit(cache=True, parallel=True)
def _density_sym_kernel(op_types, op_sites, M, N):
    """Scalar density at symmetric midpoint p = M for all samples."""
    n_samples = op_types.shape[0]
    out = np.empty(n_samples, dtype=np.float64)
    inv_N = 1.0 / N
    for s in prange(n_samples):
        state = np.zeros(N, dtype=np.int32)
        for t in range(M):
            if op_types[s, t] == -1:
                state[op_sites[s, t]] ^= 1
        total = 0
        for i in range(N):
            total += state[i]
        out[s] = total * inv_N
    return out


@njit(cache=True, parallel=True)
def _mz_sym_kernel(op_types, op_sites, M, N):
    """Staggered magnetization at symmetric midpoint (batch)."""
    n_samples = op_types.shape[0]
    out = np.empty(n_samples, dtype=np.float64)
    inv_N = 1.0 / N
    for s in prange(n_samples):
        state = np.zeros(N, dtype=np.int32)
        for t in range(M):
            if op_types[s, t] == -1:
                state[op_sites[s, t]] ^= 1
        mz = 0.0
        for i in range(N):
            phase = 1.0 if (i % 2 == 0) else -1.0
            mz += phase * (state[i] - 0.5)
        out[s] = mz * inv_N
    return out


@njit(cache=True, parallel=True)
def _nn_corr_asym_kernel(op_types, op_sites, M, N, site_i, site_j):
    """⟨n_i n_j⟩ at each forward-sweep slice (batch)."""
    n_samples = op_types.shape[0]
    out = np.empty((n_samples, M), dtype=np.float64)
    for s in prange(n_samples):
        state = np.zeros(N, dtype=np.int32)
        for p in range(M):
            out[s, p] = float(state[site_i] * state[site_j])
            if op_types[s, p] == -1:
                state[op_sites[s, p]] ^= 1
    return out


@njit(cache=True, parallel=True)
def _string_op_asym_kernel(op_types, op_sites, M, N, site_i, site_j):
    """String operator ∏_{k=i}^{j} (1-2n_k) at each slice (batch)."""
    n_samples = op_types.shape[0]
    out = np.empty((n_samples, M), dtype=np.float64)
    for s in prange(n_samples):
        state = np.zeros(N, dtype=np.int32)
        for p in range(M):
            prod = 1.0
            for k in range(site_i, site_j + 1):
                prod *= (1 - 2 * state[k])
            out[s, p] = prod
            if op_types[s, p] == -1:
                state[op_sites[s, p]] ^= 1
    return out


@njit(cache=True, parallel=True)
def _loop_string_kernel(op_types, op_sites, M, N, loop_sites):
    """Loop string operator over arbitrary sites (batch)."""
    n_samples = op_types.shape[0]
    n_loop = len(loop_sites)
    out = np.empty((n_samples, M), dtype=np.float64)
    for s in prange(n_samples):
        state = np.zeros(N, dtype=np.int32)
        for p in range(M):
            prod = 1.0
            for k in range(n_loop):
                prod *= (1 - 2 * state[loop_sites[k]])
            out[s, p] = prod
            if op_types[s, p] == -1:
                state[op_sites[s, p]] ^= 1
    return out


@njit(cache=True, parallel=True)
def _rebuild_state_batch(op_types, op_sites, N, p_target):
    """Reconstruct spin state at slice p_target for all samples (batch)."""
    n_samples = op_types.shape[0]
    states = np.zeros((n_samples, N), dtype=np.int32)
    for s in prange(n_samples):
        for t in range(p_target):
            if op_types[s, t] == -1:
                states[s, op_sites[s, t]] ^= 1
    return states


# ── Helper: rebuild spin state at slice p ────────────────────────────────────

def rebuild_state_at(op_types: np.ndarray, op_sites: np.ndarray,
                     N: int, p: int) -> np.ndarray:
    """
    Reconstruct the spin configuration at imaginary-time slice p.
    Supports both single-sample (1D) and batch (2D) inputs.
    """
    if op_types.ndim == 1:
        state = np.zeros(N, dtype=np.int32)
        for t in range(p):
            if op_types[t] == -1:
                state[op_sites[t]] ^= 1
        return state
    else:
        return _rebuild_state_batch(op_types, op_sites, N, p)


# ── Built-in observable functions ─────────────────────────────────────────────
# Batch API: fn(all_op_types, all_op_sites, arc) → (n_samples, ...) array

def obs_density_asym(op_types, op_sites, arc):
    """Rydberg density ⟨n⟩ at each asymmetric time slice p ∈ [0, M)."""
    return _density_asym_kernel(op_types, op_sites, arc.M, arc.N)


def obs_mz_asym(op_types, op_sites, arc):
    """Staggered magnetization m_z at each asymmetric slice."""
    return _mz_asym_kernel(op_types, op_sites, arc.M, arc.N)


def obs_density_sym(op_types, op_sites, arc):
    """Scalar Rydberg density at the symmetric midpoint slice p = M."""
    return _density_sym_kernel(op_types, op_sites, arc.M, arc.N)


def obs_mz_sym(op_types, op_sites, arc):
    """Staggered magnetization at the symmetric midpoint."""
    return _mz_sym_kernel(op_types, op_sites, arc.M, arc.N)


def obs_nn_corr_asym(i: int, j: int):
    """
    Factory: returns an observable function for ⟨n_i n_j⟩ along the sweep.
    Usage: arc.compute(obs_nn_corr_asym(0, 2))
    """
    def _fn(op_types, op_sites, arc):
        return _nn_corr_asym_kernel(op_types, op_sites, arc.M, arc.N, i, j)
    return _fn


def obs_string_op_asym(i: int, j: int):
    """
    Factory: string operator ∏_{k=i}^{j} (1 - 2n_k) along the sweep.
    """
    def _fn(op_types, op_sites, arc):
        return _string_op_asym_kernel(op_types, op_sites, arc.M, arc.N, i, j)
    return _fn


def obs_loop_string_op(loop_sites):
    """
    Factory: closed-loop string operator along an arbitrary ordered list of
    site indices.

        W_loop = ∏_{k ∈ loop_sites} (1 - 2 n_k)

    Parameters
    ----------
    loop_sites : list[int]
        Ordered site indices forming the loop.

    Returns
    -------
    callable
        Observable function with signature fn(op_types, op_sites, arc) → (n, M)
    """
    sites_arr = np.array(loop_sites, dtype=np.int32)

    def _fn(op_types, op_sites, arc):
        return _loop_string_kernel(op_types, op_sites, arc.M, arc.N, sites_arr)

    _fn.__doc__ = f"Loop string op over sites {list(loop_sites)}"
    return _fn
