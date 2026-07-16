"""
Quasi-Adiabatic Quantum Monte Carlo (QAQMC) Python wrapper.

Thin API layer over the C++ engine (qaqmc_cpp.QAQMCEngine).  The historical
pure-Python/Numba fallback engine was removed in 2026-07: it had no callers,
had not tracked the C++ engine's physics (periodic boundaries, seam, compact
storage), and silently degrading to it on a missing .so produced wrong-physics
data instead of a loud failure.  A missing extension now raises ImportError —
build it first (see README 部署步驟).  The old implementation remains in git
history (src/engines/qaqmc_updates.py before this commit).
"""
import numpy as np

try:
    from tqdm import trange
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import h5py
import datetime
import os
import time

try:
    import shutil
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


def _qaqmc_delta_values(p_indices, M, delta_min, delta_max):
    """Vectorized delta(p) with the same two-ramp expression as the C++ core."""
    p = np.asarray(p_indices, dtype=np.int64)
    delta = np.empty(p.shape, dtype=np.float64)
    forward = p < M
    span = delta_max - delta_min
    delta[forward] = delta_min + span * (
        p[forward].astype(np.float64) / M)
    delta[~forward] = delta_max - span * (
        (p[~forward] - M).astype(np.float64) / M)
    return delta


def _write_qaqmc_delta_schedule(group, M, delta_min, delta_max,
                                 name="delta_schedule", chunk_slots=1 << 20):
    """Write the backward-compatible 2M schedule without an O(M) temporary.

    The on-disk dataset name, shape, and dtype remain unchanged.  At most one
    ``chunk_slots`` float64 work buffer is live while HDF5 is populated.
    """
    total = 2 * int(M)
    if total <= 0:
        return group.create_dataset(name, shape=(0,), dtype=np.float64)
    chunk_slots = max(1, min(int(chunk_slots), total))
    dataset = group.create_dataset(
        name, shape=(total,), dtype=np.float64, chunks=(chunk_slots,))
    for start in range(0, total, chunk_slots):
        stop = min(start + chunk_slots, total)
        p = np.arange(start, stop, dtype=np.int64)
        dataset[start:stop] = _qaqmc_delta_values(
            p, M, delta_min, delta_max)
    return dataset


class QAQMC_Rydberg:
    def __init__(self, N: int, M: int, Omega: float = 1.0,
                 Rb: float = 1.2, delta_min: float = 0.0, delta_max: float = 1.0,
                 pos: np.ndarray = None, epsilon: float = 0.01, seed: int = 42,
                 verbose: bool = True, n_jobs: int = 1, backend: str = "process",
                 use_cpp: bool = True, omp_threads: int = 0,
                 neighbor_cutoff: int = None, delta_groups: int = 600,
                 box_vectors: np.ndarray = None, model_data=None,
                 bond_event_storage: str = "packed64"):
        # n_jobs/backend are accepted for backward compatibility but unused:
        # parallelism happens at the MPI-rank / shared-model-batch level.
        if not use_cpp:
            raise ValueError(
                "use_cpp=False (the Python/Numba fallback engine) was removed "
                "2026-07 — the C++ engine is the only implementation.")
        _require_cpp()

        self.init_kwargs = {
            'N': N, 'Omega': Omega, 'delta_min': delta_min, 'delta_max': delta_max,
            'Rb': Rb, 'M': M, 'epsilon': epsilon, 'seed': seed, 'pos': pos,
            'verbose': False, 'use_cpp': True, 'omp_threads': omp_threads,
            'neighbor_cutoff': neighbor_cutoff, 'delta_groups': delta_groups,
            'box_vectors': box_vectors, 'bond_event_storage': bond_event_storage,
        }

        # Set OpenMP threads environment variable before C++ engine usage
        if omp_threads > 0:
            os.environ["OMP_NUM_THREADS"] = str(omp_threads)
        self.N = N
        self.Omega = Omega
        self.Rb = Rb
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.verbose = verbose
        self.omp_threads = omp_threads

        self.M = M
        self.M_total = 2 * M

        self.pos = pos
        if self.pos is None:
            self.pos = np.arange(N).reshape(-1, 1).astype(np.float64)

        nc = neighbor_cutoff if neighbor_cutoff is not None else -1
        box = (np.ascontiguousarray(box_vectors, dtype=np.float64)
               if box_vectors is not None else None)
        pos_arr = np.ascontiguousarray(self.pos, dtype=np.float64)
        if model_data is None:
            self._cpp_engine = qaqmc_cpp.QAQMCEngine(
                N, Omega, delta_min, delta_max, Rb, M, epsilon, seed, pos_arr,
                neighbor_cutoff=nc, delta_groups=delta_groups,
                box_vectors=box,
            )
        else:
            if int(model_data.N) != int(N) or int(model_data.M) != int(M):
                raise ValueError(
                    "model_data N/M do not match the requested chain")
            self._cpp_engine = qaqmc_cpp.QAQMCEngine(model_data, seed)
        self._cpp_engine.bond_event_storage = bond_event_storage
        # Bond geometry is small.  Do NOT retain full int32 mirrors of the
        # C++ operator string: that would recreate 8L bytes per rank and
        # defeat the compact engine.  op_types/op_sites below are lazy
        # compatibility properties that export only when requested.
        self.bond_sites = np.array(self._cpp_engine.bond_sites, dtype=np.int32)
        if verbose:
            n_bonds = len(self.bond_sites)
            print(f"[QAQMC] Using C++ backend (N={N}, M={M}, bonds={n_bonds}, "
                  f"delta_groups={delta_groups})")

    @property
    def op_types(self):
        return np.asarray(self._cpp_engine.op_types, dtype=np.int32)

    @property
    def op_sites(self):
        return np.asarray(self._cpp_engine.op_sites, dtype=np.int32)

    def mc_step(self):
        self._cpp_engine.mc_step()

    def run_and_save(self, filepath: str, n_equil: int = 5000,
                     n_samples: int = 10000, verbose: bool = True,
                     compression: str = 'gzip', compression_opts: int = 4,
                     chunk_samples: int = 1024,
                     checkpoint_every: int = 0):
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
            op_sites       (n_samples, 2M) int32 – site/bond index per slice

        Parameters
        ----------
        filepath       : Output path, e.g. 'data/run_N8_M512.h5'
        n_equil        : Equilibration steps (discarded, not saved)
        n_samples      : Number of MCMC snapshots to save
        verbose        : Show tqdm progress bar
        compression    : HDF5 compression filter ('gzip', 'lzf', or None)
        compression_opts: gzip compression level (1=fast … 9=small)
        chunk_samples  : Write chunk size for streaming mode
        checkpoint_every: Save checkpoint every N samples (0 = disabled)
        """
        M2 = self.M_total  # 2M
        t0_overall = time.perf_counter()

        kw = dict(compression=compression, compression_opts=compression_opts) \
             if compression == 'gzip' else dict(compression=compression) \
             if compression else {}

        kw_i8  = dict(**kw, dtype='int8')
        kw_i32 = dict(**kw, dtype='int32')

        # ── Check for existing checkpoint ─────────────────────────────────
        resume_from = 0
        resume_equil_done = False

        if checkpoint_every > 0 and os.path.exists(filepath):
            try:
                with h5py.File(filepath, 'r') as f:
                    if 'checkpoint' in f:
                        ckpt = f['checkpoint']
                        resume_from = int(ckpt.attrs['n_samples_done'])
                        n_equil_done = int(ckpt.attrs['n_equil_done'])
                        resume_equil_done = (n_equil_done >= n_equil)
                        rng_state_restore = ckpt.attrs['rng_state']
                        ckpt_types = ckpt['op_types'][:]
                        ckpt_sites = ckpt['op_sites'][:]
                        self._cpp_engine.set_op_string(
                            ckpt_types.astype(np.int32),
                            ckpt_sites.astype(np.int32))
                        self._cpp_engine.set_rng_state(rng_state_restore)
                        if verbose:
                            print(f"[Checkpoint] Resuming from sample {resume_from}/{n_samples}")
            except Exception as e:
                if verbose:
                    print(f"[Checkpoint] Failed to load checkpoint: {e}, starting fresh")
                resume_from = 0
                resume_equil_done = False

        # ── Open/create HDF5 file ─────────────────────────────────────────
        file_mode = 'r+' if (resume_from > 0 and os.path.exists(filepath)) else 'w'
        with h5py.File(filepath, file_mode) as f:

            if file_mode == 'w':
                # ── metadata ──────────────────────────────────────────────
                pg = f.create_group('params')
                for k, v in self.init_kwargs.items():
                    if k != 'pos' and v is not None:
                        pg.attrs[k] = v
                pg.attrs['n_equil']        = n_equil
                pg.attrs['n_samples']       = n_samples
                pg.attrs['timestamp']       = datetime.datetime.utcnow().isoformat()
                pg.attrs['equil_time_s']    = 0.0

                # ── geometry ──────────────────────────────────────────────
                gg = f.create_group('geometry')
                pos_stored = self.pos if self.pos is not None \
                             else np.arange(self.N).reshape(-1, 1).astype(np.float64)
                gg.create_dataset('pos', data=pos_stored.astype(np.float64))

                # ── δ schedule ────────────────────────────────────────────
                sg = f.create_group('schedule')
                _write_qaqmc_delta_schedule(
                    sg, self.M, self.delta_min, self.delta_max)

                # ── sample datasets ───────────────────────────────────────
                smg = f.create_group('samples')
                ds_types = smg.create_dataset('op_types', shape=(n_samples, M2), **kw_i8)
                ds_sites = smg.create_dataset('op_sites', shape=(n_samples, M2), **kw_i32)
            else:
                pg = f['params']
                ds_types = f['samples/op_types']
                ds_sites = f['samples/op_sites']

            # Streaming mode: avoid holding all samples in RAM.
            chunk_samples = max(1, int(chunk_samples))
            t_equil = 0.0
            use_tqdm = HAS_TQDM and verbose

            # Equilibration (skip if resuming from checkpoint)
            if not resume_equil_done:
                t0_eq = time.perf_counter()
                if use_tqdm:
                    eq_bar = trange(n_equil, desc="Equil (W0)", leave=False)
                    last_eq = 0

                    def _eq_cb(done, total, phase):
                        nonlocal last_eq
                        if phase == "equil":
                            delta = int(done) - last_eq
                            if delta > 0:
                                eq_bar.update(delta)
                                last_eq = int(done)

                    try:
                        self._cpp_engine.run(n_equil, 0, _eq_cb, max(1, n_equil // 200))
                    finally:
                        eq_bar.close()
                else:
                    self._cpp_engine.run(n_equil, 0, None, 1)
                t_equil = time.perf_counter() - t0_eq

            # Sampling in chunks (resume-aware)
            samp_bar = trange(n_samples, desc="Samp  (W0)", leave=False, initial=resume_from) if use_tqdm else None
            written = resume_from
            t0_sa = time.perf_counter()
            while written < n_samples:
                cur = min(chunk_samples, n_samples - written)
                t_arr, s_arr = self._cpp_engine.run(0, cur, None, 1)
                ds_types[written:written + cur] = t_arr
                ds_sites[written:written + cur] = s_arr
                written += cur
                if samp_bar is not None:
                    samp_bar.update(cur)

                # Save checkpoint periodically
                if checkpoint_every > 0 and written < n_samples and written % checkpoint_every < cur:
                    if 'checkpoint' in f:
                        del f['checkpoint']
                    cg = f.create_group('checkpoint')
                    cg.create_dataset('op_types', data=np.array(self._cpp_engine.op_types, dtype=np.int32))
                    cg.create_dataset('op_sites', data=np.array(self._cpp_engine.op_sites, dtype=np.int32))
                    cg.attrs['rng_state'] = self._cpp_engine.get_rng_state()
                    cg.attrs['n_samples_done'] = written
                    cg.attrs['n_equil_done'] = n_equil
                    f.flush()

            if samp_bar is not None:
                samp_bar.close()
            t_sample = time.perf_counter() - t0_sa

            pg.attrs['equil_time_s'] = t_equil
            pg.attrs['sample_time_s'] = t_sample
            pg.attrs['total_time_s']  = t_equil + t_sample

            # Clean finish: remove checkpoint
            if 'checkpoint' in f:
                del f['checkpoint']

        if verbose:
            total = time.perf_counter() - t0_overall
            print(f"Saved {n_samples} samples → {filepath}  "
                  f"(equil {t_equil:.1f}s + sample {t_sample:.1f}s, overall {total:.1f}s)")
