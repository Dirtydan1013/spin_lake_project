"""
MPI-parallel SSE: each rank runs an independent Markov chain.
Rank 0 gathers samples and writes a single HDF5 file.

Usage:
    mpirun -np 4 python -m src.mpi.sse_mpi --N 64 --beta 10 --delta 2.0 ...

Requires: mpi4py, h5py, numpy
"""

import numpy as np
import h5py
import time
import datetime
import argparse
import json
import os

try:
    from mpi4py import MPI
except ImportError:
    raise ImportError("mpi4py is required for MPI mode. Install with: pip install mpi4py")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from src.engines.sse import SSE_Rydberg


def _make_tqdm_callback(total, desc):
    """Create a progress bar (rank 0 only)."""
    if not HAS_TQDM:
        return None
    return tqdm(total=total, desc=desc, unit='step', leave=True)


def run_mpi(*, N, Omega=1.0, delta=0.0, Rb=1.4, beta=10.0,
            pos=None, epsilon=0.01, seed=42,
            n_equil=5000, n_samples=50000,
            filepath='data/sse_mpi.h5', neighbor_cutoff=None,
            compression='gzip', compression_opts=4,
            verbose=True):
    """
    MPI-parallel SSE simulation.

    Each rank runs an independent Markov chain with a different seed.
    Rank 0 gathers scalar observable arrays and writes a single HDF5 file.

    Parameters
    ----------
    N, Omega, delta, Rb, beta, pos, epsilon, seed:
        Same as SSE_Rydberg.__init__
    n_equil, n_samples:
        Per-rank equilibration and total samples (split across ranks)
    filepath:
        Output HDF5 path (written by rank 0)
    neighbor_cutoff:
        Keep only first N neighbor shells (-1 = all bonds)
    compression, compression_opts:
        HDF5 compression settings
    verbose:
        Print progress info (only rank 0 prints)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # Split total samples across ranks
    base = n_samples // n_ranks
    rem = n_samples % n_ranks
    my_n_samples = base + (1 if rank < rem else 0)

    # Compute write offsets
    counts = np.array([base + (1 if r < rem else 0) for r in range(n_ranks)])
    offsets = np.zeros(n_ranks, dtype=int)
    for r in range(1, n_ranks):
        offsets[r] = offsets[r - 1] + counts[r - 1]
    my_offset = offsets[rank]

    if verbose and rank == 0:
        print(f"[MPI-SSE] {n_ranks} ranks, {n_samples} total samples "
              f"({my_n_samples} per rank +/- 1)")

    # Each rank gets a different seed
    rank_seed = seed + rank * 9973

    # Create engine
    engine = SSE_Rydberg(
        N=N, Omega=Omega, delta=delta, Rb=Rb,
        beta=beta, epsilon=epsilon, seed=rank_seed,
        pos=pos, use_cpp=True, verbose=False,
        neighbor_cutoff=neighbor_cutoff,
    )

    # Equilibration
    comm.Barrier()
    t0 = time.perf_counter()
    bar = _make_tqdm_callback(n_equil, "[MPI-SSE] Equil (rank0)") if (verbose and rank == 0) else None
    for i in range(n_equil):
        engine.mc_step()
        if bar is not None and (i + 1) % 1000 == 0:
            bar.update(1000)
    if bar is not None:
        bar.update(n_equil - bar.n)
        bar.close()
    t_equil = time.perf_counter() - t0
    comm.Barrier()

    if verbose and rank == 0:
        max_eq = comm.reduce(t_equil, op=MPI.MAX, root=0)
        print(f"[MPI-SSE] Equilibration done in {max_eq:.1f}s (slowest rank)")
    else:
        comm.reduce(t_equil, op=MPI.MAX, root=0)

    # Sampling
    t0 = time.perf_counter()
    if engine._cpp_engine is not None:
        raw = engine._cpp_engine.run(
            n_equil=0, n_samples=my_n_samples,
            progress_callback=None, progress_every=1
        )
        my_e = np.asarray(raw['energies'],  dtype=np.float64)
        my_d = np.asarray(raw['densities'], dtype=np.float64)
        my_m = np.asarray(raw['mz'],        dtype=np.float64)
        my_n = np.asarray(raw['n_ops'],     dtype=np.int32)
        M_final = int(engine._cpp_engine.M)
    else:
        my_e = np.empty(my_n_samples, dtype=np.float64)
        my_d = np.empty(my_n_samples, dtype=np.float64)
        my_m = np.empty(my_n_samples, dtype=np.float64)
        my_n = np.empty(my_n_samples, dtype=np.int32)
        bar = _make_tqdm_callback(my_n_samples, "[MPI-SSE] Samp (rank0)") if (verbose and rank == 0) else None
        for i in range(my_n_samples):
            engine.mc_step()
            obs = engine.measure_observables()
            my_e[i] = obs['energy']
            my_d[i] = obs['density']
            my_m[i] = obs['m_z']
            my_n[i] = engine.n_ops
            if bar is not None and (i + 1) % 1000 == 0:
                bar.update(1000)
        if bar is not None:
            bar.update(my_n_samples - bar.n)
            bar.close()
        M_final = engine.M

    t_sample = time.perf_counter() - t0
    comm.Barrier()

    if verbose and rank == 0:
        max_sa = comm.reduce(t_sample, op=MPI.MAX, root=0)
        print(f"[MPI-SSE] Sampling done in {max_sa:.1f}s (slowest rank)")
    else:
        comm.reduce(t_sample, op=MPI.MAX, root=0)

    # ── Rank 0 writes HDF5 ───────────────────────────────────────────────
    if rank == 0:
        kw = dict(compression=compression, compression_opts=compression_opts) \
             if compression == 'gzip' else dict(compression=compression) \
             if compression else {}

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        with h5py.File(filepath, 'w') as f:
            pg = f.create_group('params')
            pg.attrs['N']         = N
            pg.attrs['Omega']     = Omega
            pg.attrs['delta']     = delta
            pg.attrs['Rb']        = Rb
            pg.attrs['beta']      = beta
            pg.attrs['epsilon']   = epsilon
            pg.attrs['seed']      = seed
            pg.attrs['n_equil']   = n_equil
            pg.attrs['n_samples'] = n_samples
            pg.attrs['n_ranks']   = n_ranks
            pg.attrs['M_final']   = M_final
            pg.attrs['backend']   = 'cpp' if engine._cpp_engine is not None else 'python'
            pg.attrs['timestamp'] = datetime.datetime.utcnow().isoformat()
            if neighbor_cutoff is not None:
                pg.attrs['neighbor_cutoff'] = neighbor_cutoff

            gg = f.create_group('geometry')
            pos_stored = pos if pos is not None \
                         else np.arange(N).reshape(-1, 1).astype(np.float64)
            gg.create_dataset('pos', data=pos_stored.astype(np.float64))

            sg = f.create_group('samples')
            ds_e = sg.create_dataset('energies',  shape=(n_samples,), dtype='float64', **kw)
            ds_d = sg.create_dataset('densities', shape=(n_samples,), dtype='float64', **kw)
            ds_m = sg.create_dataset('mz',        shape=(n_samples,), dtype='float64', **kw)
            ds_n = sg.create_dataset('n_ops',     shape=(n_samples,), dtype='int32',   **kw)

            # Write rank 0's data
            ds_e[my_offset:my_offset + my_n_samples] = my_e
            ds_d[my_offset:my_offset + my_n_samples] = my_d
            ds_m[my_offset:my_offset + my_n_samples] = my_m
            ds_n[my_offset:my_offset + my_n_samples] = my_n

            # Receive and write data from other ranks
            for src in range(1, n_ranks):
                src_count = int(counts[src])
                src_offset = int(offsets[src])
                buf_e = np.empty(src_count, dtype=np.float64)
                buf_d = np.empty(src_count, dtype=np.float64)
                buf_m = np.empty(src_count, dtype=np.float64)
                buf_n = np.empty(src_count, dtype=np.int32)
                comm.Recv(buf_e, source=src, tag=100 + src)
                comm.Recv(buf_d, source=src, tag=200 + src)
                comm.Recv(buf_m, source=src, tag=300 + src)
                comm.Recv(buf_n, source=src, tag=400 + src)
                ds_e[src_offset:src_offset + src_count] = buf_e
                ds_d[src_offset:src_offset + src_count] = buf_d
                ds_m[src_offset:src_offset + src_count] = buf_m
                ds_n[src_offset:src_offset + src_count] = buf_n

        if verbose:
            print(f"[MPI-SSE] Saved {n_samples} samples -> {filepath}")
    else:
        # Non-root ranks send their data to rank 0
        comm.Send(np.ascontiguousarray(my_e), dest=0, tag=100 + rank)
        comm.Send(np.ascontiguousarray(my_d), dest=0, tag=200 + rank)
        comm.Send(np.ascontiguousarray(my_m), dest=0, tag=300 + rank)
        comm.Send(np.ascontiguousarray(my_n), dest=0, tag=400 + rank)

    comm.Barrier()
    return filepath


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MPI-parallel SSE simulation')
    parser.add_argument('--config', type=str, help='JSON config file')
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--Omega', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=0.0)
    parser.add_argument('--Rb', type=float, default=1.4)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_equil', type=int, default=5000)
    parser.add_argument('--n_samples', type=int, default=50000)
    parser.add_argument('--neighbor_cutoff', type=int, default=None)
    parser.add_argument('--filepath', type=str, default='data/sse_mpi.h5')
    parser.add_argument('--lattice', type=str, default='1d_chain',
                        help='Lattice type: 1d_chain, kagome_bond')
    parser.add_argument('--a', type=float, default=1.0, help='Lattice constant')
    parser.add_argument('--nx', type=int, default=1)
    parser.add_argument('--ny', type=int, default=1)
    args = parser.parse_args()

    config = vars(args)
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    # Generate lattice
    pos = None
    lattice = config.get('lattice', '1d_chain')
    if lattice == '1d_chain':
        from src.rydberg.lattices import generate_1d_chain
        pos = generate_1d_chain(config['N'], config.get('a', 1.0))
    elif lattice == 'kagome_bond':
        from src.rydberg.lattices import generate_kagome_bond_lattice
        pos = generate_kagome_bond_lattice(
            nx=config.get('nx', 1), ny=config.get('ny', 1),
            a=config.get('a', 4.0))
        config['N'] = len(pos)

    run_mpi(
        N=config['N'], Omega=config['Omega'], delta=config['delta'],
        Rb=config['Rb'], beta=config['beta'],
        pos=pos, epsilon=config['epsilon'], seed=config['seed'],
        n_equil=config['n_equil'], n_samples=config['n_samples'],
        filepath=config['filepath'],
        neighbor_cutoff=config.get('neighbor_cutoff'),
    )


if __name__ == '__main__':
    main()
