"""
MPI-parallel QAQMC: each rank runs an independent Markov chain.
Rank 0 gathers samples and writes a single HDF5 file.

Usage:
    mpirun -np 4 python -m src.qaqmc_mpi --config config.json
    mpirun -np 40 python -m src.qaqmc_mpi --N 384 --M 5000 ...

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

from src.qaqmc import QAQMC_Rydberg


def run_mpi(*, N, M, Omega=1.0, Rb=1.2, delta_min=0.0, delta_max=1.0,
            pos=None, epsilon=0.01, seed=42, n_equil=5000, n_samples=10000,
            filepath='data/qaqmc_mpi.h5', neighbor_cutoff=None,
            precompute=True, chunk_slices=0, omp_threads=0,
            compression='gzip', compression_opts=4,
            checkpoint_every=0, verbose=True):
    """
    MPI-parallel QAQMC simulation.

    Each rank runs an independent Markov chain with a different seed.
    Rank 0 creates the HDF5 file and gathers samples from all ranks.

    Parameters
    ----------
    N, M, Omega, Rb, delta_min, delta_max, pos, epsilon, seed:
        Same as QAQMC_Rydberg.__init__
    n_equil, n_samples:
        Per-rank equilibration and total samples (split across ranks)
    filepath:
        Output HDF5 path (written by rank 0)
    neighbor_cutoff, precompute, chunk_slices, omp_threads:
        Engine options passed to QAQMC_Rydberg
    compression, compression_opts:
        HDF5 compression settings
    checkpoint_every:
        Save checkpoint every N samples (per-rank, 0 = disabled).
        Currently only supported for rank 0's streaming path.
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

    # Compute write offsets for each rank
    counts = np.array([base + (1 if r < rem else 0) for r in range(n_ranks)])
    offsets = np.zeros(n_ranks, dtype=int)
    for r in range(1, n_ranks):
        offsets[r] = offsets[r - 1] + counts[r - 1]
    my_offset = offsets[rank]

    if verbose and rank == 0:
        print(f"[MPI] {n_ranks} ranks, {n_samples} total samples "
              f"({my_n_samples} per rank ± 1)")

    # Each rank gets a different seed
    rank_seed = seed + rank * 9973  # large prime to avoid correlation

    # Create the engine
    engine = QAQMC_Rydberg(
        N=N, M=M, Omega=Omega, Rb=Rb,
        delta_min=delta_min, delta_max=delta_max,
        pos=pos, epsilon=epsilon, seed=rank_seed,
        verbose=False, use_cpp=True, omp_threads=omp_threads,
        neighbor_cutoff=neighbor_cutoff, precompute=precompute,
        chunk_slices=chunk_slices,
    )

    M2 = engine.M_total

    # Equilibration
    comm.Barrier()
    t0 = time.perf_counter()

    if engine._cpp_engine is not None:
        try:
            engine._cpp_engine.run(n_equil, 0, None, 1)
        except TypeError:
            engine._cpp_engine.run(n_equil, 0)
    else:
        for _ in range(n_equil):
            engine.mc_step()

    t_equil = time.perf_counter() - t0
    comm.Barrier()

    if verbose and rank == 0:
        max_eq = comm.reduce(t_equil, op=MPI.MAX, root=0)
        print(f"[MPI] Equilibration done in {max_eq:.1f}s (slowest rank)")
    else:
        comm.reduce(t_equil, op=MPI.MAX, root=0)

    # Sampling
    t0 = time.perf_counter()

    if engine._cpp_engine is not None:
        try:
            my_types, my_sites = engine._cpp_engine.run(0, my_n_samples, None, 1)
        except TypeError:
            my_types, my_sites = engine._cpp_engine.run(0, my_n_samples)
    else:
        my_types = np.empty((my_n_samples, M2), dtype=np.int8)
        my_sites = np.empty((my_n_samples, M2), dtype=np.int32)
        for i in range(my_n_samples):
            engine.mc_step()
            my_types[i] = engine.op_types[:M2].astype(np.int8)
            my_sites[i] = engine.op_sites[:M2].astype(np.int32)

    t_sample = time.perf_counter() - t0
    comm.Barrier()

    if verbose and rank == 0:
        max_sa = comm.reduce(t_sample, op=MPI.MAX, root=0)
        print(f"[MPI] Sampling done in {max_sa:.1f}s (slowest rank)")
    else:
        comm.reduce(t_sample, op=MPI.MAX, root=0)

    # ── Rank 0 writes HDF5 ───────────────────────────────────────────────
    if rank == 0:
        kw = dict(compression=compression, compression_opts=compression_opts) \
             if compression == 'gzip' else dict(compression=compression) \
             if compression else {}

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        with h5py.File(filepath, 'w') as f:
            # metadata
            pg = f.create_group('params')
            pg.attrs['N'] = N
            pg.attrs['Omega'] = Omega
            pg.attrs['Rb'] = Rb
            pg.attrs['delta_min'] = delta_min
            pg.attrs['delta_max'] = delta_max
            pg.attrs['M'] = M
            pg.attrs['epsilon'] = epsilon
            pg.attrs['seed'] = seed
            pg.attrs['n_equil'] = n_equil
            pg.attrs['n_samples'] = n_samples
            pg.attrs['n_ranks'] = n_ranks
            pg.attrs['timestamp'] = datetime.datetime.utcnow().isoformat()

            if neighbor_cutoff is not None:
                pg.attrs['neighbor_cutoff'] = neighbor_cutoff

            # geometry
            gg = f.create_group('geometry')
            pos_stored = pos if pos is not None \
                         else np.arange(N).reshape(-1, 1).astype(np.float64)
            gg.create_dataset('pos', data=pos_stored.astype(np.float64))

            # schedule
            sg = f.create_group('schedule')
            delta_sched = np.array(engine._cpp_engine.delta_schedule
                                   if engine._cpp_engine else
                                   np.zeros(M2), dtype=np.float64)
            sg.create_dataset('delta_schedule', data=delta_sched)

            # sample datasets
            smg = f.create_group('samples')
            ds_types = smg.create_dataset('op_types', shape=(n_samples, M2),
                                          dtype='int8', **kw)
            ds_sites = smg.create_dataset('op_sites', shape=(n_samples, M2),
                                          dtype='int32', **kw)

            # Write rank 0's data
            ds_types[my_offset:my_offset + my_n_samples] = my_types
            ds_sites[my_offset:my_offset + my_n_samples] = my_sites

            # Receive and write data from other ranks
            for src in range(1, n_ranks):
                src_count = int(counts[src])
                src_offset = int(offsets[src])
                buf_t = np.empty((src_count, M2), dtype=np.int8)
                buf_s = np.empty((src_count, M2), dtype=np.int32)
                comm.Recv(buf_t, source=src, tag=100 + src)
                comm.Recv(buf_s, source=src, tag=200 + src)
                ds_types[src_offset:src_offset + src_count] = buf_t
                ds_sites[src_offset:src_offset + src_count] = buf_s

        if verbose:
            print(f"[MPI] Saved {n_samples} samples → {filepath}")
    else:
        # Non-root ranks send their data to rank 0
        comm.Send(np.ascontiguousarray(my_types), dest=0, tag=100 + rank)
        comm.Send(np.ascontiguousarray(my_sites), dest=0, tag=200 + rank)

    comm.Barrier()
    return filepath


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MPI-parallel QAQMC simulation')
    parser.add_argument('--config', type=str, help='JSON config file')
    parser.add_argument('--N', type=int, default=6)
    parser.add_argument('--M', type=int, default=1280)
    parser.add_argument('--Omega', type=float, default=1.0)
    parser.add_argument('--Rb', type=float, default=2.4)
    parser.add_argument('--delta_min', type=float, default=0.0)
    parser.add_argument('--delta_max', type=float, default=8.0)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_equil', type=int, default=5000)
    parser.add_argument('--n_samples', type=int, default=30000)
    parser.add_argument('--neighbor_cutoff', type=int, default=None)
    parser.add_argument('--precompute', action='store_true', default=True)
    parser.add_argument('--chunk_slices', type=int, default=0)
    parser.add_argument('--omp_threads', type=int, default=1)
    parser.add_argument('--filepath', type=str, default='data/qaqmc_mpi.h5')
    parser.add_argument('--lattice', type=str, default='kagome_bond',
                        help='Lattice type: kagome_bond')
    parser.add_argument('--nx', type=int, default=1)
    parser.add_argument('--ny', type=int, default=1)
    parser.add_argument('--a', type=float, default=4.0, help='Lattice constant')
    args = parser.parse_args()

    # Load config file if provided (overrides CLI args)
    config = vars(args)
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    # Generate lattice
    pos = None
    if config.get('lattice') == 'kagome_bond':
        from src.lattices import generate_kagome_bond_lattice
        pos = generate_kagome_bond_lattice(
            nx=config.get('nx', 1),
            ny=config.get('ny', 1),
            a=config.get('a', 4.0)
        )
        config['N'] = len(pos)

    run_mpi(
        N=config['N'], M=config['M'],
        Omega=config['Omega'], Rb=config['Rb'],
        delta_min=config['delta_min'], delta_max=config['delta_max'],
        pos=pos, epsilon=config['epsilon'], seed=config['seed'],
        n_equil=config['n_equil'], n_samples=config['n_samples'],
        filepath=config['filepath'],
        neighbor_cutoff=config.get('neighbor_cutoff'),
        precompute=config.get('precompute', True),
        chunk_slices=config.get('chunk_slices', 0),
        omp_threads=config.get('omp_threads', 1),
    )


if __name__ == '__main__':
    main()
