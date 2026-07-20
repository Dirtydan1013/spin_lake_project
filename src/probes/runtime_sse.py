"""
Quick runtime probe for the finite-temperature SSE MPI workflow.

Times engine init + a handful of mc_steps at the PRODUCTION geometry (after M
has grown to its steady state) and extrapolates to the full equil + sample
budget.  Mirrors run_kagome_sse.sh defaults (triangle lattice) so the estimate
is realistic.  Called by scripts/scripts/probe_sse_runtime.sh.

Typical usage
-------------
mpiexec -n 24 python -u "scripts/python script/probe_runtime_sse.py" \
    --lattice kagome_bond_triangle --nx 6 --ny 6 --a 4.0 \
    --beta 16.0 --delta 3.0 --Rb 2.4 --neighbor-cutoff -1 \
    --probe-warmup 300 --probe-steps 200 \
    --target-equil 20000 --target-samples 2000000
"""

import argparse
import os
import time

import numpy as np
from mpi4py import MPI

from src.engines.sse import SSE_Rydberg
from src.probes import costreport, memreport


def _make_pos(lattice, N, nx, ny, a):
    if lattice == "kagome_bond_triangle":
        from src.rydberg.lattices import generate_kagome_bond_triangle_lattice
        return np.ascontiguousarray(
            generate_kagome_bond_triangle_lattice(nx, ny, a), dtype=np.float64)
    if lattice == "kagome_bond":
        from src.rydberg.lattices import generate_kagome_bond_lattice
        return np.ascontiguousarray(
            generate_kagome_bond_lattice(nx, ny, a), dtype=np.float64)
    from src.rydberg.lattices import generate_1d_chain
    return np.ascontiguousarray(generate_1d_chain(N, a), dtype=np.float64)


def main():
    p = argparse.ArgumentParser(description="SSE runtime probe")
    p.add_argument("--lattice", type=str, default="kagome_bond_triangle",
                   choices=["1d_chain", "kagome_bond", "kagome_bond_triangle"])
    p.add_argument("--N", type=int, default=64, help="(1d_chain) site count")
    p.add_argument("--nx", type=int, default=6)
    p.add_argument("--ny", type=int, default=6)
    p.add_argument("--a", type=float, default=4.0)
    p.add_argument("--Omega", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=3.0)
    p.add_argument("--Rb", type=float, default=2.4)
    p.add_argument("--beta", type=float, default=16.0)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--neighbor-cutoff", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe-warmup", type=int, default=300,
                   help="mc_steps to grow M to its steady state (untimed)")
    p.add_argument("--probe-steps", type=int, default=200,
                   help="mc_steps to time")
    p.add_argument("--target-equil", type=int, default=20000)
    p.add_argument("--target-samples", type=int, default=2000000,
                   help="total samples across ranks")
    p.add_argument("--target-ranks", type=int, default=None,
                   help="production rank count for the node-memory estimate "
                        "(default: $TARGET_RANKS/$SLURM_NTASKS/$NTASKS, else probe ranks)")
    p.add_argument("--node-mem-gb", type=float, default=240.0,
                   help="node memory budget for the fit verdict")
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    pos = _make_pos(args.lattice, args.N, args.nx, args.ny, args.a)
    N = len(pos)
    ncut = None if args.neighbor_cutoff < 0 else args.neighbor_cutoff

    t0 = time.perf_counter()
    eng = SSE_Rydberg(N=N, Omega=args.Omega, delta=args.delta, Rb=args.Rb,
                      beta=args.beta, epsilon=args.epsilon, seed=args.seed + rank,
                      pos=pos, use_cpp=True, verbose=False, neighbor_cutoff=ncut)
    cpp = eng._cpp_engine
    if cpp is None:
        raise RuntimeError("probe requires the C++ backend")
    t_init = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(args.probe_warmup):
        cpp.mc_step()
    t_warm = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(args.probe_steps):
        cpp.mc_step()
    t_step = (time.perf_counter() - t0) / max(args.probe_steps, 1)

    # Slowest rank drives the wall time.
    t_init_max = comm.reduce(t_init, op=MPI.MAX, root=0)
    t_step_max = comm.reduce(t_step, op=MPI.MAX, root=0)
    M_max = comm.reduce(int(cpp.M), op=MPI.MAX, root=0)

    if rank == 0:
        per_rank = -(-int(args.target_samples) // n_ranks)   # ceil
        total_steps = int(args.target_equil) + per_rank
        est = total_steps * t_step_max
        print(f"[probe-SSE] lattice={args.lattice} N={N}, ranks={n_ranks}, "
              f"beta={args.beta}, delta={args.delta}")
        print(f"[probe-SSE] engine init (slowest): {t_init_max:.2f}s; "
              f"steady-state M(max over ranks)={M_max}")
        print(f"[probe-SSE] mc_step (slowest rank): {1e3 * t_step_max:.2f} ms/step "
              f"(warmup {args.probe_warmup} steps took {t_warm:.1f}s)")
        print(f"[probe-SSE] per-rank budget: equil={args.target_equil} + "
              f"samples/rank={per_rank} = {total_steps} steps")
        print(f"[probe-SSE] ESTIMATED wall time (slowest rank): "
              f"{est:.0f}s = {est / 3600:.2f}h")
        costreport.report(est, n_ranks,
                          int(os.environ.get("OMP_NUM_THREADS", "1")))

    core = memreport.sse_engine_core_mib(cpp)
    memreport.report(
        "probe-SSE", comm,
        engine_core_mib=core,
        core_label=f"engine core (analytic, M={int(cpp.M)}, n_ops={int(cpp.n_ops)})",
        notes=(f"adjust_M realloc transient ≤ +{int(cpp.M) * 8 / 2**20:.0f} MiB/rank "
               "(old+new op arrays coexist briefly)",),
        target_ranks=args.target_ranks, node_mem_gb=args.node_mem_gb)


if __name__ == "__main__":
    main()
