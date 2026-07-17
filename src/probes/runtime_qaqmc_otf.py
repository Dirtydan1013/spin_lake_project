"""
Quick runtime probe for QAQMC on-the-fly MPI workflow.

Purpose
-------
Run only a few equilibration/sampling steps with your target problem size,
report per-stage timing (avg/max across ranks), and estimate full-run time.

Typical usage
-------------
mpiexec -n 16 python -u "scripts/python script/probe_runtime_qaqmc_otf.py" \
  --nx 6 --ny 6 --M 2760000 --delta-groups 500 \
  --delta-min -2.0 --delta-max 6.0 \
  --Rb 2.4 --neighbor-cutoff 3 \
  --probe-equil 3 --probe-samples 16 \
  --target-equil 0 --target-samples 30000 \
  --omp-threads 8
"""

import argparse
import os
import time

import numpy as np
from mpi4py import MPI

from src.probes import memreport

# Support both repo layouts:
#   - nested (feature branches): src.rydberg.lattices, src.engines.qaqmc
#   - flat   (main):              src.lattices,         src.qaqmc
try:
    from src.rydberg.lattices import (
        generate_kagome_bond_lattice,
        kagome_loop_string_translations,
    )
except ModuleNotFoundError:
    from src.lattices import (
        generate_kagome_bond_lattice,
        kagome_loop_string_translations,
    )
try:
    from src.engines.qaqmc import QAQMC_Rydberg
except ModuleNotFoundError:
    from src.qaqmc import QAQMC_Rydberg


def _fmt_sec(seconds):
    return f"{seconds:.2f}s ({seconds/3600.0:.2f}h)"


def main():
    parser = argparse.ArgumentParser(description="Probe QAQMC on-the-fly MPI runtime")
    parser.add_argument("--lattice", type=str, default="kagome_bond_triangle",
                        choices=["kagome_bond", "kagome_bond_triangle"])
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--a", type=float, default=4.0)
    parser.add_argument("--M", type=int, default=2760000)
    parser.add_argument("--Omega", type=float, default=1.0)
    parser.add_argument("--Rb", type=float, default=2.4)
    parser.add_argument("--delta-min", type=float, default=-2.0)
    parser.add_argument("--delta-max", type=float, default=6.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neighbor-cutoff", type=int, default=3)
    parser.add_argument("--delta-groups", type=int, default=600,
                        help="Number of delta groups for shared alias tables (must be > 0).")
    parser.add_argument("--omp-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    parser.add_argument("--probe-equil", type=int, default=3, help="equil steps for timing probe")
    parser.add_argument("--probe-samples", type=int, default=16, help="total samples across all ranks")
    parser.add_argument("--measure-every", type=int, default=1)
    parser.add_argument("--target-equil", type=int, default=0, help="full-run n_equil for estimate")
    parser.add_argument("--target-samples", type=int, default=30000, help="full-run total n_samples")
    parser.add_argument("--target-ranks", type=int, default=None,
                        help="production rank count for the node-memory estimate")
    parser.add_argument("--node-mem-gb", type=float, default=240.0)
    args = parser.parse_args()

    if args.omp_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    if args.lattice == "kagome_bond_triangle":
        from src.rydberg.lattices import generate_kagome_bond_triangle_lattice
        pos = generate_kagome_bond_triangle_lattice(args.nx, args.ny, args.a)
    else:
        pos = generate_kagome_bond_lattice(nx=args.nx, ny=args.ny, a=args.a)
    n_sites = len(pos)
    m2 = 2 * args.M

    # Split probe samples across ranks
    base = args.probe_samples // n_ranks
    rem = args.probe_samples % n_ranks
    my_probe = base + (1 if rank < rem else 0)

    # Target split
    t_base = args.target_samples // n_ranks
    t_rem = args.target_samples % n_ranks
    my_target = t_base + (1 if rank < t_rem else 0)
    max_target = comm.reduce(my_target, op=MPI.MAX, root=0)

    # Build engine (init timing)
    init_t0 = time.perf_counter()
    engine = QAQMC_Rydberg(
        N=n_sites, M=args.M, Omega=args.Omega, Rb=args.Rb,
        delta_min=args.delta_min, delta_max=args.delta_max,
        pos=pos, epsilon=args.epsilon,
        seed=args.seed + rank * 9973,
        verbose=False, use_cpp=True, omp_threads=args.omp_threads,
        neighbor_cutoff=args.neighbor_cutoff,
        delta_groups=args.delta_groups,
    )
    t_init = time.perf_counter() - init_t0

    # Set observable sites (lattice-aware; identical to the production driver)
    try:
        from src.mpi.qaqmc_mpi import _lattice_observables
        (bulk, loop_sets, string_sets, _lm, _sm, vertex_sets, _m) = \
            _lattice_observables(args.lattice, args.nx, args.ny)
        engine._cpp_engine.set_bulk_sites(bulk)
        engine._cpp_engine.set_observable_sites(loop_sets + vertex_sets, string_sets)
    except ImportError:
        loop_sets, string_sets = kagome_loop_string_translations(args.nx, args.ny)
        engine._cpp_engine.set_observable_sites(loop_sets, string_sets)

    # Probe equilibration
    t0 = time.perf_counter()
    try:
        engine._cpp_engine.run(args.probe_equil, 0, None, 1)
    except TypeError:
        engine._cpp_engine.run(args.probe_equil, 0)
    t_equil = time.perf_counter() - t0

    # Probe sampling (on-the-fly)
    t0 = time.perf_counter()
    try:
        result = engine._cpp_engine.run_onthefly(0, my_probe, args.measure_every, None, 1)
    except TypeError:
        result = engine._cpp_engine.run_onthefly(0, my_probe, args.measure_every)
    t_sample = time.perf_counter() - t0

    # Get profiling from engine
    time_diag = engine._cpp_engine.time_diag
    time_clus = engine._cpp_engine.time_clus
    mc_steps = engine._cpp_engine.mc_steps

    # Reduce timings
    max_init = comm.reduce(t_init, op=MPI.MAX, root=0)
    avg_init = comm.allreduce(t_init, op=MPI.SUM) / n_ranks
    max_equil = comm.reduce(t_equil, op=MPI.MAX, root=0)
    avg_equil = comm.allreduce(t_equil, op=MPI.SUM) / n_ranks
    max_sample = comm.reduce(t_sample, op=MPI.MAX, root=0)
    avg_sample = comm.allreduce(t_sample, op=MPI.SUM) / n_ranks
    min_probe = comm.reduce(my_probe, op=MPI.MIN, root=0)
    max_probe = comm.reduce(my_probe, op=MPI.MAX, root=0)

    # Profiling breakdown
    max_diag = comm.reduce(time_diag, op=MPI.MAX, root=0)
    max_clus = comm.reduce(time_clus, op=MPI.MAX, root=0)
    total_mc = comm.reduce(mc_steps, op=MPI.MAX, root=0)

    comm.Barrier()

    if rank == 0:
        eq_step = max_equil / max(args.probe_equil, 1)
        sa_step = max_sample / max(max_probe, 1)
        est_equil = eq_step * args.target_equil
        est_sample = sa_step * max_target
        est_init = max_init
        est_total = est_init + est_equil + est_sample

        diag_pct = 100 * max_diag / (max_diag + max_clus) if (max_diag + max_clus) > 0 else 0
        clus_pct = 100 - diag_pct

        print("=" * 86)
        print("QAQMC On-The-Fly MPI Runtime Probe")
        print("=" * 86)
        print(
            f"config: ranks={n_ranks}, omp_threads={args.omp_threads}, "
            f"N={n_sites}, M={args.M}, M_total={m2}, "
            f"delta_groups={args.delta_groups}, cutoff={args.neighbor_cutoff}"
        )
        print(
            f"probe: n_equil={args.probe_equil}, n_samples_total={args.probe_samples}, "
            f"per-rank(min/max)={min_probe}/{max_probe}, measure_every={args.measure_every}"
        )
        print(
            f"observables: {len(loop_sets)} loop translations, "
            f"{len(string_sets)} string translations"
        )
        print("-" * 86)
        print(f"init    avg/max: {_fmt_sec(avg_init)} / {_fmt_sec(max_init)}")
        print(
            f"equil   avg/max: {_fmt_sec(avg_equil)} / {_fmt_sec(max_equil)} "
            f" -> max per-step {eq_step:.4f}s"
        )
        print(
            f"sample  avg/max: {_fmt_sec(avg_sample)} / {_fmt_sec(max_sample)} "
            f" -> max per-sample-step {sa_step:.4f}s"
        )
        print(f"  diag update: {_fmt_sec(max_diag)} ({diag_pct:.1f}%)")
        print(f"  cluster upd: {_fmt_sec(max_clus)} ({clus_pct:.1f}%)")
        print(f"  total mc_steps (max rank): {total_mc}")
        print("-" * 86)
        print(
            f"estimate target: n_equil={args.target_equil}, n_samples_total={args.target_samples}, "
            f"per-rank(max)={max_target}"
        )
        print(f"  est init   : {_fmt_sec(est_init)}")
        print(f"  est equil  : {_fmt_sec(est_equil)}")
        print(f"  est sample : {_fmt_sec(est_sample)}")
        print(f"  est total  : {_fmt_sec(est_total)}")
        print()
        print("  (on-the-fly mode: no write benchmark needed — output is 3 × n_samples float64)")
        print("=" * 86)

    core = memreport.qaqmc_engine_core_mib(engine._cpp_engine)
    m_total = 2 * int(args.M)
    memreport.report(
        "probe-OTF", comm,
        engine_core_mib=core,
        core_label=f"engine core (exact capacity, M_total={m_total})",
        notes=(
            f"warm-start config export transient ≈ +{m_total * 8 / 2**20:.0f} MiB/rank (int32 copies)",
            "production rank-0 additionally gathers occ-SF/profile batch arrays at the end "
            "— rank-0 peak exceeds the per-rank figure above",
        ),
        target_ranks=args.target_ranks, node_mem_gb=args.node_mem_gb)


if __name__ == "__main__":
    main()
