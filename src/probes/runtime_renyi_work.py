"""
MPI runtime probe for the QAQMC Renyi nonequilibrium-work engine.

Times the basic primitives (engine init, thermalize step, backend mc_step,
single topology-toggle proposal, and full probe trajectory), then extrapolates
to a target production scale.  Mirrors the output style of
``probe_runtime_kp_expanded.py``.

Usage (single config):
    mpiexec -n 32 python -u scripts/python\\ script/probe_runtime_renyi_work.py \\
        --lattice kagome_bond_triangle --nx 6 --ny 6 --m 1 \\
        --M 100000 --a 4.0 --Rb 2.4 --Omega 1.0 \\
        --delta-min -2.0 --delta-max 6.0 --epsilon 0.01 \\
        --neighbor-cutoff -1 --delta-groups 600 \\
        --kp-region ABC \\
        --probe-warmup-mc-steps 10 \\
        --probe-mc-steps 50 \\
        --probe-toggle-attempts 200 \\
        --probe-trajectories 3 \\
        --target-K 200 --target-n-trajectories 4000 \\
        --target-n-thermalize 5000 --target-decorrelation-steps 1000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

try:
    from mpi4py import MPI
except ImportError as exc:
    raise ImportError("mpi4py required") from exc

# Repo root importable when run via python directly
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.engines.qaqmc_renyi_work import QAQMCRenyiWorkRydberg
from src.rydberg.lattices import (
    generate_1d_chain,
    generate_kagome_bond_lattice,
    generate_kagome_bond_triangle_lattice,
)
from src.kp.kp_geometry import KP_REGION_NAMES, build_kp_region_masks_for_lattice


def _fmt(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds:8.1f}s ({seconds/3600:5.2f}h)"
    if seconds >= 60:
        return f"{seconds:8.1f}s ({seconds/60:5.2f}m)"
    return f"{seconds:8.3f}s"


def main():
    ap = argparse.ArgumentParser(description="MPI runtime probe for QAQMC Renyi work engine")
    # Lattice
    ap.add_argument("--lattice", default="kagome_bond_triangle",
                    choices=["1d_chain", "kagome_bond", "kagome_bond_triangle"])
    ap.add_argument("--N", type=int, default=0, help="(1d_chain only)")
    ap.add_argument("--nx", type=int, default=6)
    ap.add_argument("--ny", type=int, default=6)
    ap.add_argument("--a", type=float, default=4.0)
    # Hamiltonian / QAQMC params
    ap.add_argument("--M", type=int, default=100000)
    ap.add_argument("--Omega", type=float, default=1.0)
    ap.add_argument("--Rb", type=float, default=2.4)
    ap.add_argument("--delta-min", type=float, default=-2.0)
    ap.add_argument("--delta-max", type=float, default=6.0)
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--neighbor-cutoff", type=int, default=-1)
    ap.add_argument("--delta-groups", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    # Region selection: KP name OR explicit pair OR bit/site-list
    ap.add_argument("--kp-region", type=str, default="ABC",
                    help="single KP region name (probe runs ∅→region). "
                         "Use --kp-pair-start/end for nested-pair timing.")
    ap.add_argument("--kp-pair-start", type=str, default=None,
                    help="(optional) nested-pair start name; overrides --kp-region")
    ap.add_argument("--kp-pair-end", type=str, default=None,
                    help="(optional) nested-pair end name; required if --kp-pair-start given")
    ap.add_argument("--kp-m", type=int, default=1)
    ap.add_argument("--preferred-center", type=str, default="auto")
    # Probe scale (small)
    ap.add_argument("--probe-warmup-mc-steps", type=int, default=10,
                    help="cold-start steps before timed primitives")
    ap.add_argument("--probe-mc-steps", type=int, default=50,
                    help="backend mc_step calls to time")
    ap.add_argument("--probe-toggle-attempts", type=int, default=200,
                    help="topology-toggle proposals to time (averaged over D)")
    ap.add_argument("--probe-trajectories", type=int, default=3,
                    help="full probe trajectories (with K=--probe-K)")
    ap.add_argument("--probe-K", type=int, default=20,
                    help="K used for probe trajectories (small for speed)")
    ap.add_argument("--probe-decorrelation-steps", type=int, default=20)
    # Target scale (production)
    ap.add_argument("--target-K", type=int, default=200)
    ap.add_argument("--target-n-trajectories", type=int, default=4000)
    ap.add_argument("--target-n-thermalize", type=int, default=5000)
    ap.add_argument("--target-decorrelation-steps", type=int, default=1000)
    ap.add_argument("--target-n-regions", type=int, default=1,
                    help="if running KP-all (7 regions), set 7 so the estimate scales")

    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # ── Build lattice + masks ───────────────────────────────────────────────
    if args.lattice == "1d_chain":
        if args.N <= 0:
            raise ValueError("--N required for 1d_chain")
        pos = generate_1d_chain(args.N, args.a)
        N = args.N
    elif args.lattice == "kagome_bond":
        pos = generate_kagome_bond_lattice(args.nx, args.ny, args.a)
        N = len(pos)
    else:
        pos = generate_kagome_bond_triangle_lattice(args.nx, args.ny, args.a)
        N = len(pos)

    # Region masks (always built via KP unless 1d_chain)
    if args.lattice == "1d_chain":
        # Trivial: just take a half-chain
        A_end = np.zeros(N, dtype=np.uint8); A_end[: N // 2] = 1
        A_start = np.zeros(N, dtype=np.uint8)
        probe_region_name = "half_chain"
    else:
        preferred = None if args.preferred_center == "auto" else args.preferred_center
        spec = build_kp_region_masks_for_lattice(
            args.lattice, args.nx, args.ny, m=args.kp_m, a=args.a,
            preferred_center_label=preferred,
        )
        if args.kp_pair_start is not None:
            if args.kp_pair_end is None:
                raise ValueError("--kp-pair-end required with --kp-pair-start")
            for nm in (args.kp_pair_start, args.kp_pair_end):
                if nm not in KP_REGION_NAMES:
                    raise ValueError(f"unknown KP region {nm}")
            A_start = spec.region_masks[args.kp_pair_start]
            A_end   = spec.region_masks[args.kp_pair_end]
            if np.any(A_start & ~A_end):
                raise ValueError(f"--kp-pair-start={args.kp_pair_start} not ⊆ --kp-pair-end={args.kp_pair_end}")
            probe_region_name = f"{args.kp_pair_start}→{args.kp_pair_end}"
        else:
            if args.kp_region not in KP_REGION_NAMES:
                raise ValueError(f"unknown KP region {args.kp_region}")
            A_start = np.zeros(N, dtype=np.uint8)
            A_end   = spec.region_masks[args.kp_region]
            probe_region_name = f"∅→{args.kp_region}"

    D_mask = A_end & ~A_start
    D_size = int(D_mask.sum())

    # ── Build engine (timed) ────────────────────────────────────────────────
    comm.Barrier()
    t0 = time.perf_counter()
    eng = QAQMCRenyiWorkRydberg(
        N=N, M=args.M, Omega=args.Omega, Rb=args.Rb,
        delta_min=args.delta_min, delta_max=args.delta_max,
        epsilon=args.epsilon, seed=args.seed + rank * 9973,
        pos=pos, neighbor_cutoff=args.neighbor_cutoff,
        delta_groups=args.delta_groups,
    )
    eng.set_region_pair(A_start, A_end)
    t_init = time.perf_counter() - t0
    avg_init = comm.allreduce(t_init, op=MPI.SUM) / n_ranks
    max_init = comm.reduce(t_init, op=MPI.MAX, root=0)

    # Use cpp engine handles directly for the primitives below
    cpp_work = eng._cpp_engine
    cpp_backend = cpp_work.backend     # the underlying QAQMCRenyiEngine

    # ── Warmup ──────────────────────────────────────────────────────────────
    for _ in range(args.probe_warmup_mc_steps):
        cpp_backend.mc_step()

    # ── Time backend.mc_step ────────────────────────────────────────────────
    comm.Barrier()
    t0 = time.perf_counter()
    for _ in range(args.probe_mc_steps):
        cpp_backend.mc_step()
    t_mc = (time.perf_counter() - t0) / max(args.probe_mc_steps, 1)
    avg_mc = comm.allreduce(t_mc, op=MPI.SUM) / n_ranks
    max_mc = comm.reduce(t_mc, op=MPI.MAX, root=0)

    # ── Time single topology-toggle attempt ─────────────────────────────────
    # We use log_weight_ratio_for_toggle as the dominant cost in propose_toggle_site
    # (rebuilds offdiag paths twice).  Averaged over D_size choices to capture cache effects.
    D_sites = np.flatnonzero(D_mask)
    if len(D_sites) == 0:
        t_toggle = 0.0
    else:
        comm.Barrier()
        t0 = time.perf_counter()
        n_attempts = max(args.probe_toggle_attempts, 1)
        rng = np.random.default_rng(args.seed + rank)
        for _ in range(n_attempts):
            site = int(D_sites[rng.integers(0, len(D_sites))])
            _ = cpp_backend.log_weight_ratio_for_toggle(site)
        t_toggle = (time.perf_counter() - t0) / n_attempts
    avg_toggle = comm.allreduce(t_toggle, op=MPI.SUM) / n_ranks
    max_toggle = comm.reduce(t_toggle, op=MPI.MAX, root=0)

    # ── Time one full probe trajectory at probe_K ───────────────────────────
    eng.set_lambda_schedule(np.linspace(0.0, 1.0, args.probe_K + 1))
    comm.Barrier()
    t0 = time.perf_counter()
    res = eng.run_trajectories(args.probe_trajectories,
                                args.probe_decorrelation_steps)
    t_traj_block = time.perf_counter() - t0
    n_traj = max(args.probe_trajectories, 1)
    t_per_traj = t_traj_block / n_traj
    avg_per_traj = comm.allreduce(t_per_traj, op=MPI.SUM) / n_ranks
    max_per_traj = comm.reduce(t_per_traj, op=MPI.MAX, root=0)

    if rank != 0:
        return

    # ── Extrapolate to target ────────────────────────────────────────────────
    M_total = 2 * args.M
    # Production cost per region (per rank):
    #   thermalize(N_th) + n_traj_per_rank × [decorr × backend_mc_step + K × (D_size × toggle + n_qaqmc × backend)]
    # Default n_qaqmc_sweeps_per_lambda = 1 (from work engine ctor).
    n_qaqmc_per_lambda = 1
    n_topo_sweep_per_lambda = 1
    target_n_per_rank = -(-args.target_n_trajectories // n_ranks)
    target_K = args.target_K
    est_thermalize = max_mc * args.target_n_thermalize
    cost_per_traj_decorr = max_mc * args.target_decorrelation_steps
    cost_per_traj_qmc    = max_mc * target_K * n_qaqmc_per_lambda
    cost_per_traj_toggle = max_toggle * target_K * n_topo_sweep_per_lambda * D_size
    cost_per_traj = cost_per_traj_decorr + cost_per_traj_qmc + cost_per_traj_toggle
    est_traj_block = cost_per_traj * target_n_per_rank
    est_per_region = est_thermalize + est_traj_block
    est_total = max_init + args.target_n_regions * est_per_region

    print("=" * 92)
    print("QAQMC Renyi Work-engine MPI Runtime Probe")
    print("=" * 92)
    print(
        f"config: ranks={n_ranks}, lattice={args.lattice}, N={N}, M={args.M}, "
        f"M_total={M_total}, delta_groups={args.delta_groups}, cutoff={args.neighbor_cutoff}"
    )
    print(f"probe region: {probe_region_name}  (|A_start|={int(A_start.sum())}, "
          f"|A_end|={int(A_end.sum())}, |D|={D_size})")
    print("-" * 92)
    print(f"engine init                avg/max: {_fmt(avg_init)} / {_fmt(max_init)}")
    print(f"backend.mc_step            avg/max: {avg_mc*1000:7.2f}ms / {max_mc*1000:7.2f}ms  "
          f"(over {args.probe_mc_steps} steps)")
    print(f"log_weight_ratio_for_toggle avg/max: {avg_toggle*1000:7.2f}ms / {max_toggle*1000:7.2f}ms  "
          f"(over {args.probe_toggle_attempts} attempts, D_size={D_size})")
    print(f"full trajectory (K={args.probe_K:3d})    avg/max: {_fmt(avg_per_traj)} / {_fmt(max_per_traj)}  "
          f"(over {args.probe_trajectories} trajectories)")
    # Sanity: compare measured trajectory time to primitive sum
    predicted_probe_traj = (
        max_mc * args.probe_decorrelation_steps
        + max_mc * args.probe_K * n_qaqmc_per_lambda
        + max_toggle * args.probe_K * n_topo_sweep_per_lambda * D_size
    )
    print(f"  predicted from primitives at K={args.probe_K:3d}        : "
          f"{_fmt(predicted_probe_traj)}  (vs measured {_fmt(max_per_traj)})")
    print("-" * 92)
    print(f"target: K={target_K}, n_traj={args.target_n_trajectories} "
          f"({target_n_per_rank}/rank × {n_ranks} ranks), "
          f"n_thermalize={args.target_n_thermalize}, decorr={args.target_decorrelation_steps}, "
          f"n_regions={args.target_n_regions}")
    print("estimated per-rank wall-clock breakdown (max-rank dominates):")
    print(f"  init                                                : {_fmt(max_init)}")
    print(f"  thermalize × {args.target_n_thermalize:>6d} mc_step              : {_fmt(est_thermalize)}  (per region)")
    print(f"  {target_n_per_rank} trajectories × per-traj cost              : {_fmt(est_traj_block)}  (per region)")
    print(f"    per traj decorr ({args.target_decorrelation_steps} × mc_step)      : {_fmt(cost_per_traj_decorr)}")
    print(f"    per traj QAQMC ({target_K} × mc_step)             : {_fmt(cost_per_traj_qmc)}")
    print(f"    per traj toggle ({target_K} × {D_size} × ratio)    : {_fmt(cost_per_traj_toggle)}")
    print(f"  TOTAL per region                                     : {_fmt(est_per_region)}")
    if args.target_n_regions > 1:
        print(f"  × {args.target_n_regions} regions                                       : {_fmt(args.target_n_regions * est_per_region)}")
    print("-" * 92)
    print(f"OVERALL ESTIMATE (max rank, single job)              : {_fmt(est_total)}")
    print("=" * 92)


if __name__ == "__main__":
    main()
