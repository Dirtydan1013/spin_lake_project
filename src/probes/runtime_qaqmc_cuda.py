"""Compare CUDA QAQMC transition kernels against the CPU reference.

Run inside a Slurm GPU allocation after building ``build_cuda/qaqmc_cuda``.
Diagonal-only and full diagonal+cluster timings are reported separately so a
kernel-local speedup cannot be mistaken for end-to-end MC throughput.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import qaqmc_cpp
import qaqmc_cuda
from src.engines.qaqmc_cuda import CudaDiagonalBackend
from src.rydberg.lattices import (
    generate_kagome_bond_lattice,
    generate_kagome_bond_triangle_lattice,
    lattice_box_vectors,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", choices=["kagome_bond", "kagome_bond_triangle"],
                        default="kagome_bond")
    parser.add_argument("--boundary", choices=["open", "periodic"], default="periodic")
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--a", type=float, default=4.0)
    parser.add_argument("--M", type=int, default=2_760_000)
    parser.add_argument("--Omega", type=float, default=1.0)
    parser.add_argument("--Rb", type=float, default=2.4)
    parser.add_argument("--delta-min", type=float, default=-2.0)
    parser.add_argument("--delta-max", type=float, default=6.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--delta-groups", type=int, default=600)
    parser.add_argument("--neighbor-cutoff", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-warmup", type=int, default=1)
    parser.add_argument("--gpu-steps", type=int, default=5)
    parser.add_argument("--full-steps", type=int, default=3,
                        help="GPU diagonal+cluster MC steps to benchmark")
    parser.add_argument("--event-builds", type=int, default=0,
                        help="also benchmark sorted vertex-event construction")
    parser.add_argument("--cpu-steps", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if args.lattice == "kagome_bond":
        pos = generate_kagome_bond_lattice(args.nx, args.ny, args.a)
    else:
        if args.boundary == "periodic":
            raise ValueError("kagome_bond_triangle does not support periodic boundaries")
        pos = generate_kagome_bond_triangle_lattice(args.nx, args.ny, args.a)
    box = (lattice_box_vectors(args.lattice, args.nx, args.ny, args.a, N=len(pos))
           if args.boundary == "periodic" else None)

    print(f"device_info={qaqmc_cuda.device_info()}", flush=True)
    print(f"config N={len(pos)} M={args.M} L={2*args.M} "
          f"cutoff={args.neighbor_cutoff} groups={args.delta_groups}", flush=True)

    t0 = time.perf_counter()
    cpu = qaqmc_cpp.QAQMCEngine(
        len(pos), args.Omega, args.delta_min, args.delta_max, args.Rb,
        args.M, args.epsilon, args.seed, np.ascontiguousarray(pos, np.float64),
        neighbor_cutoff=args.neighbor_cutoff, delta_groups=args.delta_groups,
        box_vectors=box,
    )
    print(f"cpu_init_s={time.perf_counter()-t0:.6f}", flush=True)

    cpu.reset_timers()
    t0 = time.perf_counter()
    for _ in range(args.cpu_steps):
        cpu.mc_step()
    cpu_wall = time.perf_counter() - t0
    cpu_diag_per_step = cpu.time_diag / max(args.cpu_steps, 1)
    print(f"cpu_wall_s={cpu_wall:.6f} cpu_diag_s_per_step={cpu_diag_per_step:.6f} "
          f"cpu_cluster_s_per_step={cpu.time_clus/max(args.cpu_steps,1):.6f}", flush=True)

    t0 = time.perf_counter()
    gpu = CudaDiagonalBackend.from_cpu_engine(cpu, device=args.device)
    print(f"gpu_construct_s={time.perf_counter()-t0:.6f} "
          f"gpu_device_mib={gpu.device_bytes/2**20:.3f}", flush=True)

    sweep = 0
    for _ in range(args.gpu_warmup):
        gpu.diagonal_update(args.seed + 1000, sweep)
        sweep += 1

    elapsed_ms = []
    attempts = []
    for _ in range(args.gpu_steps):
        stats = gpu.diagonal_update(args.seed + 1000, sweep)
        sweep += 1
        if stats["failed_slots"]:
            raise RuntimeError(f"GPU diagonal update failed: {stats}")
        elapsed_ms.append(float(stats["elapsed_ms"]))
        attempts.append(stats["proposal_attempts"] / max(stats["updated_slots"], 1))

    gpu_s = float(np.median(elapsed_ms)) / 1000.0
    speedup = cpu_diag_per_step / gpu_s
    print(f"gpu_diag_ms_median={np.median(elapsed_ms):.6f} "
          f"gpu_diag_ms_min={np.min(elapsed_ms):.6f} "
          f"attempts_per_slot={np.mean(attempts):.6f}", flush=True)
    print(f"diagonal_speedup_vs_single_cpu_core={speedup:.3f}x", flush=True)

    if args.event_builds > 0:
        event_ms = []
        for _ in range(args.event_builds):
            stats = gpu.engine.build_events(download=False)
            event_ms.append(float(stats["elapsed_ms"]))
        print(f"gpu_event_ms_median={np.median(event_ms):.6f} "
              f"gpu_event_ms_min={np.min(event_ms):.6f} "
              f"gpu_device_mib_with_events={gpu.device_bytes/2**20:.3f} "
              f"site_events={stats['site_events']} bond_events={stats['bond_events']}",
              flush=True)

    if args.full_steps > 0:
        full_wall_ms = []
        full_diag_ms = []
        full_event_ms = []
        full_cluster_ms = []
        for _ in range(args.full_steps):
            t0 = time.perf_counter()
            diagonal = gpu.engine.diagonal_update(seed=args.seed + 2000,
                                                  sweep_id=sweep)
            cluster = gpu.engine.cluster_update(seed=args.seed + 3000,
                                                sweep_id=sweep)
            full_wall_ms.append((time.perf_counter() - t0) * 1000.0)
            full_diag_ms.append(float(diagonal["elapsed_ms"]))
            full_event_ms.append(float(cluster["event_ms"]))
            full_cluster_ms.append(float(cluster["sweep_ms"]))
            sweep += 1
        gpu_full_s = float(np.median(full_wall_ms)) / 1000.0
        cpu_full_s = cpu_wall / max(args.cpu_steps, 1)
        print(
            f"gpu_full_wall_ms_median={np.median(full_wall_ms):.6f} "
            f"gpu_full_diag_ms_median={np.median(full_diag_ms):.6f} "
            f"gpu_full_event_ms_median={np.median(full_event_ms):.6f} "
            f"gpu_full_cluster_ms_median={np.median(full_cluster_ms):.6f}",
            flush=True,
        )
        print(f"full_mc_speedup_vs_single_cpu_core={cpu_full_s/gpu_full_s:.3f}x",
              flush=True)


if __name__ == "__main__":
    main()
