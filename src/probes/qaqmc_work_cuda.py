"""Benchmark the off-diagonal and Renyi CUDA transition backends.

The probe reports full diagonal+cluster wall time, first/cold and cached
topology times, and the backend's exact allocated device bytes before/after
checkpoint and lazy event workspaces.
Run from ``/tmp`` with ``build_cuda`` first on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import qaqmc_cuda

from src.probes import memreport
# Resolve the portable qaqmc_cpp paired with qaqmc_cuda before the legacy CPU
# facades consider prepending the repository's generic build/ directory.
from src.engines.qaqmc_renyi_work_cuda import QAQMCRenyiWorkRydbergCUDA
from src.engines.qaqmc_string_work_cuda import QAQMCStringWorkRydbergCUDA
from src.engines.qaqmc_renyi_work import QAQMCRenyiWorkRydberg
from src.engines.qaqmc_string_work import QAQMCStringWorkRydberg
from src.rydberg.lattices import (
    generate_1d_chain,
    generate_kagome_bond_lattice,
    generate_kagome_bond_triangle_lattice,
    lattice_box_vectors,
)


def _median_step(call, count: int) -> float:
    samples = []
    for _ in range(count):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)) if samples else float("nan")


def _geometry(args):
    if args.lattice == "1d_chain":
        return generate_1d_chain(args.N, args.a), None
    if args.lattice == "kagome_bond_triangle":
        if args.boundary == "periodic":
            raise ValueError("triangle patch does not support periodic boundaries")
        return generate_kagome_bond_triangle_lattice(args.nx, args.ny, args.a), None
    pos = generate_kagome_bond_lattice(args.nx, args.ny, args.a)
    box = (lattice_box_vectors(args.lattice, args.nx, args.ny, args.a, N=len(pos))
           if args.boundary == "periodic" else None)
    return pos, box


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["string", "renyi"], required=True)
    parser.add_argument("--lattice",
                        choices=["1d_chain", "kagome_bond", "kagome_bond_triangle"],
                        default="kagome_bond_triangle")
    parser.add_argument("--boundary", choices=["open", "periodic"], default="open")
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--a", type=float, default=4.0)
    parser.add_argument("--M", type=int, default=225_700)
    parser.add_argument("--Rb", type=float, default=2.4)
    parser.add_argument("--delta-min", type=float, default=-2.0)
    parser.add_argument("--delta-max", type=float, default=4.5)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--neighbor-cutoff", type=int, default=-1)
    parser.add_argument("--delta-groups", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--gpu-steps", type=int, default=5)
    parser.add_argument("--cpu-steps", type=int, default=1)
    parser.add_argument("--topology-sweeps", type=int, default=5)
    parser.add_argument("--sites", default="0,1",
                        help="string sites or Renyi D sites")
    args = parser.parse_args()

    pos, box = _geometry(args)
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    sites = np.asarray([int(x) for x in args.sites.split(",") if x], dtype=np.int32)
    if not len(sites):
        raise ValueError("--sites must contain at least one site")
    common = dict(
        N=len(pos), M=args.M, Omega=1.0, Rb=args.Rb,
        delta_min=args.delta_min, delta_max=args.delta_max,
        epsilon=args.epsilon, seed=args.seed, pos=pos,
        neighbor_cutoff=(None if args.neighbor_cutoff < 0 else args.neighbor_cutoff),
        delta_groups=args.delta_groups, box_vectors=box,
    )
    report = {
        "engine": args.engine,
        "N": len(pos),
        "M": args.M,
        "device": qaqmc_cuda.device_info()[args.device],
    }

    if args.engine == "string":
        cpu = QAQMCStringWorkRydberg(**common)
        gpu = QAQMCStringWorkRydbergCUDA(
            **common, device=args.device, verbose=False
        )
        cpu.set_string_sites(sites)
        gpu.set_string_sites(sites)
        cpu_call = cpu._eng.mc_step
        gpu_call = gpu._eng.mc_step
        cpu_topology = lambda: cpu._eng.topology_sweep(0.5)
        gpu_topology = lambda: gpu._eng.topology_sweep(0.5)
    else:
        cpu = QAQMCRenyiWorkRydberg(**common)
        gpu = QAQMCRenyiWorkRydbergCUDA(
            **common, device=args.device, verbose=False
        )
        start = np.zeros(len(pos), dtype=np.uint8)
        end = start.copy()
        end[sites] = 1
        cpu.set_region_pair(start, end)
        gpu.set_region_pair(start, end)
        cpu_call = cpu._cpp_engine.backend.mc_step
        gpu_call = gpu._backend.mc_step
        cpu_topology = lambda: cpu._cpp_engine.backend.log_weight_ratio_for_toggle(
            int(sites[0])
        )
        gpu_topology = lambda: gpu._backend.topology_sweep(sites[:1], 0.5)

    report["device_mib_initial"] = gpu.device_bytes / 2**20
    gpu.thermalize(0)
    report["device_mib_after_checkpoint"] = gpu.device_bytes / 2**20
    for _ in range(args.warmup):
        gpu_call()
    report["gpu_full_step_s_median"] = _median_step(gpu_call, args.gpu_steps)
    report["device_mib_after_step"] = gpu.device_bytes / 2**20
    topology_start = time.perf_counter()
    gpu_topology()
    report["gpu_topology_first_s"] = time.perf_counter() - topology_start
    report["gpu_topology_s_median"] = _median_step(
        gpu_topology, args.topology_sweeps
    )
    report["device_mib_after_topology"] = gpu.device_bytes / 2**20
    report["cpu_full_step_s_median"] = _median_step(cpu_call, args.cpu_steps)
    report["cpu_topology_s_median"] = _median_step(
        cpu_topology, min(args.topology_sweeps, max(args.cpu_steps, 1))
    )
    report["full_step_speedup"] = (
        report["cpu_full_step_s_median"] / report["gpu_full_step_s_median"]
    )
    report["device_mib_peak_sampled"] = max(
        value for key, value in report.items() if key.startswith("device_mib_"))
    report["host_rss_mib"] = round(memreport.rss_mib(), 1)
    report["host_peak_mib"] = round(memreport.peak_mib(), 1)
    report["vram_total_mib"] = round(
        qaqmc_cuda.device_info()[args.device]["total_memory"] / 2**20, 1)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
