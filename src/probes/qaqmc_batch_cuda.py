"""Measure true in-process CUDA batch throughput and shared-model VRAM."""

from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np

import qaqmc_cpp
import qaqmc_cuda
from src.engines.qaqmc_batch_cuda import CudaDiagonalBatchBackend
from src.engines.qaqmc_renyi_work_batch_cuda import CudaRenyiBatchBackend
from src.rydberg.lattices import generate_kagome_bond_triangle_lattice


def _median(call, count: int) -> float:
    samples = []
    for _ in range(count):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", default="standard,string,renyi")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--M", type=int, default=2_760_000)
    parser.add_argument("--Rb", type=float, default=2.4)
    parser.add_argument("--delta-min", type=float, default=-2.0)
    parser.add_argument("--delta-max", type=float, default=4.5)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--delta-groups", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    positions = np.ascontiguousarray(
        generate_kagome_bond_triangle_lattice(args.nx, args.ny, 4.0),
        dtype=np.float64,
    )
    model = qaqmc_cpp.QAQMCEngine(
        len(positions), 1.0, args.delta_min, args.delta_max, args.Rb,
        args.M, args.epsilon, args.seed, positions,
        neighbor_cutoff=-1, delta_groups=args.delta_groups,
    )
    engines = [value for value in args.engines.split(",") if value]
    batches = [int(value) for value in args.batch_sizes.split(",") if value]
    baseline: dict[str, float] = {}
    reports = []
    for name in engines:
        for batch_size in batches:
            if name == "renyi":
                backend = CudaRenyiBatchBackend.from_cpu_model(
                    model, batch_size=batch_size,
                    device=args.device, seed=args.seed,
                )
            else:
                backend = CudaDiagonalBatchBackend.from_cpu_engine(
                    model, batch_size=batch_size,
                    device=args.device, seed=args.seed,
                )
                if name == "string":
                    backend.set_string_sites(np.array([0, 1], dtype=np.int32))
                    backend.set_seam_masks_consistent(
                        np.zeros(batch_size, dtype=np.uint64)
                    )
            initial_bytes = backend.device_bytes
            for _ in range(args.warmup):
                backend.mc_step()
            wall = _median(backend.mc_step, args.steps)
            chain_rate = batch_size / wall
            if batch_size == 1:
                baseline[name] = chain_rate
            reports.append({
                "engine": name,
                "batch_size": batch_size,
                "device": qaqmc_cuda.device_info()[args.device],
                "N": len(positions),
                "M": args.M,
                "batch_step_s_median": wall,
                "chain_steps_per_s": chain_rate,
                "throughput_vs_b1": chain_rate / baseline[name],
                "device_mib_initial": initial_bytes / 2**20,
                "device_mib_after_step": backend.device_bytes / 2**20,
                "shared_model_mib": backend.shared_model_bytes / 2**20,
                "mib_per_chain_after_step": (
                    (backend.device_bytes - backend.shared_model_bytes)
                    / batch_size / 2**20
                ),
            })
            print(json.dumps(reports[-1]), flush=True)
            del backend
            gc.collect()
    print(json.dumps({"reports": reports}, indent=2), flush=True)


if __name__ == "__main__":
    main()
