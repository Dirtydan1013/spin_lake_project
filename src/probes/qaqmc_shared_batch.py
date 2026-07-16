"""Measure shared-model CPU chain throughput and process memory."""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import time

import numpy as np

from src.engines.qaqmc_cpu_batch import QAQMCSharedModelBatch
from src.rydberg.lattices import generate_kagome_bond_lattice, lattice_box_vectors


def _memory() -> dict[str, int]:
    values: dict[str, int] = {
        "ru_maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    }
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                key = line.split(":", 1)[0]
                if key in {"VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"}:
                    values[f"{key}_kib"] = int(line.split()[1])
    except OSError:
        pass
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--M", type=int, default=2_760_000)
    parser.add_argument("--Rb", type=float, default=2.4)
    parser.add_argument("--delta-min", type=float, default=-2.0)
    parser.add_argument("--delta-max", type=float, default=6.0)
    parser.add_argument("--delta-groups", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--event-storage",
                        choices=["packed64", "p_bond16", "p_only32"],
                        default="packed64")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    pos = np.ascontiguousarray(
        generate_kagome_bond_lattice(args.nx, args.ny, 4.0),
        dtype=np.float64,
    )
    box = lattice_box_vectors(
        "kagome_bond", args.nx, args.ny, 4.0, N=len(pos)
    )
    before = _memory()
    start = time.perf_counter()
    with QAQMCSharedModelBatch(
        batch_size=args.batch_size,
        N=len(pos), M=args.M, Omega=1.0, Rb=args.Rb,
        delta_min=args.delta_min, delta_max=args.delta_max,
        pos=pos, epsilon=0.01, seed=args.seed,
        neighbor_cutoff=-1, delta_groups=args.delta_groups,
        box_vectors=box, bond_event_storage=args.event_storage,
    ) as batch:
        init_seconds = time.perf_counter() - start
        after_init = _memory()
        for _ in range(args.warmup):
            batch.mc_step()
        after_warmup = _memory()
        samples = []
        for _ in range(args.steps):
            start = time.perf_counter()
            batch.mc_step()
            samples.append(time.perf_counter() - start)
        median = statistics.median(samples)
        report = {
            "pid": os.getpid(),
            "N": len(pos),
            "M": args.M,
            "batch_size": args.batch_size,
            "event_storage": args.event_storage,
            "init_seconds": init_seconds,
            "batch_step_seconds_median": median,
            "chain_steps_per_second": args.batch_size / median,
            "shared_model_mib": batch.shared_model_bytes / 2**20,
            "dominant_resident_mib": batch.dominant_resident_bytes / 2**20,
            "dominant_chain_mib": [value / 2**20
                                    for value in batch.dominant_chain_bytes],
            "memory_before": before,
            "memory_after_init": after_init,
            "memory_after_warmup": after_warmup,
            "memory_after_timing": _memory(),
            "step_samples_seconds": samples,
        }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
