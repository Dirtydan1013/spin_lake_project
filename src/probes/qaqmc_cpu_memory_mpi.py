"""Node-level memory and throughput probe for independent QAQMC MPI chains."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time

import numpy as np
from mpi4py import MPI

import qaqmc_cpp
from src.rydberg.lattices import generate_kagome_bond_lattice, lattice_box_vectors


def _rss_kib() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--M", type=int, default=2_760_000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--event-storage",
                        choices=["packed64", "p_bond16", "p_only32"],
                        default="packed64")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    pos = np.ascontiguousarray(generate_kagome_bond_lattice(6, 6, 4.0),
                               dtype=np.float64)
    box = lattice_box_vectors("kagome_bond", 6, 6, 4.0, N=len(pos))

    before = _rss_kib()
    start = time.perf_counter()
    engine = qaqmc_cpp.QAQMCEngine(
        len(pos), 1.0, -2.0, 6.0, 2.4, args.M, 0.01,
        42 + 9973 * rank, pos, neighbor_cutoff=-1, delta_groups=600,
        box_vectors=box,
    )
    if hasattr(engine, "bond_event_storage"):
        engine.bond_event_storage = args.event_storage
    init_seconds = time.perf_counter() - start
    after_init = _rss_kib()
    for _ in range(args.warmup):
        engine.mc_step()
    after_warmup = _rss_kib()

    comm.Barrier()
    start = time.perf_counter()
    for _ in range(args.steps):
        engine.mc_step()
    elapsed = time.perf_counter() - start
    after_timing = _rss_kib()

    local = {
        "rank": rank,
        "rss_before_kib": before,
        "rss_after_init_kib": after_init,
        "rss_after_warmup_kib": after_warmup,
        "rss_after_timing_kib": after_timing,
        "init_seconds": init_seconds,
        "timed_seconds": elapsed,
    }
    rows = comm.gather(local, root=0)
    if rank == 0:
        elapsed_values = [row["timed_seconds"] for row in rows]
        rss_values = [row["rss_after_timing_kib"] for row in rows]
        init_values = [row["init_seconds"] for row in rows]
        report = {
            "label": args.label,
            "module": str(qaqmc_cpp.__file__),
            "ranks": size,
            "N": len(pos),
            "M": args.M,
            "event_storage": args.event_storage,
            "steps": args.steps,
            "node_chain_steps_per_second": size * args.steps / max(elapsed_values),
            "slowest_rank_seconds_per_step": max(elapsed_values) / args.steps,
            "median_rank_seconds_per_step": statistics.median(elapsed_values) / args.steps,
            "node_rss_gib_sum": sum(rss_values) / 2**20,
            "rank_rss_mib_median": statistics.median(rss_values) / 1024,
            "rank_rss_mib_max": max(rss_values) / 1024,
            "init_seconds_median": statistics.median(init_values),
            "init_seconds_max": max(init_values),
        }
        if hasattr(engine, "memory_breakdown"):
            report["rank0_memory_breakdown"] = {
                str(key): int(value)
                for key, value in engine.memory_breakdown.items()
            }
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
