"""NUMA-oriented MPI x threaded-chain QAQMC scaling probe."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time

import numpy as np
from mpi4py import MPI

from src.engines.qaqmc_cpu_batch import QAQMCSharedModelBatch
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
    parser.add_argument("--chains-per-rank", type=int, default=16)
    parser.add_argument("--M", type=int, default=2_760_000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--event-storage",
                        choices=["packed64", "p_bond16", "p_only32"],
                        default="packed64")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ranks = comm.Get_size()
    pos = np.ascontiguousarray(generate_kagome_bond_lattice(6, 6, 4.0),
                               dtype=np.float64)
    box = lattice_box_vectors("kagome_bond", 6, 6, 4.0, N=len(pos))
    before = _rss_kib()
    start = time.perf_counter()
    with QAQMCSharedModelBatch(
        batch_size=args.chains_per_rank,
        N=len(pos), M=args.M, Omega=1.0, Rb=2.4,
        delta_min=-2.0, delta_max=6.0, pos=pos, epsilon=0.01,
        seed=42 + rank * 1_000_003, neighbor_cutoff=-1, delta_groups=600,
        box_vectors=box, bond_event_storage=args.event_storage,
    ) as batch:
        init_seconds = time.perf_counter() - start
        after_init = _rss_kib()
        for _ in range(args.warmup):
            batch.mc_step()
        after_warmup = _rss_kib()
        comm.Barrier()
        samples = []
        for _ in range(args.steps):
            start = time.perf_counter()
            batch.mc_step()
            samples.append(time.perf_counter() - start)
        local = {
            "rank": rank,
            "rss_before_kib": before,
            "rss_after_init_kib": after_init,
            "rss_after_warmup_kib": after_warmup,
            "rss_after_timing_kib": _rss_kib(),
            "init_seconds": init_seconds,
            "median_batch_step_seconds": statistics.median(samples),
            "max_batch_step_seconds": max(samples),
            "shared_model_bytes": batch.shared_model_bytes,
            "dominant_resident_bytes": batch.dominant_resident_bytes,
        }
    rows = comm.gather(local, root=0)
    if rank == 0:
        slowest = max(row["median_batch_step_seconds"] for row in rows)
        rss = [row["rss_after_timing_kib"] for row in rows]
        report = {
            "mpi_ranks": ranks,
            "chains_per_rank": args.chains_per_rank,
            "total_chains": ranks * args.chains_per_rank,
            "M": args.M,
            "event_storage": args.event_storage,
            "node_chain_steps_per_second":
                ranks * args.chains_per_rank / slowest,
            "slowest_rank_batch_step_seconds_median": slowest,
            "node_rss_gib_sum": sum(rss) / 2**20,
            "rank_rss_mib_median": statistics.median(rss) / 1024,
            "rank_rss_mib_max": max(rss) / 1024,
            "shared_models_per_node": ranks,
            "shared_model_mib_each": rows[0]["shared_model_bytes"] / 2**20,
            "rank_rows": rows,
        }
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
