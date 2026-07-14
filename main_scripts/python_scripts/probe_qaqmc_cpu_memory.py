"""Reproducible RSS/capacity/runtime probe for the standard CPU QAQMC engine.

Run each configuration in a fresh process.  The probe records RSS before the
engine is constructed, after construction, after event scratch reaches steady
state, and after timed MC steps.  Operator-string export is optional because a
full int32 checkpoint copy can itself add O(M) RSS and contaminate the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

import qaqmc_cpp
from src.rydberg.lattices import (
    generate_kagome_bond_lattice,
    generate_kagome_bond_triangle_lattice,
    lattice_box_vectors,
)


def _proc_status_kib() -> dict[str, int]:
    wanted = {"VmRSS", "VmHWM", "VmSize", "RssAnon", "RssFile", "RssShmem"}
    result: dict[str, int] = {}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                key = line.split(":", 1)[0]
                if key in wanted:
                    result[key] = int(line.split()[1])
    except OSError:
        pass
    return result


def _smaps_rollup_kib() -> dict[str, int]:
    wanted = {"Rss", "Pss", "Private_Clean", "Private_Dirty", "Shared_Clean", "Shared_Dirty"}
    result: dict[str, int] = {}
    try:
        with open("/proc/self/smaps_rollup", "r", encoding="utf-8") as handle:
            for line in handle:
                key = line.split(":", 1)[0]
                if key in wanted:
                    result[key] = int(line.split()[1])
    except OSError:
        pass
    return result


def _memory_snapshot() -> dict[str, object]:
    return {
        "status_kib": _proc_status_kib(),
        "smaps_rollup_kib": _smaps_rollup_kib(),
        "ru_maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--neighbor-cutoff", type=int, default=-1)
    parser.add_argument("--delta-groups", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=5)
    parser.add_argument("--export-checksum", action="store_true",
                        help="export full int32 op arrays after RSS measurement and hash them")
    parser.add_argument("--snapshot-out", type=Path,
                        help="optional npz with final op arrays/RNG state (implies export)")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.lattice == "kagome_bond":
        pos = generate_kagome_bond_lattice(args.nx, args.ny, args.a)
    else:
        if args.boundary == "periodic":
            raise ValueError("kagome_bond_triangle does not support periodic boundaries")
        pos = generate_kagome_bond_triangle_lattice(args.nx, args.ny, args.a)
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    box = (lattice_box_vectors(args.lattice, args.nx, args.ny, args.a, N=len(pos))
           if args.boundary == "periodic" else None)

    result: dict[str, object] = {
        "pid": os.getpid(),
        "config": {
            "lattice": args.lattice,
            "boundary": args.boundary,
            "nx": args.nx,
            "ny": args.ny,
            "N": len(pos),
            "M": args.M,
            "L": 2 * args.M,
            "neighbor_cutoff": args.neighbor_cutoff,
            "delta_groups": args.delta_groups,
            "seed": args.seed,
        },
        "memory_before_engine": _memory_snapshot(),
    }

    start = time.perf_counter()
    engine = qaqmc_cpp.QAQMCEngine(
        len(pos), args.Omega, args.delta_min, args.delta_max, args.Rb,
        args.M, args.epsilon, args.seed, pos,
        neighbor_cutoff=args.neighbor_cutoff,
        delta_groups=args.delta_groups,
        box_vectors=box,
    )
    result["init_seconds"] = time.perf_counter() - start
    result["memory_after_init"] = _memory_snapshot()

    for _ in range(args.warmup_steps):
        engine.mc_step()
    result["memory_after_warmup"] = _memory_snapshot()

    engine.reset_timers()
    start = time.perf_counter()
    for _ in range(args.timed_steps):
        engine.mc_step()
    wall = time.perf_counter() - start
    result["timing"] = {
        "steps": args.timed_steps,
        "wall_seconds": wall,
        "wall_seconds_per_step": wall / max(args.timed_steps, 1),
        "diagonal_seconds_per_step": engine.time_diag / max(args.timed_steps, 1),
        "cluster_seconds_per_step": engine.time_clus / max(args.timed_steps, 1),
    }
    result["memory_after_timed_steps_before_export"] = _memory_snapshot()

    if hasattr(engine, "memory_breakdown"):
        breakdown = {str(k): int(v) for k, v in engine.memory_breakdown.items()}
        slots = max(breakdown.get("operator_slots", 0), 1)
        breakdown["bond_operator_fraction"] = (
            breakdown.get("bond_operator_count", 0) / slots
        )
        breakdown["offdiag_operator_fraction"] = (
            breakdown.get("offdiag_operator_count", 0) / slots
        )
        result["engine_memory_breakdown"] = breakdown

    if args.export_checksum or args.snapshot_out is not None:
        op_types = np.array(engine.op_types, dtype=np.int32, copy=True)
        op_sites = np.array(engine.op_sites, dtype=np.int32, copy=True)
        result["operator_checksum"] = {
            "types_sha256": _sha256(op_types),
            "sites_sha256": _sha256(op_sites),
            "rng_state_sha256": hashlib.sha256(
                engine.get_rng_state().encode("ascii")
            ).hexdigest(),
        }
        result["memory_after_operator_export"] = _memory_snapshot()
        if args.snapshot_out is not None:
            args.snapshot_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.snapshot_out,
                op_types=op_types,
                op_sites=op_sites,
                rng_state=np.asarray(engine.get_rng_state()),
            )

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload, flush=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
