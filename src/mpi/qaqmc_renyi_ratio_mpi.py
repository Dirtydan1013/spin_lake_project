"""
MPI entry point for a single QAQMC Renyi ratio-estimator job.

Each rank runs an independent Markov chain for the same (A_k, A_k+1) topology
pair. Rank 0 reduces the topology visit counts, writes the compact HDF5
payload, and returns the aggregated ratio result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from mpi4py import MPI
except ImportError as exc:  # pragma: no cover - exercised in environments without mpi4py
    raise ImportError("mpi4py is required for src.mpi.qaqmc_renyi_ratio_mpi") from exc

from src.engines.qaqmc_renyi import QAQMCRenyiRydberg
from src.tee.qaqmc_renyi_ratio import (
    RatioJobSpec,
    RatioRunner,
    combine_ratio_results,
    combine_visit_counts,
    save_ratio_result_hdf5,
)


def _rank_seed(seed: int, rank: int) -> int:
    return int(seed) + 9973 * int(rank)


def _to_attr_value(value):
    if isinstance(value, (str, bytes, np.number, int, float, bool)):
        return value
    if value is None:
        return "None"
    return json.dumps(value)


def run_ratio_mpi(*, N: int, M: int, A_mask, next_site: int,
                  Omega: float = 1.0, Rb: float = 1.2,
                  delta_min: float = 0.0, delta_max: float = 1.0,
                  pos=None, epsilon: float = 0.01, seed: int = 42,
                  neighbor_cutoff: int | None = None,
                  n_therm: int, n_measure: int, measure_stride: int = 1,
                  block_size: int | None = None,
                  filepath=None, region_name: str = "region", step_index: int = 0,
                  comm=None, engine_factory=None, verbose: bool = False):
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    if engine_factory is None:
        def engine_factory(local_rank):
            return QAQMCRenyiRydberg(
                N=N,
                M=M,
                Omega=Omega,
                Rb=Rb,
                delta_min=delta_min,
                delta_max=delta_max,
                pos=pos,
                epsilon=epsilon,
                seed=_rank_seed(seed, local_rank),
                neighbor_cutoff=neighbor_cutoff,
            )

    engine = engine_factory(rank)
    runner = RatioRunner(engine=engine)
    local_result = runner.run_single_ratio(
        A_mask,
        next_site=next_site,
        n_therm=n_therm,
        n_measure=n_measure,
        measure_stride=measure_stride,
        block_size=block_size,
    )
    results_per_rank = comm.gather(local_result, root=0)

    root_payload = None
    if rank == 0:
        root_payload = combine_ratio_results(results_per_rank)
        lows_per_rank = [int(item.visit_count_low) for item in results_per_rank]
        highs_per_rank = [int(item.visit_count_high) for item in results_per_rank]
        if filepath is not None:
            params = {
                "N": int(N),
                "M": int(M),
                "Omega": float(Omega),
                "Rb": float(Rb),
                "delta_min": float(delta_min),
                "delta_max": float(delta_max),
                "epsilon": float(epsilon),
                "seed": int(seed),
                "neighbor_cutoff": -1 if neighbor_cutoff is None else int(neighbor_cutoff),
                "n_therm_per_rank": int(n_therm),
                "n_measure_per_rank": int(n_measure),
                "measure_stride": int(measure_stride),
                "block_size": -1 if block_size is None else int(block_size),
                "n_ranks": int(n_ranks),
            }
            save_ratio_result_hdf5(
                filepath,
                region_name=region_name,
                step_index=int(step_index),
                A_mask=A_mask,
                next_site=int(next_site),
                result=root_payload,
                params={key: _to_attr_value(value) for key, value in params.items()},
                per_rank_visit_count_low=lows_per_rank,
                per_rank_visit_count_high=highs_per_rank,
                pos=engine.pos if hasattr(engine, "pos") else pos,
                delta_schedule=engine.delta_schedule if hasattr(engine, "delta_schedule") else None,
            )
        if verbose:
            print(
                f"[Renyi-MPI] {region_name} step={step_index} next_site={next_site} "
                f"ratio={root_payload.ratio:.6f} +/- {root_payload.ratio_err:.6f} "
                f"(low={int(root_payload.visit_count_low)}, high={int(root_payload.visit_count_high)})"
            )

    packed = comm.bcast(
        None if root_payload is None else (
            root_payload.ratio,
            root_payload.ratio_err,
            root_payload.visit_count_low,
            root_payload.visit_count_high,
        ),
        root=0,
    )
    return combine_visit_counts([packed[2]], [packed[3]]) if rank != 0 else root_payload


def run_ratio_job(job: RatioJobSpec, *, n_therm: int, n_measure: int,
                  measure_stride: int = 1, block_size: int | None = None,
                  filepath=None, **engine_kwargs):
    out_path = filepath if filepath is not None else job.output_path
    return run_ratio_mpi(
        A_mask=job.A_mask,
        next_site=job.next_site,
        region_name=job.region_name,
        step_index=job.step_index,
        n_therm=n_therm,
        n_measure=n_measure,
        measure_stride=measure_stride,
        block_size=block_size,
        filepath=out_path,
        **engine_kwargs,
    )


def _parse_mask(raw_mask: str) -> np.ndarray:
    text = raw_mask.strip()
    if text.startswith("["):
        return np.asarray(json.loads(text), dtype=np.uint8)
    if not text:
        return np.zeros(0, dtype=np.uint8)
    return np.asarray([int(item) for item in text.split(",")], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="MPI-parallel QAQMC Renyi single-ratio run")
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--Omega", type=float, default=1.0)
    parser.add_argument("--Rb", type=float, default=1.2)
    parser.add_argument("--delta_min", type=float, default=0.0)
    parser.add_argument("--delta_max", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neighbor_cutoff", type=int, default=-1)
    parser.add_argument("--A_mask", type=str, default="")
    parser.add_argument("--next_site", type=int, required=False, default=0)
    parser.add_argument("--region_name", type=str, default="region")
    parser.add_argument("--step_index", type=int, default=0)
    parser.add_argument("--n_therm", type=int, default=100)
    parser.add_argument("--n_measure", type=int, default=1000)
    parser.add_argument("--measure_stride", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=-1)
    parser.add_argument("--filepath", type=str, default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            config = json.load(fh)

    params = dict(config)
    for key, value in vars(args).items():
        if key == "config":
            continue
        if key in {"verbose"}:
            params[key] = value or params.get(key, False)
        elif key not in params or value != parser.get_default(key):
            params[key] = value

    raw_mask = params.pop("A_mask", "")
    params["A_mask"] = _parse_mask(raw_mask)
    filepath = params.get("filepath", "")
    if filepath == "":
        params["filepath"] = None
    if params.get("neighbor_cutoff", -1) < 0:
        params["neighbor_cutoff"] = None
    if params.get("block_size", -1) < 0:
        params["block_size"] = None

    run_ratio_mpi(**params)


if __name__ == "__main__":
    main()
