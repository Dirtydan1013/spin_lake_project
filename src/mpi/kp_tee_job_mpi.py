"""MPI-aware entry point for KP TEE jobs on kagome bond lattice.

This mirrors ``src.kp.kp_tee_job`` but wires MPI into the per-region / per-step
loops via the existing ``run_ratio_mpi`` and ``run_expanded_mpi`` drivers.

Usage::

    mpiexec -n <NRANKS> python -m src.mpi.kp_tee_job_mpi \
        --method ratio   --nx 8 --ny 8 --m 2 --M 500000 \
        --a 4.0 --Rb 2.4 --Omega 1.0 --delta_min -2 --delta_max 4.5 \
        --delta_groups 600 --neighbor_cutoff 3 --seed 42 \
        --n_therm 2000 --n_measure 50000 --measure_stride 1 --block_size 500 \
        --output_dir data/kp_ratio_8x8_run

Only rank 0 builds the KP geometry, collects MC results, and writes the HDF5
and JSON artefacts.  All other ranks contribute independent Markov chains
inside ``run_ratio_mpi`` / ``run_expanded_mpi``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from mpi4py import MPI
except ImportError as exc:  # pragma: no cover
    raise ImportError("mpi4py is required for src.mpi.kp_tee_job_mpi") from exc

from src.kp.kp_geometry import (
    DEFAULT_LATTICE,
    KP_REGION_NAMES,
    LATTICE_NAMES,
    attach_kp_site_orders,
    build_kp_ladders,
    build_kp_region_masks_for_lattice,
    kagome_bond_pos,
    kp_ordering_bonds,
)
from src.kp.kp_tee_job import _parse_regions, _to_attr_value
from src.engines.qaqmc_renyi import QAQMCRenyiRydberg
from src.mpi.qaqmc_renyi_ratio_mpi import run_ratio_mpi
from src.mpi.reweighting_mpi import _rank_seed, run_expanded_mpi
from src.tee.compose_tee import (
    KPResult,
    compose_kp,
    save_kp_result_hdf5,
    summarize_region,
)
from src.tee.qaqmc_renyi_ratio import RegionRunResult
from src.tee.reweighting import save_expanded_result_hdf5


def _geometry_payload(spec) -> dict:
    return {
        "center_label": str(spec.center_label),
        "region_indices": {
            name: np.asarray(values, dtype=np.int32).reshape(-1).tolist()
            for name, values in spec.region_indices.items()
        },
        "region_sizes": {
            name: int(np.sum(np.asarray(mask, dtype=np.uint8)))
            for name, mask in spec.region_masks.items()
        },
        "site_orders": None if spec.site_orders is None else {
            name: np.asarray(values, dtype=np.int32).reshape(-1).tolist()
            for name, values in spec.site_orders.items()
        },
        "outer_paths": [[str(item) for item in path] for path in spec.outer_paths],
        "branch_paths": [[str(item) for item in path] for path in spec.branch_paths],
    }


def _write_geometry_json(path, *, spec, params: dict) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"params": dict(params), "geometry": _geometry_payload(spec)},
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out_path


def run_ratio_job_mpi(
    *,
    nx: int,
    ny: int,
    m: int,
    M: int,
    Omega: float,
    Rb: float,
    delta_min: float,
    delta_max: float,
    epsilon: float,
    seed: int,
    a: float,
    neighbor_cutoff: int | None,
    delta_groups: int,
    n_therm: int,
    n_measure: int,
    measure_stride: int,
    block_size: int | None,
    preferred_center_label: str | None,
    output_dir,
    lattice: str = DEFAULT_LATTICE,
    comm=None,
) -> dict | None:
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Rank 0 builds KP geometry then broadcasts the parts everyone needs.
    if rank == 0:
        pos = kagome_bond_pos(lattice, nx, ny, a=a)
        ordering_bonds = kp_ordering_bonds(lattice, nx, ny, a=a)
        spec = build_kp_region_masks_for_lattice(
            lattice, nx, ny, m=m, a=a,
            preferred_center_label=preferred_center_label,
        )
        spec = attach_kp_site_orders(spec, ordering_bonds)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ratio_steps").mkdir(parents=True, exist_ok=True)

        params = {
            "method": "ratio",
            "lattice": str(lattice),
            "nx": int(nx), "ny": int(ny), "m": int(m),
            "N": int(pos.shape[0]), "M": int(M),
            "Omega": float(Omega), "Rb": float(Rb),
            "delta_min": float(delta_min), "delta_max": float(delta_max),
            "epsilon": float(epsilon), "seed": int(seed), "a": float(a),
            "neighbor_cutoff": -1 if neighbor_cutoff is None else int(neighbor_cutoff),
            "delta_groups": int(delta_groups),
            "n_therm": int(n_therm), "n_measure": int(n_measure),
            "measure_stride": int(measure_stride),
            "block_size": -1 if block_size is None else int(block_size),
            "preferred_center_label": str(preferred_center_label) if preferred_center_label else "auto",
            "n_ranks": int(comm.Get_size()),
        }
        geometry_path = _write_geometry_json(out_dir / "kp_geometry.json",
                                             spec=spec, params=params)
    else:
        pos = None
        spec = None
        out_dir = None
        geometry_path = None

    pos = comm.bcast(pos, root=0)
    spec = comm.bcast(spec, root=0)
    out_dir = comm.bcast(out_dir, root=0)

    N = int(pos.shape[0])
    region_runs: dict[str, RegionRunResult] = {}
    for region_name in KP_REGION_NAMES:
        site_order = np.asarray(spec.site_orders[region_name], dtype=np.int32)
        current_mask = np.zeros(N, dtype=np.uint8)
        ratio_results = []
        for step_index, next_site in enumerate(site_order):
            step_out = None
            if rank == 0:
                step_out = str(out_dir / "ratio_steps"
                               / f"{region_name}_step{int(step_index):03d}.h5")
            payload = run_ratio_mpi(
                N=N, M=int(M),
                A_mask=current_mask, next_site=int(next_site),
                Omega=float(Omega), Rb=float(Rb),
                delta_min=float(delta_min), delta_max=float(delta_max),
                pos=pos, epsilon=float(epsilon), seed=int(seed),
                neighbor_cutoff=neighbor_cutoff, delta_groups=int(delta_groups),
                n_therm=int(n_therm), n_measure=int(n_measure),
                measure_stride=int(measure_stride), block_size=block_size,
                filepath=step_out, region_name=region_name,
                step_index=int(step_index), comm=comm, verbose=False,
            )
            if rank == 0:
                ratio_results.append(payload)
            current_mask = current_mask.copy()
            current_mask[int(next_site)] = 1

        if rank == 0:
            ratios = np.array([r.ratio for r in ratio_results], dtype=np.float64)
            ratio_errs = np.array([r.ratio_err for r in ratio_results], dtype=np.float64)
            summary = summarize_region(region_name, site_order, ratios, ratio_errs)
            region_runs[region_name] = RegionRunResult(
                region_name=region_name,
                site_order=site_order.copy(),
                ratio_results=ratio_results,
                summary=summary,
            )

    if rank != 0:
        return None

    # KP composition on rank 0
    summaries = {name: region_runs[name].summary for name in KP_REGION_NAMES}
    gamma, gamma_err = compose_kp(summaries)
    kp_result = KPResult(region_summaries=summaries, gamma=gamma, gamma_err=gamma_err)
    kp_path = out_dir / "kp_ratio_result.h5"
    save_kp_result_hdf5(kp_path, kp_result)
    return {
        "geometry_path": str(geometry_path),
        "kp_result_path": str(kp_path),
        "gamma": float(gamma),
        "gamma_err": float(gamma_err),
    }


def run_expanded_job_mpi(
    *,
    nx: int,
    ny: int,
    m: int,
    M: int,
    Omega: float,
    Rb: float,
    delta_min: float,
    delta_max: float,
    epsilon: float,
    seed: int,
    a: float,
    neighbor_cutoff: int | None,
    delta_groups: int,
    regions: list[str],
    preferred_center_label: str | None,
    output_dir,
    autotune_steps_per_iter: int,
    autotune_max_iters: int,
    autotune_tol: float,
    autotune_method: str,
    autotune_damping: float,
    n_steps: int | None,
    block_size: int | None,
    target_s2_err: float | None,
    batch_steps: int | None,
    max_steps: int | None,
    min_steps: int,
    estimator: str,
    lattice: str = DEFAULT_LATTICE,
    comm=None,
) -> dict | None:
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        pos = kagome_bond_pos(lattice, nx, ny, a=a)
        ordering_bonds = kp_ordering_bonds(lattice, nx, ny, a=a)
        spec = build_kp_region_masks_for_lattice(
            lattice, nx, ny, m=m, a=a,
            preferred_center_label=preferred_center_label,
        )
        ladders = build_kp_ladders(spec, bond_sites=ordering_bonds)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        common_params = {
            "method": "expanded",
            "lattice": str(lattice),
            "nx": int(nx), "ny": int(ny), "m": int(m),
            "N": int(pos.shape[0]), "M": int(M),
            "Omega": float(Omega), "Rb": float(Rb),
            "delta_min": float(delta_min), "delta_max": float(delta_max),
            "epsilon": float(epsilon), "seed": int(seed), "a": float(a),
            "neighbor_cutoff": -1 if neighbor_cutoff is None else int(neighbor_cutoff),
            "delta_groups": int(delta_groups),
            "preferred_center_label": str(preferred_center_label) if preferred_center_label else "auto",
            "autotune_steps_per_iter": int(autotune_steps_per_iter),
            "autotune_max_iters": int(autotune_max_iters),
            "autotune_tol": float(autotune_tol),
            "autotune_method": str(autotune_method),
            "autotune_damping": float(autotune_damping),
            "n_steps": -1 if n_steps is None else int(n_steps),
            "block_size": -1 if block_size is None else int(block_size),
            "target_s2_err": None if target_s2_err is None else float(target_s2_err),
            "batch_steps": -1 if batch_steps is None else int(batch_steps),
            "max_steps": -1 if max_steps is None else int(max_steps),
            "min_steps": int(min_steps),
            "estimator": str(estimator),
            "n_ranks": int(comm.Get_size()),
        }
        geometry_path = _write_geometry_json(
            out_dir / "kp_geometry.json",
            spec=spec,
            params={**common_params, "regions": list(regions)},
        )
    else:
        pos = None
        ladders = None
        out_dir = None
        common_params = None
        geometry_path = None

    pos = comm.bcast(pos, root=0)
    ladders = comm.bcast(ladders, root=0)
    out_dir = comm.bcast(out_dir, root=0)
    common_params = comm.bcast(common_params, root=0)

    N = int(pos.shape[0])
    outputs: dict = {"geometry_path": str(geometry_path)} if rank == 0 else {}

    # Build the C++ engine ONCE per rank and reuse across regions.  Each call
    # to run_expanded_mpi resets its ladder via set_ensemble_ladder, which is
    # cheap (~0.1s) compared with engine construction (~7 min for the probed
    # 6×6 kagome).  We share one engine via engine_factory.
    shared_engine = QAQMCRenyiRydberg(
        N=N, M=int(M),
        Omega=float(Omega), Rb=float(Rb),
        delta_min=float(delta_min), delta_max=float(delta_max),
        pos=pos, epsilon=float(epsilon),
        seed=_rank_seed(int(seed), rank),
        neighbor_cutoff=neighbor_cutoff,
        delta_groups=int(delta_groups),
    )

    def _shared_engine_factory(_local_rank):
        return shared_engine

    for region_name in regions:
        ladder = ladders[region_name]
        masks = [np.asarray(x, dtype=np.uint8) for x in ladder.masks]
        neighbors = [list(map(int, row)) for row in ladder.neighbors]
        target_ensemble = int(ladder.target_ensemble)

        # Drive MPI-distributed auto_tune + production for this region.  We pass
        # filepath=None here and write the HDF5 ourselves so we can attach the
        # region_name / target_ensemble into the params block uniformly.
        mpi_result = run_expanded_mpi(
            N=N, M=int(M), masks=masks, neighbors=neighbors,
            target_ensemble=target_ensemble,
            Omega=float(Omega), Rb=float(Rb),
            delta_min=float(delta_min), delta_max=float(delta_max),
            pos=pos, epsilon=float(epsilon),
            seed=int(seed) + 7919 * hash(region_name) % 997,  # ignored when factory supplies engine
            neighbor_cutoff=neighbor_cutoff, delta_groups=int(delta_groups),
            initial_ensemble=0,
            autotune_steps_per_iter=int(autotune_steps_per_iter),
            autotune_max_iters=int(autotune_max_iters),
            autotune_tol=float(autotune_tol),
            autotune_method=str(autotune_method),
            autotune_damping=float(autotune_damping),
            n_steps=n_steps, block_size=block_size,
            target_s2_err=target_s2_err, batch_steps=batch_steps,
            max_steps=max_steps, min_steps=int(min_steps),
            estimator=str(estimator),
            reference_ensemble=0,
            filepath=None,
            comm=comm,
            engine_factory=_shared_engine_factory,
        )

        if rank != 0:
            continue

        region_path = out_dir / f"kp_expanded_{region_name}.h5"
        save_expanded_result_hdf5(
            region_path,
            masks=masks,
            neighbors=neighbors,
            result=mpi_result.production,
            auto_tune=mpi_result.auto_tune,
            params={
                key: _to_attr_value(value)
                for key, value in {
                    **common_params,
                    "region_name": str(region_name),
                    "target_ensemble": target_ensemble,
                }.items()
            },
            pos=pos,
        )
        if str(estimator).strip().lower() == "collection":
            s2, s2_err = mpi_result.production.s2_collection(target_ensemble)
        else:
            s2, s2_err = mpi_result.production.s2(target_ensemble)
        outputs[str(region_name)] = {
            "path": str(region_path),
            "s2": float(s2),
            "s2_err": float(s2_err),
        }

    return outputs if rank == 0 else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MPI-aware KP TEE jobs on kagome")
    parser.add_argument("--method", choices=["ratio", "expanded"], required=True)
    parser.add_argument("--lattice", choices=list(LATTICE_NAMES), default=DEFAULT_LATTICE)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--Omega", type=float, default=1.0)
    parser.add_argument("--Rb", type=float, default=1.2)
    parser.add_argument("--delta_min", type=float, default=0.0)
    parser.add_argument("--delta_max", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neighbor_cutoff", type=int, default=-1)
    parser.add_argument("--delta_groups", type=int, default=0)
    parser.add_argument("--preferred_center_label", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--n_therm", type=int, default=2000)
    parser.add_argument("--n_measure", type=int, default=50000)
    parser.add_argument("--measure_stride", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=-1)

    parser.add_argument("--regions", type=str, default="")
    parser.add_argument("--autotune_steps_per_iter", type=int, default=15000)
    parser.add_argument("--autotune_max_iters", type=int, default=8)
    parser.add_argument("--autotune_tol", type=float, default=1.15)
    parser.add_argument("--autotune_method", type=str, default="transition_matrix")
    parser.add_argument("--autotune_damping", type=float, default=0.7)
    parser.add_argument("--n_steps", type=int, default=-1)
    parser.add_argument("--target_s2_err", type=float, default=-1.0)
    parser.add_argument("--batch_steps", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--min_steps", type=int, default=0)
    parser.add_argument("--estimator", type=str, default="collection")
    return parser


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    parser = build_parser()
    args = parser.parse_args()

    neighbor_cutoff = None if int(args.neighbor_cutoff) < 0 else int(args.neighbor_cutoff)
    block_size = None if int(args.block_size) < 0 else int(args.block_size)
    preferred = None if str(args.preferred_center_label).strip().lower() in {"", "auto", "none"} else args.preferred_center_label

    if args.method == "ratio":
        payload = run_ratio_job_mpi(
            nx=args.nx, ny=args.ny, m=args.m, M=args.M,
            Omega=args.Omega, Rb=args.Rb,
            delta_min=args.delta_min, delta_max=args.delta_max,
            epsilon=args.epsilon, seed=args.seed, a=args.a,
            neighbor_cutoff=neighbor_cutoff,
            delta_groups=int(args.delta_groups),
            n_therm=args.n_therm, n_measure=args.n_measure,
            measure_stride=args.measure_stride, block_size=block_size,
            preferred_center_label=preferred,
            output_dir=args.output_dir,
            lattice=args.lattice,
            comm=comm,
        )
    else:
        n_steps = None if int(args.n_steps) < 0 else int(args.n_steps)
        target_s2_err = None if float(args.target_s2_err) < 0.0 else float(args.target_s2_err)
        batch_steps = None if int(args.batch_steps) < 0 else int(args.batch_steps)
        max_steps = None if int(args.max_steps) < 0 else int(args.max_steps)
        payload = run_expanded_job_mpi(
            nx=args.nx, ny=args.ny, m=args.m, M=args.M,
            Omega=args.Omega, Rb=args.Rb,
            delta_min=args.delta_min, delta_max=args.delta_max,
            epsilon=args.epsilon, seed=args.seed, a=args.a,
            neighbor_cutoff=neighbor_cutoff,
            delta_groups=int(args.delta_groups),
            regions=_parse_regions(args.regions),
            preferred_center_label=preferred,
            output_dir=args.output_dir,
            autotune_steps_per_iter=args.autotune_steps_per_iter,
            autotune_max_iters=args.autotune_max_iters,
            autotune_tol=args.autotune_tol,
            autotune_method=args.autotune_method,
            autotune_damping=args.autotune_damping,
            n_steps=n_steps, block_size=block_size,
            target_s2_err=target_s2_err, batch_steps=batch_steps,
            max_steps=max_steps, min_steps=args.min_steps,
            estimator=args.estimator,
            lattice=args.lattice,
            comm=comm,
        )

    if rank == 0 and payload is not None:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
