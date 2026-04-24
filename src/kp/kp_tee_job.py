"""
CLI helpers for launching kagome KP TEE jobs from local runs or Slurm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.tee.compose_tee import save_kp_result_hdf5
from src.kp.kp_geometry import KP_REGION_NAMES
from src.kp.kp_workflows import KagomeKPExpandedWorkflow, KagomeKPRatioWorkflow
from src.rydberg.lattices import generate_kagome_bond_lattice
from src.tee.reweighting import save_expanded_result_hdf5


def _to_attr_value(value):
    if isinstance(value, (str, bytes, bool, int, float, np.number)):
        return value
    if value is None:
        return "None"
    return json.dumps(value)


def _parse_regions(raw: str | None) -> list[str]:
    if raw is None or str(raw).strip() == "":
        return list(KP_REGION_NAMES)
    regions = []
    seen = set()
    for item in str(raw).split(","):
        region = item.strip().upper()
        if not region:
            continue
        if region not in KP_REGION_NAMES:
            raise ValueError(f"unknown KP region: {region}")
        if region not in seen:
            seen.add(region)
            regions.append(region)
    if not regions:
        raise ValueError("at least one region must be selected")
    return regions


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


def _write_geometry_json(path: str | Path, *, spec, params: dict) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": dict(params),
        "geometry": _geometry_payload(spec),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def run_ratio_job(
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
    n_therm: int,
    n_measure: int,
    measure_stride: int,
    block_size: int | None,
    preferred_center_label: str,
    output_dir: str | Path,
    workflow: KagomeKPRatioWorkflow | None = None,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos = generate_kagome_bond_lattice(nx, ny, a=a)
    if workflow is None:
        workflow = KagomeKPRatioWorkflow(
            N=int(pos.shape[0]),
            M=int(M),
            Omega=float(Omega),
            Rb=float(Rb),
            delta_min=float(delta_min),
            delta_max=float(delta_max),
            pos=pos,
            epsilon=float(epsilon),
            seed=int(seed),
            neighbor_cutoff=neighbor_cutoff,
        )

    run = workflow.run(
        nx=nx,
        ny=ny,
        m=m,
        bond_sites=pos,
        n_therm=n_therm,
        n_measure=n_measure,
        a=a,
        preferred_center_label=preferred_center_label,
        measure_stride=measure_stride,
        block_size=block_size,
    )

    params = {
        "method": "ratio",
        "nx": int(nx),
        "ny": int(ny),
        "m": int(m),
        "N": int(pos.shape[0]),
        "M": int(M),
        "Omega": float(Omega),
        "Rb": float(Rb),
        "delta_min": float(delta_min),
        "delta_max": float(delta_max),
        "epsilon": float(epsilon),
        "seed": int(seed),
        "a": float(a),
        "neighbor_cutoff": -1 if neighbor_cutoff is None else int(neighbor_cutoff),
        "n_therm": int(n_therm),
        "n_measure": int(n_measure),
        "measure_stride": int(measure_stride),
        "block_size": -1 if block_size is None else int(block_size),
        "preferred_center_label": str(preferred_center_label),
    }
    geometry_path = _write_geometry_json(out_dir / "kp_geometry.json", spec=run.spec, params=params)
    kp_path = out_dir / "kp_ratio_result.h5"
    save_kp_result_hdf5(kp_path, run.result.kp_result)
    return {
        "geometry_path": str(geometry_path),
        "kp_result_path": str(kp_path),
        "gamma": float(run.result.kp_result.gamma),
        "gamma_err": float(run.result.kp_result.gamma_err),
    }


def run_expanded_job(
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
    regions: list[str],
    preferred_center_label: str,
    output_dir: str | Path,
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
    workflow: KagomeKPExpandedWorkflow | None = None,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos = generate_kagome_bond_lattice(nx, ny, a=a)
    if workflow is None:
        workflow = KagomeKPExpandedWorkflow(
            N=int(pos.shape[0]),
            M=int(M),
            Omega=float(Omega),
            Rb=float(Rb),
            delta_min=float(delta_min),
            delta_max=float(delta_max),
            pos=pos,
            epsilon=float(epsilon),
            seed=int(seed),
            neighbor_cutoff=neighbor_cutoff,
        )

    outputs = {}
    geometry_written = False
    common_params = {
        "method": "expanded",
        "nx": int(nx),
        "ny": int(ny),
        "m": int(m),
        "N": int(pos.shape[0]),
        "M": int(M),
        "Omega": float(Omega),
        "Rb": float(Rb),
        "delta_min": float(delta_min),
        "delta_max": float(delta_max),
        "epsilon": float(epsilon),
        "seed": int(seed),
        "a": float(a),
        "neighbor_cutoff": -1 if neighbor_cutoff is None else int(neighbor_cutoff),
        "preferred_center_label": str(preferred_center_label),
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
    }

    for region_name in regions:
        result = workflow.run_region(
            region_name,
            nx=nx,
            ny=ny,
            m=m,
            bond_sites=pos,
            a=a,
            preferred_center_label=preferred_center_label,
            autotune_steps_per_iter=autotune_steps_per_iter,
            autotune_max_iters=autotune_max_iters,
            autotune_tol=autotune_tol,
            autotune_method=autotune_method,
            autotune_damping=autotune_damping,
            n_steps=n_steps,
            block_size=block_size,
            target_s2_err=target_s2_err,
            batch_steps=batch_steps,
            max_steps=max_steps,
            min_steps=min_steps,
            estimator=estimator,
        )

        if not geometry_written:
            geometry_path = _write_geometry_json(
                out_dir / "kp_geometry.json",
                spec=result.spec,
                params={**common_params, "regions": list(regions)},
            )
            outputs["geometry_path"] = str(geometry_path)
            geometry_written = True

        region_path = out_dir / f"kp_expanded_{region_name}.h5"
        save_expanded_result_hdf5(
            region_path,
            masks=result.ladder.masks,
            neighbors=result.ladder.neighbors,
            result=result.production,
            auto_tune=result.autotune,
            params={
                key: _to_attr_value(value)
                for key, value in {
                    **common_params,
                    "region_name": str(region_name),
                    "target_ensemble": int(result.ladder.target_ensemble),
                }.items()
            },
            pos=pos,
        )
        if str(estimator).strip().lower() == "collection":
            s2, s2_err = result.production.s2_collection(result.ladder.target_ensemble)
        else:
            s2, s2_err = result.production.s2(result.ladder.target_ensemble)
        outputs[str(region_name)] = {
            "path": str(region_path),
            "s2": float(s2),
            "s2_err": float(s2_err),
        }

    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run kagome KP TEE jobs")
    parser.add_argument("--method", choices=["ratio", "expanded"], required=True)
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
    parser.add_argument("--preferred_center_label", type=str, default="C35")
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
    parser = build_parser()
    args = parser.parse_args()

    neighbor_cutoff = None if int(args.neighbor_cutoff) < 0 else int(args.neighbor_cutoff)
    block_size = None if int(args.block_size) < 0 else int(args.block_size)

    if args.method == "ratio":
        payload = run_ratio_job(
            nx=args.nx,
            ny=args.ny,
            m=args.m,
            M=args.M,
            Omega=args.Omega,
            Rb=args.Rb,
            delta_min=args.delta_min,
            delta_max=args.delta_max,
            epsilon=args.epsilon,
            seed=args.seed,
            a=args.a,
            neighbor_cutoff=neighbor_cutoff,
            n_therm=args.n_therm,
            n_measure=args.n_measure,
            measure_stride=args.measure_stride,
            block_size=block_size,
            preferred_center_label=args.preferred_center_label,
            output_dir=args.output_dir,
        )
    else:
        n_steps = None if int(args.n_steps) < 0 else int(args.n_steps)
        target_s2_err = None if float(args.target_s2_err) < 0.0 else float(args.target_s2_err)
        batch_steps = None if int(args.batch_steps) < 0 else int(args.batch_steps)
        max_steps = None if int(args.max_steps) < 0 else int(args.max_steps)
        payload = run_expanded_job(
            nx=args.nx,
            ny=args.ny,
            m=args.m,
            M=args.M,
            Omega=args.Omega,
            Rb=args.Rb,
            delta_min=args.delta_min,
            delta_max=args.delta_max,
            epsilon=args.epsilon,
            seed=args.seed,
            a=args.a,
            neighbor_cutoff=neighbor_cutoff,
            regions=_parse_regions(args.regions),
            preferred_center_label=args.preferred_center_label,
            output_dir=args.output_dir,
            autotune_steps_per_iter=args.autotune_steps_per_iter,
            autotune_max_iters=args.autotune_max_iters,
            autotune_tol=args.autotune_tol,
            autotune_method=args.autotune_method,
            autotune_damping=args.autotune_damping,
            n_steps=n_steps,
            block_size=block_size,
            target_s2_err=target_s2_err,
            batch_steps=batch_steps,
            max_steps=max_steps,
            min_steps=args.min_steps,
            estimator=args.estimator,
        )

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
