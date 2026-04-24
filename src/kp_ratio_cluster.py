"""
Cluster helpers for KP ratio-estimator jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.compose_tee import KPResult, aggregate_ratios, compose_kp, save_kp_result_hdf5
from src.kp_geometry import KP_REGION_NAMES, attach_kp_site_orders, build_kp_region_masks
from src.lattices import generate_kagome_bond_lattice
from src.qaqmc_renyi_ratio import (
    build_ratio_job_specs,
    load_ratio_manifest_hdf5,
    save_ratio_manifest_hdf5,
)


def _to_attr_value(value):
    if isinstance(value, (str, bytes, bool, int, float, np.number)):
        return value
    if value is None:
        return "None"
    return json.dumps(value)


def _to_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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


def _write_json(path: str | Path, payload: dict) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def build_kp_ratio_manifest(
    *,
    nx: int,
    ny: int,
    m: int,
    a: float = 1.0,
    preferred_center_label: str = "C35",
    output_dir: str | Path,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bond_sites = generate_kagome_bond_lattice(nx, ny, a=a)
    spec = build_kp_region_masks(nx, ny, m=m, a=a, preferred_center_label=preferred_center_label)
    spec = attach_kp_site_orders(spec, bond_sites)

    ratio_dir = out_dir / "ratio_steps"
    jobs = []
    region_job_counts = {}
    for region_name in KP_REGION_NAMES:
        region_jobs = build_ratio_job_specs(
            region_name,
            spec.region_masks[region_name],
            site_order=spec.site_orders[region_name],
            output_dir=ratio_dir,
            filename_template="{region}_step{step:03d}.h5",
        )
        jobs.extend(region_jobs)
        region_job_counts[region_name] = len(region_jobs)

    manifest_path = out_dir / "ratio_manifest.h5"
    params = {
        "nx": int(nx),
        "ny": int(ny),
        "m": int(m),
        "a": float(a),
        "preferred_center_label": str(preferred_center_label),
        "center_label": str(spec.center_label),
        "n_jobs": int(len(jobs)),
    }
    save_ratio_manifest_hdf5(
        manifest_path,
        jobs,
        params={key: _to_attr_value(value) for key, value in params.items()},
    )

    geometry_path = _write_json(
        out_dir / "kp_geometry.json",
        {
            "params": params,
            "geometry": _geometry_payload(spec),
        },
    )
    summary_path = _write_json(
        out_dir / "ratio_manifest_summary.json",
        {
            "manifest_path": str(manifest_path),
            "geometry_path": str(geometry_path),
            "n_jobs": int(len(jobs)),
            "region_job_counts": region_job_counts,
        },
    )
    return {
        "manifest_path": str(manifest_path),
        "geometry_path": str(geometry_path),
        "summary_path": str(summary_path),
        "n_jobs": int(len(jobs)),
        "region_job_counts": region_job_counts,
    }


def count_manifest_jobs(manifest_path: str | Path) -> int:
    loaded = load_ratio_manifest_hdf5(manifest_path)
    return int(len(loaded["jobs"]))


def run_manifest_task(
    *,
    manifest_path: str | Path,
    task_id: int,
    M: int,
    Omega: float,
    Rb: float,
    delta_min: float,
    delta_max: float,
    epsilon: float,
    seed: int,
    neighbor_cutoff: int | None,
    n_therm: int,
    n_measure: int,
    measure_stride: int = 1,
    block_size: int | None = None,
    verbose: bool = False,
):
    loaded = load_ratio_manifest_hdf5(manifest_path)
    params = dict(loaded["params"])
    jobs = loaded["jobs"]
    task_index = int(task_id)
    if task_index < 0 or task_index >= len(jobs):
        raise IndexError(f"task_id {task_index} out of range for {len(jobs)} jobs")
    job = jobs[task_index]

    nx = int(params["nx"])
    ny = int(params["ny"])
    a = float(params["a"])
    pos = generate_kagome_bond_lattice(nx, ny, a=a)

    from src.qaqmc_renyi_ratio_mpi import run_ratio_job

    return run_ratio_job(
        job,
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
        n_therm=int(n_therm),
        n_measure=int(n_measure),
        measure_stride=int(measure_stride),
        block_size=block_size,
        filepath=job.output_path,
        verbose=bool(verbose),
    )


def collect_manifest_results(
    *,
    manifest_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict:
    loaded = load_ratio_manifest_hdf5(manifest_path)
    jobs = loaded["jobs"]
    params = dict(loaded["params"])
    manifest = Path(manifest_path)
    out_dir = manifest.parent if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    hdf5_paths = []
    for job in jobs:
        if job.output_path is None:
            raise ValueError("all jobs in the manifest must carry output_path for collection")
        path = Path(job.output_path)
        if not path.exists():
            missing.append(str(path))
        else:
            hdf5_paths.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} ratio outputs are missing; first missing path: {missing[0]}"
        )

    summaries = aggregate_ratios(hdf5_paths)
    gamma, gamma_err = compose_kp(summaries)
    kp_result = KPResult(region_summaries=summaries, gamma=gamma, gamma_err=gamma_err)
    kp_path = out_dir / "kp_result.h5"
    save_kp_result_hdf5(kp_path, kp_result)

    summary_json = _write_json(
        out_dir / "kp_result_summary.json",
        _to_json_safe({
            "manifest_path": str(manifest),
            "params": params,
            "gamma": float(gamma),
            "gamma_err": float(gamma_err),
            "regions": {
                name: {
                    "n_sites": int(summary.sites_ordered.size),
                    "S_2": float(summary.S_2),
                    "S_2_err": float(summary.S_2_err),
                }
                for name, summary in summaries.items()
            },
        }),
    )
    return {
        "kp_result_path": str(kp_path),
        "summary_path": str(summary_json),
        "gamma": float(gamma),
        "gamma_err": float(gamma_err),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KP ratio-estimator cluster helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-manifest")
    p_build.add_argument("--nx", type=int, required=True)
    p_build.add_argument("--ny", type=int, required=True)
    p_build.add_argument("--m", type=int, required=True)
    p_build.add_argument("--a", type=float, default=1.0)
    p_build.add_argument("--preferred_center_label", type=str, default="C35")
    p_build.add_argument("--output_dir", type=str, required=True)

    p_count = sub.add_parser("count-jobs")
    p_count.add_argument("--manifest", type=str, required=True)

    p_run = sub.add_parser("run-task")
    p_run.add_argument("--manifest", type=str, required=True)
    p_run.add_argument("--task_id", type=int, required=True)
    p_run.add_argument("--M", type=int, required=True)
    p_run.add_argument("--Omega", type=float, default=1.0)
    p_run.add_argument("--Rb", type=float, default=1.2)
    p_run.add_argument("--delta_min", type=float, default=0.0)
    p_run.add_argument("--delta_max", type=float, default=1.0)
    p_run.add_argument("--epsilon", type=float, default=0.01)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--neighbor_cutoff", type=int, default=-1)
    p_run.add_argument("--n_therm", type=int, required=True)
    p_run.add_argument("--n_measure", type=int, required=True)
    p_run.add_argument("--measure_stride", type=int, default=1)
    p_run.add_argument("--block_size", type=int, default=-1)
    p_run.add_argument("--verbose", action="store_true")

    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--manifest", type=str, required=True)
    p_collect.add_argument("--output_dir", type=str, default="")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-manifest":
        payload = build_kp_ratio_manifest(
            nx=args.nx,
            ny=args.ny,
            m=args.m,
            a=args.a,
            preferred_center_label=args.preferred_center_label,
            output_dir=args.output_dir,
        )
    elif args.command == "count-jobs":
        payload = {"n_jobs": count_manifest_jobs(args.manifest)}
    elif args.command == "run-task":
        payload = run_manifest_task(
            manifest_path=args.manifest,
            task_id=args.task_id,
            M=args.M,
            Omega=args.Omega,
            Rb=args.Rb,
            delta_min=args.delta_min,
            delta_max=args.delta_max,
            epsilon=args.epsilon,
            seed=args.seed,
            neighbor_cutoff=None if int(args.neighbor_cutoff) < 0 else int(args.neighbor_cutoff),
            n_therm=args.n_therm,
            n_measure=args.n_measure,
            measure_stride=args.measure_stride,
            block_size=None if int(args.block_size) < 0 else int(args.block_size),
            verbose=args.verbose,
        )
        if payload is not None:
            payload = {
                "ratio": float(payload.ratio),
                "ratio_err": float(payload.ratio_err),
                "visit_count_low": int(payload.visit_count_low),
                "visit_count_high": int(payload.visit_count_high),
            }
    else:
        payload = collect_manifest_results(
            manifest_path=args.manifest,
            output_dir=None if args.output_dir == "" else args.output_dir,
        )

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
