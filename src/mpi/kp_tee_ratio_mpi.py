"""MPI entry point for the *ratio* KP TEE method on the kagome bond lattice.

Each region's S2 is built from a sequence of independent ratio MC runs (one per
site added to the swap region).  No autotune / log_g machinery is used here.

Usage::

    mpiexec -n <NRANKS> python -m src.mpi.kp_tee_ratio_mpi \
        --nx 8 --ny 8 --m 2 --M 500000 \
        --a 4.0 --Rb 2.4 --Omega 1.0 --delta_min -2 --delta_max 4.5 \
        --delta_groups 600 --neighbor_cutoff 3 --seed 42 \
        --n_therm 2000 --n_measure 50000 --measure_stride 1 --block_size 500 \
        --output_dir data/kp_ratio_8x8_run

The legacy ``python -m src.mpi.kp_tee_job_mpi --method ratio ...`` invocation
still works via a thin compat shim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from mpi4py import MPI
except ImportError as exc:  # pragma: no cover
    raise ImportError("mpi4py is required for src.mpi.kp_tee_ratio_mpi") from exc

from src.kp.kp_geometry import (
    DEFAULT_LATTICE,
    KP_REGION_NAMES,
    attach_kp_site_orders,
    build_kp_region_masks_for_lattice,
    kagome_bond_pos,
    kp_ordering_bonds,
)
from src.mpi.kp_tee_common import (
    _resolve_total_per_rank,
    _write_geometry_json,
    add_common_args,
    normalize_common_args,
)
from src.mpi.qaqmc_renyi_ratio_mpi import run_ratio_mpi
from src.tee.compose_tee import (
    KPResult,
    compose_kp,
    save_kp_result_hdf5,
    summarize_region,
)
from src.tee.qaqmc_renyi_ratio import RegionRunResult


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MPI ratio-method KP TEE on kagome bond lattice")
    add_common_args(parser)
    # ratio-specific knobs
    parser.add_argument("--n_therm", type=int, default=2000)
    parser.add_argument("--n_measure", type=int, default=50000)
    parser.add_argument("--n_measure_total", type=int, default=-1,
                        help="total measurement sweeps across ranks; auto-divided per rank")
    parser.add_argument("--measure_stride", type=int, default=1)
    return parser


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    parser = build_parser()
    args = parser.parse_args()

    common = normalize_common_args(args)
    n_measure = _resolve_total_per_rank(
        args.n_measure, args.n_measure_total, n_ranks, name="n_measure"
    )
    if rank == 0 and int(args.n_measure_total) > 0:
        print(f"[kp_tee_ratio_mpi] n_measure_total={args.n_measure_total} → "
              f"{n_measure}/rank × {n_ranks} ranks")

    payload = run_ratio_job_mpi(
        nx=args.nx, ny=args.ny, m=args.m, M=args.M,
        Omega=args.Omega, Rb=args.Rb,
        delta_min=args.delta_min, delta_max=args.delta_max,
        epsilon=args.epsilon, seed=args.seed, a=args.a,
        neighbor_cutoff=common["neighbor_cutoff"],
        delta_groups=int(args.delta_groups),
        n_therm=args.n_therm, n_measure=n_measure,
        measure_stride=args.measure_stride,
        block_size=common["block_size"],
        preferred_center_label=common["preferred_center_label"],
        output_dir=args.output_dir,
        lattice=args.lattice,
        comm=comm,
    )

    if rank == 0 and payload is not None:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
