"""Backwards-compatible dispatcher for KP TEE jobs.

The ratio and expanded methods now live in dedicated entry-point modules:

- ``src.mpi.kp_tee_ratio_mpi``
- ``src.mpi.kp_tee_expanded_mpi``

This module preserves the legacy ``--method {ratio,expanded}`` interface used
by older Slurm scripts.  New scripts should call the dedicated entry points
directly so their CLI surface stays method-specific.

Usage (legacy, still supported)::

    mpiexec -n <NRANKS> python -m src.mpi.kp_tee_job_mpi \
        --method ratio   --nx 8 --ny 8 ... --output_dir data/...
    mpiexec -n <NRANKS> python -m src.mpi.kp_tee_job_mpi \
        --method expanded --nx 4 --ny 4 ... --output_dir data/...
"""

from __future__ import annotations

import argparse
import json

try:
    from mpi4py import MPI
except ImportError as exc:  # pragma: no cover
    raise ImportError("mpi4py is required for src.mpi.kp_tee_job_mpi") from exc

from src.kp.kp_tee_job import _parse_regions
from src.mpi.kp_tee_common import (
    _resolve_total_per_rank,
    add_common_args,
    normalize_common_args,
)

# Re-export the worker functions for any caller that still imports them from
# this module.  The implementations live in the dedicated entry points.
from src.mpi.kp_tee_ratio_mpi import run_ratio_job_mpi  # noqa: F401
from src.mpi.kp_tee_expanded_mpi import run_expanded_job_mpi  # noqa: F401


def build_parser() -> argparse.ArgumentParser:
    """Union parser supporting both ratio and expanded methods.

    Mirrors the historic flag set so existing Slurm jobs invoking
    ``python -m src.mpi.kp_tee_job_mpi --method ...`` keep working unchanged.
    """
    parser = argparse.ArgumentParser(
        description="MPI-aware KP TEE jobs on kagome (legacy dispatcher; "
                    "prefer kp_tee_ratio_mpi / kp_tee_expanded_mpi for new scripts)"
    )
    parser.add_argument("--method", choices=["ratio", "expanded"], required=True)
    add_common_args(parser)

    # ratio-specific knobs (ignored for --method expanded)
    parser.add_argument("--n_therm", type=int, default=2000)
    parser.add_argument("--n_measure", type=int, default=50000)
    parser.add_argument("--n_measure_total", type=int, default=-1,
                        help="total measurement sweeps across ranks; auto-divided per rank")
    parser.add_argument("--measure_stride", type=int, default=1)

    # expanded-specific knobs (ignored for --method ratio)
    parser.add_argument("--regions", type=str, default="")
    parser.add_argument("--autotune_steps_per_iter", type=int, default=15000)
    parser.add_argument("--autotune_max_iters", type=int, default=8)
    parser.add_argument("--autotune_tol", type=float, default=1.15)
    parser.add_argument("--autotune_method", type=str, default="transition_matrix")
    parser.add_argument("--autotune_damping", type=float, default=0.7)
    parser.add_argument("--n_steps", type=int, default=-1)
    parser.add_argument("--n_steps_total", type=int, default=-1,
                        help="total production sweeps across ranks; auto-divided per rank")
    parser.add_argument("--target_s2_err", type=float, default=-1.0)
    parser.add_argument("--batch_steps", type=int, default=-1)
    parser.add_argument("--batch_steps_total", type=int, default=-1,
                        help="total adaptive batch sweeps across ranks")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_steps_total", type=int, default=-1,
                        help="total adaptive cap across ranks")
    parser.add_argument("--min_steps", type=int, default=0)
    parser.add_argument("--min_steps_total", type=int, default=-1,
                        help="total adaptive minimum across ranks")
    parser.add_argument("--estimator", type=str, default="collection")
    return parser


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    parser = build_parser()
    args = parser.parse_args()

    common = normalize_common_args(args)

    if args.method == "ratio":
        n_measure = _resolve_total_per_rank(
            args.n_measure, args.n_measure_total, n_ranks, name="n_measure"
        )
        if rank == 0 and int(args.n_measure_total) > 0:
            print(f"[kp_tee_job_mpi] n_measure_total={args.n_measure_total} → "
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
    else:
        n_steps_resolved = _resolve_total_per_rank(
            args.n_steps, args.n_steps_total, n_ranks, name="n_steps"
        )
        batch_steps_resolved = _resolve_total_per_rank(
            args.batch_steps, args.batch_steps_total, n_ranks, name="batch_steps"
        )
        max_steps_resolved = _resolve_total_per_rank(
            args.max_steps, args.max_steps_total, n_ranks, name="max_steps"
        )
        min_steps_resolved = _resolve_total_per_rank(
            args.min_steps, args.min_steps_total, n_ranks, name="min_steps", sentinel=0
        )
        if rank == 0:
            for label, total, per_rank in [
                ("n_steps", args.n_steps_total, n_steps_resolved),
                ("batch_steps", args.batch_steps_total, batch_steps_resolved),
                ("max_steps", args.max_steps_total, max_steps_resolved),
                ("min_steps", args.min_steps_total, min_steps_resolved),
            ]:
                if int(total) > 0:
                    print(f"[kp_tee_job_mpi] {label}_total={total} → "
                          f"{per_rank}/rank × {n_ranks} ranks")
        n_steps = None if n_steps_resolved < 0 else n_steps_resolved
        target_s2_err = None if float(args.target_s2_err) < 0.0 else float(args.target_s2_err)
        batch_steps = None if batch_steps_resolved < 0 else batch_steps_resolved
        max_steps = None if max_steps_resolved < 0 else max_steps_resolved
        payload = run_expanded_job_mpi(
            nx=args.nx, ny=args.ny, m=args.m, M=args.M,
            Omega=args.Omega, Rb=args.Rb,
            delta_min=args.delta_min, delta_max=args.delta_max,
            epsilon=args.epsilon, seed=args.seed, a=args.a,
            neighbor_cutoff=common["neighbor_cutoff"],
            delta_groups=int(args.delta_groups),
            regions=_parse_regions(args.regions),
            preferred_center_label=common["preferred_center_label"],
            output_dir=args.output_dir,
            autotune_steps_per_iter=args.autotune_steps_per_iter,
            autotune_max_iters=args.autotune_max_iters,
            autotune_tol=args.autotune_tol,
            autotune_method=args.autotune_method,
            autotune_damping=args.autotune_damping,
            n_steps=n_steps, block_size=common["block_size"],
            target_s2_err=target_s2_err, batch_steps=batch_steps,
            max_steps=max_steps, min_steps=min_steps_resolved,
            estimator=args.estimator,
            lattice=args.lattice,
            comm=comm,
        )

    if rank == 0 and payload is not None:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
