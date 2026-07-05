"""MPI entry point for the *expanded* KP TEE method on the kagome bond lattice.

Each region's S2 is computed via expanded-ensemble reweighting: a single Markov
chain traverses a ladder of ensembles indexed by progressively-larger swap
masks; ``log_g`` biases keep visits balanced, then jackknife reweighting
recovers ``log Z[k] / Z[0]`` (= log Tr(rho^2_X) at the top of the ladder).

Usage::

    mpiexec -n <NRANKS> python -m src.mpi.kp_tee_expanded_mpi \
        --nx 4 --ny 4 --m 1 --M 871000 \
        --a 4.0 --Rb 2.4 --Omega 1.0 --delta_min -2 --delta_max 4.5 \
        --delta_groups 600 --neighbor_cutoff 3 --seed 42 \
        --regions A,B,C,AB,BC,CA,ABC \
        --autotune_max_iters 8 --autotune_steps_per_iter 15000 \
        --n_steps_total 500000 --block_size 2000 --estimator collection \
        --output_dir data/kp_expanded_4x4_m1

The legacy ``python -m src.mpi.kp_tee_job_mpi --method expanded ...``
invocation still works via a thin compat shim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from mpi4py import MPI
except ImportError as exc:  # pragma: no cover
    raise ImportError("mpi4py is required for src.mpi.kp_tee_expanded_mpi") from exc

from src.engines.qaqmc_renyi import QAQMCRenyiRydberg
from src.kp.kp_geometry import (
    DEFAULT_LATTICE,
    build_kp_ladders,
    build_kp_region_masks_for_lattice,
    kagome_bond_pos,
    kp_ordering_bonds,
)
from src.kp.kp_tee_job import _parse_regions, _to_attr_value
from src.mpi.kp_tee_common import (
    _resolve_total_per_rank,
    _write_geometry_json,
    add_common_args,
    normalize_common_args,
)
from src.mpi.reweighting_mpi import _rank_seed, run_expanded_mpi
from src.tee.log_g_io import load_region_log_g
from src.tee.reweighting import save_expanded_result_hdf5


# Default warm-up when log_g is loaded from disk and the user didn't override.
# Picked so the chain has time to leave the C++ constructor's fresh state but
# remains a small fraction of typical production budgets (~62k/rank).
DEFAULT_WARM_UP_WHEN_LOADING = 5000


def validate_mode_args(args) -> None:
    """Reject incoherent combinations of the three Day-3 mode flags.

    - ``--skip_autotune`` requires ``--log_g_init`` (frozen production needs
      tuned weights to freeze).
    - ``--warm_up_steps`` must be non-negative.
    """
    if getattr(args, "skip_autotune", False) and not getattr(args, "log_g_init", None):
        raise ValueError("--skip_autotune requires --log_g_init <dir>")
    warm = int(getattr(args, "warm_up_steps", 0))
    if warm < 0:
        raise ValueError("--warm_up_steps must be non-negative")


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
    log_g_init: str | None = None,
    skip_autotune: bool = False,
    warm_up_steps: int = 0,
    checkpoint_every_blocks: int = 0,
) -> dict | None:
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if skip_autotune and not log_g_init:
        raise ValueError("skip_autotune requires log_g_init to be provided")

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
            "log_g_init": "" if log_g_init is None else str(log_g_init),
            "skip_autotune": bool(skip_autotune),
            "warm_up_steps": int(warm_up_steps),
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

        # Resolve per-region log_g if --log_g_init was given.  Rank 0 reads
        # the file (with sanity checks against this region's mask ladder)
        # then broadcasts to all ranks.
        region_log_g = None
        if log_g_init is not None:
            if rank == 0:
                src_path = Path(log_g_init) / f"kp_expanded_{region_name}.h5"
                expected = np.stack([np.asarray(m, dtype=np.uint8) for m in masks], axis=0)
                expected_params_check = {
                    "M": int(M),
                    "lattice": str(lattice),
                    "region_name": str(region_name),
                }
                region_log_g = load_region_log_g(
                    src_path, expected_masks=expected,
                    expected_params=expected_params_check,
                )
                print(f"[kp_tee_expanded_mpi] loaded log_g for {region_name} "
                      f"from {src_path} (n_windows={region_log_g.size})", flush=True)
            region_log_g = comm.bcast(region_log_g, root=0)

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
            initial_log_g=region_log_g,
            skip_autotune=bool(skip_autotune),
            warm_up_steps=int(warm_up_steps),
            engine_factory=_shared_engine_factory,
            diagnostic_label=str(region_name),
            checkpoint_every_blocks=int(checkpoint_every_blocks),
            # Chunk layout: <out_dir>/expanded_chunks/{region}/rank{r}/chunk{c}.h5
            checkpoint_dir=(str(out_dir / "expanded_chunks" / str(region_name))
                            if (rank == 0 and int(checkpoint_every_blocks) > 0)
                            else None),
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
    parser = argparse.ArgumentParser(
        description="MPI expanded-ensemble KP TEE on kagome bond lattice"
    )
    add_common_args(parser)
    # expanded-specific knobs
    parser.add_argument("--regions", type=str, default="",
                        help="comma-separated region names; empty = all 7")
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
    # ----- Day-3 modes: frozen production / resume tune / warm-up -----
    parser.add_argument(
        "--log_g_init", type=str, default=None,
        help="directory containing prior tune outputs (kp_expanded_<REGION>.h5); "
             "loaded log_g seeds the chain before autotune (resume mode) or "
             "is used directly with --skip_autotune (frozen production)",
    )
    parser.add_argument(
        "--skip_autotune", action="store_true",
        help="frozen production: skip autotune entirely; requires --log_g_init",
    )
    parser.add_argument(
        "--warm_up_steps", type=int, default=0,
        help=f"MC steps before autotune/production to thermalise the chain. "
             f"Default 0; recommend ~{DEFAULT_WARM_UP_WHEN_LOADING} when "
             "loading log_g via --log_g_init so the chain leaves the C++ "
             "constructor's fresh state before measurements begin.",
    )
    parser.add_argument(
        "--checkpoint_every_blocks", type=int, default=0,
        help="incremental checkpointing (fixed n_steps production only): flush "
             "per-block counts every N blocks per rank into "
             "expanded_chunks/{region}/rank{r}/chunk{c}.h5 (requires "
             "--block_size). 0 = disabled.",
    )
    return parser


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    parser = build_parser()
    args = parser.parse_args()
    validate_mode_args(args)

    common = normalize_common_args(args)
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
                print(f"[kp_tee_expanded_mpi] {label}_total={total} → "
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
        log_g_init=args.log_g_init,
        skip_autotune=bool(args.skip_autotune),
        warm_up_steps=int(args.warm_up_steps),
        checkpoint_every_blocks=int(args.checkpoint_every_blocks),
    )

    if rank == 0 and payload is not None:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
