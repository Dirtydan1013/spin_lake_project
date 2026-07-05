"""
MPI helpers for expanded-ensemble QAQMC reweighting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

try:
    from mpi4py import MPI
except ImportError as exc:  # pragma: no cover - exercised in environments without mpi4py
    raise ImportError("mpi4py is required for src.mpi.reweighting_mpi") from exc

from src.engines.qaqmc_renyi import QAQMCRenyiRydberg
from src.tee.reweighting import (
    AutoTuneResult,
    ExpandedProductionResult,
    ReweightingDriver,
    _normalized_log_z_from_collection,
    combine_expanded_production_results,
    save_expanded_result_hdf5,
)


def _rank_seed(seed: int, rank: int) -> int:
    return int(seed) + 9973 * int(rank)


@dataclass
class ExpandedMPIResult:
    auto_tune: AutoTuneResult
    production: ExpandedProductionResult


def _print_production_diagnostics(label: str | None,
                                  root_result: ExpandedProductionResult,
                                  local_results,
                                  *,
                                  target_ensemble: int,
                                  reference_ensemble: int,
                                  estimator: str) -> None:
    prefix = "[expanded_mpi]"
    if label:
        prefix += f" {label}"

    visits = np.asarray(root_result.visit_counts, dtype=np.int64).reshape(-1)
    zero_global = np.flatnonzero(visits <= 0)
    print(
        f"{prefix} visits: sum={int(np.sum(visits))} "
        f"min={int(np.min(visits))} max={int(np.max(visits))} "
        f"zero={zero_global.tolist()}",
        flush=True,
    )

    local_zero = []
    for rank, result in enumerate(local_results):
        local_visits = np.asarray(result.visit_counts, dtype=np.int64).reshape(-1)
        zero = np.flatnonzero(local_visits <= 0)
        if zero.size:
            local_zero.append(f"r{rank}:{zero.tolist()}")
    if local_zero:
        print(f"{prefix} local zero-count ensembles: {'; '.join(local_zero)}", flush=True)

    if root_result.block_visit_counts is not None:
        blocks = np.asarray(root_result.block_visit_counts, dtype=np.int64)
        zero_cells = int(np.sum(blocks <= 0))
        zero_blocks = int(np.sum(np.any(blocks <= 0, axis=1)))
        print(
            f"{prefix} visit blocks: blocks={blocks.shape[0]} "
            f"zero_blocks={zero_blocks} zero_cells={zero_cells}",
            flush=True,
        )

    transition = np.asarray(root_result.transition_counts, dtype=np.int64)
    if transition.size:
        row_sums = np.sum(transition, axis=1)
        print(
            f"{prefix} transition row sums: "
            f"min={int(np.min(row_sums))} max={int(np.max(row_sums))} "
            f"zero_rows={np.flatnonzero(row_sums <= 0).tolist()}",
            flush=True,
        )

    if root_result.collection_counts is not None:
        collection = np.asarray(root_result.collection_counts, dtype=np.float64)
        row_sums = np.sum(collection, axis=1)
        print(
            f"{prefix} collection row sums: "
            f"min={float(np.min(row_sums)):.3g} max={float(np.max(row_sums)):.3g} "
            f"zero_rows={np.flatnonzero(row_sums <= 0.0).tolist()}",
            flush=True,
        )

    estimator_name = str(estimator).strip().lower()
    try:
        if estimator_name == "collection" and root_result.collection_log_z is not None:
            s2, s2_err = root_result.s2_collection(
                int(target_ensemble),
                reference_ensemble=int(reference_ensemble),
            )
            print(f"{prefix} S2(collection)={s2:.8g} +/- {s2_err:.3g}", flush=True)
        else:
            s2, s2_err = root_result.s2(
                int(target_ensemble),
                reference_ensemble=int(reference_ensemble),
            )
            print(f"{prefix} S2(histogram)={s2:.8g} +/- {s2_err:.3g}", flush=True)
    except ValueError as exc:
        print(f"{prefix} S2 diagnostic unavailable: {exc}", flush=True)


def _combine_results_with_optional_diagnostics(results, *, reference_ensemble: int,
                                               require_histogram: bool,
                                               require_collection: bool,
                                               diagnostic_label: str | None,
                                               target_ensemble: int,
                                               estimator: str
                                               ) -> ExpandedProductionResult:
    try:
        root_result = combine_expanded_production_results(
            results,
            reference_ensemble=reference_ensemble,
            require_histogram=require_histogram,
            require_collection=require_collection,
        )
    except ValueError:
        if diagnostic_label is not None:
            try:
                diagnostic_result = combine_expanded_production_results(
                    results,
                    reference_ensemble=reference_ensemble,
                    require_histogram=False,
                    require_collection=False,
                )
                _print_production_diagnostics(
                    diagnostic_label,
                    diagnostic_result,
                    results,
                    target_ensemble=target_ensemble,
                    reference_ensemble=reference_ensemble,
                    estimator=estimator,
                )
            except ValueError:
                pass
        raise

    if diagnostic_label is not None:
        _print_production_diagnostics(
            diagnostic_label,
            root_result,
            results,
            target_ensemble=target_ensemble,
            reference_ensemble=reference_ensemble,
            estimator=estimator,
        )
    return root_result


def _autotune_shared_log_g(driver: ReweightingDriver, *, n_steps_per_iter: int,
                           max_iters: int, tol: float, method: str,
                           damping: float, comm) -> AutoTuneResult:
    method_name = str(method).strip().lower()
    if method_name not in {"visit_count", "transition_matrix"}:
        raise ValueError("method must be 'visit_count' or 'transition_matrix'")

    log_g = driver.log_g
    visit_history: list[np.ndarray] = []
    collection_history: list[np.ndarray] = []

    for _ in range(int(max_iters)):
        driver.engine.reset_visit_counts_ext()
        if method_name == "transition_matrix":
            driver.engine.reset_collection_counts()
        driver.engine.run_steps(int(n_steps_per_iter))

        local_counts = driver.engine.get_visit_counts_ext().astype(np.int64)
        global_counts = np.asarray(comm.allreduce(local_counts, op=MPI.SUM), dtype=np.int64)
        visit_history.append(global_counts.copy())

        safe_counts = np.maximum(global_counts.astype(np.float64), 1.0)
        if method_name == "visit_count":
            target_log_g = log_g - np.log(safe_counts / safe_counts.mean())
        else:
            local_collection = driver.engine.get_collection_counts().astype(np.float64)
            global_collection = np.asarray(comm.allreduce(local_collection, op=MPI.SUM), dtype=np.float64)
            collection_history.append(global_collection.copy())
            log_z_est = _normalized_log_z_from_collection(global_collection, log_g, reference_ensemble=0)
            target_log_g = -log_z_est

        target_log_g -= target_log_g[0]
        log_g = (1.0 - float(damping)) * log_g + float(damping) * target_log_g
        log_g -= log_g[0]
        driver.set_log_g(log_g)

        if float(np.max(safe_counts) / np.min(safe_counts)) < float(tol):
            break

    return AutoTuneResult(
        log_g=driver.log_g,
        visit_counts=visit_history,
        collection_counts=collection_history if method_name == "transition_matrix" else None,
    )


def run_expanded_mpi(*, N: int, M: int, masks, neighbors, target_ensemble: int,
                     Omega: float = 1.0, Rb: float = 1.2,
                     delta_min: float = 0.0, delta_max: float = 1.0,
                     pos=None, epsilon: float = 0.01, seed: int = 42,
                     neighbor_cutoff: int | None = None,
                     delta_groups: int = 0,
                     initial_ensemble: int = 0,
                     autotune_steps_per_iter: int, autotune_max_iters: int = 8,
                     autotune_tol: float = 1.15,
                     autotune_method: str = "transition_matrix",
                     autotune_damping: float = 0.7,
                     n_steps: int | None = None,
                     block_size: int | None = None,
                     target_s2_err: float | None = None,
                     batch_steps: int | None = None,
                     max_steps: int | None = None,
                     min_steps: int = 0,
                     estimator: str = "collection",
                     reference_ensemble: int = 0,
                     filepath=None,
                     comm=None,
                     engine_factory=None,
                     diagnostic_label: str | None = None,
                     initial_log_g=None,
                     skip_autotune: bool = False,
                     warm_up_steps: int = 0,
                     checkpoint_every_blocks: int = 0,
                     checkpoint_dir=None):
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    estimator_name = str(estimator).strip().lower()
    if estimator_name not in {"histogram", "collection"}:
        raise ValueError("estimator must be 'histogram' or 'collection'")
    require_histogram = estimator_name == "histogram"
    require_collection = estimator_name == "collection"

    if skip_autotune and initial_log_g is None:
        raise ValueError("skip_autotune requires initial_log_g to be provided")

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
                delta_groups=int(delta_groups),
            )

    engine = engine_factory(rank)
    driver = ReweightingDriver(engine=engine)
    driver.set_ensemble_ladder(masks, neighbors, initial_ensemble=initial_ensemble)

    if initial_log_g is not None:
        driver.set_log_g(np.asarray(initial_log_g, dtype=np.float64))

    if int(warm_up_steps) > 0:
        driver.warm_up(int(warm_up_steps))

    if skip_autotune:
        # Frozen production: keep loaded log_g, build a stub AutoTuneResult
        # so the downstream HDF5 writer + ExpandedMPIResult shape stay valid.
        auto_result = AutoTuneResult(
            log_g=driver.log_g,
            visit_counts=[],
            collection_counts=None,
        )
    else:
        auto_result = _autotune_shared_log_g(
            driver,
            n_steps_per_iter=int(autotune_steps_per_iter),
            max_iters=int(autotune_max_iters),
            tol=float(autotune_tol),
            method=autotune_method,
            damping=float(autotune_damping),
            comm=comm,
        )

    if target_s2_err is None:
        if n_steps is None:
            raise ValueError("n_steps must be provided when target_s2_err is None")
        ckpt_blocks = int(checkpoint_every_blocks)
        if ckpt_blocks > 0:
            checkpoint_dir = comm.bcast(checkpoint_dir if rank == 0 else None, root=0)
        if ckpt_blocks > 0 and checkpoint_dir:
            # Incremental checkpointing: production runs in chunks of
            # ckpt_blocks * block_size steps; after each chunk the per-block
            # counts are flushed to checkpoint_dir/rank{r}/chunk{c}.h5 (same
            # layout as the profile driver's checkpointing).  Block boundaries
            # and the chain are identical to a single run_production call, so
            # the combined local result below is statistically identical; a
            # crash loses at most one chunk per rank.
            if block_size is None or int(block_size) <= 0:
                raise ValueError("checkpoint_every_blocks requires a positive block_size")
            from src.mpi.kp_tee_common import write_rank_chunk_h5
            from src.tee.reweighting import (
                _jackknife_log_z_from_collection_or_nan,
                _jackknife_log_z_or_nan,
            )

            chunk_steps = ckpt_blocks * int(block_size)
            parts = []
            done = 0
            c = 0
            while done < int(n_steps):
                cur = min(chunk_steps, int(n_steps) - done)
                part = driver.run_production(
                    n_steps=cur,
                    block_size=block_size,
                    reference_ensemble=reference_ensemble,
                    require_histogram=False,
                    require_collection=False,
                )
                parts.append(part)
                done += cur
                write_rank_chunk_h5(
                    checkpoint_dir, rank, c,
                    datasets=dict(
                        block_visit_counts=part.block_visit_counts,
                        block_collection_counts=part.block_collection_counts,
                        transition_counts=part.transition_counts,
                        log_g=part.log_g,
                    ),
                    attrs=dict(n_steps=int(cur), block_size=int(block_size),
                               steps_cumulative=int(done),
                               n_steps_total=int(n_steps), seed=int(seed)),
                )
                c += 1
            block_arr = np.concatenate([p.block_visit_counts for p in parts], axis=0)
            collection_block_arr = np.concatenate(
                [p.block_collection_counts for p in parts], axis=0)
            log_z, log_z_err = _jackknife_log_z_or_nan(
                block_arr, driver.log_g,
                reference_ensemble=reference_ensemble, require_histogram=False)
            collection_log_z, collection_log_z_err = _jackknife_log_z_from_collection_or_nan(
                collection_block_arr, driver.log_g,
                reference_ensemble=reference_ensemble, require_collection=False)
            local_result = ExpandedProductionResult(
                log_g=driver.log_g,
                visit_counts=np.sum(block_arr, axis=0, dtype=np.int64),
                block_visit_counts=block_arr,
                transition_counts=np.sum(
                    np.stack([p.transition_counts for p in parts]), axis=0).astype(np.int64),
                collection_counts=np.sum(
                    np.stack([p.collection_counts for p in parts]), axis=0),
                log_z=log_z,
                log_z_err=log_z_err,
                block_collection_counts=collection_block_arr,
                collection_log_z=collection_log_z,
                collection_log_z_err=collection_log_z_err,
            )
        else:
            local_result = driver.run_production(
                n_steps=int(n_steps),
                block_size=block_size,
                reference_ensemble=reference_ensemble,
                require_histogram=False,
                require_collection=False,
            )
        gathered = comm.gather(local_result, root=0)
        root_result = None
        if rank == 0:
            root_result = _combine_results_with_optional_diagnostics(
                gathered,
                reference_ensemble=reference_ensemble,
                require_histogram=require_histogram,
                require_collection=require_collection,
                diagnostic_label=diagnostic_label,
                target_ensemble=target_ensemble,
                estimator=estimator_name,
            )
    else:
        if batch_steps is None or max_steps is None or block_size is None:
            raise ValueError("batch_steps, max_steps, and block_size are required for adaptive production")
        if batch_steps <= 0 or max_steps <= 0 or block_size <= 0:
            raise ValueError("batch_steps, max_steps, and block_size must be positive")
        if batch_steps % block_size != 0:
            raise ValueError("batch_steps must be divisible by block_size")

        gathered_results = []
        steps_done = 0
        root_result = None
        done = False
        while steps_done < int(max_steps) and not done:
            current_batch = min(int(batch_steps), int(max_steps) - steps_done)
            local_batch = driver.run_production(
                n_steps=current_batch,
                block_size=block_size,
                reference_ensemble=reference_ensemble,
                require_histogram=False,
                require_collection=False,
            )
            gathered_batch = comm.gather(local_batch, root=0)
            steps_done += current_batch
            if rank == 0:
                gathered_results.extend(gathered_batch)
                root_result = _combine_results_with_optional_diagnostics(
                    gathered_results,
                    reference_ensemble=reference_ensemble,
                    require_histogram=require_histogram,
                    require_collection=require_collection,
                    diagnostic_label=None,
                    target_ensemble=target_ensemble,
                    estimator=estimator_name,
                )
                if estimator_name == "collection":
                    _, current_err = root_result.s2_collection(target_ensemble, reference_ensemble=reference_ensemble)
                else:
                    _, current_err = root_result.s2(target_ensemble, reference_ensemble=reference_ensemble)
                done = steps_done >= int(min_steps) and current_err <= float(target_s2_err)
            done = comm.bcast(done, root=0)

        if rank == 0 and diagnostic_label is not None and root_result is not None:
            _print_production_diagnostics(
                diagnostic_label,
                root_result,
                gathered_results,
                target_ensemble=target_ensemble,
                reference_ensemble=reference_ensemble,
                estimator=estimator_name,
            )

    if rank == 0 and filepath:
        save_expanded_result_hdf5(
            filepath,
            masks=masks,
            neighbors=neighbors,
            result=root_result,
            auto_tune=auto_result,
            params={
                "N": int(N),
                "M": int(M),
                "Omega": float(Omega),
                "Rb": float(Rb),
                "delta_min": float(delta_min),
                "delta_max": float(delta_max),
                "epsilon": float(epsilon),
                "seed": int(seed),
                "neighbor_cutoff": -1 if neighbor_cutoff is None else int(neighbor_cutoff),
                "delta_groups": int(delta_groups),
                "initial_ensemble": int(initial_ensemble),
                "target_ensemble": int(target_ensemble),
                "autotune_steps_per_iter": int(autotune_steps_per_iter),
                "autotune_max_iters": int(autotune_max_iters),
                "autotune_tol": float(autotune_tol),
                "autotune_method": str(autotune_method),
                "autotune_damping": float(autotune_damping),
                "block_size": -1 if block_size is None else int(block_size),
                "target_s2_err": "None" if target_s2_err is None else float(target_s2_err),
                "batch_steps": -1 if batch_steps is None else int(batch_steps),
                "max_steps": -1 if max_steps is None else int(max_steps),
                "min_steps": int(min_steps),
                "estimator": str(estimator),
                "reference_ensemble": int(reference_ensemble),
                "n_ranks": int(comm.Get_size()),
            },
            pos=engine.pos if hasattr(engine, "pos") else pos,
        )

    payload = None
    if rank == 0:
        payload = ExpandedMPIResult(auto_tune=auto_result, production=root_result)
    return comm.bcast(payload, root=0)


def _parse_json_array(raw: str):
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description="MPI-parallel expanded-ensemble QAQMC reweighting")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--Omega", type=float, default=1.0)
    parser.add_argument("--Rb", type=float, default=1.2)
    parser.add_argument("--delta_min", type=float, default=0.0)
    parser.add_argument("--delta_max", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neighbor_cutoff", type=int, default=-1)
    parser.add_argument("--delta_groups", type=int, default=0)
    parser.add_argument("--masks", type=str, required=False, default="")
    parser.add_argument("--neighbors", type=str, required=False, default="")
    parser.add_argument("--target_ensemble", type=int, default=0)
    parser.add_argument("--initial_ensemble", type=int, default=0)
    parser.add_argument("--autotune_steps_per_iter", type=int, default=15000)
    parser.add_argument("--autotune_max_iters", type=int, default=8)
    parser.add_argument("--autotune_tol", type=float, default=1.15)
    parser.add_argument("--autotune_method", type=str, default="transition_matrix")
    parser.add_argument("--autotune_damping", type=float, default=0.7)
    parser.add_argument("--n_steps", type=int, default=-1)
    parser.add_argument("--block_size", type=int, default=-1)
    parser.add_argument("--target_s2_err", type=float, default=-1.0)
    parser.add_argument("--batch_steps", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--min_steps", type=int, default=0)
    parser.add_argument("--estimator", type=str, default="collection")
    parser.add_argument("--reference_ensemble", type=int, default=0)
    parser.add_argument("--filepath", type=str, default="")
    args = parser.parse_args()

    params = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            params.update(json.load(fh))
    for key, value in vars(args).items():
        if key == "config":
            continue
        if key not in params or value != parser.get_default(key):
            params[key] = value

    if not params.get("masks"):
        raise ValueError("masks must be provided as a JSON array")
    if not params.get("neighbors"):
        raise ValueError("neighbors must be provided as a JSON array")
    params["masks"] = _parse_json_array(params["masks"])
    params["neighbors"] = _parse_json_array(params["neighbors"])
    if params.get("neighbor_cutoff", -1) < 0:
        params["neighbor_cutoff"] = None
    if params.get("n_steps", -1) < 0:
        params["n_steps"] = None
    if params.get("block_size", -1) < 0:
        params["block_size"] = None
    if params.get("target_s2_err", -1.0) <= 0.0:
        params["target_s2_err"] = None
    if params.get("batch_steps", -1) < 0:
        params["batch_steps"] = None
    if params.get("max_steps", -1) < 0:
        params["max_steps"] = None
    if params.get("filepath", "") == "":
        params["filepath"] = None

    run_expanded_mpi(**params)


if __name__ == "__main__":
    main()
