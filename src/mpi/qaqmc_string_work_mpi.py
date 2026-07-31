"""
MPI driver for the QAQMC off-diagonal string-work (Jarzynski) engine.

Each rank runs an independent set of string-toggle Jarzynski trajectories with
a different seed (no inter-rank communication during sampling).  Rank 0
aggregates all log_J samples via log-sum-exp into O_C = Z_C / Z_empty and
writes a single HDF5 result file.

Incremental checkpointing (--checkpoint-every-trajectories > 0) additionally
flushes each rank's log_J samples every N trajectories to
``<checkpoint_dir>/K{K}/rank{r}/chunk{c}.h5`` — the same
one-subdirectory-per-rank layout used by the profile / renyi-work drivers —
so a crash loses at most one chunk per rank.

Usage:
    mpiexec -n <N> python -m src.mpi.qaqmc_string_work_mpi \\
        --lattice 1d_chain --N 6 --M 100 \\
        --Omega 1.0 --Rb 1.2 --delta-min -1.0 --delta-max 2.0 \\
        --string-sites "2,3" \\
        --K-values "50,200" --schedule cosine \\
        --n-trajectories 4000 --n-thermalize 2000 --decorrelation-steps 100 \\
        --filepath data/string_work.h5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

try:
    from mpi4py import MPI
except ImportError as exc:
    raise ImportError("mpi4py is required for src.mpi.qaqmc_string_work_mpi") from exc

# Make repo root importable when launched via `python -m`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.mpi.chunk_io import (
    RankChunkWriter,
    array_fingerprint,
    collective_resume_decision,
    compact_operator_checkpoint,
    load_checkpointed_rank_chunks,
)
from src.mpi.driver_util import (rank_seed as _rank_seed,
                                 cuda_device_for_rank as _cuda_device_for_rank,
                                 permutation_checkpoint as _permutation_checkpoint)
from src.mpi.equil_progress import run_equil_with_progress
from src.mpi.site_permutation import (permute_rows, resolve_site_permutation,
                                      to_engine)
from src.rydberg.lattices import (
    generate_1d_chain,
    generate_kagome_bond_lattice,
)


def _engine_type_and_schedule_for_backend(backend: str):
    """Resolve only the selected backend, preserving CPU import semantics."""
    if backend == "cuda":
        # Load the CUDA facade first so qaqmc_cpp comes from build_cuda; its
        # CPU base import then reuses that already-loaded portable module.
        from src.engines.qaqmc_string_work_cuda import QAQMCStringWorkRydbergCUDA
        from src.engines.qaqmc_string_work import cosine_schedule

        return QAQMCStringWorkRydbergCUDA, cosine_schedule
    if backend == "cpu":
        from src.engines.qaqmc_string_work import (
            QAQMCStringWorkRydberg,
            cosine_schedule,
        )

        return QAQMCStringWorkRydberg, cosine_schedule
    raise ValueError("backend must be 'cpu' or 'cuda'")


_STRING_CHUNK_DATASETS = ("log_j_samples",)


def _string_cuda_checkpoint(eng, site_perm, n_sites: int) -> tuple[dict, dict]:
    """Export the rolling start-sector checkpoint plus Philox counters."""
    if not eng._eng.has_checkpoint:
        raise RuntimeError("string CUDA engine has no rolling checkpoint")
    eng._eng.restore_device_checkpoint()
    types, sites = eng._eng.get_operator_string()
    types, sites = compact_operator_checkpoint(types, sites)
    return (
        dict(
            op_types=types,
            op_sites=sites,
            site_perm=_permutation_checkpoint(site_perm, n_sites),
        ),
        dict(
            sweep_id=int(eng._eng.sweep_id),
            topology_id=int(eng._eng.topology_id),
            checkpoint_mask=int(eng._checkpoint_mask),
        ),
    )


def _restore_string_cuda_checkpoint(eng, checkpoint: dict, site_perm,
                                    n_sites: int, direction: str) -> None:
    datasets = checkpoint["datasets"]
    attrs = checkpoint["attrs"]
    expected_perm = _permutation_checkpoint(site_perm, n_sites)
    if "site_perm" not in datasets:
        raise ValueError("string CUDA continuation checkpoint lacks site_perm")
    if not np.array_equal(np.asarray(datasets["site_perm"]), expected_perm):
        raise ValueError("string CUDA continuation checkpoint site permutation changed")
    expected_mask = 0 if direction == "forward" else eng._full_mask()
    if int(attrs["checkpoint_mask"]) != expected_mask:
        raise ValueError("string CUDA continuation checkpoint has wrong start sector")
    eng._eng.set_op_string(
        np.ascontiguousarray(datasets["op_types"], dtype=np.int32),
        np.ascontiguousarray(datasets["op_sites"], dtype=np.int32),
    )
    eng.thermalize(0, direction=direction)
    eng._eng.sweep_id = int(attrs["sweep_id"])
    eng._eng.topology_id = int(attrs["topology_id"])




def _aggregate_log_j(log_j: np.ndarray, direction: str) -> dict:
    """log-sum-exp aggregation of Jarzynski samples into O_C (document §33)."""
    n = int(log_j.size)
    finite = np.isfinite(log_j)
    if not np.any(finite):
        return dict(o_c=0.0, log_o_c=-math.inf, log_o_c_sem_boot=math.inf,
                    n_eff=0.0, p_max=0.0,
                    zero_weight_fraction=1.0, n_trajectories=n)
    max_log = float(log_j[finite].max())
    weights = np.zeros(n, dtype=np.float64)
    weights[finite] = np.exp(log_j[finite] - max_log)
    sum_w = float(weights.sum())
    log_mean_j = max_log + math.log(sum_w / n)
    log_o_c = log_mean_j if direction == "forward" else -log_mean_j
    p = weights / sum_w
    # Bootstrap sem of log O_C (fixed rng: reproducible; needed to give the
    # drag-composed curve an anchor error bar).
    rng = np.random.default_rng(0)
    boot = np.empty(200, dtype=np.float64)
    for b in range(boot.size):
        pick = rng.choice(log_j, n)  # zero-weight samples resample too
        pmax = pick[np.isfinite(pick)].max() if np.any(np.isfinite(pick)) else 0.0
        boot[b] = pmax + math.log(max(np.mean(np.exp(pick - pmax)), 1e-300))
    return dict(
        o_c=math.exp(log_o_c), log_o_c=log_o_c,
        log_o_c_sem_boot=float(boot.std(ddof=1)),
        n_eff=1.0 / float(np.sum(p ** 2)), p_max=float(p.max()),
        zero_weight_fraction=float(np.count_nonzero(~finite)) / max(n, 1),
        n_trajectories=n,
    )


def run_string_work_mpi(*, N: int, M: int, Omega: float, Rb: float,
                        delta_min: float, delta_max: float, epsilon: float,
                        pos: np.ndarray, string_sites: list[int],
                        K_values: list[int], schedule: str,
                        n_trajectories: int, n_thermalize: int,
                        decorrelation_steps: int,
                        m_star: int | None = None,
                        direction: str = "forward",
                        n_topology_sweeps_per_lambda: int = 1,
                        n_qaqmc_sweeps_per_lambda: int = 1,
                        neighbor_cutoff: int = -1, delta_groups: int = 600,
                        seed: int = 7, box_vectors: np.ndarray | None = None,
                        filepath: str | None = None,
                        checkpoint_every_trajectories: int = 0,
                        checkpoint_dir: str | None = None,
                        config_in: str | None = None,
                        config_out: str | None = None,
                        equil_progress_every: int = 500,
                        permute_site_labels: bool = True,
                        backend: str = "cpu",
                        resume: bool = False,
                        drag_grid: list[int] | None = None,
                        drag_mirror: bool = True,
                        drag_samples_per_rung: int = 400,
                        drag_sweeps_between_samples: int = 1,
                        drag_burn_per_rung: int = 5,
                        drag_slots_per_rung: int = 1,
                        drag_repeats: int = 1,
                        drag_thermalize: int = -1,
                        drag_equil_at_anchor: int = 100,
                        verbose: bool = True) -> dict | None:
    """config_in: warm-start directory of rank{r}.h5 final configurations from
    a previous run with the same (N, M, Hamiltonian); when given, per-K
    thermalization is skipped (the loaded op string is already equilibrated —
    the configuration is K-independent).  config_out: where each rank saves
    its final configuration (default <filepath minus .h5>_configs).
    ``resume=True`` is CUDA-only and restores the exact rolling operator
    checkpoint and Philox counters committed with the last sample chunk."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    if backend not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu' or 'cuda'")
    if checkpoint_every_trajectories < 0:
        raise ValueError("checkpoint_every_trajectories must be non-negative")
    if n_trajectories <= 0:
        raise ValueError("n_trajectories must be positive")
    if n_thermalize < 0 or decorrelation_steps < 0:
        raise ValueError("thermalization/decorrelation counts must be non-negative")
    if n_topology_sweeps_per_lambda < 0 or n_qaqmc_sweeps_per_lambda < 0:
        raise ValueError("per-lambda sweep counts must be non-negative")
    if not K_values or any(int(K) < 1 for K in K_values):
        raise ValueError("K_values must contain positive integers")
    if resume and backend != "cuda":
        raise ValueError("exact trajectory resume is supported only by backend='cuda'")
    if drag_grid:
        if backend != "cpu":
            raise ValueError("the drag-ladder phase is CPU-only")
        if drag_samples_per_rung < 2 or drag_repeats < 1 or drag_slots_per_rung < 1:
            raise ValueError("drag_samples_per_rung >= 2, drag_repeats >= 1, "
                             "drag_slots_per_rung >= 1 required")
        if drag_mirror and m_star is not None and int(m_star) != int(M):
            raise ValueError("drag_mirror requires the anchor at the symmetric "
                             "point m_star = M")
    if resume and (checkpoint_every_trajectories <= 0 or not checkpoint_dir):
        raise ValueError("resume requires checkpoint_every_trajectories and checkpoint_dir")
    engine_type, cosine_schedule = _engine_type_and_schedule_for_backend(backend)

    base = n_trajectories // n_ranks
    rem = n_trajectories % n_ranks
    my_n = base + (1 if rank < rem else 0)
    rank_seed = _rank_seed(seed, rank)

    if rank == 0 and verbose:
        print(f"[MPI-STRWORK] backend={backend}, N={N}, M={M}, ranks={n_ranks}, "
              f"total_trajectories={n_trajectories}, per-rank≈{base}, "
              f"string_sites={list(string_sites)}, K_values={K_values}, "
              f"schedule={schedule}, direction={direction}", flush=True)

    # Warm-start config (K-independent) and per-rank site-label permutation
    # (scan-order decorrelation — see src/mpi/site_permutation.py).  The engine
    # runs on relabelled sites; string_sites are mapped into the engine
    # labelling and all outputs (log_J samples) are scalars.
    cfg = None
    if config_in:
        from src.mpi.chunk_io import check_config_compat, load_warm_config
        cfg = load_warm_config(config_in, rank, verbose=(rank == 0 and verbose))
        if cfg is None:
            raise FileNotFoundError(
                f"[warm-start] no rank*.h5 files in {config_in}")
        check_config_compat(
            cfg, dict(N=int(N), M_total=int(2 * M),
                      boundary=("periodic" if box_vectors is not None else "open")),
            f"string-work rank {rank}")
    site_perm, inv_perm = resolve_site_permutation(
        N, rank_seed, permute_site_labels, cfg=cfg,
        label="MPI-STRWORK")
    pos_engine = permute_rows(pos, site_perm)
    string_sites_eng = [int(s) for s in to_engine(list(string_sites), inv_perm)]

    results: dict[int, dict] = {}
    saw_committed_resume = False
    for K in K_values:
        comm.Barrier()
        t0 = time.perf_counter()
        ckpt = int(checkpoint_every_trajectories) if checkpoint_dir else 0
        k_dir = os.path.join(checkpoint_dir, f"K{K}") if ckpt > 0 else None
        run_attrs = dict(
            checkpoint_schema=1,
            K=int(K),
            seed=rank_seed,
            n_ranks=int(n_ranks),
            direction=str(direction),
            schedule=str(schedule),
            my_n_trajectories=int(my_n),
            backend=str(backend),
            N=int(N),
            M_total=int(2 * M),
            Omega=float(Omega),
            Rb=float(Rb),
            delta_min=float(delta_min),
            delta_max=float(delta_max),
            epsilon=float(epsilon),
            neighbor_cutoff=int(-1 if neighbor_cutoff is None else neighbor_cutoff),
            delta_groups=int(delta_groups),
            decorrelation_steps=int(decorrelation_steps),
            positions_sha256=array_fingerprint(
                np.asarray(pos, dtype=np.float64)),
            box_vectors_sha256=array_fingerprint(
                None if box_vectors is None else
                np.asarray(box_vectors, dtype=np.float64)),
            site_perm=_permutation_checkpoint(site_perm, N),
            m_star=int(M if m_star is None else m_star),
            string_sites=np.asarray(list(string_sites), dtype=np.int32),
            n_topology_sweeps=int(n_topology_sweeps_per_lambda),
            n_qaqmc_sweeps=int(n_qaqmc_sweeps_per_lambda),
        )
        resumed = (
            load_checkpointed_rank_chunks(
                k_dir, rank, _STRING_CHUNK_DATASETS,
                expected_run_attrs=run_attrs,
            )
            if resume and my_n > 0 else None
        )
        resume_this_k = False
        if resume:
            resume_this_k = collective_resume_decision(
                comm,
                rank=rank,
                active=my_n > 0,
                completed=(0 if resumed is None else resumed["completed"]),
                allow_all_missing=saw_committed_resume,
                label=f"string K={K}",
            )
            if resume_this_k:
                saw_committed_resume = True
            else:
                resumed = None
        if resumed is not None and resumed["completed"] > my_n:
            raise ValueError(
                f"checkpoint has {resumed['completed']} trajectories, "
                f"but this rank requests only {my_n}")

        extra = ({"device": _cuda_device_for_rank(comm),
                  "verbose": rank == 0 and verbose}
                 if backend == "cuda" else {})
        eng = engine_type(
            N=N, M=M, Omega=Omega, Rb=Rb,
            delta_min=delta_min, delta_max=delta_max,
            epsilon=epsilon, seed=rank_seed,
            pos=pos_engine,
            neighbor_cutoff=(None if neighbor_cutoff < 0 else neighbor_cutoff),
            delta_groups=delta_groups, box_vectors=box_vectors, **extra,
        )
        eng.set_string_sites(string_sites_eng, m_star)
        if schedule == "cosine":
            eng.set_lambda_schedule(cosine_schedule(int(K)))
        else:
            eng.set_lambda_schedule(np.linspace(0.0, 1.0, int(K) + 1))
        if resumed is not None and resumed["completed"] > 0:
            _restore_string_cuda_checkpoint(
                eng, resumed["checkpoint"], site_perm, N, direction
            )
            if rank == 0 and verbose:
                print(
                    f"[MPI-STRWORK] K={K} exact CUDA resume at "
                    f"{resumed['completed']}/{my_n} trajectories",
                    flush=True,
                )
        elif cfg is not None:
            eng._eng.set_op_string(
                np.ascontiguousarray(cfg["op_types"], dtype=np.int32),
                np.ascontiguousarray(cfg["op_sites"], dtype=np.int32))
            # thermalize(0) still sets the seam mask for the chosen direction.
            eng.thermalize(0, direction=direction)
            if rank == 0 and verbose and K == K_values[0]:
                print(f"[MPI-STRWORK] warm start from {config_in} — "
                      f"thermalization skipped", flush=True)
        else:
            # Chunked thermalize is equivalent to one long call: each chunk
            # just re-sets the same starting seam mask (idempotent) and keeps
            # stepping the same chain.
            run_equil_with_progress(
                lambda n: eng.thermalize(n, direction=direction),
                n_thermalize, label=f"MPI-STRWORK K={K}",
                rank=rank, print_every=equil_progress_every, verbose=verbose)

        if ckpt > 0 and my_n > 0:
            # Chunked sampling: run_trajectories resets the seam sector per
            # trajectory, so repeated calls continue the same chain and are
            # statistically identical to a single long call.  Flat per-rank
            # layout: checkpoint_dir/K{K}/rank{r}.h5 with atomically published
            # samples plus the exact state needed by the next block.
            done = int(resumed["completed"]) if resumed is not None else 0
            c = int(resumed["next_chunk"]) if resumed is not None else 0
            with RankChunkWriter(
                k_dir, rank, run_attrs=run_attrs, resume=resume
            ) as writer:
                while done < my_n:
                    n_chunk = min(ckpt, my_n - done)
                    part = eng.run_trajectories(
                        n_chunk, decorrelation_steps,
                        n_topology_sweeps_per_lambda=n_topology_sweeps_per_lambda,
                        n_qaqmc_sweeps_per_lambda=n_qaqmc_sweeps_per_lambda,
                        direction=direction)
                    done += n_chunk
                    state_data, state_attrs = _string_cuda_checkpoint(
                        eng, site_perm, N
                    ) if backend == "cuda" else ({}, {})
                    writer.write_chunk(
                        c,
                        datasets=dict(log_j_samples=part.log_j_samples),
                        attrs=dict(K=int(K), n_trajectories=int(n_chunk),
                                   trajectories_cumulative=int(done),
                                   direction=str(direction)),
                        checkpoint_datasets=state_data,
                        checkpoint_attrs=state_attrs,
                        prune_previous_checkpoints=(backend == "cuda"),
                    )
                    c += 1
                    if rank == 0 and verbose:
                        print(f"[MPI-STRWORK] K={K} rank0 chunk {c} written "
                              f"({done}/{my_n} trajectories)", flush=True)
            stored = load_checkpointed_rank_chunks(
                k_dir, rank, _STRING_CHUNK_DATASETS,
                expected_run_attrs=run_attrs,
            )
            if stored["completed"] != my_n:
                raise RuntimeError(
                    f"checkpoint reload found {stored['completed']}/{my_n} trajectories")
            local_log_j = np.asarray(
                stored["datasets"]["log_j_samples"], dtype=np.float64
            )
        else:
            local = eng.run_trajectories(
                my_n, decorrelation_steps,
                n_topology_sweeps_per_lambda=n_topology_sweeps_per_lambda,
                n_qaqmc_sweeps_per_lambda=n_qaqmc_sweeps_per_lambda,
                direction=direction)
            local_log_j = np.asarray(local.log_j_samples, dtype=np.float64)

        # Warm-start save: final op string of the last K's engine (the
        # configuration is K-independent, any equilibrated one works).
        if K == K_values[-1]:
            out_dir = config_out
            if not out_dir and filepath:
                base = str(filepath)
                out_dir = (base[:-3] if base.endswith(".h5") else base) + "_configs"
            if out_dir:
                # CUDA keeps the rolling equilibrated start-sector state in a
                # D2D checkpoint while the live arrays end in the final
                # nonequilibrium trajectory sector.  Export the checkpoint,
                # not that endpoint.  Sampling is already finished here.
                if backend == "cuda" and eng._eng.has_checkpoint:
                    eng._eng.restore_device_checkpoint()
                cfg_datasets = dict(
                    op_types=np.asarray(eng._eng.op_types, dtype=np.int32),
                    op_sites=np.asarray(eng._eng.op_sites, dtype=np.int32))
                if site_perm is not None:
                    cfg_datasets["site_perm"] = np.asarray(site_perm, dtype=np.int32)
                with RankChunkWriter(out_dir, rank) as w:
                    w.write_final_config(
                        datasets=cfg_datasets,
                        attrs=dict(N=int(N), M_total=int(2 * M), seed=int(seed),
                                   boundary=("periodic" if box_vectors is not None
                                             else "open")))
                if rank == 0 and verbose:
                    print(f"[MPI-STRWORK] final configs saved → {out_dir}", flush=True)

        all_log_j = comm.gather(local_log_j, root=0)
        t_elapsed = comm.reduce(time.perf_counter() - t0, op=MPI.MAX, root=0)

        if rank == 0:
            log_j = np.concatenate(all_log_j)
            agg = _aggregate_log_j(log_j, direction)
            agg["elapsed"] = float(t_elapsed)
            agg["log_j_samples"] = log_j
            results[int(K)] = agg
            if verbose:
                print(f"[MPI-STRWORK] K={K:4d}: O_C={agg['o_c']:.6f} "
                      f"(log O_C={agg['log_o_c']:+.4f}) "
                      f"n_eff={agg['n_eff']:.0f}/{agg['n_trajectories']} "
                      f"({agg['n_eff']/max(agg['n_trajectories'],1):.1%}) "
                      f"p_max={agg['p_max']:.3f} "
                      f"zero_frac={agg['zero_weight_fraction']:.1%} "
                      f"elapsed={t_elapsed:.1f}s", flush=True)

    # ── Drag-ladder phase: whole-curve O_C(delta) from the last K's engine ──
    # (docs/design/seam_drag_curve.md).  Each rank runs drag_repeats
    # independent mirrored (or left-only) RB-ladder passes on its own chain;
    # rank 0 aggregates pass-level log_r rows (scatter SEM across passes) and
    # composes O_C(delta) = O_C(anchor; K) * exp(log_r) per K.
    drag_payload = None
    if drag_grid:
        grid = np.asarray(sorted({int(m) for m in drag_grid}, reverse=True),
                          dtype=np.int64)
        anchor_m = int(M if m_star is None else m_star)
        n_therm_drag = int(n_thermalize if drag_thermalize < 0 else drag_thermalize)
        comm.Barrier()
        t0 = time.perf_counter()
        # Full-mask sector equilibrium at the anchor (the lambda phase left
        # the chain in its trajectory end sector).
        run_equil_with_progress(
            lambda n: eng.thermalize(n, direction="reverse"),
            n_therm_drag, label="MPI-STRWORK drag",
            rank=rank, print_every=equil_progress_every, verbose=verbose)

        drag_run_attrs = dict(
            checkpoint_schema=1, phase="drag", seed=rank_seed,
            n_ranks=int(n_ranks), N=int(N), M_total=int(2 * M),
            m_anchor=int(anchor_m), mirror=bool(drag_mirror),
            m_grid=np.asarray(grid, dtype=np.int64),
            samples_per_rung=int(drag_samples_per_rung),
            sweeps_between_samples=int(drag_sweeps_between_samples),
            burn_per_rung=int(drag_burn_per_rung),
            slots_per_rung=int(drag_slots_per_rung),
            repeats=int(drag_repeats),
            string_sites=np.asarray(list(string_sites), dtype=np.int32),
        )
        kw = dict(n_samples_per_rung=drag_samples_per_rung,
                  n_sweeps_between_samples=drag_sweeps_between_samples,
                  n_burn_per_rung=drag_burn_per_rung,
                  slots_per_rung=drag_slots_per_rung,
                  n_equil_at_anchor=drag_equil_at_anchor)

        def _one_pass():
            if drag_mirror:
                res = eng.run_drag_curve_mirrored(grid, **kw)
                return dict(log_r=np.asarray(res.log_r_mirror),
                            log_r_sem=np.asarray(res.log_r_sem),
                            log_r_left=np.asarray(res.left.log_r),
                            log_r_right=np.asarray(res.right.log_r)), (
                    int(res.left.rung_m.size + res.right.rung_m.size))
            res = eng.run_drag_ladder(grid, m_anchor=anchor_m, **kw)
            return dict(log_r=np.asarray(res.log_r),
                        log_r_sem=np.asarray(res.log_r_sem)), int(res.rung_m.size)

        rows = []
        n_rungs_per_pass = 0
        drag_dir = (os.path.join(checkpoint_dir, "drag")
                    if (checkpoint_dir and checkpoint_every_trajectories > 0)
                    else None)
        if drag_dir:
            with RankChunkWriter(drag_dir, rank, run_attrs=drag_run_attrs) as w:
                for rep in range(drag_repeats):
                    row, n_rungs_per_pass = _one_pass()
                    rows.append(row)
                    w.write_chunk(rep, datasets=row, attrs=dict(rep=int(rep)))
                    if rank == 0 and verbose:
                        print(f"[MPI-STRWORK] drag rank0 pass {rep + 1}/"
                              f"{drag_repeats} written", flush=True)
        else:
            for rep in range(drag_repeats):
                row, n_rungs_per_pass = _one_pass()
                rows.append(row)

        all_rows = comm.gather(rows, root=0)
        t_drag = comm.reduce(time.perf_counter() - t0, op=MPI.MAX, root=0)
        if rank == 0:
            flat = [row for rr in all_rows for row in rr]
            mat = np.stack([r["log_r"] for r in flat])          # (P, n_grid)
            within = np.stack([r["log_r_sem"] for r in flat])   # (P, n_grid)
            n_passes = mat.shape[0]
            log_r_mean = mat.mean(axis=0)
            sem_within = np.sqrt((within ** 2).mean(axis=0) / n_passes)
            sem_scatter = (mat.std(axis=0, ddof=1) / math.sqrt(n_passes)
                           if n_passes > 1 else sem_within)
            deltas = delta_min + (delta_max - delta_min) * grid / float(M)
            drag_payload = dict(
                m_grid=grid, deltas=deltas, m_anchor=anchor_m,
                mirror=bool(drag_mirror), n_passes=int(n_passes),
                n_rungs_per_pass=int(n_rungs_per_pass),
                samples_per_rung=int(drag_samples_per_rung),
                sweeps_between_samples=int(drag_sweeps_between_samples),
                burn_per_rung=int(drag_burn_per_rung),
                slots_per_rung=int(drag_slots_per_rung),
                thermalize=int(n_therm_drag), elapsed=float(t_drag),
                log_r_passes=mat, log_r_sem_passes=within,
                log_r_mean=log_r_mean,
                log_r_sem=np.maximum(sem_scatter, sem_within),
                curves={},
            )
            if drag_mirror:
                drag_payload["log_r_left_passes"] = np.stack(
                    [r["log_r_left"] for r in flat])
                drag_payload["log_r_right_passes"] = np.stack(
                    [r["log_r_right"] for r in flat])
            for K, res_k in results.items():
                log_o = res_k["log_o_c"] + log_r_mean
                sem_o = np.sqrt(res_k["log_o_c_sem_boot"] ** 2
                                + drag_payload["log_r_sem"] ** 2)
                drag_payload["curves"][int(K)] = dict(
                    log_o_curve=log_o, o_curve=np.exp(log_o),
                    log_o_curve_sem=sem_o)
            if verbose:
                print(f"[MPI-STRWORK] drag: {n_passes} passes x "
                      f"{grid.size} points (mirror={drag_mirror}, "
                      f"spr={drag_slots_per_rung}, rungs={n_rungs_per_pass}) "
                      f"elapsed={t_drag:.1f}s", flush=True)
                for j in range(grid.size):
                    print(f"[MPI-STRWORK]   delta={deltas[j]:+.3f} (m={grid[j]}): "
                          f"log r={log_r_mean[j]:+.4f} ± "
                          f"{drag_payload['log_r_sem'][j]:.4f}", flush=True)

    if rank != 0:
        return None

    if filepath:
        _save_hdf5(filepath, dict(
            N=N, M=M, Omega=Omega, Rb=Rb,
            delta_min=delta_min, delta_max=delta_max, epsilon=epsilon,
            neighbor_cutoff=neighbor_cutoff, delta_groups=delta_groups,
            seed=seed, n_ranks=n_ranks, n_trajectories=n_trajectories,
            n_thermalize=n_thermalize, decorrelation_steps=decorrelation_steps,
            backend=backend, resumed=bool(resume),
            checkpoint_every_trajectories=checkpoint_every_trajectories,
            positions_sha256=array_fingerprint(
                np.asarray(pos, dtype=np.float64)),
            box_vectors_sha256=array_fingerprint(
                None if box_vectors is None else
                np.asarray(box_vectors, dtype=np.float64)),
            string_sites=np.asarray(list(string_sites), dtype=np.int32),
            m_star=(-1 if m_star is None else int(m_star)),
            schedule=str(schedule), direction=str(direction),
            K_values=K_values, results=results, drag=drag_payload,
        ))
        if verbose:
            print(f"[MPI-STRWORK] saved HDF5 → {filepath}", flush=True)
    return {"K_results": results, "drag": drag_payload}


def _save_hdf5(path: str, payload: dict) -> None:
    import datetime

    import h5py

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        pg = f.create_group("params")
        for k in ("N", "M", "Omega", "Rb", "delta_min", "delta_max", "epsilon",
                  "neighbor_cutoff", "delta_groups", "seed", "n_ranks",
                  "n_trajectories", "n_thermalize", "decorrelation_steps",
                  "m_star", "schedule", "direction", "backend", "resumed",
                  "checkpoint_every_trajectories", "positions_sha256",
                  "box_vectors_sha256"):
            pg.attrs[k] = payload[k]
        pg.attrs["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        pg.create_dataset("string_sites", data=payload["string_sites"])
        pg.create_dataset("K_values",
                          data=np.asarray(payload["K_values"], dtype=np.int64))
        rg = f.create_group("K_results")
        for K, res in payload["results"].items():
            sg = rg.create_group(f"K{int(K)}")
            for key in ("o_c", "log_o_c", "log_o_c_sem_boot", "n_eff", "p_max",
                        "zero_weight_fraction", "n_trajectories", "elapsed"):
                sg.attrs[key] = res[key]
            sg.create_dataset("log_j_samples", data=res["log_j_samples"],
                              compression="gzip")
        drag = payload.get("drag")
        if drag is not None:
            dg = f.create_group("drag")
            for key in ("m_anchor", "mirror", "n_passes", "n_rungs_per_pass",
                        "samples_per_rung", "sweeps_between_samples",
                        "burn_per_rung", "slots_per_rung", "thermalize",
                        "elapsed"):
                dg.attrs[key] = drag[key]
            for key in ("m_grid", "deltas", "log_r_passes", "log_r_sem_passes",
                        "log_r_mean", "log_r_sem",
                        "log_r_left_passes", "log_r_right_passes"):
                if key in drag:
                    dg.create_dataset(key, data=np.asarray(drag[key]))
            cg = dg.create_group("curves")
            for K, cur in drag["curves"].items():
                kg = cg.create_group(f"K{int(K)}")
                for key in ("log_o_curve", "o_curve", "log_o_curve_sem"):
                    kg.create_dataset(key, data=np.asarray(cur[key]))


def _parse_int_list(text: str) -> list[int]:
    return [int(tok) for tok in text.replace(";", ",").split(",") if tok.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="MPI driver for the QAQMC off-diagonal string-work engine")
    parser.add_argument("--lattice", type=str, default="1d_chain",
                        choices=["1d_chain", "kagome_bond", "kagome_bond_triangle"])
    parser.add_argument("--N", type=int, default=0,
                        help="(1d_chain) number of sites")
    parser.add_argument("--nx", type=int, default=6, help="(kagome_bond) cells in x")
    parser.add_argument("--ny", type=int, default=6, help="(kagome_bond) cells in y")
    parser.add_argument("--a", type=float, default=1.0, help="lattice constant")
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--Omega", type=float, default=1.0)
    parser.add_argument("--Rb", type=float, default=1.2)
    parser.add_argument("--delta-min", type=float, default=-1.0)
    parser.add_argument("--delta-max", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--neighbor-cutoff", type=int, default=-1)
    parser.add_argument("--boundary", type=str, default="open",
                        choices=["open", "periodic"],
                        help="spatial lattice boundary: open (finite patch) or "
                             "periodic (torus; not valid for kagome_bond_triangle)")
    parser.add_argument("--delta-groups", type=int, default=600)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu",
                        help="transition backend; CUDA expects one Slurm GPU per MPI rank")
    parser.add_argument("--string-sites", type=str, required=True,
                        help="comma-separated site indices of the string C")
    parser.add_argument("--m-star", type=int, default=-1,
                        help="seam slice (default -1 = M, the midpoint)")
    parser.add_argument("--K-values", type=str, default="200",
                        help="comma-separated lambda-schedule segment counts")
    parser.add_argument("--schedule", type=str, default="cosine",
                        choices=["cosine", "linear"])
    parser.add_argument("--direction", type=str, default="forward",
                        choices=["forward", "reverse"])
    parser.add_argument("--n-trajectories", type=int, default=4000,
                        help="total trajectories across ranks")
    parser.add_argument("--n-thermalize", type=int, default=2000)
    parser.add_argument("--equil-progress-every", type=int, default=500,
                        help="print rank-0 thermalization progress every N steps "
                             "(<= 0 disables intermediate prints)")
    parser.add_argument("--decorrelation-steps", type=int, default=100)
    parser.add_argument("--n-topology-sweeps-per-lambda", type=int, default=1)
    parser.add_argument("--n-qaqmc-sweeps-per-lambda", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--filepath", type=str, default=None,
                        help="optional HDF5 output path")
    parser.add_argument("--checkpoint-every-trajectories", type=int, default=0,
                        help="incremental checkpointing: flush log_J samples every "
                             "N trajectories per rank into "
                             "<checkpoint_dir>/K{K}/rank{r}/chunk{c}.h5. 0 = disabled.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="checkpoint run directory (default: <filepath minus .h5>"
                             "_chunks when checkpointing is enabled)")
    parser.add_argument(
        "--resume", action="store_true",
        help="CUDA only: append an existing checkpoint run and restore the exact "
             "operator state plus Philox counters from its last committed chunk",
    )
    parser.add_argument("--config-in", type=str, default=None,
                        help="warm-start directory of rank{r}.h5 final configurations; "
                             "when given, thermalization is skipped")
    parser.add_argument("--config-out", type=str, default=None,
                        help="where to save final configurations "
                             "(default: <filepath minus .h5>_configs)")
    # ── Drag-ladder phase (whole-curve O_C(delta); docs/design/seam_drag_curve.md)
    parser.add_argument("--drag-deltas", type=str, default=None,
                        help="comma-separated delta values at which to record the "
                             "drag curve (converted to forward-branch slots "
                             "m = round(M*(delta-delta_min)/span), clipped to "
                             "[1, M-1]); enables the drag phase")
    parser.add_argument("--drag-grid", type=str, default=None,
                        help="explicit comma-separated forward-branch record slots "
                             "(alternative to --drag-deltas)")
    parser.add_argument("--drag-mirror",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="mirror-average the two branches about m=M "
                             "(odd-in-v removal; requires the default m_star)")
    parser.add_argument("--drag-samples-per-rung", type=int, default=400)
    parser.add_argument("--drag-sweeps-between-samples", type=int, default=1)
    parser.add_argument("--drag-burn-per-rung", type=int, default=5)
    parser.add_argument("--drag-slots-per-rung", type=int, default=1,
                        help="slots per RB rung; raise until the rung log-sd is "
                             "~0.3 (cost scales as 1/slots_per_rung at fixed "
                             "statistical error)")
    parser.add_argument("--drag-repeats", type=int, default=1,
                        help="independent ladder passes per rank (scatter over "
                             "ranks x repeats gives the curve error bar)")
    parser.add_argument("--drag-thermalize", type=int, default=-1,
                        help="reverse-sector thermalization before the drag phase "
                             "(-1 = reuse --n-thermalize)")
    parser.add_argument("--drag-equil-at-anchor", type=int, default=100)
    parser.add_argument("--permute-site-labels",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="per-rank random site-label permutation (identical "
                             "physics, different update scan order) — decorrelates "
                             "the ordered-phase domain selection across ranks; see "
                             "scripts/experiments/scan_order_bias_probe.py")
    args = parser.parse_args()

    if args.lattice == "1d_chain":
        if args.N <= 0:
            raise ValueError("--N must be >0 for 1d_chain")
        pos = np.asarray(generate_1d_chain(args.N, args.a), dtype=np.float64)
        N = args.N
    elif args.lattice == "kagome_bond_triangle":
        from src.rydberg.lattices import generate_kagome_bond_triangle_lattice
        pos = np.ascontiguousarray(
            generate_kagome_bond_triangle_lattice(args.nx, args.ny, args.a),
            dtype=np.float64)
        N = len(pos)
    else:
        pos = np.ascontiguousarray(
            generate_kagome_bond_lattice(args.nx, args.ny, args.a), dtype=np.float64)
        N = len(pos)

    string_sites = _parse_int_list(args.string_sites)
    if any(s < 0 or s >= N for s in string_sites):
        raise ValueError(f"string sites out of range [0, {N})")

    box_vectors = None
    if args.boundary == "periodic":
        from src.rydberg.lattices import lattice_box_vectors
        box_vectors = lattice_box_vectors(args.lattice, args.nx, args.ny, args.a, N=N)

    ckpt_dir = args.checkpoint_dir
    if int(args.checkpoint_every_trajectories) > 0 and ckpt_dir is None:
        if args.filepath:
            base = (args.filepath[:-3] if args.filepath.endswith(".h5")
                    else args.filepath)
            ckpt_dir = base + "_chunks"
        else:
            raise ValueError("--checkpoint-every-trajectories requires "
                             "--checkpoint-dir or --filepath")

    drag_grid = None
    if args.drag_deltas and args.drag_grid:
        raise ValueError("give either --drag-deltas or --drag-grid, not both")
    if args.drag_deltas:
        span = args.delta_max - args.delta_min
        drag_grid = sorted(
            {min(max(int(round(args.M * (float(tok) - args.delta_min) / span)), 1),
                 args.M - 1)
             for tok in args.drag_deltas.replace(";", ",").split(",") if tok.strip()},
            reverse=True)
    elif args.drag_grid:
        drag_grid = sorted({int(t) for t in _parse_int_list(args.drag_grid)},
                           reverse=True)
        if any(m < 1 or m >= args.M for m in drag_grid):
            raise ValueError("--drag-grid slots must lie in [1, M-1] "
                             "(forward branch)")

    run_string_work_mpi(
        N=N, M=args.M, Omega=args.Omega, Rb=args.Rb,
        delta_min=args.delta_min, delta_max=args.delta_max,
        epsilon=args.epsilon, pos=pos,
        string_sites=string_sites,
        K_values=_parse_int_list(args.K_values),
        schedule=args.schedule,
        n_trajectories=args.n_trajectories,
        n_thermalize=args.n_thermalize,
        decorrelation_steps=args.decorrelation_steps,
        m_star=(None if args.m_star < 0 else args.m_star),
        direction=args.direction,
        n_topology_sweeps_per_lambda=args.n_topology_sweeps_per_lambda,
        n_qaqmc_sweeps_per_lambda=args.n_qaqmc_sweeps_per_lambda,
        neighbor_cutoff=args.neighbor_cutoff,
        delta_groups=args.delta_groups,
        seed=args.seed, box_vectors=box_vectors,
        filepath=args.filepath,
        checkpoint_every_trajectories=args.checkpoint_every_trajectories,
        checkpoint_dir=ckpt_dir,
        config_in=args.config_in,
        config_out=args.config_out,
        equil_progress_every=args.equil_progress_every,
        permute_site_labels=args.permute_site_labels,
        backend=args.backend,
        resume=args.resume,
        drag_grid=drag_grid,
        drag_mirror=args.drag_mirror,
        drag_samples_per_rung=args.drag_samples_per_rung,
        drag_sweeps_between_samples=args.drag_sweeps_between_samples,
        drag_burn_per_rung=args.drag_burn_per_rung,
        drag_slots_per_rung=args.drag_slots_per_rung,
        drag_repeats=args.drag_repeats,
        drag_thermalize=args.drag_thermalize,
        drag_equil_at_anchor=args.drag_equil_at_anchor,
    )


if __name__ == "__main__":
    main()
