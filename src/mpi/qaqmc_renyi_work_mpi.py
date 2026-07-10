"""
MPI driver for the QAQMC Renyi nonequilibrium-work engine.

Each rank runs an independent set of Jarzynski trajectories with a different
seed (no inter-rank communication during sampling). Rank 0 reduces the work
samples across ranks via log-sum-exp aggregation and prints / saves the result.

Used for K-sweep convergence validation against ED on small systems, and as
the prototype production driver for larger systems.

Usage:
    mpiexec -n <N> python -m src.mpi.qaqmc_renyi_work_mpi \\
        --N 4 --M 100 --Omega 1.0 --Rb 1.2 \\
        --delta-min -1.0 --delta-max 2.0 \\
        --A-end "1,1,0,0" \\
        --K-values "4,20,50,200" \\
        --n-trajectories 8000 --n-thermalize 3000 --decorrelation-steps 500 \\
        [--A-start "0,0,0,0"]   # default: zeros
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

try:
    from mpi4py import MPI
except ImportError as exc:
    raise ImportError("mpi4py is required for src.mpi.qaqmc_renyi_work_mpi") from exc

# Make repo root importable when launched via `python -m`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.engines.qaqmc_renyi_work import QAQMCRenyiWorkRydberg
from src.mpi.equil_progress import run_equil_with_progress
from src.mpi.site_permutation import permute_rows, resolve_site_permutation
from src.rydberg.lattices import (
    generate_1d_chain,
    generate_kagome_bond_lattice,
    generate_kagome_bond_triangle_lattice,
)


def _parse_mask(s: str, N: int) -> np.ndarray:
    bits = [int(b.strip()) for b in s.split(",")]
    if len(bits) != N:
        raise ValueError(f"mask {s!r} has {len(bits)} bits, expected {N}")
    return np.array(bits, dtype=np.uint8)


def _ed_s2(N: int, M: int, Omega: float, Rb: float,
           delta_min: float, delta_max: float, epsilon: float,
           pos: np.ndarray, A_mask: np.ndarray) -> float:
    from src.analysis.ed_core import build_qaqmc_midpoint_state
    psi = build_qaqmc_midpoint_state(
        N=N, Omega=Omega, delta_min=delta_min, delta_max=delta_max,
        Rb=Rb, M=M, pos=pos, epsilon=epsilon, neighbor_cutoff=None,
    )
    A_sites  = [i for i in range(N) if A_mask[i] == 1]
    AC_sites = [i for i in range(N) if A_mask[i] == 0]
    nA, nAC = len(A_sites), len(AC_sites)
    M_mat = np.zeros((1 << nA, 1 << nAC), dtype=np.float64)
    for s in range(psi.size):
        bits = [(s >> i) & 1 for i in range(N)]
        a  = sum(bits[A_sites[j]]  << j for j in range(nA))
        ac = sum(bits[AC_sites[j]] << j for j in range(nAC))
        M_mat[a, ac] = psi[s]
    rho_A = M_mat @ M_mat.T
    return -float(np.log(max(np.trace(rho_A @ rho_A), 1e-300)))


def _ed_s2_difference(N, M, Omega, Rb, delta_min, delta_max, epsilon,
                      pos, A_start_mask, A_end_mask) -> float:
    s2_end   = _ed_s2(N, M, Omega, Rb, delta_min, delta_max, epsilon, pos, A_end_mask)
    if np.all(A_start_mask == 0):
        return s2_end
    s2_start = _ed_s2(N, M, Omega, Rb, delta_min, delta_max, epsilon, pos, A_start_mask)
    return s2_end - s2_start


def run_work_mpi(*, N: int, M: int, Omega: float, Rb: float,
                 delta_min: float, delta_max: float, epsilon: float,
                 pos: np.ndarray, A_start_mask: np.ndarray, A_end_mask: np.ndarray,
                 K_values: list[int],
                 n_trajectories: int, n_thermalize: int, decorrelation_steps: int,
                 neighbor_cutoff: int = -1, delta_groups: int = 200,
                 seed: int = 7, compute_ed: bool = True,
                 box_vectors: np.ndarray | None = None,
                 filepath: str | None = None,
                 checkpoint_every_trajectories: int = 0,
                 checkpoint_dir: str | None = None,
                 config_in: str | None = None,
                 config_out: str | None = None,
                 equil_progress_every: int = 500,
                 permute_site_labels: bool = True,
                 verbose: bool = True) -> dict:
    """When checkpoint_every_trajectories > 0 and checkpoint_dir is set, each
    rank additionally flushes its per-trajectory arrays every that many
    trajectories to checkpoint_dir/K{K}/rank{r}/chunk{c}.h5 (same layout as
    the profile driver's incremental checkpointing).  Repeated
    run_trajectories calls continue the same checkpoint chain, so the chunked
    run is statistically identical to a single long call; a crash loses at
    most one chunk per rank.  The final aggregate file is unchanged."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # Partition trajectories across ranks
    base = n_trajectories // n_ranks
    rem  = n_trajectories %  n_ranks
    my_n = base + (1 if rank < rem else 0)

    if rank == 0 and verbose:
        print(f"[MPI-WORK] N={N}, M={M}, ranks={n_ranks}, total_trajectories={n_trajectories}, "
              f"per-rank≈{base}, K_values={K_values}", flush=True)
        print(f"[MPI-WORK]   |A_start|={int(A_start_mask.sum())}, |A_end|={int(A_end_mask.sum())}, "
              f"|D|={int((A_end_mask & ~A_start_mask).sum())}", flush=True)
    if rank == 0 and compute_ed and N <= 16:
        ed_ref = _ed_s2_difference(N, M, Omega, Rb, delta_min, delta_max, epsilon,
                                   pos, A_start_mask, A_end_mask)
        if verbose:
            print(f"[MPI-WORK] ED reference ΔS_2 = {ed_ref:+.6f}", flush=True)
    else:
        ed_ref = None
    ed_ref = comm.bcast(ed_ref, root=0)

    # Warm-start config (loaded once; K-independent) and per-rank site-label
    # permutation (scan-order decorrelation — see src/mpi/site_permutation.py).
    # The engine runs on relabelled sites; masks are mapped into the engine
    # labelling, and all outputs (work samples etc.) are scalars, so nothing
    # needs mapping back.
    cfg = None
    if config_in:
        from src.mpi.chunk_io import check_config_compat, load_warm_config
        cfg = load_warm_config(config_in, rank, verbose=(rank == 0 and verbose))
        if cfg is None:
            raise FileNotFoundError(f"[warm-start] no rank*.h5 files in {config_in}")
        check_config_compat(
            cfg, dict(N=int(N), M_total=int(2 * M),
                      boundary=("periodic" if box_vectors is not None else "open")),
            f"renyi-work rank {rank}")
    site_perm, _inv_perm = resolve_site_permutation(
        N, seed + 9973 * rank, permute_site_labels, cfg=cfg, label="MPI-WORK")
    pos_engine = permute_rows(pos, site_perm)
    A_start_eng = permute_rows(A_start_mask, site_perm)
    A_end_eng = permute_rows(A_end_mask, site_perm)

    results = {}
    for K in K_values:
        comm.Barrier()
        t0 = time.perf_counter()

        eng = QAQMCRenyiWorkRydberg(
            N=N, M=M, Omega=Omega, Rb=Rb,
            delta_min=delta_min, delta_max=delta_max,
            epsilon=epsilon,
            seed=seed + 9973 * rank,
            pos=pos_engine, neighbor_cutoff=neighbor_cutoff,
            delta_groups=delta_groups, box_vectors=box_vectors,
        )
        eng.set_region_pair(A_start_eng, A_end_eng)
        eng.set_lambda_schedule(np.linspace(0.0, 1.0, K + 1))
        if cfg is not None:
            # Warm start: install the saved A_start-sector configuration and
            # seed the checkpoint chain — no thermalization needed.  The
            # configuration is K-independent.
            eng._cpp_engine.import_start_config(
                np.ascontiguousarray(cfg["op_types0"], dtype=np.int32),
                np.ascontiguousarray(cfg["op_sites0"], dtype=np.int32),
                np.ascontiguousarray(cfg["op_types1"], dtype=np.int32),
                np.ascontiguousarray(cfg["op_sites1"], dtype=np.int32))
            if rank == 0 and verbose and K == K_values[0]:
                print(f"[MPI-WORK] warm start from {config_in} — "
                      f"thermalization skipped", flush=True)
        else:
            # Chunked thermalize is equivalent to one long call: from the
            # second chunk on, thermalize()'s reset_to_start_sector restores
            # the checkpoint saved at the end of the previous chunk (the
            # current config) and keeps stepping.
            run_equil_with_progress(
                eng.thermalize, n_thermalize, label=f"MPI-WORK K={K}",
                rank=rank, print_every=equil_progress_every, verbose=verbose)

        ckpt = int(checkpoint_every_trajectories) if checkpoint_dir else 0
        if ckpt > 0 and my_n > 0:
            from types import SimpleNamespace

            from src.mpi.chunk_io import RankChunkWriter

            # Flat per-rank layout: checkpoint_dir/K{K}/rank{r}.h5 with one
            # chunk{i} group per flushed block of `ckpt` trajectories.
            k_dir = os.path.join(checkpoint_dir, f"K{K}")
            parts = []
            done = 0
            c = 0
            with RankChunkWriter(k_dir, rank,
                                 run_attrs=dict(K=int(K), seed=int(seed),
                                                my_n_trajectories=int(my_n))) as writer:
                while done < my_n:
                    n_chunk = min(ckpt, my_n - done)
                    part = eng.run_trajectories(n_chunk, decorrelation_steps)
                    parts.append(part)
                    done += n_chunk
                    writer.write_chunk(
                        c,
                        datasets=dict(
                            work_samples=part.work_samples,
                            final_swap_counts=part.final_swap_counts,
                            unjoined_counts_per_traj=part.unjoined_counts_per_traj,
                            topology_attempts_per_traj=part.topology_attempts_per_traj,
                            topology_accepts_per_traj=part.topology_accepts_per_traj,
                        ),
                        attrs=dict(K=int(K), n_trajectories=int(n_chunk),
                                   trajectories_cumulative=int(done)),
                    )
                    c += 1
                    if rank == 0 and verbose:
                        print(f"[MPI-WORK] K={K} rank0 chunk {c} written "
                              f"({done}/{my_n} trajectories)", flush=True)
            local = SimpleNamespace(
                work_samples=np.concatenate([p.work_samples for p in parts]),
                final_swap_counts=np.concatenate([p.final_swap_counts for p in parts]),
                unjoined_counts_per_traj=np.concatenate(
                    [p.unjoined_counts_per_traj for p in parts]),
                topology_attempts_per_traj=np.concatenate(
                    [p.topology_attempts_per_traj for p in parts]),
                topology_accepts_per_traj=np.concatenate(
                    [p.topology_accepts_per_traj for p in parts]),
            )
        else:
            local = eng.run_trajectories(my_n, decorrelation_steps)

        # Warm-start save: A_start-sector configuration from the last K.
        if K == K_values[-1]:
            out_dir = config_out
            if not out_dir and filepath:
                base = str(filepath)
                out_dir = (base[:-3] if base.endswith(".h5") else base) + "_configs"
            if out_dir:
                from src.mpi.chunk_io import RankChunkWriter
                t0c, s0c, t1c, s1c = eng._cpp_engine.export_start_config()
                cfg_datasets = dict(op_types0=t0c, op_sites0=s0c,
                                    op_types1=t1c, op_sites1=s1c)
                if site_perm is not None:
                    cfg_datasets["site_perm"] = np.asarray(site_perm, dtype=np.int32)
                with RankChunkWriter(out_dir, rank) as w:
                    w.write_final_config(
                        datasets=cfg_datasets,
                        attrs=dict(N=int(N), M_total=int(2 * M), seed=int(seed),
                                   boundary=("periodic" if box_vectors is not None
                                             else "open")))
                if rank == 0 and verbose:
                    print(f"[MPI-WORK] final configs saved → {out_dir}", flush=True)

        # Gather per-trajectory arrays to rank 0 and aggregate via log-sum-exp
        all_w           = comm.gather(local.work_samples,                root=0)
        all_final_swap  = comm.gather(local.final_swap_counts,           root=0)
        all_unjoined    = comm.gather(local.unjoined_counts_per_traj,    root=0)
        all_attempts    = comm.gather(local.topology_attempts_per_traj,  root=0)
        all_accepts     = comm.gather(local.topology_accepts_per_traj,   root=0)
        t_elapsed = comm.reduce(time.perf_counter() - t0, op=MPI.MAX, root=0)

        if rank == 0:
            w               = np.concatenate(all_w)
            final_swap      = np.concatenate(all_final_swap).astype(np.int32)
            unjoined_arr    = np.concatenate(all_unjoined).astype(np.int32)
            attempts_arr    = np.concatenate(all_attempts).astype(np.int64)
            accepts_arr     = np.concatenate(all_accepts).astype(np.int64)
            x = -w
            x_max = float(x.max())
            ex_shifted = np.exp(x - x_max)
            log_mean_exp_minus_w = x_max + np.log(ex_shifted.mean())
            delta_s2 = -log_mean_exp_minus_w
            work_mean = float(w.mean())
            work_var  = float(w.var(ddof=1))

            # bootstrap sd of delta_s2
            rng = np.random.default_rng(0)
            boot = []
            for _ in range(200):
                idx = rng.integers(0, len(w), len(w))
                xb = -w[idx]
                xb_max = float(xb.max())
                boot.append(-(xb_max + np.log(np.exp(xb - xb_max).mean())))
            boot_sd = float(np.std(boot, ddof=1))

            # ── Derived diagnostics ──────────────────────────────────────────
            unjoined_total = int(unjoined_arr.sum())
            attempts_total = int(attempts_arr.sum())
            accepts_total  = int(accepts_arr.sum())
            d_size_local = (
                int((A_end_mask & ~A_start_mask).sum()))
            # unjoined fraction = mean over traj of (unjoined / D_size).
            # =1 ⇒ every traj failed to join at all; 0 ⇒ all fully joined.
            if d_size_local > 0:
                unjoined_fraction = float(unjoined_arr.mean() / d_size_local)
            else:
                unjoined_fraction = 0.0
            topo_accept_rate = (accepts_total / attempts_total) if attempts_total > 0 else 0.0

            # Kish's effective sample size for the Jarzynski (importance) mean.
            # ESS = (Σ exp(-w))² / Σ exp(-2w), in [1, n_traj].  ESS << n_traj
            # signals that a few rare trajectories dominate <exp(-w)>.
            ex_shifted_2 = ex_shifted ** 2
            ess = float(ex_shifted.sum() ** 2 / ex_shifted_2.sum())

            # Work distribution skewness & excess kurtosis (heavy left tail →
            # Jarzynski estimator unstable).
            w_c = w - work_mean
            if work_var > 0:
                sd = np.sqrt(work_var)
                work_skew     = float((w_c**3).mean() / sd**3)
                work_kurtosis = float((w_c**4).mean() / sd**4 - 3.0)
            else:
                work_skew = 0.0
                work_kurtosis = 0.0

            diff_str = ""
            diff = None
            if ed_ref is not None:
                diff = delta_s2 - ed_ref
                diff_str = (f"diff_vs_ED={diff:+.4f} "
                            f"({diff/max(boot_sd,1e-12):+.2f}σ) ")

            if verbose:
                print(f"[MPI-WORK] K={K:4d}: ΔS_2={delta_s2:+.4f} (boot sd {boot_sd:.4f}) "
                      f"<w>={work_mean:+.3f} var={work_var:.2f} "
                      f"skew={work_skew:+.2f} kurt={work_kurtosis:+.2f} "
                      f"ESS={ess:.0f}/{n_trajectories} ({ess/n_trajectories:.1%}) "
                      f"unjoined_frac={unjoined_fraction:.1%} "
                      f"topo_acc={topo_accept_rate:.1%} "
                      f"{diff_str}elapsed={t_elapsed:.1f}s", flush=True)

            results[K] = dict(
                delta_s2=delta_s2, boot_sd=boot_sd,
                work_mean=work_mean, work_var=work_var,
                work_skew=work_skew, work_kurtosis=work_kurtosis,
                ess=ess, ess_fraction=ess / max(n_trajectories, 1),
                unjoined=unjoined_total, unjoined_fraction=unjoined_fraction,
                attempts=attempts_total, accepts=accepts_total,
                topo_accept_rate=topo_accept_rate,
                diff_vs_ED=diff if diff is not None else float("nan"),
                elapsed=t_elapsed,
                # Per-trajectory arrays for HDF5
                work_samples=w,
                final_swap_counts=final_swap,
                unjoined_counts_per_traj=unjoined_arr,
                topology_attempts_per_traj=attempts_arr,
                topology_accepts_per_traj=accepts_arr,
            )

    if rank == 0:
        if filepath:
            _save_hdf5(filepath, dict(
                N=N, M=M, Omega=Omega, Rb=Rb,
                delta_min=delta_min, delta_max=delta_max, epsilon=epsilon,
                neighbor_cutoff=neighbor_cutoff, delta_groups=delta_groups,
                seed=seed, n_ranks=n_ranks, n_trajectories=n_trajectories,
                n_thermalize=n_thermalize, decorrelation_steps=decorrelation_steps,
                A_start_mask=A_start_mask, A_end_mask=A_end_mask,
                ed_reference=ed_ref, K_values=K_values, results=results,
            ))
            if verbose:
                print(f"[MPI-WORK] saved HDF5 → {filepath}", flush=True)
        return {"ed_reference": ed_ref, "K_results": results}
    return None


def run_kp_regions_mpi(*, N, M, Omega, Rb, delta_min, delta_max, epsilon,
                       pos, region_pairs, K_values,
                       n_trajectories, n_thermalize, decorrelation_steps,
                       neighbor_cutoff=-1, delta_groups=600, seed=7,
                       compute_ed=True, box_vectors=None, filepath=None, kp_meta=None,
                       checkpoint_every_trajectories=0, checkpoint_dir=None,
                       config_in=None, config_out=None,
                       equil_progress_every=500, permute_site_labels=True,
                       verbose=True) -> dict | None:
    """Run the work engine for a list of (region_name, A_start_mask, A_end_mask).

    Each region pair is run independently (separate thermalization).  Results
    aggregated on rank 0 and (optionally) saved as a single HDF5 grouping by
    region name -- ready for downstream KP composer (γ = Σ ± S_2).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0 and verbose:
        names = [n for n, _, _ in region_pairs]
        print(f"[MPI-WORK-KP] N={N}, M={M}, regions={names}, K_values={K_values}, "
              f"n_traj={n_trajectories}", flush=True)
        if kp_meta is not None:
            print(f"[MPI-WORK-KP] kp_meta={kp_meta}", flush=True)

    all_results: dict[str, dict] = {}
    for region_name, A_start_mask, A_end_mask in region_pairs:
        if rank == 0 and verbose:
            print(f"\n[MPI-WORK-KP] ─── region {region_name} "
                  f"(|A_end|={int(A_end_mask.sum())}, "
                  f"|D|={int((A_end_mask & ~A_start_mask).sum())}) ───",
                  flush=True)
        out = run_work_mpi(
            N=N, M=M, Omega=Omega, Rb=Rb,
            delta_min=delta_min, delta_max=delta_max, epsilon=epsilon,
            pos=pos, A_start_mask=A_start_mask, A_end_mask=A_end_mask,
            K_values=K_values,
            n_trajectories=n_trajectories,
            n_thermalize=n_thermalize,
            decorrelation_steps=decorrelation_steps,
            neighbor_cutoff=neighbor_cutoff,
            delta_groups=delta_groups,
            seed=seed + 7919 * (hash(region_name) & 0xFFFF),
            compute_ed=compute_ed, box_vectors=box_vectors,
            filepath=None,    # we write a combined file at the end
            checkpoint_every_trajectories=checkpoint_every_trajectories,
            checkpoint_dir=(os.path.join(checkpoint_dir, str(region_name))
                            if checkpoint_dir else None),
            config_in=(os.path.join(config_in, str(region_name))
                       if config_in else None),
            config_out=(os.path.join(config_out, str(region_name))
                        if config_out else None),
            equil_progress_every=equil_progress_every,
            permute_site_labels=permute_site_labels,
            verbose=verbose,
        )
        if rank == 0:
            all_results[region_name] = dict(
                A_start_mask=A_start_mask, A_end_mask=A_end_mask,
                ed_reference=out["ed_reference"], K_results=out["K_results"],
            )

    if rank == 0:
        if verbose:
            print("\n[MPI-WORK-KP] ─── Summary ───", flush=True)
            # Compact one-line per region per K
            for region_name, payload in all_results.items():
                for K, res in payload["K_results"].items():
                    ed_str = (f"ed={payload['ed_reference']:+.4f}" if payload['ed_reference'] is not None
                              else "ed=N/A")
                    print(f"  {region_name:4s} K={K:4d}: ΔS_2={res['delta_s2']:+.4f} "
                          f"(boot sd {res['boot_sd']:.4f}) {ed_str}", flush=True)

            # If running all 7, also compute γ for each K from work-engine alone
            present = set(all_results.keys())
            if {"A","B","C","AB","BC","CA","ABC"}.issubset(present):
                for K in K_values:
                    s = {n: all_results[n]["K_results"][K]["delta_s2"]
                         for n in ("A","B","C","AB","BC","CA","ABC")}
                    gamma = (s["A"] + s["B"] + s["C"]
                             - s["AB"] - s["BC"] - s["CA"]
                             + s["ABC"])
                    print(f"  γ(K={K}) = S(A)+S(B)+S(C) - S(AB)-S(BC)-S(CA) + S(ABC) "
                          f"= {gamma:+.4f}", flush=True)

        if filepath is not None:
            _save_hdf5_kp(filepath, dict(
                N=N, M=M, Omega=Omega, Rb=Rb,
                delta_min=delta_min, delta_max=delta_max, epsilon=epsilon,
                neighbor_cutoff=neighbor_cutoff, delta_groups=delta_groups,
                seed=seed, n_trajectories=n_trajectories,
                n_thermalize=n_thermalize, decorrelation_steps=decorrelation_steps,
                K_values=K_values, kp_meta=kp_meta or {},
                regions=all_results,
            ))
            if verbose:
                print(f"\n[MPI-WORK-KP] saved HDF5 → {filepath}", flush=True)
        return {"regions": all_results, "K_values": K_values}
    return None


def _write_K_result_group(sg, res: dict) -> None:
    """Common writer for a per-K result group.

    Writes scalar diagnostics as attrs and per-trajectory arrays as datasets.
    """
    import h5py  # noqa: F401 (h5py group passed in)
    scalar_keys = ("delta_s2", "boot_sd", "work_mean", "work_var",
                   "work_skew", "work_kurtosis",
                   "ess", "ess_fraction",
                   "unjoined", "unjoined_fraction",
                   "attempts", "accepts", "topo_accept_rate",
                   "diff_vs_ED", "elapsed")
    for key in scalar_keys:
        if key in res:
            sg.attrs[key] = res[key]
    # Per-trajectory arrays (compressed since they can be large).
    dataset_keys = {
        "work_samples": "float64",
        "final_swap_counts": "int32",
        "unjoined_counts_per_traj": "int32",
        "topology_attempts_per_traj": "int64",
        "topology_accepts_per_traj": "int64",
    }
    for key, dtype in dataset_keys.items():
        if key in res:
            sg.create_dataset(key, data=res[key], dtype=dtype, compression="gzip")


def _save_hdf5_kp(path: str, payload: dict) -> None:
    import datetime
    import h5py

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        pg = f.create_group("params")
        for k in ("N", "M", "Omega", "Rb", "delta_min", "delta_max", "epsilon",
                  "neighbor_cutoff", "delta_groups", "seed",
                  "n_trajectories", "n_thermalize", "decorrelation_steps"):
            pg.attrs[k] = payload[k]
        pg.attrs["timestamp"] = datetime.datetime.utcnow().isoformat()
        for kk, vv in payload["kp_meta"].items():
            pg.attrs[f"kp_{kk}"] = vv

        rg = f.create_group("regions")
        rg.attrs["region_names"] = np.array(list(payload["regions"].keys()), dtype="S8")
        for name, payload_r in payload["regions"].items():
            ng = rg.create_group(name)
            ng.create_dataset("A_start_mask", data=payload_r["A_start_mask"], dtype="uint8")
            ng.create_dataset("A_end_mask",   data=payload_r["A_end_mask"],   dtype="uint8")
            ng.attrs["A_start_size"] = int(payload_r["A_start_mask"].sum())
            ng.attrs["A_end_size"]   = int(payload_r["A_end_mask"].sum())
            ng.attrs["D_size"] = int((payload_r["A_end_mask"] & ~payload_r["A_start_mask"]).sum())
            if payload_r["ed_reference"] is not None:
                ng.attrs["ed_reference"] = payload_r["ed_reference"]
            kg = ng.create_group("K_results")
            for K, res in payload_r["K_results"].items():
                sg = kg.create_group(f"K_{K}")
                _write_K_result_group(sg, res)
            kg.attrs["K_values"] = np.array(payload["K_values"], dtype=np.int64)


def _save_hdf5(path: str, payload: dict) -> None:
    import datetime
    import h5py

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        pg = f.create_group("params")
        for k in ("N", "M", "Omega", "Rb", "delta_min", "delta_max", "epsilon",
                  "neighbor_cutoff", "delta_groups", "seed",
                  "n_ranks", "n_trajectories", "n_thermalize",
                  "decorrelation_steps"):
            pg.attrs[k] = payload[k]
        pg.attrs["timestamp"] = datetime.datetime.utcnow().isoformat()

        rg = f.create_group("regions")
        rg.create_dataset("A_start_mask", data=payload["A_start_mask"], dtype="uint8")
        rg.create_dataset("A_end_mask",   data=payload["A_end_mask"],   dtype="uint8")
        rg.attrs["A_start_size"] = int(payload["A_start_mask"].sum())
        rg.attrs["A_end_size"]   = int(payload["A_end_mask"].sum())
        rg.attrs["D_size"] = int((payload["A_end_mask"] & ~payload["A_start_mask"]).sum())
        if payload["ed_reference"] is not None:
            rg.attrs["ed_reference"] = payload["ed_reference"]

        kg = f.create_group("K_results")
        for K, res in payload["results"].items():
            sg = kg.create_group(f"K_{K}")
            _write_K_result_group(sg, res)
        kg.attrs["K_values"] = np.array(payload["K_values"], dtype=np.int64)


def _parse_region_spec(spec: str, N: int) -> np.ndarray:
    """Parse a region spec as either a comma-separated bit string (len N) or
    a colon-separated list of site indices (e.g. "0:1:2") into a length-N mask."""
    if ":" in spec:
        sites = [int(s.strip()) for s in spec.split(":") if s.strip() != ""]
        mask = np.zeros(N, dtype=np.uint8)
        for s in sites:
            if s < 0 or s >= N:
                raise ValueError(f"site {s} out of range [0,{N})")
            mask[s] = 1
        return mask
    return _parse_mask(spec, N)


def main():
    parser = argparse.ArgumentParser(description="MPI driver for QAQMC Renyi work engine")
    # Lattice / Hamiltonian
    parser.add_argument("--lattice", type=str, default="1d_chain",
                        choices=["1d_chain", "kagome_bond", "kagome_bond_triangle",
                                 "kagome"])
    parser.add_argument("--N", type=int, default=0,
                        help="(1d_chain) site count. Ignored for kagome_bond (computed from nx,ny).")
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
                             "periodic (torus; not valid for kagome_bond_triangle "
                             "and inconsistent with KP region geometry)")
    parser.add_argument("--delta-groups", type=int, default=600)
    # Region pair: either bit string ("1,0,1,0,...") or site index list ("0:2:5")
    parser.add_argument("--A-end", type=str, default=None,
                        help='region mask: bit string "1,0,..." or site list "0:2:5". '
                             "Required unless --kp-regions is given.")
    parser.add_argument("--A-start", type=str, default=None,
                        help="region mask; default = zeros. Ignored if --kp-regions given.")
    # KP convenience: use src/kp/kp_geometry helpers to build region masks
    parser.add_argument("--kp-regions", type=str, default=None,
                        help='comma-separated subset of {A,B,C,AB,BC,CA,ABC} or "all" '
                             "for the standard 7-region KP set. Each region is run "
                             "as ∅→region. Triggers KP mode, overriding --A-end/--A-start.")
    parser.add_argument("--kp-start", type=str, default=None,
                        help="(kp pair mode) starting KP region name, e.g. 'A'. "
                             "Must be used together with --kp-end. The pair must be "
                             "nested (start ⊆ end), e.g. A→AB, B→BC, A→ABC.")
    parser.add_argument("--kp-end", type=str, default=None,
                        help="(kp pair mode) ending KP region name, e.g. 'AB'.")
    parser.add_argument("--kp-m", type=int, default=1,
                        help="(kp) KP construction size; m=1 is the minimal hexagon.")
    parser.add_argument("--kp-preferred-center", type=str, default=None,
                        help="(kp) preferred KP center label, e.g. 'C24'; default auto-pick.")
    # Protocol / sampling
    parser.add_argument("--K-values", type=str, default="200",
                        help="comma-separated K values to sweep (use one value for production)")
    parser.add_argument("--n-trajectories", type=int, default=4000)
    parser.add_argument("--n-thermalize", type=int, default=3000)
    parser.add_argument("--equil-progress-every", type=int, default=500,
                        help="print rank-0 thermalization progress every N steps "
                             "(<= 0 disables intermediate prints)")
    parser.add_argument("--decorrelation-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    # Output / misc
    parser.add_argument("--filepath", type=str, default=None,
                        help="optional HDF5 output path")
    parser.add_argument("--checkpoint-every-trajectories", type=int, default=0,
                        help="incremental checkpointing: flush per-trajectory arrays "
                             "every N trajectories per rank into "
                             "<checkpoint_dir>/[region/]K{K}/rank{r}/chunk{c}.h5. "
                             "0 = disabled. A crash then loses at most one chunk.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="checkpoint run directory (default: <filepath minus .h5>_chunks "
                             "when checkpointing is enabled)")
    parser.add_argument("--config-in", type=str, default=None,
                        help="warm-start directory of rank{r}.h5 final configurations "
                             "([region/]rank{r}.h5 in KP-regions mode); when given, "
                             "thermalization is skipped")
    parser.add_argument("--config-out", type=str, default=None,
                        help="where to save final configurations "
                             "(default: <filepath minus .h5>_configs)")
    parser.add_argument("--skip-ed", action="store_true",
                        help="skip ED reference computation (forced if N > 16)")
    parser.add_argument("--permute-site-labels",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="per-rank random site-label permutation (identical "
                             "physics, different update scan order) — decorrelates "
                             "the ordered-phase domain selection across ranks; see "
                             "scripts/experiments/scan_order_bias_probe.py")
    args = parser.parse_args()

    cfg_out_dir = args.config_out
    ckpt_dir = args.checkpoint_dir
    if int(args.checkpoint_every_trajectories) > 0 and ckpt_dir is None:
        if args.filepath:
            base = args.filepath[:-3] if args.filepath.endswith(".h5") else args.filepath
            ckpt_dir = base + "_chunks"
        else:
            raise ValueError("--checkpoint-every-trajectories requires "
                             "--checkpoint-dir or --filepath")

    if args.lattice == "1d_chain":
        if args.N <= 0:
            raise ValueError("--N must be >0 for 1d_chain")
        pos = generate_1d_chain(args.N, args.a)
        N = args.N
    elif args.lattice == "kagome_bond":
        pos = generate_kagome_bond_lattice(args.nx, args.ny, args.a)
        N = len(pos)
    elif args.lattice == "kagome_bond_triangle":
        pos = generate_kagome_bond_triangle_lattice(args.nx, args.ny, args.a)
        N = len(pos)
    elif args.lattice == "kagome":
        # Vertex kagome (U(1) QDM, paper/u1); nn distance = a/2 → use --a 2.0
        # for nn-distance units (paper's Rb/a_nn = 1.32 → --Rb 1.32).
        from src.rydberg.lattices import generate_kagome_lattice
        pos = generate_kagome_lattice(args.nx, args.ny, args.a,
                                      boundary=args.boundary)
        N = len(pos)
    else:
        raise ValueError(f"unsupported lattice {args.lattice}")

    box_vectors = None
    if args.boundary == "periodic":
        from src.rydberg.lattices import lattice_box_vectors
        box_vectors = lattice_box_vectors(args.lattice, args.nx, args.ny, args.a, N=N)

    K_values = [int(k) for k in args.K_values.split(",")]

    kp_pair_mode  = bool(args.kp_start or args.kp_end)
    kp_set_mode   = bool(args.kp_regions)
    if kp_pair_mode and kp_set_mode:
        raise ValueError(
            "use either --kp-regions (∅→each) or --kp-start/--kp-end (start→end pair), "
            "not both"
        )
    if kp_pair_mode and (args.kp_start is None or args.kp_end is None):
        raise ValueError("--kp-start and --kp-end must be given together")

    if kp_set_mode or kp_pair_mode:
        # ── KP mode: use src.kp.kp_geometry to build masks ───────────────────
        if args.lattice not in ("kagome_bond", "kagome_bond_triangle"):
            raise ValueError(
                "--kp-regions / --kp-start/end require a kagome BOND lattice "
                "(kp_geometry masks are not defined for the vertex 'kagome' "
                "lattice yet)")
        from src.kp.kp_geometry import (
            KP_REGION_NAMES,
            build_kp_region_masks_for_lattice,
        )
        spec = build_kp_region_masks_for_lattice(
            args.lattice, args.nx, args.ny, m=args.kp_m, a=args.a,
            preferred_center_label=args.kp_preferred_center,
        )

        if kp_pair_mode:
            for name in (args.kp_start, args.kp_end):
                if name not in KP_REGION_NAMES:
                    raise ValueError(
                        f"unknown KP region {name!r}; valid = {KP_REGION_NAMES}"
                    )
            start_mask = spec.region_masks[args.kp_start]
            end_mask   = spec.region_masks[args.kp_end]
            # validate nested
            if np.any(start_mask & ~end_mask):
                raise ValueError(
                    f"--kp-start={args.kp_start!r} is NOT a subset of --kp-end={args.kp_end!r}; "
                    f"start has {int(start_mask.sum())} sites, end has {int(end_mask.sum())}, "
                    f"sites in start∖end = {int((start_mask & ~end_mask).sum())}"
                )
            pair_name = f"{args.kp_start}_to_{args.kp_end}"
            region_pairs = [(pair_name, start_mask, end_mask)]
        else:
            names = (KP_REGION_NAMES if args.kp_regions.strip().lower() == "all"
                     else tuple(s.strip() for s in args.kp_regions.split(",") if s.strip()))
            unknown = [n for n in names if n not in KP_REGION_NAMES]
            if unknown:
                raise ValueError(f"unknown KP region names: {unknown}; "
                                 f"valid = {KP_REGION_NAMES}")
            region_pairs = [(name, np.zeros(N, dtype=np.uint8), spec.region_masks[name])
                            for name in names]

        run_kp_regions_mpi(
            N=N, M=args.M, Omega=args.Omega, Rb=args.Rb,
            delta_min=args.delta_min, delta_max=args.delta_max,
            epsilon=args.epsilon, pos=pos,
            region_pairs=region_pairs,
            K_values=K_values,
            n_trajectories=args.n_trajectories,
            n_thermalize=args.n_thermalize,
            decorrelation_steps=args.decorrelation_steps,
            neighbor_cutoff=args.neighbor_cutoff,
            delta_groups=args.delta_groups,
            seed=args.seed,
            compute_ed=(not args.skip_ed), box_vectors=box_vectors,
            filepath=args.filepath,
            checkpoint_every_trajectories=args.checkpoint_every_trajectories,
            checkpoint_dir=ckpt_dir,
            config_in=args.config_in,
            config_out=cfg_out_dir,
            equil_progress_every=args.equil_progress_every,
            permute_site_labels=args.permute_site_labels,
            kp_meta=dict(
                lattice=args.lattice, nx=args.nx, ny=args.ny, a=args.a,
                m=args.kp_m, center_label=spec.center_label,
                mode=("pair" if kp_pair_mode else "regions"),
                start_region=(args.kp_start or ""),
                end_region=(args.kp_end or ""),
            ),
        )
        return

    if args.A_end is None:
        raise ValueError("--A-end is required when --kp-regions is not given")
    A_end   = _parse_region_spec(args.A_end,   N)
    A_start = _parse_region_spec(args.A_start, N) if args.A_start else np.zeros(N, dtype=np.uint8)

    run_work_mpi(
        N=N, M=args.M, Omega=args.Omega, Rb=args.Rb,
        delta_min=args.delta_min, delta_max=args.delta_max,
        epsilon=args.epsilon, pos=pos,
        A_start_mask=A_start, A_end_mask=A_end,
        K_values=K_values,
        n_trajectories=args.n_trajectories,
        n_thermalize=args.n_thermalize,
        decorrelation_steps=args.decorrelation_steps,
        neighbor_cutoff=args.neighbor_cutoff,
        delta_groups=args.delta_groups,
        seed=args.seed,
        compute_ed=(not args.skip_ed), box_vectors=box_vectors,
        filepath=args.filepath,
        checkpoint_every_trajectories=args.checkpoint_every_trajectories,
        checkpoint_dir=ckpt_dir,
        config_in=args.config_in,
        config_out=cfg_out_dir,
        equil_progress_every=args.equil_progress_every,
        permute_site_labels=args.permute_site_labels,
    )


if __name__ == "__main__":
    main()
