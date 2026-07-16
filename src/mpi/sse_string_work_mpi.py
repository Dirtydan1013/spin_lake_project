"""
MPI driver for the thermal SSE off-diagonal string-work (Jarzynski) engine.

Measures  O_C(beta) = Tr[X_C e^{-beta H}] / Tr[e^{-beta H}],
X_C = prod_{i in C} sigma_i^x, on the equilibrium ensemble — the thermal
counterpart of src.mpi.qaqmc_string_work_mpi (same lambda-interpolation /
Jarzynski protocol, same CLI shape; the engine is the finite-temperature
SSEEngine with a periodic-tau seam, csrc/cpu/detail/sse_off_diagonal_core.hpp). Every
rank runs independent trajectories; log_J samples are aggregated with
log-sum-exp on rank 0.  A classical (diagonal) ensemble has O_C == 0
identically, so a resolved non-zero closed X-loop is a direct coherence
witness (cf. docs/progress/experiments/E14).

Example::

    mpiexec -n 16 python -m src.mpi.sse_string_work_mpi \\
        --lattice kagome_bond --nx 6 --ny 6 --a 4.0 --Rb 2.4 \\
        --boundary periodic --delta 4.25 --beta 6 \\
        --string-sites 12,17,23 --K-values 200,400 \\
        --n-trajectories 4000 --filepath data/sse_string_work.h5
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
    raise ImportError("mpi4py is required for src.mpi.sse_string_work_mpi") from exc

# Make repo root importable when launched via `python -m`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.engines.sse_string_work import SSEStringWorkRydberg, cosine_schedule
from src.mpi.chunk_io import RankChunkWriter, check_config_compat, load_warm_config
from src.mpi.equil_progress import run_equil_with_progress
from src.mpi.qaqmc_string_work_mpi import _aggregate_log_j, _parse_int_list
from src.mpi.site_permutation import (permute_rows, resolve_site_permutation,
                                      to_engine)
from src.rydberg.lattices import (
    generate_1d_chain,
    generate_kagome_bond_lattice,
)


def _rank_seed(seed: int, rank: int) -> int:
    return int(seed) + 9973 * int(rank)


def run_sse_string_work_mpi(*, N: int, beta: float, Omega: float, Rb: float,
                            delta: float, epsilon: float,
                            pos: np.ndarray, string_sites: list[int],
                            K_values: list[int], schedule: str,
                            n_trajectories: int, n_thermalize: int,
                            decorrelation_steps: int,
                            m_star: int = 0,
                            direction: str = "forward",
                            n_topology_sweeps_per_lambda: int = 1,
                            n_sse_sweeps_per_lambda: int = 1,
                            neighbor_cutoff: int = -1,
                            seed: int = 7, box_vectors: np.ndarray | None = None,
                            filepath: str | None = None,
                            checkpoint_every_trajectories: int = 0,
                            checkpoint_dir: str | None = None,
                            config_in: str | None = None,
                            config_out: str | None = None,
                            equil_progress_every: int = 500,
                            permute_site_labels: bool = True,
                            verbose: bool = True) -> dict | None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    base = n_trajectories // n_ranks
    rem = n_trajectories % n_ranks
    my_n = base + (1 if rank < rem else 0)

    if rank == 0 and verbose:
        print(f"[MPI-SSE-STRWORK] N={N}, β={beta:g}, δ={delta:g}, "
              f"ranks={n_ranks}, total_trajectories={n_trajectories}, "
              f"per-rank≈{base}, string_sites={list(string_sites)}, "
              f"K_values={K_values}, schedule={schedule}, "
              f"direction={direction}", flush=True)

    cfg = None
    if config_in:
        cfg = load_warm_config(config_in, rank, verbose=(rank == 0 and verbose))
        if cfg is None:
            raise FileNotFoundError(
                f"[warm-start] no rank*.h5 files in {config_in}")
        check_config_compat(
            cfg, dict(N=int(N),
                      boundary=("periodic" if box_vectors is not None else "open")),
            f"sse-string-work rank {rank}")
    site_perm, inv_perm = resolve_site_permutation(
        N, _rank_seed(seed, rank), permute_site_labels, cfg=cfg,
        label="MPI-SSE-STRWORK")
    pos_engine = permute_rows(pos, site_perm)
    string_sites_eng = [int(s) for s in to_engine(list(string_sites), inv_perm)]

    results: dict[int, dict] = {}
    for K in K_values:
        comm.Barrier()
        t0 = time.perf_counter()

        eng = SSEStringWorkRydberg(
            N=N, beta=beta, Omega=Omega, Rb=Rb, delta=delta,
            epsilon=epsilon, seed=_rank_seed(seed, rank),
            pos=pos_engine,
            neighbor_cutoff=(None if neighbor_cutoff < 0 else neighbor_cutoff),
            box_vectors=box_vectors,
        )
        if cfg is not None:
            eng._eng.set_config(
                np.ascontiguousarray(cfg["state"], dtype=np.int32),
                np.ascontiguousarray(cfg["op_types"], dtype=np.int32),
                np.ascontiguousarray(cfg["op_sites"], dtype=np.int32))
        eng.set_string_sites(string_sites_eng, m_star)
        if schedule == "cosine":
            eng.set_lambda_schedule(cosine_schedule(int(K)))
        else:
            eng.set_lambda_schedule(np.linspace(0.0, 1.0, int(K) + 1))
        if cfg is not None:
            # thermalize(0) still sets the seam mask (with parity repair).
            eng.thermalize(0, direction=direction)
            if rank == 0 and verbose and K == K_values[0]:
                print(f"[MPI-SSE-STRWORK] warm start from {config_in} — "
                      f"thermalization skipped", flush=True)
        else:
            run_equil_with_progress(
                lambda n: eng.thermalize(n, direction=direction),
                n_thermalize, label=f"MPI-SSE-STRWORK K={K}",
                rank=rank, print_every=equil_progress_every, verbose=verbose)

        ckpt = int(checkpoint_every_trajectories) if checkpoint_dir else 0
        if ckpt > 0 and my_n > 0:
            k_dir = os.path.join(checkpoint_dir, f"K{K}")
            parts = []
            done = 0
            c = 0
            with RankChunkWriter(k_dir, rank,
                                 run_attrs=dict(K=int(K), seed=int(seed),
                                                beta=float(beta), delta=float(delta),
                                                direction=str(direction),
                                                my_n_trajectories=int(my_n))) as writer:
                while done < my_n:
                    n_chunk = min(ckpt, my_n - done)
                    part = eng.run_trajectories(
                        n_chunk, decorrelation_steps,
                        n_topology_sweeps_per_lambda=n_topology_sweeps_per_lambda,
                        n_qaqmc_sweeps_per_lambda=n_sse_sweeps_per_lambda,
                        direction=direction)
                    parts.append(part.log_j_samples)
                    done += n_chunk
                    writer.write_chunk(
                        c, datasets=dict(log_j_samples=part.log_j_samples),
                        attrs=dict(K=int(K), n_trajectories=int(n_chunk),
                                   trajectories_cumulative=int(done),
                                   direction=str(direction)))
                    c += 1
                    if rank == 0 and verbose:
                        print(f"[MPI-SSE-STRWORK] K={K} rank0 chunk {c} written "
                              f"({done}/{my_n} trajectories)", flush=True)
            local_log_j = (np.concatenate(parts) if parts
                           else np.empty(0, dtype=np.float64))
        else:
            local = eng.run_trajectories(
                my_n, decorrelation_steps,
                n_topology_sweeps_per_lambda=n_topology_sweeps_per_lambda,
                n_qaqmc_sweeps_per_lambda=n_sse_sweeps_per_lambda,
                direction=direction)
            local_log_j = np.asarray(local.log_j_samples, dtype=np.float64)

        # Warm-start save (beta/delta-specific: SSE configs are not
        # temperature-independent, unlike the projector engine's).
        if K == K_values[-1]:
            out_dir = config_out
            if not out_dir and filepath:
                base_p = str(filepath)
                out_dir = (base_p[:-3] if base_p.endswith(".h5") else base_p) + "_configs"
            if out_dir:
                cfg_datasets = dict(
                    state=np.asarray(eng._eng.state, dtype=np.int32),
                    op_types=np.asarray(eng._eng.op_types, dtype=np.int32),
                    op_sites=np.asarray(eng._eng.op_sites, dtype=np.int32))
                if site_perm is not None:
                    cfg_datasets["site_perm"] = np.asarray(site_perm, dtype=np.int32)
                with RankChunkWriter(out_dir, rank) as w:
                    w.write_final_config(
                        datasets=cfg_datasets,
                        attrs=dict(N=int(N), beta=float(beta), seed=int(seed),
                                   boundary=("periodic" if box_vectors is not None
                                             else "open")))
                if rank == 0 and verbose:
                    print(f"[MPI-SSE-STRWORK] final configs saved → {out_dir}",
                          flush=True)

        all_log_j = comm.gather(local_log_j, root=0)
        t_elapsed = comm.reduce(time.perf_counter() - t0, op=MPI.MAX, root=0)

        if rank == 0:
            log_j = np.concatenate(all_log_j)
            agg = _aggregate_log_j(log_j, direction)
            agg["elapsed"] = float(t_elapsed)
            agg["log_j_samples"] = log_j
            results[int(K)] = agg
            if verbose:
                print(f"[MPI-SSE-STRWORK] K={K:4d}: O_C={agg['o_c']:.6f} "
                      f"(log O_C={agg['log_o_c']:+.4f}) "
                      f"n_eff={agg['n_eff']:.0f}/{agg['n_trajectories']} "
                      f"({agg['n_eff']/max(agg['n_trajectories'],1):.1%}) "
                      f"p_max={agg['p_max']:.3f} "
                      f"zero_frac={agg['zero_weight_fraction']:.1%} "
                      f"elapsed={t_elapsed:.1f}s", flush=True)

    if rank != 0:
        return None

    if filepath:
        _save_hdf5(filepath, dict(
            N=N, beta=beta, Omega=Omega, Rb=Rb, delta=delta, epsilon=epsilon,
            neighbor_cutoff=neighbor_cutoff,
            seed=seed, n_ranks=n_ranks, n_trajectories=n_trajectories,
            n_thermalize=n_thermalize, decorrelation_steps=decorrelation_steps,
            string_sites=np.asarray(list(string_sites), dtype=np.int32),
            m_star=int(m_star),
            schedule=str(schedule), direction=str(direction),
            K_values=K_values, results=results,
        ))
        if verbose:
            print(f"[MPI-SSE-STRWORK] saved HDF5 → {filepath}", flush=True)
    return {"K_results": results}


def _save_hdf5(path: str, payload: dict) -> None:
    import datetime

    import h5py

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        pg = f.create_group("params")
        for k in ("N", "beta", "Omega", "Rb", "delta", "epsilon",
                  "neighbor_cutoff", "seed", "n_ranks",
                  "n_trajectories", "n_thermalize", "decorrelation_steps",
                  "m_star", "schedule", "direction"):
            pg.attrs[k] = payload[k]
        pg.attrs["timestamp"] = datetime.datetime.now(
            datetime.timezone.utc).isoformat()
        pg.create_dataset("string_sites", data=payload["string_sites"])
        pg.create_dataset("K_values",
                          data=np.asarray(payload["K_values"], dtype=np.int64))
        rg = f.create_group("results")
        for K, agg in payload["results"].items():
            g = rg.create_group(f"K{int(K)}")
            for k in ("o_c", "log_o_c", "n_eff", "p_max",
                      "zero_weight_fraction", "n_trajectories", "elapsed"):
                g.attrs[k] = agg[k]
            g.create_dataset("log_j_samples", data=agg["log_j_samples"])


def main():
    parser = argparse.ArgumentParser(
        description="MPI driver for the thermal SSE off-diagonal string-work engine")
    parser.add_argument("--lattice", type=str, default="1d_chain",
                        choices=["1d_chain", "kagome_bond", "kagome_bond_triangle"])
    parser.add_argument("--N", type=int, default=0,
                        help="(1d_chain) number of sites")
    parser.add_argument("--nx", type=int, default=6, help="(kagome_bond) cells in x")
    parser.add_argument("--ny", type=int, default=6, help="(kagome_bond) cells in y")
    parser.add_argument("--a", type=float, default=1.0, help="lattice constant")
    parser.add_argument("--Omega", type=float, default=1.0)
    parser.add_argument("--Rb", type=float, default=1.2)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=6.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--neighbor-cutoff", type=int, default=-1)
    parser.add_argument("--boundary", type=str, default="open",
                        choices=["open", "periodic"],
                        help="spatial lattice boundary: open (finite patch) or "
                             "periodic (torus; not valid for kagome_bond_triangle)")
    parser.add_argument("--string-sites", type=str, required=True,
                        help="comma-separated site indices of the string C")
    parser.add_argument("--m-star", type=int, default=0,
                        help="seam slot (default 0 = tau=0; stays valid as M grows)")
    parser.add_argument("--K-values", type=str, default="200",
                        help="comma-separated lambda-schedule segment counts")
    parser.add_argument("--schedule", type=str, default="cosine",
                        choices=["cosine", "linear"])
    parser.add_argument("--direction", type=str, default="forward",
                        choices=["forward", "reverse"])
    parser.add_argument("--n-trajectories", type=int, default=4000,
                        help="total trajectories across ranks")
    parser.add_argument("--n-thermalize", type=int, default=2000)
    parser.add_argument("--equil-progress-every", type=int, default=500)
    parser.add_argument("--decorrelation-steps", type=int, default=50)
    parser.add_argument("--n-topology-sweeps-per-lambda", type=int, default=1)
    parser.add_argument("--n-sse-sweeps-per-lambda", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--filepath", type=str, default=None,
                        help="optional HDF5 output path")
    parser.add_argument("--checkpoint-every-trajectories", type=int, default=0,
                        help="flush log_J samples every N trajectories per rank "
                             "into <checkpoint_dir>/K{K}/rank{r}.h5. 0 = disabled.")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--config-in", type=str, default=None,
                        help="warm-start dir of rank{r}.h5 final configs from a "
                             "previous run at the SAME (N, beta, delta, boundary)")
    parser.add_argument("--config-out", type=str, default=None)
    parser.add_argument("--permute-site-labels",
                        action=argparse.BooleanOptionalAction, default=True)
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

    run_sse_string_work_mpi(
        N=N, beta=args.beta, Omega=args.Omega, Rb=args.Rb,
        delta=args.delta, epsilon=args.epsilon, pos=pos,
        string_sites=string_sites,
        K_values=_parse_int_list(args.K_values),
        schedule=args.schedule,
        n_trajectories=args.n_trajectories,
        n_thermalize=args.n_thermalize,
        decorrelation_steps=args.decorrelation_steps,
        m_star=args.m_star,
        direction=args.direction,
        n_topology_sweeps_per_lambda=args.n_topology_sweeps_per_lambda,
        n_sse_sweeps_per_lambda=args.n_sse_sweeps_per_lambda,
        neighbor_cutoff=args.neighbor_cutoff,
        seed=args.seed, box_vectors=box_vectors,
        filepath=args.filepath,
        checkpoint_every_trajectories=args.checkpoint_every_trajectories,
        checkpoint_dir=ckpt_dir,
        config_in=args.config_in,
        config_out=args.config_out,
        equil_progress_every=args.equil_progress_every,
        permute_site_labels=args.permute_site_labels,
    )


if __name__ == "__main__":
    main()
