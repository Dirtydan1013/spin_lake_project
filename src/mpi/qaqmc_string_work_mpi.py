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

from src.engines.qaqmc_string_work import QAQMCStringWorkRydberg, cosine_schedule
from src.mpi.kp_tee_common import write_rank_chunk_h5
from src.rydberg.lattices import (
    generate_1d_chain,
    generate_kagome_bond_lattice,
)


def _rank_seed(seed: int, rank: int) -> int:
    return int(seed) + 9973 * int(rank)


def _aggregate_log_j(log_j: np.ndarray, direction: str) -> dict:
    """log-sum-exp aggregation of Jarzynski samples into O_C (document §33)."""
    n = int(log_j.size)
    finite = np.isfinite(log_j)
    if not np.any(finite):
        return dict(o_c=0.0, log_o_c=-math.inf, n_eff=0.0, p_max=0.0,
                    zero_weight_fraction=1.0, n_trajectories=n)
    max_log = float(log_j[finite].max())
    weights = np.zeros(n, dtype=np.float64)
    weights[finite] = np.exp(log_j[finite] - max_log)
    sum_w = float(weights.sum())
    log_mean_j = max_log + math.log(sum_w / n)
    log_o_c = log_mean_j if direction == "forward" else -log_mean_j
    p = weights / sum_w
    return dict(
        o_c=math.exp(log_o_c), log_o_c=log_o_c,
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
                        seed: int = 7,
                        filepath: str | None = None,
                        checkpoint_every_trajectories: int = 0,
                        checkpoint_dir: str | None = None,
                        verbose: bool = True) -> dict | None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    base = n_trajectories // n_ranks
    rem = n_trajectories % n_ranks
    my_n = base + (1 if rank < rem else 0)

    if rank == 0 and verbose:
        print(f"[MPI-STRWORK] N={N}, M={M}, ranks={n_ranks}, "
              f"total_trajectories={n_trajectories}, per-rank≈{base}, "
              f"string_sites={list(string_sites)}, K_values={K_values}, "
              f"schedule={schedule}, direction={direction}", flush=True)

    results: dict[int, dict] = {}
    for K in K_values:
        comm.Barrier()
        t0 = time.perf_counter()

        eng = QAQMCStringWorkRydberg(
            N=N, M=M, Omega=Omega, Rb=Rb,
            delta_min=delta_min, delta_max=delta_max,
            epsilon=epsilon, seed=_rank_seed(seed, rank),
            pos=pos,
            neighbor_cutoff=(None if neighbor_cutoff < 0 else neighbor_cutoff),
            delta_groups=delta_groups,
        )
        eng.set_string_sites(list(string_sites), m_star)
        if schedule == "cosine":
            eng.set_lambda_schedule(cosine_schedule(int(K)))
        else:
            eng.set_lambda_schedule(np.linspace(0.0, 1.0, int(K) + 1))
        eng.thermalize(n_thermalize, direction=direction)

        ckpt = int(checkpoint_every_trajectories) if checkpoint_dir else 0
        if ckpt > 0 and my_n > 0:
            # Chunked sampling: run_trajectories resets the seam sector per
            # trajectory, so repeated calls continue the same chain and are
            # statistically identical to a single long call.
            k_dir = os.path.join(checkpoint_dir, f"K{K}")
            parts = []
            done = 0
            c = 0
            while done < my_n:
                n_chunk = min(ckpt, my_n - done)
                part = eng.run_trajectories(
                    n_chunk, decorrelation_steps,
                    n_topology_sweeps_per_lambda=n_topology_sweeps_per_lambda,
                    n_qaqmc_sweeps_per_lambda=n_qaqmc_sweeps_per_lambda,
                    direction=direction)
                parts.append(part.log_j_samples)
                done += n_chunk
                write_rank_chunk_h5(
                    k_dir, rank, c,
                    datasets=dict(log_j_samples=part.log_j_samples),
                    attrs=dict(K=int(K), n_trajectories=int(n_chunk),
                               trajectories_cumulative=int(done),
                               my_n_trajectories=int(my_n),
                               direction=str(direction), seed=int(seed)),
                )
                c += 1
                if rank == 0 and verbose:
                    print(f"[MPI-STRWORK] K={K} rank0 chunk {c} written "
                          f"({done}/{my_n} trajectories)", flush=True)
            local_log_j = (np.concatenate(parts) if parts
                           else np.empty(0, dtype=np.float64))
        else:
            local = eng.run_trajectories(
                my_n, decorrelation_steps,
                n_topology_sweeps_per_lambda=n_topology_sweeps_per_lambda,
                n_qaqmc_sweeps_per_lambda=n_qaqmc_sweeps_per_lambda,
                direction=direction)
            local_log_j = np.asarray(local.log_j_samples, dtype=np.float64)

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

    if rank != 0:
        return None

    if filepath:
        _save_hdf5(filepath, dict(
            N=N, M=M, Omega=Omega, Rb=Rb,
            delta_min=delta_min, delta_max=delta_max, epsilon=epsilon,
            neighbor_cutoff=neighbor_cutoff, delta_groups=delta_groups,
            seed=seed, n_ranks=n_ranks, n_trajectories=n_trajectories,
            n_thermalize=n_thermalize, decorrelation_steps=decorrelation_steps,
            string_sites=np.asarray(list(string_sites), dtype=np.int32),
            m_star=(-1 if m_star is None else int(m_star)),
            schedule=str(schedule), direction=str(direction),
            K_values=K_values, results=results,
        ))
        if verbose:
            print(f"[MPI-STRWORK] saved HDF5 → {filepath}", flush=True)
    return {"K_results": results}


def _save_hdf5(path: str, payload: dict) -> None:
    import datetime

    import h5py

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        pg = f.create_group("params")
        for k in ("N", "M", "Omega", "Rb", "delta_min", "delta_max", "epsilon",
                  "neighbor_cutoff", "delta_groups", "seed", "n_ranks",
                  "n_trajectories", "n_thermalize", "decorrelation_steps",
                  "m_star", "schedule", "direction"):
            pg.attrs[k] = payload[k]
        pg.attrs["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        pg.create_dataset("string_sites", data=payload["string_sites"])
        pg.create_dataset("K_values",
                          data=np.asarray(payload["K_values"], dtype=np.int64))
        rg = f.create_group("K_results")
        for K, res in payload["results"].items():
            sg = rg.create_group(f"K{int(K)}")
            for key in ("o_c", "log_o_c", "n_eff", "p_max",
                        "zero_weight_fraction", "n_trajectories", "elapsed"):
                sg.attrs[key] = res[key]
            sg.create_dataset("log_j_samples", data=res["log_j_samples"],
                              compression="gzip")


def _parse_int_list(text: str) -> list[int]:
    return [int(tok) for tok in text.replace(";", ",").split(",") if tok.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="MPI driver for the QAQMC off-diagonal string-work engine")
    parser.add_argument("--lattice", type=str, default="1d_chain",
                        choices=["1d_chain", "kagome_bond"])
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
    parser.add_argument("--delta-groups", type=int, default=600)
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
    args = parser.parse_args()

    if args.lattice == "1d_chain":
        if args.N <= 0:
            raise ValueError("--N must be >0 for 1d_chain")
        pos = np.asarray(generate_1d_chain(args.N, args.a), dtype=np.float64)
        N = args.N
    else:
        pos = np.ascontiguousarray(
            generate_kagome_bond_lattice(args.nx, args.ny, args.a), dtype=np.float64)
        N = len(pos)

    string_sites = _parse_int_list(args.string_sites)
    if any(s < 0 or s >= N for s in string_sites):
        raise ValueError(f"string sites out of range [0, {N})")

    ckpt_dir = args.checkpoint_dir
    if int(args.checkpoint_every_trajectories) > 0 and ckpt_dir is None:
        if args.filepath:
            base = (args.filepath[:-3] if args.filepath.endswith(".h5")
                    else args.filepath)
            ckpt_dir = base + "_chunks"
        else:
            raise ValueError("--checkpoint-every-trajectories requires "
                             "--checkpoint-dir or --filepath")

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
        seed=args.seed,
        filepath=args.filepath,
        checkpoint_every_trajectories=args.checkpoint_every_trajectories,
        checkpoint_dir=ckpt_dir,
    )


if __name__ == "__main__":
    main()
