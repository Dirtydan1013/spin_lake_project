"""MPI SSE thermal-entropy ladder: E(beta) on a log grid for S(T) integration.

Implements the data side of Wang-Pollet PRL 134, 086601 Eq. (7),

    S(T) = S(inf) - int_T^inf [E(T') - E(T)] / T'^2 dT'
         = N ln2 + ln Z(beta) - ln Z(0) + beta E(beta),

by measuring E(beta) on an ascending log-spaced beta ladder.  Each rung
warm-starts from the previous rung's configuration (the operator string is
valid at any beta; only the acceptance weights change), so equilibration is
paid once at the hot end.  Per-rung output uses the standard chunked format
(``<run_dir>/beta_{k:03d}/rank{r}.h5`` + ``meta.h5``), one bin per chunk, so
the usual analysis helpers work per rung.  The top-level ``meta.h5`` stores
the ladder (betas) and the analytic beta -> 0 anchor:

    E_inf   = -delta N/2 + sum_{i<j} V_ij / 4
    varH    = <H^2> - <H>^2 at infinite T   (exact, see _high_t_anchor)

so the integration can start at beta_min with ln Z(beta_min) - N ln2 =
-beta_min E_inf + beta_min^2 varH / 2 + O(beta_min^3).

Integration + plots: ``plots/plot_sse/plot_entropy.py``.

Usage (mirrors sse_mpi)::

    mpiexec -n 64 python -m src.mpi.sse_entropy_mpi --lattice kagome_bond \
        --nx 6 --ny 6 --a 4.0 --delta 4.25 --Rb 2.4 --boundary periodic \
        --beta-min 3e-4 --beta-max 40 --n-beta 60 \
        --n-samples 128000 --checkpoint 2000 --run-dir data/sse_entropy_...
"""

import argparse
import datetime
import json
import os
import time

import numpy as np
from mpi4py import MPI

from src.mpi.chunk_io import RankChunkWriter
from src.mpi.sse_mpi import _make_pos


def _min_image_vij(pos, Omega, Rb, box):
    """Full (no cutoff) V_ij = Omega (Rb/r)^6 with optional minimum image."""
    pos = np.asarray(pos, dtype=np.float64)
    N = pos.shape[0]
    d = pos[:, None, :] - pos[None, :, :]                      # (N, N, dim)
    if box is not None:
        box = np.asarray(box, dtype=np.float64)
        best = None
        rng = (-1, 0, 1)
        for m in rng:
            for n in rng if box.shape[0] > 1 else (0,):
                shift = m * box[0] + (n * box[1] if box.shape[0] > 1 else 0.0)
                r = np.linalg.norm(d + shift, axis=-1)
                best = r if best is None else np.minimum(best, r)
        r = best
    else:
        r = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(r, np.inf)
    return Omega * (Rb / r) ** 6                               # (N, N), sym


def _high_t_anchor(vij, delta, Omega, N):
    """(E_inf, varH) at beta = 0 for H = -(O/2)X - d*sum n + sum V n n.

    n_i are iid Bernoulli(1/2) at infinite T; the Omega term is traceless and
    uncorrelated with the diagonal part.  Exact subset-overlap algebra:
      var(D) = N d^2/4                                (single x single)
             + (1/4) sum_{i<j} V_ij (a_i + a_j)       (single x pair, a = -d)
             + (3/16) sum_{i<j} V_ij^2                (pair == pair)
             + (1/16) sum_i (row_i^2 - sq_i)          (pairs sharing one site)
    with row_i = sum_j V_ij, sq_i = sum_j V_ij^2.
    """
    iu = np.triu_indices(N, k=1)
    v_pairs = vij[iu]
    e_inf = -delta * N / 2.0 + v_pairs.sum() / 4.0
    row = vij.sum(axis=1)
    sq = (vij ** 2).sum(axis=1)
    var_d = (N * delta**2 / 4.0
             + 0.25 * (-2.0 * delta) * v_pairs.sum()
             + (3.0 / 16.0) * (v_pairs ** 2).sum()
             + (1.0 / 16.0) * (row ** 2 - sq).sum())
    return float(e_inf), float(var_d + Omega**2 * N / 4.0)


def run_ladder(*, lattice, N, nx, ny, a, Omega=1.0, delta=4.25, Rb=2.4,
               epsilon=0.01, seed=42, boundary="open",
               beta_min=3e-4, beta_max=40.0, n_beta=60, betas=None,
               n_equil0=2000, n_equil_warm=500,
               n_samples=128000, checkpoint=2000,
               run_dir="data/sse_entropy", verbose=True):
    import qaqmc_cpp

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    pos, N = _make_pos(lattice, N, nx, ny, a)
    box = None
    if boundary == "periodic":
        from src.rydberg.lattices import lattice_box_vectors
        box = lattice_box_vectors(lattice, nx, ny, a, N=N)

    if betas is None:
        betas = np.geomspace(beta_min, beta_max, int(n_beta))
    betas = np.sort(np.asarray(betas, dtype=np.float64))

    base = -(-int(n_samples) // n_ranks)
    n_bins = max(1, -(-base // int(checkpoint)))
    my_n = n_bins * int(checkpoint)
    rank_seed = seed + rank * 9973

    vij = _min_image_vij(pos, Omega, Rb, box)
    e_inf, var_h = _high_t_anchor(vij, delta, Omega, N)

    if verbose and rank == 0:
        print(f"[SSE-entropy] {lattice} N={N} δ={delta:g} {boundary}, "
              f"{len(betas)} rungs β∈[{betas[0]:g},{betas[-1]:g}], "
              f"{n_ranks} ranks × {n_bins}×{checkpoint} samples/rung; "
              f"E_inf/N={e_inf / N:.5f}, varH/N={var_h / N:.3f}", flush=True)

    common = dict(lattice=str(lattice), N=int(N), nx=int(nx), ny=int(ny),
                  a=float(a), Omega=float(Omega), delta=float(delta),
                  Rb=float(Rb), epsilon=float(epsilon), seed=int(seed),
                  boundary=str(boundary), n_ranks=int(n_ranks),
                  checkpoint=int(checkpoint), n_bins=int(n_bins),
                  samples_per_bin=int(checkpoint), neighbor_cutoff=-1)

    t0 = time.perf_counter()
    state = types = sites = None
    for k, beta in enumerate(betas):
        eng = qaqmc_cpp.SSEEngine(
            N=N, Omega=Omega, delta=delta, Rb=Rb, beta=float(beta),
            epsilon=epsilon, seed=rank_seed + 131 * k,
            pos=np.ascontiguousarray(pos, dtype=np.float64),
            box_vectors=box)
        if state is not None:
            eng.set_config(state, types, sites)
        n_eq = n_equil0 if k == 0 else n_equil_warm
        rung_dir = os.path.join(run_dir, f"beta_{k:03d}")
        attrs = dict(common, beta=float(beta), rung=int(k), n_equil=int(n_eq),
                     timestamp=datetime.datetime.now(
                         datetime.timezone.utc).isoformat())
        with RankChunkWriter(rung_dir, rank, run_attrs=attrs) as writer:
            first = True
            for c in range(n_bins):
                raw = eng.run(n_equil=n_eq if first else 0,
                              n_samples=int(checkpoint))
                first = False
                e = np.asarray(raw["energies"], dtype=np.float64)
                no = np.asarray(raw["n_ops"], dtype=np.float64)
                d = np.asarray(raw["densities"], dtype=np.float64)
                writer.write_chunk(c, datasets=dict(
                    energy=float(e.mean()), energy_sq=float((e**2).mean()),
                    density=float(d.mean()), n_ops=float(no.mean())),
                    attrs=dict(n_samples=int(checkpoint), M=int(eng.M)))
        state = np.ascontiguousarray(eng.state, dtype=np.int32)
        types = np.ascontiguousarray(eng.op_types, dtype=np.int32)
        sites = np.ascontiguousarray(eng.op_sites, dtype=np.int32)
        if rank == 0:
            import h5py
            with h5py.File(os.path.join(rung_dir, "meta.h5"), "w") as f:
                for kk, vv in attrs.items():
                    f.attrs[kk] = vv
                f.create_dataset("pos", data=np.asarray(pos, np.float64))
            if verbose:
                print(f"[SSE-entropy] rung {k + 1}/{len(betas)} β={beta:g} "
                      f"done (E/N={np.mean(e) / N:.5f}, M={eng.M}, "
                      f"{time.perf_counter() - t0:.0f}s)", flush=True)

    comm.Barrier()
    if rank == 0:
        import h5py
        with h5py.File(os.path.join(run_dir, "meta.h5"), "w") as f:
            for kk, vv in common.items():
                f.attrs[kk] = vv
            f.attrs["kind"] = "sse_entropy_ladder"
            f.attrs["E_inf"] = e_inf
            f.attrs["varH_inf"] = var_h
            f.attrs["n_samples_total"] = int(n_ranks * my_n)
            f.create_dataset("betas", data=betas)
            f.create_dataset("pos", data=np.asarray(pos, np.float64))
        if verbose:
            print(f"[SSE-entropy] ladder done in "
                  f"{time.perf_counter() - t0:.1f}s → {run_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="SSE thermal-entropy beta ladder (Wang-Pollet Eq. 7 data)")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON file overriding the CLI args")
    parser.add_argument("--lattice", type=str, default="kagome_bond",
                        choices=["1d_chain", "kagome_bond",
                                 "kagome_bond_triangle"])
    parser.add_argument("--N", type=int, default=0)
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--a", type=float, default=4.0)
    parser.add_argument("--Omega", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=4.25)
    parser.add_argument("--Rb", type=float, default=2.4)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary", type=str, default="open",
                        choices=["open", "periodic"])
    parser.add_argument("--beta-min", type=float, default=3e-4)
    parser.add_argument("--beta-max", type=float, default=40.0)
    parser.add_argument("--n-beta", type=int, default=60)
    parser.add_argument("--betas", type=str, default=None,
                        help="explicit comma-separated beta list (overrides "
                             "the geometric grid)")
    parser.add_argument("--n-equil0", type=int, default=2000,
                        help="equilibration sweeps for the first (hottest) rung")
    parser.add_argument("--n-equil-warm", type=int, default=500,
                        help="re-equilibration sweeps per warm-started rung")
    parser.add_argument("--n-samples", type=int, default=128000,
                        help="total samples per rung across ranks")
    parser.add_argument("--checkpoint", type=int, default=2000)
    parser.add_argument("--run-dir", type=str, default="data/sse_entropy")
    args = parser.parse_args()

    config = vars(args)
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    betas = None
    if config.get("betas"):
        betas = [float(x) for x in str(config["betas"]).split(",")]

    run_ladder(
        lattice=config["lattice"], N=config["N"], nx=config["nx"],
        ny=config["ny"], a=config["a"], Omega=config["Omega"],
        delta=config["delta"], Rb=config["Rb"], epsilon=config["epsilon"],
        seed=config["seed"], boundary=config["boundary"],
        beta_min=config["beta_min"], beta_max=config["beta_max"],
        n_beta=config["n_beta"], betas=betas,
        n_equil0=config["n_equil0"], n_equil_warm=config["n_equil_warm"],
        n_samples=config["n_samples"], checkpoint=config["checkpoint"],
        run_dir=config["run_dir"],
    )


if __name__ == "__main__":
    main()
