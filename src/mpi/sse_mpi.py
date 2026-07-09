"""
MPI-parallel finite-temperature SSE: each rank runs an independent Markov chain
and writes a single self-contained HDF5 file ``<run_dir>/rank{r}.h5`` holding
one ``chunk{i}`` group per bin plus a ``final_config`` group for warm starting.

Storage model (shared with the profile / work drivers, see src.mpi.chunk_io):
``--checkpoint`` is the merged bin==flush size.  Every ``checkpoint`` samples
each rank forms one bin (means of energy / density / mz / n_ops over those
samples) and appends it as the next chunk; a crash loses at most one bin.  The
final spin configuration + operator string + RNG state are saved in the same
file so a later run pointed at this directory (``--config-in``) resumes without
re-thermalizing.

Usage:
    mpirun -np 4 python -m src.mpi.sse_mpi \\
        --lattice kagome_bond_triangle --nx 6 --ny 6 --a 4.0 \\
        --beta 10 --delta 2.0 --Rb 2.4 \\
        --n-equil 5000 --n-samples 50000 --checkpoint 250 \\
        --run-dir data/sse_6x6

Requires: mpi4py, h5py, numpy
"""

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    raise ImportError("mpi4py is required for MPI mode. Install with: pip install mpi4py")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.engines.sse import SSE_Rydberg
from src.mpi.chunk_io import RankChunkWriter, check_config_compat, load_warm_config
from src.mpi.equil_progress import run_equil_with_progress


def _make_pos(lattice, N, nx, ny, a):
    """Build atom positions for the requested lattice; returns (pos, N)."""
    if lattice == "1d_chain":
        from src.rydberg.lattices import generate_1d_chain
        if N <= 0:
            raise ValueError("--N must be > 0 for 1d_chain")
        return np.ascontiguousarray(generate_1d_chain(N, a), dtype=np.float64), N
    if lattice == "kagome_bond":
        from src.rydberg.lattices import generate_kagome_bond_lattice
        pos = np.ascontiguousarray(generate_kagome_bond_lattice(nx, ny, a),
                                   dtype=np.float64)
        return pos, len(pos)
    if lattice == "kagome_bond_triangle":
        from src.rydberg.lattices import generate_kagome_bond_triangle_lattice
        pos = np.ascontiguousarray(
            generate_kagome_bond_triangle_lattice(nx, ny, a), dtype=np.float64)
        return pos, len(pos)
    raise ValueError(f"unsupported lattice {lattice!r}")


def _setup_diag_observables(cpp, lattice, nx, ny, boundary, occ_sf_grid_n,
                            verbose):
    """Configure QAQMC-profile-parity diagonal observables on the SSE engine.

    Returns extra meta.h5 datasets (occ geometry) or None when skipped.
    Skips silently when the lattice has no kagome observable geometry, and
    with a note when the loaded qaqmc_cpp predates the SSE observable API —
    so a driver newer than the deployed .so degrades to the legacy behaviour
    instead of crashing a production run.
    """
    if lattice not in ("kagome_bond", "kagome_bond_triangle"):
        return None
    if not hasattr(cpp, "set_observable_sites"):
        if verbose:
            print("[MPI-SSE] loaded qaqmc_cpp lacks the SSE diagonal-observable "
                  "API — measuring legacy scalars only", flush=True)
        return None

    from src.mpi.qaqmc_mpi import (
        _build_occ2_sf_geometry,
        _build_occ2_sf_geometry_tri,
        _build_occ_q_grid,
        _build_occ_sf_geometry,
        _build_occ_sf_geometry_tri,
        _build_vbs_triangles,
        _build_vbs_triangles_tri,
        _lattice_observables,
    )

    (bulk, loop_sets, string_sets, loop_meta, string_meta,
     vertex_sets, ijk_map) = _lattice_observables(lattice, nx, ny,
                                                  boundary=boundary)
    cpp.set_bulk_sites([int(s) for s in bulk])
    # A_v vertex sets appended after the loops → trailing Z_l size group,
    # exactly like the QAQMC profile driver.
    cpp.set_observable_sites(
        [[int(s) for s in st] for st in list(loop_sets) + list(vertex_sets)],
        [[int(s) for s in st] for st in string_sets])

    if lattice == "kagome_bond_triangle":
        vbs = _build_vbs_triangles_tri(nx, ny, ijk_map)
    else:
        vbs = _build_vbs_triangles(nx, ny)
    if vbs is not None:
        cpp.set_vbs_triangles(*vbs)

    meta_extra = {}
    if int(occ_sf_grid_n) > 0:
        if lattice == "kagome_bond_triangle":
            cell_R, basis, in_bulk = _build_occ_sf_geometry_tri(
                nx, ny, 1.0, ijk_map, bulk)
            cell2_R, basis2 = _build_occ2_sf_geometry_tri(nx, ny, 1.0, ijk_map)
        else:
            cell_R, basis, in_bulk = _build_occ_sf_geometry(
                nx, ny, 1.0, boundary=boundary)
            cell2_R, basis2 = _build_occ2_sf_geometry(nx, ny, 1.0)
        q_points, _ = _build_occ_q_grid(int(occ_sf_grid_n), 1.0)
        cpp.set_occ_sf_site_map(np.asarray(cell_R, dtype=np.float64),
                                np.asarray(basis, dtype=np.int32),
                                np.asarray(in_bulk, dtype=np.int32), 6)
        cpp.set_occ_sf_q_points(np.asarray(q_points, dtype=np.float64))
        cpp.set_occ2_sf_site_map(np.asarray(cell2_R, dtype=np.float64),
                                 np.asarray(basis2, dtype=np.int32), 6)
        meta_extra = dict(
            occ_q_points=np.asarray(q_points, dtype=np.float64),
            occ_cell_R=np.asarray(cell_R, dtype=np.float64),
            occ_basis=np.asarray(basis, dtype=np.int32))

    if verbose:
        loops = [m["size"] for m in loop_meta if m["n_copies"] > 0]
        strings = [m["size"] for m in string_meta if m["n_copies"] > 0]
        occ_note = (f", occ-SF {occ_sf_grid_n}×{occ_sf_grid_n} q-grid"
                    if int(occ_sf_grid_n) > 0 else "")
        print(f"[MPI-SSE] diagonal observables on: bulk={len(bulk)}/{cpp.N}, "
              f"loops s={loops} + A_v({len(vertex_sets)}), strings s={strings}, "
              f"VBS/SS={'on' if vbs is not None else 'off'}{occ_note}",
              flush=True)
    return meta_extra


# Per-chunk occ-SF datasets forwarded from the C++ run() (one super-bin/chunk).
_OCC_CHUNK_KEYS = ("occ_S_full_re", "occ_S_full_im", "occ_S_bulk_re",
                   "occ_S_bulk_im", "occ2_S_re", "occ2_S_im", "occ_nprof")


def run_mpi(*, lattice, N, nx, ny, a, Omega=1.0, delta=0.0, Rb=1.4, beta=10.0,
            epsilon=0.01, seed=42, n_equil=5000, n_samples=50000,
            checkpoint=250, run_dir="data/sse_mpi", neighbor_cutoff=None,
            boundary="open", config_in=None, equil_progress_every=500,
            occ_sf_grid_n=0, n_snapshots=0,
            verbose=True):
    """MPI-parallel SSE with per-rank chunked output and warm start.

    Each rank runs an independent chain (seed = seed + rank*9973), bins its
    per-step scalar observables every ``checkpoint`` samples into a chunk, and
    saves its final configuration for warm starting.  ``n_samples`` is the
    total across ranks; each rank gets ``ceil`` its share.  ``boundary`` selects
    the spatial lattice boundary: 'open' (finite patch) or 'periodic' (torus via
    minimum-image; undefined for the cropped kagome_bond_triangle patch).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    pos, N = _make_pos(lattice, N, nx, ny, a)

    box_vectors = None
    if boundary == "periodic":
        from src.rydberg.lattices import lattice_box_vectors
        box_vectors = lattice_box_vectors(lattice, nx, ny, a, N=N)

    # Per-rank sample budget, rounded up to a whole number of bins.
    base = -(-int(n_samples) // n_ranks)  # ceil so every rank has >= n_samples/n_ranks
    if checkpoint <= 0:
        raise ValueError("--checkpoint (samples per bin) must be > 0")
    n_bins = max(1, -(-base // int(checkpoint)))
    my_n_samples = n_bins * int(checkpoint)

    if verbose and rank == 0:
        print(f"[MPI-SSE] {lattice} N={N}, {n_ranks} ranks, {boundary} boundary, "
              f"{n_bins} bins × {checkpoint} samples = {my_n_samples}/rank "
              f"({n_ranks * my_n_samples} total)", flush=True)

    rank_seed = seed + rank * 9973
    engine = SSE_Rydberg(
        N=N, Omega=Omega, delta=delta, Rb=Rb, beta=beta,
        epsilon=epsilon, seed=rank_seed, pos=pos, use_cpp=True,
        verbose=False, neighbor_cutoff=neighbor_cutoff, box_vectors=box_vectors,
    )
    cpp = engine._cpp_engine
    if cpp is None:
        raise RuntimeError("SSE MPI mode requires the C++ backend")

    # ── Warm start: install a saved configuration, skip thermalization ──────
    if config_in:
        cfg = load_warm_config(config_in, rank, verbose=(rank == 0 and verbose))
        if cfg is None:
            raise FileNotFoundError(f"[warm-start] no rank*.h5 files in {config_in}")
        check_config_compat(cfg, dict(N=int(N), lattice=str(lattice),
                                      boundary=str(boundary)),
                            f"sse rank {rank}")
        cpp.set_config(
            np.ascontiguousarray(cfg["state"], dtype=np.int32),
            np.ascontiguousarray(cfg["op_types"], dtype=np.int32),
            np.ascontiguousarray(cfg["op_sites"], dtype=np.int32))
        rng_state = cfg["attrs"].get("rng_state")
        if rng_state is not None and int(cfg["attrs"].get("rank", -1)) == rank:
            cpp.set_rng_state(rng_state)
        n_equil = 0
        if rank == 0 and verbose:
            print(f"[MPI-SSE] warm start from {config_in} — thermalization skipped",
                  flush=True)

    # ── Equilibration ───────────────────────────────────────────────────────
    comm.Barrier()

    def _advance_equil(n):
        for _ in range(n):
            cpp.mc_step()

    t_equil = run_equil_with_progress(
        _advance_equil, n_equil, label="MPI-SSE", rank=rank,
        print_every=equil_progress_every, verbose=verbose)
    t_equil = comm.reduce(t_equil, op=MPI.MAX, root=0)
    if verbose and rank == 0:
        print(f"[MPI-SSE] Equilibration done in {t_equil:.1f}s (slowest rank)",
              flush=True)

    # ── Diagonal observables (QAQMC-profile parity; needs a new-enough .so) ──
    diag_meta = _setup_diag_observables(cpp, lattice, nx, ny, boundary,
                                        occ_sf_grid_n,
                                        verbose=(verbose and rank == 0))
    supports_new_run = hasattr(cpp, "set_observable_sites")

    # ── Sampling in bins → chunks ───────────────────────────────────────────
    comm.Barrier()
    t0 = time.perf_counter()

    run_attrs = dict(
        lattice=str(lattice), N=int(N), nx=int(nx), ny=int(ny), a=float(a),
        Omega=float(Omega), delta=float(delta),
        Rb=float(Rb), beta=float(beta), epsilon=float(epsilon), seed=int(seed),
        rank_seed=int(rank_seed), n_ranks=int(n_ranks), checkpoint=int(checkpoint),
        n_bins=int(n_bins), samples_per_bin=int(checkpoint),
        neighbor_cutoff=(-1 if neighbor_cutoff is None else int(neighbor_cutoff)),
        boundary=str(boundary),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    run_kwargs = dict(n_snapshots=int(n_snapshots)) if supports_new_run else {}

    with RankChunkWriter(run_dir, rank, run_attrs=run_attrs) as writer:
        for c in range(n_bins):
            raw = cpp.run(n_equil=0, n_samples=int(checkpoint),
                          progress_callback=None, progress_every=1,
                          **run_kwargs)
            e = np.asarray(raw["energies"], dtype=np.float64)
            d = np.asarray(raw["densities"], dtype=np.float64)
            m = np.asarray(raw["mz"], dtype=np.float64)
            no = np.asarray(raw["n_ops"], dtype=np.float64)
            datasets = dict(
                energy=float(e.mean()), energy_sq=float((e**2).mean()),
                density=float(d.mean()), density_sq=float((d**2).mean()),
                mz=float(m.mean()), mz_abs=float(np.abs(m).mean()),
                n_ops=float(no.mean()),
            )
            # Bin means of the diagonal observables (chunk == one bin, same
            # convention as the QAQMC profile chunks).
            if "Z_l" in raw:
                datasets["Z_l"] = np.asarray(raw["Z_l"], np.float64).mean(axis=0)
            if "C_m_l" in raw:
                datasets["C_m_l"] = np.asarray(raw["C_m_l"], np.float64).mean(axis=0)
            if "M_vbs" in raw:
                mv = np.asarray(raw["M_vbs"], np.float64)
                ms = np.asarray(raw["M_ss"], np.float64)
                datasets.update(M_vbs=float(mv.mean()), M_vbs2=float((mv**2).mean()),
                                M_ss=float(ms.mean()), M_ss2=float((ms**2).mean()))
            if "density_bulk" in raw:
                datasets["density_bulk"] = float(
                    np.asarray(raw["density_bulk"], np.float64).mean())
            for key in _OCC_CHUNK_KEYS:
                if key in raw:
                    datasets[key] = np.asarray(raw[key], dtype=np.float32)
            if "snapshots" in raw:
                datasets["snapshots"] = np.asarray(raw["snapshots"], dtype=np.int8)
            writer.write_chunk(c, datasets=datasets,
                               attrs=dict(n_samples=int(checkpoint), M=int(cpp.M)))
            if verbose and rank == 0:
                el = time.perf_counter() - t0
                print(f"[MPI-SSE] rank0 chunk {c + 1}/{n_bins} "
                      f"(⟨n⟩={d.mean():.4f}, {el:.0f}s)", flush=True)

    # Warm-start config (spin state + operator string + RNG state) →
    # <run_dir>/configs/rank{r}.h5, separate from the observable chunk files so
    # it can be reclaimed independently (rm -rf <run_dir>/configs).
    with RankChunkWriter(os.path.join(run_dir, "configs"), rank) as cfg_writer:
        cfg_writer.write_final_config(
            datasets=dict(
                state=np.asarray(cpp.state, dtype=np.int32),
                op_types=np.asarray(cpp.op_types, dtype=np.int32),
                op_sites=np.asarray(cpp.op_sites, dtype=np.int32)),
            attrs=dict(N=int(N), lattice=str(lattice), boundary=str(boundary),
                       beta=float(beta), seed=int(rank_seed),
                       rng_state=cpp.get_rng_state()))

    t_sample = comm.reduce(time.perf_counter() - t0, op=MPI.MAX, root=0)
    comm.Barrier()

    # One-time run metadata (geometry + params) from rank 0.
    if rank == 0:
        import h5py
        with h5py.File(os.path.join(run_dir, "meta.h5"), "w") as f:
            for k, v in run_attrs.items():
                f.attrs[k] = v
            f.attrs["n_samples_total"] = int(n_ranks * my_n_samples)
            f.create_dataset("pos", data=np.asarray(pos, dtype=np.float64))
            for k, v in (diag_meta or {}).items():
                f.create_dataset(k, data=v)
        if verbose:
            print(f"[MPI-SSE] Sampling done in {t_sample:.1f}s; "
                  f"{n_bins} chunk(s)/rank → {run_dir}", flush=True)
    return run_dir


def combine_run(run_dir, burn_in_fraction=0.5):
    """Read all rank files and return burn-in-trimmed observable means/errors.

    Convenience aggregator (rank-0 / post-processing use).  Each chunk is one
    bin; per-rank burn-in drops the first ``burn_in_fraction`` of chunks.
    """
    from src.mpi.chunk_io import iter_rank_chunks
    keys = ("energy", "density", "mz", "mz_abs", "n_ops")
    acc = {k: [] for k in keys}
    for _r, _c, g in iter_rank_chunks(run_dir, burn_in_fraction=burn_in_fraction):
        for k in keys:
            if k in g:
                acc[k].append(float(g[k][()]))
    out = {}
    for k, vals in acc.items():
        if vals:
            arr = np.asarray(vals)
            out[k] = float(arr.mean())
            out[k + "_err"] = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    out["n_bins_kept"] = len(acc["density"])
    return out


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MPI-parallel SSE simulation")
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--lattice", type=str, default="kagome_bond_triangle",
                        choices=["1d_chain", "kagome_bond", "kagome_bond_triangle"])
    parser.add_argument("--N", type=int, default=0, help="(1d_chain) site count")
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--a", type=float, default=4.0, help="lattice constant")
    parser.add_argument("--Omega", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--Rb", type=float, default=2.4)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-equil", type=int, default=5000)
    parser.add_argument("--equil-progress-every", type=int, default=500,
                        help="print rank-0 equilibration progress every N steps "
                             "(<= 0 disables intermediate prints)")
    parser.add_argument("--n-samples", type=int, default=50000,
                        help="total samples across ranks")
    parser.add_argument("--checkpoint", type=int, default=250,
                        help="samples per bin == chunk flush size (merged)")
    parser.add_argument("--neighbor-cutoff", type=int, default=-1)
    parser.add_argument("--boundary", type=str, default="open",
                        choices=["open", "periodic"],
                        help="spatial lattice boundary: open (finite patch) or "
                             "periodic (torus; not valid for kagome_bond_triangle)")
    parser.add_argument("--occ-sf-grid-n", type=int, default=0,
                        help="side length of the 2D BZ q-grid for the occupation "
                             "SF matrix (one super-bin per chunk). 0 = disabled.")
    parser.add_argument("--n-snapshots", type=int, default=0,
                        help="full-state snapshots per rank per chunk (int8, "
                             "for real-space excitation-pattern plots). "
                             "0 = disabled.")
    parser.add_argument("--run-dir", type=str, default="data/sse_mpi",
                        help="output directory; each rank writes rank{r}.h5 here")
    parser.add_argument("--config-in", type=str, default=None,
                        help="warm-start dir of a previous run (rank{r}.h5 with "
                             "final_config); when given, thermalization is skipped")
    args = parser.parse_args()

    config = vars(args)
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    neighbor_cutoff = None if int(config["neighbor_cutoff"]) < 0 else int(config["neighbor_cutoff"])

    run_mpi(
        lattice=config["lattice"], N=config["N"], nx=config["nx"], ny=config["ny"],
        a=config["a"], Omega=config["Omega"], delta=config["delta"], Rb=config["Rb"],
        beta=config["beta"], epsilon=config["epsilon"], seed=config["seed"],
        n_equil=config["n_equil"], n_samples=config["n_samples"],
        checkpoint=config["checkpoint"], run_dir=config["run_dir"],
        neighbor_cutoff=neighbor_cutoff, boundary=config.get("boundary", "open"),
        config_in=config.get("config_in"),
        equil_progress_every=config.get("equil_progress_every", 500),
        occ_sf_grid_n=config.get("occ_sf_grid_n", 0),
        n_snapshots=config.get("n_snapshots", 0),
    )


if __name__ == "__main__":
    main()
