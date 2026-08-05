"""Growth-anchor mixing diagnostic: per-sample seam-bit occupancy time series.

Why this exists (2026-08-05): the growth residence-ladder anchor on the
production kagome geometry (6x6 PBC, M=227600, hexagon C={84..89}) does NOT
converge in the sampling window — log O_C drifts +0.6-0.75 per e-fold of
window (1k: -6.68, 8k: -5.50, 32k: -4.46) and the drift rejects a 1/T
transient model at ~3 sigma.  This driver measures the in-sector relaxation
DIRECTLY: it replays the production ladder protocol (same shared lambdas,
same equilibration sequence, same toggle cadence as
``run_growth_residence_ladder``) but records the toggling bit's occupancy at
EVERY sample with no burn-in discard.  Half the ranks start each stage
bit-ON, half bit-OFF: the gap between the two arms' ensemble occupancy
curves p_ON(t | start) decays with the true sector mixing time, and its
functional form (exponential / stretched / log) tells us whether longer
windows can converge the anchor or an extrapolation is required.

Stages listed in --record-stages get the long recorded window (--T samples);
the other stages run a short pass (--short-samples) purely to carry the
worldline through the ladder order (production history fidelity is
approximate there — the short pass replaces the full production window).

Output HDF5 (rank 0): group stage{k}/ with occ (n_ranks, T_k) int8 and
attrs lam, recorded; root attrs start_on (n_ranks,), plus the run params.

Usage (mirrors probe_string_work_runtime.sh geometry):
  $MPIEXEC python -u -m src.mpi.growth_mixing_diag_mpi \
      --out data/growth_mixing_diag_${SLURM_JOB_ID}.h5
"""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    from mpi4py import MPI
except ImportError as exc:
    raise ImportError("mpi4py is required for src.mpi.growth_mixing_diag_mpi") from exc

from src.engines.qaqmc_string_work import QAQMCStringWorkRydberg
from src.mpi.driver_util import rank_seed as _rank_seed
from src.mpi.site_permutation import (permute_rows, resolve_site_permutation,
                                      to_engine)
from src.rydberg.lattices import generate_kagome_bond_lattice, lattice_box_vectors

# Rank-0-tuned shared lambdas from the consistency probes (27119/27125 —
# deterministic tune at seed=7; 27121 production retuned to slightly
# different values, the drift is protocol-independent).
DEFAULT_LAMBDAS = "0.8922,0.8677,0.2497,0.9653,0.6832,0.2177"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=6)
    p.add_argument("--ny", type=int, default=6)
    p.add_argument("--a", type=float, default=4.0)
    p.add_argument("--M", type=int, default=227600)
    p.add_argument("--Omega", type=float, default=1.0)
    p.add_argument("--Rb", type=float, default=2.4)
    p.add_argument("--delta-min", type=float, default=-2.0)
    p.add_argument("--delta-max", type=float, default=6.0)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--delta-groups", type=int, default=600)
    p.add_argument("--string-sites", type=str, default="84,85,86,87,88,89")
    p.add_argument("--stage-lambdas", type=str, default=DEFAULT_LAMBDAS)
    p.add_argument("--record-stages", type=str, default="1,3")
    p.add_argument("--T", type=int, default=48000,
                   help="samples recorded per long stage")
    p.add_argument("--short-samples", type=int, default=500,
                   help="samples for non-recorded (history-carrying) stages")
    p.add_argument("--n-thermalize", type=int, default=50,
                   help="initial thermalization (probe family used 50)")
    p.add_argument("--n-equil-per-stage", type=int, default=200)
    p.add_argument("--n-toggle-attempts", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    seed_r = _rank_seed(args.seed, rank)

    sites = [int(s) for s in args.string_sites.split(",")]
    lambdas = np.asarray([float(x) for x in args.stage_lambdas.split(",")])
    record_stages = {int(s) for s in args.record_stages.split(",")}
    L = len(sites)
    if lambdas.shape != (L,):
        raise ValueError("stage-lambdas length must match string-sites length")

    pos = np.ascontiguousarray(
        generate_kagome_bond_lattice(args.nx, args.ny, args.a), dtype=np.float64)
    N = len(pos)
    box = lattice_box_vectors("kagome_bond", args.nx, args.ny, args.a, N=N)

    site_perm, inv_perm = resolve_site_permutation(
        N, seed_r, True, cfg=None, label="MIXDIAG")
    pos_eng = permute_rows(pos, site_perm)
    sites_eng = [int(s) for s in to_engine(list(sites), inv_perm)]

    if rank == 0:
        print(f"[MIXDIAG] N={N}, M={args.M}, ranks={n_ranks}, sites={sites}, "
              f"lambdas={np.round(lambdas, 4).tolist()}, "
              f"record_stages={sorted(record_stages)}, T={args.T}, "
              f"short={args.short_samples}", flush=True)

    eng = QAQMCStringWorkRydberg(
        N=N, M=args.M, Omega=args.Omega, Rb=args.Rb,
        delta_min=args.delta_min, delta_max=args.delta_max,
        epsilon=args.epsilon, seed=seed_r, pos=pos_eng,
        neighbor_cutoff=None, delta_groups=args.delta_groups, box_vectors=box)
    eng.set_string_sites(sites_eng)
    eng.thermalize(args.n_thermalize, direction="forward")
    core = eng._eng

    start_on = (rank % 2 == 1)
    stage_occ: dict[int, np.ndarray] = {}
    for k in range(L):
        lam = float(lambdas[k])
        base = (1 << k) - 1
        # identical sequence to run_growth_residence_ladder with
        # stage_lambdas given, EXCEPT no n_equil_at_lambda burn — the
        # transient IS the measurement
        core.set_seam_mask_consistent(base)
        for _ in range(args.n_equil_per_stage):
            core.mc_step()
        if start_on:
            core.set_seam_mask_consistent(base | (1 << k))
            for _ in range(max(args.n_equil_per_stage // 4, 10)):
                core.mc_step()
        T_k = args.T if k in record_stages else args.short_samples
        occ = np.empty(T_k, dtype=np.int8)
        t0 = time.perf_counter()
        for i in range(T_k):
            # cluster_update stales the seam snapshots (E14 family) — any
            # path using attempt_string_toggle directly must recompute first
            core.recompute_seam_snapshots()
            for _ in range(args.n_toggle_attempts):
                core.attempt_string_toggle(k, lam)
            core.mc_step()
            occ[i] = (core.seam_mask >> k) & 1
            if rank == 0 and (i + 1) % 4000 == 0:
                el = time.perf_counter() - t0
                print(f"[MIXDIAG] stage {k}: {i + 1}/{T_k} "
                      f"({el:.0f}s, {(i + 1) / el:.1f} samp/s)", flush=True)
        stage_occ[k] = occ
        if rank == 0:
            print(f"[MIXDIAG] stage {k} done: mean occ={occ.mean():.4f}, "
                  f"flips={int(np.abs(np.diff(occ.astype(np.int16))).sum())}",
                  flush=True)

    all_occ = comm.gather(stage_occ, root=0)
    all_start = comm.gather(start_on, root=0)
    if rank == 0:
        import h5py
        with h5py.File(args.out, "w") as f:
            f.attrs["n_ranks"] = n_ranks
            f.attrs["seed"] = args.seed
            f.attrs["M"] = args.M
            f.attrs["N"] = N
            f.attrs["string_sites"] = np.asarray(sites, dtype=np.int32)
            f.attrs["stage_lambdas"] = lambdas
            f.attrs["n_thermalize"] = args.n_thermalize
            f.attrs["n_equil_per_stage"] = args.n_equil_per_stage
            f.attrs["n_toggle_attempts"] = args.n_toggle_attempts
            f.attrs["start_on"] = np.asarray(all_start, dtype=np.int8)
            for k in range(L):
                g = f.create_group(f"stage{k}")
                g.create_dataset(
                    "occ", data=np.stack([r[k] for r in all_occ]),
                    compression="gzip", compression_opts=4)
                g.attrs["lam"] = float(lambdas[k])
                g.attrs["recorded"] = bool(k in record_stages)
        print(f"[MIXDIAG] saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
