"""Probe: does the fixed site-scan order pick the stripe pattern in QAQMC sweeps?

Background (2026-07-09 analysis of qaqmc_profile_M=2760000_6x6 + SSE β=20):
64 independent-seed sweep chains froze into essentially the SAME real-space
M-stripe pattern (cross-chain per-site ⟨n⟩ std ≈ 0.24, sites 0.00–0.94), while
the SSE equilibrium ensemble at the same point is symmetric (std ≈ 0.05,
M1:M2:M3 domains ≈ 1:1:1).  Geometry, allowed momenta and the measurement
chain were all verified C3-fair, so the selection must be kinetic.  The one
deterministic structure shared by every rank is the update scan order
(slots p ascending; sites in (j*nx+i)*6+k raster order).

Test: run n_chains short sweeps twice —
  control : all chains use the canonical site labelling (like production),
  permuted: each chain randomly permutes site labels before building the
            engine (identical physics, different scan geometry), snapshots
            un-permuted back to canonical labels before analysis.

Verdict logic on the cross-chain mean profile std and domain counts:
  control locked + permuted unlocked  -> scan order causes the selection
  both locked (same real-space pattern) -> some real-space mechanism (not labels)
  both unlocked -> lock needs slower ramps / more equil than this probe

Usage (bare server or inside an allocation)::

    python scripts/experiments/scan_order_bias_probe.py \
        --n-chains 16 --workers 10 --M 552000 --n-equil 1000 --n-samples 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

NX = NY = 6
A = 4.0
N_SNAP = 4


def _run_chain(args):
    """One sweep chain; returns (n_snap, N) snapshots in CANONICAL site labels."""
    (chain, M, n_equil, n_samples, permute, seed0) = args
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from src.engines.qaqmc import QAQMC_Rydberg
    from src.rydberg.lattices import (generate_kagome_bond_lattice,
                                      lattice_box_vectors)

    pos = generate_kagome_bond_lattice(NX, NY, A)
    N = len(pos)
    box = np.asarray(lattice_box_vectors("kagome_bond", NX, NY, A, N=N), float)

    if permute:
        perm = np.random.RandomState(910 + chain).permutation(N)
    else:
        perm = np.arange(N)
    pos_p = np.ascontiguousarray(pos[perm])   # engine site i == canonical perm[i]

    eng = QAQMC_Rydberg(N=N, M=M, Omega=1.0, Rb=2.4,
                        delta_min=-2.0, delta_max=6.0, pos=pos_p,
                        epsilon=0.01, seed=seed0 + 9973 * chain, verbose=False,
                        use_cpp=True, omp_threads=0, neighbor_cutoff=None,
                        delta_groups=600, box_vectors=box)
    cpp = eng._cpp_engine

    # snapshot at forward δ≈5.5 on a 100-point profile grid
    profile_step = cpp.M_total // 100
    sched = np.asarray(cpp.delta_schedule)
    p_idx = np.array([(k + 1) * profile_step - 1 for k in range(100)])
    prof_delta = sched[p_idx]
    turn = int(np.argmax(prof_delta))
    snap_pt = int(np.argmin(np.abs(prof_delta[:turn + 1] - 5.5)))
    cpp.set_snapshot_point_indices(np.array([snap_pt], dtype=np.int32))

    cpp.run(n_equil, 0)
    res = cpp.run_profile(0, n_samples, 1, 1, 1, profile_step, n_samples,
                          None, 1, N_SNAP, 0)
    snaps = np.asarray(res["snapshots"], dtype=np.int8)[:, 0, :]  # (n_snap, N)

    # back to canonical labels: canonical site perm[i] = engine site i
    out = np.zeros_like(snaps)
    out[:, perm] = snaps
    return out


def _m_lambdas(states):
    """Per-state λ(M1), λ(M2), λ(M3), per-cell normalised (canonical labels)."""
    from src.mpi.qaqmc_mpi import _build_occ_sf_geometry
    cell_R, basis, _ = _build_occ_sf_geometry(NX, NY, 1.0, boundary="periodic")
    cell_R = np.asarray(cell_R); basis = np.asarray(basis)
    b1 = 2 * np.pi * np.array([1.0, -1 / np.sqrt(3)])
    b2 = 2 * np.pi * np.array([0.0, 2 / np.sqrt(3)])
    out = []
    for q in (b1 / 2, b2 / 2, (b1 + b2) / 2):
        ph = np.exp(1j * (cell_R @ q))
        s = np.stack([((states * (basis == aa)) * ph[None, :]).sum(axis=1)
                      for aa in range(6)], axis=1)          # (n_states, 6)
        out.append((np.abs(s) ** 2).sum(axis=1) / 36.0)
    return np.stack(out, axis=1)                             # (n_states, 3)


def run_group(tag, permute, args):
    t0 = time.perf_counter()
    jobs = [(c, args.M, args.n_equil, args.n_samples, permute, args.seed0)
            for c in range(args.n_chains)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        snaps = list(ex.map(_run_chain, jobs))               # n_chains × (n_snap, N)
    snaps = np.stack(snaps)                                  # (chains, n_snap, N)

    mean_profile = snaps.reshape(-1, snaps.shape[-1]).mean(axis=0)
    lam = _m_lambdas(snaps.reshape(-1, snaps.shape[-1]))
    lam_chain = lam.reshape(args.n_chains, N_SNAP, 3).mean(axis=1)
    dom = np.array(["M1", "M2", "M3"])[np.argmax(lam_chain, axis=1)]
    strong = lam_chain.max(axis=1) > 1.0
    counts = {m: int(((dom == m) & strong).sum()) for m in ("M1", "M2", "M3")}
    counts["weak"] = int((~strong).sum())

    print(f"\n== {tag} ({time.perf_counter() - t0:.0f}s) ==")
    print(f"  cross-chain mean-profile: ⟨n⟩={mean_profile.mean():.3f}  "
          f"site std={mean_profile.std():.3f}  "
          f"min/max={mean_profile.min():.2f}/{mean_profile.max():.2f}")
    print(f"  chain domains (λ>1): {counts}")
    print(f"  per-chain λ(M) mean: {np.round(lam_chain.mean(axis=0), 2)}")
    return mean_profile.std(), counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-chains", type=int, default=16)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--M", type=int, default=552000)
    ap.add_argument("--n-equil", type=int, default=1000)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--pilot", action="store_true",
                    help="2 chains, control only — check the probe orders at all")
    args = ap.parse_args()

    if args.pilot:
        args.n_chains = 2
        run_group("pilot (control)", False, args)
        return

    std_c, cnt_c = run_group("control (canonical labels)", False, args)
    std_p, cnt_p = run_group("permuted (per-chain labels)", True, args)

    print("\n== verdict ==")
    locked_c = std_c > 0.12
    locked_p = std_p > 0.12
    if locked_c and not locked_p:
        print("  control locked, permuted unlocked -> pattern follows the site")
        print("  LABELLING: the update scan order causes the domain selection.")
    elif locked_c and locked_p:
        print("  both locked -> real-space mechanism independent of labels.")
    else:
        print("  control not locked at this ramp speed -> probe inconclusive;")
        print("  increase --M / --n-equil.")


if __name__ == "__main__":
    main()
