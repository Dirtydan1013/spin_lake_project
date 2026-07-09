"""Real-space excitation patterns from stored full-state snapshots.

Draws the lattice (atoms + nearest-neighbour guide bonds) with Rydberg-excited
atoms filled in, one column per snapshot δ point and one row per sampled
snapshot.  Snapshots are pooled from all ranks/chunks (each chunk stores one
per rank) and sampled evenly from that pool.

Usage::

    python plots/plot_diagonal/plot_snapshots.py [--run_dir data/M=...] \
        [--n_samples 3] [--deltas 3.5 4.0 4.5] [--seed 0] [--out plots/foo.png]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_data import (default_fig_path, latest_run_dir, load_meta,
                          load_run_attrs, resolve_point_indices, stack_chunks)


def _nn_bonds(pos, tol=1e-6):
    """(n_bonds, 2) index pairs of the first distance shell (guide lines only)."""
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    iu = np.triu_indices(len(pos), k=1)
    d_min = dist[iu].min()
    keep = dist[iu] <= d_min * (1.0 + tol)
    return np.column_stack([iu[0][keep], iu[1][keep]])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", default=None,
                        help="chunked run dir (default: newest data/M=*_*)")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="snapshot rows (drawn evenly from the pooled "
                             "rank×chunk snapshots)")
    parser.add_argument("--deltas", type=float, nargs="+", default=None,
                        help="subset of the snapshot δ points (default: all)")
    parser.add_argument("--request_deltas", type=float, nargs="+", default=None,
                        help="the --snapshot_deltas the RUN used, for old runs "
                             "whose meta.h5 lacks snap_pt_indices")
    parser.add_argument("--seed", type=int, default=0,
                        help="which snapshot of the pool the sampling starts at")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir()
    meta = load_meta(run_dir)
    attrs = load_run_attrs(run_dir)
    pos = np.asarray(meta["pos"], dtype=np.float64)

    pt_idx = resolve_point_indices(meta, "snap", args.request_deltas)
    pt_delta = meta["prof_delta"][pt_idx]
    # requested δ may have snapped to the backward (δ-decreasing) ramp
    turn = int(np.argmax(meta["prof_delta"]))
    pt_back = pt_idx > turn

    snaps = stack_chunks(run_dir, ["snapshots"])["snapshots"]
    if snaps is None:
        raise KeyError("run has no snapshots datasets (--n_snapshots 0?)")
    # (pool, n_snap_pts, N) — pool = all ranks × chunks
    if snaps.shape[1] != len(pt_delta):
        raise RuntimeError(
            f"stored snapshot points ({snaps.shape[1]}) != resolved δ points "
            f"({len(pt_delta)}); pass --request_deltas with the run's "
            f"--snapshot_deltas values")

    if args.deltas is not None:
        cols = list({int(np.argmin(np.abs(pt_delta - d))) for d in args.deltas})
    else:
        cols = list(range(len(pt_delta)))
    # columns left→right by δ value (storage order is by profile-point index,
    # which puts backward-ramp points after the largest forward δ)
    cols.sort(key=lambda pt: pt_delta[pt])

    pool = snaps.shape[0]
    n_rows = min(args.n_samples, pool)
    rows = (np.round(np.linspace(0, pool - 1, n_rows)).astype(int)
            + args.seed) % pool

    bonds = _nn_bonds(pos)
    pad = 0.05 * (pos.max(axis=0) - pos.min(axis=0))

    n_cols = len(cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.4 * n_cols + 0.6, 2.3 * n_rows + 0.8),
                             squeeze=False)
    nx, ny = int(attrs["nx"]), int(attrs["ny"])
    fig.suptitle(f"{attrs.get('lattice', '?')} {nx}×{ny} — snapshot excitation "
                 f"patterns ({pool} snapshots pooled)", fontsize=13)

    for r, snap_i in enumerate(rows):
        for c, pt in enumerate(cols):
            ax = axes[r][c]
            state = snaps[snap_i, pt].astype(bool)
            for i, j in bonds:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color="0.85", lw=0.7, zorder=1)
            ax.scatter(pos[~state, 0], pos[~state, 1], s=12,
                       facecolors="white", edgecolors="0.6", lw=0.6, zorder=2)
            ax.scatter(pos[state, 0], pos[state, 1], s=26,
                       facecolors="#C0392B", edgecolors="none", zorder=3)
            n_exc = int(state.sum())
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(pos[:, 0].min() - pad[0], pos[:, 0].max() + pad[0])
            ax.set_ylim(pos[:, 1].min() - pad[1], pos[:, 1].max() + pad[1])
            if r == 0:
                tag = " ↓" if pt_back[pt] else ""
                ax.set_title(f"δ≈{pt_delta[pt]:.2f}{tag}", fontsize=10)
            ax.text(0.02, 0.02, f"n={n_exc}", transform=ax.transAxes,
                    fontsize=7, color="0.4")
        axes[r][0].set_ylabel(f"snapshot #{snap_i}", fontsize=8)

    fig.tight_layout()
    out = args.out or default_fig_path(run_dir, "snapshots")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
