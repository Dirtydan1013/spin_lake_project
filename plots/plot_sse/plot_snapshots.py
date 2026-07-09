"""Real-space excitation patterns from SSE full-state snapshots.

Single (δ, β) point: draws a grid of snapshots pooled from all ranks/chunks
(requires a run with ``--n-snapshots > 0``).  Same drawing conventions as
plots/plot_diagonal/plot_snapshots.py.

Usage::

    python plots/plot_sse/plot_snapshots.py --run_dir data/sse_6x6_... \
        [--n_samples 8] [--seed 0] [--out plots/foo.png]
"""

import argparse
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "plot_diagonal"))
from plot_snapshots import _nn_bonds          # same NN guide-bond convention
from profile_data import default_fig_path, load_run_attrs, stack_chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=8,
                        help="snapshots to draw (sampled evenly from the pool)")
    parser.add_argument("--n_cols", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0,
                        help="offset into the snapshot pool")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    attrs = load_run_attrs(args.run_dir)
    with h5py.File(os.path.join(args.run_dir, "meta.h5"), "r") as f:
        pos = f["pos"][:]

    snaps = stack_chunks(args.run_dir, ["snapshots"])["snapshots"]
    if snaps is None:
        raise KeyError("run has no snapshots (re-run with --n-snapshots > 0)")
    snaps = snaps.reshape(-1, snaps.shape[-1])       # pool × N
    pool = snaps.shape[0]

    n_show = min(args.n_samples, pool)
    picks = (np.round(np.linspace(0, pool - 1, n_show)).astype(int)
             + args.seed) % pool

    bonds = _nn_bonds(pos)
    pad = 0.05 * (pos.max(axis=0) - pos.min(axis=0))
    n_cols = min(args.n_cols, n_show)
    n_rows = -(-n_show // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.6 * n_cols + 0.5, 2.4 * n_rows + 0.7),
                             squeeze=False)
    nx, ny = int(attrs["nx"]), int(attrs["ny"])
    fig.suptitle(f"SSE {attrs.get('lattice', '?')} {nx}×{ny} — snapshots at "
                 f"δ={attrs['delta']:g}, β={attrs['beta']:g} "
                 f"({pool} pooled)", fontsize=12)

    for k, ax in enumerate(axes.flat):
        if k >= n_show:
            ax.axis("off")
            continue
        state = snaps[picks[k]].astype(bool)
        for i, j in bonds:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color="0.85", lw=0.7, zorder=1)
        ax.scatter(pos[~state, 0], pos[~state, 1], s=12,
                   facecolors="white", edgecolors="0.6", lw=0.6, zorder=2)
        ax.scatter(pos[state, 0], pos[state, 1], s=26,
                   facecolors="#C0392B", edgecolors="none", zorder=3)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(pos[:, 0].min() - pad[0], pos[:, 0].max() + pad[0])
        ax.set_ylim(pos[:, 1].min() - pad[1], pos[:, 1].max() + pad[1])
        ax.set_title(f"#{picks[k]}  (n={int(state.sum())})", fontsize=8)

    fig.tight_layout()
    out = args.out or default_fig_path(args.run_dir, "snapshots")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
