"""2D Brillouin-zone heatmaps of the sublattice-resolved occupation SF matrix.

For each requested δ point the 6×6 matrix S_αβ(q) is reduced to a scalar per q
(default: max eigenvalue of the Hermitian matrix) and drawn on the reciprocal
unit-cell grid.  Rows = unit-cell conventions (hexagon-void 'full' and
triangle-pair 'tri' by default), columns = δ points.  Supports the connected
(⟨s s*⟩ − ⟨s⟩⟨s⟩*, default) and unconnected (raw ⟨s s*⟩) definitions; both are
normalised per cell.

Usage::

    python plots/plot_diagonal/plot_occ_sf_bz.py [--run_dir data/M=...] \
        [--cells full tri] [--mode connected|unconnected] [--stat max_eig|trace] \
        [--deltas 3.5 4.0 4.5] [--burn_frac 0.1] [--out plots/foo.png]
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
                          load_run_attrs, occ_sf_matrices,
                          resolve_point_indices)

CELL_LABEL = {"full": "hexagon cell (full)", "bulk": "hexagon cell (bulk)",
              "tri": "triangle-pair cell"}


def _reduce(s_mat, stat):
    """(pt, q) scalar field from the (pt, q, 6, 6) Hermitian matrices."""
    herm = 0.5 * (s_mat + np.conj(np.swapaxes(s_mat, -1, -2)))
    if stat == "max_eig":
        return np.linalg.eigvalsh(herm)[..., -1]
    if stat == "trace":
        return np.trace(herm, axis1=-2, axis2=-1).real
    raise ValueError(f"unknown stat {stat!r}")


def _bz_mesh(q_points):
    """Corner meshes (X, Y) of the fractional q-grid parallelogram (units of 2π)."""
    n_q = q_points.shape[0]
    grid_n = int(round(np.sqrt(n_q)))
    if grid_n * grid_n != n_q:
        raise ValueError(f"q grid is not square: n_q={n_q}")
    # _build_occ_q_grid: index qi = m*grid_n + n, q = (m/g) b1 + (n/g) b2
    b1 = np.array([1.0, -1.0 / np.sqrt(3)])
    b2 = np.array([0.0, 2.0 / np.sqrt(3)])
    f = np.arange(grid_n + 1) / grid_n
    fm, fn = np.meshgrid(f, f, indexing="ij")
    X = fm * b1[0] + fn * b2[0]
    Y = fm * b1[1] + fn * b2[1]
    return X, Y, grid_n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", default=None,
                        help="chunked run dir (default: newest data/M=*_*)")
    parser.add_argument("--cells", nargs="+", default=["full", "tri"],
                        choices=["full", "bulk", "tri"],
                        help="unit-cell conventions to plot (one row each)")
    parser.add_argument("--mode", choices=["connected", "unconnected"],
                        default="connected")
    parser.add_argument("--stat", choices=["max_eig", "trace"], default="max_eig",
                        help="scalar per q from the 6×6 matrix")
    parser.add_argument("--deltas", type=float, nargs="+", default=None,
                        help="subset of the measured occ-SF δ points "
                             "(default: all; matched to the nearest stored point)")
    parser.add_argument("--request_deltas", type=float, nargs="+", default=None,
                        help="the --occ_sf_delta_points the RUN used, for old "
                             "runs whose meta.h5 lacks occ_pt_indices "
                             "(default: the run script's standard 11-value list)")
    parser.add_argument("--burn_frac", type=float, default=0.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir()
    meta = load_meta(run_dir)
    attrs = load_run_attrs(run_dir)

    pt_idx = resolve_point_indices(meta, "occ", args.request_deltas)
    pt_delta = meta["prof_delta"][pt_idx]              # δ of each stored point
    # requested δ may have snapped to the backward (δ-decreasing) ramp
    turn = int(np.argmax(meta["prof_delta"]))
    pt_back = pt_idx > turn
    if args.deltas is not None:
        cols = list({int(np.argmin(np.abs(pt_delta - d))) for d in args.deltas})
    else:
        cols = list(range(len(pt_delta)))
    # columns left→right by δ value (storage order is by profile-point index,
    # which puts backward-ramp points after the largest forward δ)
    cols.sort(key=lambda pt: pt_delta[pt])

    X, Y, grid_n = _bz_mesh(np.asarray(meta["occ_q_points"]))
    connected = args.mode == "connected"

    fields = {}
    for cell in args.cells:
        s_mat = occ_sf_matrices(run_dir, attrs, meta, cell=cell,
                                connected=connected,
                                burn_in_fraction=args.burn_frac)
        if s_mat.shape[0] != len(pt_delta):
            raise RuntimeError(
                f"stored occ points ({s_mat.shape[0]}) != resolved δ points "
                f"({len(pt_delta)}); pass --request_deltas with the run's "
                f"--occ_sf_delta_points values")
        fields[cell] = _reduce(s_mat, args.stat)       # (pt, q)

    n_rows, n_cols = len(args.cells), len(cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.6 * n_cols + 1.2, 2.8 * n_rows + 0.8),
                             squeeze=False)
    nx, ny = int(attrs["nx"]), int(attrs["ny"])
    fig.suptitle(f"{attrs.get('lattice', '?')} {nx}×{ny} — occupation SF "
                 f"S_αβ(q): {args.stat}, {args.mode}", fontsize=13)

    for r, cell in enumerate(args.cells):
        vals = fields[cell]
        vmax = max(vals[cols].max(), 1e-12)
        vmin = min(vals[cols].min(), 0.0)
        for c, pt in enumerate(cols):
            ax = axes[r][c]
            C = vals[pt].reshape(grid_n, grid_n)
            im = ax.pcolormesh(X, Y, C, cmap="inferno", vmin=vmin, vmax=vmax,
                               shading="flat")
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                tag = " ↓" if pt_back[pt] else ""
                ax.set_title(f"δ≈{pt_delta[pt]:.2f}{tag}", fontsize=10)
            if c == 0:
                ax.set_ylabel(CELL_LABEL.get(cell, cell), fontsize=9)
        fig.colorbar(im, ax=[axes[r][c] for c in range(n_cols)],
                     shrink=0.85, pad=0.01)

    out = args.out or default_fig_path(run_dir, f"occ_sf_{args.stat}_{args.mode}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
