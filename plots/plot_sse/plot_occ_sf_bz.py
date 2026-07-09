"""SSE 2D Brillouin-zone heatmap of the occupation SF matrix S_αβ(q).

Single (δ, β) point; one panel per unit-cell convention (hexagon 'full',
'bulk', triangle-pair 'tri').  Same reduction/normalisation conventions as
plots/plot_diagonal/plot_occ_sf_bz.py: default max eigenvalue of the Hermitian
6×6 matrix, connected (⟨s s*⟩ − ⟨s⟩⟨s⟩*) or unconnected, per-cell normalised.

Usage::

    python plots/plot_sse/plot_occ_sf_bz.py --run_dir data/sse_6x6_... \
        [--cells full tri] [--mode connected|unconnected] [--stat max_eig|trace] \
        [--burn_frac 0.5] [--out plots/foo.png]
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
from profile_data import (default_fig_path, load_run_attrs, occ_geometry,
                          stack_chunk_bins)

CELL_DS = {"full": ("occ_S_full_re", "occ_S_full_im"),
           "bulk": ("occ_S_bulk_re", "occ_S_bulk_im"),
           "tri": ("occ2_S_re", "occ2_S_im")}
CELL_LABEL = {"full": "hexagon cell (full)", "bulk": "hexagon cell (bulk)",
              "tri": "triangle-pair cell"}


def load_meta(run_dir):
    meta = {}
    with h5py.File(os.path.join(run_dir, "meta.h5"), "r") as f:
        for key in f.keys():
            meta[key] = f[key][:]
        meta["attrs"] = dict(f.attrs)
    return meta


def sf_matrix(run_dir, attrs, meta, cell, connected, burn_frac):
    """Bin-averaged S[q, a, b] (complex, per-cell normalised) for one cell."""
    ds = CELL_DS[cell]
    data = stack_chunk_bins(run_dir, [ds[0], ds[1], "occ_nprof"],
                            burn_in_fraction=burn_frac)
    if data[ds[0]] is None:
        raise KeyError(f"run has no {ds[0]} (run with --occ-sf-grid-n > 0)")
    s_mat = (data[ds[0]].astype(np.float64).mean(axis=0)
             + 1j * data[ds[1]].astype(np.float64).mean(axis=0))   # (q, 6, 6)

    cell_R, basis, mask, n_cells = occ_geometry(attrs, meta, cell)
    if connected:
        nbar = data["occ_nprof"].astype(np.float64).mean(axis=0)   # (N,)
        q_points = np.asarray(meta["occ_q_points"], dtype=np.float64)
        phase = np.exp(1j * (cell_R[mask] @ q_points.T))           # (n_sel, q)
        b_sel = basis[mask]
        w = nbar[mask]
        m = np.zeros((q_points.shape[0], 6), dtype=np.complex128)
        for a in range(6):
            sel = b_sel == a
            m[:, a] = (w[sel][:, None] * phase[sel, :]).sum(axis=0)
        s_mat -= np.einsum("qa,qb->qab", m, m.conj())
    return s_mat / max(n_cells, 1)


def reduce_stat(s_mat, stat):
    herm = 0.5 * (s_mat + np.conj(np.swapaxes(s_mat, -1, -2)))
    if stat == "max_eig":
        return np.linalg.eigvalsh(herm)[..., -1]
    if stat == "trace":
        return np.trace(herm, axis1=-2, axis2=-1).real
    raise ValueError(f"unknown stat {stat!r}")


def bz_mesh(n_q):
    grid_n = int(round(np.sqrt(n_q)))
    if grid_n * grid_n != n_q:
        raise ValueError(f"q grid is not square: n_q={n_q}")
    b1 = np.array([1.0, -1.0 / np.sqrt(3)])
    b2 = np.array([0.0, 2.0 / np.sqrt(3)])
    f = np.arange(grid_n + 1) / grid_n
    fm, fn = np.meshgrid(f, f, indexing="ij")
    return fm * b1[0] + fn * b2[0], fm * b1[1] + fn * b2[1], grid_n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--cells", nargs="+", default=["full", "tri"],
                        choices=["full", "bulk", "tri"])
    parser.add_argument("--mode", choices=["connected", "unconnected"],
                        default="connected")
    parser.add_argument("--stat", choices=["max_eig", "trace"], default="max_eig")
    parser.add_argument("--burn_frac", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    attrs = load_run_attrs(args.run_dir)
    meta = load_meta(args.run_dir)
    connected = args.mode == "connected"

    fig, axes = plt.subplots(1, len(args.cells),
                             figsize=(4.2 * len(args.cells) + 1.0, 4.2),
                             squeeze=False)
    nx, ny = int(attrs["nx"]), int(attrs["ny"])
    fig.suptitle(f"SSE {attrs.get('lattice', '?')} {nx}×{ny} — "
                 f"δ={attrs['delta']:g}, β={attrs['beta']:g}: occ-SF "
                 f"{args.stat}, {args.mode}", fontsize=12)

    for ax, cell in zip(axes[0], args.cells):
        s_mat = sf_matrix(args.run_dir, attrs, meta, cell, connected,
                          args.burn_frac)
        vals = reduce_stat(s_mat, args.stat)
        X, Y, grid_n = bz_mesh(vals.size)
        im = ax.pcolormesh(X, Y, vals.reshape(grid_n, grid_n),
                           cmap="inferno", shading="flat")
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(CELL_LABEL.get(cell, cell), fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    fig.tight_layout()
    out = args.out or default_fig_path(args.run_dir,
                                       f"occ_sf_{args.stat}_{args.mode}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
