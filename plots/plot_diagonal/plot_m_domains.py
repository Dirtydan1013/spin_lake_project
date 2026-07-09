"""Per-bin M-point domain analysis: orientation populations and exclusivity.

With domain-averaged ensembles (site permutation ON, or equilibrium SSE) the
ensemble-mean S(q) understates stripe order — the order lives in the per-bin
λ_max(S(M)) values, chain by chain.  This script pools every super-bin from
every rank and shows, at one δ point:

  1. per-bin histograms of λ_max at M1/M2/M3 (equal curves = C3-fair),
  2. per-bin M1 vs M2 scatter, coloured by M3 (points hugging the axes =
     single-Q stripes, a diagonal cloud = double-Q order),
  3. per-rank means sorted by M1−M2 (which orientation each chain sits in),

plus printed per-rank/per-bin domain counts and corr(M1,M2).  All values are
per-cell normalised (λ ≳ 1 = ordered bin, tunable via --strong).

Works on both chunked formats: qaqmc profile runs (occ arrays with a δ-point
axis; pick the point with --delta) and SSE runs (single point).

Usage::

    python plots/plot_diagonal/plot_m_domains.py --run_dir data/qaqmc_profile_M=... --delta 5.5
    python plots/plot_diagonal/plot_m_domains.py --run_dir data/sse_6x6_...
"""

import argparse
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_data import (default_fig_path, load_run_attrs, occ_geometry,
                          resolve_point_indices)

CELL_DS = {"full": ("occ_S_full_re", "occ_S_full_im"),
           "bulk": ("occ_S_bulk_re", "occ_S_bulk_im"),
           "tri": ("occ2_S_re", "occ2_S_im")}


def _m_indices(n_q):
    g = int(round(np.sqrt(n_q)))
    if g * g != n_q or g % 2:
        raise ValueError(f"q grid must be square with even side, n_q={n_q}")
    h = g // 2
    return g, {"M1": h * g, "M2": h, "M3": h * g + h}


def load_bins(run_dir, cell, burn_frac, delta):
    """(lam (n_bins, 3), ranks (n_bins,), delta_label, n_cells) for M1/M2/M3."""
    from src.mpi.chunk_io import iter_rank_chunks

    attrs = load_run_attrs(run_dir)
    meta = {}
    with h5py.File(os.path.join(run_dir, "meta.h5"), "r") as f:
        for key in f.keys():
            meta[key] = f[key][:]
        meta["attrs"] = dict(f.attrs)

    is_profile = "delta_schedule" in meta
    if is_profile:
        meta["prof_delta"] = meta["delta_schedule"][meta["p_indices"]]
        pt_idx = resolve_point_indices(meta, "occ")
        pt_delta = meta["prof_delta"][pt_idx]
        req = float(delta) if delta is not None else pt_delta.max()
        pt = int(np.argmin(np.abs(pt_delta - req)))
        delta_label = f"δ≈{pt_delta[pt]:.2f}"
    else:
        pt = None
        delta_label = f"δ={attrs['delta']:g}, β={attrs['beta']:g}"

    ds_re, ds_im = CELL_DS[cell]
    lam, ranks = [], []
    qsel = None
    for r, _c, grp in iter_rank_chunks(run_dir, burn_in_fraction=burn_frac):
        if ds_re not in grp:
            raise KeyError(f"run has no {ds_re} (occ-SF not enabled)")
        re = grp[ds_re]
        if qsel is None:
            n_q = re.shape[-3]
            _g, m_map = _m_indices(n_q)
            qsel = sorted(m_map.values())            # ascending for h5py
            order = np.argsort(np.argsort([m_map[k] for k in ("M1", "M2", "M3")]))
        if pt is not None:                            # profile: (nb, n_pt, q, 6, 6)
            h = (re[:, pt, qsel].astype(np.float64)
                 + 1j * grp[ds_im][:, pt, qsel].astype(np.float64))
        else:                                         # SSE: (q, 6, 6), one bin/chunk
            h = (re[qsel].astype(np.float64)
                 + 1j * grp[ds_im][qsel].astype(np.float64))[None]
        h = 0.5 * (h + np.conj(np.swapaxes(h, -1, -2)))
        v = np.linalg.eigvalsh(h)[..., -1]            # (nb, 3) in qsel order
        lam.append(v[:, order])                       # reorder to M1, M2, M3
        ranks += [r] * v.shape[0]

    _cr, _b, _mask, n_cells = occ_geometry(attrs, meta, cell)
    lam = np.concatenate(lam) / max(n_cells, 1)
    return lam, np.asarray(ranks), delta_label, attrs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", required=True,
                        help="profile (qaqmc_profile_M=*) or SSE run dir")
    parser.add_argument("--delta", type=float, default=None,
                        help="(profile runs) which occ-SF δ point "
                             "(default: the largest measured δ)")
    parser.add_argument("--cell", choices=["full", "bulk", "tri"], default="full")
    parser.add_argument("--strong", type=float, default=1.0,
                        help="per-cell λ threshold counting a bin/rank as ordered")
    parser.add_argument("--burn_frac", type=float, default=0.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    lam, ranks, delta_label, attrs = load_bins(
        args.run_dir, args.cell, args.burn_frac, args.delta)
    n_bins = lam.shape[0]
    labels = np.array(["M1", "M2", "M3"])

    # ── printed summary ──────────────────────────────────────────────────────
    dom = labels[np.argmax(lam, axis=1)]
    strong = lam.max(axis=1) > args.strong
    r12 = np.corrcoef(lam[:, 0], lam[:, 1])[0, 1]
    print(f"{delta_label}: {n_bins} bins / {len(set(ranks))} ranks "
          f"(cell={args.cell}, per-cell λ, strong>{args.strong:g})")
    print("  per-bin domains: " + "  ".join(
        f"{m}: {int(((dom == m) & strong).sum())}" for m in labels)
        + f"  weak: {int((~strong).sum())}")
    rank_ids = sorted(set(ranks))
    rmeans = np.stack([lam[ranks == r].mean(axis=0) for r in rank_ids])
    rdom = labels[np.argmax(rmeans, axis=1)]
    rstrong = rmeans.max(axis=1) > args.strong
    print("  per-rank domains: " + "  ".join(
        f"{m}: {int(((rdom == m) & rstrong).sum())}" for m in labels)
        + f"  weak: {int((~rstrong).sum())}")
    print(f"  corr(M1,M2) over bins = {r12:+.3f} "
          f"(negative = mutually exclusive single-Q domains)")

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle(f"{attrs.get('lattice', '?')} {attrs['nx']}×{attrs['ny']} — "
                 f"M-point domains at {delta_label} "
                 f"({n_bins} bins / {len(rank_ids)} ranks, cell={args.cell})",
                 fontsize=12)

    ax = axes[0]
    bins = np.linspace(0, max(lam.max(), 1e-9) * 1.02, 40)
    for i, (lab, col) in enumerate(zip(labels, ("#2B4EAE", "#D35400", "#27AE60"))):
        ax.hist(lam[:, i], bins=bins, histtype="step", lw=1.8, label=lab, color=col)
    ax.axvline(args.strong, color="k", lw=0.9, ls=":", alpha=0.6)
    ax.set_xlabel("per-bin λ_max(S(M)) / cell")
    ax.set_ylabel("bins")
    ax.set_title("per-bin histograms", fontsize=10)
    ax.legend(fontsize=9)

    ax = axes[1]
    sc = ax.scatter(lam[:, 0], lam[:, 1], s=6, c=lam[:, 2], cmap="viridis",
                    alpha=0.6)
    lim = max(lam[:, :2].max(), 1e-9) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    fig.colorbar(sc, ax=ax, label="M3")
    ax.set_xlabel("M1 per bin"); ax.set_ylabel("M2 per bin")
    ax.set_title(f"M1 vs M2 (corr={r12:+.2f})", fontsize=10)

    ax = axes[2]
    srt = np.argsort(rmeans[:, 0] - rmeans[:, 1])
    x = np.arange(len(rank_ids))
    ax.bar(x, rmeans[srt, 0], width=0.9, color="#2B4EAE", alpha=0.85, label="M1")
    ax.bar(x, -rmeans[srt, 1], width=0.9, color="#D35400", alpha=0.85,
           label="M2 (down)")
    ax.plot(x, rmeans[srt, 2], "o", ms=3, color="#27AE60", label="M3")
    ax.plot(x, -rmeans[srt, 2], "o", ms=3, color="#27AE60")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("rank (sorted by M1−M2)")
    ax.set_ylabel("rank-mean λ / cell")
    ax.set_title("chain-by-chain orientation", fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    name = f"m_domains_{args.cell}" + ("" if args.delta is None
                                       else f"_d{args.delta:g}")
    out = args.out or default_fig_path(args.run_dir, name)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
