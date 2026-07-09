"""Standalone BFFM (Fredenhagen–Marcu) order parameter: C_m(l−1) / √|Z(l)|.

Signed ratio (no |·| on C_m), one curve per (loop size l, string size l−1)
pair, from the chunked run format of ``src.mpi.qaqmc_mpi --mode profile``.

Usage::

    python plots/plot_diagonal/plot_bffm.py [--run_dir data/M=...] \
        [--delta_min 2.0 --delta_max 6.0] [--sweep forward|backward] \
        [--burn_frac 0.1] [--out plots/foo.png]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_data import (bin_mean_sem, default_fig_path, latest_run_dir,
                          load_meta, load_run_attrs, loop_string_sizes,
                          stack_chunks, sweep_split)

COLORS = ["#C0392B", "#D35400", "#27AE60", "#8E44AD", "#2B4EAE"]
EPS = 1e-4


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", default=None,
                        help="chunked run dir (default: newest data/M=*_*)")
    parser.add_argument("--delta_min", type=float, default=2.0)
    parser.add_argument("--delta_max", type=float, default=6.0)
    parser.add_argument("--sweep", choices=["forward", "backward"],
                        default="forward")
    parser.add_argument("--burn_frac", type=float, default=0.0)
    parser.add_argument("--ylim", type=float, default=2.0,
                        help="clip y-axis to ±this (BFFM lives in [-1, 1]; "
                             "error bands blow up where Z→0). <=0 disables")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir()
    meta = load_meta(run_dir)
    attrs = load_run_attrs(run_dir)
    prof_delta = meta["prof_delta"]
    fwd, bwd = sweep_split(prof_delta)
    sl = fwd if args.sweep == "forward" else bwd

    data = stack_chunks(run_dir, ["Z_l", "C_m_l"], burn_in_fraction=args.burn_frac)
    z_loops = data["Z_l"][:, :, :-1]          # drop trailing A_v column
    c_all = data["C_m_l"]
    n_bins = z_loops.shape[0]

    loop_sizes, string_sizes = loop_string_sizes(attrs)
    if loop_sizes is None or len(loop_sizes) != z_loops.shape[2] \
            or string_sizes is None or len(string_sizes) != c_all.shape[2]:
        raise RuntimeError("cannot reconstruct loop/string size labels; "
                           "BFFM needs the (l, l-1) size pairing")

    z_mean, z_sem = bin_mean_sem(z_loops)
    c_mean, c_sem = bin_mean_sem(c_all)

    delta = prof_delta[sl]
    mask = (delta >= args.delta_min) & (delta <= args.delta_max)
    d = delta[mask]

    pairs = [(ls_, ls_ - 1) for ls_ in loop_sizes if (ls_ - 1) in string_sizes]
    nx, ny = int(attrs["nx"]), int(attrs["ny"])

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"{attrs.get('lattice', '?')} {nx}×{ny} — BFFM: C_m / √|Z| "
                 f"(δ = {args.delta_min}–{args.delta_max}, sweep={args.sweep}, "
                 f"{n_bins} bins)", fontsize=13)

    for (lsz, ssz), col in zip(pairs, COLORS):
        gi = loop_sizes.index(lsz)
        gj = string_sizes.index(ssz)
        zm = z_mean[sl, gi][mask]
        cm = c_mean[sl, gj][mask]
        zs = z_sem[sl, gi][mask]
        cs = c_sem[sl, gj][mask]

        safe_z = np.where(np.abs(zm) > EPS, np.abs(zm), np.nan)
        ratio = cm / np.sqrt(safe_z)
        ratio_sem = np.abs(ratio) * np.sqrt(
            (cs / np.where(np.abs(cm) > EPS, np.abs(cm), np.nan)) ** 2
            + (0.5 * zs / safe_z) ** 2)

        ax.fill_between(d, ratio - ratio_sem, ratio + ratio_sem,
                        color=col, alpha=0.2, lw=0)
        ax.plot(d, ratio, color=col, lw=1.8, label=f"s_loop={lsz}, s_str={ssz}")

    ax.axhline(1.0, color="k", lw=1.2, ls="--", alpha=0.6, label="BFFM = 1")
    ax.axhline(-1.0, color="k", lw=1.2, ls="--", alpha=0.6)
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    if args.ylim > 0:
        ax.set_ylim(-args.ylim, args.ylim)
    ax.set_xlabel("δ / Ω")
    ax.set_ylabel("C_m(l−1) / √|Z(l)|")
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = args.out or default_fig_path(run_dir, "bffm")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
