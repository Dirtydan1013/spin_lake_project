"""Four-panel diagonal profile: Rydberg density, A_v, Z_loop, C_m string vs δ.

Reads the chunked run format of ``src.mpi.qaqmc_mpi --mode profile``.

Usage::

    python plots/plot_diagonal/plot_profile_panels.py                 # latest data/M=* run
    python plots/plot_diagonal/plot_profile_panels.py --run_dir data/M=2760000_6x6_...
        [--sweep forward|backward|both] [--burn_frac 0.1] [--out plots/foo.png]
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

COLORS = ["#C0392B", "#D35400", "#27AE60", "#8E44AD", "#2B4EAE", "#16A085"]


def _plot_band(ax, x, mean, sem, color, label, ls="-"):
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.18, lw=0)
    ax.plot(x, mean, color=color, lw=1.6, ls=ls, label=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", default=None,
                        help="chunked run dir (default: newest data/M=*_*)")
    parser.add_argument("--sweep", choices=["forward", "backward", "both"],
                        default="forward",
                        help="which half of the δ up-down ramp to show")
    parser.add_argument("--burn_frac", type=float, default=0.0,
                        help="skip this leading fraction of each rank's chunks")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir()
    meta = load_meta(run_dir)
    attrs = load_run_attrs(run_dir)
    prof_delta = meta["prof_delta"]
    fwd, bwd = sweep_split(prof_delta)

    data = stack_chunks(run_dir, ["density", "Z_l", "C_m_l"],
                        burn_in_fraction=args.burn_frac)
    density = data["density"]                      # (bins, pts)
    z_all = data["Z_l"]                            # (bins, pts, n_lg + 1)
    c_all = data["C_m_l"]                          # (bins, pts, n_sg)
    n_bins = density.shape[0]
    z_loops, a_v = z_all[:, :, :-1], z_all[:, :, -1]

    loop_sizes, string_sizes = loop_string_sizes(attrs)
    if loop_sizes is None or len(loop_sizes) != z_loops.shape[2]:
        loop_sizes = [f"g{k}" for k in range(z_loops.shape[2])]
    if string_sizes is None or len(string_sizes) != c_all.shape[2]:
        string_sizes = [f"g{k}" for k in range(c_all.shape[2])]

    d_mean, d_sem = bin_mean_sem(density)
    v_mean, v_sem = bin_mean_sem(a_v)
    z_mean, z_sem = bin_mean_sem(z_loops)
    c_mean, c_sem = bin_mean_sem(c_all)

    sweeps = {"forward": [(fwd, "-")], "backward": [(bwd, "--")],
              "both": [(fwd, "-"), (bwd, "--")]}[args.sweep]

    nx, ny = int(attrs["nx"]), int(attrs["ny"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{attrs.get('lattice', '?')} {nx}×{ny} — diagonal profile "
                 f"({n_bins} bins × {attrs.get('samples_per_bin', '?')} samples, "
                 f"sweep={args.sweep})", fontsize=13)

    ax = axes[0, 0]
    for sl, ls in sweeps:
        x = prof_delta[sl]
        order = slice(None) if ls == "-" else slice(None, None, -1)
        _plot_band(ax, x[order], d_mean[sl][order], d_sem[sl][order],
                   COLORS[4], "⟨n⟩" + (" (back)" if ls == "--" else ""), ls)
    ax.set_ylabel("Rydberg density ⟨n⟩")

    ax = axes[0, 1]
    for sl, ls in sweeps:
        x = prof_delta[sl]
        order = slice(None) if ls == "-" else slice(None, None, -1)
        _plot_band(ax, x[order], v_mean[sl][order], v_sem[sl][order],
                   COLORS[0], "⟨A_v⟩" + (" (back)" if ls == "--" else ""), ls)
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.set_ylabel("vertex ⟨A_v⟩")

    ax = axes[1, 0]
    for g, (size, col) in enumerate(zip(loop_sizes, COLORS)):
        for sl, ls in sweeps:
            x = prof_delta[sl]
            order = slice(None) if ls == "-" else slice(None, None, -1)
            _plot_band(ax, x[order], z_mean[sl, g][order], z_sem[sl, g][order],
                       col, f"s={size}" if ls == "-" else None, ls)
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.set_ylabel("loop ⟨Z_l⟩ (signed)")
    ax.set_xlabel("δ / Ω")
    ax.legend(fontsize=8, title="loop size", ncol=2)

    ax = axes[1, 1]
    for g, (size, col) in enumerate(zip(string_sizes, COLORS)):
        for sl, ls in sweeps:
            x = prof_delta[sl]
            order = slice(None) if ls == "-" else slice(None, None, -1)
            _plot_band(ax, x[order], c_mean[sl, g][order], c_sem[sl, g][order],
                       col, f"s={size}" if ls == "-" else None, ls)
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.set_ylabel("string ⟨C_m⟩ (signed)")
    ax.set_xlabel("δ / Ω")
    ax.legend(fontsize=8, title="string size", ncol=2)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3, ls="--")
    fig.tight_layout()

    out = args.out or default_fig_path(run_dir, "profile_panels")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
