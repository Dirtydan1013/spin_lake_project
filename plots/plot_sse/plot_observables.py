"""SSE fixed-point observable summary: Z_l/A_v, C_m, order parameters.

Reads the chunked run dir of ``src.mpi.sse_mpi`` (one bin per chunk).  Three
panels: loop operators vs size (+ A_v), string operators vs size, and the
scalar order parameters (⟨n⟩, ⟨n⟩_bulk, Ψ_VBS, Ψ_SS, ⟨|mz|⟩).  The suptitle
carries the energy estimate.  Errors are SEMs over bins.

Usage::

    python plots/plot_sse/plot_observables.py --run_dir data/sse_6x6_... \
        [--burn_frac 0.5] [--out plots/foo.png]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "plot_diagonal"))
from profile_data import (bin_mean_sem, default_fig_path, load_run_attrs,
                          loop_string_sizes, stack_chunk_bins)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--burn_frac", type=float, default=0.5,
                        help="per-rank burn-in fraction of chunks (default 0.5, "
                             "matching combine_run)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    attrs = load_run_attrs(args.run_dir)
    keys = ["energy", "density", "density_bulk",
            "Z_l", "C_m_l", "M_vbs2", "M_ss2"]
    data = stack_chunk_bins(args.run_dir, keys, burn_in_fraction=args.burn_frac)
    if data["Z_l"] is None:
        raise KeyError("run has no Z_l datasets — re-run with the new SSE "
                       "observable support")
    n_bins = data["Z_l"].shape[0]

    e_mean, e_sem = bin_mean_sem(data["energy"].reshape(-1))
    z_mean, z_sem = bin_mean_sem(data["Z_l"])
    c_mean, c_sem = bin_mean_sem(data["C_m_l"])

    loop_sizes, string_sizes = loop_string_sizes(attrs)
    if loop_sizes is None or len(loop_sizes) != z_mean.size - 1:
        loop_sizes = list(range(z_mean.size - 1))
    if string_sizes is None or len(string_sizes) != c_mean.size:
        string_sizes = list(range(c_mean.size))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    nx, ny = int(attrs["nx"]), int(attrs["ny"])
    N = int(attrs["N"])
    fig.suptitle(
        f"SSE {attrs.get('lattice', '?')} {nx}×{ny} — δ={attrs['delta']:g}, "
        f"β={attrs['beta']:g}:  E = {e_mean:.4f} ± {e_sem:.4f}  "
        f"(E/N = {e_mean / N:.5f} ± {e_sem / N:.5f}, {n_bins} bins)",
        fontsize=12)

    ax = axes[0]
    ax.errorbar(loop_sizes, z_mean[:-1], yerr=z_sem[:-1], fmt="o-",
                color="#2B4EAE", lw=1.5, capsize=3, label="⟨Z_l⟩")
    ax.errorbar([max(loop_sizes) + 1], [z_mean[-1]], yerr=[z_sem[-1]],
                fmt="s", color="#C0392B", capsize=3, ms=8, label="⟨A_v⟩")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("loop size s (A_v offset right)")
    ax.set_ylabel("signed expectation")
    ax.set_title("loops + vertex", fontsize=10)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.errorbar(string_sizes, c_mean, yerr=c_sem, fmt="o-",
                color="#D35400", lw=1.5, capsize=3)
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("string size s")
    ax.set_ylabel("⟨C_m⟩ (signed)")
    ax.set_title("strings", fontsize=10)

    ax = axes[2]
    labels, vals, errs = [], [], []
    for key, label in [("density", "⟨n⟩"), ("density_bulk", "⟨n⟩ bulk")]:
        if data[key] is not None:
            m, s = bin_mean_sem(data[key].reshape(-1))
            labels.append(label); vals.append(m); errs.append(s)
    for key, label in [("M_vbs2", "Ψ_VBS"), ("M_ss2", "Ψ_SS")]:
        if data[key] is not None:
            m2, s2 = bin_mean_sem(data[key].reshape(-1))
            psi = np.sqrt(max(m2, 0.0))
            labels.append(label)
            vals.append(psi)
            errs.append(s2 / (2.0 * max(psi, 1e-12)))
    x = np.arange(len(labels))
    ax.errorbar(x, vals, yerr=errs, fmt="o", color="#27AE60", capsize=3, ms=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("order parameters", fontsize=10)

    for ax in axes:
        ax.grid(True, alpha=0.3, ls="--")
    fig.tight_layout()

    out = args.out or default_fig_path(args.run_dir, "observables")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
