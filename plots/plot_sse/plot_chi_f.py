"""SSE fidelity susceptibility chi_F(delta) across a set of run dirs.

Reads runs produced by ``src.mpi.sse_mpi --chi-f`` (per-chunk bin means
chi_gl / chi_gr / chi_glgr) and assembles the Wang-Liu-Troyer estimator

    chi_F = ( <G_L G_R> - <G_L><G_R> ) / 2

with GLOBAL means over all kept bins and a leave-one-bin-out jackknife for
the error.  One point per run dir; plots chi_F/N vs delta (the peak position
locates the trivial -> RCSL transition, cf. Wang-Pollet PRL 134, 086601).

Usage::

    python plots/plot_sse/plot_chi_f.py --run_dirs data/sse_6x6_*_chif_delta* \
        [--burn_frac 0.5] [--out plots/foo.png]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "plot_diagonal"))
from profile_data import load_run_attrs, stack_chunk_bins


def chi_f_jackknife(run_dir, burn_frac):
    """(chi_F, err, n_bins) from the chi_gl/chi_gr/chi_glgr bin means."""
    data = stack_chunk_bins(run_dir, ["chi_gl", "chi_gr", "chi_glgr"],
                            burn_in_fraction=burn_frac)
    if data["chi_gl"] is None:
        raise KeyError(f"{run_dir} has no chi_gl — run sse_mpi with --chi-f")
    gl = data["chi_gl"].reshape(-1)
    gr = data["chi_gr"].reshape(-1)
    glgr = data["chi_glgr"].reshape(-1)
    n = gl.size

    def est(mask):
        return 0.5 * (glgr[mask].mean() - gl[mask].mean() * gr[mask].mean())

    full = est(np.ones(n, dtype=bool))
    jk = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jk[i] = est(mask)
    err = np.sqrt((n - 1) * np.mean((jk - jk.mean()) ** 2))
    return full, err, n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--burn_frac", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = []
    for rd in args.run_dirs:
        attrs = load_run_attrs(rd)
        chi, err, n = chi_f_jackknife(rd, args.burn_frac)
        rows.append((float(attrs["delta"]), float(attrs["beta"]),
                     int(attrs["N"]), chi, err, n))
        print(f"δ={attrs['delta']:g} β={attrs['beta']:g}: "
              f"chi_F = {chi:.5f} ± {err:.5f}  "
              f"(chi_F/N = {chi / attrs['N']:.6f}, {n} bins)")
    rows.sort()

    deltas = np.array([r[0] for r in rows])
    chi_n = np.array([r[3] / r[2] for r in rows])
    err_n = np.array([r[4] / r[2] for r in rows])
    beta_set = sorted({r[1] for r in rows})
    n_site = rows[0][2]

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.errorbar(deltas, chi_n, yerr=err_n, fmt="o-", color="#2B4EAE",
                lw=1.5, capsize=3)
    ipk = int(np.argmax(chi_n))
    ax.axvline(deltas[ipk], color="#C0392B", lw=0.9, ls=":",
               label=f"peak δ≈{deltas[ipk]:g}")
    ax.set_xlabel("δ/Ω")
    ax.set_ylabel("χ_F / N")
    ax.set_title(f"SSE fidelity susceptibility — N={n_site}, "
                 f"β={','.join(f'{b:g}' for b in beta_set)}", fontsize=11)
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = args.out or "figures/sse_chi_f/chi_f_vs_delta.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
