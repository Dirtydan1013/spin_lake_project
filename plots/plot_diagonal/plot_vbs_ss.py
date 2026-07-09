"""VBS / SS order parameters Ψ = √⟨M²⟩ vs δ (paper Eq. 5-6).

Reads the chunked run format of ``src.mpi.qaqmc_mpi --mode profile``
(M_vbs2 / M_ss2 bin means).  Random baseline = 1/√N_tri.

Usage::

    python plots/plot_diagonal/plot_vbs_ss.py [--run_dir data/M=...] \
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
from profile_data import (default_fig_path, latest_run_dir, load_meta,
                          load_run_attrs, stack_chunks, sweep_split)


def order_param(m2_bins):
    """Ψ = sqrt(mean over bins of ⟨M²⟩), SEM propagated through the sqrt."""
    mean = m2_bins.mean(axis=0)
    sem = m2_bins.std(axis=0, ddof=1) / np.sqrt(m2_bins.shape[0])
    psi = np.sqrt(np.maximum(mean, 0.0))
    return psi, sem / (2.0 * np.maximum(psi, 1e-12))


def n_vbs_triangles(attrs):
    """Number of up-triangles entering M_vbs/M_ss (for the random baseline)."""
    lattice = str(attrs.get("lattice", "kagome_bond"))
    nx, ny = int(attrs["nx"]), int(attrs["ny"])
    try:
        from src.mpi.qaqmc_mpi import (_build_vbs_triangles,
                                       _build_vbs_triangles_tri,
                                       _lattice_observables)
        if lattice == "kagome_bond_triangle":
            (_b, _l, _s, _lm, _sm, _v, ijk_map) = _lattice_observables(
                lattice, nx, ny, boundary=str(attrs.get("boundary", "open")))
            tri = _build_vbs_triangles_tri(nx, ny, ijk_map)
        else:
            tri = _build_vbs_triangles(nx, ny)
        return len(tri[1]) if tri is not None else None
    except Exception:
        return (nx - 1) * (ny - 1) if lattice == "kagome_bond" else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", default=None,
                        help="chunked run dir (default: newest data/M=*_*)")
    parser.add_argument("--sweep", choices=["forward", "backward", "both"],
                        default="both")
    parser.add_argument("--burn_frac", type=float, default=0.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir()
    meta = load_meta(run_dir)
    attrs = load_run_attrs(run_dir)
    prof_delta = meta["prof_delta"]
    fwd, bwd = sweep_split(prof_delta)

    data = stack_chunks(run_dir, ["M_vbs2", "M_ss2"],
                        burn_in_fraction=args.burn_frac)
    if data["M_vbs2"] is None:
        raise KeyError("run has no M_vbs2 datasets (VBS/SS not measured)")
    n_bins = data["M_vbs2"].shape[0]

    psi_vbs, err_vbs = order_param(data["M_vbs2"])
    psi_ss, err_ss = order_param(data["M_ss2"])

    n_tri = n_vbs_triangles(attrs)
    baseline = None if n_tri is None else 1.0 / np.sqrt(n_tri)

    sweeps = {"forward": [(fwd, "-", "up")], "backward": [(bwd, "--", "down")],
              "both": [(fwd, "-", "up"), (bwd, "--", "down")]}[args.sweep]

    nx, ny = int(attrs["nx"]), int(attrs["ny"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(f"{attrs.get('lattice', '?')} {nx}×{ny} — VBS/SS order "
                 f"parameters Ψ = √⟨M²⟩ ({n_bins} bins)", fontsize=13)

    for ax, (psi, err, name, col) in zip(
            axes, [(psi_vbs, err_vbs, "VBS (checkerboard)", "#8E44AD"),
                   (psi_ss, err_ss, "SS (stripe)", "#27AE60")]):
        for sl, ls, tag in sweeps:
            x = prof_delta[sl]
            ax.fill_between(x, psi[sl] - err[sl], psi[sl] + err[sl],
                            color=col, alpha=0.18, lw=0)
            ax.plot(x, psi[sl], color=col, lw=1.6, ls=ls, label=f"δ {tag}")
        if baseline is not None:
            ax.axhline(baseline, color="k", lw=1.0, ls=":",
                       label=f"random 1/√N_tri (N_tri={n_tri})")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("δ / Ω")
        ax.grid(True, alpha=0.3, ls="--")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Ψ")
    fig.tight_layout()

    out = args.out or default_fig_path(run_dir, "vbs_ss")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
