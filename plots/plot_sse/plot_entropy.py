"""Thermal entropy S(T)/N from an SSE beta ladder (Wang-Pollet Eq. 7).

Reads a ``src.mpi.sse_entropy_mpi`` run dir (rung subdirs beta_{k:03d}/ +
top-level meta.h5 with betas / E_inf / varH_inf) and computes

    S(beta) = N ln2 - b0 E_inf + b0^2 varH/2 - int_{b0}^{beta} E db' + beta E(beta)

with a trapezoid over the measured rungs and the analytic high-T anchor
applied at the first rung b0 (valid when b0^2 varH / 2 is small; the script
prints the anchor residual E(b0) - [E_inf - b0 varH] as a check).  S is
linear in the measured E_k, so the error is exact linear propagation of the
per-rung SEMs.  Plots S/N vs T with the ln2/6 CSL plateau and the finite-size
line ln2 (1/6 + 1/N), cf. Wang-Pollet PRL 134, 086601 Fig. 5.

Usage::

    python plots/plot_sse/plot_entropy.py --run_dir data/sse_entropy_... \
        [--burn_frac 0.25] [--out plots/foo.png]
"""

import argparse
import glob
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "plot_diagonal"))
from profile_data import bin_mean_sem, stack_chunk_bins


def load_ladder(run_dir, burn_frac):
    with h5py.File(os.path.join(run_dir, "meta.h5"), "r") as f:
        meta = dict(f.attrs)
        betas = f["betas"][:]
    rungs = sorted(glob.glob(os.path.join(run_dir, "beta_[0-9]*")))
    if len(rungs) < len(betas):
        print(f"[warn] {len(rungs)}/{len(betas)} rungs present — "
              "ladder incomplete, integrating what exists")
        betas = betas[:len(rungs)]
    e_mean = np.empty(len(rungs))
    e_sem = np.empty(len(rungs))
    for k, rd in enumerate(rungs[:len(betas)]):
        data = stack_chunk_bins(rd, ["energy"], burn_in_fraction=burn_frac)
        e_mean[k], e_sem[k] = bin_mean_sem(data["energy"].reshape(-1))
    return meta, np.asarray(betas, float), e_mean, e_sem


def entropy_curve(meta, betas, e_mean, e_sem):
    """S(beta_m), err for every rung m >= 0 (anchor at rung 0)."""
    N = int(meta["N"])
    e_inf, var_h = float(meta["E_inf"]), float(meta["varH_inf"])
    b0 = betas[0]
    anchor = N * np.log(2.0) - b0 * e_inf + 0.5 * b0**2 * var_h
    resid = e_mean[0] - (e_inf - b0 * var_h)
    print(f"anchor check @β0={b0:g}: E measured {e_mean[0]:.4f} vs analytic "
          f"{e_inf - b0 * var_h:.4f} (resid {resid:+.4f}, "
          f"{resid / max(e_sem[0], 1e-300):+.1f}σ)")

    n = len(betas)
    S = np.empty(n)
    S_err = np.empty(n)
    for m in range(n):
        w = np.zeros(n)                      # trapezoid weights for ∫_{b0}^{bm}
        for k in range(m):
            h = betas[k + 1] - betas[k]
            w[k] += 0.5 * h
            w[k + 1] += 0.5 * h
        coeff = -w
        coeff[m] += betas[m]
        S[m] = anchor + float(coeff @ e_mean)
        S_err[m] = float(np.sqrt(((coeff * e_sem) ** 2).sum()))
    return S, S_err


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--burn_frac", type=float, default=0.25)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    meta, betas, e_mean, e_sem = load_ladder(args.run_dir, args.burn_frac)
    N = int(meta["N"])
    S, S_err = entropy_curve(meta, betas, e_mean, e_sem)

    print(f"\n{'beta':>10} {'T':>10} {'E/N':>12} {'S/N':>10} {'err':>8}")
    for b, e, s, se in zip(betas, e_mean, S, S_err):
        print(f"{b:>10.4g} {1.0 / b:>10.4g} {e / N:>12.5f} "
              f"{s / N:>10.5f} {se / N:>8.5f}")

    T = 1.0 / betas
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.errorbar(T, S / N, yerr=S_err / N, fmt="o-", ms=3.5, lw=1.2,
                color="#2B4EAE", capsize=2)
    ax.axhline(np.log(2) / 6, color="#C0392B", lw=1.0, ls="--",
               label="ln2/6 (CSL plateau)")
    ax.axhline(np.log(2) * (1.0 / 6.0 + 1.0 / N), color="#C0392B", lw=0.8,
               ls=":", label="ln2(1/6 + 1/N)")
    ax.axhline(np.log(2), color="k", lw=0.7, ls=":", alpha=0.5,
               label="ln2 (free spins)")
    ax.set_xscale("log")
    ax.set_xlabel("T / Ω")
    ax.set_ylabel("S / N")
    ax.set_ylim(bottom=0)
    ax.set_title(f"SSE entropy — {meta.get('lattice', '?')} "
                 f"{meta.get('nx', '?')}×{meta.get('ny', '?')} "
                 f"δ={meta.get('delta', float('nan')):g} "
                 f"({meta.get('boundary', '?')})", fontsize=11)
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = args.out or os.path.join(
        "figures", os.path.basename(os.path.normpath(args.run_dir)),
        "entropy.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
