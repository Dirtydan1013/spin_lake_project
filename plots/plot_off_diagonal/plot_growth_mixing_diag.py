"""Analyze the growth-anchor mixing diagnostic (two-arm occupancy series).

Input: HDF5 from src/mpi/growth_mixing_diag_mpi.py.  For each recorded
stage: bin the per-rank occupancy series, split ranks by starting arm
(ON/OFF), and plot the two ensemble curves p(t | arm) with sems.  The arm
gap g(t) = p_ON(t) - p_OFF(t) is fit against exponential and power-law
decay; the merged (arm-averaged) curve shows where the pooled estimate
stops drifting.  Prints tau estimates and the window needed for the gap to
fall below the target sem.

Usage: python plots/plot_off_diagonal/plot_growth_mixing_diag.py <h5> [outdir]
"""

import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BIN = 1000


def _binned(series: np.ndarray, bin_size: int) -> np.ndarray:
    """(R, T) -> (R, T//bin) per-rank bin means."""
    R, T = series.shape
    nb = T // bin_size
    return series[:, :nb * bin_size].reshape(R, nb, bin_size).mean(axis=2)


def _arm_curve(binned: np.ndarray):
    """(R_arm, nb) -> mean, sem over ranks."""
    m = binned.mean(axis=0)
    s = binned.std(axis=0, ddof=1) / np.sqrt(binned.shape[0])
    return m, s


def _fit_exp(t, g, sg):
    """log|g| = log A - t/tau weighted fit; returns (A, tau) or None."""
    ok = g > 3 * sg
    if ok.sum() < 3:
        return None
    w = (g[ok] / sg[ok]) ** 2
    coef = np.polyfit(t[ok], np.log(g[ok]), 1, w=w)
    tau = -1.0 / coef[0] if coef[0] < 0 else np.inf
    return float(np.exp(coef[1])), float(tau)


def main():
    path = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "plots"
    with h5py.File(path, "r") as f:
        start_on = np.asarray(f.attrs["start_on"], dtype=bool)
        stages = sorted(int(k[5:]) for k in f
                        if k.startswith("stage") and f[k].attrs["recorded"])
        fig, axes = plt.subplots(2, len(stages), figsize=(7 * len(stages), 8),
                                 squeeze=False)
        for j, k in enumerate(stages):
            occ = np.asarray(f[f"stage{k}/occ"], dtype=np.float64)
            lam = float(f[f"stage{k}"].attrs["lam"])
            b = _binned(occ, BIN)
            t = (np.arange(b.shape[1]) + 0.5) * BIN
            p_on, s_on = _arm_curve(b[start_on])
            p_off, s_off = _arm_curve(b[~start_on])
            p_all, s_all = _arm_curve(b)
            gap = p_on - p_off
            sgap = np.sqrt(s_on ** 2 + s_off ** 2)

            ax = axes[0][j]
            ax.errorbar(t, p_on, s_on, fmt="o-", ms=3, capsize=2,
                        label="start ON", color="C3")
            ax.errorbar(t, p_off, s_off, fmt="s-", ms=3, capsize=2,
                        label="start OFF", color="C0")
            ax.errorbar(t, p_all, s_all, fmt="-", alpha=0.6,
                        label="pooled", color="k")
            ax.set_title(f"stage {k} (λ={lam:.4f})")
            ax.set_xlabel("sample index t")
            ax.set_ylabel("p_ON(t)")
            ax.legend()

            ax2 = axes[1][j]
            ax2.errorbar(t, gap, sgap, fmt="o", ms=3, capsize=2, color="C2")
            ax2.axhline(0, color="gray", lw=0.5)
            ax2.set_xlabel("sample index t")
            ax2.set_ylabel("arm gap p_ON − p_OFF")
            fit = _fit_exp(t, gap, sgap)
            if fit is not None and np.isfinite(fit[1]):
                A, tau = fit
                ax2.plot(t, A * np.exp(-t / tau), "r--",
                         label=f"exp fit τ={tau:.0f}")
                ax2.legend()
            merged = np.abs(gap) < 2 * sgap
            first = int(np.argmax(merged)) if merged.any() else -1
            print(f"stage {k}: lam={lam:.4f}")
            print(f"  final p_ON={p_on[-1]:.4f}±{s_on[-1]:.4f} "
                  f"p_OFF={p_off[-1]:.4f}±{s_off[-1]:.4f} "
                  f"gap={gap[-1]:+.4f}±{sgap[-1]:.4f}")
            if fit is not None:
                print(f"  exp fit: A={fit[0]:.4f}, tau={fit[1]:.0f} samples")
            print(f"  first bin with |gap|<2σ: "
                  f"{'never' if first < 0 or not merged[first:].all() else int(t[first])}")
            drift = p_all[-1] - p_all[len(p_all) // 4]
            print(f"  pooled drift last-3/4: {drift:+.4f} "
                  f"(±{np.sqrt(s_all[-1]**2 + s_all[len(p_all)//4]**2):.4f})")

        fig.tight_layout()
        out = f"{outdir}/growth_mixing_diag.png"
        fig.savefig(out, dpi=150)
        print(f"saved → {out}")


if __name__ == "__main__":
    main()
