"""Blocking-analysis error diagnostic for the off-diagonal string-work engine.

O_C = ⟨J⟩ over Jarzynski trajectories (J = e^{log J}; failed trajectories have
J = 0, i.e. log J = −inf).  Two error curves are drawn:

- block-SEM of ⟨J⟩ (the estimator is linear in J), and
- leave-one-block-out jackknife of log O_C,

both as RELATIVE errors so they are directly comparable (err(log x) ≈ δx/x).
The two agreeing at the plateau is the healthy case; a large gap is a
heavy-tail warning (consistent with a small ESS / large zero_frac).

Data sources (auto-detected): chunk dir <dir>/K{K}/rank{r}.h5 with
chunk{i}/log_j_samples (per-rank order preserved — preferred), or the
aggregate HDF5 with K{K}/log_j_samples.

Usage::

    python plots/plot_off_diagonal/plot_error_vs_binsize.py --data data/strwork_..._chunks
    python plots/plot_off_diagonal/plot_error_vs_binsize.py --data data/strwork.h5
"""

import argparse
import glob
import os
import re
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.blocking import blocking_curve_jackknife, blocking_curve_linear


def j_weights(log_j):
    """J = e^{log J} with log J = −inf → 0 (failed trajectories)."""
    log_j = np.asarray(log_j, dtype=np.float64)
    j = np.zeros_like(log_j)
    finite = np.isfinite(log_j)
    j[finite] = np.exp(log_j[finite])
    return j


def log_oc(j_samples):
    m = np.mean(j_samples)
    return np.log(m) if m > 0 else -np.inf


def ess_fraction(j):
    s = j.sum()
    return float(s * s / (np.sum(j * j) * len(j))) if s > 0 else 0.0


def _load_chunk_dir(k_dir):
    chains = []
    for fn in sorted(glob.glob(os.path.join(k_dir, "rank*.h5")),
                     key=lambda p: int(re.search(r"rank(\d+)\.h5$", p).group(1))):
        with h5py.File(fn, "r") as f:
            idxs = sorted(int(m.group(1)) for name in f.keys()
                          for m in [re.fullmatch(r"chunk(\d+)", name)] if m)
            series = [f[f"chunk{i}"]["log_j_samples"][:] for i in idxs]
        if series:
            chains.append(np.concatenate(series))
    return chains


def load_chains(data_path):
    """{K: [per-rank 1D log_j arrays]} from a chunk dir or aggregate .h5."""
    out = {}
    if os.path.isdir(data_path):
        k_dirs = sorted(glob.glob(os.path.join(data_path, "K*")))
        if not k_dirs and glob.glob(os.path.join(data_path, "rank*.h5")):
            k_dirs = [data_path]
        for kd in k_dirs:
            m = re.search(r"K(\d+)$", kd.rstrip("/"))
            K = int(m.group(1)) if m else -1
            chains = _load_chunk_dir(kd)
            if chains:
                out[K] = chains
        return out

    with h5py.File(data_path, "r") as f:
        root = f["K_results"] if "K_results" in f else f
        params = f["params"].attrs if "params" in f else f.attrs
        n_ranks = int(params.get("n_ranks", 1))
        for name in root:
            m = re.fullmatch(r"K_?(\d+)", name)
            if not m or "log_j_samples" not in root[name]:
                continue
            lj = root[name]["log_j_samples"][:]
            n = len(lj)
            base_n, rem = divmod(n, n_ranks)
            chains, ofs = [], 0
            for r in range(n_ranks):
                cnt = base_n + (1 if r < rem else 0)
                if cnt > 0:
                    chains.append(lj[ofs:ofs + cnt])
                ofs += cnt
            out[int(m.group(1))] = chains
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True,
                        help="chunk dir (K{K}/rank{r}.h5) or aggregate .h5")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    by_k = load_chains(args.data)
    if not by_k:
        raise FileNotFoundError(f"no log_j_samples found under {args.data!r}")

    ks = sorted(by_k)
    fig, axes = plt.subplots(1, len(ks), figsize=(5.2 * len(ks), 4.4),
                             squeeze=False)
    label = os.path.basename(os.path.normpath(args.data))
    fig.suptitle(f"String work O_C — blocking error diagnostic ({label})",
                 fontsize=13)

    for ax, K in zip(axes[0], ks):
        j_chains = [j_weights(c) for c in by_k[K]]
        pooled = np.concatenate(j_chains)
        oc = pooled.mean()
        zero_frac = float(np.mean(pooled == 0.0))

        bs_l, sems, _ = blocking_curve_linear(j_chains)
        bs_j, jerrs, _ = blocking_curve_jackknife(j_chains, log_oc)

        rel_sem = sems / oc if oc > 0 else sems * np.nan
        ax.plot(bs_l, rel_sem, "o-", color="#2B4EAE", lw=1.5, ms=4,
                label="block-SEM(⟨J⟩)/O_C")
        ax.plot(bs_j, jerrs, "s--", color="#C0392B", lw=1.5, ms=4,
                label="jackknife err(log O_C)")
        ax.axhline(rel_sem[0], color="k", lw=0.9, ls=":", alpha=0.6,
                   label=f"naive (b=1): {rel_sem[0]:.4f}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("block size b (trajectories)")
        ax.set_ylabel("relative error")
        ax.grid(True, alpha=0.3, ls="--")
        title_oc = (f"O_C = {oc:.4g} ± {sems[-1]:.2g} "
                    f"(log O_C = {log_oc(pooled):+.4f})")
        ax.set_title(f"K={K}:  {title_oc}\n"
                     f"n_traj={len(pooled)} ({len(j_chains)} chains), "
                     f"ESS={ess_fraction(pooled) * 100:.1f}%, "
                     f"zero_frac={zero_frac * 100:.1f}%",
                     fontsize=10)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig_base = os.path.basename(os.path.normpath(args.data))
    if fig_base.endswith(".h5"):
        fig_base = fig_base[:-3]
    out = args.out or os.path.join("figures", fig_base, "err_vs_binsize.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
