"""Blocking-analysis error diagnostic for the Renyi work engine (ΔS₂).

ΔS₂ = −log⟨e^{−W}⟩ over Jarzynski trajectories.  The estimator is nonlinear,
so the error at each block size is a leave-one-block-out jackknife; the curve
err(b) should plateau once blocks exceed the residual trajectory correlation
(introduced by the shared checkpoint chain).  A still-rising curve means the
naive error is an underestimate.  The panel title shows the estimate ± the
largest-block error; ESS = (Σw)²/Σw² with w = e^{−W} is annotated as a
heavy-tail diagnostic.

Data sources (auto-detected):
- chunk dir from --checkpoint-every-trajectories:  <dir>/K{K}/rank{r}.h5
  (per-rank trajectory order preserved — preferred), or a single K{K} dir.
- aggregate HDF5 from --filepath: K{K}/work_samples (rank-concatenated;
  chain boundaries reconstructed from the n_ranks attr).

Usage::

    python plots/plot_renyi_work/plot_error_vs_binsize.py --data data/foo_chunks
    python plots/plot_renyi_work/plot_error_vs_binsize.py --data data/foo.h5
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
from common.blocking import blocking_curve_jackknife

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def delta_s2(work_samples):
    """ΔS₂ = −log⟨e^{−W}⟩, evaluated with a max-shift for stability."""
    w = np.asarray(work_samples, dtype=np.float64)
    shift = (-w).max()
    return -(shift + np.log(np.mean(np.exp(-w - shift))))


def ess_fraction(work_samples):
    w = np.exp(-(np.asarray(work_samples) - np.asarray(work_samples).min()))
    return float((w.sum() ** 2) / (np.sum(w ** 2) * len(w)))


def _load_chunk_dir(k_dir):
    """{rank: 1D work series in trajectory order} from K{K}/rank{r}.h5 files."""
    chains = []
    for fn in sorted(glob.glob(os.path.join(k_dir, "rank*.h5")),
                     key=lambda p: int(re.search(r"rank(\d+)\.h5$", p).group(1))):
        with h5py.File(fn, "r") as f:
            idxs = sorted(int(m.group(1)) for name in f.keys()
                          for m in [re.fullmatch(r"chunk(\d+)", name)] if m)
            series = [f[f"chunk{i}"]["work_samples"][:] for i in idxs]
        if series:
            chains.append(np.concatenate(series))
    return chains


def load_chains(data_path, region=None):
    """{K: [per-rank 1D work arrays]} from a chunk dir or an aggregate .h5."""
    out = {}
    if os.path.isdir(data_path):
        base = data_path if region is None else os.path.join(data_path, region)
        k_dirs = sorted(glob.glob(os.path.join(base, "K*")))
        if not k_dirs and glob.glob(os.path.join(base, "rank*.h5")):
            k_dirs = [base]                     # already inside a K{K} dir
        for kd in k_dirs:
            m = re.search(r"K(\d+)$", kd.rstrip("/"))
            K = int(m.group(1)) if m else -1
            chains = _load_chunk_dir(kd)
            if chains:
                out[K] = chains
        return out

    with h5py.File(data_path, "r") as f:
        root = f
        if region is not None and "regions" in f:
            root = f[f"regions/{region}"]
        if "K_results" in root:
            root = root["K_results"]
        params = f["params"].attrs if "params" in f else f.attrs
        n_ranks = int(params.get("n_ranks", 1))
        for name in root:
            m = re.fullmatch(r"K_?(\d+)", name)
            if not m or "work_samples" not in root[name]:
                continue
            w = root[name]["work_samples"][:]
            # gather order = rank order; rank r holds base + (1 if r < rem)
            n = len(w)
            base_n, rem = divmod(n, n_ranks)
            chains, ofs = [], 0
            for r in range(n_ranks):
                cnt = base_n + (1 if r < rem else 0)
                if cnt > 0:
                    chains.append(w[ofs:ofs + cnt])
                ofs += cnt
            out[int(m.group(1))] = chains
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True,
                        help="chunk dir (K{K}/rank{r}.h5) or aggregate .h5")
    parser.add_argument("--region", default=None,
                        help="region subdir / group for KP-regions outputs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    by_k = load_chains(args.data, region=args.region)
    if not by_k:
        raise FileNotFoundError(f"no work_samples found under {args.data!r}")

    ks = sorted(by_k)
    fig, axes = plt.subplots(1, len(ks), figsize=(5.2 * len(ks), 4.4),
                             squeeze=False)
    label = os.path.basename(os.path.normpath(args.data))
    if args.region:
        label += f" [{args.region}]"
    fig.suptitle(f"Renyi work ΔS₂ — blocking error diagnostic ({label})",
                 fontsize=13)

    for ax, K in zip(axes[0], ks):
        chains = by_k[K]
        pooled = np.concatenate(chains)
        est = delta_s2(pooled)
        bs, errs, counts = blocking_curve_jackknife(chains, delta_s2)

        ax.plot(bs, errs, "o-", color="#2B4EAE", lw=1.5, ms=4)
        ax.axhline(errs[0], color="k", lw=0.9, ls=":", alpha=0.6,
                   label=f"naive (b=1): {errs[0]:.4f}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("block size b (trajectories)")
        ax.set_ylabel("jackknife error of ΔS₂")
        ax.grid(True, alpha=0.3, ls="--")
        ax.set_title(f"K={K}:  ΔS₂ = {est:.4f} ± {errs[-1]:.4f}\n"
                     f"n_traj={len(pooled)} ({len(chains)} chains), "
                     f"ESS={ess_fraction(pooled) * 100:.1f}%",
                     fontsize=10)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig_base = os.path.basename(os.path.normpath(args.data))
    if fig_base.endswith(".h5"):
        fig_base = fig_base[:-3]
    name = "err_vs_binsize" + (f"_{args.region}" if args.region else "")
    out = args.out or os.path.join("figures", fig_base, f"{name}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
