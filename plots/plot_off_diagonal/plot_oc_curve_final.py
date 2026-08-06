"""Final composed O_C(delta) curve: 27121 drag shape x 27135 corrected anchor.

log O_C(delta) = log O_C(delta=6) + log r(delta); anchor sem enters every
point as a common (fully correlated) offset, drawn as a band; drag sems are
per-point scatter errors.

Usage: python plots/plot_off_diagonal/plot_oc_curve_final.py
"""

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANCHOR, ANCHOR_SEM = -4.2886, 0.2417          # job 27135 (burn 16k, record 32k)
DRAG_H5 = "data/string_work_kagome_bond_6x6_M227600_K200_n4000_27121.h5"

with h5py.File(DRAG_H5) as f:
    g = f["drag"]
    deltas = np.asarray(g["deltas"])
    log_r = np.asarray(g["log_r_mean"])
    log_r_sem = np.asarray(g["log_r_sem"])

order = np.argsort(deltas)
deltas, log_r, log_r_sem = deltas[order], log_r[order], log_r_sem[order]
log_oc = ANCHOR + log_r

fig, ax = plt.subplots(figsize=(7.2, 5))
ax.errorbar(deltas, log_oc, log_r_sem, fmt="o-", ms=5, capsize=3,
            color="C0", label=r"$\log O_C(\delta)$ (drag pts, per-point sem)")
ax.fill_between(deltas, log_oc - ANCHOR_SEM, log_oc + ANCHOR_SEM,
                alpha=0.18, color="C0",
                label=f"anchor sem ±{ANCHOR_SEM:.2f} (common offset)")
ax.errorbar([6.0], [ANCHOR], [ANCHOR_SEM], fmt="s", ms=8, color="C3",
            capsize=4, label=fr"anchor: $\log O_C(6)={ANCHOR:.2f}\pm{ANCHOR_SEM:.2f}$")
ax.axvspan(4.0, 4.5, alpha=0.10, color="C2",
           label=r"SL-candidate window ($Z_l(2)$ peak)")
ax.set_xlabel(r"$\delta/\Omega$ (seam position $m \to \delta(m)$)")
ax.set_ylabel(r"$\log\,O_C = \log\langle \prod_{i\in C} \sigma^x_i\rangle$  (hexagon plaquette $B_p$)")
ax.set_title("Sweep-state hexagon plaquette coherence, kagome_bond 6x6 PBC, "
             "M=227600\n(drag 27121 x corrected growth anchor 27135)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("plots/oc_curve_final_M227600.png", dpi=150)
print("saved -> plots/oc_curve_final_M227600.png")
for d, lo, se in zip(deltas, log_oc, log_r_sem):
    print(f"  delta={d:4.2f}  log O_C={lo:+.3f}±{se:.3f}(pt)±{ANCHOR_SEM:.3f}(anchor)"
          f"  O_C={np.exp(lo):.4g}")
