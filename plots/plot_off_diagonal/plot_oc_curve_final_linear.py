"""Final O_C(delta) curve in raw expectation values (linear axis).

Same data as plot_oc_curve_final.py (drag 27121 x anchor 27135); errors
propagated from log: point sem -> O_C*sem, anchor band -> O_C*e^{+-sem}
(asymmetric, fully correlated across points).

Usage: python plots/plot_off_diagonal/plot_oc_curve_final_linear.py
"""

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANCHOR, ANCHOR_SEM = -4.2886, 0.2417          # job 27135
DRAG_H5 = "data/string_work_kagome_bond_6x6_M227600_K200_n4000_27121.h5"

with h5py.File(DRAG_H5) as f:
    g = f["drag"]
    deltas = np.asarray(g["deltas"])
    log_r = np.asarray(g["log_r_mean"])
    log_r_sem = np.asarray(g["log_r_sem"])

order = np.argsort(deltas)
deltas, log_r, log_r_sem = deltas[order], log_r[order], log_r_sem[order]
oc = np.exp(ANCHOR + log_r)
oc_sem = oc * log_r_sem

fig, ax = plt.subplots(figsize=(7.2, 5))
ax.errorbar(deltas, oc, oc_sem, fmt="o-", ms=5, capsize=3, color="C0",
            label=r"$O_C(\delta)$ (drag pts, per-point sem)")
ax.fill_between(deltas, oc * np.exp(-ANCHOR_SEM), oc * np.exp(ANCHOR_SEM),
                alpha=0.18, color="C0",
                label=r"anchor uncertainty $\times e^{\pm0.24}$ (common)")
a_oc = np.exp(ANCHOR)
ax.errorbar([6.0], [a_oc],
            [[a_oc - a_oc * np.exp(-ANCHOR_SEM)],
             [a_oc * np.exp(ANCHOR_SEM) - a_oc]],
            fmt="s", ms=8, color="C3", capsize=4,
            label=fr"anchor: $O_C(6)={a_oc:.4f}^{{+{a_oc*(np.exp(ANCHOR_SEM)-1):.4f}}}"
                  fr"_{{-{a_oc*(1-np.exp(-ANCHOR_SEM)):.4f}}}$")
ax.axvspan(4.0, 4.5, alpha=0.10, color="C2",
           label=r"SL-candidate window ($Z_l(2)$ peak)")
ax.set_xlabel(r"$\delta/\Omega$ (seam position $m \to \delta(m)$)")
ax.set_ylabel(r"$O_C = \langle \prod_{i\in C} \sigma^x_i\rangle$  (hexagon plaquette $B_p$)")
ax.set_title("Sweep-state hexagon plaquette coherence, kagome_bond 6x6 PBC, "
             "M=227600\n(raw expectation values; drag 27121 x anchor 27135)")
ax.set_ylim(bottom=0)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("plots/oc_curve_final_M227600_linear.png", dpi=150)
print("saved -> plots/oc_curve_final_M227600_linear.png")
