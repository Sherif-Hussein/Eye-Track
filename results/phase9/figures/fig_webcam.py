"""Combined webcam figure for Paper A: (a) gaze against known targets; (b) offset at the central point over the same session.
Reads fig_accuracy_targets_samples.csv, fig_accuracy_targets_targets.csv and fig_drift.csv (numbers only; EMEET recording R4). Output: fig_webcam.pdf/.png."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#f0efec"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42})
s = pd.read_csv(OUT / "fig_accuracy_targets_samples.csv")
t = pd.read_csv(OUT / "fig_accuracy_targets_targets.csv")
d = pd.read_csv(OUT / "fig_drift.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.6), gridspec_kw={"width_ratios": [1.12, 1]})
a.scatter(s.x_deg, s.y_deg, s=7, color=BLUE, alpha=0.4, lw=0)
for _, r in t.iterrows():
    a.plot([r.tx_deg, r.median_x_deg], [r.ty_deg, r.median_y_deg], color=INK, lw=1.2)
a.scatter(t.median_x_deg, t.median_y_deg, s=26, color=INK, zorder=3)
a.scatter(t.tx_deg, t.ty_deg, s=60, marker="+", color=ORANGE, lw=1.8, zorder=4)
a.set_xlabel("Horizontal position (°)")
a.set_ylabel("Vertical position (°)")
a.set_xlim(-9, 9)
a.set_ylim(7, -7)
a.text(-8.6, -6.3, "mean offset %.2f°" % t.offset_deg.mean(), fontsize=7)
a.text(-0.2, 1.03, "a", transform=a.transAxes, fontweight="bold", fontsize=10)

prac, sc = d[d.kind == "practice"], d[d.kind != "practice"]
xp = np.arange(len(prac))
xs = np.arange(len(prac), len(prac) + len(sc))
b.axvspan(-0.5, len(prac) - 0.5, color=BAND, lw=0)
b.plot(np.r_[xp, xs], np.r_[prac.drift_deg, sc.drift_deg], color=INK, lw=1, zorder=2)
b.scatter(xp, prac.drift_deg, s=22, facecolor="white", edgecolor=INK, zorder=3)
b.scatter(xs, sc.drift_deg, s=18, color=INK, zorder=3)
b.axhline(4, color=INK2, lw=0.8, ls="--")
med = sc.drift_deg.median()
b.axhline(med, color=BLUE, lw=1.2)
b.text(xs[0] - 0.3, 4.12, "re-check limit 4°", ha="left", fontsize=7, color=INK2)
b.text(xs[-1], med - 0.42, "median %.1f°" % med, ha="right", fontsize=7, color=BLUE)
b.set_xticks(np.r_[[xp.mean()], xs[::3]], ["P"] + [str(i + 1) for i in range(len(sc))][::3])
b.set_xlabel("Trial (P = practice)")
b.set_ylabel("Offset at central point (°)")
b.set_ylim(0, 7)
b.text(-0.2, 1.03, "b", transform=b.transAxes, fontweight="bold", fontsize=10)
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / "fig_webcam.pdf")
fig.savefig(OUT / "fig_webcam.png", dpi=300)
print("mean offset", round(t.offset_deg.mean(), 3), "median drift", round(med, 3), "n scored", len(sc))
