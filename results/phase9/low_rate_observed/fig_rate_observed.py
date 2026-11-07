"""Frame-rate figure with the observed-webcam condition (Task 2.2b, exploratory). Style, palette and rcParams of results/phase9/figures/fig_rate.py.
(a) AUC by frame rate: dwell from samples (blue) and I-VT (orange), each on clean data, under simulated measured noise, and under the observed-webcam
    condition without and with the app's per-trial correction. (b) Change from 30 Hz for the dwell model with its 95% paired interval and the 0.02 margin.
Reads rate_table_observed.csv (this folder). Output: fig_rate_observed.pdf, .png, .csv."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "pdf.fonttype": 42})
RATES = [30, 20, 15, 10]
d = pd.read_csv(OUT / "rate_table_observed.csv")
COND = [("clean", "No noise", "o", "-"), ("anch", "Simulated noise", "s", "--"), ("obsnone", "Observed webcam, no correction", "^", ":"),
        ("obsapp", "Observed webcam, with correction", "D", "-.")]
ROUTES = [("sb", "Dwell from samples", BLUE, -0.09), ("ivt", "Fixation detection (I-VT)", ORANGE, 0.09)]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.25, 1]})
x = np.arange(len(RATES))
rows = []
for route, rname, col, off in ROUTES:
    for ck, cname, mk, ls in COND:
        s = d[(d.route == route) & (d.condition_key == ck)].set_index("rate_hz").loc[RATES]
        a.plot(x + off, s.auc.to_numpy(), color=col, marker=mk, ms=4.2, lw=1.1, ls=ls, mec="white", mew=0.4, alpha=0.95)
        for r, v, lo, hi in zip(RATES, s.auc, s.auc_lo, s.auc_hi):
            rows.append({"panel": "a", "series": f"{rname}; {cname}", "rate_hz": r, "value": v, "ci_lo": lo, "ci_hi": hi})
a.set_xticks(x, [str(r) for r in RATES])
a.set_xlabel("Frame rate (Hz)")
a.set_ylabel("AUC, known vs unknown item")
a.set_ylim(0.60, 0.82)
h1 = [Line2D([], [], color=c, lw=1.6, label=n) for _, n, c, _ in ROUTES]
h2 = [Line2D([], [], color=INK2, marker=m, ms=4.2, lw=1.1, ls=l, mec="white", mew=0.4, label=n) for _, n, m, l in COND]
leg1 = a.legend(handles=h1, frameon=False, loc="lower left", fontsize=6.5, handlelength=1.8, borderaxespad=0.1, labelspacing=0.25)
a.add_artist(leg1)
a.legend(handles=h2, frameon=False, loc="lower right", fontsize=6.5, handlelength=2.4, borderaxespad=0.1, labelspacing=0.25)
a.text(-0.17, 1.02, "a", transform=a.transAxes, fontweight="bold", fontsize=10)

R2 = [20, 15, 10]
shade = {"clean": BLUE, "anch": "#6aa0e6", "obsnone": "#8a4ea8", "obsapp": "#1baf7a"}
for i, (ck, cname, mk, ls) in enumerate(COND):
    s = d[(d.route == "sb") & (d.condition_key == ck)].set_index("rate_hz").loc[R2]
    xs = np.arange(len(R2)) + (i - 1.5) * 0.14
    y, lo, hi = s.delta_vs_30hz.to_numpy(), s.delta_lo.to_numpy(), s.delta_hi.to_numpy()
    b.errorbar(xs, y, yerr=[y - lo, hi - y], color=shade[ck], marker=mk, ms=4.2, lw=0, elinewidth=1.0, capsize=0, mec="white", mew=0.4, label=cname)
    for r, v, l, h in zip(R2, y, lo, hi):
        rows.append({"panel": "b", "series": f"Dwell from samples; {cname}", "rate_hz": r, "value": v, "ci_lo": l, "ci_hi": h})
b.axhline(0, color=INK2, lw=0.6)
b.axhline(-0.02, color=ORANGE, lw=0.9, ls="--")
b.text(-0.45, -0.0185, "margin -0.02", color=ORANGE, fontsize=6.5, ha="left", va="bottom")
b.set_xticks(np.arange(len(R2)), [str(r) for r in R2])
b.set_xlim(-0.5, len(R2) - 0.5)
b.set_xlabel("Frame rate (Hz)")
b.set_ylabel("Change in AUC from 30 Hz (dwell from samples)")
b.legend(frameon=False, loc="upper center", ncol=2, fontsize=6.0, handlelength=0.8, borderaxespad=0.1, labelspacing=0.25, columnspacing=0.8)
b.text(-0.2, 1.02, "b", transform=b.transAxes, fontweight="bold", fontsize=10)
b.set_ylim(-0.035, 0.03)
fig.tight_layout(w_pad=1.6)
fig.savefig(OUT / "fig_rate_observed.pdf")
fig.savefig(OUT / "fig_rate_observed.png", dpi=300)
pd.DataFrame(rows).to_csv(OUT / "fig_rate_observed.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string())
