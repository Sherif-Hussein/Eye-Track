"""Frame-rate figure for Paper A: (a) recognition AUC by rate; (b) agreement with the research tracker's own parser by rate.
Reads results/phase9/low_rate/rate_table.csv and event_diag_agreement_long.csv. Output: fig_rate.pdf, fig_rate.png, fig_rate.csv in this folder."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
LR = OUT.parent / "low_rate"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "pdf.fonttype": 42})
RATES = [30, 20, 15, 10]
d = pd.read_csv(LR / "rate_table.csv")
d = d[d.dataset == "Delmas"]
d = d[d.rate_hz.isin(RATES)]


def pick(model_part, cond_part):
    s = d[d.model.str.contains(model_part, regex=False) & d.condition.str.contains(cond_part, regex=False)].set_index("rate_hz")
    return s.loc[RATES]


SERIES = [("Dwell from samples, webcam noise", pick("sb: dwell", "measured"), BLUE, "o", "-"),
          ("Fixation detection, webcam noise", pick("I-VT", "measured"), ORANGE, "s", "-"),
          ("Fuller model with pupil, no noise", pick("our 13", "clean"), AQUA, "^", "--")]
fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.45), gridspec_kw={"width_ratios": [1.15, 1]})
x = np.arange(len(RATES))
rows = []
for i, (name, s, col, mk, ls) in enumerate(SERIES):
    off = (i - 1) * 0.07
    y, lo, hi = s.auc.to_numpy(), s.ci_lo.to_numpy(), s.ci_hi.to_numpy()
    a.errorbar(x + off, y, yerr=[y - lo, hi - y], color=col, marker=mk, ms=4.5, lw=1.4, ls=ls, elinewidth=0.7, capsize=0, label=name, mec="white", mew=0.5)
    for r, v, l, h in zip(RATES, y, lo, hi):
        rows.append({"panel": "a", "series": name, "rate_hz": r, "value": v, "ci_lo": l, "ci_hi": h})
a.set_xticks(x, [str(r) for r in RATES])
a.set_xlabel("Frame rate (Hz)")
a.set_ylabel("AUC, known vs unknown item")
a.set_ylim(0.68, 0.90)
a.legend(frameon=False, loc="upper center", ncol=1, handlelength=2.2, borderaxespad=0.1, labelspacing=0.25)
a.text(-0.2, 1.02, "a", transform=a.transAxes, fontweight="bold", fontsize=10)

g = pd.read_csv(LR / "event_diag_agreement_long.csv")
SER2 = [("sample_dwell_2_vs_parser", "Dwell from samples", BLUE, "o"), ("dwell_2", "Dwell from detected fixations", AQUA, "^"), ("count", "Fixation count", ORANGE, "s")]
for key, name, col, mk in SER2:
    s = g[g.metric == key].set_index("rate_hz").loc[RATES, "r_participant_x_image"].to_numpy()
    b.plot(x, s, color=col, marker=mk, ms=4.5, lw=1.4, mec="white", mew=0.5)
    b.text(x[-1] - 0.05, s[-1] + (0.035 if key == "sample_dwell_2_vs_parser" else -0.05 if key == "dwell_2" else 0.04), name, color=INK, ha="right", fontsize=7)
    for r, v in zip(RATES, s):
        rows.append({"panel": "b", "series": name, "rate_hz": r, "value": v, "ci_lo": np.nan, "ci_hi": np.nan})
b.set_xticks(x, [str(r) for r in RATES])
b.set_xlabel("Frame rate (Hz)")
b.set_ylabel("Agreement with tracker's parser ($r$)")
b.set_ylim(0.4, 1.05)
b.text(-0.2, 1.02, "b", transform=b.transAxes, fontweight="bold", fontsize=10)
fig.tight_layout(w_pad=1.6)
fig.savefig(OUT / "fig_rate.pdf")
fig.savefig(OUT / "fig_rate.png", dpi=300)
pd.DataFrame(rows).to_csv(OUT / "fig_rate.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string())
