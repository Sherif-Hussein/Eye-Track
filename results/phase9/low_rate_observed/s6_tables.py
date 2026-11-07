"""Task 2.2b: markdown tables for card.md from rate_table_observed.csv and paired_contrasts_observed.csv (numbers only)."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
H = Path(__file__).resolve().parent
d = pd.read_csv(H / "rate_table_observed.csv")
N = {"clean": "clean", "anch": "simulated measured noise", "obsnone": "observed webcam, no correction", "obsapp": "observed webcam, with correction"}
R = {"sb": "Dwell from samples (camera model)", "ivt": "I-VT fixation detection"}
f = lambda v: f"{v:.4f}"
sg = lambda v: f"{v:+.4f}"
out = []
for route in ("sb", "ivt"):
    out.append(f"\n**{R[route]}**\n\n| Condition | Rate (Hz) | AUC [95% CI] | Change from 30 Hz [95% CI] | Lower one-sided 95% bound | p (inferior by 0.02) | Within 0.02 (point / bound) |\n|---|---|---|---|---|---|---|")
    for c in N:
        for fs in (30, 20, 15, 10):
            r = d[(d.route == route) & (d.condition_key == c) & (d.rate_hz == fs)].iloc[0]
            if fs == 30:
                out.append(f"| {N[c]} | 30 | {f(r.auc)} [{f(r.auc_lo)}, {f(r.auc_hi)}] | (reference) | | | |")
            else:
                out.append(f"| {N[c]} | {fs} | {f(r.auc)} [{f(r.auc_lo)}, {f(r.auc_hi)}] | {sg(r.delta_vs_30hz)} [{sg(r.delta_lo)}, {sg(r.delta_hi)}] | {sg(r.ni_lower_onesided95)} | {r.ni_p:.4f} | {'yes' if r.point_within_0_02 else 'no'} / {'yes' if r.ni_pass_lower_above_ni_0_02 else 'no'} |" if False else
                           f"| {N[c]} | {fs} | {f(r.auc)} [{f(r.auc_lo)}, {f(r.auc_hi)}] | {sg(r.delta_vs_30hz)} [{sg(r.delta_lo)}, {sg(r.delta_hi)}] | {sg(r.ni_lower_onesided95)} | {r.ni_p:.4f} | {'yes' if r['point_within_0.02'] else 'no'} / {'yes' if r['ni_pass_lower_above_-0.02'] else 'no'} |")
out.append("\n**Dwell from samples minus I-VT at the same rate (paired)**\n\n| Condition | 30 Hz | 20 Hz | 15 Hz | 10 Hz |\n|---|---|---|---|---|")
for c in N:
    cells = []
    for fs in (30, 20, 15, 10):
        r = d[(d.route == "sb") & (d.condition_key == c) & (d.rate_hz == fs)].iloc[0]
        cells.append(f"{sg(r.dwell_minus_ivt)} [{sg(r.dwell_minus_ivt_lo)}, {sg(r.dwell_minus_ivt_hi)}]")
    out.append(f"| {N[c]} | " + " | ".join(cells) + " |")
out.append("\n**EER and HTER (nested-CV track values, mean over the six group sets and over draws; image level | participant level)**\n\n| Measure | Condition | Rate (Hz) | EER image | HTER image | EER participant | HTER participant |\n|---|---|---|---|---|---|---|")
for route in ("sb", "ivt"):
    for c in N:
        for fs in (30, 20, 15, 10):
            r = d[(d.route == route) & (d.condition_key == c) & (d.rate_hz == fs)].iloc[0]
            out.append(f"| {R[route]} | {N[c]} | {fs} | {r.eer_img:.3f} | {r.hter_img:.3f} | {r.eer_cit:.3f} | {r.hter_cit:.3f} |")
x = pd.read_csv(H / "paired_contrasts_observed.csv")
out.append("\n**Other paired contrasts (AUC difference [95% CI])**\n\n| Measure | Contrast | 30 Hz | 20 Hz | 15 Hz | 10 Hz |\n|---|---|---|---|---|---|")
for route in ("sb", "ivt"):
    for name in x.contrast.unique():
        cells = []
        for fs in (30, 20, 15, 10):
            r = x[(x.route == route) & (x.rate_hz == fs) & (x.contrast == name)].iloc[0]
            cells.append(f"{sg(r['diff'])} [{sg(r.lo)}, {sg(r.hi)}]")
        out.append(f"| {R[route]} | {name} | " + " | ".join(cells) + " |")
(H / "card_tables.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out[:48]))
