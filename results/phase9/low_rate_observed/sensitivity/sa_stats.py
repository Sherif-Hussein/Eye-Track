"""Task 2.2c Part A (EXPLORATORY): statistics and tables, 2.2b observed conditions beside the reduced-slow-jitter conditions.
Reads (read only) results/phase9/low_rate/checkpoints_webcam (clean, 'anch'), results/phase9/low_rate_observed/checkpoints (2.2b observed) and
sensitivity/checkpoints (reduced). Same method as 2.2b s5_stats.py (re-typed because s5_stats writes into the 2.2b folder): ONE call of
results/phase9/low_rate/lr_common.py paired_bootstrap (unchanged), 2000 participant-cluster resamples, seed 20261102, AUC interval = first 300
resamples, every difference = all 2000; non-inferiority against 0.02 (lower bound = 5th percentile, p = (1 + #{diff <= -0.02}) / 2001).
The 2.2b rows must be reproduced exactly (checked against rate_table_observed.csv). Writes sens_rate_table.csv, sens_contrasts.csv, sens_tables.md,
sens_repro_check.json here.
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
LRO = HERE.parent
sys.path.insert(0, str(LRO))
import lro_common as C  # noqa: E402  (2.2b, unchanged; gives the paths and the thread settings)

sys.stdout.reconfigure(encoding="utf-8")
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lr_common as L  # noqa: E402

RATES = [30, 20, 15, 10]
MARGIN = 0.02
LOWCK = C.LOWRATE / "checkpoints_webcam"
CK22B = LRO / "checkpoints"
CKRED = HERE / "checkpoints"
CONDS = ["clean", "anch", "obsnone", "rednone", "obsapp", "redapp"]
NAME = {"clean": "clean EyeLink data", "anch": "simulated measured noise", "obsnone": "observed webcam (2.2b), no correction",
        "rednone": "observed webcam, reduced slow jitter, no correction", "obsapp": "observed webcam (2.2b), with correction",
        "redapp": "observed webcam, reduced slow jitter, with correction"}
ROUTE = {"sb": "Dwell from samples (camera model)", "ivt": "I-VT fixation detection"}


def paths(route, cond, fs):
    if cond == "clean":
        return [(LOWCK, f"{route}np_clean_{fs}Hz_d0")]
    if cond == "anch":
        return [(LOWCK, f"{route}np_anch_{fs}Hz_d{d}") for d in (1, 2, 3)]
    base = CK22B if cond.startswith("obs") else CKRED
    return [(base, f"{route}np_obs{cond[3:]}_{fs}Hz_d{d}") for d in (1, 2, 3)]


def metrics(route, cond, fs):
    v = {"eer_img": [], "hter_img": [], "eer_cit": [], "hter_cit": []}
    for dirp, tag in paths(route, cond, fs):
        r = json.loads((dirp / f"{tag}.json").read_text())["res"]["mean_over_sets"]
        v["eer_img"].append(r["image"]["eer"]); v["hter_img"].append(r["image"]["hter"])
        v["eer_cit"].append(r["participant_cit"]["eer"]); v["hter_cit"].append(r["participant_cit"]["hter"])
    return {k: float(np.mean(x)) for k, x in v.items()}


def main():
    cfg = {f"{r}|{c}|{fs}": [d / f"{t}_oof.csv.gz" for d, t in paths(r, c, fs)] for r in ("sb", "ivt") for c in CONDS for fs in RATES}
    missing = [str(p) for v in cfg.values() for p in v if not p.exists()]
    assert not missing, missing[:5]
    res, est = L.paired_bootstrap(cfg, ref="sb|clean|30", n_boot=2000, seed=20261102, return_draws=True)
    rows = []
    for route in ("sb", "ivt"):
        for cond in CONDS:
            for fs in RATES:
                k, k30 = f"{route}|{cond}|{fs}", f"{route}|{cond}|30"
                d = est[k] - est[k30]
                v = res[k]
                row = {"route": route, "model": ROUTE[route], "condition_key": cond, "condition": NAME[cond], "rate_hz": fs, "auc": v["auc"], "auc_lo": v["auc_lo"],
                       "auc_hi": v["auc_hi"], "delta_vs_30hz": v["auc"] - res[k30]["auc"], "delta_lo": float(np.percentile(d, 2.5)), "delta_hi": float(np.percentile(d, 97.5))}
                if fs != 30:
                    row.update({"ni_lower_onesided95": float(np.percentile(d, 5)), "ni_p": float((1 + np.sum(d <= -MARGIN)) / (1 + len(d))),
                                "ni_pass_lower_above_-0.02": bool(np.percentile(d, 5) > -MARGIN), "point_within_0.02": bool(row["delta_vs_30hz"] > -MARGIN)})
                if route == "sb":
                    c = est[k] - est[f"ivt|{cond}|{fs}"]
                    row.update({"dwell_minus_ivt": v["auc"] - res[f"ivt|{cond}|{fs}"]["auc"], "dwell_minus_ivt_lo": float(np.percentile(c, 2.5)),
                                "dwell_minus_ivt_hi": float(np.percentile(c, 97.5))})
                row.update(metrics(route, cond, fs))
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "sens_rate_table.csv", index=False)
    ex = []
    pairs = (("rednone", "clean", "reduced, no correction - clean"), ("redapp", "clean", "reduced, with correction - clean"),
             ("rednone", "obsnone", "reduced - 2.2b observed, no correction"), ("redapp", "obsapp", "reduced - 2.2b observed, with correction"),
             ("rednone", "anch", "reduced, no correction - simulated measured noise"), ("redapp", "rednone", "correction effect, reduced (with - without)"),
             ("obsnone", "clean", "2.2b observed, no correction - clean"), ("obsapp", "clean", "2.2b observed, with correction - clean"))
    for route in ("sb", "ivt"):
        for fs in RATES:
            for a, b, name in pairs:
                ka, kb = f"{route}|{a}|{fs}", f"{route}|{b}|{fs}"
                d = est[ka] - est[kb]
                ex.append({"route": route, "rate_hz": fs, "contrast": name, "diff": res[ka]["auc"] - res[kb]["auc"], "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))})
    pd.DataFrame(ex).to_csv(HERE / "sens_contrasts.csv", index=False)
    # the 2.2b rows must be reproduced exactly
    old = pd.read_csv(LRO / "rate_table_observed.csv")
    cols = ["auc", "auc_lo", "auc_hi", "delta_vs_30hz", "delta_lo", "delta_hi", "eer_img", "hter_img", "eer_cit", "hter_cit"]
    mx = 0.0
    for _, o in old[old.condition_key.isin(["clean", "anch", "obsnone", "obsapp"])].iterrows():
        r = df[(df.route == o.route) & (df.condition_key == o.condition_key) & (df.rate_hz == o.rate_hz)].iloc[0]
        mx = max(mx, max(abs(r[c] - o[c]) for c in cols))
        if o.route == "sb":
            mx = max(mx, abs(r.dwell_minus_ivt_lo - o.dwell_minus_ivt_lo), abs(r.dwell_minus_ivt_hi - o.dwell_minus_ivt_hi))
    C.jdump(HERE / "sens_repro_check.json", {"max_abs_difference_2.2b_rows_vs_rate_table_observed": mx, "passed": bool(mx < 1e-9)})
    C.log("2.2b rows reproduced, max abs difference", mx)
    write_tables(df, pd.DataFrame(ex))


def write_tables(d, x):
    f = lambda v: f"{v:.4f}"  # noqa: E731
    sg = lambda v: f"{v:+.4f}"  # noqa: E731
    out = []
    for route in ("sb", "ivt"):
        for corr, (o, r) in {"no correction": ("obsnone", "rednone"), "with the app's correction": ("obsapp", "redapp")}.items():
            out.append(f"\n**{ROUTE[route]}, {corr}** (2.2b = full slow jitter; reduced = slow jitter variance scaled down)\n\n| Rate (Hz) | 2.2b AUC [95% CI] | 2.2b change from 30 Hz [95% CI] | Reduced AUC [95% CI] | Reduced change from 30 Hz [95% CI] | Reduced lower one-sided 95% bound | p (inferior by 0.02) | Within 0.02 (point / bound) |\n|---|---|---|---|---|---|---|---|")
            for fs in RATES:
                a = d[(d.route == route) & (d.condition_key == o) & (d.rate_hz == fs)].iloc[0]
                b = d[(d.route == route) & (d.condition_key == r) & (d.rate_hz == fs)].iloc[0]
                if fs == 30:
                    out.append(f"| 30 | {f(a.auc)} [{f(a.auc_lo)}, {f(a.auc_hi)}] | (reference) | {f(b.auc)} [{f(b.auc_lo)}, {f(b.auc_hi)}] | (reference) | | | |")
                else:
                    out.append(f"| {fs} | {f(a.auc)} [{f(a.auc_lo)}, {f(a.auc_hi)}] | {sg(a.delta_vs_30hz)} [{sg(a.delta_lo)}, {sg(a.delta_hi)}] | {f(b.auc)} [{f(b.auc_lo)}, {f(b.auc_hi)}] | "
                               f"{sg(b.delta_vs_30hz)} [{sg(b.delta_lo)}, {sg(b.delta_hi)}] | {sg(b.ni_lower_onesided95)} | {b.ni_p:.4f} | {'yes' if b['point_within_0.02'] else 'no'} / {'yes' if b['ni_pass_lower_above_-0.02'] else 'no'} |")
    out.append("\n**Dwell from samples minus I-VT at the same rate (paired)**\n\n| Condition | 30 Hz | 20 Hz | 15 Hz | 10 Hz |\n|---|---|---|---|---|")
    for c in ("obsnone", "rednone", "obsapp", "redapp"):
        cells = []
        for fs in RATES:
            r = d[(d.route == "sb") & (d.condition_key == c) & (d.rate_hz == fs)].iloc[0]
            cells.append(f"{sg(r.dwell_minus_ivt)} [{sg(r.dwell_minus_ivt_lo)}, {sg(r.dwell_minus_ivt_hi)}]")
        out.append(f"| {NAME[c]} | " + " | ".join(cells) + " |")
    out.append("\n**Paired AUC differences [95% CI]**\n\n| Measure | Contrast | 30 Hz | 20 Hz | 15 Hz | 10 Hz |\n|---|---|---|---|---|---|")
    for route in ("sb", "ivt"):
        for name in x.contrast.unique():
            cells = [f"{sg(r['diff'])} [{sg(r.lo)}, {sg(r.hi)}]" for fs in RATES for r in [x[(x.route == route) & (x.rate_hz == fs) & (x.contrast == name)].iloc[0]]]
            out.append(f"| {ROUTE[route]} | {name} | " + " | ".join(cells) + " |")
    out.append("\n**EER and HTER (nested-CV track values, mean over the six group sets and over draws)**\n\n| Measure | Condition | Rate (Hz) | EER image | HTER image | EER participant | HTER participant |\n|---|---|---|---|---|---|---|")
    for route in ("sb", "ivt"):
        for c in ("obsnone", "rednone", "obsapp", "redapp"):
            for fs in RATES:
                r = d[(d.route == route) & (d.condition_key == c) & (d.rate_hz == fs)].iloc[0]
                out.append(f"| {ROUTE[route]} | {NAME[c]} | {fs} | {r.eer_img:.3f} | {r.hter_img:.3f} | {r.eer_cit:.3f} | {r.hter_cit:.3f} |")
    (HERE / "sens_tables.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[:30]))


if __name__ == "__main__":
    main()
