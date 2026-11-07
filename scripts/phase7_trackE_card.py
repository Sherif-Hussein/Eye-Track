"""Phase 7 Track E: merge results_E1.json into results.json and write card.md from the CSVs (no new computation)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results" / "phase7" / "trackE"


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{100 * x:.1f}%"


def e2_text():
    d = pd.read_csv(OUT / "E2_conservative_threshold.csv")
    lines = []
    for (t, s), g in d.groupby(["task", "subset"], sort=False):
        if t.startswith("chen") and s != "all":
            continue
        nom = g[g.threshold == "nominal"].iloc[0]
        con = g[g.threshold == "conservative"].iloc[0]
        feas = f"{int(con.folds_feasible)}/{int(con.n_folds)} folds feasible"
        if con.folds_feasible == 0:
            c_txt = f"conservative: infeasible in every fold (m = {con.mean_m_train_neg:.0f} training negatives; bound at k = 0 is {pct(con.cp_upper_at_k0)}), flags nobody (detection 0, FPR 0)"
        else:
            c_txt = (f"conservative ({feas}, k* = {con.mean_kstar:.1f}): detection {pct(con.det)} [{pct(con.det_cp_lo)}, {pct(con.det_cp_hi)}], "
                     f"FPR {pct(con.fpr)} [{pct(con.fpr_cp_lo)}, {pct(con.fpr_cp_hi)}]")
        lines.append(f"{t} ({con.role}): nominal detection {pct(nom.det)} [{pct(nom.det_cp_lo)}, {pct(nom.det_cp_hi)}], FPR {pct(nom.fpr)} [{pct(nom.fpr_cp_lo)}, {pct(nom.fpr_cp_hi)}]; {c_txt}")
    return lines


def e3_text():
    d = pd.read_csv(OUT / "E3_chen_corrected_intervals.csv")
    lines = []
    for _, r in d.iterrows():
        lines.append(f"{r.comparison} on {r.metric}: mean per-fold delta {r.mean_per_fold_delta:+.3f}, corrected 95% CI [{r.corrected_lo:+.3f}, {r.corrected_hi:+.3f}] "
                     f"(naive [{r.naive_lo:+.3f}, {r.naive_hi:+.3f}]; across-repeat [{r.across_repeat_lo:+.3f}, {r.across_repeat_hi:+.3f}])")
    return lines


def e1_text(R):
    if "E1" not in R:
        return ["E1 not run yet."]
    c = pd.DataFrame(R["E1"]["cells"]).set_index("cell")
    dec = pd.DataFrame(R["E1"]["decomposition"]).set_index("quantity")
    out = []
    for m, lab in (("pub", "published P4 SVM"), ("ours", "our pipeline")):
        for fs in (1000, 30):
            o, v = c.loc[f"{m}|{fs}|orig"], c.loc[f"{m}|{fs}|vec"]
            out.append(f"{lab}, {fs} Hz: original code p50 {o.p50_ms:.1f} ms (p95 {o.p95_ms_pooled:.1f}), vectorised p50 {v.p50_ms:.2f} ms (p95 {v.p95_ms_pooled:.2f})")
    d = dec
    g = lambda q: d.loc[q]
    tot = g("total speed-up (pub|1000|orig / ours|30|vec)")
    out.append(f"total speed-up (published SVM, 1000 Hz, original code -> ours, 30 Hz, vectorised): {tot.value:.1f}x [{tot.ci_lo:.1f}, {tot.ci_hi:.1f}] (bootstrap over rounds)")
    for f in ("rate", "code", "model"):
        r = g(f"Shapley (mean log over the six orders) factor: {f}")
        out.append(f"Shapley factor {f}: {r.value:.2f}x [{r.ci_lo:.2f}, {r.ci_hi:.2f}]")
    for q in d.index:
        if q.startswith("edge") or q.startswith("microsaccade"):
            out.append(f"{q}: {d.loc[q].value:.2f}x [{d.loc[q].ci_lo:.2f}, {d.loc[q].ci_hi:.2f}]")
    for tag in ("run1", "run2"):
        f = OUT / f"{tag}_E1_timing_cells.csv"
        if f.exists():
            cc = pd.read_csv(f).set_index("cell")
            out.append(f"replicate {tag} p50 ms: " + ", ".join(f"{k} {v:.1f}" for k, v in cc.p50_ms.items()))
    return out


def main():
    R = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
    p = OUT / "results_E1.json"
    if p.exists():
        R.update(json.loads(p.read_text()))
        (OUT / "results.json").write_text(json.dumps(R, indent=1))
    parts = ["# Phase 7 Track E card", "", "Definitions: `PREREG.md`; deviations: `DEVIATIONS.md`. No model was refitted. No charts.", ""]
    parts += ["## E1 (like-for-like timing)", ""] + [f"- {x}" for x in e1_text(R)] + [""]
    parts += ["## E2 (conservative threshold; unit-level Clopper-Pearson 95% CIs)", ""] + [f"- {x}" for x in e2_text()] + [""]
    parts += ["## E3 (Chen corrected repeated-CV intervals)", ""] + [f"- {x}" for x in e3_text()] + [""]
    (OUT / "card.md").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
