"""Task 2.2b (EXPLORATORY): statistics and tables. Reads (read only) the Task 2.2 checkpoints of results/phase9/low_rate/checkpoints_webcam
(clean and measured-noise 'anch' conditions) and this folder's checkpoints/ (observed-webcam, without and with the app's per-trial correction).

All configurations go through ONE call of results/phase9/low_rate/lr_common.py paired_bootstrap (unchanged): 2000 participant-cluster
resamples per group set, seed 20261102, the same resamples for every configuration, so all differences are paired. A condition's AUC is the mean over
its noise draws (3; clean 1), as in s4_webcam.py. AUC interval: first 300 resamples (as in the earlier tables); every difference: all 2000.
Non-inferiority (one-sided, margin 0.02 AUC): delta = AUC(rate) - AUC(30 Hz) of the same route and condition; lower one-sided 95% bound = 5th
percentile of the 2000 paired differences; p = (1 + #{diff <= -0.02}) / (1 + 2000), the formula of scripts/phase6_trackC_c1.py stage_stats and
edd.delmas_raw_matched.noninf_row. Exploratory; no multiplicity correction.
EER / HTER: the nested-CV track's own image-level and participant-level values (mean over the six group sets), mean over draws.
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import lro_common as C  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lr_common as L  # noqa: E402

RATES = [30, 20, 15, 10]
MARGIN = 0.02
LOWCK = C.LOWRATE / "checkpoints_webcam"
ROUTE = {"sb": "Dwell from samples (camera model, 10 features, no pupil)", "ivt": "I-VT fixation detection (11 features, no pupil)"}
COND = {"clean": "clean EyeLink data", "anch": "simulated measured noise (constant offset, white jitter, 3% loss)",
        "obsnone": "observed webcam, no correction", "obsapp": "observed webcam, with the app's per-trial correction"}


def paths(route, cond, fs):
    if cond == "clean":
        return [(LOWCK, f"{route}np_clean_{fs}Hz_d0")]
    if cond == "anch":
        return [(LOWCK, f"{route}np_anch_{fs}Hz_d{d}") for d in (1, 2, 3)]
    return [(C.CK, f"{route}np_{cond[:3]}{cond[3:]}_{fs}Hz_d{d}") for d in (1, 2, 3)]


def metrics(route, cond, fs):
    import json
    v = {"eer_img": [], "hter_img": [], "eer_cit": [], "hter_cit": []}
    for dirp, tag in paths(route, cond, fs):
        r = json.loads((dirp / f"{tag}.json").read_text())["res"]["mean_over_sets"]
        v["eer_img"].append(r["image"]["eer"]); v["hter_img"].append(r["image"]["hter"])
        v["eer_cit"].append(r["participant_cit"]["eer"]); v["hter_cit"].append(r["participant_cit"]["hter"])
    return {k: float(np.mean(x)) for k, x in v.items()}


def main():
    cfg = {}
    for route in ("sb", "ivt"):
        for cond in COND:
            for fs in RATES:
                cfg[f"{route}|{cond}|{fs}"] = [d / f"{t}_oof.csv.gz" for d, t in paths(route, cond, fs)]
        for fs in RATES:
            cfg[f"{route}|regular|{fs}"] = [C.CK / f"{route}np_obsregular_{fs}Hz_d1_oof.csv.gz"]
    missing = [str(p) for v in cfg.values() for p in v if not p.exists()]
    assert not missing, missing[:5]
    res, est = L.paired_bootstrap(cfg, ref="sb|clean|30", n_boot=2000, seed=20261102, return_draws=True)
    rows = []
    for route in ("sb", "ivt"):
        for cond in list(COND) + ["regular"]:
            for fs in RATES:
                k = f"{route}|{cond}|{fs}"
                k30 = f"{route}|{cond}|30"
                d = est[k] - est[k30]
                v = res[k]
                row = {"dataset": "Delmas", "route": route, "model": ROUTE[route], "condition_key": cond,
                       "condition": COND.get(cond, "regular frame times, no error (pipeline check)"), "rate_hz": fs,
                       "auc": v["auc"], "auc_lo": v["auc_lo"], "auc_hi": v["auc_hi"], "auc_draw_sd": float(np.std(v["per_draw_auc"])) if len(v["per_draw_auc"]) > 1 else np.nan,
                       "delta_vs_30hz": v["auc"] - res[k30]["auc"], "delta_lo": float(np.percentile(d, 2.5)), "delta_hi": float(np.percentile(d, 97.5))}
                if fs != 30:
                    row.update({"ni_lower_onesided95": float(np.percentile(d, 5)), "ni_p": float((1 + np.sum(d <= -MARGIN)) / (1 + len(d))),
                                "ni_pass_lower_above_-0.02": bool(np.percentile(d, 5) > -MARGIN), "point_within_0.02": bool(row["delta_vs_30hz"] > -MARGIN)})
                if route == "sb":
                    c = est[k] - est[f"ivt|{cond}|{fs}"]
                    row.update({"dwell_minus_ivt": v["auc"] - res[f"ivt|{cond}|{fs}"]["auc"], "dwell_minus_ivt_lo": float(np.percentile(c, 2.5)),
                                "dwell_minus_ivt_hi": float(np.percentile(c, 97.5))})
                if cond != "regular":
                    row.update(metrics(route, cond, fs))
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(C.HERE / "rate_table_observed.csv", index=False)
    # extra paired contrasts (same resamples)
    ex = []
    for route in ("sb", "ivt"):
        for fs in RATES:
            for a, b, name in (("obsapp", "obsnone", "correction effect (with - without)"), ("obsnone", "anch", "observed (no correction) - simulated measured noise"),
                               ("obsapp", "anch", "observed (with correction) - simulated measured noise"), ("anch", "clean", "simulated measured noise - clean"),
                               ("obsapp", "clean", "observed (with correction) - clean"), ("obsnone", "clean", "observed (no correction) - clean"),
                               ("regular", "clean", "regular timing through this pipeline - clean (pipeline check)")):
                ka, kb = f"{route}|{a}|{fs}", f"{route}|{b}|{fs}"
                d = est[ka] - est[kb]
                ex.append({"route": route, "rate_hz": fs, "contrast": name, "diff": res[ka]["auc"] - res[kb]["auc"], "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))})
    pd.DataFrame(ex).to_csv(C.HERE / "paired_contrasts_observed.csv", index=False)
    # check against the earlier stored table (same resamples, same seed): sb and ivt, anch and clean, delta vs 30 Hz and dwell-minus-ivt
    old = pd.read_csv(C.LOWRATE / "rows_webcam.csv")
    chk = []
    for _, o in old.iterrows():
        route = "sb" if o.model.startswith("sb") else "ivt"
        cond = "clean" if o.condition.startswith("clean") else "anch"
        r = df[(df.route == route) & (df.condition_key == cond) & (df.rate_hz == o.rate_hz)].iloc[0]
        chk.append({"route": route, "cond": cond, "rate": int(o.rate_hz), "d_auc": abs(r.auc - o.auc), "d_delta": abs(r.delta_vs_30hz - o.delta_vs_30hz),
                    "d_delta_lo": abs(r.delta_lo - o.delta_lo), "d_delta_hi": abs(r.delta_hi - o.delta_hi),
                    "d_auc_lo": abs(r.auc_lo - o.auc_lo), "d_auc_hi": abs(r.auc_hi - o.auc_hi)})
        if route == "sb":
            chk[-1]["d_sb_minus_ivt_lo"] = abs(r.dwell_minus_ivt_lo - o.sb_minus_ivt_lo)
    chk = pd.DataFrame(chk)
    mx = {c: float(chk[c].max()) for c in chk.columns if c.startswith("d_")}
    C.jdump(C.HERE / "repro_stats_vs_stored.json", {"max_abs_differences": mx, "n_rows": int(len(chk)), "passed": bool(max(mx.values()) < 1e-6)})
    C.log("stats vs stored:", mx)
    print(df[["route", "condition_key", "rate_hz", "auc", "delta_vs_30hz", "delta_lo", "delta_hi"]].round(4).to_string())


if __name__ == "__main__":
    main()
