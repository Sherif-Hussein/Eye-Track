"""Phase 2 (celniak track): descriptive checks behind the mechanism section of comparison_card.md.

Full-sample and unsupervised-by-design: these numbers explain the cross-validated results and are not
performance estimates. AUCs are oriented so that values above 0.5 mean higher in the guilty group.

Usage: python -I scripts/phase2_celniak_mechanism.py <derived_dir> <cache_dir> <out.json>
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score as auc

KEYS = ["recording", "question", "repetition"]


def block(g, col):
    et, pg = g.order_et == 1, g.order_et == 0
    ok = g[col].notna()
    return {"auc_guilty_higher": round(auc(g.y[ok], g[col][ok]), 3),
            "auc_within_ET_first": round(auc(g.y[et & ok], g[col][et & ok]), 3),
            "auc_within_PG_first": round(auc(g.y[pg & ok], g[col][pg & ok]), 3),
            "median_innocent": round(float(g[col][g.y == 0].median()), 2),
            "median_guilty": round(float(g[col][g.y == 1].median()), 2)}


def main(der, cache, out):
    it = pd.read_csv(Path(der) / "celniak_p2_items.csv")
    tr = pd.read_csv(Path(der) / "celniak_p2_trials.csv")
    n = it.groupby(KEYS).agg(a_d=("dwell", "sum"), a_n=("nfix", "sum")).reset_index().merge(tr, on=KEYS)
    n["all_d"] = n.sl_fixdur * n.sl_nfix
    g = n.groupby("recording").agg(y=("label", "first"), exam=("first_exam", "first"), all_d=("all_d", "sum"),
                                   all_n=("sl_nfix", "sum"), a_d=("a_d", "sum"), a_n=("a_n", "sum"),
                                   fixdur_all=("sl_fixdur", "mean"), fixations_per_slide=("sl_nfix", "mean"),
                                   valid_frac=("valid_frac", "mean"), pupil_valid=("pupil_valid_frac", "mean"),
                                   fixdur_video=("vid_fixdur", "mean"), fixdur_cross=("cross_fixdur", "mean"))
    g["order_et"] = (g.exam == "Eyetracker").astype(int)
    g["fixdur_on_items"] = g.a_d / g.a_n
    g["fixdur_off_items"] = (g.all_d - g.a_d) / (g.all_n - g.a_n)
    g["off_item_time_share"] = (g.all_d - g.a_d) / g.all_d
    g["off_item_fixations_per_slide"] = (g.all_n - g.a_n) / 21
    g["on_item_fixations_per_slide"] = g.a_n / 21

    def on(d):
        h = d.groupby("recording")
        return h.dwell.sum() / h.nfix.sum()
    g["fixdur_on_probe"] = on(it[it.is_probe == 1])
    g["fixdur_on_irrelevant"] = on(it[it.is_probe == 0])
    meta = {}
    for f in glob.glob(str(Path(cache) / "*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        meta[d["Recording name"]] = d
    g["val_acc_deg"] = [meta[r].get("Average validation accuracy (degrees)") for r in g.index]
    g["val_rms_deg"] = [meta[r].get("Average validation precision RMS (degrees)") for r in g.index]
    g["cal_acc_deg"] = [meta[r].get("Average calibration accuracy (degrees)") for r in g.index]
    g[["val_acc_deg", "val_rms_deg", "cal_acc_deg"]] = g[["val_acc_deg", "val_rms_deg", "cal_acc_deg"]].astype(float)
    cols = ["fixdur_all", "fixdur_on_items", "fixdur_off_items", "fixdur_on_probe", "fixdur_on_irrelevant",
            "off_item_time_share", "off_item_fixations_per_slide", "on_item_fixations_per_slide",
            "fixations_per_slide", "fixdur_video", "fixdur_cross", "val_acc_deg", "val_rms_deg", "cal_acc_deg",
            "valid_frac", "pupil_valid"]
    res = {"univariate": {c: block(g, c) for c in cols},
           "fixdur_all_by_repetition_auc_guilty_lower": {
               str(r): round(1 - auc(g.y, tr[tr.repetition == r].groupby("recording").sl_fixdur.mean().loc[g.index]), 3)
               for r in (1, 2, 3)},
           "spearman_fixdur_all": {c: round(float(g.fixdur_all.corr(g[c], method="spearman")), 3)
                                   for c in ("val_acc_deg", "val_rms_deg", "valid_frac")},
           "spearman_fixdur_all_within_label": {
               str(y): {c: round(float(g[g.y == y].fixdur_all.corr(g[g.y == y][c], method="spearman")), 3)
                        for c in ("val_acc_deg", "val_rms_deg", "valid_frac")} for y in (0, 1)},
           "sampling_rate_counts": {str(k): int(v) for k, v in tr.groupby("recording").fs_hz.first().value_counts().items()},
           "lowest_fixdur_recordings": g.sort_values("fixdur_all")[["y", "order_et", "fixdur_all", "pupil_valid", "valid_frac"]]
           .head(6).round(3).reset_index().to_dict("records")}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
