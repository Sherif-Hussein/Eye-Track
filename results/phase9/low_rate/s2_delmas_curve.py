"""Task 2.2 (EXPLORATORY, not preregistered), step 2: Delmas sampling-rate curve extended to 20, 15, 10 Hz.

Source of the curve: results/phase2/delmas_raw (scripts/phase2_delmas_raw.py, stage `rate`: P4_svm_<rate> = the published four
features with the RBF-SVM; ALLz_lr_<rate> = our 13 standardised features without microsaccades, L2 logistic regression), drawn by
scripts/phase5_figures.py -> paper/figures/data_delmas_rate_curve.csv. This script loads the track script UNCHANGED
(edd.delmas_raw_matched.load_track) and calls its build_tables / run_grid on the frozen folds
(results/phase2/delmas_raw/folds.json), so tuning grids, seeds and nested CV are the originals. Only difference: run single-process
(N_JOBS = 1; results do not depend on it).
Stages: build -> repro (stops on mismatch) -> run -> stats.   Usage: python -I s2_delmas_curve.py [build,repro,run,stats]
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lr_common as L  # noqa: E402  (sets thread env vars first)

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from edd import delmas_raw as dr  # noqa: E402
from edd import delmas_raw_matched as M  # noqa: E402

OUT = L.OUT
DERIVED = OUT / "derived"
CK = OUT / "checkpoints_delmas"
BOOK64 = L.ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx"
BOOK32 = L.ROOT / "data" / "delmas" / "BookFIP_32_trials.xlsx"
CURVE = L.ROOT / "results" / "old_paper_data" / "data_delmas_rate_curve.csv"
MODELS = {"P4_svm": ("svm", None), "ALLz_lr": ("lr", None)}


def build():
    DERIVED.mkdir(exist_ok=True)
    rows = []
    for f in sorted((OUT / "cache").glob("p??.npz")):
        rows.append(np.load(f)["face"])
    face = pd.DataFrame(np.vstack(rows), columns=["participant", "group", "rate", "trial", "quad", "image", "fam_trial", "el_nfix"] + dr.TRIAL_FEATURES)
    for c in ["participant", "group", "rate", "trial", "quad", "image", "fam_trial"]:
        face[c] = face[c].astype(int)
    face.to_csv(DERIVED / "delmas_raw_face_trials.csv.gz", index=False)       # same file name/columns as data/derived/
    L.log("face table", face.shape, "participants", face.participant.nunique(), "rates", sorted(face.rate.unique()))
    # feature identity at 30 Hz against the stored table
    st = pd.read_csv(L.ROOT / "data" / "derived" / "delmas_raw_face_trials.csv.gz")
    st = st[st.rate == 30].sort_values(["participant", "trial", "quad"]).reset_index(drop=True)
    mine = face[face.rate == 30].sort_values(["participant", "trial", "quad"]).reset_index(drop=True)
    a, b = st[dr.TRIAL_FEATURES].to_numpy(float), mine[dr.TRIAL_FEATURES].to_numpy(float)
    key = ["participant", "trial", "quad", "image"]
    res = {"rows": int(len(a)), "same_keys": bool((st[key].to_numpy() == mine[key].to_numpy()).all()),
           "nan_pattern_identical": bool((np.isnan(a) == np.isnan(b)).all()),
           "max_abs_diff_trial_features_30Hz": float(np.nanmax(np.abs(a - b)))}
    L.jdump(OUT / "repro_features_30hz.json", res)
    L.log("30 Hz feature identity vs stored table:", res)
    assert res["same_keys"] and res["nan_pattern_identical"] and res["max_abs_diff_trial_features_30Hz"] < 1e-9
    return face


def get_track():
    CK.mkdir(exist_ok=True)
    track = M.load_track()
    track.CKPT = CK
    track.N_JOBS = 1
    df = track.build_tables(DERIVED, BOOK64, BOOK32)
    frozen = json.loads((L.STORED / "folds.json").read_text())["repeats"]
    return track, df, frozen


def feats_for(track, tag):
    if tag == "P4_svm":
        return track.P4, "svm"
    return [f for f in track.FEATURE_SETS["ALL_z"] if f != "msacc_rate_z"], "lr"


def run_rate(track, df, frozen, tag, rate):
    feats, model = feats_for(track, tag)
    name = f"{tag}_{rate}"
    L.log("run", name)
    return track.run_grid(df, frozen, feats, model, rate, name, {}, {})


def repro(track, df, frozen):
    out = {}
    curve = pd.read_csv(CURVE)
    ok = True
    for tag in MODELS:
        run_rate(track, df, frozen, tag, 30)
        name = f"{tag}_30"
        mine = pd.read_csv(CK / f"{name}_oof.csv.gz").sort_values(["set", "repeat", "participant", "image"]).reset_index(drop=True)
        ref = pd.read_csv(L.STORED / "checkpoints" / f"{name}_oof.csv.gz").sort_values(["set", "repeat", "participant", "image"]).reset_index(drop=True)
        k = ["set", "repeat", "participant", "image", "y"]
        same = len(mine) == len(ref) and (mine[k].to_numpy() == ref[k].to_numpy()).all()
        md = float(np.abs(mine.score.to_numpy() - ref.score.to_numpy()).max()) if same else None
        auc_m, auc_r = L.pooled_auc(CK / f"{name}_oof.csv.gz"), float(curve[(curve.model == tag) & (curve.rate == 30)].auc.iloc[0])
        j_m = json.loads((CK / f"{name}.json").read_text())["res"]["mean_over_sets"]["image"]["auc"]
        j_r = json.loads((L.STORED / "checkpoints" / f"{name}.json").read_text())["res"]["mean_over_sets"]["image"]["auc"]
        out[tag] = {"same_rows": bool(same), "max_abs_oof_score_diff": md, "repeat_averaged_pooled_auc_new": auc_m,
                    "repeat_averaged_pooled_auc_stored_curve": auc_r, "mean_of_fold_auc_new": j_m, "mean_of_fold_auc_stored": j_r}
        ok = ok and same and md is not None and md < 1e-9 and abs(auc_m - auc_r) < 1e-9
        L.log("repro", tag, out[tag])
    out["all_passed"] = bool(ok)
    L.jdump(OUT / "repro_delmas.json", out)
    if not ok:
        L.log("REPRODUCTION FAILED - stopping")
        sys.exit(2)


def stats():
    curve = {}
    rows = []
    for tag in MODELS:
        cfg = {}
        for r in (1000, 60):
            cfg[r] = [L.STORED / "checkpoints" / f"{tag}_{r}_oof.csv.gz"]
        for r in L.RATES_NEW:
            cfg[r] = [CK / f"{tag}_{r}_oof.csv.gz"]
        order = [1000, 60, 30, 20, 15, 10]
        res = L.paired_bootstrap({str(r): cfg[r] for r in order}, ref="30", n_boot=2000, seed=20261102)
        curve[tag] = res
        for r in order:
            v = res[str(r)]
            rows.append({"dataset": "Delmas", "model": tag, "condition": "clean EyeLink (anti-aliased resampling)", "rate_hz": r,
                         "auc": v["auc"], "auc_lo": v["auc_lo"], "auc_hi": v["auc_hi"], "delta_vs_30hz": v["delta_vs_ref"],
                         "delta_lo": v["delta_lo"], "delta_hi": v["delta_hi"],
                         "source": "stored (phase2)" if r in (1000, 60) else ("reproduced (identical to stored)" if r == 30 else "new")})
        L.log(tag, {r: round(res[str(r)]["auc"], 4) for r in order})
    L.jdump(OUT / "results_delmas.json", {"note": "AUC CI from first 300 resamples (as original curve); delta CI from 2000 paired resamples", "curve": curve})
    pd.DataFrame(rows).to_csv(OUT / "rows_delmas.csv", index=False)


def main():
    st = (sys.argv[1] if len(sys.argv) > 1 else "build,repro,run,stats").split(",")
    if "build" in st:
        build()
    if {"repro", "run"} & set(st):
        track, df, frozen = get_track()
        if "repro" in st:
            repro(track, df, frozen)
        if "run" in st:
            for rate in (20, 15, 10):
                for tag in MODELS:
                    run_rate(track, df, frozen, tag, rate)
    if "stats" in st:
        stats()


if __name__ == "__main__":
    main()
