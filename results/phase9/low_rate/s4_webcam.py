"""Task 2.2 (EXPLORATORY, not preregistered), step 4: Delmas, measured-webcam-noise condition at 30 / 20 / 15 / 10 Hz,
dwell-from-samples ("sb") versus fixed-threshold event detection (I-VT 30 deg/s, 60 ms), both WITHOUT pupil features.

Sources, imported UNCHANGED: scripts/phase7_trackD_run.py (loaded through scripts/phase6_trackC_c1.py: noise model `noisy` incl. the
"anch" condition = measured webcam values: 1.45 deg mean offset, jitter SD 1.66/sqrt2 and 1.42/sqrt2 deg per axis in x/y, 3% sample
loss; `sb_features_batch`; `make_table`; feature lists FEATS[("ivt"|"sb", "np")]), edd.delmas_fast.trial_features_batch (I-VT),
edd.delmas_raw_matched.load_track (track.run_grid: nested LR on the frozen folds). Noise draws 1, 2, 3 (seeds SeedSequence
[W.SEED, draw, participant]); the same participant offset is used at every rate, jitter and loss are drawn per sample at that rate.
Module attribute C1.FS is set to the rate before building features (the originals hard-code 30 Hz; nothing on disk is edited).
Clean (noise-free) is run alongside as the reference. AUC of a condition = mean over the 3 draws; CI = participant-cluster bootstrap
(2000 paired resamples, mean over draws inside each resample, as in scripts/phase6_trackC_c1.py stage_stats).
Stages: run (30 Hz first) -> repro (compares with results/phase7/trackD/d1_table.csv, stops on mismatch) -> stats.
Usage: python -I s4_webcam.py [run30,repro,run,stats]
"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lr_common as L  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

spec = importlib.util.spec_from_file_location("trackC_c1", L.ROOT / "scripts" / "phase6_trackC_c1.py")
C1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C1)
dr, df_, M = C1.dr, C1.df_, C1.M

OUT = L.OUT
CK = OUT / "checkpoints_webcam"
PUPIL = ("pupil_face_z", "pupil_trial_z")
FEATS = {"ivt": [f for f in C1.FEATS["ivt"] if f not in PUPIL], "sb": [f for f in C1.FEATS["sb"] if f not in PUPIL]}
assert len(FEATS["ivt"]) == 11 and len(FEATS["sb"]) == 10
UNITS = [("clean", 0.0, 0)] + [("anch", "A", d) for d in (1, 2, 3)]
RATES = [30, 15, 10, 20]
SHOW = [30, 20, 15, 10]
EXPECTED30 = {("ivt", "clean"): 0.7862610230322313, ("sb", "clean"): 0.7731838961824132,
              ("ivt", "anch"): 0.7343513124910731, ("sb", "anch"): 0.7737061625087726}      # results/phase7/trackD/d1_table.csv


def tag(route, cond, fs, d):
    return f"{route}np_{cond}_{fs}Hz_d{d}"


def load_rate(fs):
    segs = {}
    for f in sorted((OUT / "cache").glob("p??.npz")):
        z = np.load(f)
        segs[int(f.stem[1:])] = {"x": z[f"x{fs}"], "y": z[f"y{fs}"], "p": z[f"p{fs}"], "images": z["images"], "group": int(z["group"])}
    return segs


def get_track():
    CK.mkdir(exist_ok=True)
    track = M.load_track()
    track.CKPT = CK
    track.N_JOBS = 1
    return track


def run_rate(track, frozen, bk, fs):
    C1.FS = fs
    segs = load_rate(fs)
    for cond, Lv, d in UNITS:
        need = [r for r in ("ivt", "sb") if not (CK / f"{tag(r, cond, fs, d)}.json").exists()]
        if not need:
            continue
        ivt_arr, sb_arr = {}, {}
        for pid, s in segs.items():
            x, y, p = C1.noisy(s["x"], s["y"], s["p"], cond, Lv, d, pid)
            if "ivt" in need:
                ivt_arr[pid] = (df_.trial_features_batch(x, y, p, fs, with_msacc=False), s["images"], s["group"])
            if "sb" in need:
                sb_arr[pid] = (C1.sb_features_batch(x, y, p), s["images"], s["group"])
        tabs = {}
        if ivt_arr:
            tabs["ivt"] = C1.make_table(ivt_arr, dr.TRIAL_FEATURES, bk)
        if sb_arr:
            tabs["sb"] = C1.make_table(sb_arr, C1.SB_NAMES, bk)
        for r in need:
            t = tag(r, cond, fs, d)
            res = track.run_grid(tabs[r], frozen, FEATS[r], "lr", fs, t, {}, {})
            L.log(f"{t}: pooled AUC {L.pooled_auc(CK / (t + '_oof.csv.gz')):.4f} (track mean-of-fold {res['mean_over_sets']['image']['auc']:.4f})")


def aucs(route, cond, fs):
    ds = [d for c, _, d in UNITS if c == cond]
    return float(np.mean([L.pooled_auc(CK / f"{tag(route, cond, fs, d)}_oof.csv.gz") for d in ds]))


def repro():
    out = {}
    # decimated 30 Hz samples vs the cache used by the earlier webcam studies
    mx = 0.0
    same_nan = True
    for pid in (2, 3, 10, 40):
        z = np.load(L.ROOT / "results" / "phase3" / "webcam_noise" / "cache_30hz" / f"p{pid:02d}.npz")
        if not (L.OUT / "cache" / f"p{pid:02d}.npz").exists():
            continue
        m = np.load(L.OUT / "cache" / f"p{pid:02d}.npz")
        for a, b in ((z["x30"], m["x30"]), (z["y30"], m["y30"]), (z["p30"], m["p30"])):
            same_nan &= bool((np.isnan(a) == np.isnan(b)).all())
            mx = max(mx, float(np.nanmax(np.abs(a - b))))
    out["cache30_vs_earlier_cache_max_abs_diff"], out["cache30_nan_pattern_identical"] = mx, same_nan
    ok = same_nan and mx < 1e-9
    for (route, cond), exp in EXPECTED30.items():
        a = aucs(route, cond, 30)
        out[f"{route}_{cond}_30Hz"] = {"auc_new": a, "auc_stored_trackD": exp, "abs_diff": abs(a - exp)}
        ok = ok and abs(a - exp) < 5e-5
    out["all_passed"] = bool(ok)
    L.jdump(OUT / "repro_webcam.json", out)
    L.log("repro", out)
    if not ok:
        L.log("REPRODUCTION FAILED - stopping")
        sys.exit(2)


def stats():
    rows, res_all = [], {}
    names_route = {"sb": "sb: dwell from samples, no event detection (no pupil)", "ivt": "I-VT event detection 30 deg/s, 60 ms (no pupil)"}
    cond_label = {"clean": "clean, no added noise", "anch": "measured webcam values (1.45 deg offset, ~1.1 deg jitter, 3% loss), mean of 3 draws"}
    for cond in ("clean", "anch"):
        cfg = {}
        for route in ("ivt", "sb"):
            for fs in RATES:
                cfg[f"{route}@{fs}"] = [CK / f"{tag(route, cond, fs, d)}_oof.csv.gz" for c, _, d in UNITS if c == cond]
        res, est = L.paired_bootstrap(cfg, ref="ivt@30", n_boot=2000, seed=20261102, return_draws=True)
        res_all[cond] = {}
        for route in ("ivt", "sb"):
            for fs in SHOW:
                k = f"{route}@{fs}"
                d = est[k] - est[f"{route}@30"]
                v = res[k]
                dd = {"auc": v["auc"], "auc_lo": v["auc_lo"], "auc_hi": v["auc_hi"], "per_draw_auc": v["per_draw_auc"],
                      "delta_vs_30hz": v["auc"] - res[f"{route}@30"]["auc"], "delta_lo": float(np.percentile(d, 2.5)), "delta_hi": float(np.percentile(d, 97.5))}
                if route == "sb":
                    c = est[f"sb@{fs}"] - est[f"ivt@{fs}"]
                    dd.update({"sb_minus_ivt_same_rate": v["auc"] - res[f"ivt@{fs}"]["auc"], "sb_minus_ivt_lo": float(np.percentile(c, 2.5)),
                               "sb_minus_ivt_hi": float(np.percentile(c, 97.5))})
                res_all[cond][k] = dd
                rows.append({"dataset": "Delmas", "model": names_route[route], "condition": cond_label[cond], "rate_hz": fs, "auc": dd["auc"],
                             "auc_lo": dd["auc_lo"], "auc_hi": dd["auc_hi"], "delta_vs_30hz": dd["delta_vs_30hz"], "delta_lo": dd["delta_lo"],
                             "delta_hi": dd["delta_hi"], "sb_minus_ivt": dd.get("sb_minus_ivt_same_rate"), "sb_minus_ivt_lo": dd.get("sb_minus_ivt_lo"),
                             "sb_minus_ivt_hi": dd.get("sb_minus_ivt_hi"),
                             "source": "reproduced (matches phase7 trackD)" if fs == 30 else "new"})
        L.log(cond, {k: round(v["auc"], 4) for k, v in res_all[cond].items()})
    L.jdump(OUT / "results_webcam.json", {"note": "AUC CI from first 300 of 2000 paired resamples; delta CIs from all 2000; mean over 3 noise draws", "conditions": res_all})
    pd.DataFrame(rows).to_csv(OUT / "rows_webcam.csv", index=False)


def main():
    st = (sys.argv[1] if len(sys.argv) > 1 else "run30,repro,run,stats").split(",")
    track = get_track()
    frozen = json.loads((L.STORED / "folds.json").read_text())["repeats"]
    bk = C1.books()
    if "run30" in st:
        run_rate(track, frozen, bk, 30)
    if "repro" in st:
        repro()
    if "run" in st:
        for fs in RATES[1:]:
            run_rate(track, frozen, bk, fs)
    if "stats" in st:
        stats()


if __name__ == "__main__":
    main()
