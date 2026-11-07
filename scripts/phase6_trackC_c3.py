"""Phase 6, Track C, analysis C3: data-quality table per dataset (feeds the Dunn et al. 2023 reporting checklist).
Pre-registration: results/phase6/trackC/PREREG.md (C3). Values come from the project data; device attributes that the
data cannot give (head support, device model for some sets) are taken from the source papers or the repository notes and
are marked in the `source` column. "not available" where neither exists. Nothing is used to select or drop data.

Usage (project root):  .venv/Scripts/python.exe -I scripts/phase6_trackC_c3.py
Needs the C2 scratch cache (`scripts/phase6_trackC_c2.py --stages extract`) for Delmas sample-level missing data.
"""
import json
import os
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from edd import celniak as cel  # noqa: E402
from edd import chen  # noqa: E402
from edd import delmas_raw as dr  # noqa: E402
from edd.delmas import load_trials  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "phase6" / "trackC"
SCRATCH = Path(os.environ.get("TRACKC_SCRATCH", r"<local path>"))
NA = "not available"
VAL_RE = re.compile(r"!CAL VALIDATION (\S+) (\S+) (\S+) (\S+) ERROR ([\d.]+) avg\. ([\d.]+) max")


def auc_boot(v, y, n_boot=2000, seed=0):
    """AUC of a quality variable for the class label (rank formula) with a bootstrap over participants (one value each)."""
    v, y = np.asarray(v, float), np.asarray(y, int)
    ok = np.isfinite(v)
    v, y = v[ok], y[ok]

    def auc(a, b):
        r = pd.Series(a).rank().to_numpy()
        n1, n0 = int(b.sum()), int(len(b) - b.sum())
        return (r[b == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(v), len(v))
        if len(np.unique(y[i])) < 2:
            continue
        bs.append(auc(v[i], y[i]))
    p = float(mannwhitneyu(v[y == 1], v[y == 0]).pvalue) if (y == 1).any() and (y == 0).any() else float("nan")
    return {"auc": auc(v, y), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "mannwhitney_p": p,
            "median_class1": float(np.median(v[y == 1])), "median_class0": float(np.median(v[y == 0])), "n1": int((y == 1).sum()), "n0": int((y == 0).sum())}


def summ(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


# ----------------------------------------------------------------------------- Delmas
def delmas():
    raw = ROOT / "data" / "delmas" / "raw" / "data_raw"
    book = load_trials(ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx")
    parts = {int(p): int(g) for p, g in book.groupby("participant").group.first().items()}
    val, header = [], None
    for pid in parts:
        for asc in sorted((raw / f"Sj_{pid:02d}").glob(f"I{pid:02d}c2_*.asc")):
            txt = asc.read_text(errors="ignore")
            if header is None:
                header = [l for l in txt.splitlines()[:14] if l.startswith("**")]
            for j, m in enumerate(VAL_RE.finditer(txt)):
                val.append((pid, asc.name, j, float(m.group(5)), float(m.group(6)), m.group(4)))
    v_all = pd.DataFrame(val, columns=["participant", "file", "order", "avg_err", "max_err", "quality"])
    n_failed = int((v_all.avg_err >= 100).sum())            # eye not found: EyeLink prints errors of 1e5 deg and above
    v = v_all[v_all.avg_err < 100]
    first = v.sort_values(["participant", "file", "order"]).groupby(["participant", "file"]).head(1)
    per_first = first.groupby("participant").avg_err.mean()
    per = v.groupby("participant").agg(avg_err=("avg_err", "mean"), max_err=("max_err", "max"), n_val=("avg_err", "size"))
    # sample-level missing data in the 5-s display windows (1000 Hz)
    miss = {}
    for pid in parts:
        z = np.load(SCRATCH / "c2_raw" / f"p{pid:02d}.npz")
        miss[pid] = float(np.mean(~np.isfinite(z["x"][:, dr.PRE_MS:dr.PRE_MS + dr.POST_MS])))
    ms = pd.Series(miss)
    grp = pd.Series(parts)
    by_group = {int(g): float(100 * ms[grp == g].mean()) for g in sorted(grp.unique())}
    n_rows = len(book)
    return {"dataset": "Delmas et al. 2024 (famous-face CIT)", "device": "SR Research EyeLink 1000 (EDF header: " + (header[3].strip("* ") if header else "") + "; " + (header[5].strip("* ") if header else "") + ")",
            "device_source": "EDF/ASC header in the data; model name from the paper",
            "sampling_rate_hz": 1000, "rate_source": "EDF sample timestamps (1 ms)", "eyes": "monocular, right eye (validation lines: RIGHT)",
            "head_support": "chin and forehead rest, 57 cm (paper)",
            "pct_samples_missing": 100 * float(ms.mean()), "pct_samples_missing_definition": "gaze or pupil not valid in the 5-s display windows, 1000 Hz, mean over 75 participants",
            "pct_samples_missing_range_participants": [100 * float(ms.min()), 100 * float(ms.max())], "pct_samples_missing_by_group": by_group,
            "pct_trials_excluded": f"{100 * (1 - n_rows / (75 * 64)):.1f}% of image rows (4,525 of 4,800 participant x image rows kept in the authors' table: ambiguous or misidentified faces removed by the authors); main-session trials: 64 of 64 kept per participant (0%)",
            "pct_participants_excluded": "0 of 75 in the released tables (participants removed by the authors before release: not available)",
            "calibration_validation_error_deg": {"quantity": "EyeLink HV9 validation, average error per validation, mean over the validations of the main session (per participant)",
                                                 "mean_of_participant_means": float(per.avg_err.mean()), "median": float(per.avg_err.median()),
                                                 "range_participant_means": [float(per.avg_err.min()), float(per.avg_err.max())],
                                                 "worst_max_error_any_validation": float(v.max_err.max()), "n_validations": int(len(v)), "n_failed_validations_excluded_eye_not_found": n_failed,
                                                 "first_validation_per_session_file": {"mean_of_participant_means": float(per_first.mean()), "median": float(per_first.median()), "range": [float(per_first.min()), float(per_first.max())], "n_participants": int(len(per_first))},
                                                 "n_participants_with_validation": int(per.shape[0]), "pct_participants_mean_error_above_1deg": float(100 * (per.avg_err > 1).mean()),
                                                 "quality_labels": v.quality.value_counts().to_dict()},
            "class_difference": {"by_group_missing_pct": by_group, "note": "classes are known versus unknown faces within each participant; data loss compared across the five groups"}}, per, ms


# ----------------------------------------------------------------------------- Celniak
def celniak():
    cache = ROOT / "data" / "celniak_cache"
    rows = []
    for js in sorted(cache.glob("Recording*.json")):
        name = js.stem
        s, meta = cel.load_cached(cache, name)
        ts = s["ts_us"].to_numpy(float)
        smp = s[s.valid_l.notna() | s.valid_r.notna()]
        dt = np.median(np.diff(smp.ts_us.to_numpy(float)))
        slide = smp[smp.stimulus.astype(str).str.match(cel.SLIDE_RE)]
        both_inv = ((smp.valid_l != "Valid") & (smp.valid_r != "Valid")).mean()
        any_inv = ((smp.valid_l != "Valid") | (smp.valid_r != "Valid")).mean()
        sl_both = ((slide.valid_l != "Valid") & (slide.valid_r != "Valid")).mean()
        sl_any = ((slide.valid_l != "Valid") | (slide.valid_r != "Valid")).mean()
        # slides = trials: valid fraction per slide (both eyes valid)
        vf = slide.groupby("stimulus").apply(lambda d: ((d.valid_l == "Valid") & (d.valid_r == "Valid")).mean())
        rows.append({"recording": name, "label": int(str(meta["group"]).strip().lower() == "guilty"), "rate_hz": 1e6 / dt,
                     "pct_both_eyes_invalid_all": 100 * both_inv, "pct_either_eye_invalid_all": 100 * any_inv,
                     "pct_both_eyes_invalid_slides": 100 * sl_both, "pct_either_eye_invalid_slides": 100 * sl_any,
                     "n_slides": int(len(vf)), "n_slides_valid_below_50": int((vf < 0.5).sum()),
                     "calibration_accuracy_deg": meta.get("Average calibration accuracy (degrees)", np.nan),
                     "validation_accuracy_deg": meta.get("Average validation accuracy (degrees)", np.nan),
                     "validation_precision_rms_deg": meta.get("Average validation precision RMS (degrees)", np.nan)})
    d = pd.DataFrame(rows)
    cls = {}
    for c in ("pct_either_eye_invalid_slides", "pct_both_eyes_invalid_slides", "validation_accuracy_deg", "calibration_accuracy_deg", "validation_precision_rms_deg"):
        cls[c] = auc_boot(d[c], d.label)
    rate = d.rate_hz.round(-1)
    out = {"dataset": "Celniak et al. 2026 (mock-crime CIT)", "device": "Tobii Pro Fusion (remote), Tobii Pro Lab export (events from the vendor classifier)",
           "device_source": "dataset loader docstring and Zenodo description in the repository; model not stored in the sample files",
           "sampling_rate_hz": {"measured_median_by_recording": d.rate_hz.round(1).value_counts().to_dict(), "median_of_recordings": float(d.rate_hz.median())},
           "rate_source": "sample timestamps (median inter-sample interval) of each recording", "eyes": "both eyes recorded; validity per eye in the export; pupil = Tobii filtered diameter (mm), fallback mean of the two eyes",
           "head_support": "remote tracker, head free (dataset description)",
           "pct_samples_missing": float(d.pct_either_eye_invalid_slides.mean()),
           "pct_samples_missing_definition": "samples inside the 7.5-s question slides where at least one eye is not Valid (mean over recordings); both eyes invalid: %.2f%%" % d.pct_both_eyes_invalid_slides.mean(),
           "pct_samples_missing_range_participants": [float(d.pct_either_eye_invalid_slides.min()), float(d.pct_either_eye_invalid_slides.max())],
           "pct_trials_excluded": "0%% by us (21 slides per recording kept); recordings with a slide valid below 50%%: %d of %d recordings, %d of %d slides" % ((d.n_slides_valid_below_50 > 0).sum(), len(d), d.n_slides_valid_below_50.sum(), d.n_slides.sum()),
           "pct_participants_excluded": "0 of %d recordings in the deposit (exclusions made by the authors before deposit: not available)" % len(d),
           "calibration_validation_error_deg": {"quantity": "Tobii Pro Lab 'Average validation accuracy (degrees)' (the variable used in the confound analysis, as log), 'Average calibration accuracy' and 'Average validation precision RMS'",
                                                "validation_accuracy": summ(d.validation_accuracy_deg), "calibration_accuracy": summ(d.calibration_accuracy_deg),
                                                "validation_precision_rms": summ(d.validation_precision_rms_deg)},
           "class_difference": {k: v for k, v in cls.items()}, "n_recordings": int(len(d)), "label_definition": "class 1 = guilty"}
    return out, d


# ----------------------------------------------------------------------------- Chen
def _chen_val(f):
    rec = chen.read_edf_fast(f, phase="__none__", max_samples=10)
    out = []
    for t, m in rec["messages"]:
        mt = VAL_RE.search(m)
        if mt:
            out.append((float(mt.group(5)), float(mt.group(6))))
    return out


def chen_ds():
    import importlib.util
    spec = importlib.util.spec_from_file_location("phase2_chen_mod", ROOT / "scripts" / "phase2_chen.py")
    pc = importlib.util.module_from_spec(spec)
    sys.argv = ["x"]
    spec.loader.exec_module(pc)
    res = {}
    rows = []
    for exp in ("exp1", "exp2"):
        meta_all = pd.read_csv(ROOT / "data" / "derived" / f"chen_{exp}_meta.csv")
        z = np.load(ROOT / "data" / "derived" / f"chen_{exp}_traces.npz")
        n_files = meta_all.subject_nr.nunique()
        excl = sorted(pc.EXCLUDE[exp])
        meta, arr = pc.load(exp)
        m2, ch, info = pc.preprocess(meta, arr)
        # sample-level missing: fraction of missing 1 kHz pupil samples (blink events included) per 10-ms bin, RSVP phase
        miss_all = np.nanmean(z["miss"], axis=1)
        mm = pd.DataFrame({"subject_nr": meta_all.subject_nr, "trialtype": meta_all.trialtype, "T1": meta_all.T1, "miss": miss_all})
        kept = mm[(mm.trialtype == 1) & ~mm.subject_nr.isin(excl) & mm.T1.isin([1, 2, 3, 4])]
        per_p = kept.groupby("subject_nr").miss.mean()
        # class contrast within participant: familiar (T1 = 2) vs control (T1 = 3, 4)
        fam = kept[kept.T1 == 2].groupby("subject_nr").miss.mean()
        ctl = kept[kept.T1.isin([3, 4])].groupby("subject_nr").miss.mean()
        d = (fam - ctl).dropna()
        rng = np.random.default_rng(0)
        bs = [d.to_numpy()[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        # EDF messages: validation
        files = [f for f in sorted((ROOT / "data" / "chen" / exp / "EDF").glob("*.edf")) if int(re.search(r"(\d+)", f.stem).group(1)) not in excl]
        got = Parallel(n_jobs=6)(delayed(_chen_val)(f) for f in files)
        vals = []
        for f, g in zip(files, got):
            sid = int(re.search(r"(\d+)", f.stem).group(1))
            for (a, b) in g:
                vals.append((sid, a, b, f.name, len(vals)))
        v_all = pd.DataFrame(vals, columns=["participant", "avg", "max", "file", "order"])
        v_all.to_csv(SCRATCH / f"chen_validation_{exp}.csv", index=False)
        n_failed = int((v_all.avg >= 100).sum())
        v = v_all[v_all.avg < 100]
        pv_first = v.sort_values(["participant", "order"]).groupby("participant").head(1).set_index("participant").avg
        pv = v.groupby("participant").avg.mean() if len(v) else pd.Series(dtype=float)
        res[exp] = {"n_edf_files": int(n_files), "n_participants_excluded_by_authors_scripts": excl,
                    "n_participants_used": int(m2.subject_nr.nunique()),
                    "pct_participants_excluded": float(100 * len(excl) / n_files),
                    "n_trials_in_after_participant_filter": info["n_trials_in"], "n_undefined_baseline": info["n_undefined_baseline"],
                    "n_baseline_outliers": info["n_baseline_outliers"], "n_trials_kept": info["n_kept"],
                    "pct_trials_excluded_baseline_rules": float(100 * (info["n_trials_in"] - info["n_kept"]) / info["n_trials_in"]),
                    "pct_samples_missing_pupil_rsvp_phase": float(100 * per_p.mean()),
                    "pct_samples_missing_range_participants": [float(100 * per_p.min()), float(100 * per_p.max())],
                    "missing_diff_familiar_minus_control_pct_points": {"mean": float(100 * d.mean()), "ci95": [float(100 * np.percentile(bs, 2.5)), float(100 * np.percentile(bs, 97.5))],
                                                                       "n_participants": int(len(d))},
                    "validation": {"n_validations": int(len(v)), "n_failed_validations_excluded_eye_not_found": n_failed, "n_participants_with_validation": int(len(pv)),
                                   "first_validation_per_participant": {"mean": float(pv_first.mean()), "median": float(pv_first.median()), "range": [float(pv_first.min()), float(pv_first.max())]} if len(pv_first) else NA,
                                   "mean_of_participant_mean_avg_error_deg": float(pv.mean()) if len(pv) else NA,
                                   "median": float(pv.median()) if len(pv) else NA,
                                   "range": [float(pv.min()), float(pv.max())] if len(pv) else NA,
                                   "worst_max_error": float(v["max"].max()) if len(v) else NA}}
    return res


# ----------------------------------------------------------------------------- Preti, Colasanti
def preti():
    x = pd.read_excel(ROOT / "data" / "preti" / "preti2026.xlsx", sheet_name=None, engine="calamine")
    raw, err, pup = x["raw data"], x["errors"], x["pupil_corrected"]
    n_sub = raw.SUBJECT_LABEL.nunique()
    per = raw.groupby("SUBJECT_LABEL").agg(blink=("BLINK_0_1", "mean"), cond=("CONDITION", "first"))
    cls = auc_boot(per.blink.to_numpy(), (per.cond == 0).astype(int).to_numpy())     # class 1 = liar (CONDITION 0)
    return {"dataset": "Preti et al. 2026 (fake-identity lying)", "device": "eye tracker (model not reported in the deposit)", "device_source": NA,
            "sampling_rate_hz": NA, "rate_source": "deposit holds fixation-level rows (about %.1f per trial), no samples" % (len(raw) / (n_sub * 32)),
            "eyes": NA, "head_support": NA,
            "pct_samples_missing": NA, "pct_samples_missing_definition": "no sample-level data; proxy: share of fixation rows flagged as blink = %.1f%%; fixation rows with missing pupil = %.1f%%" % (100 * raw.BLINK_0_1.mean(), 100 * raw.CURRENT_FIX_PUP.isna().mean()),
            "pct_trials_excluded": "error trials (outcome 1) are %.1f%% of %d trials; pupil sheet holds %d of %d trials (%.2f%% without a corrected pupil value); none removed by us" % (100 * (err.outcome == 1).mean(), len(err), len(pup), len(err), 100 * (1 - len(pup) / len(err))),
            "pct_participants_excluded": "v1 deposit holds 60 participants, the analysed v2 deposit %d (3 removed by the authors, all liars)" % n_sub,
            "calibration_validation_error_deg": NA,
            "class_difference": {"blink_flagged_fixation_share_liar_vs_truth": cls, "note": "class 1 = liar; blink share per participant"}}


def colasanti():
    d = pd.read_csv(ROOT / "data" / "mazza" / "data.csv")
    return {"dataset": "Colasanti et al. 2024 (RealEye webcam crime scene; Mazza data)", "device": "RealEye webcam eye tracking (HD webcam 1080p at 60 fps, Chrome)", "device_source": "Colasanti et al. 2023 (repository copy of the paper)",
            "sampling_rate_hz": "60 nominal (paper: a gaze sample at least every 16 ms); effective rate not recorded", "rate_source": "paper text; deposit has per-participant aggregates only",
            "eyes": NA, "head_support": "none (online, free head)",
            "pct_samples_missing": NA, "pct_samples_missing_definition": "aggregate fixation tables only, no sample-level data",
            "pct_trials_excluded": NA, "pct_participants_excluded": f"3 of 160 (1.8%) excluded by the authors for technical problems; {len(d)} analysed in the deposit",
            "calibration_validation_error_deg": {"value": NA, "note": "RealEye platform accuracy 90 to 156 px, 113 px on average (Lewandowska 2020, quoted in the paper); no per-participant calibration value in the deposit"},
            "class_difference": NA}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = {}
    R["delmas"], per_d, ms = delmas()
    print("delmas done", flush=True)
    R["celniak"], dc = celniak()
    print("celniak done", flush=True)
    R["chen"] = chen_ds()
    print("chen done", flush=True)
    R["preti"] = preti()
    R["colasanti"] = colasanti()
    (OUT / "results_c3.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    dc.to_csv(OUT / "c3_celniak_per_recording.csv", index=False)
    pd.DataFrame({"participant": per_d.index, "mean_validation_avg_error_deg": per_d.avg_err.values, "max_error_deg": per_d.max_err.values,
                  "n_validations": per_d.n_val.values, "pct_missing_display_windows": [100 * ms[p] for p in per_d.index]}).to_csv(OUT / "c3_delmas_per_participant.csv", index=False)
    print(json.dumps(R, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
