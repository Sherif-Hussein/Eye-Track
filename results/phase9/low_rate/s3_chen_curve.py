"""Task 2.2 (EXPLORATORY, not preregistered), step 3: Chen et al. sampling-rate test extended to 20, 15, 10 Hz.

Source: scripts/phase4_unified.py stage_chen_rate (results/phase4/unified/results_chen_rate.json, log_chen_rate.txt). Everything is
imported UNCHANGED from src/edd/unified.py (build_chen, decimate_to_rate = anti-aliased polyphase resampling of the released 100 Hz
epochs, run_task = nested participant-grouped CV with L2 logistic regression, run_chen_i2b, oof_rows, auc_items, BOOT_SEED) on the
frozen confirmatory folds (results/phase3/chen_confirm/folds.json). Models: "unified" = our pipeline (25 x 100 ms bins of pupil,
dx, dy, within-participant z), "I2b" = pupil bins + gaze (125 features). Trial-level AUC (repeat-averaged scores).
Two adaptations for rates that do not divide 100 ms into whole samples (the originals handle 100/60/30 Hz only):
  (a) "unified" features at 20/15/10 Hz are bin means over round(k * fs / 10) sample edges (the edges that
      unified.chen_i2b_features already uses; identical to the reshape of unified.chen_features for 100/60/30/20/10 Hz;
      1-2 samples per 100 ms bin at 15 Hz). Checked equal to unified.chen_features at 30 and 60 Hz (repro_chen.json).
  (b) I2b "rate" channel: the 250 ms Hanning smoothing window has int(round(0.25*fs)) samples (as unified.chen_i2b_features); at 10 Hz
      that is 2 samples (np.hanning(2) = [0, 0] -> division by zero), so a 3-sample window (300 ms) is used there.
Stages: repro (stops on mismatch) -> run -> stats.   Usage: python -I s3_chen_curve.py [repro,run,stats]
"""
import pickle
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

from edd import chen, chen_confirm as cc  # noqa: E402
from edd import unified as U  # noqa: E402

OUT = L.OUT
CK = OUT / "checkpoints_chen"
STORED_CK = L.ROOT / "results" / "phase4" / "unified" / "checkpoints"
STORED_JSON = L.ROOT / "results" / "phase4" / "unified" / "results_chen_rate.json"
NEW = [20, 15, 10]


def edges_for(fs):
    return np.round(np.arange(26) * fs * 0.1).astype(int)


def unified_X(ch, g, fs):
    feats = [chen.bin_means(U.decimate_to_rate(ch[c], fs), edges_for(fs)) for c in U.CHEN_CHANNELS]
    return U.zscore_within(np.hstack(feats), g)


def i2b_X(ch, fs):
    P = cc.p2()
    d = {c: U.decimate_to_rate(ch[c], fs) for c in ("pupil", "dx", "dy", "miss")}
    win = max(3, int(round(0.25 * fs)))
    d["rate"] = chen.smooth_hanning(np.diff(d["pupil"], axis=1), win)
    X, _ = P.bin_features(d, P.FUSION, edges_for(fs))
    return X


def setup():
    CK.mkdir(exist_ok=True)
    tasks, (meta, ch) = U.build_chen(L.log)
    return tasks["chen_unified"], ch


def run_unified(base, ch, fs):
    X = base["X"] if fs == 30 else unified_X(ch, base["g"], fs)
    res = U.run_task(f"chen_unified_r{fs}", X, base["y"], base["g"], base["folds"], CK, n_jobs=1, val=False, log=L.log)
    return U.oof_rows(res, base["meta"])


def run_i2b(base, ch, fs):
    X = U.chen_i2b_features(ch, 30)[0] if fs == 30 else i2b_X(ch, fs)
    res = U.run_chen_i2b(f"chen_i2b_r{fs}", X, base, CK, n_jobs=1, log=L.log)
    return U.oof_rows(res, base["meta"])


def load_rows(base, kind, fs):
    """Long table of out-of-fold scores of a finished arm (new checkpoints for 30/20/15/10 Hz, stored ones for 100/60 Hz)."""
    d = CK if fs in (30, 20, 15, 10) else STORED_CK
    name = f"chen_{'unified' if kind == 'unified' else 'i2b'}_r{fs}"
    parts = {}
    for r in range(3):
        f = d / f"{name}__f100__r{r}.pkl"
        assert f.exists(), f
        parts[r] = pickle.load(open(f, "rb"))
    return U.oof_rows(parts, base["meta"])


def repro(base, ch):
    stored = json.loads(STORED_JSON.read_text())["trial_auc"]
    out = {}
    for fs in (30, 60):          # feature identity of the edge-based builders against the original ones
        out[f"unified_features_edges_vs_original_{fs}Hz_max_abs_diff"] = float(np.nanmax(np.abs(
            unified_X(ch, base["g"], fs) - U.zscore_within(U.chen_features(ch, fs=fs)[0], base["g"]))))
        out[f"i2b_features_{fs}Hz_max_abs_diff"] = float(np.nanmax(np.abs(i2b_X(ch, fs) - U.chen_i2b_features(ch, fs)[0])))
    L.log("feature identity", out)
    run_unified(base, ch, 30)
    run_i2b(base, ch, 30)
    for name, key, sck, mck in (("unified", "unified@30", "chen_unified", "chen_unified_r30"), ("I2b", "I2b@30", "chen_i2b_r30", "chen_i2b_r30")):
        rows = load_rows(base, "unified" if name == "unified" else "i2b", 30)
        auc = U.auc_items(list(U._trial_groups(rows, rows).values()))
        ref = [pickle.load(open(STORED_CK / f"{sck}__f100__r{r}.pkl", "rb")) for r in range(3)]
        mine = [pickle.load(open(CK / f"{mck}__f100__r{r}.pkl", "rb")) for r in range(3)]
        md = max(float(np.abs(a["score"] - b["score"]).max()) for ra, rb in zip(ref, mine) for a, b in zip(ra, rb))
        out[name] = {"trial_auc_new": auc, "trial_auc_stored": stored[key], "max_abs_score_diff_vs_stored_checkpoint": md}
        L.log("repro", name, out[name])
    ok = all(abs(out[k]["trial_auc_new"] - out[k]["trial_auc_stored"]) < 1e-9 and out[k]["max_abs_score_diff_vs_stored_checkpoint"] < 1e-9
             for k in ("unified", "I2b"))
    out["all_passed"] = bool(ok)
    L.jdump(OUT / "repro_chen.json", out)
    if not ok:
        L.log("REPRODUCTION FAILED - stopping")
        sys.exit(2)


def stats(base):
    rows_by = {}
    for fs in (100, 60, 30, 20, 15, 10):
        rows_by[("unified", fs)] = load_rows(base, "unified", fs)
    for fs in (60, 30, 20, 15, 10):
        rows_by[("I2b", fs)] = load_rows(base, "i2b", fs)
    _, ct = U.chen_checkpoint_persons()
    rows_by[("I2b", 100)] = ct[ct.arm == "I2b"][["repeat", "fold", "row", "unit", "y", "score"]].reset_index(drop=True)
    ref_rows = rows_by[("unified", 30)]
    groups = {k: U._trial_groups(v, ref_rows) for k, v in rows_by.items()}
    common = sorted(set.intersection(*[set(g) for g in groups.values()]))
    keys = list(groups)
    A = {k: [groups[k][u] for u in common] for k in keys}
    n = len(common)
    est = {k: U.auc_items(A[k]) for k in keys}
    rng = np.random.default_rng(U.BOOT_SEED + 11)
    N = 2000
    draws = {k: np.empty(N) for k in keys}
    for b in range(N):
        pick = rng.integers(0, n, n)
        for k in keys:
            draws[k][b] = U.auc_items([A[k][i] for i in pick])
    rows, res = [], {}
    for m in ("unified", "I2b"):
        for fs in (100, 60, 30, 20, 15, 10):
            k = (m, fs)
            d = draws[k] - draws[(m, 30)]
            r = {"auc": est[k], "auc_lo": float(np.percentile(draws[k], 2.5)), "auc_hi": float(np.percentile(draws[k], 97.5)),
                 "delta_vs_30hz": est[k] - est[(m, 30)], "delta_lo": float(np.percentile(d, 2.5)), "delta_hi": float(np.percentile(d, 97.5))}
            res[f"{m}@{fs}"] = r
            rows.append({"dataset": "Chen", "model": {"unified": "our pipeline (unified)", "I2b": "pupil bins + gaze (I2b)"}[m],
                         "condition": "released 100 Hz epochs, anti-aliased resampling", "rate_hz": fs, "auc": r["auc"], "auc_lo": r["auc_lo"],
                         "auc_hi": r["auc_hi"], "delta_vs_30hz": r["delta_vs_30hz"], "delta_lo": r["delta_lo"], "delta_hi": r["delta_hi"],
                         "source": "stored (phase4)" if fs in (100, 60) else ("reproduced (identical to stored)" if fs == 30 else "new")})
    L.jdump(OUT / "results_chen.json", {"n_units": n, "n_boot": N, "bootstrap_seed": U.BOOT_SEED + 11,
                                        "note": "trial-level AUC, participant-cluster paired bootstrap (2000 resamples, same seed and scheme as the stored analysis)",
                                        "rates": res})
    pd.DataFrame(rows).to_csv(OUT / "rows_chen.csv", index=False)
    L.log({k: round(v["auc"], 4) for k, v in res.items()})


def main():
    st = (sys.argv[1] if len(sys.argv) > 1 else "repro,run,stats").split(",")
    base, ch = setup()
    if "repro" in st:
        repro(base, ch)
    if "run" in st:
        for fs in NEW:
            run_unified(base, ch, fs)
            run_i2b(base, ch, fs)
    if "stats" in st:
        stats(base)


if __name__ == "__main__":
    main()
