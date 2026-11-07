"""Task 2.2b (EXPLORATORY), step 0: measure the observed-webcam parameters from our own recordings (numbers only).

Sources (read only): app/data/{test05_faces_K_emeet_glasses, test06_faces_I_emeet_glasses, test08_v1_18trials, test09_v1_18trials}/
  trials.csv, frames.csv, events.csv, meta.json ; results/phase9/figures/fig_drift.csv ; results/app_pilot/vertical_check/output.txt.
Reuses UNCHANGED app/features.py ear_threshold (the app's blink threshold). No video, no images; only numeric columns are read.

Measures
 (a) per-trial offset: trials.csv drift_dx_px, drift_dy_px (scored trials with fixation_ok = 1) / session px_per_deg -> deg. The offset of the
     GAZE is minus the logged drift (the drift is the correction that moves the median gaze onto the cross).
 (b) gains: median over the EMEET calibration segments (sessions test05-test09, 11 segments) of the slope of (target + error) on target, the rule of
     results/app_pilot/vertical_check/vertical_check.py applied to vertical_check_targets.csv (base variant), EMEET sessions only.
 (c) jitter: gaze_x_live/gaze_y_live during validation and re-calibration dots (settled part: first 0.7 s dropped as the app does, valid frames),
     residual around each dot's median, per axis, in deg; lag 1..5 autocorrelation (pairs of valid frames exactly k frames apart) and as a
     function of the elapsed time; RMS of consecutive differences (the app's rms_s2s definition).
 (d) loss: trials.csv valid_share of scored trials; validation loss_pct from meta.json.
 (e) frame timing: t_ms gaps of frames.csv (processed frames, as recorded; frames dropped in the backlog appear as longer gaps).
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import lro_common as C  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from features import ear_threshold  # noqa: E402  (app/features.py, unchanged)

C.MEAS.mkdir(exist_ok=True)
SKIP_MS = 700.0
KAD = (1.66, 1.42)
GROSS_RMS_DEG = 20.0      # a settled dot whose residual RMS exceeds this (on either axis) is a tracking failure, not jitter (screen is 28 x 16 deg); excluded and listed


def load(s):
    d = C.APPDATA / s
    return (pd.read_csv(d / "trials.csv"), pd.read_csv(d / "frames.csv"), pd.read_csv(d / "events.csv"), json.loads((d / "meta.json").read_text(encoding="utf-8")))


json = C.json
out = {"sessions": C.SESSIONS}

# ----------------------------------------------------------------------------------- (a) drift, (d) valid share
dr_rows, vs_rows, per_sess = [], [], {}
for s in C.SESSIONS:
    t, fr, ev, meta = load(s)
    ppd = float(meta["px_per_deg"])
    sc = t[(t.lookaway == 0)]
    ok = sc[sc.fixation_ok == 1]
    for _, r in ok.iterrows():
        dr_rows.append({"session": s, "trial": int(r.trial), "drift_dx_px": r.drift_dx_px, "drift_dy_px": r.drift_dy_px, "px_per_deg": ppd,
                        "gaze_off_x_deg": -r.drift_dx_px / ppd, "gaze_off_y_deg": -r.drift_dy_px / ppd})
    for _, r in sc.iterrows():
        vs_rows.append({"session": s, "trial": int(r.trial), "valid_share": float(r.valid_share)})
    ve = meta.get("validation_end") or {}
    per_sess[s] = {"px_per_deg": ppd, "n_scored": int(len(sc)), "n_check_ok": int(len(ok)), "n_check_failed": int(len(sc) - len(ok)),
                   "mean_valid_share": float(sc.valid_share.mean()), "validation_end_loss_pct": float(ve.get("loss_pct", float("nan"))),
                   "validation_end_rms_s2s_x": float(ve.get("rms_s2s_x_deg", float("nan"))), "validation_end_rms_s2s_y": float(ve.get("rms_s2s_y_deg", float("nan")))}
drift = pd.DataFrame(dr_rows)
drift.to_csv(C.MEAS / "drift_offsets.csv", index=False)
vs = pd.DataFrame(vs_rows)
vs.to_csv(C.MEAS / "valid_share.csv", index=False)
mag = np.hypot(drift.gaze_off_x_deg, drift.gaze_off_y_deg)
fd = pd.read_csv(C.ROOT / "results" / "phase9" / "figures" / "fig_drift.csv")
fdsc = fd[fd.kind == "scored"] if "scored" in set(fd.kind) else fd
out["drift"] = {"n_trials": int(len(drift)), "n_failed_checks_excluded": int(sum(v["n_check_failed"] for v in per_sess.values())),
                "magnitude_deg": {"mean": float(mag.mean()), "median": float(mag.median()) if hasattr(mag, "median") else float(np.median(mag)),
                                  "p10": float(np.percentile(mag, 10)), "p90": float(np.percentile(mag, 90)), "min": float(mag.min()), "max": float(mag.max())},
                "gaze_offset_x_deg": {"mean": float(drift.gaze_off_x_deg.mean()), "sd": float(drift.gaze_off_x_deg.std())},
                "gaze_offset_y_deg": {"mean": float(drift.gaze_off_y_deg.mean()), "sd": float(drift.gaze_off_y_deg.std())},
                "corr_x_y": float(np.corrcoef(drift.gaze_off_x_deg, drift.gaze_off_y_deg)[0, 1]),
                "fig_drift_csv_scored_median_deg": float(np.median(fdsc.drift_deg)), "fig_drift_csv_n_scored": int(len(fdsc)),
                "fig_drift_csv_range": [float(fdsc.drift_deg.min()), float(fdsc.drift_deg.max())],
                "per_session_median_magnitude_deg": {s: float(np.median(np.hypot(g.gaze_off_x_deg, g.gaze_off_y_deg))) for s, g in drift.groupby("session")}}
out["valid_share"] = {"n_trials": int(len(vs)), "mean": float(vs.valid_share.mean()), "mean_loss": float(1 - vs.valid_share.mean()),
                      "p10": float(np.percentile(vs.valid_share, 10)), "min": float(vs.valid_share.min()),
                      "validation_loss_pct_mean_over_sessions": float(np.nanmean([v["validation_end_loss_pct"] for v in per_sess.values()]))}
out["per_session"] = per_sess

# ----------------------------------------------------------------------------------- (b) gains
vt = pd.read_csv(C.ROOT / "results" / "app_pilot" / "vertical_check" / "vertical_check_targets.csv")
vt = vt[(vt.variant == "base") & vt.session.isin(C.SESSIONS)]
gy, gx = [], []
for _, g in vt.groupby(["session", "segment"]):
    if g.y_frac.nunique() >= 2:
        gy.append(np.polyfit(g.target_y, g.target_y + g.dy_deg * g.ppd, 1)[0])
    if g.target_x.nunique() >= 2:
        gx.append(np.polyfit(g.target_x, g.target_x + g.dx_deg * g.ppd, 1)[0])
out["gain"] = {"y": round(float(np.median(gy)), 2), "x": round(float(np.median(gx)), 2), "y_unrounded": float(np.median(gy)), "x_unrounded": float(np.median(gx)),
               "n_segments": len(gy), "source": "results/app_pilot/vertical_check/vertical_check_targets.csv, base variant, EMEET sessions test05-test09 (rule of vertical_check.py)"}
assert (out["gain"]["y"], out["gain"]["x"]) == (0.79, 0.88), out["gain"]

# ----------------------------------------------------------------------------------- (c) jitter from dots
LAGS = 5
resx, resy, tt, fidx, dotid, sess_of = [], [], [], [], [], []
dot_n = []
excluded_dots = []
n_dot = 0
for s in C.SESSIONS:
    t, fr, ev, meta = load(s)
    ppd = float(meta["px_per_deg"])
    thr = ear_threshold(fr.ear.to_numpy(), fr.face.to_numpy())
    on = ev[ev.event == "dot_onset"].reset_index(drop=True)
    off = ev[ev.event == "dot_offset"].reset_index(drop=True)
    assert len(on) == len(off) and (on.detail.values == off.detail.values).all()
    gx, gy = fr.gaze_x_live.to_numpy(float), fr.gaze_y_live.to_numpy(float)
    valid = fr.face.to_numpy(bool) & np.isfinite(gx) & np.isfinite(gy) & ~(np.isfinite(thr) & (fr.ear.to_numpy() < thr))
    for i in range(len(on)):
        m_ = (fr.t_ms.to_numpy() >= on.t_ms[i] + SKIP_MS) & (fr.t_ms.to_numpy() <= off.t_ms[i]) & valid
        if m_.sum() < 20:
            continue
        x, y = gx[m_] / ppd, gy[m_] / ppd
        if max(np.sqrt(np.mean((x - np.median(x)) ** 2)), np.sqrt(np.mean((y - np.median(y)) ** 2))) > GROSS_RMS_DEG:
            excluded_dots.append({"session": s, "dot": on.detail[i], "rms_x_deg": float(np.sqrt(np.mean((x - np.median(x)) ** 2))),
                                  "rms_y_deg": float(np.sqrt(np.mean((y - np.median(y)) ** 2)))})
            continue
        resx.append(x - np.median(x)); resy.append(y - np.median(y))
        tt.append(fr.t_ms.to_numpy()[m_]); fidx.append(fr.frame_idx.to_numpy()[m_])
        dotid.append(np.full(m_.sum(), n_dot)); sess_of.append(np.full(m_.sum(), C.SESSIONS.index(s)))
        dot_n.append(int(m_.sum()))
        n_dot += 1
pk = lambda L: np.concatenate(L)  # noqa: E731
Rx, Ry, T_, F_, D_, S_ = pk(resx), pk(resy), pk(tt), pk(fidx), pk(dotid), pk(sess_of)
dfj = pd.DataFrame({"session": [C.SESSIONS[k] for k in S_], "dot": D_, "frame_idx": F_, "t_ms": T_, "res_x_deg": Rx, "res_y_deg": Ry})
dfj.to_csv(C.MEAS / "jitter_residuals.csv", index=False)


def stats_for(mask_sess=None):
    sel = np.ones(len(Rx), bool) if mask_sess is None else (S_ == mask_sess)
    res = {}
    for ax, R in (("x", Rx), ("y", Ry)):
        r = R[sel]
        dots = D_[sel]
        n_d = len(np.unique(dots))
        sd = float(np.sqrt(np.sum(r ** 2) / (len(r) - n_d)))              # df-corrected for the removed dot medians
        rho = {}
        for k in range(1, LAGS + 1):
            same = (dots[k:] == dots[:-k]) & (F_[sel][k:] - F_[sel][:-k] == k)
            a, b = r[:-k][same], r[k:][same]
            rho[k] = float(np.mean(a * b) / np.mean(r ** 2))
        # RMS of consecutive-frame differences (the app's definition)
        same1 = (dots[1:] == dots[:-1]) & (F_[sel][1:] - F_[sel][:-1] == 1)
        d1 = (r[1:] - r[:-1])[same1]
        res[ax] = {"sd_resid_deg": sd, "rho": rho, "rms_s2s_deg": float(np.sqrt(np.mean(d1 ** 2))), "n_pairs_lag1": int(same1.sum()), "n_samples": int(len(r)), "n_dots": int(n_d),
                   "excess_kurtosis": float(((r / r.std()) ** 4).mean() - 3)}
    sel1 = np.flatnonzero(sel)
    res["corr_xy_lag0"] = float(np.corrcoef(Rx[sel], Ry[sel])[0, 1])
    return res


out["jitter"] = {"excluded_gross_failure_dots": excluded_dots, "gross_rms_rule_deg": GROSS_RMS_DEG, "pooled": stats_for(), "per_session": {s: stats_for(i) for i, s in enumerate(C.SESSIONS)}, "n_dots": int(n_dot),
                 "dots_per_session": {s: int(len(np.unique(D_[S_ == i]))) for i, s in enumerate(C.SESSIONS)},
                 "settled_frames_per_dot_median": float(np.median(dot_n)), "kaduk_rms_s2s_deg": {"x": KAD[0], "y": KAD[1]},
                 "kaduk_white_equiv_sd_deg": {"x": KAD[0] / 2 ** 0.5, "y": KAD[1] / 2 ** 0.5}}

# autocorrelation against elapsed time (pairs of valid frames k = 1..5 frames apart, both within one dot)
rows = []
for ax, R in (("x", Rx), ("y", Ry)):
    v0 = np.mean(R ** 2)
    for k in range(1, LAGS + 1):
        same = (D_[k:] == D_[:-k]) & (F_[k:] - F_[:-k] == k)
        dt = (T_[k:] - T_[:-k])[same]
        pr = (R[:-k] * R[k:])[same] / v0
        rows.append(pd.DataFrame({"axis": ax, "k": k, "dt_ms": dt, "prod": pr}))
pairs = pd.concat(rows, ignore_index=True)
pairs.to_csv(C.MEAS / "jitter_pairs.csv.gz", index=False)

# ----------------------------------------------------------------------------------- (e) frame gaps
gaps = {}
for s in C.SESSIONS:
    t, fr, ev, meta = load(s)
    g = np.diff(fr.t_ms.to_numpy(float))
    gaps[s] = g
    np.save(C.MEAS / f"gaps_{s}.npy", g)
allg = np.concatenate(list(gaps.values()))
med = float(np.median(allg))
out["timing"] = {"n_gaps": int(len(allg)), "mean_gap_ms": float(allg.mean()), "median_gap_ms": med, "fps_from_mean": float(1000 / allg.mean()),
                 "pctl_1_5_25_75_95_99_99.9": [float(v) for v in np.percentile(allg, [1, 5, 25, 75, 95, 99, 99.9])], "max_gap_ms": float(allg.max()),
                 "share_gap_over_1.4_median": float(np.mean(allg > 1.4 * med)), "share_gap_over_2.2_median": float(np.mean(allg > 2.2 * med)),
                 "per_session_mean_gap_ms": {s: float(g.mean()) for s, g in gaps.items()},
                 "camera_meta": {s: {k: load(s)[3]["camera"][k] for k in ("achieved_fps", "frames_read", "frames_dropped_backlog", "frames_processed")} for s in C.SESSIONS}}
C.jdump(C.MEAS / "measured_params.json", out)
print(json.dumps({k: out[k] for k in ("drift", "valid_share", "gain", "timing")}, indent=1)[:5000])
print(json.dumps(out["jitter"]["pooled"], indent=1))
for s, v in out["jitter"]["per_session"].items():
    print(s, {ax: (round(v[ax]["sd_resid_deg"], 3), round(v[ax]["rho"][1], 3), round(v[ax]["rms_s2s_deg"], 3)) for ax in "xy"})
