"""Task 2.3 eye-movement figures for Paper A (numbers only, no camera images).

  fig_trajectory        (a) gaze path of one Delmas trial, 1000 Hz vs 30 Hz (anti-aliased polyphase downsampling)
  fig_accuracy_targets  (b) webcam gaze vs the five validation targets, recording R4 (Paper A Fig. 2d)
  fig_drift             (c) per-trial offset at the fixation cross over the same R4 session

Usage (project root):  .venv/Scripts/python.exe -I results/phase9/figures/make_figures.py
Single process, single thread. Output: this folder (PDF + PNG 300 dpi + the CSV each figure is drawn from).
Selection for (a) is fixed in select_trial() and runs before any figure is drawn.
"""
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

from edd import delmas_raw as dr  # noqa: E402
from edd.delmas import load_trials  # noqa: E402

# ---------------------------------------------------------------- style: same setup as scripts/phase5_figures.py
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAND = "#f0efec"
W = 5.5
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
# single-hue sequential map for time (blue; start at 30% so the first samples stay visible on white)
TIME_CMAP = LinearSegmentedColormap.from_list("time_blue", plt.cm.Blues(np.linspace(0.3, 1.0, 256)))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("wrote", name, flush=True)


def panel_tag(ax, t):
    ax.text(-0.02, 1.02, t, transform=ax.transAxes, fontweight="bold", fontsize=9, ha="right", va="bottom")


# ================================================================ (a) trajectory
CACHE = Path(r"<local path>")
# ^ scratch copy of scripts/phase6_trackC_c2.load_halves output (1000 Hz, float32); the chosen participant is re-read from the EDF and compared.
BOOK64 = ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx"
KNOWN_MIN_ID = 57          # image ids 57-64 are the eight famous (known) faces (face = 1 in the authors' tables)
MIN_VALID = 0.90
PPD, CTR = dr.PIX_PER_DEG, dr.CENTER


def dwell_ms(x, y):
    """Total time (ms, 1000 Hz samples) with valid gaze inside any of the four item rectangles in the 5-s display window."""
    xs, ys = x[dr.PRE_MS:dr.PRE_MS + dr.POST_MS], y[dr.PRE_MS:dr.PRE_MS + dr.POST_MS]
    ok = np.isfinite(xs) & np.isfinite(ys)
    q = dr.aoi_of(np.where(ok, xs, -1e9), np.where(ok, ys, -1e9))
    return int((q >= 0).sum()), float(ok.mean())


def select_trial():
    book = load_trials(BOOK64)
    pids = sorted(int(p) for p in book.participant.unique())
    rows = []
    for pid in pids:
        d = np.load(CACHE / f"p{pid:02d}.npz")
        x, y, img = d["x"].astype(np.float64), d["y"].astype(np.float64), d["images"]
        for k in range(len(img)):
            if not (img[k] >= KNOWN_MIN_ID).any():
                continue
            dw, vs = dwell_ms(x[k], y[k])
            rows.append((pid, k, dw, vs))
    t = pd.DataFrame(rows, columns=["participant", "trial", "dwell_ms", "valid_share"])
    n_known = len(t)
    t = t[t.valid_share >= MIN_VALID].reset_index(drop=True)
    pt = t.groupby("participant").dwell_ms.mean().sort_values(kind="mergesort")      # participant total dwell = mean over eligible trials
    pid = int(pt.index[(len(pt) - 1) // 2])                                           # median participant (lower middle if even)
    tp = t[t.participant == pid].sort_values(["dwell_ms", "trial"], kind="mergesort").reset_index(drop=True)
    k = int(tp.trial[(len(tp) - 1) // 2])                                             # median trial of that participant
    info = {"n_participants_total": len(pids), "n_known_item_trials": n_known, "n_eligible_trials": len(t),
            "n_eligible_participants": len(pt), "participant_rank_used": (len(pt) - 1) // 2 + 1,
            "participant_median_of_means_ms": float(np.median(pt.values)), "participant": pid,
            "participant_mean_dwell_ms": float(pt[pid]), "n_eligible_trials_of_participant": len(tp),
            "trial_index_0based": k, "trial_number_1based": k + 1,
            "trial_dwell_ms": int(tp.dwell_ms[(len(tp) - 1) // 2]), "trial_valid_share": float(tp.valid_share[(len(tp) - 1) // 2]),
            "trial_dwell_range_of_participant_ms": [int(tp.dwell_ms.min()), int(tp.dwell_ms.max())]}
    return info, t


def load_raw_trial(pid, k):
    """Re-read the chosen participant from the EDF files (scripts/phase6_trackC_c2.load_halves) and compare with the cache."""
    spec = importlib.util.spec_from_file_location("trackC_c2", ROOT / "scripts" / "phase6_trackC_c2.py")
    c2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c2)
    x, y, p, img, _ = c2.load_halves(pid)
    d = np.load(CACHE / f"p{pid:02d}.npz")
    same = bool(np.array_equal(x, d["x"], equal_nan=True) and np.array_equal(y, d["y"], equal_nan=True) and np.array_equal(img, d["images"]))
    return x[k], y[k], p[k], img[k], same


def to_deg(xp, yp):
    return (xp - CTR[0]) / PPD, (yp - CTR[1]) / PPD


def make_trajectory():
    info, elig = select_trial()
    print("SELECTION", json.dumps(info), flush=True)
    pid, k = info["participant"], info["trial_index_0based"]
    x, y, p, img, same = load_raw_trial(pid, k)
    info["cache_equals_raw_edf"] = same
    print("cache equals raw EDF for this participant:", same, flush=True)
    assert same
    x1, y1, _ = dr.degrade(x, y, p, 1000)
    x3, y3, _ = dr.degrade(x, y, p, 30)
    # display window 0 .. 5000 ms after onset
    t1 = np.arange(len(x1)) - dr.PRE_MS
    t3 = np.arange(len(x3)) * 1000.0 / 30 - dr.PRE_MS
    m1, m3 = (t1 >= 0) & (t1 < 5000), (t3 >= 0) & (t3 < 5000)
    series = {1000: (t1[m1], x1[m1], y1[m1]), 30: (t3[m3], x3[m3], y3[m3])}
    rows = []
    for fs, (t, xx, yy) in series.items():
        dxd, dyd = to_deg(xx, yy)
        for i in range(len(t)):
            rows.append((fs, i, t[i], xx[i], yy[i], dxd[i], dyd[i], int(np.isfinite(xx[i]) and np.isfinite(yy[i]))))
    path = pd.DataFrame(rows, columns=["rate_hz", "sample", "t_ms", "x_px", "y_px", "x_deg", "y_deg", "valid"])
    path.to_csv(OUT / "fig_trajectory.csv", index=False, float_format="%.4f")
    # item layout
    known = [bool(img[q] >= KNOWN_MIN_ID) for q in range(4)]
    items = []
    for q in range(4):
        x0, y0, x1_, y1_ = dr.QUAD_RECTS[q]
        a, b = to_deg(np.array([x0, x1_]), np.array([y0, y1_]))
        items.append((q, int(img[q]), "known item" if known[q] else "unknown item", x0, y0, x1_, y1_, a[0], b[0], a[1], b[1]))
    pd.DataFrame(items, columns=["slot", "image_id", "role", "x0_px", "y0_px", "x1_px", "y1_px", "x0_deg", "y0_deg", "x1_deg", "y1_deg"]
                 ).to_csv(OUT / "fig_trajectory_items.csv", index=False, float_format="%.4f")
    # quantities for the card
    stats = {}
    for fs, (t, xx, yy) in series.items():
        ok = np.isfinite(xx) & np.isfinite(yy)
        dxd, dyd = to_deg(xx, yy)
        q = dr.aoi_of(np.where(ok, xx, -1e9), np.where(ok, yy, -1e9))
        dwell = (q >= 0).sum() * (1000.0 / fs)
        per_item = [float((q == j).sum() * 1000.0 / fs) for j in range(4)]
        stats[fs] = {"n_samples": int(len(t)), "n_valid": int(ok.sum()), "dwell_ms_on_items": float(dwell), "dwell_per_slot_ms": per_item,
                     "x_deg_range": [float(np.nanmin(dxd)), float(np.nanmax(dxd))], "y_deg_range": [float(np.nanmin(dyd)), float(np.nanmax(dyd))]}
    info["stats"] = stats
    info["image_ids_by_slot"] = [int(v) for v in img]
    info["known_by_slot"] = known
    (OUT / "fig_trajectory_selection.json").write_text(json.dumps(info, indent=1), encoding="utf-8")

    # ---- figure
    xl, yl = (-18.0, 18.0), (-17.5, 17.5)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.05), gridspec_kw={"wspace": 0.22})
    norm = Normalize(0, 5)
    for ax, fs, lab in [(a, 1000, "1000 Hz"), (b, 30, "30 Hz")]:
        for q, (_, _, role, _, _, _, _, ax0, ay0, ax1, ay1) in enumerate(items):
            isk = role == "known item"
            ax.add_patch(Rectangle((ax0, ay0), ax1 - ax0, ay1 - ay0, fill=True, fc=BAND if not isk else "#fbe3d8",
                                   ec=ORANGE if isk else MUTED, lw=1.0 if isk else 0.6, zorder=1))
            ax.text((ax0 + ax1) / 2, ay0 + 0.5 if ay0 < 0 else ay1 - 0.5, role.replace(" ", "\n"), ha="center", va="top" if ay0 < 0 else "bottom",
                    fontsize=5.5, color=ORANGE if isk else INK2, zorder=2)
        t, xx, yy = series[fs]
        xd, yd = to_deg(xx, yy)
        ok = np.isfinite(xd) & np.isfinite(yd)
        seg_ok = ok[:-1] & ok[1:]
        pts = np.column_stack([xd, yd])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)[seg_ok]
        tm = ((t[:-1] + t[1:]) / 2000.0)[seg_ok]
        lc = LineCollection(segs, cmap=TIME_CMAP, norm=norm, linewidths=0.8 if fs == 1000 else 1.1, zorder=3)
        lc.set_array(tm)
        ax.add_collection(lc)
        if fs == 30:
            ax.scatter(xd[ok], yd[ok], c=(t[ok] / 1000.0), cmap=TIME_CMAP, norm=norm, s=7, edgecolor="white", linewidths=0.3, zorder=4)
        i0 = np.flatnonzero(ok)[0]
        ax.scatter([xd[i0]], [yd[i0]], s=22, marker="o", facecolor="white", edgecolor=INK, linewidths=0.8, zorder=5)
        ax.annotate("start", (xd[i0], yd[i0]), xytext=(6, 5), textcoords="offset points", fontsize=6, color=INK, zorder=6,
                    ha="left", va="bottom", arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
        ax.set_xlim(*xl)
        ax.set_ylim(yl[1], yl[0])             # screen y grows downwards
        ax.set_aspect("equal")
        ax.set_xlabel("Horizontal position (°)")
        ax.set_xticks([-15, -10, -5, 0, 5, 10, 15])
        ax.set_yticks([-15, -10, -5, 0, 5, 10, 15])
        ax.text(0, -16.0, f"{lab}, {stats[fs]['n_valid']} valid samples", fontsize=7, color=INK, ha="center", va="center", zorder=7)
    a.set_ylabel("Vertical position (°)")
    b.set_yticklabels([])
    panel_tag(a, "a")
    panel_tag(b, "b")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=TIME_CMAP), ax=[a, b], orientation="horizontal", fraction=0.045, pad=0.17, aspect=40)
    cb.set_label("Time since display onset (s)")
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=2)
    save(fig, "fig_trajectory")
    return info


# ================================================================ (b),(c) webcam recording R4
SESSION = ROOT / "app" / "data" / "test06_faces_I_emeet_glasses"       # R4 (see results/app_pilot/paper_summary/export_fig3de.py)
PS = ROOT / "results" / "app_pilot" / "paper_summary"
SKIP_MS = 700.0
CFG_RECHECK = None


def webcam_from_logs():
    m = json.loads((SESSION / "meta.json").read_text())
    f = pd.read_csv(SESSION / "frames.csv")               # numeric landmark-derived columns only
    ev = pd.read_csv(SESSION / "events.csv")
    Wp, Hp = m["screen_px"]
    ppd = m["calibration"]["px_per_deg"]
    on = ev[ev.event == "dot_onset"].reset_index(drop=True)
    off = ev[ev.event == "dot_offset"].reset_index(drop=True)
    samples, targets = [], []
    for i in range(len(on)):
        kind, idx, x, y, att = on.detail[i].split(":")
        if kind != "validation":
            continue
        if len(targets) == 5:
            break
        w = f[(f.t_ms >= on.t_ms[i] + SKIP_MS) & (f.t_ms <= off.t_ms[i]) & f.gaze_x_live.notna() & (f.face == 1)]
        gx, gy = (w.gaze_x_live.to_numpy() - Wp / 2) / ppd, (w.gaze_y_live.to_numpy() - Hp / 2) / ppd
        tx, ty = (float(x) - Wp / 2) / ppd, (float(y) - Hp / 2) / ppd
        mx, my = float(np.median(gx)), float(np.median(gy))
        k = len(targets)
        samples += [{"target": k, "x_deg": a, "y_deg": b} for a, b in zip(gx, gy)]
        targets.append({"target": k, "tx_deg": tx, "ty_deg": ty, "median_x_deg": mx, "median_y_deg": my, "offset_deg": float(np.hypot(mx - tx, my - ty)),
                        "n_samples": len(gx)})
    return pd.DataFrame(samples), pd.DataFrame(targets), m, ppd


def make_accuracy():
    smp, tg, m, ppd = webcam_from_logs()
    # reproduce the existing figure's inputs and its number (phase5_figures.py: tg.offset_deg.mean() in the panel title)
    old_s, old_t = pd.read_csv(PS / "fig3d_samples.csv"), pd.read_csv(PS / "fig3d_targets.csv")
    d_s = float(np.abs(smp[["x_deg", "y_deg"]].to_numpy() - old_s[["x_deg", "y_deg"]].to_numpy()).max())
    d_t = float(np.abs(tg[["tx_deg", "ty_deg", "median_x_deg", "median_y_deg", "offset_deg"]].to_numpy()
                       - old_t[["tx_deg", "ty_deg", "median_x_deg", "median_y_deg", "offset_deg"]].to_numpy()).max())
    mean_new, mean_old = float(tg.offset_deg.mean()), float(old_t.offset_deg.mean())
    sess = pd.read_csv(PS / "emeet_sessions.csv")
    acc_tbl = float(sess[sess.recording == SESSION.name].accuracy_deg.iloc[0])
    print(f"(b) rows {len(smp)} (old {len(old_s)}); max abs diff samples {d_s:.2e}, targets {d_t:.2e}; mean offset new {mean_new:.6f} old {mean_old:.6f} "
          f"table {acc_tbl:.6f}; title text of old figure '{mean_old:.2f}°'", flush=True)
    assert len(smp) == len(old_s) and d_s < 1e-3 and d_t < 1e-3 and abs(mean_new - mean_old) < 1e-4
    smp.to_csv(OUT / "fig_accuracy_targets_samples.csv", index=False, float_format="%.4f")
    tg.to_csv(OUT / "fig_accuracy_targets_targets.csv", index=False, float_format="%.4f")
    chk = {"n_samples": int(len(smp)), "max_abs_diff_samples_deg": d_s, "max_abs_diff_targets_deg": d_t, "mean_offset_new_deg": mean_new,
           "mean_offset_existing_fig_code_deg": mean_old, "accuracy_deg_in_emeet_sessions_csv": acc_tbl, "px_per_deg_meta": ppd,
           "screen_px": m["screen_px"], "screen_deg": [m["screen_px"][0] / ppd, m["screen_px"][1] / ppd],
           "per_target_offset_deg": [float(v) for v in tg.offset_deg], "n_samples_per_target": [int(v) for v in tg.n_samples],
           "rms_sample_scatter_per_target_deg": [float(np.sqrt(((smp[smp.target == i][["x_deg", "y_deg"]] - smp[smp.target == i][["x_deg", "y_deg"]].median()) ** 2).sum(1).mean()))
                                                 for i in range(5)]}
    (OUT / "fig_accuracy_targets_check.json").write_text(json.dumps(chk, indent=1), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(W * 0.66, 3.1))
    ax.scatter(smp.x_deg, smp.y_deg, s=6, color=BLUE, alpha=0.35, lw=0, zorder=2)
    for r in tg.itertuples():
        ax.plot([r.tx_deg, r.median_x_deg], [r.ty_deg, r.median_y_deg], color=INK, lw=0.9, zorder=3)
    ax.scatter(tg.median_x_deg, tg.median_y_deg, s=16, color=INK, zorder=4)
    ax.scatter(tg.tx_deg, tg.ty_deg, s=55, marker="+", color=ORANGE, lw=1.5, zorder=5)
    ax.set_xlim(-9.0, 8.2)
    ax.set_ylim(6.6, -6.6)                   # screen y grows downwards
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal position (°)")
    ax.set_ylabel("Vertical position (°)")
    ax.set_xticks([-8, -4, 0, 4, 8])
    ax.set_yticks([-6, -3, 0, 3, 6])
    # offset values: at the median end of each line, set away from the line
    for r in tg.itertuples():
        dx, dy = r.median_x_deg - r.tx_deg, r.median_y_deg - r.ty_deg
        ox = 0.45 if dx >= 0 else -0.45
        ax.text(r.median_x_deg + ox, r.median_y_deg + (0.55 if dy >= 0 else -0.25), f"{r.offset_deg:.1f}°",
                ha="left" if dx >= 0 else "right", va="center", fontsize=6.5, color=INK2, zorder=6,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    # direct labels for the three marks (placed in empty space, thin leader lines)
    r0 = tg.iloc[1]
    ax.annotate("target", (r0.tx_deg, r0.ty_deg), xytext=(-5.2, -0.3), fontsize=6.5, color=ORANGE, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=ORANGE, shrinkA=1, shrinkB=2))
    r2 = tg.iloc[2]
    ax.annotate("median gaze", (r2.median_x_deg, r2.median_y_deg), xytext=(6.5, 0.6), fontsize=6.5, color=INK, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=1, shrinkB=2))
    ax.text(0.02, 0.98, f"gaze samples (30 Hz), n = {len(smp)}", transform=ax.transAxes, fontsize=6.5, color=BLUE, ha="left", va="top", zorder=7)
    ax.text(0.02, 0.905, f"mean offset {mean_new:.2f}° over 5 targets", transform=ax.transAxes, fontsize=6.5, color=INK, ha="left", va="top", zorder=7)
    save(fig, "fig_accuracy_targets")
    return chk


def make_drift():
    t = pd.read_csv(SESSION / "trials.csv").sort_values("trial")
    pr = pd.read_csv(SESSION / "practice.csv").sort_values("trial")
    ev = pd.read_csv(SESSION / "events.csv")
    m = json.loads((SESSION / "meta.json").read_text())
    sys.path.insert(0, str(ROOT / "app"))
    from config import Config                                               # app/config.py (defaults; no config.json override is used here)
    cfgj = (ROOT / "app" / "config.json")
    lim = float(Config().drift_recheck_deg)
    n_consec = int(Config().auto_recheck_trials)
    cfg_override = cfgj.exists() and "drift_recheck_deg" in json.loads(cfgj.read_text())
    mcfg = m.get("config", {}) if isinstance(m, dict) else {}
    lim_session = mcfg.get("drift_recheck_deg") if isinstance(mcfg, dict) else None
    ar = ev[ev.event == "auto_recheck"]
    rows = []
    for r in pr.itertuples():
        rows.append(("practice", int(r.trial), float(r.fix_on_ms) / 1000.0, float(r.drift_deg), int(r.segment), int(r.fixation_ok)))
    for r in t.itertuples():
        rows.append(("scored", int(r.trial), float(r.fix_on_ms) / 1000.0, float(r.drift_deg), int(r.segment), int(r.fixation_ok)))
    d = pd.DataFrame(rows, columns=["kind", "trial_in_kind", "time_s", "drift_deg", "calibration_segment", "fixation_ok"])
    d["order"] = np.arange(1, len(d) + 1)
    d["above_recheck_limit"] = (d.drift_deg > lim).astype(int)
    t_re = float(ar.t_ms.iloc[0]) / 1000.0 if len(ar) else np.nan
    d.to_csv(OUT / "fig_drift.csv", index=False, float_format="%.4f")
    old = pd.read_csv(PS / "fig3e_drift.csv")
    sc = d[d.kind == "scored"]
    diff = float(np.abs(sc.drift_deg.to_numpy() - old.drift_deg.to_numpy()).max())
    med_sc, med_all = float(sc.drift_deg.median()), float(d.drift_deg.median())
    print(f"(c) limit {lim}; scored median {med_sc:.4f}; all-20 median {med_all:.4f}; diff to fig3e {diff:.2e}; recheck t {t_re:.1f}s; "
          f"emeet table drift_median {pd.read_csv(PS / 'emeet_sessions.csv').query('recording == @SESSION.name').drift_median_deg.iloc[0]:.4f}", flush=True)
    assert diff < 1e-3
    chk = {"recheck_limit_deg": lim, "auto_recheck_trials": n_consec, "config_override_of_limit": bool(cfg_override), "session_meta_limit": lim_session,
           "median_scored_deg": med_sc, "median_all_deg": med_all, "n_scored": int(len(sc)), "n_practice": int((d.kind == "practice").sum()),
           "scored_above_limit": [[int(a), float(b)] for a, b in zip(sc.trial_in_kind, sc.drift_deg) if b > lim],
           "practice_drift_deg": [float(v) for v in d[d.kind == "practice"].drift_deg], "auto_recheck_event": ar.detail.tolist(),
           "auto_recheck_time_s": t_re, "scored_min_max_deg": [float(sc.drift_deg.min()), float(sc.drift_deg.max())],
           "scored_p90_deg": float(np.percentile(sc.drift_deg, 90)), "diff_to_fig3e_csv_deg": diff,
           "session_span_s": [float(d.time_s.min()), float(d.time_s.max())]}
    (OUT / "fig_drift_check.json").write_text(json.dumps(chk, indent=1), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(W * 0.72, 2.3))
    n_pr = int((d.kind == "practice").sum())
    ax.axhline(lim, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(len(d) + 0.4, lim + 0.12, f"re-check limit {lim:.0f}°", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.axhline(med_sc, color=BLUE, lw=0.9, zorder=1)
    ax.text(len(d) + 0.4, med_sc - 0.12, f"median of trials 1–{len(sc)}: {med_sc:.1f}°", ha="right", va="top", fontsize=6.5, color=BLUE)
    ax.axvspan(0.5, n_pr + 0.5, color=BAND, lw=0, zorder=0)
    ax.axvline(n_pr + 0.5, color=INK2, lw=0.7, zorder=1)
    ax.text(n_pr + 0.5 + 0.25, 0.25, "automatic re-check,\nrecalibration", ha="left", va="bottom", fontsize=6, color=INK2)
    ax.plot(d.order, d.drift_deg, color=INK, lw=0.7, zorder=2)
    pr_d, sc_d = d[d.kind == "practice"], d[d.kind == "scored"]
    ax.scatter(pr_d.order, pr_d.drift_deg, s=20, facecolor="white", edgecolor=INK, lw=0.9, zorder=3)
    ax.scatter(sc_d.order, sc_d.drift_deg, s=20, color=INK, edgecolor="white", lw=0.5, zorder=3)
    ax.set_xlim(0.5, len(d) + 0.5)
    ax.set_ylim(0, 7)
    ax.set_xticks([1, 2] + list(range(n_pr + 1, len(d) + 1, 3)) + [len(d)])
    lab = {1: "P1", 2: "P2"}
    ax.set_xticklabels([lab.get(v, str(v - n_pr)) for v in ax.get_xticks()])
    ax.set_xlabel("Trial (P = practice)")
    ax.set_ylabel("Offset at fixation cross (°)")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_drift")
    return chk


if __name__ == "__main__":
    info = make_trajectory()
    chk_b = make_accuracy()
    chk_c = make_drift()
    print("done")
