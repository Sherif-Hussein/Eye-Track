"""Light, grey or dark screen for webcam gaze? One 14-dot calibration and one 5-dot check in each mode, in random order; every calibration is then tested on every mode's check.
Camera frames are processed in memory and discarded. Saved: numbers only (features, head pose, brightness, glare, targets); no camera image and no picture."""
import csv, json, os, random, sys, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "./app")
import numpy as np
import cv2
from calibrate import GazeModel, CAL_POINTS, CAL2_POINTS, VAL_POINTS
from camera import CameraThread
from display import Display
from landmarks import FaceTracker, px_per_deg, FEATURE_NAMES, EAR_R, EAR_L

OUT = os.path.dirname(os.path.abspath(__file__))
STAMP = time.strftime("%H%M%S")
CAM_INDEX, FOCAL_PX, SCREEN_W_MM = 0, 529.0, 294.0
DOT_S, SETTLE_S, ADAPT_S = 2.0, 0.7, 4.0
MODES = {"light": {"bg": (232, 232, 232), "dot": (30, 30, 30), "text": (30, 30, 30)},
         "grey": {"bg": (128, 128, 128), "dot": (245, 245, 245), "text": (245, 245, 245)},
         "dark": {"bg": (22, 22, 22), "dot": (235, 235, 235), "text": (235, 235, 235)}}

disp = Display(fullscreen=True)
disp.max_fps = 30.0
pg = disp.pg
disp.screen = pg.display.set_mode((1280, 720), pg.FULLSCREEN | pg.SCALED, vsync=1)
disp.size = disp.screen.get_size()
disp.ui = 1.0
disp._bring_to_front()
W, H = disp.size
_c = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
_c.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
for _ in range(5):
    _c.read()
_c.release()
cam = CameraThread(index=CAM_INDEX, width=640, height=480, fps=30)
cam.start()
cam.opened.wait(10)
if cam.error:
    disp.close()
    sys.exit(cam.error)
tr = FaceTracker(focal_px=FOCAL_PX, keep_pts=True)
rows = []
summary = {"camera": "built-in", "order": [], "glasses": None}
t_zero = time.perf_counter()


def glare(bgr, p):
    """% of pixels at 240 or more in a box around each eye (mean of the two eyes); computed in memory only."""
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = []
    for idx in (EAR_R, EAR_L):
        e = p[list(idx)]
        x0, y0 = e.min(0)
        x1, y1 = e.max(0)
        m = 0.3 * (x1 - x0)
        box = grey[max(0, int(y0 - m)):int(y1 + m) + 1, max(0, int(x0 - m)):int(x1 + m) + 1]
        if box.size:
            out.append(100.0 * float(np.mean(box >= 240)))
    return float(np.mean(out)) if out else float("nan")


def next_obs():
    fr = cam.get(0.004)
    if fr is None:
        return None, float("nan")
    idx, t_ns, bgr = fr
    o = tr.process(bgr, t_ns, idx)
    g = glare(bgr, o.pts) if o.face and o.pts is not None else float("nan")
    del bgr, fr                                               # discard the frame
    return o, g


def screen(mode, lines):
    m = MODES[mode]
    disp.clear(m["bg"])
    y = H // 2 - 30 * len(lines)
    for i, s in enumerate(lines):
        disp.text(s, (W // 2, y + i * 56), size=25, color=m["text"], raw=True)
    disp.flip()


def show_dot(mode, frac, block):
    m = MODES[mode]
    p = (int(frac[0] * W), int(frac[1] * H))
    t0 = time.perf_counter()
    while True:
        el = time.perf_counter() - t0
        if el >= DOT_S:
            break
        disp.poll()
        disp.clear(m["bg"])
        pg.draw.circle(disp.screen, m["dot"], p, int(max(9, 30 - 34 * el)))
        pg.draw.circle(disp.screen, m["bg"], p, 3)
        disp.flip()
        o, g = next_obs()
        if o is None:
            continue
        ok = bool(o.face) and np.isfinite(o.feat).all() if o.face else False
        rows.append([round(time.perf_counter() - t_zero, 3), mode, block, p[0], p[1], int(el > SETTLE_S), int(ok), round(float(o.img_lum), 1), round(g, 3) if np.isfinite(g) else "",
                     round(float(o.proc_ms), 2)] + ([round(float(o.ear), 4), round(float(o.yaw), 2), round(float(o.pitch), 2), round(float(o.dist_mm), 1)] + [round(float(v), 5) for v in o.feat] if ok else [""] * (4 + len(FEATURE_NAMES))))


try:
    screen("grey", ["Light, grey or dark screen?", "Are you wearing glasses now?  1 = Yes   2 = No"])
    summary["glasses"] = {"1": True, "2": False}[disp.wait_key({"1", "2"})]
    seen, ready = 0, False
    disp.clicks.clear()
    while not ready:
        o, _ = next_obs()
        if o is not None:
            seen = seen + 1 if o.face else 0
        okf = seen >= 10
        screen("grey", ["Face found. Sit normally and keep your head still. Press SPACE or click to start (about 3 minutes)." if okf else "Looking for your face...", "Look at each dot until the next one appears. The screen changes between parts."])
        for k, _t in disp.poll():
            if k == "space" and okf:
                ready = True
        if disp.clicks:
            ready = ready or okf
            disp.clicks.clear()
    order = random.sample(list(MODES), 3)
    summary["order"] = order
    for mode in order:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < ADAPT_S:                # let the eyes and the camera's exposure adapt
            disp.poll()
            screen(mode, ["%s screen. Get ready..." % mode.capitalize()])
            next_obs()
        for pts in (random.sample(CAL_POINTS, len(CAL_POINTS)), random.sample(CAL2_POINTS, len(CAL2_POINTS))):
            for fr in pts:
                show_dot(mode, fr, "cal")
        for fr in random.sample(VAL_POINTS, len(VAL_POINTS)):
            show_dot(mode, fr, "val")
    summary["complete"] = True
except KeyboardInterrupt:
    summary["stopped_with"] = "Esc"
finally:
    cols = ["t_s", "mode", "block", "target_x", "target_y", "settled", "ok", "lum", "glare_pct", "proc_ms", "ear", "yaw", "pitch", "dist_mm"] + FEATURE_NAMES
    fn = os.path.join(OUT, "mode_test_%s.csv" % STAMP)
    with open(fn, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(rows)
    cam.stop()
    tr.close()

    # ---- analysis: every calibration tested on every mode's check
    import pandas as pd
    d = pd.read_csv(fn)
    lines = []
    try:
        d["t_block"] = d.groupby(["mode", "block"]).t_s.transform("min")
        q = d.groupby("mode").apply(lambda g: pd.Series({"fps": (len(g) - 1) / max(g.t_s.max() - g.t_s.min(), 1e-6) if (g.t_s.diff().median() < 1) else float("nan"),
                                                         "face_pct": 100 * g.ok.mean(), "lum": g.lum.median(), "glare_pct": g.glare_pct.median(), "proc_ms": g.proc_ms.median(), "yaw": g.yaw.median(), "pitch": g.pitch.median(), "dist_mm": g.dist_mm.median()}), include_groups=False)
        jit = {}
        for mode, g in d[(d.ok == 1) & (d.settled == 1)].groupby("mode"):
            v = []
            for _, t in g.groupby(["block", "target_x", "target_y"]):
                v.append(np.median(np.abs(np.diff(t[["ix_r", "iy_r", "ix_l", "iy_l"]].to_numpy(float), axis=0))))
            jit[mode] = float(np.median(v)) * 1000
        q["iris_jitter_x1000"] = pd.Series(jit)
        print(q.round(2).to_string(), flush=True)
        summary["quality"] = q.round(3).to_dict(orient="index")
        fits = {}
        ok = d[(d.ok == 1) & (d.settled == 1)]
        ok = ok[ok.ear >= 0.7 * ok.groupby("mode").ear.transform(lambda s: s.quantile(0.85))]
        for mode, g in ok[ok.block == "cal"].groupby("mode"):
            F, T = g[FEATURE_NAMES].to_numpy(float), g[["target_x", "target_y"]].to_numpy(float)
            m = GazeModel(kind="m8").fit(F, T)
            u, inv = np.unique(T, axis=0, return_inverse=True)
            inv = inv.ravel()
            p_ = m.predict(np.array([np.median(F[inv == i], axis=0) for i in range(len(u))]))
            fits[mode] = (m, np.polyfit(p_[:, 0], u[:, 0], 1), np.polyfit(p_[:, 1], u[:, 1], 1), float(g.dist_mm.median()))
        res = {}
        for cm, (m, ax, ay, dist) in fits.items():
            ppd = px_per_deg(dist, SCREEN_W_MM, W)
            for vm, g in ok[ok.block == "val"].groupby("mode"):
                errs, dy = [], []
                for (x, y), t in g.groupby(["target_x", "target_y"]):
                    if len(t) >= 5:
                        q_ = m.predict(t[FEATURE_NAMES].to_numpy(float))
                        e = np.median(np.column_stack([np.polyval(ax, q_[:, 0]), np.polyval(ay, q_[:, 1])]), axis=0)
                        errs.append(float(np.hypot(e[0] - x, e[1] - y) / ppd))
                        dy.append(float((e[1] - y) / ppd))
                res["%s->%s" % (cm, vm)] = round(float(np.mean(errs)), 2) if errs else None
        summary["errors_deg_calibrate_to_check"] = res
        print(json.dumps(res), flush=True)
        lines = ["%s -> %s: %.2f deg" % (k.split("->")[0], k.split("->")[1], v) for k, v in res.items() if v is not None and k.split("->")[0] == k.split("->")[1]]
    except Exception as e:
        print("analysis error:", repr(e), flush=True)
    with open(os.path.join(OUT, "mode_test_%s_summary.json" % STAMP), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("SUMMARY " + json.dumps({k: summary.get(k) for k in ("glasses", "order", "complete", "stopped_with")}), flush=True)
    try:
        screen("grey", ["Finished. Same mode for calibration and check:"] + lines + ["Press SPACE or click to close."])
        disp.clicks.clear()
        disp.wait_key({"space"})
    except KeyboardInterrupt:
        pass
    disp.close()
