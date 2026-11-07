"""Live gaze line on the laptop camera: calibration, 5-dot accuracy check, then a line follows the gaze over simple shapes until SPACE.
Keys while tracking: R recalibrate, G guided check, A apply the correction found by G, N more shapes (4, 6, 9, 12), C re-centre on the cross, X clear the line, SPACE stop, Esc quit.
Camera frames are processed in memory and discarded. Saved: gaze numbers (CSV), a summary (JSON) and a picture of the drawn line (no camera image)."""
import csv, json, math, os, random, sys, time
from collections import deque
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
sys.stdout.reconfigure(encoding="utf-8")
A = "./app"
sys.path.insert(0, A)
import numpy as np
import cv2
from calibrate import GazeModel, CAL_POINTS, CAL2_POINTS, VAL_POINTS
from camera import CameraThread
from display import Display
from landmarks import FaceTracker, px_per_deg

OUT = os.path.dirname(os.path.abspath(__file__))
CAM_INDEX, FOCAL_PX, SCREEN_W_MM = 0, 529.0, 294.0          # built-in camera; focal as in the earlier sessions on this camera
DOT_S, SETTLE_S = 2.0, 0.7
GUARD_YAW, GUARD_PITCH, GUARD_DIST = 25.0, 20.0, 0.30         # head limits against the calibration position: deg, deg, fraction of the distance
GUIDED_S, GUIDED_SETTLE_S = 3.0, 1.0
BG, INK, LINE, DOT, RING = (128, 128, 128), (15, 15, 15), (10, 40, 120), (225, 50, 20), (170, 170, 170)
PALE, LIT, TARGET = (205, 218, 235), (250, 205, 120), (240, 130, 40)


class OneEuro:
    """Adaptive smoothing: strong when the gaze is steady, light during a fast movement."""

    def __init__(self, mincut=0.4, beta=0.0003, dcut=1.0):
        self.mincut, self.beta, self.dcut, self.x, self.dx, self.t = mincut, beta, dcut, None, 0.0, None

    @staticmethod
    def _a(cut, dt):
        return 1.0 / (1.0 + 1.0 / (2 * math.pi * cut * dt))

    def __call__(self, x, t):
        if self.x is None or t <= self.t:
            self.x, self.t = x, t
            return x
        dt = t - self.t
        self.t = t
        self.dx += self._a(self.dcut, dt) * ((x - self.x) / dt - self.dx)
        self.x += self._a(self.mincut + self.beta * abs(self.dx), dt) * (x - self.x)
        return self.x


disp = Display(fullscreen=True)
disp.max_fps = 30.0
pg = disp.pg
# draw at 1280x720, scaled up by the graphics card: drawing at 4K took processor time from the camera
disp.screen = pg.display.set_mode((1280, 720), pg.FULLSCREEN | pg.SCALED, vsync=1)
disp.size = disp.screen.get_size()
disp.ui = 1.0
disp._bring_to_front()
W, H = disp.size
_c = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)              # the driver keeps a fixed exposure left by an earlier program: restore auto
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
tr = FaceTracker(focal_px=FOCAL_PX)


def next_obs():
    fr = cam.get(0.004)
    if fr is None:
        return None
    idx, t_ns, bgr = fr
    o = tr.process(bgr, t_ns, idx)
    del bgr, fr                                               # discard the frame
    return o


def message(lines):
    disp.clear(BG)
    y = H // 2 - 30 * len(lines)
    for i, s in enumerate(lines):
        disp.text(s, (W // 2, y + i * 60), size=26, color=INK, raw=True)
    disp.flip()


def show_dot(frac, seconds=DOT_S, label=""):
    """Shows one dot; returns (features, eye openness, distances, head yaw/pitch) from after the eyes have settled."""
    p = (int(frac[0] * W), int(frac[1] * H))
    t0 = time.perf_counter()
    F, E, D, HD = [], [], [], []
    while True:
        el = time.perf_counter() - t0
        if el >= seconds:
            break
        disp.poll()
        disp.clear(BG)
        pg.draw.circle(disp.screen, DOT, p, int(max(9, 30 - 34 * el)))
        pg.draw.circle(disp.screen, (255, 255, 255), p, 3)
        if label:
            disp.text(label, (W // 2, 30), size=23, color=INK, raw=True)
        disp.flip()
        o = next_obs()
        if o is not None and o.face and el > SETTLE_S and np.isfinite(o.feat).all():
            F.append(o.feat.copy())
            E.append(o.ear)
            D.append(o.dist_mm)
            HD.append((o.yaw, o.pitch))
    return F, E, D, HD


def face_check():
    """Waits until the face is found on 15 frames in a row, then for SPACE or a click."""
    seen, ready = 0, False
    disp.clicks.clear()
    while not ready:
        o = next_obs()
        if o is not None:
            seen = seen + 1 if o.face else 0
        ok = seen >= 15
        message(["Face found. Keep your head still, facing the screen. Press SPACE or click to start the dots." if ok else "Looking for your face... please face the camera.",
                 "Look at each red dot until the next one appears (about 40 seconds)."])
        for k, _ in disp.poll():
            if k == "space" and ok:
                ready = True
        if disp.clicks:
            ready = ready or ok
            disp.clicks.clear()


def iris_pose(F):
    ix, iy = 0.5 * (F[:, 0] + F[:, 2]), 0.5 * (F[:, 1] + F[:, 3])
    return np.column_stack([ix, iy, ix ** 2, iy ** 2, ix * iy, F[:, 5], F[:, 6]])


class SimpleModel:
    """Ridge on iris position and head angles only. No head-position columns: a small shift of the face in the image gave about 20 deg of error in the fuller model."""

    def __init__(self, F, T, alpha=1.0):
        X = iris_pose(F)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        A = np.column_stack([(X - self.mu) / self.sd, np.ones(len(X))])
        P = alpha * np.eye(A.shape[1])
        P[-1, -1] = 0
        self.W = np.linalg.solve(A.T @ A + P, A.T @ T)

    def predict(self, F):
        return np.column_stack([(iris_pose(np.atleast_2d(F)) - self.mu) / self.sd, np.ones(len(np.atleast_2d(F)))]) @ self.W


def calibrate_now():
    """14 dots, model, 5-dot accuracy check. Returns the calibration, or None if the face was not found often enough."""
    face_check()
    tgt, F, E, D, HD = [], [], [], [], []
    for pts in (random.sample(CAL_POINTS, len(CAL_POINTS)), random.sample(CAL2_POINTS, len(CAL2_POINTS))):
        for fr in pts:
            f, e, d, h = show_dot(fr)
            F += f
            E += e
            D += d
            HD += h
            tgt += [(fr[0] * W, fr[1] * H)] * len(f)
    if len(F) < 60:
        return None
    F_a, T_a = np.array(F), np.array(tgt, float)
    message(["Now keep your eyes on the red dot and slowly turn your head", "left and right, then a little up and down (15 seconds).", "Starts in 4 seconds."])
    t_wait = time.perf_counter()
    while time.perf_counter() - t_wait < 4.0:
        disp.poll()
        next_obs()
    Fh, Th = [], []
    for fr in [(0.5, 0.5), (0.25, 0.5), (0.75, 0.5)]:
        f, _, _, _ = show_dot(fr, 5.0, "Eyes on the dot. Slowly move your head.")
        Fh += f
        Th += [(fr[0] * W, fr[1] * H)] * len(f)
    model = SimpleModel(np.vstack([F_a, np.array(Fh)]), np.vstack([T_a, np.array(Th, float)])) if len(Fh) >= 15 else SimpleModel(F_a, T_a)
    # rescale per axis from the median estimate at each calibration dot to its target (vertical gain was 0.65 in the saved sessions)
    u, inv = np.unique(T_a, axis=0, return_inverse=True)
    inv = inv.ravel()
    pm = model.predict(np.array([np.median(F_a[inv == i], axis=0) for i in range(len(u))]))
    ax, ay = np.polyfit(pm[:, 0], u[:, 0], 1), np.polyfit(pm[:, 1], u[:, 1], 1)

    def predict(X):
        p = model.predict(X)
        return np.column_stack([np.polyval(ax, p[:, 0]), np.polyval(ay, p[:, 1])])
    dist = float(np.median(D))
    ppd = px_per_deg(dist, SCREEN_W_MM, W)
    yaw0, pitch0 = (float(v) for v in np.median(np.array(HD), axis=0))
    errs, off = [], []
    for fr in VAL_POINTS:
        f, _, _, _ = show_dot(fr)
        if len(f) >= 5:
            g = np.median(predict(np.array(f)), axis=0)
            dx, dy = (g[0] - fr[0] * W) / ppd, (g[1] - fr[1] * H) / ppd
            errs.append(float(np.hypot(dx, dy)))
            off.append((float(dx), float(dy)))
    return {"predict": predict, "rescale_x": round(float(ax[0]), 3), "rescale_y": round(float(ay[0]), 3), "ear": float(np.median(E)), "dist": dist, "ppd": ppd, "yaw0": yaw0, "pitch0": pitch0,
            "samples": len(F), "errs": errs, "acc": float(np.mean(errs)) if errs else float("nan"),
            "mean_shift_deg": [round(float(v), 2) for v in np.mean(off, axis=0)] if off else None}


def calibrate_accept():
    """Calibrates until accepted with SPACE (R repeats). Records every calibration in the summary."""
    while True:
        c = calibrate_now()
        if c is None:
            message(["The face was not found often enough. Press SPACE or click to try again."])
            disp.wait_key({"space"})
            continue
        summary["calibrations"].append({"samples": c["samples"], "distance_cm": round(c["dist"] / 10, 1), "px_per_deg": round(c["ppd"], 1),
                                        "validation_offsets_deg": [round(e, 2) for e in c["errs"]], "validation_mean_deg": round(c["acc"], 2),
                                        "validation_mean_shift_deg_x_y": c["mean_shift_deg"], "yaw0": round(c["yaw0"], 1), "pitch0": round(c["pitch0"], 1),
                                        "rescale_x": c["rescale_x"], "rescale_y": c["rescale_y"], "delivered_fps": round(cam.achieved_fps(), 1)})
        print(json.dumps(summary["calibrations"][-1]), flush=True)
        message(["Accuracy check: %.1f degrees on average, at %.0f cm." % (c["acc"], c["dist"] / 10),
                 "Next, look at the red dot in each orange shape. Then move normally: when your head settles somewhere new, one red dot appears; just look at it.",
                 "D = dot check   G = guided check   A = correct   N = more shapes   R = recalibrate   X = clear   SPACE = stop",
                 "Press SPACE or click to begin, or R to calibrate again."])
        if disp.wait_key({"space", "r"}) == "space":
            return c


KINDS = ["circle", "square", "triangle", "diamond", "star", "plus"]
GRIDS = {4: ((0.18, 0.82), (0.22, 0.78)), 6: ((0.18, 0.50, 0.82), (0.22, 0.78)), 9: ((0.18, 0.50, 0.82), (0.15, 0.50, 0.85)), 12: ((0.12, 0.37, 0.63, 0.88), (0.15, 0.50, 0.85))}
LEVELS = [4, 6, 9, 12]
GRIDS["after"] = ((0.34, 0.66), (0.35, 0.65, 0.85))             # second check: positions not used to make the correction


def new_check(shapes, stage=0):
    """Guided check over `shapes`. stage 1: correct automatically, then stage 2 measures the result at new positions; 0: started with G."""
    return {"i": 0, "t0": time.perf_counter(), "rec": [], "res": [], "order": random.sample(range(len(shapes)), len(shapes)), "shapes": shapes, "stage": stage}




def make_shapes(n):
    xs, ys = GRIDS[n]
    return [(KINDS[i % len(KINDS)], x, y) for i, (y, x) in enumerate((y, x) for y in ys for x in xs)]


SHAPE_R, HIT_R = 62, 110                                      # drawn size and how near the line must come for the shape to light up, px


def draw_shape(name, c, r, fill):
    x, y = c
    if name == "circle":
        pg.draw.circle(disp.screen, fill, c, r)
        pg.draw.circle(disp.screen, INK, c, r, 2)
        return
    if name == "square":
        pts = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
    elif name == "triangle":
        pts = [(x, y - r), (x + r, y + r), (x - r, y + r)]
    elif name == "diamond":
        pts = [(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
    elif name == "star":
        pts = [(x + (r if i % 2 == 0 else r * 0.42) * math.sin(i * math.pi / 5), y - (r if i % 2 == 0 else r * 0.42) * math.cos(i * math.pi / 5)) for i in range(10)]
    else:
        a = r * 0.34
        pts = [(x - a, y - r), (x + a, y - r), (x + a, y - a), (x + r, y - a), (x + r, y + a), (x + a, y + a), (x + a, y + r),
               (x - a, y + r), (x - a, y + a), (x - r, y + a), (x - r, y - a), (x - a, y - a)]
    pg.draw.polygon(disp.screen, fill, pts)
    pg.draw.polygon(disp.screen, INK, pts, 2)


summary = {"camera": "built-in", "index": CAM_INDEX, "calibrations": [], "guided_checks": [], "shift_applied_deg_x_y": []}
rows = []
try:
    message(["Live gaze line", "Sit comfortably, facing the screen. Press SPACE or click to continue."])
    disp.wait_key({"space"})
    cal = calibrate_accept()
    trace = pg.Surface((W, H), pg.SRCALPHA)
    fx, fy = OneEuro(), OneEuro()
    fixes = []                                                # straight-line corrections per axis, applied in order: ((a, b) for x, (a, b) for y)
    recent, head = deque(maxlen=3), deque(maxlen=5)
    pos, target, last = None, None, None
    last_seen = time.perf_counter()
    t_start = t_prev = time.perf_counter()
    n_frames = n_face = 0
    guard_s, guard_events, guard_on = 0.0, 0, False
    guard_text = ""
    level = 4
    SHAPES = make_shapes(level)
    near = {}
    guided, result_text, pending = None, "", None
    check, live_pts, live_fix = None, deque(maxlen=12), None  # dot checks while running: the app chooses the dot, never the gaze
    summary["dot_checks"] = []
    guided = new_check(make_shapes(9), 1)                     # after calibration: look at each shape when asked, then the app corrects itself
    err_first = None
    t_dot_order = []
    # offset per head position: each entry is (scaled yaw, pitch, face x, face y) and the offset measured there
    POSE_SCALE = np.array([6.0, 6.0, 0.015, 0.015])          # one unit = this much change counts as a new head position
    pose_recent, pose_map, t_last_auto = deque(maxlen=15), [], 0.0

    def pose_offset(v):
        if not pose_map:
            return np.zeros(2)
        d2 = np.array([np.sum(((v - e[0]) / POSE_SCALE) ** 2) for e in pose_map])
        w_ = 1.0 / (d2 + 0.1)
        return (w_[:, None] * np.array([e[1] for e in pose_map])).sum(0) / w_.sum()
    disp.clicks.clear()
    stop = False
    while not stop:
        for k, _ in disp.poll():
            if k == "space":
                stop = True
            elif k == "x":
                trace.fill((0, 0, 0, 0))
            elif k == "n" and guided is None:                 # next number of shapes: 4, 6, 9, 12, then back to 4
                level = LEVELS[(LEVELS.index(level) + 1) % len(LEVELS)]
                SHAPES = make_shapes(level)
                t_dot_order, check = [], None
                trace.fill((0, 0, 0, 0))
                result_text, pending = "%d shapes. Press G for the guided check." % level, None
            elif k == "c" and target is not None:
                fixes.append(((1.0, W / 2 - target[0]), (1.0, H / 2 - target[1])))
                summary["shift_applied_deg_x_y"].append(["C", round((W / 2 - target[0]) / cal["ppd"], 2), round((H / 2 - target[1]) / cal["ppd"], 2)])
            elif k == "d" and guided is None and check is None:
                if not t_dot_order:
                    t_dot_order = random.sample(range(len(SHAPES)), len(SHAPES))
                check = {"i": min(t_dot_order.pop(), len(SHAPES) - 1), "t0": time.perf_counter(), "rec": []}
            elif k == "g" and guided is None and check is None:
                guided = new_check(SHAPES)
                result_text, pending = "", None
            elif k == "a" and pending is not None:
                fixes.append(pending)
                summary["shift_applied_deg_x_y"].append(["A", [round(float(v), 3) for v in pending[0]], [round(float(v), 3) for v in pending[1]]])
                result_text, pending = "Correction applied. Press G to check it.", None
            elif k == "r":
                cal = calibrate_accept()
                trace.fill((0, 0, 0, 0))
                fx, fy = OneEuro(), OneEuro()
                fixes = []
                check, live_fix = None, None
                live_pts.clear()
                pose_map, pose_recent = [], deque(maxlen=15)
                recent.clear()
                head.clear()
                pos, target, last, guided, result_text, pending = None, None, None, new_check(make_shapes(9), 1), "", None
                err_first = None
                guard_on = False
                last_seen = t_prev = time.perf_counter()
        if disp.clicks:
            stop = True
        o = next_obs()
        now = time.perf_counter()
        if o is not None:
            n_frames += 1
            face_ok = bool(o.face) and np.isfinite(o.feat).all()
            dyaw = dpitch = dd = float("nan")
            if o.face:
                dyaw, dpitch, dd = o.yaw - cal["yaw0"], o.pitch - cal["pitch0"], o.dist_mm / cal["dist"] - 1.0
                head.append((dyaw, dpitch, dd))
                if np.isfinite(o.feat).all():
                    pose_recent.append(np.array([o.yaw, o.pitch, o.feat[8], o.feat[9]], float))
            if len(head) >= 3:                                # median of the last 5 frames, so one bad frame does not stop the line
                my, mp, md = np.median(np.array(head), axis=0)
                new_guard = abs(my) > GUARD_YAW or abs(mp) > GUARD_PITCH or abs(md) > GUARD_DIST
                if new_guard and not guard_on:
                    guard_events += 1
                guard_on = new_guard
                guard_text = "Head very far away; the line continues when it is nearer."
            ok = face_ok and o.ear > 0.65 * cal["ear"] and not guard_on    # no face, a blink or a moved head: no new point
            raw = (float("nan"), float("nan"))
            if ok:
                n_face += 1
                g = cal["predict"](o.feat[None])[0]
                for fxl, fyl in fixes:
                    g = np.array([np.polyval(fxl, g[0]), np.polyval(fyl, g[1])])
                g_pre = np.array(g, float)                    # before the running dot correction
                if len(pose_recent):
                    g = g_pre + pose_offset(np.median(np.array(pose_recent), axis=0))
                if check is not None and now - check["t0"] > 0.8:
                    check["rec"].append((float(g_pre[0]), float(g_pre[1])))
                raw = (float(g[0]), float(g[1]))
                recent.append(raw)                            # median of the last 3 estimates removes single-frame spikes
                m = np.median(np.array(recent), axis=0)
                mx, my_ = float(np.clip(m[0], -0.1 * W, 1.1 * W)), float(np.clip(m[1], -0.1 * H, 1.1 * H))
                target = (float(np.clip(fx(mx, now), 0, W - 1)), float(np.clip(fy(my_, now), 0, H - 1)))
                last_seen = now
                if guided is not None and now - guided["t0"] > GUIDED_SETTLE_S:
                    guided["rec"].append(raw)
            sm = target if target is not None else (float("nan"), float("nan"))
            rows.append([round(now - t_start, 3), round(raw[0], 1), round(raw[1], 1), round(sm[0], 1), round(sm[1], 1), int(ok), int(guard_on),
                         round(float(dyaw), 1), round(float(dpitch), 1), round(float(dd), 3)])
        if guard_on:
            guard_s += now - t_prev
        if guided is None and check is None and len(pose_recent) == 15:
            P_ = np.array(pose_recent)
            pv_ = np.median(P_, axis=0)
            if not pose_map:
                pose_map.append((pv_, np.zeros(2)))           # the position right after the guided checks
            elif np.all(np.ptp(P_, axis=0) / POSE_SCALE < 0.6) and now - t_last_auto > 5.0:      # head settled
                if min(np.sqrt(np.sum(((pv_ - e[0]) / POSE_SCALE) ** 2)) for e in pose_map) > 1.0:   # at a position not learned yet: one dot, no need to go back
                    check = {"i": random.randrange(len(SHAPES)), "t0": now, "rec": [], "auto": True}
                    t_last_auto = now
        if check is not None and now - check["t0"] >= 2.0:           # one dot check finished
            s = SHAPES[check["i"]]
            tgt = (s[1] * W, s[2] * H)
            if len(check["rec"]) >= 5:
                m = np.median(np.array(check["rec"]), axis=0)
                pv = np.median(np.array(pose_recent), axis=0)
                shown = m + pose_offset(pv)
                err = float(math.hypot(shown[0] - tgt[0], shown[1] - tgt[1]) / cal["ppd"])      # error BEFORE this check changes the correction
                pose_map.append((pv, np.array([tgt[0] - m[0], tgt[1] - m[1]])))
                summary["dot_checks"].append({"t_s": round(now - t_start, 1), "n_shapes": level, "shape": check["i"], "error_before_deg": round(err, 2),
                                              "dx_deg": round((shown[0] - tgt[0]) / cal["ppd"], 2), "dy_deg": round((shown[1] - tgt[1]) / cal["ppd"], 2), "n_points": len(pose_map), "auto": bool(check.get("auto")), "yaw": round(float(pv[0]), 1), "pitch": round(float(pv[1]), 1)})
                print(json.dumps(summary["dot_checks"][-1]), flush=True)
                result_text = "Dot check: error %.1f deg before correcting (%d positions learned). D = another." % (err, len(pose_map))
            else:
                result_text = "Dot check: not enough valid samples. D = try again."
            check = None
        if guided is not None and now - guided["t0"] >= GUIDED_S:     # next shape of the guided check
            idx = guided["order"][guided["i"]]
            s = guided["shapes"][idx]
            if len(guided["rec"]) >= 5:
                g = np.median(np.array(guided["rec"]), axis=0)
                chosen = int(np.argmin([math.hypot(g[0] - q[1] * W, g[1] - q[2] * H) for q in guided["shapes"]]))      # the shape nearest the gaze
                guided["res"].append((idx, float((g[0] - s[1] * W) / cal["ppd"]), float((g[1] - s[2] * H) / cal["ppd"]), chosen == idx))
            guided["i"] += 1
            guided["t0"], guided["rec"] = now, []
            if guided["i"] >= len(guided["order"]):
                res = guided["res"]
                if res:
                    dx, dy = float(np.mean([r[1] for r in res])), float(np.mean([r[2] for r in res]))
                    e = float(np.mean([math.hypot(r[1], r[2]) for r in res]))
                    tx = np.array([guided["shapes"][r[0]][1] * W for r in res])
                    ty = np.array([guided["shapes"][r[0]][2] * H for r in res])
                    n_ok = sum(1 for r in res if r[3])
                    mx_ = tx + np.array([r[1] for r in res]) * cal["ppd"]
                    my2 = ty + np.array([r[2] for r in res]) * cal["ppd"]
                    gain_x, gain_y = None, None
                    if len(set(tx)) >= 2 and len(set(ty)) >= 2:
                        curved = len(set(ty)) >= 3                  # three rows: a curved vertical correction for the weak lower screen
                        pending = (np.polyfit(mx_, tx, 1), np.polyfit(my2, ty, 2 if curved else 1))
                        gain_x, gain_y = float(1 / pending[0][0]), None if curved else float(1 / pending[1][0])
                        by_row = {}
                        for r_, yy in zip(res, ty):
                            by_row.setdefault(round(yy / H, 2), []).append(math.hypot(r_[1], r_[2]))
                        summary.setdefault("error_by_row_deg", []).append({"stage": guided["stage"], **{str(k_): round(float(np.mean(v_)), 2) for k_, v_ in sorted(by_row.items())}})
                    result_text = "%d shapes: %d of %d correct, error %.1f deg, on average %+.1f right, %+.1f down. A = correct, G = check, N = next." % (level, n_ok, len(res), e, dx, dy)
                    summary["guided_checks"].append({"stage": guided["stage"], "n_shapes": len(guided["shapes"]), "n_correct": n_ok, "n_measured": len(res), "per_shape_deg_dx_dy_correct": [[r[0], round(r[1], 2), round(r[2], 2), bool(r[3])] for r in res], "mean_error_deg": round(e, 2),
                                                     "mean_shift_deg_x_y": [round(dx, 2), round(dy, 2)], "gain_x": gain_x, "gain_y": gain_y, "n_corrections_before": len(fixes)})
                    print(json.dumps(summary["guided_checks"][-1]), flush=True)
                else:
                    result_text = "Guided check: not enough valid samples."
                stage = guided["stage"]
                guided = None
                if stage == 1 and res:                        # correct from the first check, then measure at new positions
                    err_first = e
                    if pending is not None:
                        fixes.append(pending)
                        summary["shift_applied_deg_x_y"].append(["auto", [round(float(v), 3) for v in pending[0]], [round(float(v), 3) for v in pending[1]]])
                    pending = None
                    guided = new_check(make_shapes("after"), 2)
                elif stage == 2 and res:
                    pending = None
                    result_text = "Ready. Error %.1f deg before correcting, %.1f deg after (new positions). Look at the shapes. D = check, N = more shapes." % (err_first, e)
                    summary["error_before_after_deg"] = [round(err_first, 2), round(e, 2)]
        if target is not None and not guard_on:
            pos = list(target) if pos is None else [pos[0] + 0.4 * (target[0] - pos[0]), pos[1] + 0.4 * (target[1] - pos[1])]
            q = (int(pos[0]), int(pos[1]))
            if now - last_seen > 0.4:
                last = None                                   # gap in tracking: break the line
            elif last is not None and last != q:
                pg.draw.line(trace, LINE, last, q, 2)
                last = q
            else:
                last = q
        if guard_on:
            last = None
        disp.clear(BG)
        active = guided["order"][guided["i"]] if guided is not None else None
        for i_s, (name, fx_, fy_) in enumerate(guided["shapes"] if guided is not None else SHAPES):
            c = (int(fx_ * W), int(fy_ * H))
            hit = pos is not None and not guard_on and math.hypot(pos[0] - c[0], pos[1] - c[1]) < HIT_R
            if hit:
                key_s = "%d:%d" % (level, i_s)
                near[key_s] = near.get(key_s, 0.0) + now - t_prev
            fill = TARGET if i_s == active else (LIT if hit and guided is None else PALE)
            draw_shape(name, c, SHAPE_R, fill)
            is_target = i_s == active or (check is not None and check["i"] == i_s)
            if is_target:                                     # a dot only while checking; plain shapes otherwise
                pg.draw.circle(disp.screen, DOT, c, 8)
        t_prev = now
        pg.draw.line(disp.screen, RING, (W // 2 - 16, H // 2), (W // 2 + 16, H // 2), 2)
        pg.draw.line(disp.screen, RING, (W // 2, H // 2 - 16), (W // 2, H // 2 + 16), 2)
        disp.screen.blit(trace, (0, 0))
        if pos is not None:
            pg.draw.circle(disp.screen, DOT, (int(pos[0]), int(pos[1])), 9)
        if guard_on:
            disp.text(guard_text, (W // 2, 26), size=20, color=DOT, raw=True)
        elif now - last_seen > 0.4:
            disp.text("face or eyes not found", (W // 2, 26), size=22, color=DOT, raw=True)
        elif check is not None:
            disp.text("Look at the red dot.", (W // 2, 26), size=22, color=INK, raw=True)
        elif guided is not None:
            disp.text("Guided check: look at the red dot in the orange shape.", (W // 2, 26), size=22, color=INK, raw=True)
        elif result_text:
            disp.text(result_text, (W // 2, 26), size=20, color=INK, raw=True)
        disp.flip()
    dur = time.perf_counter() - t_start
    summary.update(live_seconds=round(dur, 1), live_fps=round(n_frames / max(dur, 1e-6), 1), live_valid_percent=round(100 * n_face / max(n_frames, 1), 1),
                   guard_seconds=round(guard_s, 1), guard_events=guard_events, seconds_near_shape={k: round(v, 1) for k, v in near.items()})
    pg.image.save(disp.screen, os.path.join(OUT, "gaze_line.png"))
    with open(os.path.join(OUT, "gaze_line.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "raw_x", "raw_y", "smooth_x", "smooth_y", "valid", "guard", "yaw_change", "pitch_change", "dist_change_frac"])
        w.writerows(rows)
    disp.text("Stopped after %.0f s. Press SPACE or click to close." % dur, (W // 2, H - 26), size=22, color=INK, raw=True)
    disp.flip()
    disp.wait_key({"space"})
except KeyboardInterrupt:
    summary["stopped_with"] = "Esc"
finally:
    if rows and not os.path.exists(os.path.join(OUT, "gaze_line.csv")):         # also keep the numbers when stopped with Esc
        with open(os.path.join(OUT, "gaze_line.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t_s", "raw_x", "raw_y", "smooth_x", "smooth_y", "valid", "guard", "yaw_change", "pitch_change", "dist_change_frac"])
            w.writerows(rows)
    with open(os.path.join(OUT, "gaze_line_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary), flush=True)
    cam.stop()
    tr.close()
    disp.close()
