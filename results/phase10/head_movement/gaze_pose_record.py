"""Records one fixed session for offline work on head movement: 9 known dots at each of 7 head positions, on a grey screen. There is no live gaze line and no model.
Camera frames are processed in memory and discarded. Saved: numbers only (eye and head features, target positions); no camera image and no picture. Esc stops and keeps what was recorded."""
import csv, json, os, random, sys, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "./app")
import numpy as np
import cv2
from camera import CameraThread
from display import Display
from landmarks import FaceTracker, FEATURE_NAMES

OUT = os.path.dirname(os.path.abspath(__file__))
STAMP = time.strftime("%H%M%S")
BG, INK, DOT = (128, 128, 128), (15, 15, 15), (235, 235, 235)
DOT_S, SETTLE_S, PROMPT_S = 2.0, 0.7, 5.0
GRID = [(x, y) for y in (0.12, 0.5, 0.88) for x in (0.12, 0.5, 0.88)]
POSES = [("straight", "Sit normally, facing the screen."),
         ("left", "Turn your head a little to the LEFT and keep it there."),
         ("right", "Turn your head a little to the RIGHT and keep it there."),
         ("up", "Tilt your head a little UP and keep it there."),
         ("down", "Tilt your head a little DOWN and keep it there."),
         ("closer", "Lean a little CLOSER to the screen and stay there."),
         ("straight_end", "Sit normally again, facing the screen.")]

disp = Display(fullscreen=True)
disp.max_fps = 30.0
pg = disp.pg
disp.screen = pg.display.set_mode((1280, 720), pg.FULLSCREEN | pg.SCALED, vsync=1)
disp.size = disp.screen.get_size()
disp.ui = 1.0
disp._bring_to_front()
W, H = disp.size
_c = cv2.VideoCapture(0, cv2.CAP_DSHOW)
_c.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
for _ in range(5):
    _c.read()
_c.release()
cam = CameraThread(index=0, width=640, height=480, fps=30)
cam.start()
cam.opened.wait(10)
if cam.error:
    disp.close()
    sys.exit(cam.error)
tr = FaceTracker(focal_px=529.0)
rows, summary = [], {"camera": "built-in", "glasses": None, "screen_px": [W, H], "screen_width_mm": 294.0, "poses": [p[0] for p in POSES]}
t_zero = time.perf_counter()


def next_obs():
    fr = cam.get(0.004)
    if fr is None:
        return None
    idx, t_ns, bgr = fr
    o = tr.process(bgr, t_ns, idx)
    del bgr, fr                                               # discard the frame
    return o


def screen(lines):
    disp.clear(BG)
    y = H // 2 - 30 * len(lines)
    for i, s in enumerate(lines):
        disp.text(s, (W // 2, y + i * 56), size=25, color=INK, raw=True)
    disp.flip()


def ask(lines, keys):
    """Keeps redrawing until a key is pressed: one drawing can be lost while the window is still appearing on a busy laptop."""
    disp._bring_to_front()
    while True:
        screen(lines)
        next_obs()
        for k, _ in disp.poll():
            k = k.strip("[]") if len(k) == 3 and k.startswith("[") else k
            if k in keys:
                return k


try:
    summary["glasses"] = {"1": True, "2": False}[ask(["Recording for the head-movement work (about 3 minutes).", "Are you wearing glasses now?  1 = Yes   2 = No"], {"1", "2"})]
    seen, ready = 0, False
    disp.clicks.clear()
    while not ready:
        o = next_obs()
        if o is not None:
            seen = seen + 1 if o.face else 0
        ok = seen >= 10
        screen(["Face found. Press SPACE or click to start." if ok else "Looking for your face...", "Hold each head position and look at each dot until the next one appears."])
        for k, _ in disp.poll():
            if k == "space" and ok:
                ready = True
        if disp.clicks:
            ready = ready or ok
            disp.clicks.clear()
    for pose, text in POSES:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < PROMPT_S:
            disp.poll()
            screen([text, "Dots start in %d seconds." % max(1, int(PROMPT_S - (time.perf_counter() - t0)) + 1)])
            next_obs()
        for fr in random.sample(GRID, len(GRID)):
            p = (int(fr[0] * W), int(fr[1] * H))
            t1 = time.perf_counter()
            while True:
                el = time.perf_counter() - t1
                if el >= DOT_S:
                    break
                disp.poll()
                disp.clear(BG)
                pg.draw.circle(disp.screen, DOT, p, int(max(9, 30 - 34 * el)))
                pg.draw.circle(disp.screen, INK, p, 3)
                disp.flip()
                o = next_obs()
                if o is None:
                    continue
                good = bool(o.face) and np.isfinite(o.feat).all()
                rows.append([round(time.perf_counter() - t_zero, 3), pose, p[0], p[1], int(el > SETTLE_S), int(good), round(float(o.img_lum), 1)] +
                            ([round(float(o.ear), 4), round(float(o.yaw), 2), round(float(o.pitch), 2), round(float(o.roll), 2), round(float(o.dist_mm), 1)] + [round(float(v), 5) for v in o.feat] if good else [""] * (5 + len(FEATURE_NAMES))))
    summary["complete"] = True
    summary["delivered_fps"] = round(cam.achieved_fps(), 1)
except KeyboardInterrupt:
    summary["stopped_with"] = "Esc"
finally:
    with open(os.path.join(OUT, "pose_rec_%s.csv" % STAMP), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "pose", "target_x", "target_y", "settled", "ok", "lum", "ear", "yaw", "pitch", "roll", "dist_mm"] + FEATURE_NAMES)
        w.writerows(rows)
    with open(os.path.join(OUT, "pose_rec_%s.json" % STAMP), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary), "rows", len(rows), flush=True)
    cam.stop()
    tr.close()
    try:
        t_end = time.perf_counter()
        done = False
        while not done and time.perf_counter() - t_end < 30:
            screen(["Finished. Thank you. Press SPACE to close."])
            time.sleep(0.03)
            done = any(k == "space" for k, _ in disp.poll())
    except KeyboardInterrupt:
        pass
    disp.close()
