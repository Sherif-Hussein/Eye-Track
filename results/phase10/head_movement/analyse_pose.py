"""Exploratory, one person: which gaze model handles head movement? Every error is on head positions NOT used for fitting.
Input: a recording of 9 known dots at 7 head positions (numbers only). Output: tables printed and saved as CSV next to the recording."""
import json, os, sys
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "./app")
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from calibrate import GazeModel
from landmarks import FEATURE_NAMES as FN

base = sys.argv[1]
d = pd.read_csv(base + ".csv")
meta = json.load(open(base + ".json", encoding="utf-8"))
W, H = meta["screen_px"]
PX_PER_MM = W / meta["screen_width_mm"]
d = d[(d.ok == 1) & (d.settled == 1)].copy()
d = d[d.ear >= 0.7 * d.groupby("pose").ear.transform(lambda s: s.quantile(0.85))]
POSES = [p for p in meta["poses"] if p in set(d.pose)]
TEST_ONLY = "straight_end"
print("glasses:", meta.get("glasses"), "| delivered fps:", meta.get("delivered_fps"), "| complete:", meta.get("complete"))
s0 = d[d.pose == "straight"]
yaw0, pitch0, roll0, dist0, cx0, cy0 = s0.yaw.median(), s0.pitch.median(), s0.roll.median(), s0.dist_mm.median(), s0.cx.median(), s0.cy.median()
t = d.groupby("pose", sort=False).agg(frames=("t_s", "size"), yaw=("yaw", "median"), pitch=("pitch", "median"), roll=("roll", "median"), dist_mm=("dist_mm", "median"), cx=("cx", "median"), cy=("cy", "median"), ear=("ear", "median"))
t["yaw_change"] = t.yaw - yaw0
t["pitch_change"] = t.pitch - pitch0
t["dist_change_pct"] = 100 * (t.dist_mm / dist0 - 1)
print(t.round(3).to_string())
ppd = float(np.tan(np.radians(1)) * dist0 * PX_PER_MM)
print("px per deg %.1f" % ppd)


def iris(f):
    return 0.5 * (f.ix_r.to_numpy() + f.ix_l.to_numpy()), 0.5 * (f.iy_r.to_numpy() + f.iy_l.to_numpy())


def x_iris(f):
    a, b = iris(f)
    return np.column_stack([a, b, a ** 2, b ** 2, a * b])


def x_pose(f):
    return np.column_stack([x_iris(f), f.yaw, f.pitch])


def x_pose_int(f):
    a, b = iris(f)
    return np.column_stack([x_pose(f), a * f.yaw, b * f.pitch, a * f.pitch, b * f.yaw])


def x_pose_roll(f):
    return np.column_stack([x_pose(f), f.roll])


def x_full(f):
    return np.column_stack([x_pose_roll(f), f.cx, f.cy, f.dist])


def x_full_int(f):
    a, b = iris(f)
    return np.column_stack([x_full(f), a * f.yaw, b * f.pitch, a * f.pitch, b * f.yaw])


def ridge(mk, alpha):
    def fit(rows):
        X, Y = mk(rows), rows[["target_x", "target_y"]].to_numpy(float)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        A = np.column_stack([(X - mu) / sd, np.ones(len(X))])
        P = alpha * np.eye(A.shape[1])
        P[-1, -1] = 0
        w = np.linalg.solve(A.T @ A + P, A.T @ Y)
        return lambda f: np.column_stack([(mk(f) - mu) / sd, np.ones(len(f))]) @ w
    return fit


def app_m8(rows):
    m = GazeModel(kind="m8").fit(rows[FN].to_numpy(float), rows[["target_x", "target_y"]].to_numpy(float))
    return lambda f: m.predict(f[FN].to_numpy(float))


def geo(rows):
    """Head angle (gain limited to 0.4-1.3 in size) + eye-in-head angle, then distance x tan(angle) + head position."""
    def pred(th, f):
        a, b = iris(f)
        dp = f.dist_mm.to_numpy() * PX_PER_MM
        ax = np.radians(th[0] * (f.yaw.to_numpy() - yaw0) + th[1] * a + th[2] * b + th[3])
        ay = np.radians(th[4] * (f.pitch.to_numpy() - pitch0) + th[5] * b + th[6] * a + th[7])
        return np.column_stack([W / 2 + th[8] * (f.cx.to_numpy() - cx0) * dp + dp * np.tan(ax), H / 2 + th[9] * (f.cy.to_numpy() - cy0) * dp + dp * np.tan(ay)])
    T = rows[["target_x", "target_y"]].to_numpy(float)
    best = None
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            lo = [min(.4 * sx, 1.3 * sx), -500, -200, -40, min(.4 * sy, 1.3 * sy), -500, -200, -40, -3, -3]
            hi = [max(.4 * sx, 1.3 * sx), 500, 200, 40, max(.4 * sy, 1.3 * sy), 500, 200, 40, 3, 3]
            r = least_squares(lambda th: (pred(th, rows) - T).ravel() / ppd, np.array([sx, 100, 0, 0, sy, 100, 0, 0, 0, 0.0]), bounds=(lo, hi), loss="soft_l1", f_scale=2.0, max_nfev=300)
            if best is None or r.cost < best.cost:
                best = r
    th = best.x
    return lambda f: pred(th, f)


MODELS = {"current app model": app_m8, "iris only": ridge(x_iris, 1.0), "iris + yaw, pitch": ridge(x_pose, 1.0), "iris + yaw, pitch, interactions": ridge(x_pose_int, 1.0),
          "iris + pose + position": ridge(x_full, 1.0), "all + interactions": ridge(x_full_int, 1.0), "geometric": geo}


def dot_points(p, f):
    out = []
    for (x, y), g in f.groupby(["target_x", "target_y"]):
        if len(g) >= 5:
            m = np.median(p(g), axis=0)
            out.append((m[0], m[1], x, y))
    return np.array(out)


def score(P):
    return float(np.mean(np.hypot(P[:, 0] - P[:, 2], P[:, 1] - P[:, 3])) / ppd)


def one_dot(P):
    """Error at this head position after one known dot there sets a shift, for each dot in turn; tested on the other dots."""
    e = []
    for i in range(len(P)):
        s = P[i, 2:4] - P[i, :2]
        Q = np.delete(P, i, axis=0)
        e.append(np.mean(np.hypot(Q[:, 0] + s[0] - Q[:, 2], Q[:, 1] + s[1] - Q[:, 3])) / ppd)
    return float(np.mean(e))


rows = []
train_pool = [p for p in POSES if p != TEST_ONLY]
for name, fit in MODELS.items():
    # still calibration: fit on 'straight' only
    p0 = fit(d[d.pose == "straight"])
    for pose in POSES:
        if pose == "straight":
            continue
        P = dot_points(p0, d[d.pose == pose])
        rows.append({"model": name, "training": "straight only", "test_pose": pose, "error": score(P), "one_dot": one_dot(P)})
    # many poses: leave the test position out
    for pose in POSES:
        if pose == TEST_ONLY:
            tr = d[d.pose.isin(train_pool)]
        else:
            tr = d[d.pose.isin([q for q in train_pool if q != pose])]
        if pose == "straight":
            continue
        P = dot_points(fit(tr), d[d.pose == pose])
        rows.append({"model": name, "training": "other positions", "test_pose": pose, "error": score(P), "one_dot": one_dot(P)})
r = pd.DataFrame(rows)
r.to_csv(base + "_models.csv", index=False)
print("\nError in degrees on a head position not used for fitting")
for tr_name, g in r.groupby("training", sort=False):
    print("\nTrained on:", tr_name)
    pv = g.pivot(index="model", columns="test_pose", values="error").reindex(MODELS).round(2)
    pv = pv[[c for c in POSES if c in pv.columns]]
    pv["mean"] = pv.mean(1).round(2)
    print(pv.to_string())
    pv1 = g.pivot(index="model", columns="test_pose", values="one_dot").reindex(MODELS).round(2)
    pv1 = pv1[[c for c in POSES if c in pv1.columns]]
    pv1["mean"] = pv1.mean(1).round(2)
    print("  ...after one known dot at that position (error on the other dots):")
    print(pv1.to_string())

best = r[r.training == "other positions"].groupby("model").one_dot.mean().idxmin()
print("\nLowest mean after one dot, trained on other positions:", best)
print("How many positions does calibration need? (model above; test: 'straight_end', and each position not trained on)")
fit = MODELS[best]
sets = {"straight": ["straight"], "straight + left, right": ["straight", "left", "right"], "straight + up, down": ["straight", "up", "down"], "straight + left, right, up, down": ["straight", "left", "right", "up", "down"], "all six": train_pool}
out = []
for nm, tr in sets.items():
    p = fit(d[d.pose.isin(tr)])
    te = [q for q in POSES if q not in tr]
    vals = {q: score(dot_points(p, d[d.pose == q])) for q in te}
    out.append({"trained on": nm, "dots": 9 * len(tr), "mean on unseen": round(float(np.mean(list(vals.values()))), 2), **{q: round(v, 2) for q, v in vals.items()}})
print(pd.DataFrame(out).to_string(index=False))
print("\nBy screen row, same model, trained on other positions: signed vertical error (deg; negative = estimate above the target)")
rw = []
for pose in POSES:
    if pose == "straight":
        continue
    tr = d[d.pose.isin(train_pool if pose == TEST_ONLY else [q for q in train_pool if q != pose])]
    P = dot_points(fit(tr), d[d.pose == pose])
    for row in sorted(set(P[:, 3])):
        q = P[P[:, 3] == row]
        rw.append({"pose": pose, "row": round(row / H, 2), "dy": float(np.mean(q[:, 1] - q[:, 3]) / ppd), "err": float(np.mean(np.hypot(q[:, 0] - q[:, 2], q[:, 1] - q[:, 3])) / ppd)})
rw = pd.DataFrame(rw)
print(rw.pivot(index="pose", columns="row", values="dy").round(2).to_string())
print(rw.groupby("row")[["dy", "err"]].mean().round(2).to_string())
