"""Phase 7, Track D: no-pupil webcam simulation (D1) and face-area size sensitivity (D2).
Pre-registration: results/phase7/trackD/PREREG.md (written before any result).

Reuses the Track C harness (scripts/phase6_trackC_c1.py, imported unchanged): noise model (bit-identical to
edd.webcam_noise.apply_noise), sample-based features, table building, frozen-fold nested CV.

Stages (each skips what is on disk):
  check  - harness reproduction (0.7723 / 0.7854) and AOI-patching checks; run this FIRST
  run    - features + frozen-fold nested CV for every D1/D2 configuration (checkpoints in the scratch folder)

Usage (project root, at most 2 workers):
  OMP_NUM_THREADS=1 .venv/Scripts/python.exe -I scripts/phase7_trackD_run.py --stages check
  OMP_NUM_THREADS=1 .venv/Scripts/python.exe -I scripts/phase7_trackD_run.py --stages run --shard 0/2
"""
import argparse
import contextlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location("trackC_c1", ROOT / "scripts" / "phase6_trackC_c1.py")
C1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C1)
dr, df_, M = C1.dr, C1.df_, C1.M

OUT = ROOT / "results" / "phase7" / "trackD"
SCRATCH = Path(os.environ.get("TRACKD_SCRATCH", r"<local path>"))
CK = SCRATCH / "ck"
LEVELS, DRAWS, FS = C1.LEVELS, C1.DRAWS, C1.FS
PUPIL = ("pupil_face_z", "pupil_trial_z")
FEATS = {("ivt", "wp"): C1.FEATS["ivt"], ("sb", "wp"): C1.FEATS["sb"]}
FEATS[("ivt", "np")] = [f for f in FEATS[("ivt", "wp")] if f not in PUPIL]
FEATS[("sb", "np")] = [f for f in FEATS[("sb", "wp")] if f not in PUPIL]
assert len(FEATS[("ivt", "wp")]) == 13 and len(FEATS[("sb", "wp")]) == 12
assert len(FEATS[("ivt", "np")]) == 11 and len(FEATS[("sb", "np")]) == 10
FULL_RECTS = dr.QUAD_RECTS.copy()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def scaled_rects(s):
    r = FULL_RECTS.copy()
    cx, cy = (r[:, 0] + r[:, 2]) / 2, (r[:, 1] + r[:, 3]) / 2
    hw, hh = (r[:, 2] - r[:, 0]) / 2 * s, (r[:, 3] - r[:, 1]) / 2 * s
    return np.column_stack([cx - hw, cy - hh, cx + hw, cy + hh])


@contextlib.contextmanager
def aoi_scale(s):
    """Temporarily replace the four face rectangles by their shrunken versions (scale of width and height around the
    centre). Patches the module-level bounds read by `delmas_fast.aoi_batch` and `delmas_raw.aoi_of`."""
    R = scaled_rects(s)
    old = (dr.QUAD_RECTS, df_._X0, df_._X1, df_._X2, df_._X3, df_._Y0, df_._Y1, df_._Y2, df_._Y3)
    dr.QUAD_RECTS = R
    df_._X0, df_._X1, df_._X2, df_._X3 = R[0, 0], R[0, 2], R[1, 0], R[1, 2]
    df_._Y0, df_._Y1, df_._Y2, df_._Y3 = R[0, 1], R[0, 3], R[2, 1], R[2, 3]
    try:
        yield R
    finally:
        (dr.QUAD_RECTS, df_._X0, df_._X1, df_._X2, df_._X3, df_._Y0, df_._Y1, df_._Y2, df_._Y3) = old


def stag(s):
    return f"s{int(round(s * 100)):03d}"


def tag_of(v, pup, s, cond, L, d):
    return f"{v}_{pup}_{stag(s)}_{cond}_L{C1.lv(L)}_d{d}"


def d1_units():
    out = [(1.0, "clean", 0.0, 0)]
    for L in LEVELS:
        out += [(1.0, "all", L, d) for d in DRAWS]
    out += [(1.0, "anch", "A", d) for d in DRAWS]
    return out


def d2_units():
    out = []
    for s in (0.75, 0.5):
        out.append((s, "clean", 0.0, 0))
        out += [(s, "anch", "A", d) for d in DRAWS]
    return out


def unit_jobs():
    """(scale, cond, L, draw, [(route, pupil)...])."""
    jobs = [(s, c, L, d, [("ivt", "np"), ("sb", "np")]) for s, c, L, d in d1_units()]
    jobs += [(s, c, L, d, [("ivt", "np"), ("sb", "np"), ("ivt", "wp"), ("sb", "wp")]) for s, c, L, d in d2_units()]
    return jobs


def all_tags():
    return [(tag_of(v, p, s, c, L, d), v, p, s, c, L, d) for s, c, L, d, vp in unit_jobs() for v, p in vp]


def build(segs, cond, L, draw, routes, bk):
    """Feature tables for the routes needed ('ivt' / 'sb') with whatever AOI is currently patched in."""
    ivt_arr, sb_arr = {}, {}
    for pid, seg in segs.items():
        x, y, p = C1.noisy(seg["x30"], seg["y30"], seg["p30"], cond, L, draw, pid)
        if "ivt" in routes:
            ivt_arr[pid] = (df_.trial_features_batch(x, y, p, FS, with_msacc=False), seg["images"], seg["group"])
        if "sb" in routes:
            sb_arr[pid] = (C1.sb_features_batch(x, y, p), seg["images"], seg["group"])
    out = {}
    if ivt_arr:
        out["ivt"] = C1.make_table(ivt_arr, dr.TRIAL_FEATURES, bk)
    if sb_arr:
        out["sb"] = C1.make_table(sb_arr, C1.SB_NAMES, bk)
    return out


def get_track():
    track = M.load_track()
    track.CKPT = CK
    return track


def stage_run(shard):
    assert (OUT / "PREREG.md").exists()
    assert (OUT / "checks.json").exists() and json.loads((OUT / "checks.json").read_text())["all_passed"], "run `check` first"
    CK.mkdir(parents=True, exist_ok=True)
    track = get_track()
    frozen = json.loads((C1.TRACK / "folds.json").read_text())["repeats"]
    segs, bk = C1.load_cache(), C1.books()
    i0, n = shard
    for ui, (s, cond, L, d, vp) in enumerate(unit_jobs()):
        if ui % n != i0:
            continue
        need = [(v, p) for v, p in vp if not (CK / f"{tag_of(v, p, s, cond, L, d)}.json").exists()]
        if not need:
            continue
        t0 = time.time()
        with aoi_scale(s):
            tabs = build(segs, cond, L, d, sorted({v for v, _ in need}), bk)
        log(f"unit scale={s} {cond} L={L} d={d}: features {time.time() - t0:.0f}s for {need}")
        for v, p in need:
            tag = tag_of(v, p, s, cond, L, d)
            t1 = time.time()
            res = track.run_grid(tabs[v], frozen, FEATS[(v, p)], "lr", FS, tag, {}, {})
            log(f"  {tag}: mean AUC {res['mean_over_sets']['image']['auc']:.4f} ({time.time() - t1:.0f}s)")


def pooled_auc(f):
    """Non-inferiority AUC definition: repeat-averaged OOF scores, AUC pooled within each set, mean over sets."""
    from sklearn.metrics import roc_auc_score
    d = pd.read_csv(f).groupby(["set", "participant", "image"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))
    return float(np.mean([roc_auc_score(g.y, g.score) for _, g in d.groupby("set")]))


def load_oof_sorted(f):
    return pd.read_csv(f).sort_values(["set", "repeat", "participant", "image"]).reset_index(drop=True)


def stage_check():
    res = {}
    segs, bk = C1.load_cache(), C1.books()
    # (0) AOI geometry
    res["aoi_deg"] = {str(s): {"width_px": float(scaled_rects(s)[0, 2] - scaled_rects(s)[0, 0]),
                               "height_px": float(scaled_rects(s)[0, 3] - scaled_rects(s)[0, 1]),
                               "width_deg": float((scaled_rects(s)[0, 2] - scaled_rects(s)[0, 0]) / dr.PIX_PER_DEG),
                               "height_deg": float((scaled_rects(s)[0, 3] - scaled_rects(s)[0, 1]) / dr.PIX_PER_DEG),
                               "centres_px": ((scaled_rects(s)[:, :2] + scaled_rects(s)[:, 2:]) / 2).tolist()}
                      for s in (1.0, 0.75, 0.5)}
    lay = json.loads((ROOT / "results" / "phase5" / "samples" / "delmas_layout.json").read_text())
    rect = lay["face_rectangles_px"]
    res["layout_matches_QUAD_RECTS"] = bool(np.allclose(
        FULL_RECTS, [[rect[k]["x0"], rect[k]["y0"], rect[k]["x1"], rect[k]["y1"]] for k in ("upper_left", "upper_right", "lower_left", "lower_right")]))
    res["layout_px_per_degree_equals_code"] = bool(abs(lay["px_per_degree"] - dr.PIX_PER_DEG) < 1e-9)
    res["scale1_patch_is_identity"] = True
    with aoi_scale(1.0) as R:
        res["scale1_patch_is_identity"] = bool(np.array_equal(R, FULL_RECTS))
    # (1) vectorised vs reference implementations at scales 0.75, 0.5 (sample-based and I-VT)
    for s in (0.75, 0.5):
        with aoi_scale(s):
            mx_sb = mx_iv = 0.0
            for pid in list(segs)[:2]:
                seg = segs[pid]
                x, y, p = C1.noisy(seg["x30"], seg["y30"], seg["p30"], "all", 1.5, 1, pid)
                Fv = C1.sb_features_batch(x[:40], y[:40], p[:40])
                Fi = df_.trial_features_batch(x[:40], y[:40], p[:40], FS, with_msacc=False)
                for k in range(40):
                    Fr = C1.sb_reference_trial(x[k], y[k], p[k])
                    both = np.isfinite(Fv[k]) & np.isfinite(Fr)
                    assert (np.isfinite(Fv[k]) == np.isfinite(Fr)).all(), ("sb", s, pid, k)
                    mx_sb = max(mx_sb, float(np.abs(Fv[k][both] - Fr[both]).max()))
                    Fo = dr.trial_features(x[k], y[k], p[k], FS, with_msacc=False)
                    both = np.isfinite(Fi[k]) & np.isfinite(Fo)
                    assert (np.isfinite(Fi[k]) == np.isfinite(Fo)).all(), ("ivt", s, pid, k)
                    mx_iv = max(mx_iv, float(np.abs(Fi[k][both] - Fo[both]).max()))
            res[f"scale{s}_sb_vectorised_vs_reference_max_abs_diff"] = mx_sb
            res[f"scale{s}_ivt_batch_vs_unbatched_max_abs_diff"] = mx_iv
    # (1b) shrinking really changes the features (sanity)
    seg = segs[2]
    with aoi_scale(0.5):
        a = C1.sb_features_batch(seg["x30"][:20], seg["y30"][:20], seg["p30"][:20])
    b = C1.sb_features_batch(seg["x30"][:20], seg["y30"][:20], seg["p30"][:20])
    res["scale0.5_changes_sb_features"] = bool(np.nanmax(np.abs(a - b)) > 1e-6)
    # (2) harness reproduction: clean, full-size AOI, with pupil; OOF equal to the stored Track C files
    CK.mkdir(parents=True, exist_ok=True)
    track = get_track()
    frozen = json.loads((C1.TRACK / "folds.json").read_text())["repeats"]
    stored = {"ivt": C1.WEB / "checkpoints" / "ALLz_fixed_lr_30_L000_d0_oof.csv.gz",
              "sb": C1.SCRATCH / "c1_ck" / "sb_clean_L000_d0_oof.csv.gz"}
    with aoi_scale(1.0):
        tabs = build(segs, "clean", 0.0, 0, ["ivt", "sb"], bk)
    for v in ("ivt", "sb"):
        tag = f"chk_{v}_wp_clean"
        r = track.run_grid(tabs[v], frozen, FEATS[(v, "wp")], "lr", FS, tag, {}, {})
        a = load_oof_sorted(CK / f"{tag}_oof.csv.gz")
        b = load_oof_sorted(stored[v])
        key = ["set", "repeat", "participant", "image", "y"]
        same = len(a) == len(b) and (a[key].to_numpy() == b[key].to_numpy()).all()
        res[f"harness_{v}_clean_with_pupil"] = {"mean_auc_image": pooled_auc(CK / f"{tag}_oof.csv.gz"),
                                                "mean_of_fold_aucs_run_grid": r["mean_over_sets"]["image"]["auc"], "same_rows": bool(same),
                                                "max_abs_score_diff": float(np.abs(a.score.to_numpy() - b.score.to_numpy()).max()) if same else None}
    res["expected_auc"] = {"sb": 0.7723, "ivt": 0.7854}
    ok = (res["layout_matches_QUAD_RECTS"] and res["layout_px_per_degree_equals_code"] and res["scale1_patch_is_identity"]
          and res["scale0.5_changes_sb_features"])
    ok = ok and all(res[f"scale{s}_{k}"] < 1e-9 for s in (0.75, 0.5) for k in ("sb_vectorised_vs_reference_max_abs_diff", "ivt_batch_vs_unbatched_max_abs_diff"))
    for v in ("ivt", "sb"):
        h = res[f"harness_{v}_clean_with_pupil"]
        ok = ok and h["same_rows"] and h["max_abs_score_diff"] < 1e-9 and abs(h["mean_auc_image"] - res["expected_auc"][v]) < 5e-5
    res["all_passed"] = bool(ok)
    C1.jdump(OUT / "checks.json", res)
    log("checks:", json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="check")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CK.mkdir(parents=True, exist_ok=True)
    st = a.stages.split(",")
    i0, n = (int(v) for v in a.shard.split("/"))
    if "check" in st:
        stage_check()
    if "run" in st:
        stage_run((i0, n))


if __name__ == "__main__":
    main()
