"""Task M (model search) library: data bundles, extra features, candidate models, nested-CV engine.
EXPLORATORY. Development datasets only (Delmas; Chen). The held-out dataset is never touched.

Design (all choices fixed in PLAN.md before any full run):
  * The unit of evaluation is the same as the frozen pipeline: Delmas = image-level rows (participant, image) of the six group sets,
    frozen folds (results/phase2/delmas_raw/folds.json, 5 participant-disjoint folds x 3 repeats); Chen = trial rows, frozen folds
    (results/phase3/chen_confirm/folds.json, 10 folds x 3 repeats).
  * Every candidate is a Spec with fit(bundle, train_rows, hyper, seed) -> predictor(rows) -> scores. One engine (run_fold) tunes the
    hyper-parameters by participant-grouped inner folds INSIDE the training participants (mean inner-fold AUC, first maximum wins), refits,
    scores the test participants, and keeps the inner out-of-fold scores at the chosen setting (for thresholds, voting, stacking).
  * The frozen reference F0 is run through the same engine; it must reproduce the stored scores exactly (checked by ms_run.py repro).
Imports the project code UNCHANGED; writes nothing outside results/phase10/model_search/.
"""
import importlib.util
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "results" / "phase9" / "low_rate")]
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import lr_common as LRC  # noqa: E402  (thread env vars; read-only helpers: load_oof, paired_bootstrap)
from edd import delmas_fast as df_  # noqa: E402
from edd import delmas_raw as dr  # noqa: E402
from edd import delmas_raw_matched as M  # noqa: E402
from edd import unified as U  # noqa: E402

CK = OUT / "ck"
CACHE = ROOT / "results" / "phase9" / "low_rate" / "cache"
DERIVED = ROOT / "results" / "phase9" / "low_rate" / "derived"
BOOK64 = ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx"
BOOK32 = ROOT / "data" / "delmas" / "BookFIP_32_trials.xlsx"
SETS = ["64_g1", "64_g2", "64_g3", "32_g3", "32_g4", "32_g5"]
SETDEF = {"64_g1": ("64", 1), "64_g2": ("64", 2), "64_g3": ("64", 3), "32_g3": ("32", 3), "32_g4": ("32", 4), "32_g5": ("32", 5)}
C_SK = [0.01, 0.1, 1.0, 10.0]                       # the Delmas track's grid (scripts/phase2_delmas_raw.py model_factory 'lr')
SCAN_NAMES = ["visit_order", "is_first", "is_last", "n_visits", "n_returns", "mean_run_ms", "max_run_ms", "trans_out", "trans_in",
              "trans_with_first", "trans_with_last", "trans_entropy_out"]
NBIN = 50                                           # 100 ms bins over the 5 s display


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================= small helpers
def zwithin(M_, part):
    """Within-participant z-score of each column (pandas convention: ddof=1, +1e-9), NaN-aware. Label-free. Vectorised over participants."""
    M_ = np.asarray(M_, float)
    u, inv = np.unique(part, return_inverse=True)
    ok = np.isfinite(M_)
    A = np.where(ok, M_, 0.0)
    n = np.zeros((len(u), M_.shape[1]))
    s1 = np.zeros_like(n)
    np.add.at(n, inv, ok)
    np.add.at(s1, inv, A)
    mu = s1 / np.maximum(n, 1)
    D = np.where(ok, M_ - mu[inv], 0.0)
    s2 = np.zeros_like(n)
    np.add.at(s2, inv, D * D)
    sd = np.sqrt(s2 / np.maximum(n - 1, 1))
    out = (M_ - mu[inv]) / (sd[inv] + 1e-9)
    return np.where(ok, out, np.nan)


def agg_to_rows(F, rowmap, nrow):
    """Mean (NaN-aware) of trial-item rows F (N, nf) over the rows mapped to the same image row; rowmap < 0 = not in the book."""
    F = np.asarray(F, float)
    sel = rowmap >= 0
    ok = np.isfinite(F[sel])
    s = np.zeros((nrow, F.shape[1]))
    c = np.zeros((nrow, F.shape[1]))
    np.add.at(s, rowmap[sel], np.where(ok, F[sel], 0.0))
    np.add.at(c, rowmap[sel], ok)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


# ============================================================================= sample-based extra features (Delmas)
def scan_trial(lab, fs):
    """Scan-path features of the 4 items of one trial from the AOI label of each valid sample (-1 = background), in time order."""
    out = np.zeros((4, len(SCAN_NAMES)))
    out[:, 0] = 5.0
    if len(lab) == 0:
        return out
    chg = np.flatnonzero(np.diff(lab) != 0) + 1
    st, en = np.r_[0, chg], np.r_[chg, len(lab)]
    rl, rlen = lab[st], (en - st) * 1000.0 / fs
    keep = rl >= 0
    seq, slen = rl[keep], rlen[keep]
    if len(seq) == 0:
        return out
    first, last = int(seq[0]), int(seq[-1])
    order = []
    for a in seq:
        if int(a) not in order:
            order.append(int(a))
    a_, b_ = seq[:-1], seq[1:]
    d = a_ != b_
    a_, b_ = a_[d], b_[d]
    for k in range(4):
        m = seq == k
        nv = int(m.sum())
        if nv:
            out[k, 0] = order.index(k) + 1
            out[k, 1] = float(k == first)
            out[k, 2] = float(k == last)
            out[k, 3] = nv
            out[k, 4] = nv - 1
            out[k, 5] = slen[m].mean()
            out[k, 6] = slen[m].max()
        out[k, 7] = float((a_ == k).sum())
        out[k, 8] = float((b_ == k).sum())
        dest = b_[a_ == k]
        if len(dest) > 1:
            p = np.bincount(dest, minlength=4) / len(dest)
            p = p[p > 0]
            out[k, 11] = float(-(p * np.log2(p)).sum())
        if k != first:
            out[k, 9] = float(((a_ == k) & (b_ == first)).sum() + ((a_ == first) & (b_ == k)).sum())
        if k != last:
            out[k, 10] = float(((a_ == k) & (b_ == last)).sum() + ((a_ == last) & (b_ == k)).sum())
    return out


def sample_blocks(x, y, fs):
    """(T, n) samples starting 500 ms before onset -> per trial and item: 100 ms occupancy time course (T,4,50) and scan-path (T,4,12).
    Occupancy of a bin = samples inside the item's area / valid samples (0 when the bin has no valid sample)."""
    T, n = x.shape
    t = np.arange(n) * 1000.0 / fs - dr.PRE_MS
    win = (t >= 0) & (t < dr.POST_MS)
    q = df_.aoi_batch(x, y)
    valid = np.isfinite(x) & np.isfinite(y)
    qw, vw, tw = q[:, win], valid[:, win], t[win]
    Bm = np.zeros((len(tw), NBIN))
    Bm[np.arange(len(tw)), np.minimum((tw // 100).astype(int), NBIN - 1)] = 1.0
    nv = vw.astype(float) @ Bm
    tc = np.zeros((T, 4, NBIN))
    scan = np.zeros((T, 4, len(SCAN_NAMES)))
    for k in range(4):
        ins = ((qw == k) & vw).astype(float) @ Bm
        tc[:, k] = np.where(nv > 0, ins / np.maximum(nv, 1), 0.0)
    for i in range(T):
        scan[i] = scan_trial(qw[i][vw[i]], fs)
    return tc, scan


def catch22_rows(series, n_chunks=1):
    """series (N, ch, L) -> (N, 22*ch) catch22 via aeon (NaN replaced by 0 by aeon; non-finite input replaced by 0)."""
    from aeon.transformations.collection.feature_based import Catch22
    s = np.nan_to_num(np.asarray(series, float), nan=0.0, posinf=0.0, neginf=0.0)
    return Catch22(replace_nans=True, n_jobs=1).fit_transform(s)


# ============================================================================= bundles
class Bundle:
    pass


def _base_bundle_common(B):
    B._cache = {}
    return B


def load_participant_samples(pid, fs):
    z = np.load(CACHE / f"p{pid:02d}.npz")
    return z[f"x{fs}"].astype(float), z[f"y{fs}"].astype(float), z[f"p{fs}"].astype(float), z["images"], int(z["group"])


class DelmasCond:
    """One Delmas condition: frozen image-level table + per-participant trial-level arrays + extra blocks.
    name D30 = clean 30 Hz, 13 standardised features (ALL_z without microsaccades) as 'ALLz_lr_30';
    name W30 / W15 = measured webcam noise ('anch'), sample-based dwell without pupil (10 features), draw d = 1..3 (the Phase 7 construction)."""

    def __init__(self, name, draw=None, want=("scan", "tc"), log_=log):
        self.name, self.draw = name, draw
        self.fs = 30 if name in ("D30", "W30") else 15
        self.rate = self.fs
        self.trials = {}
        with np.errstate(all="ignore"):
            if name == "D30":
                self._build_clean(want, log_)
            else:
                self._build_web(want, log_)

    def _pids(self):
        from edd.delmas import load_trials
        b = load_trials(BOOK64)
        return {int(p): int(g) for p, g in b.groupby("participant").group.first().items()}

    def _finish_blocks(self, pid, x, y, extra):
        tc, scan = sample_blocks(x, y, self.fs)
        self.trials[pid].update({"tc": tc, "scan": scan, **extra})

    def _build_clean(self, want, log_):
        track = M.load_track()
        track.N_JOBS = 1
        self.df = track.build_tables(DERIVED, BOOK64, BOOK32)
        self.feats = [f for f in track.FEATURE_SETS["ALL_z"] if f != "msacc_rate_z"]
        self.base_names = [f[:-2] for f in self.feats]
        face = pd.read_csv(DERIVED / "delmas_raw_face_trials.csv.gz")
        face = face[face.rate == 30].sort_values(["participant", "trial", "quad"])
        for pid in self._pids():
            x, y, p, images, g = load_participant_samples(pid, 30)
            f = face[face.participant == pid]
            T = len(images)
            assert len(f) == T * 4 and (f.image.to_numpy().reshape(T, 4) == images).all()
            self.trials[pid] = {"images": images, "group": g, "base": f[self.base_names].to_numpy(float).reshape(T, 4, -1)}
            self._finish_blocks(pid, x, y, {})
        self.track = track

    def _build_web(self, want, log_):
        spec = importlib.util.spec_from_file_location("trackC_c1", ROOT / "scripts" / "phase6_trackC_c1.py")
        C1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(C1)
        self.C1 = C1
        C1.FS = self.fs
        nonp = [n for n in C1.SB_NAMES if n not in ("pupil_face", "pupil_trial", "n_seg")]
        assert len(nonp) == 10
        cols = [C1.SB_NAMES.index(n) for n in nonp]
        arrays = {}
        for pid in self._pids():
            x, y, p, images, g = load_participant_samples(pid, self.fs)
            xn, yn, pn = C1.noisy(x, y, p, "anch", "A", self.draw, pid)
            F = C1.sb_features_batch(xn, yn, pn)
            arrays[pid] = (F, images, g)
            self.trials[pid] = {"images": images, "group": g, "base": F[:, :, cols]}
            self._finish_blocks(pid, xn, yn, {})
        self.df = C1.make_table(arrays, C1.SB_NAMES, C1.books())
        self.feats = [f"{n}_z" for n in nonp]
        self.base_names = nonp


def delmas_set_bundle(cond, sk, want=("scan", "tc", "c22")):
    s, g = SETDEF[sk]
    d = cond.df[(cond.df.set == s) & (cond.df.group == g) & (cond.df.rate == cond.rate)].reset_index(drop=True)
    B = Bundle()
    B._cache = {}
    B.kind, B.key, B.cond = "delmas", sk, cond.name
    B.y = d.face.to_numpy(int)
    B.part = d.participant.to_numpy(int)
    B.image = d.image.to_numpy(int)
    B.n = len(d)
    B.X = {"base": d[cond.feats].to_numpy(float)}
    parts = sorted(pd.unique(B.part))
    t0 = 32 if s == "32" else 0
    t_part, t_img, tb, ttc, tsc = [], [], [], [], []
    for p in parts:
        tr_ = cond.trials[p]
        T = len(tr_["images"])
        idx = np.arange(t0, T)
        t_part.append(np.full(len(idx) * 4, p))
        t_img.append(tr_["images"][idx].reshape(-1))
        tb.append(tr_["base"][idx].reshape(len(idx) * 4, -1))
        ttc.append(tr_["tc"][idx].reshape(len(idx) * 4, -1))
        tsc.append(tr_["scan"][idx].reshape(len(idx) * 4, -1))
    B.t_part, B.t_img = np.concatenate(t_part), np.concatenate(t_img)
    tb, ttc, tsc = np.vstack(tb), np.vstack(ttc), np.vstack(tsc)
    key = pd.MultiIndex.from_arrays([B.part, B.image])
    B.t_rowmap = key.get_indexer(pd.MultiIndex.from_arrays([B.t_part, B.t_img]))
    B.t_Xz = zwithin(tb, B.t_part)                       # trial-level frozen features, z within participant over the set's trial items
    B.X["scan"] = zwithin(agg_to_rows(tsc, B.t_rowmap, B.n), B.part)
    B.X["tc"] = agg_to_rows(ttc, B.t_rowmap, B.n)
    if "c22" in want:
        c = catch22_rows(ttc.reshape(-1, 1, NBIN))
        B.X["c22"] = zwithin(agg_to_rows(c, B.t_rowmap, B.n), B.part)
    B.t_raw_base = tb
    # displays: 4 consecutive rows per trial; trainable = exactly one known item
    known = np.zeros(len(B.t_rowmap), bool)
    sel = B.t_rowmap >= 0
    known[sel] = B.y[B.t_rowmap[sel]] == 1
    kn = known.reshape(-1, 4)
    B.d_ok = kn.sum(1) == 1
    B.d_k = kn.argmax(1)
    B.d_part = B.t_part[::4]
    B.nd = len(B.d_ok)
    B.folds = [[[int(u) for u in f] for f in rep] for rep in json.loads((LRC.STORED / "folds.json").read_text())["repeats"][sk]]
    B.seed_fn = lambda r, f: r
    B.inner_k = lambda n_units: 4
    B.auc_fn = roc_auc_score
    B.grid_default = C_SK
    B.grid_pca = C_SK
    B.grid_en = C_SK
    B.l1_grid = (0.2, 0.5, 0.8)
    B.cv_mean = np.mean
    return B


def chen_bundle(want=("tc", "c22")):
    tasks, (meta, ch) = U.build_chen(lambda *a: None)
    b = tasks["chen_unified"]
    B = Bundle()
    B._cache = {}
    B.kind, B.key, B.cond = "chen", "chen", "C30"
    B.y, B.part = b["y"].astype(int), np.asarray(b["g"]).astype(str)
    B.n = len(B.y)
    B.X = {"base": b["X"], "tc": b["X"]}
    if "c22" in want:
        ser = np.stack([U.decimate_to_rate(ch[c], 30) for c in U.CHEN_CHANNELS], axis=1)      # (n, 3, 75)
        B.X["c22"] = zwithin(catch22_rows(ser), B.part)
    B.chen = b
    B.folds = b["folds"]
    B.seed_fn = lambda r, f: 1000 * r + f
    B.inner_k = U.inner_k
    B.auc_fn = U.fast_auc
    B.grid_default = list(U.C_GRID)
    B.grid_pca = list(U.C_GRID)
    B.grid_en = [U.C_GRID[0], U.C_GRID[2], U.C_GRID[4]]      # 0.001, 0.01, 0.1: larger C makes saga very slow on 7,742 rows
    B.l1_grid = (0.2, 0.8)
    B.cv_mean = np.nanmean
    B.cnn_shape = (3, 25)
    return B


# ============================================================================= candidate specs
def sk_pipe(C, l1=None):
    if l1 is None:
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=C)
    else:
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=C, solver="saga", l1_ratio=l1, tol=1e-3)
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", clf)])


def _xmat(B, blocks):
    k = tuple(blocks)
    if k not in B._cache:
        B._cache[k] = np.hstack([B.X[b] for b in blocks])
    return B._cache[k]


def lr_predictor(X, tr, y, C, flavor):
    """L2 logistic regression on rows `tr` of X; flavor 'sk' = the Delmas track's pipeline, 'uni' = the unified (Chen) pipeline."""
    if flavor == "sk":
        m = sk_pipe(C).fit(X[tr], y[tr])
        return lambda rows: m.decision_function(X[rows])
    P = U.prep_fit(X[tr])
    m = U.fit_lr(U.prep_apply(P, X[tr]), y[tr], C)
    return lambda rows: U.prep_apply(P, X[rows]) @ m.coef_[0] + float(m.intercept_[0])


class LRSpec:
    """L2 logistic regression on the listed feature blocks."""

    def __init__(self, name, blocks, flavor):
        self.name, self.blocks, self.flavor = name, blocks, flavor
        self.inner = True

    def grid(self, B):
        return B.grid_default

    def fit(self, B, tr, hp, seed):
        return lr_predictor(_xmat(B, self.blocks), tr, B.y, hp, self.flavor)

    def n_params(self, B):
        return _xmat(B, self.blocks).shape[1] * 4 + 1


class PCASpec:
    """PCA (k components, fitted on the training people's per-image 100 ms time courses only) -> within-person z -> L2 LR; optional base features."""

    def __init__(self, name, with_base, flavor):
        self.name, self.with_base, self.flavor = name, with_base, flavor
        self.inner = True

    def grid(self, B):
        return [(k, C) for k in (3, 4, 5, 6, 7, 8) for C in B.grid_pca]

    def fit(self, B, tr, hp, seed):
        k, C = hp
        key = ("pca", len(tr), int(tr[0]), int(tr[-1]), int(tr.sum()), k)
        if key not in B._cache:
            if len(B._cache) > 400:
                B._cache.clear()
            tc = np.nan_to_num(B.X["tc"])
            skey = key[:-1]
            if skey not in B._cache:
                mu = tc[tr].mean(0)
                B._cache[skey] = (mu, np.linalg.svd(tc[tr] - mu, full_matrices=False)[2])
            mu, Vt = B._cache[skey]
            B._cache[key] = zwithin((tc - mu) @ Vt[:k].T, B.part)
        Z = B._cache[key]
        X = np.hstack([B.X["base"], Z]) if self.with_base else Z
        return lr_predictor(X, tr, B.y, C, self.flavor)

    def n_params(self, B):
        nb = B.X["base"].shape[1] if self.with_base else 0
        return B.X["tc"].shape[1] * 9 + 8 * 4 + nb * 4 + 1


class ENSpec:
    """Elastic-net logistic regression (saga), same features and preprocessing as the frozen model."""

    def __init__(self, name, blocks):
        self.name, self.blocks = name, blocks
        self.inner = True

    def grid(self, B):
        return [(C, l1) for C in B.grid_en for l1 in B.l1_grid]

    def fit(self, B, tr, hp, seed):
        C, l1 = hp
        X = _xmat(B, self.blocks)
        m = sk_pipe(C, l1).fit(X[tr], B.y[tr])
        return lambda rows: m.decision_function(X[rows])

    def n_params(self, B):
        return _xmat(B, self.blocks).shape[1] * 4 + 1


class RFSpec:
    """Random forest, 100 shallow trees (depth 2 or 4, chosen in the inner folds), min leaf 5, balanced subsample, same features."""

    def __init__(self, name, blocks):
        self.name, self.blocks = name, blocks
        self.inner = True

    def grid(self, B):
        return [2, 4]

    def fit(self, B, tr, hp, seed):
        X = _xmat(B, self.blocks)
        m = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("clf", RandomForestClassifier(n_estimators=100, max_depth=hp, min_samples_leaf=5, max_features="sqrt",
                                                     class_weight="balanced_subsample", random_state=seed, n_jobs=1))]).fit(X[tr], B.y[tr])
        return lambda rows: m.predict_proba(X[rows])[:, 1] - 0.5

    def n_params(self, B):
        return 100 * (2 ** 4 - 1) * 2


def _prep_trial(B, tr_disp_items):
    """Median-impute and standardise trial-item features with training items only."""
    X = B.t_Xz
    med = np.nanmedian(tr_disp_items, 0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xi = np.where(np.isnan(tr_disp_items), med, tr_disp_items)
    mu, sd = Xi.mean(0), Xi.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return lambda A: (np.where(np.isnan(A), med, A) - mu) / sd


def cl_fit(Xd, k, C):
    """Conditional logit: softmax over the 4 items of each display, trained on which item is known. L2 penalty 1/(2C)."""
    D, _, p = Xd.shape
    ar = np.arange(D)

    def f(w):
        s = Xd @ w
        m = s.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(s - m).sum(1))
        P = np.exp(s - lse[:, None])
        loss = (lse - s[ar, k]).sum() + 0.5 / C * (w @ w)
        P[ar, k] -= 1.0
        return loss, np.einsum("di,dip->p", P, Xd) + w / C
    r = minimize(f, np.zeros(p), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return r.x


class DesignSpec:
    """Display-aware models on trial-item rows: 'cl' conditional logit (softmax over the 4 items); 'pw' pairwise logistic on known-minus-unknown
    differences (3 pairs per display, both signs, no intercept). Trained on displays with exactly one known item, scored on every trial item,
    image score = mean over the image's presentations."""

    def __init__(self, name, kind):
        self.name, self.kind = name, kind
        self.inner = True

    def grid(self, B):
        return B.grid_default

    def fit(self, B, tr, hp, seed):
        parts = np.unique(B.part[tr])
        dm = B.d_ok & np.isin(B.d_part, parts)
        Xd = B.t_Xz.reshape(B.nd, 4, -1)[dm]
        kd = B.d_k[dm]
        prep = _prep_trial(B, Xd.reshape(-1, Xd.shape[2]))
        Xs = prep(Xd.reshape(-1, Xd.shape[2])).reshape(Xd.shape)
        if self.kind == "cl":
            w = cl_fit(Xs, kd, hp)
        else:
            ar = np.arange(len(kd))
            kn = Xs[ar, kd]
            diffs = np.concatenate([kn - Xs[ar, (kd + j) % 4] for j in (1, 2, 3)])
            Xp = np.vstack([diffs, -diffs])
            yp = np.r_[np.ones(len(diffs)), np.zeros(len(diffs))]
            w = LogisticRegression(C=hp, fit_intercept=False, max_iter=2000).fit(Xp, yp).coef_[0]
        st = prep(B.t_Xz) @ w
        simg = agg_to_rows(st[:, None], B.t_rowmap, B.n)[:, 0]
        simg = np.where(np.isfinite(simg), simg, 0.0)
        return lambda rows: simg[rows]

    def n_params(self, B):
        return B.t_Xz.shape[1] * 3


class TinyCNN:
    """1-D CNN in numpy: conv(F filters, width K) -> ReLU -> [global mean, global max] -> linear. Weighted logistic loss, Adam, fixed epochs."""

    def __init__(self, ch, L, F=8, K=5, seed=0):
        rng = np.random.default_rng(seed)
        self.ch, self.L, self.F, self.K = ch, L, F, K
        self.W1 = rng.normal(0, np.sqrt(2.0 / (ch * K)), (F, ch * K))
        self.b1 = np.zeros(F)
        self.W2 = rng.normal(0, 0.3, 2 * F)
        self.b2 = 0.0

    @property
    def n_params(self):
        return int(self.W1.size + self.b1.size + self.W2.size + 1)

    def _win(self, X):
        w = sliding_window_view(X, self.K, axis=2)
        return w.transpose(0, 2, 1, 3).reshape(X.shape[0], w.shape[2], self.ch * self.K)

    def _fwd(self, X):
        Wn = self._win(X)
        A = Wn @ self.W1.T + self.b1
        R = np.maximum(A, 0.0)
        H = np.concatenate([R.mean(1), R.max(1)], 1)
        return Wn, A, R, H, H @ self.W2 + self.b2

    def decision(self, X):
        return self._fwd(X)[4]

    def fit(self, X, y, epochs=40, bs=64, lr=0.01, wd=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        n = len(y)
        w = np.where(y == 1, 0.5 / max((y == 1).sum(), 1), 0.5 / max((y == 0).sum(), 1)) * n
        params = [self.W1, self.b1, self.W2]
        m = [np.zeros_like(p) for p in params] + [0.0]
        v = [np.zeros_like(p) for p in params] + [0.0]
        t = 0
        for _ in range(epochs):
            for idx in np.array_split(rng.permutation(n), max(1, n // bs)):
                Wn, A, R, H, z = self._fwd(X[idx])
                dz = (1 / (1 + np.exp(-z)) - y[idx]) * w[idx] / len(idx)
                F, Lo = self.F, R.shape[1]
                dW2 = H.T @ dz + wd * self.W2
                db2 = dz.sum()
                dH = dz[:, None] * self.W2[None, :]
                dR = np.repeat(dH[:, None, :F] / Lo, Lo, axis=1)
                am = R.argmax(1)
                dR[np.arange(len(idx))[:, None], am, np.arange(F)[None, :]] += dH[:, F:]
                dA = dR * (A > 0)
                dW1 = np.einsum("nlf,nlk->fk", dA, Wn) + wd * self.W1
                db1 = dA.sum((0, 1))
                grads = [dW1, db1, dW2, db2]
                t += 1
                for i, g in enumerate(grads):
                    if i < 3:
                        m[i] = 0.9 * m[i] + 0.1 * g
                        v[i] = 0.999 * v[i] + 0.001 * g * g
                        params[i] -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
                    else:
                        m[3] = 0.9 * m[3] + 0.1 * g
                        v[3] = 0.999 * v[3] + 0.001 * g * g
                        self.b2 -= lr * (m[3] / (1 - 0.9 ** t)) / (np.sqrt(v[3] / (1 - 0.999 ** t)) + 1e-8)
        return self


class CNNSpec:
    """One-try tiny CNN: Delmas input = per-image 100 ms occupancy time course (1 x 50); Chen input = the 3 channel x 25 bin features (3 x 25).
    No hyper-parameter search (single fixed setting); no inner folds, so threshold-based metrics and ensembles do not use it."""

    def __init__(self, name):
        self.name = name
        self.inner = False

    def grid(self, B):
        return [None]

    def _x(self, B):
        if "cnn_x" not in B._cache:
            if B.kind == "delmas":
                X = np.nan_to_num(B.X["tc"])[:, None, :]
            else:
                X = np.nan_to_num(B.X["tc"]).reshape(B.n, *B.cnn_shape)
            B._cache["cnn_x"] = X
        return B._cache["cnn_x"]

    def fit(self, B, tr, hp, seed):
        X = self._x(B)
        mu, sd = X[tr].mean((0, 2), keepdims=True), X[tr].std((0, 2), keepdims=True) + 1e-9
        Xs = (X - mu) / sd
        net = TinyCNN(X.shape[1], X.shape[2], seed=seed).fit(Xs[tr], B.y[tr], seed=seed)
        self.n_par = net.n_params
        return lambda rows: net.decision(Xs[rows])

    def n_params(self, B):
        return TinyCNN(self._x(B).shape[1], self._x(B).shape[2]).n_params


def spec_registry(kind):
    """Single-source list of the pre-specified candidates (see PLAN.md). kind 'delmas' | 'chen'."""
    fl = "sk" if kind == "delmas" else "uni"
    R = {"F0": LRSpec("F0", ["base"], fl)}
    if kind == "delmas":
        R["M1_cl"] = DesignSpec("M1_cl", "cl")
        R["M2_pw"] = DesignSpec("M2_pw", "pw")
        R["F3a_scan"] = LRSpec("F3a_scan", ["scan"], fl)
        R["F3b_scan+base"] = LRSpec("F3b_scan+base", ["base", "scan"], fl)
    R["F4a_pca"] = PCASpec("F4a_pca", False, fl)
    if kind == "delmas":
        R["F4b_pca+base"] = PCASpec("F4b_pca+base", True, fl)
    R["F5a_c22"] = LRSpec("F5a_c22", ["c22"], fl)
    R["F5b_c22+base"] = LRSpec("F5b_c22+base", ["base", "c22"], fl)
    R["C6_enet"] = ENSpec("C6_enet", ["base"])
    R["C7_rf"] = RFSpec("C7_rf", ["base"])
    R["C8_cnn"] = CNNSpec("C8_cnn")
    return R


# ============================================================================= engine
def _auc(B, y, s):
    if len(np.unique(y)) < 2:
        return np.nan
    return B.auc_fn(y, s)


def run_fold(spec, B, tr, te, seed):
    """Nested step: tune on the training participants' inner folds, refit, score test. Returns the fold record."""
    grid = spec.grid(B)
    rec = {"tr": tr, "te": te}
    if spec.inner and len(grid) > 1:
        cv = StratifiedGroupKFold(B.inner_k(len(np.unique(B.part[tr]))), shuffle=True, random_state=seed)
        inner = list(cv.split(tr, B.y[tr], B.part[tr]))
        oof = np.full((len(tr), len(grid)), np.nan)
        aucs = np.full((len(inner), len(grid)), np.nan)
        for fi, (a, b) in enumerate(inner):
            for j, hp in enumerate(grid):
                s = spec.fit(B, tr[a], hp, seed)(tr[b])
                oof[b, j] = s
                aucs[fi, j] = _auc(B, B.y[tr[b]], s)
        cvm = B.cv_mean(aucs, axis=0)
        j = int(np.argmax(np.where(np.isfinite(cvm), cvm, -np.inf)))
        rec["inner_oof"], rec["cv_auc"] = oof[:, j], cvm
    else:
        j = 0
        rec["inner_oof"], rec["cv_auc"] = None, None
    pred = spec.fit(B, tr, grid[j], seed)
    rec["score"] = pred(te)
    rec["hp"] = grid[j]
    return rec


def subsample_participants(tr, B, frac, rseed):
    ug = np.unique(B.part[tr])
    keep = np.random.default_rng(rseed).choice(ug, max(4, int(round(frac * len(ug)))), replace=False)
    return tr[np.isin(B.part[tr], keep)]


def run_set(spec, B, repeats=(0, 1, 2), folds_limit=None, frac=1.0):
    out = {}
    for r in repeats:
        out[r] = []
        for f, test_units in enumerate(B.folds[r]):
            if folds_limit is not None and f >= folds_limit:
                break
            te_m = np.isin(B.part, test_units)
            te, tr = np.flatnonzero(te_m), np.flatnonzero(~te_m)
            if frac < 1.0:
                tr = subsample_participants(tr, B, frac, 7000 + 1000 * r + f)
            out[r].append(run_fold(spec, B, tr, te, B.seed_fn(r, f)))
    return out


def ck_path(cond_tag, cand, sk, frac=1.0):
    d = CK / cond_tag / cand
    d.mkdir(parents=True, exist_ok=True)
    return d / (f"{sk}.pkl" if frac == 1.0 else f"{sk}__f{int(round(frac * 100))}.pkl")


def save_ck(path, obj):
    tmp = Path(str(path) + ".tmp")
    pickle.dump(obj, open(tmp, "wb"))
    tmp.replace(path)


def oof_frame(B, outs, sk=None):
    """Test-fold scores in the (set, repeat, participant, image, y, score) layout of the stored OOF files."""
    rows = []
    for r, folds in outs.items():
        for d in folds:
            te = d["te"]
            rows.append(pd.DataFrame({"set": sk or B.key, "repeat": r, "participant": B.part[te], "image": B.image[te] if hasattr(B, "image") else te,
                                      "y": B.y[te], "score": d["score"]}))
    return pd.concat(rows, ignore_index=True)


# ============================================================================= person level (Delmas CIT statistic)
def _centre(v, how):
    from scipy.stats import trim_mean
    if how == "mean":
        return float(np.mean(v))
    if how == "trim20":
        return float(trim_mean(v, 0.2))
    return float(np.median(v))


def _spread(v, how):
    if how == "median":
        return float(1.4826 * np.median(np.abs(v - np.median(v))))
    return float(np.std(v))


def cit_z(s, y, rng, how="mean"):
    """U._cit_z with the centre/spread of the person statistic selectable. how='mean' is identical to U._cit_z."""
    sf, su = s[y == 1], s[y == 0]
    zk = (_centre(sf, how) - _centre(su, how)) / (_spread(su, how) + 1e-9)
    pick = rng.choice(len(su), len(sf), replace=False)
    rest = np.delete(su, pick)
    zi = (_centre(su[pick], how) - _centre(rest, how)) / (_spread(rest, how) + 1e-9)
    return float(zk), float(zi)


def persons_cit2(B, outs, how="mean", score_key="score", oof_key="inner_oof"):
    """Same layout and seeds as U.persons_cit; thresholds are set only when the fold record has inner out-of-fold scores."""
    g, y = B.part, B.y
    rows = []
    for r, folds in outs.items():
        for f, d in enumerate(folds):
            th = {"tau_eer": np.nan, "tau_fpr7": np.nan}
            if d.get(oof_key) is not None:
                pos, neg = [], []
                tr_g = g[d["tr"]]
                for u in np.unique(tr_g):
                    m = tr_g == u
                    zk, zi = cit_z(d[oof_key][m], y[d["tr"]][m], np.random.default_rng(500 + int(u)), how)
                    pos.append(zk)
                    neg.append(zi)
                th = U.thresholds(pos, neg)
            te_g = g[d["te"]]
            for u in np.unique(te_g):
                m = te_g == u
                zk, zi = cit_z(d[score_key][m], y[d["te"]][m], np.random.default_rng(r + int(u)), how)
                for kind, yy, z in (("known", 1, zk), ("sim_unknown", 0, zi)):
                    rows.append({"repeat": r, "fold": f, "unit": int(u), "kind": kind, "y": yy, "stat": z, **th})
    return pd.DataFrame(rows)
