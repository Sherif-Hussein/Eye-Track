"""Preti et al. (2026) fake-identity lying dataset: loaders, trial/participant features, and a
participant-disjoint nested-CV engine with inner-tuned trial-to-participant aggregation.

Data facts established in Phase 2 (see results/phase2/preti/comparison_card.md):
- `raw data` is fixation-level (one row per fixation, ~7 per trial, 32 trials per subject:
  C1-C8 control, E1-E8 expected, U1-U16 unexpected). RT and KEYPRESS are constant within a trial.
- CONDITION 0 = liar (slower RTs, flat pupil profile), 1 = truth-teller. Positive class = liar.
- `pupil_corrected`: baseline = first fixation's pupil, activation = mean of the remaining
  fixations, corrected = activation - baseline (subtractive).
- `blink_aggregated`: blink_rate = blinks / mean RT (ms) per subject x question type.
- The pupil and blink sheets number subjects differently from `raw data`/`RTs_aggregated`: pupil sheet id s
  matches raw id s+1 for s = 3..56, blink sheet likewise for 50 of 57 ids. Pupil id 57 matches no raw
  subject. Everything here is rebuilt from `raw data`.
- Zenodo v1 (18093935) has 60 subjects; v2 drops v1 ids 8, 13, 49 (all liars) and renumbers.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .metrics import binary_metrics, bootstrap_ci, summarise_repeats

QT = ["C", "E", "U"]


# ----------------------------------------------------------------------------- loading
def load_workbook(path):
    return pd.read_excel(path, sheet_name=None, engine="calamine")


def fixations(xl):
    f = xl["raw data"].copy()
    f = f.rename(columns={"SUBJECT_LABEL": "subject", "QUESTION_TYPE": "qt", "TRIAL": "trial",
                          "CURRENT_FIX_PUP": "pup", "BLINK_0_1": "blink"})
    f["pos"] = f.groupby(["subject", "trial"]).cumcount()
    f["n_fix"] = f.groupby(["subject", "trial"])["RT"].transform("size")
    f["y"] = (f["CONDITION"] == 0).astype(int)  # 1 = liar
    return f


def _trial_stats(g):
    p = g["pup"].to_numpy(float)
    b = g["blink"].to_numpy(int)
    n = len(p)
    base = p[0]
    rest = p[1:]
    rest_clean = p[1:][b[1:] == 0]
    half = rest[len(rest) // 2:] if len(rest) else rest
    chosen = (g["KEYPRESS"] == g["ANSWER_LABEL"]).to_numpy(int)
    return {
        "n_fix": n,
        "blinks": int(b.sum()),
        "blink_any": int(b.max()),
        "base": base,
        "act": rest.mean() if len(rest) else np.nan,
        "act_clean": rest_clean.mean() if len(rest_clean) else np.nan,
        "peak": rest.max() if len(rest) else np.nan,
        "late": half.mean() if len(half) else np.nan,
        "base_blink": int(b[0]),
        "on_chosen": chosen.mean(),
    }


def trials_from_fixations(f):
    keys = ["subject", "trial"]
    first = f.groupby(keys).agg(y=("y", "first"), qt=("qt", "first"), idx=("TRIAL_INDEX", "first"),
                                RT=("RT", "first")).reset_index()
    stats = pd.DataFrame([dict(subject=s, trial=t, **_trial_stats(g)) for (s, t), g in f.groupby(keys, sort=False)])
    t = first.merge(stats, on=keys)
    t["logRT"] = np.log(t["RT"])
    t["pc_sub"] = t["act"] - t["base"]                      # published (subtractive)
    t["pc_div"] = (t["act"] - t["base"]) / t["base"]        # divisive / relative change
    t["pc_peak"] = t["peak"] - t["base"]
    t["pc_late"] = t["late"] - t["base"]
    t["pc_clean"] = t["act_clean"] - t["base"]              # blink-flagged fixations removed
    t.loc[t["base_blink"] == 1, "pc_clean"] = np.nan        # and blink-contaminated baselines rejected
    t["pc_none"] = t["act"]                                  # no baseline (raw level)
    t["blink_per_s"] = t["blinks"] / t["RT"] * 1000
    t["fix_per_s"] = t["n_fix"] / t["RT"] * 1000
    return t.sort_values(["subject", "idx"]).reset_index(drop=True)


def apply_rejection(t, rule):
    """Pupil trial-rejection rules. 'published': only trials with no post-baseline fixation
    are lost (NaN). 'minfix3': also reject trials with < 3 fixations. 'z3': also reject trials whose
    corrected pupil is > 3 SD from the subject's own mean."""
    t = t.copy()
    cols = [c for c in t.columns if c.startswith("pc_")]
    if rule == "minfix3":
        t.loc[t["n_fix"] < 3, cols] = np.nan
    elif rule == "z3":
        for c in cols:
            z = t.groupby("subject")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
            t.loc[z.abs() > 3, c] = np.nan
    return t


# ----------------------------------------------------------------------------- subject-level features
def subject_features(t, measures, kinds):
    """Wide table, one row per subject. kinds subset of:
    level (mean per qt), diff (U-C, E-C, U-E), ratio (U/C, E/C), dev ((qt mean - C mean) / SD of the
    person's own trials), devc ((qt mean - C mean) / SD of the person's own control trials)."""
    out = t.groupby("subject").agg(y=("y", "first"))
    for m in measures:
        mean = t.pivot_table(index="subject", columns="qt", values=m, aggfunc="mean").reindex(columns=QT)
        sd_all = t.groupby("subject")[m].std()
        sd_c = t[t.qt == "C"].groupby("subject")[m].std()
        if "level" in kinds:
            for q in QT:
                out[f"{m}_{q}"] = mean[q]
        if "diff" in kinds:
            out[f"{m}_UmC"] = mean["U"] - mean["C"]
            out[f"{m}_EmC"] = mean["E"] - mean["C"]
            out[f"{m}_UmE"] = mean["U"] - mean["E"]
        if "ratio" in kinds:
            out[f"{m}_UdC"] = mean["U"] / mean["C"]
            out[f"{m}_EdC"] = mean["E"] / mean["C"]
        if "dev" in kinds:
            s = sd_all.replace(0, np.nan)
            out[f"{m}_devU"] = ((mean["U"] - mean["C"]) / s).fillna(0)
            out[f"{m}_devE"] = ((mean["E"] - mean["C"]) / s).fillna(0)
        if "devc" in kinds:
            s = sd_c.where(sd_c > 0, sd_all).replace(0, np.nan)
            out[f"{m}_devcU"] = ((mean["U"] - mean["C"]) / s).fillna(0)
            out[f"{m}_devcE"] = ((mean["E"] - mean["C"]) / s).fillna(0)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.reset_index()


def published_style_features(t):
    """The paper's per-subject aggregates rebuilt from raw: mean RT, blink rate (blinks / mean RT)
    and mean subtractive corrected pupil, per question type."""
    out = t.groupby("subject").agg(y=("y", "first"))
    for q in QT:
        tq = t[t.qt == q].groupby("subject")
        out[f"RT_{q}"] = tq["RT"].mean()
        out[f"blink_rate_{q}"] = tq["blinks"].sum() / tq["RT"].mean()
        out[f"pupil_{q}"] = tq["pc_sub"].mean()
    return out.reset_index()


def published_sheets_merged(xl):
    """As-deposited aggregate sheets merged on SUBJECT_LABEL (label from the RT sheet)."""
    rt = xl["RTs_aggregated"].pivot(index="SUBJECT_LABEL", columns="QUESTION_TYPE", values="RT").add_prefix("RT_")
    bl = xl["blink_aggregated"].pivot(index="SUBJECT_LABEL", columns="QUESTION_TYPE", values="blink_rate").add_prefix("blink_rate_")
    pu = xl["pupil_corrected"].groupby(["SUBJECT_LABEL", "QUESTION_TYPE"]).pupil_corrected.mean().unstack().add_prefix("pupil_")
    y = (xl["RTs_aggregated"].groupby("SUBJECT_LABEL").CONDITION.first() == 0).astype(int).rename("y")
    d = pd.concat([y, rt, bl, pu], axis=1)
    d.index.name = "subject"
    return d.reset_index()


# ----------------------------------------------------------------------------- arm registry
_OC, _RT, _LDD = ["pc_sub", "blink_per_s"], ["RT", "logRT"], ["level", "diff", "dev"]
SUBJECT_SPECS = {  # arm -> (rejection rule, [(measures, kinds), ...]); parts are merged column-wise
    "I1_diff_ocular": (None, [(_OC, _LDD)]),
    "I1_diff_rt": (None, [(_RT, ["level", "diff", "ratio", "dev"])]),
    "I1_diff_fusion": (None, [(_OC, _LDD), (_RT, ["level", "diff", "ratio", "dev"])]),
    "I1b_contrast_only_ocular": (None, [(_OC, ["diff", "dev"])]),
    "I2_neutralz_subject_ocular": (None, [(_OC, ["devc"])]),
    "I2_neutralz_subject_rt": (None, [(_RT, ["devc"])]),
    "I2_neutralz_subject_fusion": (None, [(_OC, ["devc"]), (_RT, ["devc"])]),
    "I4_blink_only": (None, [(["blink_per_s"], _LDD)]),
    "I4_pupil_pc_sub_minfix3": ("minfix3", [(["pc_sub"], _LDD)]),
    "I4_pupil_pc_sub_z3": ("z3", [(["pc_sub"], _LDD)]),
    "I5_ocular_plus_fixcount": (None, [(["pc_sub", "blink_per_s", "n_fix"], _LDD)]),
    "I5_ocular_plus_fixrate": (None, [(["pc_sub", "blink_per_s", "fix_per_s"], _LDD)]),
}
for _pc in ("pc_sub", "pc_div", "pc_peak", "pc_clean", "pc_none"):
    SUBJECT_SPECS[f"I4_pupil_{_pc}"] = (None, [([_pc], _LDD)])
for _fam in ("lr_en", "lgbm"):
    for _mod in ("ocular", "rt", "fusion"):
        SUBJECT_SPECS[f"I7_{_fam}_I1_{_mod}"] = SUBJECT_SPECS[f"I1_diff_{_mod}"]


def arm_grid(name):
    """Learner space of a subject-level arm."""
    for fam in ("lr_en", "lgbm"):
        if name.startswith(f"I7_{fam}_"):
            return family_grid((fam,))
    return family_grid(("lr", "svm"))


def arm_subject_table(t, name):
    """Subject-level feature table of a named arm; the same function serves CV, permutations and timing."""
    if name.startswith("B_"):
        return published_style_features(t)
    rule, parts = SUBJECT_SPECS[name]
    if rule:
        t = apply_rejection(t, rule)
    out = None
    for measures, kinds in parts:
        d = subject_features(t, measures, kinds)
        out = d if out is None else out.merge(d.drop(columns="y"), on="subject")
    return out


# ----------------------------------------------------------------------------- learners
def pipe(clf):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("clf", clf)])


def family_grid(families=("lr", "svm", "rf")):
    g = []
    if "lr" in families:
        g.append({"clf": [LogisticRegression(max_iter=5000)], "clf__C": [0.01, 0.1, 1.0, 10.0]})
    if "svm" in families:
        g.append({"clf": [SVC(kernel="rbf")], "clf__C": [0.3, 1.0, 3.0]})
    if "rf" in families:
        g.append({"clf": [RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=0, n_jobs=1)],
                  "clf__max_depth": [2, None]})
    if "lr_en" in families:
        g.append({"clf": [LogisticRegression(solver="saga", l1_ratio=0.5, max_iter=20000)],
                  "clf__C": [0.1, 1.0], "clf__l1_ratio": [0.5, 1.0]})
    if "lgbm" in families:
        from lightgbm import LGBMClassifier
        g.append({"clf": [LGBMClassifier(n_estimators=100, learning_rate=0.05, min_child_samples=5,
                                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                         verbose=-1, n_jobs=1, random_state=0)],
                  "clf__num_leaves": [3, 7]})
    return g


def signed_score(model, X):
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except AttributeError:
            pass
    return model.predict_proba(X)[:, 1] - 0.5


def proba(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return 1 / (1 + np.exp(-signed_score(model, X)))


# ----------------------------------------------------------------------------- nested CV, subject level
def _subject_task(r, f, X, y, units, test_units, grid, seed):
    test = np.isin(units, test_units)
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + r)
    gs = GridSearchCV(pipe(LogisticRegression()), grid, cv=inner, scoring="roc_auc", n_jobs=1, refit=True)
    gs.fit(X[~test], y[~test], groups=units[~test])
    est = gs.best_estimator_
    return pd.DataFrame({"repeat": r, "fold": f, "unit": units[test], "y": y[test],
                         "score": signed_score(est, X[test]), "prob": proba(est, X[test]),
                         "params": str({k: (type(v).__name__ if k == "clf" else v) for k, v in gs.best_params_.items()})})


def run_subject_level(df, features, folds_per_repeat, grid, n_jobs=2, seed=0):
    X = df[features].to_numpy(float)
    y = df["y"].to_numpy(int)
    units = df["subject"].astype(str).to_numpy()
    tasks = [(r, f, tu) for r, folds in enumerate(folds_per_repeat) for f, tu in enumerate(folds)]
    out = Parallel(n_jobs=n_jobs)(delayed(_subject_task)(r, f, X, y, units, tu, grid, seed) for r, f, tu in tasks)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- nested CV, trial level + aggregation
AGG_RULES = ("mean", "median", "trim20", "vote")


def agg_scores(units, scores, rule):
    s = pd.Series(scores, index=units)
    g = s.groupby(level=0)
    if rule == "mean":
        return g.mean()
    if rule == "median":
        return g.median()
    if rule == "trim20":
        from scipy.stats import trim_mean
        return g.apply(lambda v: trim_mean(v.to_numpy(), 0.2))
    if rule == "vote":  # fraction of trials called deceptive, centred so 0 is the threshold
        return g.apply(lambda v: float((v.to_numpy() >= 0).mean()) - 0.5 + 1e-3 * float(np.mean(v.to_numpy())))
    raise ValueError(rule)


def per_subject_neutral_z(t, cols, ref="C"):
    """z-score each subject's trials with the mean/SD of that subject's own control (neutral)
    trials. Uses no labels and no other participant, so it is leakage-free."""
    t = t.copy()
    for c in cols:
        ref_stats = t[t.qt == ref].groupby("subject")[c].agg(["mean", "std"])
        all_sd = t.groupby("subject")[c].std()
        sd = ref_stats["std"].where(ref_stats["std"] > 0, all_sd).replace(0, np.nan)
        mu = t["subject"].map(ref_stats["mean"])
        t[c + "_z"] = ((t[c] - mu) / t["subject"].map(sd)).fillna(0)
    return t


def _trial_task(r, f, X, y, units, test_units, grid_list, rules, seed):
    test = np.isin(units, test_units)
    Xtr, ytr, gtr = X[~test], y[~test], units[~test]
    inner = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + r).split(Xtr, ytr, gtr))
    best = (-np.inf, None, None)
    for params in grid_list:
        oof = np.full(len(ytr), np.nan)
        for tr, va in inner:
            m = clone(pipe(LogisticRegression()).set_params(**params)).fit(Xtr[tr], ytr[tr])
            oof[va] = signed_score(m, Xtr[va])
        yu = pd.Series(ytr, index=gtr).groupby(level=0).first()
        for rule in rules:
            a = agg_scores(gtr, oof, rule)
            auc = roc_auc_score(yu.loc[a.index], a.to_numpy())
            if auc > best[0]:
                best = (auc, params, rule)
    _, params, rule = best
    m = clone(pipe(LogisticRegression()).set_params(**params)).fit(Xtr, ytr)
    s = signed_score(m, X[test])
    p = proba(m, X[test])
    a = agg_scores(units[test], s, rule)
    pa = pd.Series(p, index=units[test]).groupby(level=0).mean()
    yu = pd.Series(y[test], index=units[test]).groupby(level=0).first()
    return pd.DataFrame({"repeat": r, "fold": f, "unit": a.index, "y": yu.loc[a.index].to_numpy(),
                         "score": a.to_numpy(), "prob": pa.loc[a.index].to_numpy(),
                         "params": str({"rule": rule, **{k: (type(v).__name__ if k == "clf" else v) for k, v in params.items()}})})


def expand_grid(grid):
    from sklearn.model_selection import ParameterGrid
    return list(ParameterGrid(grid))


def run_trial_level(t, features, folds_per_repeat, grid, rules=AGG_RULES, n_jobs=2, seed=0):
    X = t[features].to_numpy(float)
    y = t["y"].to_numpy(int)
    units = t["subject"].astype(str).to_numpy()
    gl = expand_grid(grid)
    tasks = [(r, f, tu) for r, folds in enumerate(folds_per_repeat) for f, tu in enumerate(folds)]
    out = Parallel(n_jobs=n_jobs)(delayed(_trial_task)(r, f, X, y, units, tu, gl, rules, seed) for r, f, tu in tasks)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- evaluation
def ece(y, p, n_bins=5):
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def evaluate_participants(parts, n_boot=2000, ci=True):
    """parts: one row per (repeat, unit) with y, score, prob."""
    per_rep = []
    for _, p in parts.groupby("repeat"):
        m = binary_metrics(p.y, p.score)
        m["brier"] = float(np.mean((p.prob - p.y) ** 2))
        m["ece"] = ece(p.y, p.prob)
        per_rep.append(m)
    avg = parts.groupby("unit").agg(y=("y", "first"), score=("score", "mean"), prob=("prob", "mean"))
    res = {"participant": summarise_repeats(per_rep),
           "participant_avg_scores": binary_metrics(avg.y, avg.score),
           "n_repeats": len(per_rep)}
    if ci:
        res["participant_ci95"] = bootstrap_ci(avg.y.to_numpy(), avg.score.to_numpy(), n_boot=n_boot)
    return res, avg


def paired_auc_diff(avg_a, avg_b, n_boot=2000, seed=0):
    """Stratified participant bootstrap of AUC(a) - AUC(b) on repeat-averaged scores (same units)."""
    j = avg_a[["y", "score"]].join(avg_b[["score"]], rsuffix="_b").dropna()
    y, sa, sb = j.y.to_numpy(int), j.score.to_numpy(), j.score_b.to_numpy()
    obs = roc_auc_score(y, sa) - roc_auc_score(y, sb)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

    def wins(v):  # Mann-Whitney win matrix: AUC is its mean, a bootstrap AUC its weighted mean
        d = v[pos][:, None] - v[neg][None, :]
        return (d > 0) + 0.5 * (d == 0)

    delta = wins(sa) - wins(sb)
    wp = rng.multinomial(len(pos), np.full(len(pos), 1 / len(pos)), size=n_boot) / len(pos)
    wn = rng.multinomial(len(neg), np.full(len(neg), 1 / len(neg)), size=n_boot) / len(neg)
    d = np.einsum("bi,ij,bj->b", wp, delta, wn)
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {"delta_auc": float(obs), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_boot": p}


def holm(pvals):
    keys = list(pvals)
    order = sorted(keys, key=lambda k: pvals[k])
    m = len(keys)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj
