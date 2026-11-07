"""Phase 2, track delmas_raw, step 2: screening experiments on the raw-sample features.

Stages (all on frozen participant-disjoint folds, nested participant-grouped tuning):
  audit     - how the published per-image table was aggregated (label-dependent presentation subsets)
  baseline  - published features (BookFIP) + the paper's RBF-SVM under our protocol
  rate      - idea 1: sampling-rate degradation curve (1000 -> 30 Hz), deployable per-presentation features
  features  - idea 3: new raw features (pupil, first look, microsaccades, time-course bins, participant z-scoring)
  transfer  - idea 2: train on no-countermeasure groups, test on countermeasure groups, and the reverse
  models    - idea 4: SVM vs logistic regression vs small LightGBM, accuracy + deployment cost end to end
  identity  - idea 5: how well trial-level gaze summaries identify the participant, per sensor rate

Usage: python -I scripts/phase2_delmas_raw.py <derived_dir> <BookFIP_64.xlsx> <BookFIP_32.xlsx> <out_dir> <data_raw_dir>
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from edd import delmas_raw as dr  # noqa: E402
from edd import efficiency as eff  # noqa: E402
from edd.delmas import FEATURES as BOOK_FEATURES, load_trials  # noqa: E402
from edd.harness import scores_of  # noqa: E402
from edd.metrics import binary_metrics, bootstrap_ci, summarise_repeats  # noqa: E402

warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
N_JOBS = 2
N_REPEATS = 3
N_FOLDS = 5
RATES = [1000, 500, 250, 120, 60, 30]
SETS = [("64", 1), ("64", 2), ("64", 3), ("32", 3), ("32", 4), ("32", 5)]
GROUP_NAMES = {1: "control", 2: "concealment", 3: "simple CM", 4: "enhanced CM", 5: "CM + feedback"}
PUBLISHED = {("64", 1): {"auc": 1.00, "ba": .98, "f1": .97}, ("64", 2): {"auc": .93, "ba": .89, "f1": .83},
             ("64", 3): {"auc": 1.00, "ba": .96, "f1": .93}, ("32", 3): {"auc": .82, "ba": .94, "f1": .73},
             ("32", 4): {"auc": .99, "ba": 1.00, "f1": .98}, ("32", 5): {"auc": 1.00, "ba": 1.00, "f1": .99}}
P4 = ["dwell_1", "dwell_2", "n_fix", "n_visit"]
BINS = ["dwell_b1", "dwell_b2", "dwell_b3", "dwell_b4", "dwell_b5"]
NEW = ["first_fix", "lat_first", "pupil_face", "pupil_trial", "msacc_rate"]
FEATURE_SETS = {
    "P4": P4, "P4+bins": P4 + BINS, "NEW": NEW, "P4+NEW": P4 + NEW, "ALL": P4 + BINS + NEW,
    "P4_z": [f + "_z" for f in P4], "ALL_z": [f + "_z" for f in P4 + BINS + NEW],
    "pupil_only": ["pupil_face", "pupil_trial"], "firstlook_only": ["first_fix", "lat_first"],
}


# ----------------------------------------------------------------------------- data
def build_tables(derived, book64, book32):
    """Per (rate, set, participant, image) rows: per-presentation means over every presentation in the
    set's trial window (label-free: no label is used to choose presentations)."""
    face = pd.read_csv(Path(derived) / "delmas_raw_face_trials.csv.gz")
    books = {"64": load_trials(book64), "32": load_trials(book32)}
    out = []
    for s, book in books.items():
        f = face if s == "64" else face[face.trial >= 32]
        f = f[f.participant.isin(book.participant.unique())]
        agg = f.groupby(["rate", "participant", "image"])[dr.TRIAL_FEATURES].mean().reset_index()
        npres = f.groupby(["rate", "participant", "image"]).size().rename("n_pres").reset_index()
        agg = agg.merge(npres, on=["rate", "participant", "image"])
        # presentations the published table appears to aggregate: famous -> all, unknown -> non-famous trials
        pub = f[(f.image >= 57) | (f.fam_trial == 0)]
        el = f[f.rate == 1000].groupby(["participant", "image"]).el_nfix.sum().rename("el_nfix_all").reset_index()
        elp = pub[pub.rate == 1000].groupby(["participant", "image"]).agg(
            el_nfix_pub=("el_nfix", "sum"), n_pres_pub=("el_nfix", "size")).reset_index()
        agg = agg.merge(el, on=["participant", "image"], how="left").merge(elp, on=["participant", "image"], how="left")
        agg[["el_nfix_pub", "n_pres_pub"]] = agg[["el_nfix_pub", "n_pres_pub"]].fillna(0)
        # our own detector, aggregated with the published (label-dependent) rule: dwell = mean, counts = sums
        pr = pub.groupby(["rate", "participant", "image"]).agg(
            dwell_1_pr=("dwell_1", "mean"), dwell_2_pr=("dwell_2", "mean"),
            n_fix_pr=("n_fix", "sum"), n_visit_pr=("n_visit", "sum")).reset_index()
        agg = agg.merge(pr, on=["rate", "participant", "image"], how="left")
        agg[["n_fix_pr", "n_visit_pr"]] = agg[["n_fix_pr", "n_visit_pr"]].fillna(0)
        b = book.rename(columns={c: "book_" + c for c in BOOK_FEATURES})
        agg = agg.merge(b, on=["participant", "image"], how="inner")
        agg["set"] = s
        out.append(agg)
    df = pd.concat(out, ignore_index=True)
    cols = P4 + BINS + NEW
    z = df.groupby(["rate", "set", "participant"])[cols].transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
    for c in cols:
        df[c + "_z"] = z[c]
    return df


def make_folds(df, path):
    folds = {}
    for s, g in SETS:
        parts = np.array(sorted(df[(df.set == s) & (df.group == g)].participant.unique()))
        reps = []
        for r in range(N_REPEATS):
            perm = np.random.default_rng(2026 + r).permutation(parts)
            reps.append([sorted(int(p) for p in perm[i::N_FOLDS]) for i in range(N_FOLDS)])
        folds[f"{s}_g{g}"] = reps
    Path(path).write_text(json.dumps({"meta": {"design": "within-group, 5 participant-disjoint folds x 3 repeats, "
                                               "seeds 2026+r; same folds for every rate, feature set and model"},
                                      "repeats": folds}, indent=1))
    return folds


# ----------------------------------------------------------------------------- models
def model_factory(name, seed):
    inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
    pre = [("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if name == "svm":
        pipe = Pipeline(pre + [("clf", SVC(class_weight="balanced"))])
        return GridSearchCV(pipe, {"clf__C": [0.1, 1, 10], "clf__gamma": [0.01, 0.1, 1]}, cv=inner,
                            scoring="roc_auc", n_jobs=1)
    if name == "lr":
        pipe = Pipeline(pre + [("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
        return GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1, 10]}, cv=inner, scoring="roc_auc", n_jobs=1)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        pipe = Pipeline([("clf", LGBMClassifier(n_estimators=100, num_leaves=7, learning_rate=0.05,
                                                 min_child_samples=10, class_weight="balanced", n_jobs=1,
                                                 verbose=-1, random_state=seed))])
        return GridSearchCV(pipe, {"clf__n_estimators": [50, 150]}, cv=inner, scoring="roc_auc", n_jobs=1)
    raise ValueError(name)


def _fold_task(X, y, g, test_units, model, seed):
    te = np.isin(g, test_units)
    m = model_factory(model, seed).fit(X[~te], y[~te], groups=g[~te])
    return np.flatnonzero(te), scores_of(m.best_estimator_, X[te])


def run_cv(d, feats, folds, model):
    """Within-group outer CV. Returns OOF rows (repeat, participant, image, y, score)."""
    X, y, g = d[feats].to_numpy(float), d.face.to_numpy(int), d.participant.to_numpy()
    tasks = [(r, t) for r, fl in enumerate(folds) for t in fl]
    res = Parallel(n_jobs=N_JOBS)(delayed(_fold_task)(X, y, g, t, model, r) for r, t in tasks)
    rows = []
    for (r, _), (idx, sc) in zip(tasks, res):
        rows.append(pd.DataFrame({"repeat": r, "participant": g[idx], "image": d.image.to_numpy()[idx],
                                  "y": y[idx], "score": sc}))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- metrics
def cit_scores(o, seed=0):
    """Participant-level CIT: z of probe (famous) vs irrelevant scores; simulated innocent = same person
    with |famous| random unknown faces as pseudo-probes (as in the paper's unknowledgeable simulation)."""
    rows = []
    for p, d in o.groupby("participant"):
        s, y = d.score.to_numpy(), d.y.to_numpy()
        sf, su = s[y == 1], s[y == 0]
        zk = (sf.mean() - su.mean()) / (su.std() + 1e-9)
        rng = np.random.default_rng(seed + int(p))
        pick = rng.choice(len(su), len(sf), replace=False)
        rest = np.delete(su, pick)
        zi = (su[pick].mean() - rest.mean()) / (rest.std() + 1e-9)
        rows += [(p, 1, zk - 1.645), (p, 0, zi - 1.645)]
    return pd.DataFrame(rows, columns=["participant", "y", "score"])


def evaluate(oof):
    """Image-level pooled metrics, paper-style per-participant means, participant-level CIT metrics;
    each per repeat then mean/SD."""
    img, pp, cit, per_part = [], [], [], []
    for r, o in oof.groupby("repeat"):
        img.append(binary_metrics(o.y, o.score))
        pa = []
        for p, d in o.groupby("participant"):
            m = binary_metrics(d.y, d.score)
            pa.append({"auc": m["auc"], "balanced_accuracy": m["balanced_accuracy"], "f1": m["f1"]})
            per_part.append((r, p, m["auc"]))
        pp.append({k: float(np.mean([x[k] for x in pa])) for k in pa[0]})
        c = cit_scores(o, seed=r)
        cit.append(binary_metrics(c.y, c.score))
    pa_auc = pd.DataFrame(per_part, columns=["repeat", "participant", "auc"]).groupby("participant").auc.mean()
    return {"image": summarise_repeats(img), "per_participant_mean": summarise_repeats(pp),
            "participant_cit": summarise_repeats(cit)}, pa_auc


def cluster_bootstrap_auc(oof, n_boot=2000, seed=0):
    """Participant-cluster bootstrap CI of the pooled image-level AUC (repeat-averaged scores)."""
    avg = oof.groupby(["participant", "image"]).agg(y=("y", "first"), score=("score", "mean")).reset_index()
    parts = avg.participant.unique()
    by = {p: avg[avg.participant == p] for p in parts}
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        d = pd.concat([by[p] for p in rng.choice(parts, len(parts))])
        aucs.append(roc_auc_score(d.y, d.score))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def short(ev):
    i, c, p = ev["image"], ev["participant_cit"], ev["per_participant_mean"]
    return (f"img AUC {i['auc']['mean']:.3f}±{i['auc']['sd']:.3f} BA {i['balanced_accuracy']['mean']:.3f} "
            f"EER {i['eer']['mean']:.3f} HTER {i['hter']['mean']:.3f} | pp AUC {p['auc']['mean']:.3f} | "
            f"CIT AUC {c['auc']['mean']:.3f} HTER {c['hter']['mean']:.3f}")


# ----------------------------------------------------------------------------- stages
def stage_audit(df):
    """Is the published table aggregated over a label-dependent subset of presentations?"""
    d = df[df.rate == 1000]
    out = {}
    for s in ("64", "32"):
        x = d[d.set == s]
        r_all = np.corrcoef(x.el_nfix_all, x.book_n_fix)[0, 1]
        r_pub = np.corrcoef(x.el_nfix_pub, x.book_n_fix)[0, 1]
        exact = float((x.el_nfix_pub == x.book_n_fix).mean())
        out[s] = {
            "n_rows": int(len(x)),
            "corr_book_nfix_vs_eyelink_all_presentations": float(r_all),
            "corr_book_nfix_vs_eyelink_famous_all_unknown_nonfamous_trials": float(r_pub),
            "exact_match_rate_label_dependent_rule": exact,
            "mean_presentations_counted_famous": float(x[x.face == 1].n_pres_pub.mean()),
            "mean_presentations_counted_unknown": float(x[x.face == 0].n_pres_pub.mean()),
            "auc_presentations_counted_alone": float(roc_auc_score(x.face, x.n_pres_pub)),
            "auc_book_n_fix_alone": float(roc_auc_score(x.face, x.book_n_fix)),
            "auc_book_n_visit_alone": float(roc_auc_score(x.face, x.book_n_visit)),
            "auc_eyelink_nfix_sum_label_dependent": float(roc_auc_score(x.face, x.el_nfix_pub)),
            "auc_eyelink_nfix_sum_all_presentations": float(roc_auc_score(x.face, x.el_nfix_all)),
            "auc_eyelink_nfix_per_presentation": float(roc_auc_score(x.face, x.el_nfix_all / x.n_pres)),
            "corr_book_dwell_1_vs_ours_all": float(x.dwell_1.corr(x.book_dwell_1)),
            "corr_book_dwell_1_vs_ours_label_rule": float(x.dwell_1_pr.corr(x.book_dwell_1)),
            "corr_book_dwell_2_vs_ours_all": float(x.dwell_2.corr(x.book_dwell_2)),
            "corr_book_dwell_2_vs_ours_label_rule": float(x.dwell_2_pr.corr(x.book_dwell_2)),
            "corr_book_n_visit_vs_ours_all": float((x.n_visit * x.n_pres).corr(x.book_n_visit)),
            "corr_book_n_visit_vs_ours_label_rule": float(x.n_visit_pr.corr(x.book_n_visit)),
            "auc_book_dwell_2_alone_reversed": float(roc_auc_score(x.face, -x.book_dwell_2)),
            "auc_rebuilt_dwell_2_alone_reversed": float(roc_auc_score(x.face, -x.dwell_2)),
        }
        print(f"[audit {s}] " + ", ".join(f"{k}={v:.3f}" for k, v in out[s].items() if k != "n_rows"), flush=True)
    return out


def subset(df, s, g, rate):
    return df[(df.set == s) & (df.group == g) & (df.rate == rate)].reset_index(drop=True)


CKPT = None      # results/phase2/delmas_raw/checkpoints, set in main()


def _ck(name):
    return CKPT / (name.replace("|", "_").replace(" ", "_").replace(">", "").replace("(", "").replace(")", "") )


def cached(name, fn):
    """Return the saved JSON result for `name`, or compute it with fn(), save it at once, and return it."""
    f = _ck(name).with_suffix(".json")
    if f.exists():
        print(f"[skip, cached] {name}", flush=True)
        return json.loads(f.read_text())
    r = fn()
    f.write_text(json.dumps(r, indent=1, default=str))
    progress_note(name, r)
    return r


def progress_note(name, r):
    """Append one headline line per finished unit of work to progress.md (read by RESUME.md)."""
    line = f"- {time.strftime('%H:%M:%S')} {name}"
    try:
        if "mean_over_sets" in r:
            m, c = r["mean_over_sets"]["image"], r["mean_over_cm_sets"]["image"]
            line += (f": mean AUC {m['auc']:.3f} EER {m['eer']:.3f} HTER {m['hter']:.3f}; "
                     f"countermeasure/concealment sets AUC {c['auc']:.3f}")
    except Exception:
        pass
    with open(CKPT.parent / "progress.md", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_grid(df, folds, feats, model, rate, tag, store, oofs):
    """One configuration over the six group-sets. Saved to checkpoints/<tag>.json (+ OOF scores) the moment it
    finishes, and loaded instead of recomputed on a rerun."""
    store.setdefault("_tags", {})[tag] = {"features": list(feats), "model": model, "rate": rate}
    fj, fo = _ck(tag).with_suffix(".json"), _ck(tag + "_oof").with_suffix(".csv.gz")
    if fj.exists():
        c = json.loads(fj.read_text())
        store[tag] = c["res"]
        for k, v in c["pa"].items():
            store.setdefault("_per_participant_auc", {})[k] = {int(p): float(x) for p, x in v.items()}
        if fo.exists():
            o = pd.read_csv(fo)
            for key, oo in o.groupby("set"):
                oofs[(tag, key)] = oo.drop(columns="set").reset_index(drop=True)
        print(f"[skip, cached] {tag}", flush=True)
        return c["res"]
    res = {}
    for s, g in SETS:
        key = f"{s}_g{g}"
        d = subset(df, s, g, rate)
        o = run_cv(d, feats, folds[key], model)
        ev, pa = evaluate(o)
        res[key] = ev
        oofs[(tag, key)] = o
        store.setdefault("_per_participant_auc", {})[f"{tag}|{key}"] = {int(p): float(v) for p, v in pa.items()}
        print(f"[{tag}] {key}: {short(ev)}", flush=True)
    res["mean_over_sets"] = {lvl: {m: float(np.mean([res[k][lvl][m]["mean"] for k in res]))
                                   for m in ("auc", "balanced_accuracy", "eer", "hter", "mcc")}
                             for lvl in ("image", "participant_cit")}
    res["mean_over_cm_sets"] = {lvl: {m: float(np.mean([res[k][lvl][m]["mean"] for k in ("64_g2", "32_g3", "32_g4", "32_g5")]))
                                      for m in ("auc", "balanced_accuracy", "eer", "hter", "mcc")}
                                for lvl in ("image", "participant_cit")}
    store[tag] = res
    pa_all = {k: v for k, v in store["_per_participant_auc"].items() if k.startswith(tag + "|")}
    pd.concat([oofs[(tag, f"{s}_g{g}")].assign(set=f"{s}_g{g}") for s, g in SETS]).to_csv(fo, index=False)
    fj.write_text(json.dumps({"res": res, "pa": pa_all, "spec": store["_tags"][tag]}, indent=1, default=str))
    progress_note(tag, res)
    return res


def stage_transfer(df, feats, model, rate, tag):
    """Train on one block of groups, test on another (participants are disjoint by design)."""
    out = {}
    A = df[(df.rate == rate) & (df.set == "64") & df.group.isin([1, 2])].reset_index(drop=True)
    B = df[(df.rate == rate) & (df.set == "32") & df.group.isin([3, 4, 5])].reset_index(drop=True)
    for name, tr, te in [("noCM(g1-2)->CM(g3-5 last32)", A, B), ("CM(g3-5 last32)->noCM(g1-2)", B, A)]:
        reps = []
        for r in range(N_REPEATS):
            m = model_factory(model, r).fit(tr[feats].to_numpy(float), tr.face.to_numpy(), groups=tr.participant.to_numpy())
            reps.append(pd.DataFrame({"repeat": r, "participant": te.participant, "image": te.image, "y": te.face,
                                      "score": scores_of(m.best_estimator_, te[feats].to_numpy(float)),
                                      "group": te.group}))
        o = pd.concat(reps, ignore_index=True)
        ev, _ = evaluate(o)
        per_group = {int(gg): evaluate(oo)[0] for gg, oo in o.groupby("group")}
        out[name] = {"all": ev, "per_group": per_group}
        print(f"[{tag}] {name}: {short(ev)}", flush=True)
        for gg, e in per_group.items():
            print(f"      group {gg}: {short(e)}", flush=True)
    return out


def holm(pvals):
    keys = sorted(pvals, key=pvals.get)
    m, adj, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        adj[k] = run
    return adj


def paired_tests(store, base_tag, cand_tags):
    """Wilcoxon signed-rank on per-participant AUC (repeat-averaged) over the six group-sets, identical
    folds, one unit per participant (group 3 sits in two sets, so it is averaged first); Holm across candidates."""
    pp = store["_per_participant_auc"]
    res, pv = {}, {}
    for c in cand_tags:
        by = {}
        for s_, g in SETS:
            k = f"{s_}_g{g}"
            x, y = pp[f"{base_tag}|{k}"], pp[f"{c}|{k}"]
            for p in x:
                by.setdefault(p, []).append((x[p], y[p]))
        a = np.array([np.mean([v[0] for v in vs]) for vs in by.values()])
        b = np.array([np.mean([v[1] for v in vs]) for vs in by.values()])
        diff = b - a
        p = float(wilcoxon(b, a, zero_method="zsplit").pvalue) if np.any(diff != 0) else 1.0
        res[c] = {"mean_diff_pp_auc": float(diff.mean()), "n_units": int(len(diff)), "p_raw": p}
        pv[c] = p
    for c, p in holm(pv).items():
        res[c]["p_holm"] = p
    return res


def permutation_test(df, feats, folds, rate, n_perm=30, seed=0):
    """Item labels are within-person, so shuffle which images are famous within each participant and rerun
    the nested pipeline on repeat 0 of every group-set. Logistic regression is used for the real labels and
    every permutation alike, so the test stays affordable; statistic = mean pooled image AUC over sets."""
    rng = np.random.default_rng(seed)

    def stat(shuffle):
        aucs = []
        for s, g in SETS:
            d = subset(df, s, g, rate).copy()
            if shuffle:
                d["face"] = d.groupby("participant").face.transform(lambda v: rng.permutation(v.to_numpy()))
            o = run_cv(d, feats, folds[f"{s}_g{g}"][:1], "lr")
            aucs.append(roc_auc_score(o.y, o.score))
        return float(np.mean(aucs))

    observed = stat(False)
    null = np.array([stat(True) for _ in range(n_perm)])
    return {"n_perm": n_perm, "model": "lr", "observed_mean_auc": observed, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "p_value": float((1 + np.sum(null >= observed)) / (1 + n_perm))}


def identity_probe(derived):
    """Participant identification from per-trial gaze summaries (train odd trials, test even, and back)."""
    t = pd.read_csv(Path(derived) / "delmas_raw_trial_summary.csv.gz")
    gaze = ["s_nfix", "s_fixdur_mean", "s_fixdur_sd", "s_sacc_amp", "s_fix_vel", "s_missing", "s_fix_x_med",
            "s_fix_y_med", "s_lat_first"]
    allf = gaze + ["s_pupil_mean", "s_pupil_sd"]
    out = {}
    n_cls = t.participant.nunique()
    for rate in RATES:
        d = t[t.rate == rate]
        res = {}
        for name, cols in [("gaze_only", gaze), ("gaze+pupil", allf)]:
            accs = []
            for tr_par in (0, 1):
                tr, te = d[d.trial % 2 == tr_par], d[d.trial % 2 != tr_par]
                pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                 ("clf", LogisticRegression(C=1.0, max_iter=3000))])
                pipe.fit(tr[cols], tr.participant)
                accs.append(float((pipe.predict(te[cols]) == te.participant).mean()))
            res[name] = {"trial_top1_acc": float(np.mean(accs)), "chance": 1.0 / n_cls}
        out[rate] = res
        print(f"[identity] {rate} Hz: " + ", ".join(f"{k} {v['trial_top1_acc']:.3f}" for k, v in res.items())
              + f" (chance {1 / n_cls:.3f})", flush=True)
    return out


# ----------------------------------------------------------------------------- efficiency
def raw_to_scores(pre, images, fs, m, feats):
    """Raw samples of one participant -> I-VT -> features -> per-image aggregation -> model scores."""
    base = [f[:-2] if f.endswith("_z") else f for f in feats]
    idx = [dr.TRIAL_FEATURES.index(f) for f in base]
    msacc = "msacc_rate" in base
    rows = {}
    for k in range(len(pre)):
        F = dr.trial_features(*pre[k], fs, with_msacc=msacc)
        for q in range(4):
            rows.setdefault(int(images[k, q]), []).append(F[q])
    A = np.array([np.nanmean(v, 0) for v in rows.values()])
    if feats[0].endswith("_z"):
        A = (A - np.nanmean(A, 0)) / (np.nanstd(A, 0, ddof=1) + 1e-9)
    return scores_of(m, A[:, idx])


def cost_from_raw(seg, fs, m, feats):
    pre = [dr.degrade(seg["x"][k], seg["y"][k], seg["p"][k], fs) for k in range(64)]   # simulated sensor output
    images = seg["images"]
    base = [f[:-2] if f.endswith("_z") else f for f in feats]
    idx = [dr.TRIAL_FEATURES.index(f) for f in base]
    msacc = "msacc_rate" in base

    def one_trial():
        F = dr.trial_features(*pre[0], fs, with_msacc=msacc)
        return scores_of(m, F[:, idx])

    def one_participant():
        return raw_to_scores(pre, images, fs, m, feats)

    lt = eff.latency_ms(one_trial, n_runs=200, warmup=10)
    lp = eff.latency_ms(one_participant, n_runs=10, warmup=2)
    n_in = 64 * len(pre[0][0])      # input samples per participant, including the 500 ms pre-onset window
    sps = n_in / (lp["p50"] / 1000)
    return {"per_trial_ms": lt, "per_participant_ms": lp, "input_samples_per_participant": int(n_in),
            "samples_per_s": float(sps), "real_time_factor": float(sps / fs),
            "peak_memory_bytes": eff.peak_memory_bytes(one_participant)}


def efficiency(df, raw_dir, best=None):
    """Deployment cost of each model on P4, and of the best candidate, including end-to-end from raw samples."""
    from edd.delmas_raw import load_session
    d = df[(df.rate == 1000) & (df.set == "64")].reset_index(drop=True)
    X, y, g = d[P4].to_numpy(float), d.face.to_numpy(), d.participant.to_numpy()
    res = {"hardware": eff.hardware_info()}
    # one participant's raw session (64 trials x 5.5 s at 1000 Hz)
    pid = 4
    seg = load_session(Path(raw_dir) / f"Sj_{pid:02d}" / f"I{pid:02d}c2_1.edf",
                       Path(raw_dir) / f"Sj_{pid:02d}" / f"FI{pid:02d}c2_1.mat")
    n_samples_display = 64 * dr.POST_MS          # samples of 5-s displays at 1000 Hz
    for name in ("svm", "lr", "lgbm"):
        m = model_factory(name, 0).fit(X, y, groups=g).best_estimator_
        clf = m.named_steps["clf"]
        if name == "svm":
            n_par = eff.svm_parameter_count(clf, len(P4)) + 2 * len(P4) * 2
        elif name == "lr":
            n_par = clf.coef_.size + clf.intercept_.size + 2 * len(P4) * 2
        else:
            dfm = clf.booster_.trees_to_dataframe()
            n_par = int(len(dfm)) * 2      # split threshold or leaf value + feature/child index per node
        x64 = X[:64]
        r = {"model_size_bytes": eff.model_size_bytes(m), "stored_parameters": int(n_par),
             "params": str({k: v for k, v in clf.get_params().items() if k in ("C", "gamma", "n_estimators", "num_leaves")}),
             "inference_per_face_row_ms": eff.latency_ms(lambda: scores_of(m, x64[:1]), n_runs=500),
             "inference_per_trial_4_faces_ms": eff.latency_ms(lambda: scores_of(m, x64[:4]), n_runs=500),
             "inference_per_participant_64_images_ms": eff.latency_ms(lambda: scores_of(m, x64), n_runs=500),
             "inference_peak_memory_bytes": eff.peak_memory_bytes(lambda: scores_of(m, x64)),
             "decisions_per_s_images": eff.throughput_per_s(lambda: scores_of(m, x64), 64, seconds=1.0)}
        # end to end from raw samples: I-VT + features + aggregation + model, on a simulated native-fs stream
        e2e = {fs: cost_from_raw(seg, fs, m, P4) for fs in (1000, 250, 60, 30)}
        r["end_to_end_from_raw"] = e2e
        res[name] = r
        print(f"[eff] {name}: size {r['model_size_bytes']} B, params {n_par}, per-participant "
              f"{r['inference_per_participant_64_images_ms']['p50']:.3f} ms; e2e 1000 Hz "
              f"{e2e[1000]['per_participant_ms']['p50']:.1f} ms/participant, RTF {e2e[1000]['real_time_factor']:.0f}x; "
              f"60 Hz RTF {e2e[60]['real_time_factor']:.0f}x", flush=True)
    if best:
        tag, spec = best
        d = df[(df.rate == spec["rate"]) & (df.set == "64")].reset_index(drop=True)
        feats = spec["features"]
        m = model_factory(spec["model"], 0).fit(d[feats].to_numpy(float), d.face.to_numpy(),
                                                groups=d.participant.to_numpy()).best_estimator_
        x64 = d[feats].to_numpy(float)[:64]
        res["best_candidate"] = {
            "tag": tag, "model_size_bytes": eff.model_size_bytes(m),
            "inference_per_participant_64_images_ms": eff.latency_ms(lambda: scores_of(m, x64), n_runs=500),
            "inference_peak_memory_bytes": eff.peak_memory_bytes(lambda: scores_of(m, x64)),
            "end_to_end_from_raw": {spec["rate"]: cost_from_raw(seg, spec["rate"], m, feats)}}
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("derived")
    ap.add_argument("book64")
    ap.add_argument("book32")
    ap.add_argument("out_dir")
    ap.add_argument("raw_dir")
    ap.add_argument("--stages", default="audit,baseline,rate,features,transfer,models,stats,identity,efficiency")
    a = ap.parse_args()
    out = Path(a.out_dir)
    global CKPT
    CKPT = out / "checkpoints"
    CKPT.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    stages = a.stages.split(",")
    t0 = time.perf_counter()
    df = build_tables(a.derived, a.book64, a.book32)
    folds = make_folds(df, out / "folds.json")
    R, oofs = {"n_participants": int(df.participant.nunique()), "rows_per_rate": int(len(df) / len(RATES))}, {}
    timing = {}

    def tick(name):
        timing[name] = round(time.perf_counter() - t0, 1)
        R["timing_s"] = timing
        (out / "results.json").write_text(json.dumps(R, indent=1, default=str))
        print(f"[saved after {name}: {timing[name]} s]", flush=True)

    if "audit" in stages:
        R["audit"] = stage_audit(df)
        tick("audit")
    if "baseline" in stages:
        # honest baseline: published table + paper's SVM under our protocol
        d1000 = df[df.rate == 1000].copy()
        for c in BOOK_FEATURES:
            d1000["B_" + c] = d1000["book_" + c]
        R["baseline_book_svm"] = run_grid(d1000.assign(rate=1000), folds, ["B_" + c for c in BOOK_FEATURES],
                                          "svm", 1000, "book_svm", R, oofs)
        # same detector as P4, but presentations chosen with the published label-dependent rule
        R["pubrule_svm"] = run_grid(df, folds, ["dwell_1_pr", "dwell_2_pr", "n_fix_pr", "n_visit_pr"], "svm", 1000,
                                    "P4_pubrule_svm_1000", R, oofs)
        R["pubrule_counts_only_svm"] = run_grid(df, folds, ["n_fix_pr", "n_visit_pr"], "svm", 1000,
                                                "counts_pubrule_svm_1000", R, oofs)
        R["counts_all_svm"] = run_grid(df, folds, ["n_fix", "n_visit"], "svm", 1000, "counts_all_svm_1000", R, oofs)
        tick("baseline")
    if "rate" in stages:
        R["rate_curve"] = {}
        for rate in RATES:
            R["rate_curve"][rate] = {"svm": run_grid(df, folds, P4, "svm", rate, f"P4_svm_{rate}", R, oofs)}
        for rate in RATES:
            R["rate_curve"][rate]["lr_ALL_z"] = run_grid(df, folds, FEATURE_SETS["ALL_z"] if rate >= 500 else
                                                          [f for f in FEATURE_SETS["ALL_z"] if f != "msacc_rate_z"],
                                                          "lr", rate, f"ALLz_lr_{rate}", R, oofs)
        tick("rate")
    if "features" in stages:
        R["feature_sets"] = {}
        for fsn in ("P4+bins", "NEW", "pupil_only", "firstlook_only", "P4+NEW", "ALL", "P4_z", "ALL_z"):
            for model in ("lr", "svm") if fsn in ("P4_z", "ALL_z") else ("lr",):
                R["feature_sets"][f"{fsn}|{model}"] = run_grid(df, folds, FEATURE_SETS[fsn], model, 1000,
                                                               f"{fsn}_{model}_1000", R, oofs)
        R["feature_sets"]["P4|lr"] = run_grid(df, folds, P4, "lr", 1000, "P4_lr_1000", R, oofs)
        tick("features")
    if "transfer" in stages:
        R["transfer"] = {}
        for fsn in ("P4", "P4_z", "ALL_z"):
            for rate in (1000, 60):
                feats = [f for f in FEATURE_SETS[fsn] if not (rate < 500 and f.startswith("msacc"))]
                R["transfer"][f"{fsn}|lr|{rate}"] = cached(f"transfer_{fsn}_lr_{rate}", lambda feats=feats, rate=rate, fsn=fsn:
                                                           stage_transfer(df, feats, "lr", rate, f"transfer {fsn} lr {rate}"))
        R["transfer"]["book|svm|1000"] = cached("transfer_book_svm_1000", lambda: stage_transfer(
            df.assign(**{"B_" + c: df["book_" + c] for c in BOOK_FEATURES}), ["B_" + c for c in BOOK_FEATURES],
            "svm", 1000, "transfer BOOK svm 1000"))
        tick("transfer")
    if "models" in stages:
        R["models"] = {"lgbm|P4": run_grid(df, folds, P4, "lgbm", 1000, "P4_lgbm_1000", R, oofs),
                       "lgbm|ALL_z": run_grid(df, folds, FEATURE_SETS["ALL_z"], "lgbm", 1000, "ALLz_lgbm_1000", R, oofs)}
        tick("models")
    if "stats" in stages:
        base = "P4_svm_1000"
        cands = [k for k in ("ALL_z_lr_1000", "ALL_z_svm_1000", "P4_z_lr_1000", "P4_z_svm_1000", "P4+bins_lr_1000",
                             "P4+NEW_lr_1000", "ALL_lr_1000", "P4_lr_1000", "P4_lgbm_1000", "ALLz_lgbm_1000",
                             "P4_svm_60", "P4_svm_30", "P4_svm_120", "ALLz_lr_60")
                 if f"{k}|64_g1" in R["_per_participant_auc"]]
        R["paired_tests_vs_P4_svm_1000"] = paired_tests(R, base, cands)
        R["paired_tests_vs_book_svm"] = paired_tests(R, "book_svm", [base] + cands)
        for k, v in R["paired_tests_vs_P4_svm_1000"].items():
            print(f"[stats] {k}: d_ppAUC {v['mean_diff_pp_auc']:+.4f} p {v['p_raw']:.4f} holm {v['p_holm']:.4f}", flush=True)
        # CIs for honest baseline, deployable baseline and best candidate
        best = max(cands, key=lambda c: np.mean([R["_per_participant_auc"][f"{c}|{s}_g{g}"][p]
                                                 for s, g in SETS for p in R["_per_participant_auc"][f"{c}|{s}_g{g}"]]))
        R["best_candidate"] = best
        R["ci"] = {}
        for tag in ("book_svm", base, best):
            R["ci"][tag] = {}
            for s, g in SETS:
                k = f"{s}_g{g}"
                if (tag, k) not in oofs:
                    continue
                o = oofs[(tag, k)]
                c = cit_scores(o.groupby(["participant", "image"]).agg(y=("y", "first"), score=("score", "mean")).reset_index())
                R["ci"][tag][k] = {"image_auc_cluster_boot_95": cluster_bootstrap_auc(o, n_boot=500),
                                   "participant_cit_bootstrap_ci": bootstrap_ci(c.y.to_numpy(), c.score.to_numpy(), n_boot=1000)}
        print("[stats] best candidate:", best, flush=True)
        spec = R["_tags"][best]
        R["permutation_best"] = cached(f"permutation_{best}", lambda: permutation_test(df, spec["features"], folds, spec["rate"]))
        print("[stats] permutation:", R["permutation_best"], flush=True)
        tick("stats")
    if "identity" in stages:
        R["identity_probe"] = cached("identity_probe", lambda: identity_probe(a.derived))
        tick("identity")
    if "efficiency" in stages:
        b = R.get("best_candidate")
        R["efficiency"] = cached("efficiency", lambda: efficiency(df, a.raw_dir, (b, R["_tags"][b]) if b else None))
        tick("efficiency")
    R["published"] = {f"{s}_g{g}": v for (s, g), v in PUBLISHED.items()}
    R["timing_s"] = timing
    if oofs:
        pd.concat([o.assign(tag=t, set=k) for (t, k), o in oofs.items()]).to_csv(out / "oof_scores.csv.gz", index=False)
    (out / "results.json").write_text(json.dumps(R, indent=1, default=str))
    print("done in", round(time.perf_counter() - t0, 1), "s")


if __name__ == "__main__":
    main()
