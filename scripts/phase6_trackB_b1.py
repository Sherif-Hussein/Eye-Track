"""Phase 6 Track B, analysis B1: refit-aware label-permutation tests for the seven pre-registered Phase 4 primary tests.

Pre-registration: results/phase6/trackB/PREREG.md. Every permutation re-runs the whole nested pipeline (folds rebuilt where the design
needs class-paired folds, C and all other hyperparameters re-tuned) for the honest baseline and for the unified pipeline.

Usage (python -I scripts/phase6_trackB_b1.py <mode> [options]):
  check                      identity run (real labels) of every task; compares with the stored Phase 4 values
  time   --scratch DIR       one permutation per task, results to a scratch folder (timing only)
  run    [--workers 7]       production: perms/<task>.jsonl in results/phase6/trackB (resumable)
         [--n delmas=300,chen=...]   per-task N (from PREREG Amendment 1)
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = ROOT / "results" / "phase6" / "trackB"
SEED_BASE = 20261201
TASK_IDS = {"delmas": 1, "chen": 2, "celniak": 3, "preti": 4, "mazza_g2_vs_g3": 5, "mazza_guilty_vs_innocent": 6}
BLOCK = 25
R1 = 1   # repeats per permutation for delmas, chen, celniak, preti (PREREG Amendment 2 / DEVIATIONS.md); mazza keeps its stored repeats
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name.replace(".py", "")] = m
    spec.loader.exec_module(m)
    return m


def rng_for(task, i):
    return np.random.default_rng([SEED_BASE, TASK_IDS[task], i])


def U():
    from edd import unified
    return unified


def fast_auc(y, s):
    return U().fast_auc(y, s)


def unified_scores(t, y, folds, n_rep=None):
    """Re-run the unified nested pipeline (exactly `unified.run_task` without checkpoints and without joblib).
    Returns per repeat the list of fold outputs and the row-level score array of every repeat."""
    u = U()
    g = np.asarray(t["g"]).astype(str)
    y = np.asarray(y).astype(int)
    res, rows = {}, []
    for r, fl in enumerate(folds):
        if n_rep is not None and r >= n_rep:
            break
        out = []
        sc = np.full(len(y), np.nan)
        for f, tu in enumerate(fl):
            d = u._fold(t["X"], y, g, [str(x) for x in tu], 1000 * r + f, 1.0, 7000 + 1000 * r + f, False)
            sc[d["te"]] = d["score"]
            out.append(d)
        res[r] = out
        rows.append(sc)
    return res, np.vstack(rows)


def unit_mean_auc(units, y, score_by_repeat):
    """Participant-level AUC as in unified._unit_mean_scores: mean over rows within a repeat, then over repeats."""
    parts = []
    for r, sc in enumerate(score_by_repeat):
        parts.append(pd.DataFrame({"u": units, "y": y, "s": sc}).groupby("u").agg(y=("y", "first"), s=("s", "mean")).assign(r=r))
    a = pd.concat(parts).reset_index().groupby("u").agg(y=("y", "first"), s=("s", "mean"))
    return fast_auc(a.y.to_numpy(), a.s.to_numpy())


# =============================================================================== Delmas
class DelmasSetup:
    def __init__(self):
        u = U()
        self.dr = load_script("phase2_delmas_raw.py")
        self.tasks = u.build_delmas(ROOT / "results" / "phase4" / "unified" / "cache" / "delmas_30hz_features.csv", lambda *a: None)
        df = self.dr.build_tables(ROOT / "data" / "derived", ROOT / "data" / "delmas" / "BookFIP_64_trials.xlsx",
                                  ROOT / "data" / "delmas" / "BookFIP_32_trials.xlsx")
        self.base = {}
        self.folds_dr = json.loads((ROOT / "results" / "phase2" / "delmas_raw" / "folds.json").read_text())["repeats"]
        for s, g in self.dr.SETS:
            key = f"{s}_g{g}"
            d = self.dr.subset(df, s, g, 1000)[["participant", "image", "face"] + self.dr.P4].reset_index(drop=True)
            t = self.tasks[f"delmas_{key}"]
            ku = pd.MultiIndex.from_arrays([t["g"].astype(int), t["meta"].image.to_numpy()])
            kb = pd.MultiIndex.from_arrays([d.participant.to_numpy(), d.image.to_numpy()])
            assert len(ku) == len(kb) and set(ku) == set(kb), f"Delmas {key}: unified and baseline rows differ"
            # baseline row -> position in the unified table
            pos = pd.Series(np.arange(len(ku)), index=ku).loc[kb].to_numpy()
            self.base[key] = {"d": d, "pos": pos}
            assert (t["y"][pos] == d.face.to_numpy()).all()
        del df

    def run(self, i):
        """i = None: real labels. Returns the dict of statistics."""
        rng = None if i is None else rng_for("delmas", i)
        per = {}
        for s, gp in self.dr.SETS:
            key = f"{s}_g{gp}"
            t = self.tasks[f"delmas_{key}"]
            y = t["y"].copy()
            if rng is not None:
                g = t["g"]
                for p in np.unique(g):
                    ix = np.flatnonzero(g == p)
                    y[ix] = rng.permutation(y[ix])
            # unified
            _, sc = unified_scores(t, y, t["folds"][:R1])
            auc_u = fast_auc(y, sc.mean(0))
            # baseline (P4 SVM 1000 Hz), same folds, labels mapped by (participant, image)
            b = self.base[key]
            d = b["d"]
            yb = y[b["pos"]]
            X, gb = d[self.dr.P4].to_numpy(float), d.participant.to_numpy()
            sb = np.full((R1, len(d)), np.nan)
            for r, fl in enumerate(self.folds_dr[key][:R1]):
                for tu in fl:
                    idx, scr = self.dr._fold_task(X, yb, gb, tu, "svm", r)
                    sb[r, idx] = scr
            auc_b = fast_auc(yb, sb.mean(0))
            per[key] = (auc_b, auc_u)
        ab = float(np.mean([v[0] for v in per.values()]))
        au = float(np.mean([v[1] for v in per.values()]))
        return {"delmas_noninferiority": {"base": ab, "ours": au, "delta": au - ab,
                                          "per_set_base": {k: v[0] for k, v in per.items()}, "per_set_ours": {k: v[1] for k, v in per.items()}}}


# =============================================================================== Chen
class ChenSetup:
    def __init__(self):
        from edd import chen_confirm as cc
        u = U()
        tasks, (self.meta, self.ch) = u.build_chen(lambda *a: None)
        self.t = tasks["chen_unified"]
        self.cc = cc
        self.units = self.meta.unit.to_numpy()
        self.exps = self.meta.exp.to_numpy()
        self.rate = self.ch["rate"]
        self.T1_real = self.t["T1"].copy()

    def rate_rule_persons(self, T1, y, folds):
        cc = self.cc
        P = cc.p2()
        units, exps = self.units, self.exps
        rows = []
        for r, fl in enumerate(folds):
            for f, tu in enumerate(fl):
                te_all = np.isin(units.astype(str), [str(x) for x in tu])
                for e in (1, 2):
                    tr = np.flatnonzero(~te_all & (exps == e))
                    te = np.flatnonzero(te_all & (exps == e))
                    score, stat, oof, params = cc._peak_arm(self.rate, y, units, tr, te)
                    null = cc.null_from_oof(oof, T1[tr], units[tr])
                    tau = cc.threshold_for_fpr(null)
                    for uu in np.unique(units[te]):
                        m = units[te] == uu
                        d = cc.person_z(stat[m], T1[te][m])
                        rows.append({"repeat": r, "unit": int(uu), "z_fam": d["z_fam"], "z_c1": d["z_c1"], "z_c2": d["z_c2"],
                                     "detect": float(d["z_fam"] > tau), "fpr": float(((d["z_c1"] > tau) + (d["z_c2"] > tau)) / 2)})
        return pd.DataFrame(rows)

    def person_auc_from(self, df, zcols):
        long = pd.concat([pd.DataFrame({"unit": df.unit, "kind": k, "stat": df[c]}) for k, c in zcols])
        d = long.groupby(["unit", "kind"], as_index=False).agg(s=("stat", "mean"))
        d["y"] = (d.kind == "fam").astype(int)
        ok = np.isfinite(d.s.to_numpy())
        return fast_auc(d.y.to_numpy()[ok], d.s.to_numpy()[ok])

    def run(self, i):
        u = U()
        t = dict(self.t)
        T1 = self.T1_real.copy()
        if i is not None:
            rng = rng_for("chen", i)
            g = t["g"]
            for p in np.unique(g):
                ix = np.flatnonzero(g == p)
                T1[ix] = rng.permutation(T1[ix])
        y = (T1 == 2).astype(int)
        t["T1"], t["y"] = T1, y
        res, _ = unified_scores(t, y, t["folds"][:R1])
        pu = u.persons_chen(t, res)
        det_u = float(u.chen_detection_matrix(pu).mean().mean())
        fpr_u = u.chen_realised_fpr(pu)
        fam = pu[pu.kind == "fam"].rename(columns={"stat": "z_fam"})[["unit", "repeat", "z_fam"]]
        c1 = pu[pu.kind == "c1"].rename(columns={"stat": "z_c1"})[["unit", "repeat", "z_c1"]]
        c2 = pu[pu.kind == "c2"].rename(columns={"stat": "z_c2"})[["unit", "repeat", "z_c2"]]
        pj = fam.merge(c1, on=["unit", "repeat"]).merge(c2, on=["unit", "repeat"])
        auc_u = self.person_auc_from(pj, [("fam", "z_fam"), ("c1", "z_c1"), ("c2", "z_c2")])
        pb = self.rate_rule_persons(T1, y, t["folds"][:R1])
        det_b, fpr_b = float(pb.detect.mean()), float(pb.fpr.mean())
        auc_b = self.person_auc_from(pb, [("fam", "z_fam"), ("c1", "z_c1"), ("c2", "z_c2")])
        return {"chen_detection_fpr7": {"base": det_b, "ours": det_u, "delta": det_u - det_b, "fpr_base": fpr_b, "fpr_ours": fpr_u,
                                        "auc_base": auc_b, "auc_ours": auc_u, "auc_delta": auc_u - auc_b}}


# =============================================================================== Celniak
class CelniakSetup:
    def __init__(self):
        u = U()
        self.sc = load_script("phase2_celniak_screen.py")
        self.slides = pd.read_csv(ROOT / "data" / "derived" / "celniak_slide_features.csv")
        self.t = u.build_celniak(ROOT / "results" / "phase4" / "unified" / "cache" / "celniak_fast_slides.csv", lambda *a: None)["celniak_unified"]
        from edd.celniak import PAPER_FEATURES
        self.feats = PAPER_FEATURES
        s = self.slides
        self.per_p = s.groupby("recording")["label"].first()
        first_exam = s.groupby("recording")["first_exam"].first()
        self.order = (first_exam == "Eyetracker").astype(int)
        self.order.index = self.order.index.astype(str)
        self.frozen = json.loads((ROOT / "results" / "phase2" / "celniak" / "folds.json").read_text())["repeats"]

    def folds_for(self, per_p):
        return [self.sc.order_matched_folds(per_p, self.order.loc[per_p.index], 2026 + r) for r in range(R1)]

    def run(self, i):
        from edd.harness import aggregate, grouped_inner, make_search, nested_fit, run_outer
        u = U()
        per_p = self.per_p.copy()
        if i is not None:
            per_p = pd.Series(rng_for("celniak", i).permutation(per_p.to_numpy()), index=per_p.index)
        folds = self.folds_for(per_p)
        s = self.slides.copy()
        s["label"] = s["recording"].map(per_p).to_numpy(int)
        fit = nested_fit(lambda: make_search(self.sc.svm_pipe(), self.sc.SVM_GRID, grouped_inner(self.sc.INNER_SPLITS)))
        tr = run_outer(s, self.feats, "label", "recording", folds, fit, n_jobs=1)
        a = tr.groupby(["repeat", "unit"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))
        a = a.groupby("unit", as_index=False).agg(y=("y", "first"), score=("score", "mean"))
        auc_b = fast_auc(a.y.to_numpy(), a.score.to_numpy())
        t = self.t
        g = np.asarray(t["g"]).astype(str)
        y = per_p.loc[g].to_numpy(int)
        _, sc = unified_scores(t, y, folds)
        auc_u = unit_mean_auc(g, y, sc)
        return {"celniak_participant_auc": {"base": auc_b, "ours": auc_u, "delta": auc_u - auc_b}}


# =============================================================================== Preti
class PretiSetup:
    def __init__(self):
        from edd import preti as P
        u = U()
        self.P = P
        xl = P.load_workbook(ROOT / "data" / "preti" / "preti2026.xlsx")
        trials = P.trials_from_fixations(P.fixations(xl))
        self.pub = P.published_style_features(trials)
        self.subj = trials.groupby("subject").y.first()
        PUB = [f"{m}_{q}" for m in ("RT", "blink_rate", "pupil") for q in P.QT]
        self.sets = {"B_fusion": PUB, "B_ocular": [f for f in PUB if not f.startswith("RT")]}
        self.grid = P.family_grid(("lr", "svm"))
        tasks = u.build_preti(lambda *a: None)
        self.tA, self.tB = tasks["preti_A_ocular"], tasks["preti_B_ocular_rt"]
        self.g = np.asarray(self.tA["g"]).astype(str)
        assert (np.asarray(self.tB["g"]).astype(str) == self.g).all()
        self.units = self.subj.index.astype(str).to_numpy()
        self.frozen = json.loads((ROOT / "results" / "phase2" / "preti" / "folds.json").read_text())["repeats"]

    def run(self, i):
        from edd.folds import make_repeated_folds
        u = U()
        lab = self.subj.to_numpy(int).copy()
        if i is not None:
            lab = rng_for("preti", i).permutation(lab)
        by_unit = dict(zip(self.units, lab))
        folds = make_repeated_folds(self.units, lab, R1, base_seed=2026)
        pub = self.pub.copy()
        pub["y"] = pub["subject"].astype(str).map(by_unit).to_numpy(int)
        out = {}
        y_u = np.array([by_unit[x] for x in self.g], int)
        for tname, bname, t in (("preti_A_participant_auc", "B_ocular", self.tA), ("preti_B_participant_auc", "B_fusion", self.tB)):
            parts = self.P.run_subject_level(pub, self.sets[bname], folds, self.grid, n_jobs=1)
            a = parts.groupby(["repeat", "unit"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))
            a = a.groupby("unit", as_index=False).agg(y=("y", "first"), score=("score", "mean"))
            auc_b = fast_auc(a.y.to_numpy(), a.score.to_numpy())
            _, sc = unified_scores(t, y_u, folds)
            auc_u = unit_mean_auc(self.g, y_u, sc)
            out[tname] = {"base": auc_b, "ours": auc_u, "delta": auc_u - auc_b}
        return out


# =============================================================================== Mazza
class MazzaSetup:
    def __init__(self, task):
        from sklearn.model_selection import StratifiedKFold
        from edd import mazza as mz
        u = U()
        self.mz, self.task, self.SKF = mz, task, StratifiedKFold
        self.cfgmod = load_script("phase2_mazza_configs.py")
        d = mz.load(ROOT / "data" / "mazza" / "data.csv")
        self.raw = mz.raw_columns(d)
        groups, pos, _ = mz.TASKS[task]
        self.m = d.group.isin(groups).to_numpy()
        self.s = d[self.m].reset_index(drop=True)
        self.pos = pos
        self.Xraw = self.s[self.raw].to_numpy(float)
        self.g = self.s.unit.astype(str).to_numpy()
        self.ugroup = self.s.group.to_numpy()
        ut = u.build_mazza(lambda *a: None)[f"mazza_{task}"]
        # unified table rows are in the same order as `self.s` (both d[m] in file order)
        assert (np.asarray(ut["g"]).astype(str) == self.g).all()
        self.t = ut
        self.sub = {"g2_vs_g3": "g23", "guilty_vs_innocent": "all"}[task]
        self.frozen = json.loads((ROOT / "results" / "phase2" / "mazza" / "folds.json").read_text())["repeats"][self.sub]
        self.cfg = self.cfgmod.CONFIGS["base_pub36_lr_unreg"]

    def folds_for(self, grp):
        out = []
        for r in range(5):
            skf = self.SKF(10, shuffle=True, random_state=2026 + r)
            out.append([self.s.unit.iloc[te].tolist() for _, te in skf.split(self.s, grp)])
        return out

    def run(self, i):
        grp = self.ugroup.copy()
        if i is not None:
            grp = rng_for("mazza_" + self.task, i).permutation(grp)
        y = np.isin(grp, self.pos).astype(int)
        folds = self.folds_for(grp)
        idea, sets, model, select, _ = self.cfg
        _, sc = unified_scores(self.t, y, folds)
        auc_u = unit_mean_auc(self.g, y, sc)
        oof = self.mz.run_binary(self.Xraw, y, self.g, [[[str(u) for u in f] for f in rep] for rep in folds[:3]],
                                 self.cfgmod.build(self.raw, sets, model, select), n_jobs=1)
        a = oof.groupby(["repeat", "unit"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))
        a = a.groupby("unit", as_index=False).agg(y=("y", "first"), score=("score", "mean"))
        auc_b = fast_auc(a.y.to_numpy(), a.score.to_numpy())
        return {f"mazza_{self.task}_participant_auc": {"base": auc_b, "ours": auc_u, "delta": auc_u - auc_b}}


SETUPS = {"delmas": DelmasSetup, "chen": ChenSetup, "celniak": CelniakSetup, "preti": PretiSetup,
          "mazza_g2_vs_g3": lambda: MazzaSetup("g2_vs_g3"), "mazza_guilty_vs_innocent": lambda: MazzaSetup("guilty_vs_innocent")}
_CACHE = {}


def setup_for(task):
    if task not in _CACHE:
        _CACHE[task] = SETUPS[task]()
    return _CACHE[task]


def run_one(job):
    task, i = job
    t0 = time.time()
    c0 = time.process_time()
    for attempt in (1, 2):
        try:
            res = setup_for(task).run(i)
            return {"task": task, "perm": i, "seconds": time.time() - t0, "cpu_seconds": time.process_time() - c0, "tests": res, "attempt": attempt}
        except Exception:
            err = traceback.format_exc()
            if attempt == 2:
                return {"task": task, "perm": i, "error": err, "seconds": time.time() - t0}


# =============================================================================== modes
STORED = {"delmas_noninferiority": ("delmas_noninferiority", "est_base", "est_unified"),
          "celniak_participant_auc": ("celniak_participant_auc", "est_base", "est_unified"),
          "preti_A_participant_auc": ("preti_A_participant_auc", "est_base", "est_unified"),
          "preti_B_participant_auc": ("preti_B_participant_auc", "est_base", "est_unified"),
          "mazza_g2_vs_g3_participant_auc": ("mazza_g2_vs_g3_participant_auc", "est_base", "est_unified"),
          "mazza_guilty_vs_innocent_participant_auc": ("mazza_guilty_vs_innocent_participant_auc", "est_base", "est_unified")}


def _auc_units(df, y="y", s="score", unit="unit"):
    a = df.groupby(unit).agg(y=(y, "first"), s=(s, "mean"))
    return fast_auc(a.y.to_numpy(), a.s.to_numpy())


def stored_repeat0():
    """Stored Phase 2 / Phase 4 out-of-fold scores restricted to repeat 0 (no refitting): the observed values of the R1 = 1 tasks."""
    p2 = ROOT / "results" / "phase2"
    p4 = ROOT / "results" / "phase4" / "unified" / "oof"
    u = U()
    out = {}
    rd = lambda f: pd.read_csv(f)  # noqa: E731
    def r0(df):
        return df[df["repeat"] == 0]
    out["celniak_participant_auc"] = {"base": _auc_units(r0(rd(p2 / "celniak" / "trials_baseline_svm_paper.csv"))),
                                      "ours": _auc_units(r0(rd(p4 / "celniak_unified_rows.csv.gz")))}
    out["preti_A_participant_auc"] = {"base": _auc_units(r0(rd(p2 / "preti" / "scores" / "B_ocular.csv"))),
                                      "ours": _auc_units(r0(rd(p4 / "preti_A_ocular_rows.csv.gz")))}
    out["preti_B_participant_auc"] = {"base": _auc_units(r0(rd(p2 / "preti" / "scores" / "B_fusion.csv"))),
                                      "ours": _auc_units(r0(rd(p4 / "preti_B_ocular_rt_rows.csv.gz")))}
    base = r0(rd(p2 / "delmas_raw" / "oof_scores.csv.gz"))
    base = base[base.tag == "P4_svm_1000"]
    ab, au = [], []
    for s, g in (("64", 1), ("64", 2), ("64", 3), ("32", 3), ("32", 4), ("32", 5)):
        key = f"{s}_g{g}"
        b = base[base.set == key]
        ab.append(fast_auc(b.y.to_numpy(), b.score.to_numpy()))
        w = r0(rd(p4 / f"delmas_{key}_rows.csv.gz"))
        au.append(fast_auc(w.y.to_numpy(), w.score.to_numpy()))
    out["delmas_noninferiority"] = {"base": float(np.mean(ab)), "ours": float(np.mean(au))}
    pu = r0(rd(p4 / "chen_unified_persons.csv.gz"))
    fam = pu[pu.kind == "fam"]
    cp, _ = u.chen_checkpoint_persons()
    cb = cp[(cp.arm == "rate_rule") & (cp["repeat"] == 0)]
    out["chen_detection_fpr7"] = {"base": float(cb.detect_cal.mean()), "ours": float((fam.stat > fam.tau_fpr7).mean()),
                                  "fpr_base": float(cb.fpr_cal.mean()),
                                  "fpr_ours": float((pu[pu.kind.isin(["c1", "c2"])].stat > pu[pu.kind.isin(["c1", "c2"])].tau_fpr7).mean())}
    return out


def mode_check(a):
    stored = json.loads((ROOT / "results" / "phase4" / "unified" / "results.json").read_text())["primary_tests"]
    tasks = a.tasks.split(",") if a.tasks else list(SETUPS)
    R0 = stored_repeat0() if R1 == 1 else {}
    report = {}
    for task in tasks:
        t0 = time.time()
        res = setup_for(task).run(None)
        for tn, v in res.items():
            if tn in R0:
                s = R0[tn]
                diffs = {k: v[k] - s[k] for k in s}
            elif tn == "chen_detection_fpr7":
                s = stored[tn]["detection"]
                diffs = {"base_detection": v["base"] - s["rate_rule"], "ours_detection": v["ours"] - s["unified"],
                         "fpr_base": v["fpr_base"] - s["realised_fpr_rate_rule"], "fpr_ours": v["fpr_ours"] - s["realised_fpr_unified"]}
            else:
                s = stored[tn]
                diffs = {"base": v["base"] - s["est_base"], "ours": v["ours"] - s["est_unified"]}
            report[tn] = {"computed": v, "max_abs_diff_vs_stored": max(abs(x) for x in diffs.values()), "diffs": diffs}
            log(f"CHECK {tn}: " + json.dumps({k: (round(x, 9)) for k, x in diffs.items()}) + f"  [{time.time() - t0:.0f}s]")
    out = Path(a.scratch) / "identity_check.json" if a.scratch else OUT / "identity_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(report)
    out.write_text(json.dumps(old, indent=1, default=float))


def mode_time(a):
    out = Path(a.scratch)
    out.mkdir(parents=True, exist_ok=True)
    tasks = a.tasks.split(",") if a.tasks else list(SETUPS)
    timing = {}
    for task in tasks:
        t0 = time.time()
        s = setup_for(task)
        t_setup = time.time() - t0
        r = run_one((task, 10_000 + 1))               # scratch permutation index, never used in production
        timing[task] = {"setup_s": t_setup, "perm_s": r.get("seconds"), "error": r.get("error"), "tests": r.get("tests")}
        log(f"TIME {task}: setup {t_setup:.0f}s, one permutation {r.get('seconds'):.0f}s", "ERR" if r.get("error") else "")
        (out / "timing.json").write_text(json.dumps(timing, indent=1, default=float))


def done_set(task_dir, task):
    f = task_dir / f"{task}.jsonl"
    done = set()
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["perm"])
                except Exception:
                    pass
    return done


def mode_run(a):
    pdir = OUT / "perms"
    pdir.mkdir(parents=True, exist_ok=True)
    N = {t: 1000 for t in SETUPS}
    if a.n:
        for kv in a.n.split(","):
            k, v = kv.split("=")
            N[k] = int(v)
    order = a.order.split(",") if a.order else list(SETUPS)
    done = {t: done_set(pdir, t) for t in SETUPS}
    jobs = []
    maxb = max(N.values()) // BLOCK
    for b in range(maxb):
        for t in order:
            if (b + 1) * BLOCK <= N[t]:
                jobs += [(t, i) for i in range(b * BLOCK, (b + 1) * BLOCK) if i not in done[t]]
    if a.reverse:
        jobs = jobs[::-1]
    log(f"N per task {N}; {len(jobs)} permutations to run; already done { {t: len(v) for t, v in done.items()} }")
    counts = {t: len(done[t]) for t in SETUPS}
    tsum = {t: 0.0 for t in SETUPS}
    n_since = 0
    errs = 0
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap_unordered(run_one, jobs, chunksize=1):
            t = r["task"]
            if "error" in r:
                errs += 1
                log("ERROR", t, r["perm"], r["error"][-400:])
                with open(OUT / "perm_errors.log", "a") as fh:
                    fh.write(json.dumps(r) + "\n")
                continue
            with open(pdir / f"{t}.jsonl", "a") as fh:
                fh.write(json.dumps(r, default=float) + "\n")
                fh.flush()
            counts[t] += 1
            tsum[t] += r["seconds"]
            n_since += 1
            if n_since % BLOCK == 0:
                log("progress", json.dumps(counts), "mean s/perm", {k: round(tsum[k] / max(1, counts[k] - len(done[k])), 1) for k in counts if counts[k] > len(done[k])})
                (OUT / f"progress_{order[0]}.json").write_text(json.dumps({"tasks": order, "counts": counts, "N_target": N, "errors": errs, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    log("run finished", json.dumps(counts), "errors", errs)
    (OUT / f"progress_{order[0]}.json").write_text(json.dumps({"tasks": order, "counts": counts, "N_target": N, "errors": errs, "finished": True, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "time", "run"])
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--n", default=None)
    ap.add_argument("--order", default=None)
    ap.add_argument("--reverse", action="store_true", help="take the jobs from the end of the list (second worker group on the same task)")
    a = ap.parse_args()
    if a.mode == "check":
        mode_check(a)
    elif a.mode == "time":
        mode_time(a)
    else:
        mode_run(a)


if __name__ == "__main__":
    main()
