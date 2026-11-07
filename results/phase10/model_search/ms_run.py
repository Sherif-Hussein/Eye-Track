"""Task M driver. EXPLORATORY. Development datasets only.

Usage (project root, one process, one thread):
  .venv/Scripts/python.exe -I results/phase10/model_search/ms_run.py repro  D30
  .venv/Scripts/python.exe -I results/phase10/model_search/ms_run.py repro  W30 <draw>
  .venv/Scripts/python.exe -I results/phase10/model_search/ms_run.py repro  C30 [repeat]
  .venv/Scripts/python.exe -I results/phase10/model_search/ms_run.py run    <D30|W15|C30> <cand,cand,...|all> [draw] [--sets a,b] [--repeats 0,1] [--folds n] [--frac 0.5] [--ck dir]
  .venv/Scripts/python.exe -I results/phase10/model_search/ms_run.py ens    <D30|W15|C30> [draw] [--sets ...] [--repeats ...] [--ck dir]
Stages are resumable (a finished candidate x set is skipped). Nothing outside results/phase10/model_search/ is written.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ms_lib as L  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pickle  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

POOL_AB = ["M1_cl", "M2_pw", "F3a_scan", "F3b_scan+base", "F4a_pca", "F4b_pca+base", "F5a_c22", "F5b_c22+base"]   # the A-B candidates (vote pool)
POOL_AB_CHEN = ["F4a_pca", "F5a_c22", "F5b_c22+base"]
STACK_EXTRA = ["C6_enet", "C7_rf"]


def cond_tag(cond, draw):
    return cond if draw is None else f"{cond}_d{draw}"


def need_c22(cands):
    return any("c22" in c for c in cands)


def sets_of(cond, sets=None):
    return ["chen"] if cond.startswith("C") else (sets or L.SETS)


def get_bundle(cond_obj, cond, sk, want, tag=None):
    B = L.chen_bundle(want=want) if cond.startswith("C") else L.delmas_set_bundle(cond_obj, sk, want=want)
    if tag is not None:
        meta = L.CK / tag / f"meta_{sk}.pkl"
        meta.parent.mkdir(parents=True, exist_ok=True)
        if not meta.exists():
            L.save_ck(meta, {"part": B.part, "y": B.y, "image": B.image if hasattr(B, "image") else np.arange(B.n)})
    return B


def run_candidates(cond, cands, draw=None, sets=None, repeats=(0, 1, 2), folds_limit=None, frac=1.0):
    kind = "chen" if cond.startswith("C") else "delmas"
    reg = L.spec_registry(kind)
    cands = list(reg) if cands == ["all"] else cands
    want = ("scan", "tc", "c22") if need_c22(cands) else ("scan", "tc")
    tag = cond_tag(cond, draw)
    cobj = None if kind == "chen" else L.DelmasCond(cond, draw=draw)
    for sk in sets_of(cond, sets):
        todo = [c for c in cands if not L.ck_path(tag, c, sk, frac).exists()]
        if not todo:
            continue
        B = get_bundle(cobj, cond, sk, want, tag)
        for c in todo:
            t = time_now()
            outs = L.run_set(reg[c], B, repeats=repeats, folds_limit=folds_limit, frac=frac)
            L.save_ck(L.ck_path(tag, c, sk, frac), outs)
            L.log(f"{tag} {c:15s} {sk:6s} done in {time_now() - t:6.1f}s")


def time_now():
    import time
    return time.time()


def load_outs(tag, cand, sk, frac=1.0):
    return pickle.load(open(L.ck_path(tag, cand, sk, frac), "rb"))


# ----------------------------------------------------------------------------- reproduction checks
def repro(cond, draw, repeat):
    from sklearn.metrics import roc_auc_score
    kind = "chen" if cond.startswith("C") else "delmas"
    spec = L.spec_registry(kind)["F0"]
    ok = True
    if cond == "D30":
        cobj = L.DelmasCond("D30")
        frames = []
        for sk in L.SETS:
            B = L.delmas_set_bundle(cobj, sk, want=("tc",))
            frames.append(L.oof_frame(B, L.run_set(spec, B)))
        mine = pd.concat(frames)
        ref = pd.read_csv(L.LRC.STORED / "checkpoints" / "ALLz_lr_30_oof.csv.gz")
        keys = ["set", "repeat", "participant", "image", "y"]
        m = mine.merge(ref, on=keys, suffixes=("", "_ref"))
        md = float(np.abs(m.score - m.score_ref).max())
        auc = L.LRC.pooled_auc(mine)
        res = {"rows": len(mine), "rows_matched": len(m), "max_abs_score_diff": md, "pooled_auc": auc, "stored_auc": 0.7854480364491726}
        ok = len(m) == len(mine) == len(ref) and md < 1e-9 and abs(auc - 0.7854480364491726) < 1e-9
    elif cond == "W30":
        cobj = L.DelmasCond("W30", draw=draw)
        frames = []
        for sk in L.SETS:
            B = L.delmas_set_bundle(cobj, sk, want=("tc",))
            frames.append(L.oof_frame(B, L.run_set(spec, B)))
        mine = pd.concat(frames)
        ref = pd.read_csv(L.ROOT / "results" / "phase9" / "low_rate" / "checkpoints_webcam" / f"sbnp_anch_30Hz_d{draw}_oof.csv.gz")
        keys = ["set", "repeat", "participant", "image", "y"]
        m = mine.merge(ref, on=keys, suffixes=("", "_ref"))
        md = float(np.abs(m.score - m.score_ref).max())
        res = {"draw": draw, "rows": len(mine), "rows_matched": len(m), "max_abs_score_diff": md, "pooled_auc": L.LRC.pooled_auc(mine),
               "stored_pooled_auc": L.LRC.pooled_auc(ref)}
        ok = len(m) == len(mine) == len(ref) and md < 1e-9
    else:
        B = L.chen_bundle(want=("tc",))
        reps = [repeat] if repeat is not None else [0, 1, 2]
        outs = L.run_set(spec, B, repeats=reps)
        md = 0.0
        for r in reps:
            ref = pickle.load(open(L.ROOT / "results" / "phase4" / "unified" / "checkpoints" / f"chen_unified__f100__r{r}.pkl", "rb"))
            for a, b in zip(ref, outs[r]):
                md = max(md, float(np.abs(a["score"] - b["score"]).max()), float(np.abs(a["inner_oof"] - b["inner_oof"]).max()))
        res = {"repeats": reps, "max_abs_score_and_inner_oof_diff": md}
        if len(reps) == 3:
            o = L.oof_frame(B, outs)
            avg = o.groupby("image", as_index=False).agg(y=("y", "first"), score=("score", "mean"))
            res["trial_auc_pooled_over_participants"] = float(roc_auc_score(avg.y, avg.score))
            res["stored_trial_auc"] = 0.6015502983896992
        ok = md < 1e-9
    res["passed"] = bool(ok)
    name = cond + ("" if draw is None else f"_d{draw}") + ("" if not (cond == "C30" and repeat is not None) else f"_r{repeat}")
    L.LRC.jdump(L.OUT / f"repro_{name}.json", res)
    L.log("REPRO", cond, res)
    if not ok:
        L.log("REPRODUCTION FAILED - stopping")
        sys.exit(2)


# ----------------------------------------------------------------------------- nested ensembles from stored fold records
def zvec(v, part):
    return L.zwithin(np.asarray(v, float)[:, None], part)[:, 0]


def ens_set(tag, sk, B, pool, repeats):
    """Vote (frozen + best of pool by inner AUC) and stack (level-2 LR on inner-out-of-fold scores). Returns fold records of both."""
    recs = {m: load_outs(tag, m, sk) for m in ["F0"] + pool + STACK_EXTRA if L.ck_path(tag, m, sk).exists()}
    members_stack = [m for m in recs]
    vote, stack = {}, {}
    for r in repeats:
        vote[r], stack[r] = [], []
        for f in range(len(recs["F0"][r])):
            base = recs["F0"][r][f]
            tr, te = base["tr"], base["te"]
            for m in recs:
                assert np.array_equal(recs[m][r][f]["tr"], tr) and np.array_equal(recs[m][r][f]["te"], te)
            zt = {m: zvec(recs[m][r][f]["score"], B.part[te]) for m in recs}
            zi = {m: zvec(recs[m][r][f]["inner_oof"], B.part[tr]) for m in recs}
            # vote: best of the pool by pooled inner AUC (training people only)
            cand = [m for m in pool if m in recs]
            best = max(cand, key=lambda m: (B.auc_fn(B.y[tr], zi[m]), -cand.index(m)))
            vote[r].append({"tr": tr, "te": te, "score": (zt["F0"] + zt[best]) / 2, "inner_oof": (zi["F0"] + zi[best]) / 2, "hp": best})
            # stack: level-2 L2 logistic regression, C chosen by participant-grouped inner folds on the inner-OOF matrix
            Ztr = np.column_stack([zi[m] for m in members_stack])
            Zte = np.column_stack([zt[m] for m in members_stack])
            cv = StratifiedGroupKFold(B.inner_k(len(np.unique(B.part[tr]))), shuffle=True, random_state=B.seed_fn(r, f))
            inner = list(cv.split(tr, B.y[tr], B.part[tr]))
            Cs = [0.01, 0.1, 1.0]
            oof = np.full((len(tr), len(Cs)), np.nan)
            aucs = np.full((len(inner), len(Cs)), np.nan)
            for fi, (a, b) in enumerate(inner):
                for j, C in enumerate(Cs):
                    mdl = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Ztr[a], B.y[tr][a])
                    oof[b, j] = mdl.decision_function(Ztr[b])
                    aucs[fi, j] = B.auc_fn(B.y[tr][b], oof[b, j]) if len(np.unique(B.y[tr][b])) == 2 else np.nan
            j = int(np.argmax(np.where(np.isfinite(B.cv_mean(aucs, axis=0)), B.cv_mean(aucs, axis=0), -np.inf)))
            mdl = LogisticRegression(C=Cs[j], class_weight="balanced", max_iter=2000).fit(Ztr, B.y[tr])
            stack[r].append({"tr": tr, "te": te, "score": mdl.decision_function(Zte), "inner_oof": oof[:, j], "hp": Cs[j],
                             "coef": dict(zip(members_stack, [float(c) for c in mdl.coef_[0]]))})
    return vote, stack


def run_ens(cond, draw=None, sets=None, repeats=(0, 1, 2)):
    tag = cond_tag(cond, draw)
    kind = "chen" if cond.startswith("C") else "delmas"
    pool = POOL_AB_CHEN if kind == "chen" else POOL_AB
    cobj = None if kind == "chen" else L.DelmasCond(cond, draw=draw)
    for sk in sets_of(cond, sets):
        if L.ck_path(tag, "E9_vote", sk).exists() and L.ck_path(tag, "E10_stack", sk).exists():
            continue
        B = get_bundle(cobj, cond, sk, ("scan", "tc"), tag)
        vote, stack = ens_set(tag, sk, B, pool, repeats)
        L.save_ck(L.ck_path(tag, "E9_vote", sk), vote)
        L.save_ck(L.ck_path(tag, "E10_stack", sk), stack)
        L.log(f"{tag} ensembles {sk} done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("cond")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--sets", default=None)
    ap.add_argument("--repeats", default="0,1,2")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--ck", default=None)
    a = ap.parse_args()
    if a.ck:
        L.CK = Path(a.ck)
    sets = a.sets.split(",") if a.sets else None
    reps = tuple(int(x) for x in a.repeats.split(","))
    if a.stage == "repro":
        draw = int(a.rest[0]) if a.rest and a.cond == "W30" else None
        rep = int(a.rest[0]) if a.rest and a.cond == "C30" else None
        repro(a.cond, draw, rep)
    elif a.stage == "run":
        cands = a.rest[0].split(",")
        draw = int(a.rest[1]) if len(a.rest) > 1 else None
        run_candidates(a.cond, cands, draw, sets, reps, a.folds, a.frac)
    elif a.stage == "ens":
        draw = int(a.rest[0]) if a.rest else None
        run_ens(a.cond, draw, sets, reps)


if __name__ == "__main__":
    main()
