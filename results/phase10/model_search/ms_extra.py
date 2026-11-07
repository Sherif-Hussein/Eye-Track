"""Task M extras. EXPLORATORY. Development datasets only.
  displays : does the frozen model's person-level AUC rise with the number of displays per person (8, 16, 32, 64)?
  lc       : learning curve (fraction of the training participants) for named candidates.
  cost     : model size, ms per person, peak memory.
Usage: .venv/Scripts/python.exe -I results/phase10/model_search/ms_extra.py displays <D30|W15> [--sets a,b] [--ns 8,16] [--q 3] [--repeats 0,1,2] [--out name]
       .venv/Scripts/python.exe -I results/phase10/model_search/ms_extra.py lc <cond> <cand,cand> [--fracs 0.25,0.5,0.75] [--draw d]
       .venv/Scripts/python.exe -I results/phase10/model_search/ms_extra.py cost <D30|W15|C30> [--cands a,b] [--out name]
"""
import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ms_lib as L  # noqa: E402
import ms_eval as E  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pickle  # noqa: E402

NS = (8, 16, 32, 64)
DISPLAY_SEED = 20261108


def reduced_bundle(B, n_disp, q, sk):
    """Bundle where each person contributes only n_disp displays (half with a known item, half without; random, seeded), features rebuilt
    from those displays only (mean over the image's presentations in them, then within-person z)."""
    kn_any = np.zeros(len(B.t_rowmap), bool)
    sel = B.t_rowmap >= 0
    kn_any[sel] = B.y[B.t_rowmap[sel]] == 1
    kn_any = kn_any.reshape(-1, 4).any(1)
    selD = np.zeros(B.nd, bool)
    for p in np.unique(B.d_part):
        idx = np.flatnonzero(B.d_part == p)
        rng = np.random.default_rng([DISPLAY_SEED, n_disp, q, int(p)])
        fam, non = idx[kn_any[idx]], idx[~kn_any[idx]]
        k1 = min(len(fam), n_disp // 2)
        k0 = min(len(non), n_disp - k1)
        selD[rng.choice(fam, k1, replace=False)] = True
        selD[rng.choice(non, k0, replace=False)] = True
    selT = np.repeat(selD, 4)
    F = L.agg_to_rows(B.t_raw_base[selT], B.t_rowmap[selT], B.n)
    pres = np.bincount(B.t_rowmap[selT & (B.t_rowmap >= 0)], minlength=B.n) > 0
    B2 = L.Bundle()
    B2._cache = {}
    B2.kind, B2.key, B2.cond = "delmas", sk, B.cond
    B2.y, B2.part, B2.image, B2.n = B.y[pres], B.part[pres], B.image[pres], int(pres.sum())
    B2.X = {"base": L.zwithin(F[pres], B2.part)}
    B2.folds, B2.seed_fn, B2.inner_k, B2.auc_fn = B.folds, B.seed_fn, B.inner_k, B.auc_fn
    B2.grid_default, B2.cv_mean = B.grid_default, B.cv_mean
    return B2


def displays(cond, sets, ns, qn, repeats, out_name):
    spec = L.spec_registry("delmas")["F0"]
    draws = [None] if cond == "D30" else [1, 2, 3]
    res = {}
    for n in ns:
        fr_item, fr_person, fr_pt = [], [], []
        for d in draws:
            cobj = L.DelmasCond(cond, draw=d)
            for sk in sets:
                if n > (64 if sk.startswith("64") else 32):
                    continue
                B = L.delmas_set_bundle(cobj, sk, want=("tc",))
                for q in range(1, (1 if n == 64 else qn) + 1):
                    B2 = reduced_bundle(B, n, q, sk)
                    outs = L.run_set(spec, B2, repeats=repeats)
                    meta = SimpleNamespace(part=B2.part, y=B2.y, image=B2.image)
                    it = L.oof_frame(meta, outs, sk).assign(draw=d or 0, q=q)
                    pe = L.persons_cit2(meta, outs, "mean")
                    pa = pe.groupby(["unit", "kind"], as_index=False).agg(y=("y", "first"), score=("stat", "mean"))
                    pa["set"], pa["participant"], pa["image"] = sk, pa.unit, (pa.kind == "known").astype(int)
                    fr_item.append(it)
                    fr_person.append(pa[["set", "participant", "image", "y", "score"]].assign(draw=d or 0, q=q))
                    L.log(f"displays n={n} {cond} d={d} {sk} q={q}: images {B2.n}")
        res[n] = (fr_item, fr_person)
    # summarise: per n, AUC per configuration (draw x subset), mean over sets and configurations; participant-cluster bootstrap CI (500 resamples)
    def conf_auc(g):
        a = g.groupby(["participant", "image"], as_index=False).agg(y=("y", "first"), s=("score", "mean"))
        return L.roc_auc_score(a.y, a.s) if a.y.nunique() == 2 else np.nan

    rows = []
    for n, (fi, fp) in res.items():
        for level, frs in (("item", fi), ("person", fp)):
            allf = pd.concat(frs, ignore_index=True)
            confs = sorted(set(zip(allf.draw, allf.q)))
            sets_present = sorted(allf.set.unique())
            auc = float(np.mean([np.mean([conf_auc(allf[(allf.draw == d) & (allf.q == q) & (allf.set == sk)]) for sk in sets_present]) for d, q in confs]))
            parts = {sk: sorted(allf[allf.set == sk].participant.unique()) for sk in sets_present}
            rng = np.random.default_rng(E.BOOT_SEED + n)
            boots = []
            for _ in range(500):
                vals = []
                for sk in sets_present:
                    pk = [parts[sk][i] for i in rng.integers(0, len(parts[sk]), len(parts[sk]))]
                    sub = allf[allf.set == sk]
                    cs = []
                    for d, q in confs:
                        g = sub[(sub.draw == d) & (sub.q == q)]
                        m = pd.concat([g[g.participant == p_] for p_ in pk])
                        cs.append(conf_auc(m))
                    vals.append(np.nanmean(cs))
                boots.append(np.mean(vals))
            rows.append({"cond": cond, "n_displays": n, "level": level, "auc": auc, "auc_lo": float(np.percentile(boots, 2.5)),
                         "auc_hi": float(np.percentile(boots, 97.5)), "n_configs": len(confs), "sets": ",".join(sets_present)})
            L.log(rows[-1])
    od = L.OUT / out_name
    od.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(od / f"displays_{cond}.csv", index=False)


# ----------------------------------------------------------------------------- learning curve
def lc_table(cond, cands, fracs, sets_arg, out_name):
    """Item AUC (mean over sets; bootstrap CI) against the fraction of training participants kept; fraction 1.0 = the main run.
    Runs for the fractions < 1 are made with: ms_run.py run <cond> <cands> [draw] --frac f"""
    tags = E.tags_of(cond)
    sets = E.sets_of(cond, sets_arg)
    frames = {}
    for c in cands:
        for f in fracs:
            frames[f"{c}@{f}"] = [pd.concat([L.oof_frame(E.load_meta(t, sk), pickle.load(open(L.ck_path(t, c, sk, f), "rb")), sk) for sk in sets], ignore_index=True)
                                  for t in tags]
    sdb, owner, names = E.to_sets(frames, sets)
    pt, bt = E.point(sdb, owner, names), E.boot(sdb, owner, names, 2000, E.BOOT_SEED + 5)
    rows = [{"cond": cond, "cand": k.split("@")[0], "frac_train_participants": float(k.split("@")[1]), "item_auc": pt[k],
             "lo": float(np.percentile(bt[k], 2.5)), "hi": float(np.percentile(bt[k], 97.5))} for k in names]
    od = L.OUT / out_name
    od.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(od / f"learning_curve_{cond}.csv", index=False)
    for r in rows:
        L.log(r)


# ----------------------------------------------------------------------------- cost
def _time(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000)


def cost(cond, cands, out_name):
    kind = "chen" if cond.startswith("C") else "delmas"
    reg = L.spec_registry(kind)
    cands = cands or list(reg)
    rows = []
    if kind == "delmas":
        cobj = L.DelmasCond(cond, draw=None if cond == "D30" else 1)
        B = L.delmas_set_bundle(cobj, "64_g1", want=("scan", "tc", "c22"))
        pid = int(np.unique(B.part)[0])
        # extra-feature extraction cost for one person (64 trials), from the samples
        x, y, p, images, g = L.load_participant_samples(pid, cobj.fs)
        T = {"tc+scan": lambda: L.sample_blocks(x, y, cobj.fs)}
        tc, _ = L.sample_blocks(x, y, cobj.fs)
        L.catch22_rows(tc.reshape(-1, 1, L.NBIN)[:8])
        T["catch22"] = lambda: L.catch22_rows(tc.reshape(-1, 1, L.NBIN))
        feat_ms, feat_kib = {}, {}
        for k, fn in T.items():
            feat_ms[k] = _time(fn, 5)
            tracemalloc.start()
            fn()
            feat_kib[k] = tracemalloc.get_traced_memory()[1] / 1024
            tracemalloc.stop()
    else:
        B = L.chen_bundle(want=("tc", "c22"))
        pid = str(B.part[0])
        feat_ms, feat_kib = {}, {}
    f0 = reg["F0"]
    te_m = B.part == pid
    te = np.flatnonzero(te_m)
    tr = np.flatnonzero(~te_m)
    extra = {"M1_cl": [], "M2_pw": [], "F3a_scan": ["tc+scan"], "F3b_scan+base": ["tc+scan"], "F4a_pca": ["tc+scan"], "F4b_pca+base": ["tc+scan"],
             "F5a_c22": ["tc+scan", "catch22"], "F5b_c22+base": ["tc+scan", "catch22"], "C8_cnn": ["tc+scan"]}
    for c in cands:
        if c not in reg:
            continue
        spec = reg[c]
        hp = spec.grid(B)[0] if len(spec.grid(B)) == 1 else spec.grid(B)[len(spec.grid(B)) // 2]
        pred = spec.fit(B, tr, hp, 0)
        ms = _time(lambda: pred(te), 30)
        tracemalloc.start()
        pred(te)
        kib = tracemalloc.get_traced_memory()[1] / 1024
        tracemalloc.stop()
        ex = extra.get(c, [])
        rows.append({"cond": cond, "cand": c, "n_params_or_stored_numbers": int(spec.n_params(B)), "score_ms_per_person": ms, "score_peak_kib": kib,
                     "extra_feature_ms_per_person": float(sum(feat_ms.get(k, 0.0) for k in ex)),
                     "extra_feature_peak_kib": float(max([feat_kib.get(k, 0.0) for k in ex] + [0.0])),
                     "note": "model scoring from precomputed features of one person; extra features = samples -> block features beyond the frozen ones"})
        L.log(rows[-1])
    od = L.OUT / out_name
    od.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(od / f"cost_{cond}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("cond")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--sets", default="64_g1,64_g2,64_g3")
    ap.add_argument("--ns", default="8,16,32,64")
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--repeats", default="0,1,2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cands", default=None)
    ap.add_argument("--ck", default=None)
    ap.add_argument("--fracs", default=None)
    ap.add_argument("--sets_given", action="store_true")
    a = ap.parse_args()
    if a.ck:
        L.CK = Path(a.ck)
    reps = tuple(int(x) for x in a.repeats.split(","))
    if a.stage == "displays":
        displays(a.cond, a.sets.split(","), [int(x) for x in a.ns.split(",")], a.q, reps, a.out)
    elif a.stage == "lc":
        lc_table(a.cond, a.rest[0].split(","), [float(x) for x in (a.fracs or "0.25,0.5,0.75,1.0").split(",")], a.sets.split(",") if a.sets_given else None, a.out)
    elif a.stage == "cost":
        cost(a.cond, a.cands.split(",") if a.cands else None, a.out)


if __name__ == "__main__":
    main()
