"""Task M evaluation: item and person metrics, paired participant bootstrap, paired score-swap tests (2,000 swaps), Holm. EXPLORATORY.

Usage: .venv/Scripts/python.exe -I results/phase10/model_search/ms_eval.py [--ck dir] [--sets a,b] [--out name] [--conds D30,W15,C30] [--nboot N] [--nperm N]

Primary metric of every row = item AUC (Delmas: image-level AUC pooled within each of the six group sets, repeat-averaged out-of-fold scores,
mean over sets, mean over the 3 noise draws for W15; Chen: trial-level AUC pooled over participants), EXCEPT the two pooling rows
(E11a, E11b) whose primary metric is person AUC (pooling does not change item scores). Win = paired participant-bootstrap CI of the difference
from F0 excludes 0 with a positive difference AND Holm-adjusted two-sided swap p < .05 over the whole family (all candidate x condition rows).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ms_lib as L  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from edd import unified as U  # noqa: E402
from edd import webcam_noise as W  # noqa: E402
from edd.metrics import eer as eer_fn  # noqa: E402

BOOT_SEED, SWAP_SEED = 20261106, 20261107
CAND_ORDER = ["F0", "M1_cl", "M2_pw", "F3a_scan", "F3b_scan+base", "F4a_pca", "F4b_pca+base", "F5a_c22", "F5b_c22+base", "C6_enet", "C7_rf", "C8_cnn",
              "E9_vote", "E10_stack", "E11a_trim20", "E11b_median"]
LABEL = {"F0": "frozen pipeline (reference)", "M1_cl": "A1 conditional logit (softmax over the 4 items)", "M2_pw": "A2 pairwise ranking",
         "F3a_scan": "B3 scan-path features alone", "F3b_scan+base": "B3 scan-path + base features", "F4a_pca": "B4 time-course PCA alone",
         "F4b_pca+base": "B4 time-course PCA + base", "F5a_c22": "B5 catch22 alone", "F5b_c22+base": "B5 catch22 + base",
         "C6_enet": "C6 elastic net", "C7_rf": "C7 random forest (shallow)", "C8_cnn": "C8 tiny 1-D CNN (one try)",
         "E9_vote": "D9 nested soft vote (frozen + best of A-B)", "E10_stack": "D10 nested stacking",
         "E11a_trim20": "E11a person statistic: 20% trimmed mean", "E11b_median": "E11b person statistic: median"}
POOL_ROWS = ("E11a_trim20", "E11b_median")


def tags_of(cond):
    return [f"{cond}_d{d}" for d in (1, 2, 3)] if cond == "W15" else [cond]


def sets_of(cond, sets):
    return ["chen"] if cond.startswith("C") else (sets or L.SETS)


def load_meta(tag, sk):
    m = pickle.load(open(L.CK / tag / f"meta_{sk}.pkl", "rb"))
    return SimpleNamespace(part=m["part"], y=m["y"], image=m["image"])


def have(tag, cand, sets):
    return all(L.ck_path(tag, cand, sk).exists() for sk in sets)


def read_outs(tag, cand, sk):
    return pickle.load(open(L.ck_path(tag, cand, sk), "rb"))


# ----------------------------------------------------------------------------- per candidate tables
def item_frame(tag, cand, sets, repeats=None):
    fr = []
    for sk in sets:
        meta = load_meta(tag, sk)
        outs = read_outs(tag, cand, sk)
        if repeats is not None:
            outs = {r: v for r, v in outs.items() if r in repeats}
        fr.append(L.oof_frame(meta, outs, sk))
    return pd.concat(fr, ignore_index=True)


def person_frame(tag, cand, sets, how="mean", repeats=None):
    """Person-level table: stat averaged over folds/repeats per (set, unit, kind); plus the per-fold table for threshold metrics."""
    parts, avg = [], []
    base = "F0" if cand in POOL_ROWS else cand
    for sk in sets:
        meta = load_meta(tag, sk)
        outs = read_outs(tag, base, sk)
        if repeats is not None:
            outs = {r: v for r, v in outs.items() if r in repeats}
        pdf = L.persons_cit2(meta, outs, how=how)
        pdf["set"] = sk
        parts.append(pdf)
        a = pdf.groupby(["unit", "kind"], as_index=False).agg(y=("y", "first"), score=("stat", "mean"))
        a["set"], a["participant"], a["image"] = sk, a.unit, (a.kind == "known").astype(int)
        avg.append(a[["set", "participant", "image", "y", "score"]])
    return pd.concat(parts, ignore_index=True), pd.concat(avg, ignore_index=True)


def item_metrics(tag, cand, sets, repeats=None):
    """AUC, EER (repeat-averaged scores, per set then mean), HTER / detection / realised false-alarm rate at the training-set 7% threshold."""
    aucs, eers, hters, dets, fprs = [], [], [], [], []
    for sk in sets:
        meta = load_meta(tag, sk)
        outs = read_outs(tag, cand, sk)
        if repeats is not None:
            outs = {r: v for r, v in outs.items() if r in repeats}
        fr = L.oof_frame(meta, outs, sk).groupby(["participant", "image"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))
        aucs.append(roc_auc_score(fr.y, fr.score))
        eers.append(eer_fn(fr.y.to_numpy(), fr.score.to_numpy()))
        for r, folds in outs.items():
            tp = fn = fp = tn = 0
            ok = all(d.get("inner_oof") is not None for d in folds)
            if not ok:
                continue
            for d in folds:
                tau = U.thresholds(d["inner_oof"][meta.y[d["tr"]] == 1], d["inner_oof"][meta.y[d["tr"]] == 0])["tau_fpr7"]
                yt = meta.y[d["te"]]
                dec = d["score"] > tau
                tp += int((dec & (yt == 1)).sum())
                fn += int((~dec & (yt == 1)).sum())
                fp += int((dec & (yt == 0)).sum())
                tn += int((~dec & (yt == 0)).sum())
            tpr, fpr = tp / max(tp + fn, 1), fp / max(fp + tn, 1)
            dets.append(tpr)
            fprs.append(fpr)
            hters.append((fpr + (1 - tpr)) / 2)
    f = lambda v: float(np.mean(v)) if len(v) else float("nan")  # noqa: E731
    return {"item_auc": f(aucs), "item_eer": f(eers), "item_hter_fpr7": f(hters), "item_detection_fpr7": f(dets), "item_realised_fpr7": f(fprs)}


def person_metrics(pdf):
    out = {}
    per_set = []
    for sk, d in pdf.groupby("set"):
        rows = []
        for r, dd in d.groupby("repeat"):
            y, s = dd.y.to_numpy(), dd.stat.to_numpy()
            m = {"auc": roc_auc_score(y, s), "eer": eer_fn(y, s)}
            if np.isfinite(dd.tau_fpr7).all():
                dec = s > dd.tau_fpr7.to_numpy()
                tpr, fpr = dec[y == 1].mean(), dec[y == 0].mean()
                m.update({"detection_fpr7": tpr, "realised_fpr7": fpr, "hter_fpr7": (fpr + 1 - tpr) / 2})
                dec2 = s > dd.tau_eer.to_numpy()
                m["hter_eer"] = ((dec2[y == 0].mean()) + (1 - dec2[y == 1].mean())) / 2
            rows.append(m)
        per_set.append({k: float(np.mean([x[k] for x in rows if k in x])) for k in rows[0]})
    for k in per_set[0]:
        out["person_" + k] = float(np.mean([p[k] for p in per_set]))
    return out


# ----------------------------------------------------------------------------- statistics (paired, participant clusters)
def to_sets(frames, sets):
    """frames: {name: [DataFrame per draw]} (identical rows) -> {set: SetData with all columns}, owner array, names order."""
    names = list(frames)
    cols, owner, base = [], [], None
    for n in names:
        for fr in frames[n]:
            d = L.LRC.load_oof(fr)
            if base is None:
                base = d
            else:
                assert len(d) == len(base) and (d[["set", "participant", "image", "y"]].to_numpy() == base[["set", "participant", "image", "y"]].to_numpy()).all(), n
            cols.append(d.score.to_numpy())
            owner.append(n)
    S = np.column_stack(cols)
    out = {}
    for sk in sets:
        m = (base.set == sk).to_numpy()
        out[sk] = (W.SetData(base.y.to_numpy()[m], base.participant.to_numpy()[m], S[m]), base.participant.to_numpy()[m])
    return out, np.array(owner), names


def boot(sd_by_set, owner, names, n_boot, seed):
    ncol = next(iter(sd_by_set.values()))[0].S.shape[1]
    rng = np.random.default_rng(seed)
    B = np.full((n_boot, len(sd_by_set), ncol), np.nan)
    for b in range(n_boot):
        for si, (sk, (sd, _)) in enumerate(sd_by_set.items()):
            idx = sd.rows_of(rng.integers(0, sd.n_part, sd.n_part))
            y = sd.y[idx]
            if len(np.unique(y)) < 2:
                continue
            B[b, si] = W.auc_matrix(y, sd.S[idx])
    est_cols = np.nanmean(B, axis=1)
    return {n: est_cols[:, owner == n].mean(1) for n in names}


def point(sd_by_set, owner, names):
    P = np.array([W.auc_matrix(sd.y, sd.S) for sd, _ in sd_by_set.values()])
    return {n: float(P[:, owner == n].mean()) for n in names}


def swap_p(sd_by_set, owner, ref, cand, n_perm, seed):
    """Paired participant-level score swap (as results/phase8/R4_paired_swap): in each permutation every participant (across all sets and draws
    together) independently, with probability 0.5, swaps which method its scores belong to; statistic = mean AUC(cand') - mean AUC(ref').
    Scores are replaced by normalised within-set ranks (a monotone transform of the whole set's score vector: the AUC is unchanged), so
    methods with different score scales can be exchanged."""
    allp = sorted({p for _, (sd, parts) in sd_by_set.items() for p in np.unique(parts)})
    pidx = {p: i for i, p in enumerate(allp)}
    masks = np.random.default_rng(seed).random((n_perm, len(allp))) < 0.5            # (P, n_part)
    ia, ib = np.flatnonzero(owner == ref), np.flatnonzero(owner == cand)
    obs, perm = [], np.zeros(n_perm)
    for sd, parts in sd_by_set.values():
        M = masks[:, [pidx[p] for p in parts]].T                                       # (n, P)
        for ca, cb in zip(ia, ib):
            ra = rankdata(sd.S[:, ca]) / len(sd.y)
            rb = rankdata(sd.S[:, cb]) / len(sd.y)
            obs.append(W.auc_matrix(sd.y, rb[:, None])[0] - W.auc_matrix(sd.y, ra[:, None])[0])
            A = np.where(M, rb[:, None], ra[:, None])
            Bm = np.where(M, ra[:, None], rb[:, None])
            perm += W.auc_matrix(sd.y, Bm) - W.auc_matrix(sd.y, A)
    k = len(obs)
    obs_d = float(np.mean(obs))
    perm /= k
    return obs_d, float((1 + np.sum(np.abs(perm) >= abs(obs_d) - 1e-12)) / (1 + n_perm))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ----------------------------------------------------------------------------- main
def evaluate(conds, ck, sets_arg, n_boot, n_perm, out_name, repeats=None):
    if ck:
        L.CK = Path(ck)
    rows = []
    for cond in conds:
        tags = tags_of(cond)
        sets = sets_of(cond, sets_arg)
        cands = [c for c in CAND_ORDER if c not in POOL_ROWS and all(have(t, c, sets) for t in tags)]
        if "F0" not in cands:
            L.log("no F0 for", cond)
            continue
        L.log(cond, "candidates:", cands)
        # ---- item level
        frames = {c: [item_frame(t, c, sets, repeats) for t in tags] for c in cands}
        sdb, owner, names = to_sets(frames, sets)
        pt = point(sdb, owner, names)
        bt = boot(sdb, owner, names, n_boot, BOOT_SEED)
        # ---- person level (Delmas only: CIT statistic)
        pers = {}
        if not cond.startswith("C"):
            pfr = {}
            for c in cands:
                tabs = [person_frame(t, c, sets, "mean", repeats) for t in tags]
                pfr[c] = [t[1] for t in tabs]
                pers[c] = {"tables": [t[0] for t in tabs]}
            for how, c in (("trim20", "E11a_trim20"), ("median", "E11b_median")):
                tabs = [person_frame(t, c, sets, how, repeats) for t in tags]
                pfr[c] = [t[1] for t in tabs]
                pers[c] = {"tables": [t[0] for t in tabs]}
            psd, powner, pnames = to_sets(pfr, sets)
            ppt = point(psd, powner, pnames)
            pbt = boot(psd, powner, pnames, n_boot, BOOT_SEED + 1)
        for c in cands + ([] if cond.startswith("C") else list(POOL_ROWS)):
            row = {"cond": cond, "cand": c, "label": LABEL[c], "n_draws": len(tags)}
            if c not in POOL_ROWS:
                im = [item_metrics(t, c, sets, repeats) for t in tags]
                row.update({k: float(np.mean([m[k] for m in im])) for k in im[0]})
                row["item_auc"] = pt[c]
                row["item_auc_lo"], row["item_auc_hi"] = (float(np.percentile(bt[c], 2.5)), float(np.percentile(bt[c], 97.5)))
                d = bt[c] - bt["F0"]
                row["item_delta"] = pt[c] - pt["F0"]
                row["item_delta_lo"], row["item_delta_hi"] = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
                if c != "F0":
                    row["item_swap_obs"], row["item_swap_p"] = swap_p(sdb, owner, "F0", c, n_perm, SWAP_SEED)
            if not cond.startswith("C"):
                ms = [person_metrics(t) for t in pers[c]["tables"]]
                row.update({k: float(np.mean([m[k] for m in ms])) for k in ms[0]})
                row["person_auc"] = ppt[c]
                row["person_auc_lo"], row["person_auc_hi"] = (float(np.percentile(pbt[c], 2.5)), float(np.percentile(pbt[c], 97.5)))
                dp = pbt[c] - pbt["F0"]
                row["person_delta"] = ppt[c] - ppt["F0"]
                row["person_delta_lo"], row["person_delta_hi"] = (float(np.percentile(dp, 2.5)), float(np.percentile(dp, 97.5)))
                if c != "F0":
                    row["person_swap_obs"], row["person_swap_p"] = swap_p(psd, powner, "F0", c, n_perm, SWAP_SEED + 1)
            rows.append(row)
            L.log(cond, c, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)})
    df = pd.DataFrame(rows)
    # ---- family, Holm, decision
    fam = df[df.cand != "F0"].copy()
    nan = pd.Series(np.nan, index=fam.index)
    isp = fam.cand.isin(POOL_ROWS)
    pick = lambda a, b: np.where(isp, fam.get(a, nan), fam.get(b, nan)).astype(float)  # noqa: E731
    fam["p_primary"], fam["delta_primary"] = pick("person_swap_p", "item_swap_p"), pick("person_delta", "item_delta")
    fam["lo_primary"], fam["hi_primary"] = pick("person_delta_lo", "item_delta_lo"), pick("person_delta_hi", "item_delta_hi")
    fam["p_holm"] = holm(fam.p_primary.to_numpy())
    fam["ci_excludes_0"] = (fam.lo_primary > 0) | (fam.hi_primary < 0)
    fam["WIN"] = (fam.lo_primary > 0) & (fam.p_holm < 0.05)
    fam["WORSE"] = (fam.hi_primary < 0) & (fam.p_holm < 0.05)
    df = df.merge(fam[["cond", "cand", "p_primary", "delta_primary", "lo_primary", "hi_primary", "p_holm", "ci_excludes_0", "WIN", "WORSE"]], on=["cond", "cand"], how="left")
    od = L.OUT / out_name
    od.mkdir(exist_ok=True)
    df.to_csv(od / "results_table.csv", index=False)
    (od / "results.json").write_text(json.dumps({"n_boot": n_boot, "n_perm": n_perm, "boot_seed": BOOT_SEED, "swap_seed": SWAP_SEED, "family_size": int(len(fam)),
                                                 "rows": json.loads(df.to_json(orient="records"))}, indent=1))
    L.log("family size", len(fam), "WIN:", fam[fam.WIN][["cond", "cand"]].values.tolist(), "WORSE:", fam[fam.WORSE][["cond", "cand"]].values.tolist())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", default=None)
    ap.add_argument("--sets", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--conds", default="D30,W15,C30")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--repeats", default=None)
    a = ap.parse_args()
    evaluate(a.conds.split(","), a.ck, a.sets.split(",") if a.sets else None, a.nboot, a.nperm, a.out,
             [int(x) for x in a.repeats.split(",")] if a.repeats else None)


if __name__ == "__main__":
    main()
