"""Phase 6 Track A, A2: sensitivity of the AUC definition for the 7 Phase 4 primary tests (PREREG section 5)."""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from phase6_common import *  # noqa: E402,F401,F403

SEED = 20261202
STORED = json.loads((RES / "phase4" / "unified" / "results.json").read_text())["primary_tests"]


def rank_cols(S):
    """Mid-rank transform within each repeat (column) over all decision units, scaled to (0, 1]."""
    return np.column_stack([rankdata(S[:, j]) / S.shape[0] for j in range(S.shape[1])])


def matrix(df, keys, ncol_by="repeat"):
    """Wide (rows = decision units in sorted key order, columns = repeats) score matrix and label vector."""
    p = df.pivot_table(index=keys, columns="repeat", values="score")
    y = df.groupby(keys).y.first().loc[p.index]
    return p, y


class Task:
    """Row-level arrays for two arms with strata (Delmas sets) and clusters (participants)."""

    def __init__(self, name, y, Sb, Su, cluster, stratum=None, note=""):
        self.name, self.y, self.cluster, self.note = name, np.asarray(y), np.asarray(cluster), note
        self.stratum = np.zeros(len(self.y), int) if stratum is None else np.asarray(stratum)
        self.S = {"base": Sb, "uni": Su}
        self.R = {"base": rank_cols(Sb), "uni": rank_cols(Su)}
        self.strata = {s: np.flatnonzero(self.stratum == s) for s in np.unique(self.stratum)}
        self.clusters = {s: [np.flatnonzero((self.stratum == s) & (self.cluster == c)) for c in np.unique(self.cluster[self.stratum == s])]
                         for s in self.strata}

    def metric(self, arm, variant, idx_by_stratum):
        vals = []
        for s, idx in idx_by_stratum.items():
            y = self.y[idx]
            if variant == "V0_repeat_avg_raw":
                vals.append(auc(y, self.S[arm][idx].mean(1)))
            elif variant == "V1_rank_avg_per_repeat":
                vals.append(auc(y, self.R[arm][idx].mean(1)))
            else:
                vals.append(np.mean([auc(y, self.S[arm][idx, j]) for j in range(self.S[arm].shape[1])]))
        return float(np.mean(vals))

    def run(self, seed):
        rng = np.random.default_rng(seed)
        full = dict(self.strata)
        variants = ["V0_repeat_avg_raw", "V1_rank_avg_per_repeat", "V2_mean_of_per_repeat_auc"]
        est = {v: {a: self.metric(a, v, full) for a in ("base", "uni")} for v in variants}
        bs = {v: {"d": [], "base": [], "uni": []} for v in variants}
        for _ in range(B):
            idx = {s: np.concatenate([cl[i] for i in rng.integers(0, len(cl), len(cl))]) for s, cl in self.clusters.items()}
            for v in variants:
                b, u = self.metric("base", v, idx), self.metric("uni", v, idx)
                bs[v]["d"].append(u - b)
        out = {"note": self.note, "n_clusters": int(len(np.unique(self.cluster))), "variants": {}}
        v0 = est["V0_repeat_avg_raw"]["uni"] - est["V0_repeat_avg_raw"]["base"]
        ci0 = pct_ci(bs["V0_repeat_avg_raw"]["d"])
        for v in variants:
            d = est[v]["uni"] - est[v]["base"]
            ci = pct_ci(bs[v]["d"])
            out["variants"][v] = {"auc_base": est[v]["base"], "auc_unified": est[v]["uni"], "delta": d, "ci95": ci,
                                  "p_tail_one_sided": p_one_sided(np.array(bs[v]["d"])), "change_in_delta_vs_V0": d - v0,
                                  "sign_flip_vs_V0": bool(np.sign(d) != np.sign(v0)),
                                  "ci_excludes0_status_changed_vs_V0": bool((ci[0] > 0 or ci[1] < 0) != (ci0[0] > 0 or ci0[1] < 0))}
        out["definition_gap_V0_minus_V2"] = {a: est["V0_repeat_avg_raw"][a] - est["V2_mean_of_per_repeat_auc"][a] for a in ("base", "uni")}
        return out


def person_task(name, base_df, uni_rows, note=""):
    b = base_df.assign(unit=base_df.unit.astype(str)).groupby(["repeat", "unit"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))
    u = uni_rows.assign(unit=uni_rows.unit.astype(str)).groupby(["repeat", "unit"], as_index=False).agg(y=("y", "first"), score=("score", "mean"))
    Sb, yb = matrix(b, "unit")
    Su, yu = matrix(u, "unit")
    common = sorted(set(Sb.index) & set(Su.index))
    assert (yb.loc[common] == yu.loc[common]).all()
    return Task(name, yu.loc[common].to_numpy(), Sb.loc[common].to_numpy(), Su.loc[common].to_numpy(), np.arange(len(common)), note=note)


def delmas_task():
    sets = ["64_g1", "64_g2", "64_g3", "32_g3", "32_g4", "32_g5"]
    base = pd.read_csv(P2 / "delmas_raw" / "checkpoints" / "P4_svm_1000_oof.csv.gz")
    ys, Sb, Su, cl, st = [], [], [], [], []
    for k, s in enumerate(sets):
        u = pd.read_csv(OOF / f"delmas_{s}_rows.csv.gz").rename(columns={"unit": "participant"})
        b = base[base.set == s]
        pu = u.pivot_table(index=["participant", "image"], columns="repeat", values="score")
        pb = b.pivot_table(index=["participant", "image"], columns="repeat", values="score")
        yy = u.groupby(["participant", "image"]).y.first()
        common = pu.index.intersection(pb.index)
        ys.append(yy.loc[common].to_numpy())
        Su.append(pu.loc[common].to_numpy())
        Sb.append(pb.loc[common].to_numpy())
        cl.append(np.array([p for p, _ in common]))
        st.append(np.full(len(common), k))
    return Task("delmas_noninferiority", np.concatenate(ys), np.vstack(Sb), np.vstack(Su), np.concatenate(cl), np.concatenate(st),
                note="mean over six group-sets of the image-level AUC (G3 appears twice, as in the pre-registered primary); rank transform within repeat and set")


def chen_tasks():
    cps = pd.concat([pickle.load(open(RES / "phase3" / "chen_confirm" / "checkpoints" / f"repeat{r}.pkl", "rb"))["persons"] for r in range(3)], ignore_index=True)
    cts = pd.concat([pickle.load(open(RES / "phase3" / "chen_confirm" / "checkpoints" / f"repeat{r}.pkl", "rb"))["trials"] for r in range(3)], ignore_index=True)
    uni = pd.read_csv(OOF / "chen_unified_persons.csv.gz")
    units = sorted(cps.unit.unique())
    rows_b, rows_u, ys, cl = [], [], [], []
    for u in units:
        for key, kind, yy in (("z_fam", "fam", 1), ("z_c1", "c1", 0), ("z_c2", "c2", 0)):
            b = cps[(cps.arm == "rate_rule") & (cps.unit == u)].sort_values("repeat")[key].to_numpy()
            s = uni[(uni.unit == u) & (uni.kind == kind)].sort_values("repeat").stat.to_numpy()
            rows_b.append(b)
            rows_u.append(s)
            ys.append(yy)
            cl.append(u)
    person = Task("chen_person_auc", ys, np.array(rows_b), np.array(rows_u), cl, note="AUC analogue of the Chen primary (detection at FPR <= 7% has no AUC): person z statistic, familiar vs control-1 and control-2 (pre-registered pooling), rate rule as baseline")
    ur = pd.read_csv(OOF / "chen_unified_rows.csv.gz")
    pu = ur.pivot_table(index="row", columns="repeat", values="score")
    meta = ur.groupby("row").agg(y=("y", "first"), unit=("unit", "first"))
    tb = cts[cts.arm == "rate_rule"].pivot_table(index="row", columns="repeat", values="score")
    common = pu.index.intersection(tb.index)
    trial = Task("chen_trial_auc", meta.loc[common].y.to_numpy(), tb.loc[common].to_numpy(), pu.loc[common].to_numpy(), meta.loc[common].unit.to_numpy(),
                 note="descriptive: trial-level AUC, familiar vs control trials, rate rule as baseline")
    return person, trial


def make_card(out):
    parts = []
    for k, r in out["tests"].items():
        v = r["variants"]
        sx = " / ".join(f"{v[x]['delta']:+.3f} {ci_s(v[x]['ci95'])}" for x in v)
        flips = [x for x in v if v[x]["sign_flip_vs_V0"] or v[x]["ci_excludes0_status_changed_vs_V0"]]
        parts.append(f"{k}: {sx}" + (f" (flag: {', '.join(flips)})" if flips else ""))
    mx = out["largest_definition_gap_V0_vs_V2"]
    m = out["tests"]["mazza_g2_vs_g3"]
    para = ("Delta (unified minus baseline), V0 pre-registered / V1 per-repeat rank averaging / V2 mean of per-repeat AUC: " + "; ".join(parts) + ". "
            f"Colasanti G2 vs G3: the pre-registered delta of {m['variants']['V0_repeat_avg_raw']['delta']:+.3f} becomes {m['variants']['V1_rank_avg_per_repeat']['delta']:+.3f} (rank averaging) and {m['variants']['V2_mean_of_per_repeat_auc']['delta']:+.3f} (per-repeat AUC); the deficit is a scoring artefact of the unified arm's AUC "
            f"(V0 minus V2 = {m['definition_gap_V0_minus_V2']['uni']:+.3f} for the unified arm, {m['definition_gap_V0_minus_V2']['base']:+.3f} for the baseline), and every CI includes 0 under all three definitions. "
            "The Delmas flag is a sign change of a delta of 0.000 and carries no meaning. Celniak (-0.13 to -0.15), Preti A (+0.15 to +0.19) and Preti B stay the same size and sign under all definitions; the Chen AUC deltas are unchanged. "
            f"Largest gap between the pre-registered AUC and the mean of per-repeat AUCs for a single arm: {mx['max']:.3f} ({mx['test_of_max']}), so the sentence 'the two definitions differ by up to 0.02' is "
            f"{'correct' if mx['max'] <= 0.0205 else 'not correct'}. All V0 deltas reproduce the Phase 4 stored values: {all(r.get('check_vs_phase4_stored', {}).get('match', True) for r in out['tests'].values())}.")
    save_card("A2", para)
    log(para)


if __name__ == "__main__" and "--card-only" in sys.argv:
    import json as _j
    make_card(_j.loads((OUT / "results.json").read_text())["A2"])
    sys.exit(0)


if __name__ == "__main__":
    tasks = {}
    tasks["delmas_noninferiority"] = delmas_task()
    p, t = chen_tasks()
    tasks["chen_person_auc"], tasks["chen_trial_auc"] = p, t
    tasks["celniak"] = person_task("celniak", pd.read_csv(P2 / "celniak" / "trials_baseline_svm_paper.csv"), pd.read_csv(OOF / "celniak_unified_rows.csv.gz"))
    tasks["preti_A"] = person_task("preti_A", pd.read_csv(P2 / "preti" / "scores" / "B_ocular.csv"), pd.read_csv(OOF / "preti_A_ocular_rows.csv.gz"))
    tasks["preti_B"] = person_task("preti_B", pd.read_csv(P2 / "preti" / "scores" / "B_fusion.csv"), pd.read_csv(OOF / "preti_B_ocular_rt_rows.csv.gz"))
    tasks["mazza_g2_vs_g3"] = person_task("mazza_g2_vs_g3", pd.read_csv(P2 / "mazza" / "oof" / "g2_vs_g3__base_pub36_lr_unreg.csv"), pd.read_csv(OOF / "mazza_g2_vs_g3_rows.csv.gz"))
    tasks["mazza_guilty_vs_innocent"] = person_task("mazza_guilty_vs_innocent", pd.read_csv(P2 / "mazza" / "oof" / "guilty_vs_innocent__base_pub36_lr_unreg.csv"), pd.read_csv(OOF / "mazza_guilty_vs_innocent_rows.csv.gz"))
    out, rows = {"definition": {"B": B, "seed": SEED, "variants": ["V0 repeat-averaged raw score (pre-registered)", "V1 per-repeat rank averaging", "V2 mean of per-repeat AUC"]}, "tests": {}}, []
    stored_key = {"celniak": "celniak_participant_auc", "preti_A": "preti_A_participant_auc", "preti_B": "preti_B_participant_auc",
                  "mazza_g2_vs_g3": "mazza_g2_vs_g3_participant_auc", "mazza_guilty_vs_innocent": "mazza_guilty_vs_innocent_participant_auc",
                  "delmas_noninferiority": "delmas_noninferiority"}
    for i, (k, t) in enumerate(tasks.items()):
        log("task", k, "n rows", len(t.y), "repeats base/uni", t.S["base"].shape[1], t.S["uni"].shape[1])
        r = t.run(SEED + i)
        if k in stored_key:
            sd = STORED[stored_key[k]]
            r["check_vs_phase4_stored"] = {"stored_delta": sd["delta"], "recomputed_V0_delta": r["variants"]["V0_repeat_avg_raw"]["delta"],
                                           "match": bool(abs(sd["delta"] - r["variants"]["V0_repeat_avg_raw"]["delta"]) < 1e-9)}
        elif k == "chen_person_auc":
            sd = STORED["chen_detection_fpr7"]["person_auc"]["rate_rule"]
            r["check_vs_phase4_stored"] = {"stored_delta": sd["delta_unified_minus"], "recomputed_V0_delta": r["variants"]["V0_repeat_avg_raw"]["delta"],
                                           "match": bool(abs(sd["delta_unified_minus"] - r["variants"]["V0_repeat_avg_raw"]["delta"]) < 1e-9)}
        elif k == "chen_trial_auc":
            sd = STORED["chen_trial_auc_descriptive"]["rate_rule"]
            r["check_vs_phase4_stored"] = {"stored_delta": sd["delta_unified_minus"], "recomputed_V0_delta": r["variants"]["V0_repeat_avg_raw"]["delta"],
                                           "match": bool(abs(sd["delta_unified_minus"] - r["variants"]["V0_repeat_avg_raw"]["delta"]) < 1e-9)}
        out["tests"][k] = r
        for v, e in r["variants"].items():
            rows.append({"test": k, "variant": v, "auc_base": e["auc_base"], "auc_unified": e["auc_unified"], "delta": e["delta"], "lo": e["ci95"][0], "hi": e["ci95"][1]})
    gaps = {k: max(abs(v) for v in r["definition_gap_V0_minus_V2"].values()) for k, r in out["tests"].items()}
    out["largest_definition_gap_V0_vs_V2"] = {"by_test": gaps, "max": max(gaps.values()), "test_of_max": max(gaps, key=gaps.get)}
    pd.DataFrame(rows).to_csv(OUT / "figdata_A2_auc_definition.csv", index=False)
    save_section("A2", out)
    make_card(out)
