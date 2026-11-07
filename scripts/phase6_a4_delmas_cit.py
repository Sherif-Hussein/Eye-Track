"""Phase 6 Track A, A4: Delmas participant-level CIT AUC per group, published model vs ours, 1000 and 30 Hz (PREREG section 3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phase6_common import *  # noqa: E402,F401,F403

SEED = 20261204
CK = P2 / "delmas_raw" / "checkpoints"
GROUPS = {"G1": "64_g1", "G2": "64_g2", "G3": "64_g3", "G4": "32_g4", "G5": "32_g5"}
EXTRA = {"G3_last32": "32_g3"}
MODELS = {  # label -> (checkpoint tag, rate, role)
    "published_table_SVM_1000": ("book_svm", 1000, "published table (authors' label-dependent features) + SVM"),
    "published_model_P4_SVM_1000": ("P4_svm_1000", 1000, "published model on rebuilt label-free features"),
    "published_model_P4_SVM_30": ("P4_svm_30", 30, "published model on rebuilt label-free features"),
    "ours_ALLz_LR_1000": ("ALLz_lr_1000", 1000, "our features (within-person z, 14 incl. microsaccades) + LR"),
    "ours_ALLz_LR_30": ("ALLz_lr_30", 30, "our features (13) + LR"),
}
N_DRAWS = 50


def cit(s, y, rng):
    sf, su = s[y == 1], s[y == 0]
    zk = (sf.mean() - su.mean()) / (su.std() + 1e-9)          # numpy ddof = 0, as edd.unified._cit_z
    pick = rng.choice(len(su), len(sf), replace=False)
    rest = np.delete(su, pick)
    zi = (su[pick].mean() - rest.mean()) / (rest.std() + 1e-9)
    return float(zk), float(zi)


def person_stats(oof, setname):
    """Per participant: repeat-mean known z, repeat-mean simulated-innocent z (rng = default_rng(repeat + id)), per-repeat AUCs,
    and the expected-innocent statistics over N_DRAWS independent pseudo-probe draws."""
    o = oof[oof.set == setname]
    parts = sorted(o.participant.unique())
    reps = sorted(o["repeat"].unique())
    K = np.zeros((len(parts), len(reps)))
    I = np.zeros_like(K)
    IX = np.zeros((len(parts), len(reps), N_DRAWS))
    for j, r in enumerate(reps):
        for i, p in enumerate(parts):
            d = o[(o["repeat"] == r) & (o.participant == p)]
            s, y = d.score.to_numpy(), d.y.to_numpy()
            K[i, j], I[i, j] = cit(s, y, np.random.default_rng(r + int(p)))
            for k in range(N_DRAWS):
                IX[i, j, k] = cit(s, y, np.random.default_rng([7, r, int(p), k]))[1]
    return {"parts": parts, "known": K, "innoc": I, "innoc_draws": IX}


def grp_auc(known, innoc):
    return auc(np.r_[np.ones(len(known)), np.zeros(len(innoc))], np.r_[known, innoc])


def main():
    cache = {}
    for m, (tag, rate, _) in MODELS.items():
        oof = pd.read_csv(CK / f"{tag}_oof.csv.gz")
        for g, s in {**GROUPS, **EXTRA}.items():
            cache[(m, g)] = person_stats(oof, s)
        log("done", m)
    # Phase 4 unified at 30 Hz: stored statistics
    uni = {}
    for g, s in {**GROUPS, **EXTRA}.items():
        d = pd.read_csv(OOF / f"delmas_{s}_persons.csv.gz")
        parts = sorted(d.unit.unique())
        kn = d[d.kind == "known"].groupby("unit").stat.mean().loc[parts].to_numpy()
        iv = d[d.kind == "sim_innocent"].groupby("unit").stat.mean().loc[parts].to_numpy()
        det = (d[d.kind == "known"].assign(x=lambda t: t.stat > t.tau_fpr7).groupby("unit").x.mean().loc[parts]).to_numpy()
        fp = (d[d.kind == "sim_innocent"].assign(x=lambda t: t.stat > t.tau_fpr7).groupby("unit").x.mean().loc[parts]).to_numpy()
        per_rep = np.mean([grp_auc(x[x.kind == "known"].stat.to_numpy(), x[x.kind == "sim_innocent"].stat.to_numpy()) for _, x in d.groupby("repeat")])
        uni[g] = {"parts": parts, "known": kn, "innoc": iv, "det": det, "fp": fp, "per_rep_auc": float(per_rep)}
    # sanity: participants agree across models
    for g in list(GROUPS) + list(EXTRA):
        ref = cache[("ours_ALLz_LR_30", g)]["parts"]
        assert ref == uni[g]["parts"], g
        for m in MODELS:
            assert cache[(m, g)]["parts"] == ref, (m, g)
    names = list(MODELS) + ["ours_unified_phase4_30"]

    def stat(m, g):
        if m == "ours_unified_phase4_30":
            return uni[g]["known"], uni[g]["innoc"]
        c = cache[(m, g)]
        return c["known"].mean(1), c["innoc"].mean(1)

    glist = list(GROUPS) + list(EXTRA)
    rng = np.random.default_rng(SEED)
    boot = {}                                         # (model, group) -> list of AUC
    pooled_defs = {"no_countermeasure_G1_G2": ["G1", "G2"], "countermeasure_G3_G4_G5": ["G3", "G4", "G5"], "strong_countermeasure_G4_G5": ["G4", "G5"]}
    boot_pool = {}
    draws = [{g: rng.integers(0, 15, 15) for g in list(GROUPS) + list(EXTRA)} for _ in range(B)]
    for m in names:
        for g in glist:
            k, i = stat(m, g)
            boot[(m, g)] = np.array([grp_auc(k[d[g]], i[d[g]]) for d in draws])
        for pn, gs in pooled_defs.items():
            vals = []
            for d in draws:
                kk = np.concatenate([stat(m, g)[0][d[g]] for g in gs])
                ii = np.concatenate([stat(m, g)[1][d[g]] for g in gs])
                vals.append(grp_auc(kk, ii))
            boot_pool[(m, pn)] = np.array(vals)
        log("boot", m)
    rows, out = [], {"groups": {}, "pooled": {}, "delta": {}, "definition": {"G3": "64_g3 (once); 32_g3 reported as G3_last32 sensitivity", "B": B, "seed": SEED}}
    for g in glist:
        out["groups"][g] = {}
        for m in names:
            k, i = stat(m, g)
            a = grp_auc(k, i)
            ci = pct_ci(boot[(m, g)])
            e = {"auc": a, "ci95": ci, "n": 15}
            if m in cache or True:
                if m != "ours_unified_phase4_30":
                    c = cache[(m, g)]
                    e["auc_per_repeat_mean"] = float(np.mean([grp_auc(c["known"][:, j], c["innoc"][:, j]) for j in range(c["known"].shape[1])]))
                    e["auc_expected_innocent_50draws"] = float(np.mean([grp_auc(c["known"].mean(1), c["innoc_draws"][:, :, kk].mean(1)) for kk in range(N_DRAWS)]))
                else:
                    e["auc_per_repeat_mean"] = uni[g]["per_rep_auc"]
                    e["detection_at_7pct_threshold"] = float(uni[g]["det"].mean())
                    e["realised_fpr_at_7pct_threshold"] = float(uni[g]["fp"].mean())
            out["groups"][g][m] = e
            rows.append({"group": g, "set": {**GROUPS, **EXTRA}[g], "model": m, "rate_hz": (MODELS[m][1] if m in MODELS else 30), "auc": a, "lo": ci[0], "hi": ci[1], "n": 15,
                         "auc_expected_innocent": e.get("auc_expected_innocent_50draws", np.nan)})
    for pn, gs in pooled_defs.items():
        out["pooled"][pn] = {}
        for m in names:
            kk = np.concatenate([stat(m, g)[0] for g in gs])
            ii = np.concatenate([stat(m, g)[1] for g in gs])
            a = grp_auc(kk, ii)
            ci = pct_ci(boot_pool[(m, pn)])
            out["pooled"][pn][m] = {"auc": a, "ci95": ci, "n": 15 * len(gs)}
            rows.append({"group": pn, "set": "+".join(GROUPS[g] for g in gs), "model": m, "rate_hz": (MODELS[m][1] if m in MODELS else 30), "auc": a, "lo": ci[0], "hi": ci[1], "n": 15 * len(gs)})
    pd.DataFrame(rows).to_csv(OUT / "figdata_A4_delmas_cit_auc.csv", index=False)
    pairs = [("ours_ALLz_LR_1000", "published_model_P4_SVM_1000", "ours - published model, 1000 Hz"),
             ("ours_ALLz_LR_30", "published_model_P4_SVM_30", "ours - published model, 30 Hz"),
             ("ours_unified_phase4_30", "published_model_P4_SVM_30", "Phase 4 unified (30 Hz) - published model 30 Hz"),
             ("published_model_P4_SVM_30", "published_model_P4_SVM_1000", "published model 30 Hz - 1000 Hz"),
             ("ours_ALLz_LR_30", "ours_ALLz_LR_1000", "ours 30 Hz - 1000 Hz"),
             ("published_model_P4_SVM_1000", "published_table_SVM_1000", "rebuilt published model - published table (1000 Hz)")]
    drows = []
    for a_, b_, lab in pairs:
        out["delta"][lab] = {}
        for g in glist + list(pooled_defs):
            if g in glist:
                bd = boot[(a_, g)] - boot[(b_, g)]
                k1, i1 = stat(a_, g)
                k2, i2 = stat(b_, g)
            else:
                bd = boot_pool[(a_, g)] - boot_pool[(b_, g)]
                gs = pooled_defs[g]
                k1 = np.concatenate([stat(a_, x)[0] for x in gs]); i1 = np.concatenate([stat(a_, x)[1] for x in gs])
                k2 = np.concatenate([stat(b_, x)[0] for x in gs]); i2 = np.concatenate([stat(b_, x)[1] for x in gs])
            dl = grp_auc(k1, i1) - grp_auc(k2, i2)
            out["delta"][lab][g] = {"delta": dl, "ci95": pct_ci(bd)}
            drows.append({"comparison": lab, "group": g, "delta": dl, "lo": out["delta"][lab][g]["ci95"][0], "hi": out["delta"][lab][g]["ci95"][1]})
    pd.DataFrame(drows).to_csv(OUT / "figdata_A4_delmas_cit_delta.csv", index=False)
    save_section("A4", out)
    make_card(out)


def make_card(out):
    g = out["groups"]
    pm = out["pooled"]

    def f(m, k):
        e = g[k][m]
        return f"{e['auc']:.2f} {ci_u(e['ci95'], 2)}"
    txt = []
    for k in ["G1", "G2", "G3", "G4", "G5"]:
        txt.append(f"{k}: published model 1000/30 Hz {f('published_model_P4_SVM_1000', k)} / {f('published_model_P4_SVM_30', k)}, ours 1000/30 Hz {f('ours_ALLz_LR_1000', k)} / {f('ours_ALLz_LR_30', k)}")
    pc = pm["countermeasure_G3_G4_G5"]
    pn = pm["no_countermeasure_G1_G2"]
    ps = pm["strong_countermeasure_G4_G5"]
    d30 = out["delta"]["ours - published model, 30 Hz"]
    d1000 = out["delta"]["ours - published model, 1000 Hz"]
    tb = out["delta"]["rebuilt published model - published table (1000 Hz)"]
    para = ("Delmas participant-level CIT AUC (15 known vs 15 simulated innocents per group; G3 = 64-trial window, counted once; G4 and G5 have only the 32-trial window). " + "; ".join(txt) + ". "
            f"Pooled, ours 30 Hz: no countermeasure (G1+G2, n = 30) {pn['ours_ALLz_LR_30']['auc']:.2f} {ci_u(pn['ours_ALLz_LR_30']['ci95'], 2)}, G3-G5 (n = 45) {pc['ours_ALLz_LR_30']['auc']:.2f} {ci_u(pc['ours_ALLz_LR_30']['ci95'], 2)}, G4+G5 (n = 30) {ps['ours_ALLz_LR_30']['auc']:.2f} {ci_u(ps['ours_ALLz_LR_30']['ci95'], 2)}. "
            f"Ours minus the published model: 1000 Hz G4+G5 {d1000['strong_countermeasure_G4_G5']['delta']:+.2f} {ci_s(d1000['strong_countermeasure_G4_G5']['ci95'], 2)}; 30 Hz G4 {d30['G4']['delta']:+.2f} {ci_s(d30['G4']['ci95'], 2)}, G5 {d30['G5']['delta']:+.2f} {ci_s(d30['G5']['ci95'], 2)}, G4+G5 {d30['strong_countermeasure_G4_G5']['delta']:+.2f} {ci_s(d30['strong_countermeasure_G4_G5']['ci95'], 2)}. "
            f"Phase 4 unified (30 Hz) at the stored 7% threshold: detection / realised FPR G4 {g['G4']['ours_unified_phase4_30']['detection_at_7pct_threshold']:.2f} / {g['G4']['ours_unified_phase4_30']['realised_fpr_at_7pct_threshold']:.2f}, G5 {g['G5']['ours_unified_phase4_30']['detection_at_7pct_threshold']:.2f} / {g['G5']['ours_unified_phase4_30']['realised_fpr_at_7pct_threshold']:.2f}. "
            f"The authors' published feature table + SVM stays at AUC 1.00 in G4 and G5 (label-dependent aggregation), so the countermeasure collapse appears only with label-free features: rebuilt published model minus published table, G4+G5 {tb['strong_countermeasure_G4_G5']['delta']:+.2f} {ci_s(tb['strong_countermeasure_G4_G5']['ci95'], 2)}. "
            "G3 (simple countermeasure) stays at 0.93 in both windows. Per-group CIs span about +-0.2 at n = 15, and a perfect AUC has a degenerate bootstrap interval; the model change from 1000 to 30 Hz is within these intervals in every group except possibly the published model in G4+G5.")
    save_card("A4", para)
    log(para)


if __name__ == "__main__" and "--card-only" in sys.argv:
    import json as _j
    make_card(_j.loads((OUT / "results.json").read_text())["A4"])
    sys.exit(0)


if __name__ == "__main__":
    main()
