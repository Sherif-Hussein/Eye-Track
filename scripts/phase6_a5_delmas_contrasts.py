"""Phase 6 Track A, A5: Delmas known-vs-unknown univariate contrasts per group (PREREG section 4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ttest_1samp  # noqa: E402

from phase6_common import *  # noqa: E402,F401,F403
from edd.delmas import load_trials  # noqa: E402

SEED = 20261205
PRIMARY = ["first_fix", "lat_first", "dwell_1", "dwell_2"]
SECOND = ["dwell_b1", "dwell_b2", "dwell_b3", "dwell_b4", "dwell_b5", "n_fix", "n_visit", "pupil_face"]
FEATS = PRIMARY + SECOND
GROUPS = ["G1", "G2", "G3", "G4", "G5"]
DATA = ROOT / "data"

face = pd.read_csv(DATA / "derived" / "delmas_raw_face_trials.csv.gz")
book = load_trials(DATA / "delmas" / "BookFIP_64_trials.xlsx")[["participant", "image", "face"]]
gmap = face.drop_duplicates("participant").set_index("participant").group.to_dict()


def contrasts(rate, window):
    """window: 'benchmark' (64 trials for G1-G3, last 32 for G4-G5), 'last32' (all groups), 'G3last32' (G3 only)."""
    f = face[face.rate == rate].merge(book, on=["participant", "image"], how="inner")
    f = f[f.face.notna()]
    out = {}
    for g in GROUPS + ["G3_last32"]:
        gg = 3 if g == "G3_last32" else int(g[1])
        d = f[f.participant.map(gmap) == gg]
        if window == "last32" or g in ("G4", "G5", "G3_last32"):
            d = d[d.trial >= 32]
        nk = d.groupby(["participant", "trial"]).face.sum()
        keep = nk[nk == 1].index
        d = d.set_index(["participant", "trial"]).loc[keep].reset_index()
        kn = d[d.face == 1].groupby("participant")[FEATS].mean()
        un = d[d.face == 0].groupby("participant")[FEATS].mean()
        diff = (kn - un.loc[kn.index]).dropna(how="all")
        diff["n_known_pres"] = d[d.face == 1].groupby("participant").size().loc[diff.index]
        out[g] = diff
    return out


def dz(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1)) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


def hedges(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (len(a) + len(b)) - 9)))


def analyse(rate, window, seed, full=True):
    C = contrasts(rate, window)
    rng = np.random.default_rng(seed)
    n = {g: len(C[g]) for g in C}
    draws = {g: [rng.integers(0, n[g], n[g]) for _ in range(B)] for g in C}
    res = {"n": n, "per_group": {}, "interaction_vs_G1": {}}
    for g in C:
        res["per_group"][g] = {}
        for ft in FEATS:
            x = C[g][ft].to_numpy()
            ok = np.isfinite(x)
            bs = [dz(x[idx]) for idx in draws[g]]
            bm = [np.nanmean(x[idx]) for idx in draws[g]]
            t = ttest_1samp(x[ok], 0) if ok.sum() > 2 else None
            res["per_group"][g][ft] = {"dz": dz(x), "dz_ci95": pct_ci(bs), "mean_diff": float(np.nanmean(x)), "mean_diff_ci95": pct_ci(bm),
                                       "p_t_two_sided": float(t.pvalue) if t is not None else np.nan, "n": int(ok.sum())}
    # Holm over 20 primary contrasts (4 features x 5 groups, G3 once)
    keys = [(g, ft) for g in GROUPS for ft in PRIMARY]
    ps = holm([res["per_group"][g][ft]["p_t_two_sided"] for g, ft in keys])
    for (g, ft), p in zip(keys, ps):
        res["per_group"][g][ft]["p_holm_20"] = float(p)
    # interactions
    pooled = lambda gs, ft: np.concatenate([C[g][ft].to_numpy() for g in gs])  # noqa: E731
    for ft in FEATS:
        res["interaction_vs_G1"][ft] = {}
        for k in ["G2", "G3", "G4", "G5", "G2-G5 pooled"]:
            gs = ["G2", "G3", "G4", "G5"] if k == "G2-G5 pooled" else [k]
            x1 = C["G1"][ft].to_numpy()
            xk = pooled(gs, ft)
            bg, bd = [], []
            for b in range(B):
                i1 = draws["G1"][b]
                parts = []
                for g in gs:
                    parts.append(C[g][ft].to_numpy()[draws[g][b]])
                ik = np.concatenate(parts)
                bg.append(hedges(ik, x1[i1]))                       # Gk minus G1, standardised
                bd.append(dz(ik) - dz(x1[i1]))
            res["interaction_vs_G1"][ft][k] = {"hedges_g_Gk_minus_G1": hedges(xk, x1), "g_ci95": pct_ci(bg),
                                               "dz_Gk_minus_dz_G1": dz(xk) - dz(x1), "dz_diff_ci95": pct_ci(bd)}
    # decision rules (PREREG 4)
    pg, it = res["per_group"], res["interaction_vs_G1"]
    g2 = pg["G2"]
    early_g2 = g2["first_fix"]["dz_ci95"][0] > 0 and g2["lat_first"]["dz_ci95"][1] < 0
    late_g2 = g2["dwell_2"]["dz_ci95"][1] < 0
    inter_pred = (it["first_fix"]["G2"]["g_ci95"][0] > 0) or (it["lat_first"]["G2"]["g_ci95"][1] < 0) or (it["dwell_2"]["G2"]["g_ci95"][1] < 0)
    res["decision_rules"] = {
        "G2_early_orienting_present": bool(early_g2), "G2_late_avoidance_present": bool(late_g2), "G2_vs_G1_interaction_in_predicted_direction": bool(inter_pred),
        "concealment_specific_account_supported": bool(early_g2 and late_g2 and inter_pred),
        "G1_late_avoidance_present": bool(pg["G1"]["dwell_2"]["dz_ci95"][1] < 0),
        "G2_below_G1_on_late_dwell_CI_excludes0": bool(it["dwell_2"]["G2"]["g_ci95"][1] < 0),
        "memory_task_account_supported": bool(pg["G1"]["dwell_2"]["dz_ci95"][1] < 0 and not (it["dwell_2"]["G2"]["g_ci95"][1] < 0)),
        "G1_first_fix_dz": pg["G1"]["first_fix"]["dz"], "G1_lat_first_dz": pg["G1"]["lat_first"]["dz"], "G1_dwell2_dz": pg["G1"]["dwell_2"]["dz"],
        "G2_first_fix_dz": g2["first_fix"]["dz"], "G2_lat_first_dz": g2["lat_first"]["dz"], "G2_dwell2_dz": g2["dwell_2"]["dz"]}
    return res, C


def table(res, tag):
    rows = []
    for g, d in res["per_group"].items():
        for ft, e in d.items():
            rows.append({"config": tag, "group": g, "feature": ft, "n": e["n"], "dz": e["dz"], "dz_lo": e["dz_ci95"][0], "dz_hi": e["dz_ci95"][1],
                         "mean_diff": e["mean_diff"], "md_lo": e["mean_diff_ci95"][0], "md_hi": e["mean_diff_ci95"][1],
                         "p_t": e["p_t_two_sided"], "p_holm_20": e.get("p_holm_20", np.nan)})
    return pd.DataFrame(rows)


def make_card(out):
    r = out["primary_1000Hz_benchmark_window"]
    pg = r["per_group"]
    it = r["interaction_vs_G1"]

    def f(g, ft):
        e = pg[g][ft]
        return f"{e['dz']:+.2f} {ci_s(e['dz_ci95'], 2)}"
    dr = r["decision_rules"]
    s30 = out["sens_30Hz_benchmark_window"]["per_group"]
    s32 = out["sens_1000Hz_last32_all_groups"]["per_group"]
    para = ("Delmas known-minus-unknown contrasts within trials containing a known face (paired d_z with 95% CI, 1000 Hz, 15 per group; G4 and G5 over their 32-trial window; negative latency and negative dwell mean the known face is looked at earlier / less). "
            + " | ".join(f"{g}: first-fixation {f(g, 'first_fix')}, latency {f(g, 'lat_first')}, dwell 0.2-1 s {f(g, 'dwell_1')}, dwell 1-5 s {f(g, 'dwell_2')}" for g in ["G1", "G2", "G3", "G4", "G5"]) + ". "
            f"Group x face interaction (Hedges g, G2 minus G1): first fixation {it['first_fix']['G2']['hedges_g_Gk_minus_G1']:+.2f} {ci_s(it['first_fix']['G2']['g_ci95'], 2)}, latency {it['lat_first']['G2']['hedges_g_Gk_minus_G1']:+.2f} {ci_s(it['lat_first']['G2']['g_ci95'], 2)}, "
            f"dwell 1-5 s {it['dwell_2']['G2']['hedges_g_Gk_minus_G1']:+.2f} {ci_s(it['dwell_2']['G2']['g_ci95'], 2)}; pooled G2-G5 minus G1 on dwell 1-5 s {it['dwell_2']['G2-G5 pooled']['hedges_g_Gk_minus_G1']:+.2f} {ci_s(it['dwell_2']['G2-G5 pooled']['g_ci95'], 2)} (late avoidance is weaker, not stronger, in the countermeasure groups). "
            f"Decision rules (PREREG 4): G2 early orienting present {dr['G2_early_orienting_present']}, G2 late avoidance present {dr['G2_late_avoidance_present']}, G2-vs-G1 interaction in the predicted direction {dr['G2_vs_G1_interaction_in_predicted_direction']} "
            f"-> concealment-specific 'quick first look then avoidance' supported: {dr['concealment_specific_account_supported']}; G1 (memory task only, no concealment instruction) late avoidance present {dr['G1_late_avoidance_present']}, "
            f"G2 below G1 on late dwell with CI excluding 0 {dr['G2_below_G1_on_late_dwell_CI_excludes0']} -> memory-task account supported: {dr['memory_task_account_supported']}. "
            f"Sensitivities: at 30 Hz G1/G2 dwell 1-5 s d_z {s30['G1']['dwell_2']['dz']:+.2f}/{s30['G2']['dwell_2']['dz']:+.2f} and first-fixation {s30['G1']['first_fix']['dz']:+.2f}/{s30['G2']['first_fix']['dz']:+.2f}; "
            f"with the common last-32-trial window G1/G2 first-fixation {s32['G1']['first_fix']['dz']:+.2f}/{s32['G2']['first_fix']['dz']:+.2f}, dwell 1-5 s {s32['G1']['dwell_2']['dz']:+.2f}/{s32['G2']['dwell_2']['dz']:+.2f}.")
    save_card("A5", para)
    log(para)


if __name__ == "__main__" and "--card-only" in sys.argv:
    import json as _j
    make_card(_j.loads((OUT / "results.json").read_text())["A5"])
    sys.exit(0)


if __name__ == "__main__":
    configs = {"primary_1000Hz_benchmark_window": (1000, "benchmark"), "sens_1000Hz_last32_all_groups": (1000, "last32"),
               "sens_30Hz_benchmark_window": (30, "benchmark")}
    out = {"definition": {"contrast": "known minus unknown, within trials containing exactly one known face, per participant mean over presentations; d_z = mean/SD of the participant contrasts",
                          "B": B, "seed": SEED}}
    tabs = []
    for i, (name, (rate, win)) in enumerate(configs.items()):
        log(name)
        res, C = analyse(rate, win, SEED + i)
        out[name] = res
        tabs.append(table(res, name))
    pd.concat(tabs).to_csv(OUT / "figdata_A5_delmas_contrasts.csv", index=False)
    save_section("A5", out)
    make_card(out)
