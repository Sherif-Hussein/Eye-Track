"""Phase 7, Track D: statistics for D1 (no-pupil webcam simulation) and D2 (face-area size sensitivity).
Reads finished out-of-fold scores only. Pre-registration: results/phase7/trackD/PREREG.md.

Usage (project root):  .venv/Scripts/python.exe -I scripts/phase7_trackD_stats.py
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location("trackD_run", ROOT / "scripts" / "phase7_trackD_run.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
C1, W, OUT, CK = R.C1, R.C1.W, R.OUT, R.CK
N_BOOT, MARGIN, BOOT_SEED, REF = C1.N_BOOT, C1.MARGIN, C1.BOOT_SEED, C1.REF
C1_CK = C1.SCRATCH / "c1_ck"
CONDS = [("clean", 0.0), ("1", 1.0), ("1.5", 1.5), ("2", 2.0), ("3", 3.0), ("4", 4.0), ("anch", "A")]
CLABEL = {"clean": "0 (clean)", "1": "1", "1.5": "1.5", "2": "2", "3": "3", "4": "4", "anch": "anchored"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cond_tags(v, p, s, ck):
    """List of (name, path) of the draws of one configuration."""
    cname = "clean" if ck == "clean" else ("anch" if ck == "anch" else "all")
    L = dict(CONDS)[ck]
    draws = [0] if ck == "clean" else R.DRAWS
    out = []
    for d in draws:
        if p == "np" or s != 1.0:
            t = R.tag_of(v, p, s, cname, L, d)
            out.append((t, CK / f"{t}_oof.csv.gz"))
        elif v == "ivt" and ck != "anch":
            t = f"ALLz_fixed_lr_30_L{C1.lv(L)}_d{d}"
            out.append((t, C1.WEB / "checkpoints" / f"{t}_oof.csv.gz"))
        else:
            t = C1.tag_of(v, cname, L, d)
            out.append((t, C1_CK / f"{t}_oof.csv.gz"))
    return out


def config_keys():
    ks = []
    for v in ("ivt", "sb"):
        for p in ("wp", "np"):
            for ck, _ in CONDS:
                ks.append((1.0, v, p, ck))
    for s in (0.75, 0.5):
        for v in ("ivt", "sb"):
            for p in ("wp", "np"):
                for ck in ("clean", "anch"):
                    ks.append((s, v, p, ck))
    return ks


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[k]))
        adj[k] = run
    return adj


def main():
    keys = config_keys()
    paths = {REF: C1.oof_path(REF)}
    ctags = {}
    for k in keys:
        ctags[k] = cond_tags(k[1], k[2], k[0], k[3])
        for n, pth in ctags[k]:
            paths[n] = pth
    missing = [n for n, pth in paths.items() if not Path(pth).exists()]
    assert not missing, f"missing OOF ({len(missing)}): {missing[:6]}"
    names = [REF] + sorted(n for n in paths if n != REF)
    base = C1.load_oof(paths[REF])
    rowkeys = base[["set", "participant", "image"]].to_numpy()
    S = np.empty((len(base), len(names)))
    S[:, 0] = base.score.to_numpy()
    for j, n in enumerate(names[1:], 1):
        o = C1.load_oof(paths[n])
        assert len(o) == len(base) and (o[["set", "participant", "image"]].to_numpy() == rowkeys).all() and (o.y.to_numpy() == base.y.to_numpy()).all(), n
        S[:, j] = o.score.to_numpy()
    col = {n: j for j, n in enumerate(names)}
    sets = {}
    for s in sorted(base.set.unique()):
        m = (base.set == s).to_numpy()
        sets[s] = W.SetData(base.y.to_numpy()[m], base.participant.to_numpy()[m], S[m])
    set_names = list(sets)
    point = np.array([W.auc_matrix(sets[s].y, sets[s].S) for s in set_names])
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.full((N_BOOT, len(set_names), len(names)), np.nan)
    for b in range(N_BOOT):
        for si, s in enumerate(set_names):
            sd = sets[s]
            pick = rng.integers(0, sd.n_part, sd.n_part)
            idx = sd.rows_of(pick)
            y = sd.y[idx]
            if len(np.unique(y)) < 2:
                continue
            boot[b, si] = W.auc_matrix(y, sd.S[idx])
    boot_auc = np.nanmean(boot, axis=1)
    log("bootstrap done; columns", len(names))
    pt_ref = float(point[:, 0].mean())

    rows = {}
    for k in keys:
        cs = [col[n] for n, _ in ctags[k]]
        cb = boot_auc[:, cs].mean(1)
        pts = [float(point[:, c].mean()) for c in cs]
        auc = float(np.mean(pts))
        db = cb - boot_auc[:, 0]
        rows[k] = {"scale": k[0], "route": k[1], "pupil": k[2], "cond": k[3], "n_draws": len(cs), "tags": [n for n, _ in ctags[k]],
                   "auc": auc, "auc_ci95": [float(x) for x in np.percentile(cb, [2.5, 97.5])],
                   "auc_per_draw": pts, "auc_draw_sd": float(np.std(pts)) if len(pts) > 1 else 0.0,
                   "delta_vs_ref": auc - pt_ref, "delta_vs_ref_ci95": [float(x) for x in np.percentile(db, [2.5, 97.5])],
                   "p_noninferior_raw": float((1 + np.sum(db <= -MARGIN)) / (1 + N_BOOT)), "_b": cb}

    def diff(a, b):
        d = rows[a]["_b"] - rows[b]["_b"]
        return rows[a]["auc"] - rows[b]["auc"], [float(x) for x in np.percentile(d, [2.5, 97.5])]

    # D1 family: 14 no-pupil configurations at full size; Holm over 14 and within route (7)
    fam = [(1.0, v, "np", ck) for v in ("ivt", "sb") for ck, _ in CONDS]
    for k, a in zip(fam, holm([rows[k]["p_noninferior_raw"] for k in fam])):
        rows[k]["p_holm_d1_family14"] = float(a)
        rows[k]["noninferior_d1_family14"] = bool(a < 0.05)
    for v in ("ivt", "sb"):
        ks = [(1.0, v, "np", ck) for ck, _ in CONDS]
        for k, a in zip(ks, holm([rows[k]["p_noninferior_raw"] for k in ks])):
            rows[k]["p_holm_d1_route7"] = float(a)
    # contrasts
    for k, r in rows.items():
        s, v, p, ck = k
        if s == 1.0 and p == "np":
            r["delta_vs_own_clean"], r["delta_vs_own_clean_ci95"] = diff(k, (1.0, v, "np", "clean"))
            o = (1.0, "ivt" if v == "sb" else "sb", "np", ck)
            if v == "sb":
                r["delta_sb_minus_ivt"], r["delta_sb_minus_ivt_ci95"] = diff(k, o)
        if p == "np":
            r["pupil_contribution_with_minus_no"], r["pupil_contribution_ci95"] = diff((s, v, "wp", ck), k)
        if s < 1.0:
            r["delta_vs_fullsize_same_config"], r["delta_vs_fullsize_ci95"] = diff(k, (1.0, v, p, ck))

    out = [{a: b for a, b in r.items() if a != "_b"} for r in rows.values()]
    # CSVs
    d1 = []
    for ck, _ in CONDS:
        for v in ("ivt", "sb"):
            r = rows[(1.0, v, "np", ck)]
            d1.append({"route": v, "level_deg": CLABEL[ck], "n_features": len(R.FEATS[(v, "np")]), "auc": r["auc"], "auc_lo": r["auc_ci95"][0], "auc_hi": r["auc_ci95"][1],
                       "delta_vs_ref": r["delta_vs_ref"], "delta_lo": r["delta_vs_ref_ci95"][0], "delta_hi": r["delta_vs_ref_ci95"][1],
                       "p_noninferior_raw": r["p_noninferior_raw"], "p_holm_family14": r["p_holm_d1_family14"], "p_holm_route7": r["p_holm_d1_route7"],
                       "noninferior_holm14": r["noninferior_d1_family14"], "delta_vs_own_clean": r["delta_vs_own_clean"],
                       "delta_vs_own_clean_lo": r["delta_vs_own_clean_ci95"][0], "delta_vs_own_clean_hi": r["delta_vs_own_clean_ci95"][1],
                       "sb_minus_ivt": r.get("delta_sb_minus_ivt"), "sb_minus_ivt_lo": r.get("delta_sb_minus_ivt_ci95", [None, None])[0],
                       "sb_minus_ivt_hi": r.get("delta_sb_minus_ivt_ci95", [None, None])[1], "auc_draw_sd": r["auc_draw_sd"]})
    pd.DataFrame(d1).to_csv(OUT / "d1_table.csv", index=False)
    pc = []
    for k, r in rows.items():
        if "pupil_contribution_with_minus_no" in r:
            w = rows[(k[0], k[1], "wp", k[3])]
            pc.append({"scale": k[0], "route": k[1], "cond": CLABEL[k[3]], "auc_with_pupil": w["auc"], "auc_no_pupil": r["auc"],
                       "with_minus_no": r["pupil_contribution_with_minus_no"], "lo": r["pupil_contribution_ci95"][0], "hi": r["pupil_contribution_ci95"][1]})
    pd.DataFrame(pc).to_csv(OUT / "d1_pupil_contribution.csv", index=False)
    d2 = []
    geo = json.loads((OUT / "checks.json").read_text())["aoi_deg"]
    for k, r in rows.items():
        if k[0] == 1.0 and k[3] not in ("clean", "anch"):
            continue
        g = geo[str(k[0])]
        e = {"scale": k[0], "aoi_deg": f'{g["width_deg"]:.2f} x {g["height_deg"]:.2f}', "route": k[1], "pupil": k[2], "cond": k[3], "auc": r["auc"],
             "auc_lo": r["auc_ci95"][0], "auc_hi": r["auc_ci95"][1], "delta_vs_fullsize": r.get("delta_vs_fullsize_same_config", 0.0),
             "delta_vs_fullsize_lo": r.get("delta_vs_fullsize_ci95", [0, 0])[0], "delta_vs_fullsize_hi": r.get("delta_vs_fullsize_ci95", [0, 0])[1],
             "delta_vs_ref": r["delta_vs_ref"], "delta_vs_ref_lo": r["delta_vs_ref_ci95"][0], "delta_vs_ref_hi": r["delta_vs_ref_ci95"][1],
             "p_noninferior_raw_descriptive": r["p_noninferior_raw"], "auc_draw_sd": r["auc_draw_sd"]}
        d2.append(e)
    pd.DataFrame(d2).sort_values(["cond", "route", "pupil", "scale"], ascending=[True, True, True, False]).to_csv(OUT / "d2_table.csv", index=False)
    res = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "reference": REF, "reference_auc": pt_ref, "margin": MARGIN, "n_boot": N_BOOT,
           "bootstrap_seed": BOOT_SEED, "anchored_noise": C1.ANCH, "aoi_deg": geo, "checks": json.loads((OUT / "checks.json").read_text()),
           "trackC_script_sha256": M_sha(ROOT / "scripts" / "phase6_trackC_c1.py"), "rows": out}
    C1.jdump(OUT / "results.json", res)
    log("written")


def M_sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
