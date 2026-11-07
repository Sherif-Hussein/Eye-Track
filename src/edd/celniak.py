"""Celniak et al. (2026) mock-crime CIT, Tobii Pro Fusion at 120 Hz (Zenodo 10.5281/zenodo.18525952).

Raw Tobii Pro Lab exports are cached as compact CSVs, then each 7.5 s slide (7 questions x 3
repetitions, 4 item AOIs each) is reduced to the paper's AOI metrics: fixation, visit and glance
duration statistics for the crime-related (probe) item, normalised against the median of the three
irrelevant items, plus baseline-corrected pupil diameter.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Probe (crime-related) item per question, from Table 1 of the paper mapped to the export's AOI names.
PROBES = {1: "rękawiczki", 2: "klucz", 3: "asystent", 4: "google", 5: "egzamin", 6: "czapka", 7: "torba"}
SLIDE_RE = re.compile(r"^(\d)_(\d)$")
KINDS = ("fixation", "visit", "glance")
STATS = ("total", "mean", "min", "max", "count", "first")
PAPER_FEATURES = [f"norm_{k}_{s}" for k in KINDS for s in STATS] + ["pupil_bc"]
# The paper's Table 3 reports raw probe-AOI values, and its text is ambiguous about whether the
# classifier saw normalised or raw values, so both readings are evaluated.
RAW_TARGET_FEATURES = [f"tgt_{k}_{s}" for k in KINDS for s in STATS] + ["pupil_bc"]

SAMPLE_COLUMNS = {
    "Recording timestamp": "ts_us",
    "Presented Stimulus name": "stimulus",
    "Event": "event",
    "Event value": "event_value",
    "Gaze point X": "gaze_x",
    "Gaze point Y": "gaze_y",
    "Pupil diameter left": "pupil_l",
    "Pupil diameter right": "pupil_r",
    "Pupil diameter filtered": "pupil_f",
    "Validity left": "valid_l",
    "Validity right": "valid_r",
    "Eye movement type": "move_type",
    "Eye movement type index": "move_index",
    "Gaze event duration": "event_dur_ms",
    "Fixation point X": "fix_x",
    "Fixation point Y": "fix_y",
}
META_COLUMNS = ["Participant name", "gender", "FirstExamination", "group", "Recording name",
                "Average calibration accuracy (degrees)", "Average validation accuracy (degrees)",
                "Average validation precision RMS (degrees)"]


def cache_recording(xlsx_path, cache_dir):
    """Convert one Tobii export to `<name>.csv.gz` (samples) and `<name>.json` (participant metadata)."""
    try:
        df = pd.read_excel(xlsx_path, engine="calamine")  # ~10x faster than openpyxl when installed
    except ImportError:
        df = pd.read_excel(xlsx_path, engine="openpyxl")
    meta = {c: (df[c].dropna().iloc[0].item() if hasattr(df[c].dropna().iloc[0], "item")
                else df[c].dropna().iloc[0]) for c in META_COLUMNS if c in df and df[c].notna().any()}
    aoi = {c: "aoi:" + c[len("AOI hit ["):-1] for c in df.columns if str(c).startswith("AOI hit [")}
    out = df[list(SAMPLE_COLUMNS) + list(aoi)].rename(columns={**SAMPLE_COLUMNS, **aoi})
    out[list(aoi.values())] = out[list(aoi.values())].fillna(0).astype("int8")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = meta["Recording name"]
    out.to_csv(cache_dir / f"{name}.csv.gz", index=False, compression="gzip")
    (cache_dir / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return name


def load_cached(cache_dir, name):
    cache_dir = Path(cache_dir)
    samples = pd.read_csv(cache_dir / f"{name}.csv.gz", low_memory=False)
    meta = json.loads((cache_dir / f"{name}.json").read_text())
    return samples, meta


def _pupil(samples):
    p = samples["pupil_f"].to_numpy(float)
    fallback = samples[["pupil_l", "pupil_r"]].mean(axis=1).to_numpy(float)
    return np.where(np.isnan(p), fallback, p)


def pupil_baseline(samples):
    """Mean pupil diameter over the first fixation-cross screen of the recording (paper's baseline)."""
    cross = samples["stimulus"].eq("cross").to_numpy()
    first = np.flatnonzero(cross)
    if len(first) == 0:
        return float("nan")
    # The first contiguous run of 'cross' samples.
    breaks = np.flatnonzero(np.diff(first) > 1)
    end = first[breaks[0]] if len(breaks) else first[-1]
    return float(np.nanmean(_pupil(samples.iloc[first[0]:end + 1])))


def _events(slide, aoi_cols, dt_us):
    """Fixation and saccade events within one slide, clipped to the slide, sorted by onset."""
    events = []
    for kind in ("Fixation", "Saccade"):
        sub = slide[slide["move_type"] == kind]
        for _, g in sub.groupby("move_index", sort=False):
            ev = {"kind": kind, "start": g["ts_us"].iloc[0], "end": g["ts_us"].iloc[-1] + dt_us,
                  "aoi": None, "rows": g.index}
            if kind == "Fixation":
                hits = g[aoi_cols].sum()
                if hits.max() > len(g) / 2:
                    ev["aoi"] = hits.idxmax()
            events.append(ev)
    events.sort(key=lambda e: e["start"])
    return events


def _visits_and_glances(events):
    """Tobii Pro Lab definitions. Visit: first fixation onset to last fixation offset of consecutive
    fixations inside an AOI. Glance: the visit plus the saccade that entered the AOI."""
    visits, glances = {}, {}
    run = None
    for i, ev in enumerate(events):
        if ev["kind"] != "Fixation":
            continue
        if run is not None and ev["aoi"] == run["aoi"]:
            run["end"] = ev["end"]
            continue
        if run is not None:
            visits.setdefault(run["aoi"], []).append(run["end"] - run["start"])
            glances.setdefault(run["aoi"], []).append(run["end"] - run["glance_start"])
        run = None
        if ev["aoi"] is not None:
            prev = events[i - 1] if i > 0 else None
            glance_start = prev["start"] if prev is not None and prev["kind"] == "Saccade" else ev["start"]
            run = {"aoi": ev["aoi"], "start": ev["start"], "end": ev["end"], "glance_start": glance_start}
    if run is not None:
        visits.setdefault(run["aoi"], []).append(run["end"] - run["start"])
        glances.setdefault(run["aoi"], []).append(run["end"] - run["glance_start"])
    return visits, glances


def _stats(durations_us):
    if not durations_us:
        return dict.fromkeys(STATS, 0.0)
    d = np.asarray(durations_us, float) / 1e3  # milliseconds
    return {"total": d.sum(), "mean": d.mean(), "min": d.min(), "max": d.max(),
            "count": float(len(d)), "first": d[0]}


def aoi_table(samples, meta):
    """One row per (slide, item AOI): the same fixation/visit/glance metrics and pupil for every item,
    with `is_probe` marking the crime-related one. Supports within-slide (item-ranking) models."""
    dt_us = float(samples["ts_us"].diff().median())
    baseline = pupil_baseline(samples)
    pupil = _pupil(samples)
    rows = []
    for stim, slide in samples.groupby("stimulus", sort=False):
        m = SLIDE_RE.match(str(stim))
        if not m:
            continue
        question, rep = int(m.group(1)), int(m.group(2))
        aoi_cols = [c for c in samples.columns if c.startswith(f"aoi:{stim} - ")]
        events = _events(slide, aoi_cols, dt_us)
        visits, glances = _visits_and_glances(events)
        for c in aoi_cols:
            fx = [e for e in events if e["kind"] == "Fixation" and e["aoi"] == c]
            row = {"participant": meta["Participant name"], "recording": meta["Recording name"],
                   "label": int(str(meta["group"]).strip().lower() == "guilty"),
                   "first_exam": meta.get("FirstExamination"), "question": question, "repetition": rep,
                   "item": c.split(" - ", 1)[1], "is_probe": int(c.endswith(f" - {PROBES[question]}"))}
            for kind, durs in (("fixation", [e["end"] - e["start"] for e in fx]),
                               ("visit", visits.get(c, [])), ("glance", glances.get(c, []))):
                row.update({f"{kind}_{s}": v for s, v in _stats(durs).items()})
            idx = np.concatenate([e["rows"].to_numpy() for e in fx]) if fx else np.array([], int)
            row["pupil_bc"] = float(np.nanmean(pupil[idx]) - baseline) if len(idx) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def slide_features(samples, meta):
    """One row per probe slide: raw target / non-target-median values and the paper's normalised features."""
    dt_us = float(samples["ts_us"].diff().median())
    baseline = pupil_baseline(samples)
    pupil = _pupil(samples)
    rows = []
    for stim, slide in samples.groupby("stimulus", sort=False):
        m = SLIDE_RE.match(str(stim))
        if not m:
            continue
        question, rep = int(m.group(1)), int(m.group(2))
        aoi_cols = [c for c in samples.columns if c.startswith(f"aoi:{stim} - ")]
        target = f"aoi:{stim} - {PROBES[question]}"
        others = [c for c in aoi_cols if c != target]
        events = _events(slide, aoi_cols, dt_us)
        fix = {}
        for ev in events:
            if ev["kind"] == "Fixation" and ev["aoi"] is not None:
                fix.setdefault(ev["aoi"], []).append(ev)
        visits, glances = _visits_and_glances(events)
        per_aoi = {}
        for c in aoi_cols:
            fx = fix.get(c, [])
            vals = {}
            for kind, durs in (("fixation", [e["end"] - e["start"] for e in fx]),
                               ("visit", visits.get(c, [])), ("glance", glances.get(c, []))):
                vals.update({f"{kind}_{s}": v for s, v in _stats(durs).items()})
            rows_idx = np.concatenate([e["rows"].to_numpy() for e in fx]) if fx else np.array([], int)
            vals["pupil_bc"] = float(np.nanmean(pupil[rows_idx]) - baseline) if len(rows_idx) else np.nan
            per_aoi[c] = vals
        row = {"participant": meta["Participant name"], "recording": meta["Recording name"],
               "label": int(str(meta["group"]).strip().lower() == "guilty"),
               "first_exam": meta.get("FirstExamination"), "question": question, "repetition": rep,
               "valid_frac": float(((slide["valid_l"] == "Valid") & (slide["valid_r"] == "Valid")).mean()),
               "slide_pupil_bc": float(np.nanmean(pupil[slide.index.to_numpy()]) - baseline),
               "n_samples": int(len(slide))}
        for key in per_aoi[target]:
            t = per_aoi[target][key]
            nt = float(np.nanmedian([per_aoi[c][key] for c in others])) if others else np.nan
            row[f"tgt_{key}"], row[f"ntm_{key}"] = t, nt
            if key != "pupil_bc":
                row[f"norm_{key}"] = t / (t + nt) if (t + nt) > 0 else 0.5
        # Paper feature: target-AOI pupil; fall back to the slide mean when the probe was never fixated.
        row["pupil_bc"] = row["tgt_pupil_bc"] if not np.isnan(row["tgt_pupil_bc"]) else row["slide_pupil_bc"]
        rows.append(row)
    return pd.DataFrame(rows)
