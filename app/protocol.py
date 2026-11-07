"""Item sets, trial construction and the session runner for the three modes.

  A  parallel 2x2 display (Delmas-like): 4 options for 5 s, then a yes/no target question (F = yes, J = no).
  B  sequential Yes/No (classic reaction-time CIT): one item at a time.
  C  polygraph-style irrelevant / relevant / comparison questions, Yes/No.

Before the real trials, modes A and B run 2 unscored practice trials with feedback (saved in practice.csv). Each trial starts
only when gaze is on the fixation cross (gaze gate); a mode-A trial whose display had too little valid on-screen gaze is flagged
`lookaway`, kept out of scoring and repeated once (see run_trial / run_session).

Face item sets with "task": "memory" (items/faces_mockcrime_1man.json, items/faces_mockcrime.json) run in mode A only: a mock-crime scene
with one or two men, then 4 faces (of one of one or two test pools, each with its own probe) for 5 s and the question "was this test
face one of the four?" (short-term-memory CIT of Delmas et al. 2024). Within a pool every face is shown equally often (see build_memory_trials); after the end validation a
line-up asks which men were in the story.

Item-set JSON schema: see app/README.md and app/items/demo_shapes.json.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

import lang as L
from camera import CLOCK
from config import APP_DIR, Config
from features import onscreen_valid_share
from landmarks import px_per_deg
from recorder import GROUP_LABEL, SessionRecorder

# Delmas geometry (scripts/phase6 + src/edd/delmas_raw.py): four 300 x 400 px faces centred at +-320, +-270 px from the screen
# centre, 34.9869 px/deg. App layouts have their own gaps (sized from the gaze error, see make_layout); `to_delmas` maps each
# app card onto the matching Delmas card, so AOI membership maps one-to-one onto the paper's.
DELMAS_PPD = 34.9868554165592
DELMAS_AOI_PX = (300.0, 400.0)
DELMAS_CENTRES_PX = np.array([[-320, -270], [320, -270], [-320, 270], [320, 270]], float)      # TL, TR, BL, BR
DELMAS_EXTENT_DEG = (2 * 320 + 300) / DELMAS_PPD                                                 # 26.9 deg both ways
VERSION = "0.1.0"


# ----------------------------------------------------------------------------- item sets
FACES_ALIAS = "faces"                                    # `--items faces`: the faces task; the exam (1 or 2 men) is asked on the first screen
EXAM_ITEMS = {1: "items/faces_mockcrime_1man.json", 2: "items/faces_mockcrime.json"}      # exam choice -> item set (relative to app/)


def exam_itemset_path(exam: int) -> Path:
    return APP_DIR / EXAM_ITEMS[int(exam)]


def load_itemset(path: Path | str) -> dict:
    p = Path(path)
    raw = p.read_bytes()
    d = json.loads(raw)
    d["_sha256"] = hashlib.sha256(raw).hexdigest()
    d["_path"] = str(p)
    for i in d.get("items", []):                     # picture paths are relative to the item-set file
        for holder, key in ((i.get("draw", {}), "path"), (i, "scene_path")):
            if holder.get(key) and not Path(holder[key]).is_absolute():
                holder[key] = str((p.parent / holder[key]).resolve())
    validate_itemset(d)
    return d


def validate_itemset(d: dict) -> None:
    ids = [i["id"] for i in d.get("items", [])]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate item ids")
    if d.get("task") == "memory":
        _validate_memory(d, ids)
        return
    for q in d.get("questions", []):
        refs = [q["probe"]] if q["probe"] != "@pick" else []
        refs += q["irrelevants"] if isinstance(q["irrelevants"], list) else []
        refs += [q["target"]] if q.get("target") else []
        for r in refs:
            if r not in ids:
                raise ValueError(f"question {q['id']} refers to unknown item {r}")
        if isinstance(q["irrelevants"], list) and len(q["irrelevants"]) < 3:
            raise ValueError(f"question {q['id']} needs at least 3 irrelevant items")
    for q in d.get("polygraph_questions", []):
        if q["type"] not in ("relevant", "irrelevant", "comparison"):
            raise ValueError(f"polygraph question {q['id']}: type must be relevant, irrelevant or comparison")
    if not d.get("questions") and not d.get("polygraph_questions"):
        raise ValueError("item set has neither 'questions' nor 'polygraph_questions'")
    from display import SHAPES
    for i in d.get("items", []):
        dr = i.get("draw", {})
        if dr.get("kind") == "shape" and dr.get("shape") not in SHAPES:
            raise ValueError(f"item {i['id']}: unknown shape '{dr.get('shape')}' (one of {', '.join(SHAPES)})")


def _validate_memory(d: dict, ids: list) -> None:
    pools = d.get("pools", {})
    test = pools.get("test")
    if not isinstance(test, list) or len(test) not in (1, 2):
        raise ValueError('memory item set: "pools" must be {"test": [n1] or [n1, n2], "scene": 1 or 2, "practice": 2} (one or two test pools, '
                         'each with one probe and one innocent-story face; the single-probe structure of the first version, "test": 9, '
                         'is no longer supported)')
    n_test, n_scene, n_prac = [int(x) for x in test], int(pools.get("scene", len(test))), int(pools.get("practice", 2))
    if sum(n_test) + n_scene + n_prac != len(ids):
        raise ValueError(f"memory item set: pools {n_test}+{n_scene}+{n_prac} must add up to the {len(ids)} items")
    if n_scene != len(n_test) or min(n_test) < 5:
        raise ValueError(f"memory item set: needs exactly {len(n_test)} scene face(s) (one per test pool) and at least 5 faces in each test pool")
    n, k = int(d.get("n_trials", 0)), len(n_test)
    if n < k or n % k or any((n // k) % m for m in n_test):
        raise ValueError(f"memory item set: n_trials ({n}) must be even, and half of it a multiple of each test pool size ({n_test})" if k == 2 else
                         f"memory item set: n_trials ({n}) must be a multiple of the test pool size ({n_test[0]})")
    if len(d.get("questions", [])) != 1 or not d["questions"][0].get("text"):
        raise ValueError("memory item set: needs one question with a text")
    for g in ("knowledgeable", "innocent"):
        sc = d.get("scene", {}).get(g, {})
        if len(sc.get("lines", [])) != k or not sc.get("remember"):
            raise ValueError(f"memory item set: scene.{g} needs {k} story line(s) (one per man) and a 'remember' text")
    if not d.get("lineup", {}).get("question"):
        raise ValueError("memory item set: needs lineup.question")
    for i in d["items"]:
        for path in (i.get("draw", {}).get("path"), i.get("scene_path")):
            if not path or not Path(path).exists():
                raise ValueError(f"item {i['id']}: picture file not found ({path})")


# ----------------------------------------------------------------------------- layout
GAP_RULE_NOTE = ("gap = max(min_gap_deg, gap_error_factor x accuracy). The validation SD is not used: it is sample-to-sample jitter, "
                 "which averages out over the roughly 150 samples of a 5 s display, whereas the accuracy offset does not; and the "
                 "memory task makes gaze land on the faces.")


def required_gap_deg(cfg: Config, accuracy_deg: float | None = None) -> float:
    """Minimum gap between any two options: max(min_gap_deg, gap_error_factor x accuracy) of the chosen calibration (docs/app_plan.md
    7c). Only the ACCURACY (the person's constant offset) enters. The sample-to-sample SD does not: it is jitter that averages out
    over the ~150 samples of a 5 s display, the offset does not, and in the memory task the gaze lands on the faces. (The earlier
    rule 2 x (accuracy + SD) demanded 9.5 deg in a live session and shrank the faces to 1.9 x 2.5 deg.)"""
    acc = accuracy_deg if accuracy_deg is not None and np.isfinite(accuracy_deg) else 0.0
    return float(max(cfg.min_gap_deg, cfg.gap_error_factor * acc))


def make_layout(screen_px: tuple, ppd: float, name: str, aoi_deg: dict, cfg: Config | None = None,
                accuracy_deg: float | None = None) -> dict:
    """2x2 AOI rectangles in screen pixels with EXPLICIT gaps (docs/app_plan.md 7c).

    The gap g between any two options (horizontal and vertical) is at least max(min_gap_deg, gap_error_factor x accuracy) of the
    person's validation (see required_gap_deg: the SD is not in the rule). The card keeps the requested aspect ratio
    (`aoi_deg[name]`); its height is
        h = min(requested h, (0.92 screen_h - g) / 2, ((0.92 screen_w - g) / 2) / aspect)
    so the gap is kept and the cards shrink instead. Cards are centred at +-(w+g)/2 and +-(h+g)/2 from the screen centre
    (order TL, TR, BL, BR). If h < cfg.min_card_deg the layout is flagged (`layout_too_small`); the session still runs, with
    cards of min_card_deg and whatever gap then fits (`gap_ok` False). The export to the paper's Delmas geometry is a
    piecewise affine map per card (`to_delmas`), so the Delmas screen's own gaps are used there."""
    cfg = cfg or Config()
    req_w, req_h = aoi_deg[name]
    aspect = req_w / req_h
    g = required_gap_deg(cfg, accuracy_deg)
    sw, sh = np.array(screen_px, float) / ppd                                    # screen size in degrees
    h = min(req_h, (0.92 * sh - g) / 2, ((0.92 * sw - g) / 2) / aspect)
    too_small = bool(h < cfg.min_card_deg)
    gap = g
    if too_small:
        h = float(cfg.min_card_deg)
        gap = float(max(0.0, min(g, 0.92 * sh - 2 * h, 0.92 * sw - 2 * h * aspect)))   # what still fits on the screen
    w = h * aspect
    cx, cy = np.array(screen_px, float) / 2
    ox, oy = (w + gap) / 2 * ppd, (h + gap) / 2 * ppd
    ctr = np.array([[cx - ox, cy - oy], [cx + ox, cy - oy], [cx - ox, cy + oy], [cx + ox, cy + oy]])
    wp, hp = w * ppd, h * ppd
    rects = np.column_stack([ctr[:, 0] - wp / 2, ctr[:, 1] - hp / 2, ctr[:, 0] + wp / 2, ctr[:, 1] + hp / 2])
    return {"name": name, "requested_deg": [req_w, req_h], "realised_deg": [w, h],
            # realised card width / Delmas card width (8.57 deg): size relative to the paper's cards, gaps excluded
            "scale_vs_delmas": float(w / (DELMAS_AOI_PX[0] / DELMAS_PPD)),
            "scaled_down": bool(h < req_h - 1e-9), "rects": rects.tolist(), "px_per_deg": ppd,
            "gap_deg": [float(gap), float(gap)],                                  # horizontal, vertical (realised)
            "required_gap_deg": g, "gap_ok": bool(gap >= g - 1e-9),
            "gap_basis": {"accuracy_deg": accuracy_deg, "min_gap_deg": cfg.min_gap_deg, "gap_error_factor": cfg.gap_error_factor,
                          "note": GAP_RULE_NOTE},
            "edge_margin_deg": [float((sw - 2 * w - gap) / 2), float((sh - 2 * h - gap) / 2)],   # left/right, top/bottom
            "min_card_deg": cfg.min_card_deg, "layout_too_small": too_small}


DELMAS_RECTS_PX = np.column_stack([960.0 + DELMAS_CENTRES_PX[:, 0] - DELMAS_AOI_PX[0] / 2, 540.0 + DELMAS_CENTRES_PX[:, 1] - DELMAS_AOI_PX[1] / 2,
                                   960.0 + DELMAS_CENTRES_PX[:, 0] + DELMAS_AOI_PX[0] / 2, 540.0 + DELMAS_CENTRES_PX[:, 1] + DELMAS_AOI_PX[1] / 2])


def _piecewise(v, src: list, dst: list) -> np.ndarray:
    """Monotone piecewise-linear map through the breakpoints src -> dst, extended beyond the ends with the end slopes."""
    v = np.asarray(v, float)
    out = np.interp(v, src, dst)
    out = np.where(v < src[0], dst[0] + (v - src[0]) * (dst[1] - dst[0]) / (src[1] - src[0]), out)
    out = np.where(v > src[-1], dst[-1] + (v - src[-1]) * (dst[-1] - dst[-2]) / (src[-1] - src[-2]), out)
    return out


def to_delmas(x, y, rects, delmas_rects=None) -> tuple[np.ndarray, np.ndarray]:
    """Map gaze (screen px) onto the Delmas screen: affine per card (app card k -> Delmas card k), piecewise linear in the
    gaps and outside. A sample inside app card k lands inside Delmas card k; a sample in a gap (or off the cards) lands
    outside all Delmas cards (the gap maps onto the Delmas gap, the outside keeps the nearest card's scale), so AOI
    membership is preserved. NaN stays NaN. `rects` is the 2x2 grid in the order TL, TR, BL, BR."""
    r = np.asarray(rects, float)
    d = np.asarray(DELMAS_RECTS_PX if delmas_rects is None else delmas_rects, float)
    xs, xd = [r[0, 0], r[0, 2], r[1, 0], r[1, 2]], [d[0, 0], d[0, 2], d[1, 0], d[1, 2]]
    ys, yd = [r[0, 1], r[0, 3], r[2, 1], r[2, 3]], [d[0, 1], d[0, 3], d[2, 1], d[2, 3]]
    return _piecewise(x, xs, xd), _piecewise(y, ys, yd)


def mode_b_geometry(screen_px: tuple, ppd: float, layout: dict, text_h_px: float) -> dict:
    """Mode B: question text above the item. The gap between the text's bottom and the item's top is at least the layout's
    required gap (px via ppd), so a look at the text cannot be mistaken for a look at the item. The item keeps the layout's
    card size, shrunk (same aspect, gap kept) only if text + gap + item do not fit in 92% of the screen height. The block is
    centred vertically."""
    W, H = screen_px
    g_px = float(layout["required_gap_deg"]) * ppd
    w, h = layout["realised_deg"][0] * ppd, layout["realised_deg"][1] * ppd
    avail = 0.92 * H - text_h_px - g_px
    if h > avail:
        k = max(avail, 0.5 * ppd) / h
        w, h = w * k, h * k
    top = (H - (text_h_px + g_px + h)) / 2
    text_bottom = top + text_h_px
    item_top = text_bottom + g_px
    return {"text_mid_bottom_px": [W / 2, text_bottom], "item_rect_px": [W / 2 - w / 2, item_top, w, h],
            "gap_px": item_top - text_bottom, "gap_deg": (item_top - text_bottom) / ppd, "item_deg": [w / ppd, h / ppd]}


# ----------------------------------------------------------------------------- trial construction
def _balanced_positions(n: int, rng) -> list[int]:
    """Probe quadrant for n trials: shuffled blocks of the four quadrants (balanced to within one)."""
    out: list[int] = []
    while len(out) < n:
        blk = list(range(4))
        rng.shuffle(blk)
        out += blk
    return out[:n]


def build_trials_A(items: dict, qs: list[dict], rng, picked: str | None, default_repeats: int) -> list[dict]:
    trials = []
    for q in qs:
        probe = picked if q["probe"] == "@pick" else q["probe"]
        irr = [i for i in (items["_pool"] if q["irrelevants"] == "@rest" else q["irrelevants"]) if i != probe]
        n = int(q.get("repeats", default_repeats))
        pos = _balanced_positions(n, rng)
        n_tgt = int(round(float(q.get("target_rate", 0.15)) * n)) if q.get("target") else 0
        tgt_idx = set(rng.choice(n, n_tgt, replace=False).tolist()) if n_tgt else set()
        counts = {x: 0 for x in irr}

        def draw(need: int) -> list[str]:
            """The `need` irrelevants shown least often so far (random tie-break): every irrelevant is shown equally often (+-1)."""
            pick = sorted(irr, key=lambda x: (counts[x], rng.random()))[:need]
            for x in pick:
                counts[x] += 1
            return pick

        for t in range(n):
            pick = draw(3 - (1 if t in tgt_idx else 0))
            shown = [probe] + pick
            roles = ["probe"] + ["irrelevant"] * len(pick)
            if t in tgt_idx:
                shown.append(q["target"])
                roles.append("target")
            order = [0, 1, 2, 3]
            rng.shuffle(order)
            slot_item, slot_role = [None] * 4, [None] * 4
            # the probe goes to its balanced quadrant, the others fill the rest in random order
            rest = [k for k in order if k != pos[t]]
            slot_item[pos[t]], slot_role[pos[t]] = shown[0], roles[0]
            for k, it, ro in zip(rest, shown[1:], roles[1:]):
                slot_item[k], slot_role[k] = it, ro
            trials.append({"mode": "A", "question_id": q["id"], "text": q["text"], "items": slot_item, "roles": slot_role,
                           "target_pos": slot_role.index("target") if "target" in slot_role else -1})
    rng.shuffle(trials)
    return trials


def build_trials_B(items: dict, qs: list[dict], rng, picked: str | None, default_repeats: int, expected: dict) -> list[dict]:
    blocks = []
    nrep = max(int(q.get("repeats", default_repeats)) for q in qs)
    for b in range(nrep):
        blk = []
        for q in qs:
            probe = picked if q["probe"] == "@pick" else q["probe"]
            irr = [i for i in (items["_pool"] if q["irrelevants"] == "@rest" else q["irrelevants"]) if i != probe]
            blk.append((q, probe, "probe"))
            blk += [(q, i, "irrelevant") for i in irr]
            if q.get("target"):
                blk.append((q, q["target"], "target"))
        rng.shuffle(blk)
        blocks.append(blk)
    seq: list = []
    for blk in blocks:
        if seq and blk[0][1] == seq[-1][1]:                 # no immediate repeat across the block boundary
            blk.append(blk.pop(0))
        seq += blk
    return [{"mode": "B", "question_id": q["id"], "text": q["text"], "items": [it], "roles": [ro], "expected": expected[ro]}
            for q, it, ro in seq]


def build_trials_C(qs: list[dict], rng, default_repeats: int, expected: dict) -> list[dict]:
    nrep = max(int(q.get("repeats", default_repeats)) for q in qs)
    seq = []
    for b in range(nrep):
        blk = list(qs)
        rng.shuffle(blk)
        if seq and blk[0]["id"] == seq[-1]["id"]:
            blk.append(blk.pop(0))
        seq += blk
    return [{"mode": "C", "question_id": q["id"], "text": q["text"], "items": [q["id"]], "roles": [q["type"]],
             "expected": expected[q["type"]]} for q in seq]


DEFAULT_EXPECTED = {"probe": "no", "irrelevant": "no", "target": "yes", "relevant": "no", "comparison": "no"}


def target_name(iset: dict, items: dict, tid: str, lang: str) -> str:
    """How feedback names the target: the item set's `target_name`, else "the <label>" (English) or the label (other languages)."""
    if iset.get("target_name"):
        return str(iset["target_name"])
    label = str(items[tid].get("label", tid))
    return f"the {label.lower()}" if lang == "en" else label


def build_practice(iset: dict, items: dict, qs: list[dict], rng, picked: str | None, mode: str, expected: dict, lang: str,
                   n: int = 2) -> list[dict]:
    """Two unscored practice trials for modes A and B, built only from items that are neither the probe nor the target: one with
    the target shown (answer yes) and one without (answer no), in random order. Returns [] when the item set has no target or
    too few other items (mode A needs 4)."""
    if n <= 0 or mode not in ("A", "B") or not qs or not qs[0].get("target"):
        return []
    q = qs[0]
    probe = picked if q["probe"] == "@pick" else q["probe"]
    tgt = q["target"]
    pool = [i for i in (iset["_pool"] if q["irrelevants"] == "@rest" else q["irrelevants"]) if i not in (probe, tgt)]
    if len(pool) < (4 if mode == "A" else 1):
        return []
    name = target_name(iset, items, tgt, lang)
    out = []
    for show in rng.permutation([True, False]).tolist():
        if mode == "A":
            others = rng.choice(pool, 3 if show else 4, replace=False).tolist()
            shown = others + ([tgt] if show else [])
            roles = ["practice"] * len(others) + (["target"] if show else [])
            order = rng.permutation(4).tolist()
            slot_item, slot_role = [shown[j] for j in order], [roles[j] for j in order]
            out.append({"mode": "A", "question_id": q["id"], "text": q["text"], "items": slot_item, "roles": slot_role,
                        "target_pos": slot_role.index("target") if show else -1, "target_name": name})
        else:
            it = tgt if show else str(rng.choice(pool))
            out.append({"mode": "B", "question_id": q["id"], "text": q["text"], "items": [it],
                        "roles": ["target" if show else "practice"], "expected": expected["target" if show else "irrelevant"],
                        "target_name": name})
    return out


# ----------------------------------------------------------------------------- memory task (faces, short-term-memory CIT)
def _match(adj: list, n_right: int, rng) -> list | None:
    """Matching of every left node i to one of its allowed right nodes adj[i], each right node used once (augmenting paths,
    random order). Returns the right node of each left node, or None when some left node cannot be matched."""
    adj = [rng.permutation(a).tolist() for a in adj]
    owner = [-1] * n_right

    def grab(i: int, seen: set) -> bool:
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                if owner[j] < 0 or grab(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    for i in rng.permutation(len(adj)).tolist():
        if not grab(i, set()):
            return None
    out = [-1] * len(adj)
    for j, i in enumerate(owner):
        if i >= 0:
            out[i] = j
    return out


def memory_pools(iset: dict, rng) -> dict:
    """Split the item set's faces with the session's seeded RNG: one or two test pools (each with its own probe, chosen at random among
    its faces for every participant, plus the irrelevants), the innocent-scene faces (one per pool, never in the test) and the practice
    faces (practice only). `test_pool` / `irrelevants` are the flat lists over all pools (9 / 8 faces with one pool, 18 / 16 with two)."""
    ids = [i["id"] for i in iset["items"]]
    pools = iset.get("pools", {})
    sizes, n_scene = [int(x) for x in pools.get("test", [9, 9])], int(pools.get("scene", 2))
    order = [str(x) for x in rng.permutation(ids)]
    test_pools, k = [], 0
    for m in sizes:
        test_pools.append(order[k:k + m])
        k += m
    probes = [str(rng.choice(p)) for p in test_pools]
    irr = [[f for f in p if f != pr] for p, pr in zip(test_pools, probes)]
    return {"test_pools": test_pools, "probes": probes, "irrelevants_by_pool": irr, "test_pool": sum(test_pools, []),
            "irrelevants": sum(irr, []), "scene_faces": order[k:k + n_scene], "practice_faces": order[k + n_scene:]}


def _memory_sets(faces: list, n: int, rng) -> list:
    """n displays of 4 distinct faces; each display takes the 4 faces shown least often so far (random tie-break), so after n displays
    (4n a multiple of len(faces)) every face has been shown exactly 4n / len(faces) times."""
    counts = {f: 0 for f in faces}
    out = []
    for _ in range(n):
        pick = sorted(faces, key=lambda f: (counts[f], rng.random()))[:4]
        for f in pick:
            counts[f] += 1
        out.append(pick)
    return out


def _memory_tests(sets: list, faces: list, rng) -> list | None:
    """Test face of every display: half the displays (n // 2, one more at random when n is odd) test a face that IS in the display
    (answer yes), the rest a test-pool face that is NOT (answer no); each face is the test face n / len(faces) times, as evenly as
    possible yes and no (1 + 1 for n = 18). Random split into yes and no displays, then matching; None if it does not work out."""
    n = len(sets)
    n_yes = n // 2 + (int(rng.integers(0, 2)) if n % 2 else 0)
    per = n // len(faces)
    extra = set(rng.permutation(len(faces))[:n_yes % len(faces)].tolist())
    yes_q = [n_yes // len(faces) + (k in extra) for k in range(len(faces))]
    yes_idx = rng.permutation(n)[:n_yes].tolist()
    no_idx = [t for t in range(n) if t not in set(yes_idx)]
    yes_slots = [f for f, q in zip(faces, yes_q) for _ in range(q)]
    no_slots = [f for f, q in zip(faces, yes_q) for _ in range(per - q)]
    my = _match([[j for j, t in enumerate(yes_idx) if f in sets[t]] for f in yes_slots], len(yes_idx), rng)
    mn = _match([[j for j, t in enumerate(no_idx) if f not in sets[t]] for f in no_slots], len(no_idx), rng)
    if my is None or mn is None:
        return None
    test = [""] * n
    for f, j in zip(yes_slots, my):
        test[yes_idx[j]] = f
    for f, j in zip(no_slots, mn):
        test[no_idx[j]] = f
    return test


def _memory_slots(sets: list, faces: list, rng) -> list:
    """Quadrant (0..3 = TL, TR, BL, BR) of every face in every display, such that each face is in each quadrant equally often.
    Each face is split into len(sets) / len(faces) 'copies' of 4 displays; displays x copies is a 4-regular bipartite graph, which
    splits into 4 perfect matchings, one per quadrant: every display gets each quadrant once and every copy (so every face) is in
    each quadrant the same number of times."""
    n = len(sets)
    per = n // len(faces)
    copy_of = {}
    for fi, f in enumerate(faces):
        shown = rng.permutation([t for t in range(n) if f in sets[t]]).tolist()
        for c in range(per):
            for t in shown[4 * c:4 * c + 4]:
                copy_of[(t, f)] = fi * per + c
    adj = [[copy_of[(t, f)] for f in sets[t]] for t in range(n)]
    out = [[None] * 4 for _ in range(n)]
    for quad in rng.permutation(4).tolist():
        m = _match(adj, n, rng)
        if m is None:
            raise RuntimeError("quadrant assignment failed")
        for t in range(n):
            out[t][quad] = faces[m[t] // per]
            adj[t].remove(m[t])
    return out


def _memory_order_ok(order: list, trials: list) -> bool:
    """Consecutive displays differ (as sets of faces), consecutive test faces differ, no run of more than 3 equal answers."""
    for a, b in zip(order, order[1:]):
        if set(trials[a]["items"]) == set(trials[b]["items"]) or trials[a]["test_item"] == trials[b]["test_item"]:
            return False
    ans = [trials[t]["test_in_display"] for t in order]
    return all(len(set(ans[i:i + 4])) > 1 for i in range(len(ans) - 3))


def build_memory_trials(pool: list, probe: str, n: int, rng, qid: str, text: str, tries: int = 400, pool_no: int = 1) -> list[dict]:
    """The n test displays of ONE test pool of the memory task with EQUAL EXPOSURE (avoids Delmas' presentation-count confound): every test-pool face,
    the probe included, is in exactly 4n / len(pool) displays (8 of 18), in each quadrant equally often (2 times for n = 18), never
    twice in one display. The test face is in the display in half the trials (answer yes) and a test-pool face that is not in it
    in the other half; every face is the test face n / len(pool) times (the probe once with yes and once with no for n = 18).
    Roles are `probe` / `irrelevant` (there is no target in this task); `pool` = pool_no tags every trial."""
    if n % len(pool):
        raise ValueError(f"n_trials ({n}) must be a multiple of the test pool size ({len(pool)})")
    for _ in range(tries):
        sets = _memory_sets(pool, n, rng)
        test = _memory_tests(sets, pool, rng)
        if test is None:
            continue
        slots = _memory_slots(sets, pool, rng)
        trials = [{"mode": "A", "task": "memory", "question_id": qid, "text": text, "items": slots[t],
                   "roles": ["probe" if f == probe else "irrelevant" for f in slots[t]], "target_pos": -1,
                   "test_item": test[t], "test_in_display": int(test[t] in sets[t]), "test_is_probe": int(test[t] == probe),
                   "pool": pool_no}
                  for t in range(n)]
        for _ in range(200):
            order = rng.permutation(n).tolist()
            if _memory_order_ok(order, trials):
                return [trials[t] for t in order]
    raise RuntimeError("could not build a memory trial plan with the required constraints")


def _pool_counts(trials: list, pool: list, probe: str) -> dict:
    """What the plan guarantees within one pool, counted: exposures per face, face x quadrant counts, the probe's quadrant counts,
    test-face counts (total / yes / no per face) and the number of yes answers."""
    quad = {f: [sum(t["items"][q] == f for t in trials) for q in range(4)] for f in pool}
    tests = {f: {"total": sum(t["test_item"] == f for t in trials),
                 "yes": sum(t["test_item"] == f and t["test_in_display"] == 1 for t in trials),
                 "no": sum(t["test_item"] == f and t["test_in_display"] == 0 for t in trials)} for f in pool}
    return {"n_trials": len(trials), "exposures": {f: sum(quad[f]) for f in pool}, "quadrant_counts": quad,
            "probe_quadrants": quad[probe], "test_counts": tests, "n_yes": sum(t["test_in_display"] for t in trials)}


def _max_run(seq: list) -> int:
    """Length of the longest run of equal consecutive values."""
    best = run = 1 if seq else 0
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def memory_plan_counts(trials: list, pools: list, probes: list) -> dict:
    """What the plan guarantees, counted: per pool (`pools`) the exposures, quadrant and test-face counts; over all trials the pool
    sequence, its longest run, the longest run of equal answers, identical consecutive displays or test faces, and faces twice in
    a display."""
    per = [_pool_counts([t for t in trials if t["pool"] == k + 1], pool, probe) for k, (pool, probe) in enumerate(zip(pools, probes))]
    seq = [t["pool"] for t in trials]
    return {"n_trials": len(trials), "pools": per, "pool_sequence": seq, "max_pool_run": _max_run(seq),
            "max_answer_run": _max_run([t["test_in_display"] for t in trials]), "n_yes": sum(t["test_in_display"] for t in trials),
            "repeated_displays": sum(set(a["items"]) == set(b["items"]) for a, b in zip(trials, trials[1:])),
            "repeated_test_faces": sum(a["test_item"] == b["test_item"] for a, b in zip(trials, trials[1:])),
            "duplicate_faces_in_display": sum(len(set(t["items"])) != 4 for t in trials)}


def interleave_pools(per_pool: list, rng, max_run: int = 3, tries: int = 20000) -> list[dict]:
    """Mix the trial lists of the pools in random order (each list keeps its own order, which already has consecutive displays and test
    faces differing): no more than `max_run` consecutive trials from the same pool, and no run of more than 3 equal answers over the
    whole sequence (the per-pool plans only guarantee that inside a pool). Two displays of different pools never share a face."""
    base = [k for k, tr in enumerate(per_pool) for _ in tr]
    for _ in range(tries):
        order = rng.permutation(base).tolist()
        if _max_run(order) > max_run:
            continue
        nxt, out = [0] * len(per_pool), []
        for k in order:
            out.append(per_pool[k][nxt[k]])
            nxt[k] += 1
        if _max_run([t["test_in_display"] for t in out]) <= 3:
            return out
    raise RuntimeError("could not interleave the pools with the required constraints")


def build_memory_practice(plan: dict, qid: str, text: str, rng, n: int = 2) -> list[dict]:
    """Unscored practice trials from the 2 practice faces (used nowhere else): one display holds practice face A and the test face is
    A (answer yes); in the other the display holds practice face B and the test face is A (answer no), in random order. The other
    three faces of a practice display are test-pool irrelevants (never a probe), the least used so far."""
    pf, irr = plan["practice_faces"], plan["irrelevants"]
    if n <= 0 or len(pf) < 2 or len(irr) < 3:
        return []
    kinds = rng.permutation([True, False]).tolist() if n == 2 else ([True, False] * n)[:n]
    used = {f: 0 for f in irr}
    out = []
    for yes in kinds:
        fill = sorted(irr, key=lambda f: (used[f], rng.random()))[:3]
        for f in fill:
            used[f] += 1
        shown = fill + [pf[0] if yes else pf[1]]
        slot = [shown[j] for j in rng.permutation(4).tolist()]
        out.append({"mode": "A", "task": "memory", "question_id": qid, "text": text, "items": slot, "roles": ["practice"] * 4,
                    "target_pos": -1, "test_item": pf[0], "test_in_display": int(yes), "test_is_probe": 0})
    return out


LINEUP_N = 8


def build_memory_lineup(plan: dict, rng) -> list:
    """The 8 faces of the story-recognition line-up in their random positions 1..8: all probes, all innocent-scene faces and test-pool
    irrelevants to fill up to 8 (4 from two pools, 2 from each; 6 from the one pool). The set does not depend on the group: everyone
    sees the same kinds of faces."""
    k = len(plan["probes"])
    per_pool = (LINEUP_N - k - len(plan["scene_faces"])) // k
    irr = [str(f) for p in plan["irrelevants_by_pool"] for f in rng.permutation(p)[:per_pool]]
    faces = [*plan["probes"], *plan["scene_faces"], *irr]
    return [faces[j] for j in rng.permutation(len(faces)).tolist()]


def lineup_score(selected: list, story: list, probes: list) -> dict:
    """Score of the line-up. `story` = the person's OWN story faces (knowledgeable / countermeasure: the probes; innocent: the innocent-scene
    faces; one or two men). lineup_story_hits: own story faces selected (0-2, 0-1 with one man); lineup_probe_selected: probes selected
    (for an innocent a false recognition); lineup_false: faces selected that are neither own story faces nor probes; lineup_both: all
    the story men recognised (with one man: the story man)."""
    sel = set(selected)
    hits = len(sel & set(story))
    return {"lineup_story_hits": hits, "lineup_probe_selected": len(sel & set(probes)),
            "lineup_false": len(sel - set(story) - set(probes)), "lineup_n_selected": len(sel), "lineup_both": int(hits == len(story))}


def build_memory_plan(iset: dict, rng, n_trials: int | None = None, n_practice: int = 2) -> dict:
    """Pools, trials, practice trials, line-up and the counts of the plan for one session (everything comes from `rng`, the session's
    seeded generator). `iset` is the (localised) item set. The n trials are n / (number of pools) per pool (each from build_memory_trials
    with the pool's own probe); two pools are mixed in random order (interleave_pools), with one pool its own plan is the trial list."""
    pools = memory_pools(iset, rng)
    q = iset["questions"][0]
    n = int(n_trials or iset["n_trials"])
    k = len(pools["test_pools"])
    if n % k or any((n // k) % len(p) for p in pools["test_pools"]):
        raise ValueError(f"n_trials ({n}) must be a multiple of {k} x the test pool size ({[len(p) for p in pools['test_pools']]}); "
                         f"{k * len(pools['test_pools'][0])} is the smallest")
    for _ in range(50):          # a pair of per-pool orders that cannot be interleaved within the limits (rare) is replaced by a new pair
        per = [build_memory_trials(p, pr, n // k, rng, q["id"], q["text"], pool_no=i + 1)
               for i, (p, pr) in enumerate(zip(pools["test_pools"], pools["probes"]))]
        try:
            trials = per[0] if k == 1 else interleave_pools(per, rng, tries=2000)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("could not build a memory trial plan with the required constraints")
    plan = {**pools, "n_trials": n, "trials": trials, "practice": build_memory_practice(pools, q["id"], q["text"], rng, n_practice)}
    plan["lineup_faces"] = build_memory_lineup(pools, rng)
    plan["counts"] = memory_plan_counts(trials, pools["test_pools"], pools["probes"])
    return plan


# ----------------------------------------------------------------------------- drift check
def drift_from_frames(obs: list, cross_px: tuple, ppd: float, max_deg: float, thr: float) -> tuple[float, float, float, bool]:
    """Offset (dx, dy px) to add to the gaze estimates so the median gaze during fixation lands on the cross; `ok` is False
    (and the offset 0) when too few samples or the offset exceeds `max_deg`."""
    ok = [o for o in obs if o.face and np.isfinite(o.gaze_x) and not (np.isfinite(thr) and o.ear < thr)]
    if len(ok) < 8:
        return 0.0, 0.0, float("nan"), False
    mx, my = np.median([o.gaze_x for o in ok]), np.median([o.gaze_y for o in ok])
    dx, dy = cross_px[0] - mx, cross_px[1] - my
    deg = float(np.hypot(dx, dy) / ppd)
    if deg > max_deg:
        return 0.0, 0.0, deg, False
    return float(dx), float(dy), deg, True


# ----------------------------------------------------------------------------- session runner
def new_session_id(group: str) -> str:
    return time.strftime("S%Y%m%d_%H%M%S")


def _wait_until(disp, t_end_ns: int, keys: set | None = None, ignored: list | None = None):
    """Poll keys every ~1 ms until t_end_ns; returns (key, t_ns) of the first key in `keys`, or None.
    With keys=None nothing ends the wait early (a stray key must not cut a timed display short); other presses are
    appended to `ignored` as (key, t_ns) when it is given. Keypad digits count as the same digits ("[1]" -> "1").
    Sleeps with time.sleep (high-resolution timer): pygame.time.wait fell back to 15 ms ticks after pygame.quit() / init cycles."""
    while True:
        for k, t in disp.poll():
            k = k.strip("[]") if len(k) == 3 and k.startswith("[") else k
            if keys is not None and k in keys:
                return k, t
            if ignored is not None:
                ignored.append((k, t))
        left = t_end_ns - CLOCK.now_ns()
        if left <= 0:
            return None
        if left > 3_000_000:                 # sleep while far from the end, spin (still polling keys) over the last 3 ms
            time.sleep(0.001)


def _auto_continue(disp, seconds: float) -> None:
    """Hold the current screen for `seconds`, then go on by itself; SPACE or a click goes on at once. Esc aborts (KeyboardInterrupt
    from poll), as everywhere."""
    disp.clicks.clear()
    t_end = CLOCK.now_ns() + int(seconds * 1e9)
    while CLOCK.now_ns() < t_end:
        if any(k == "space" for k, _ in disp.poll()) or disp.clicks:
            break
        time.sleep(0.005)
    disp.clicks.clear()


def gate_thresholds(cfg: Config, accuracy_deg: float | None = None, sd_deg: float | None = None) -> dict:
    """Gaze-gate limits (deg) for one calibration segment, from its own validation. The config values are floors:
        narrow = max(fixation_gate_deg, 1.5 x accuracy)        all frames of the hold within this of the target
        wide   = max(gate_wide_deg, accuracy + 2 x SD)         frames within this also pass when the fixation is steady ...
        stable = max(gate_stable_deg, 1.5 x SD)                ... radial SD of the gaze over the hold at most this
    A person whose webcam error is large is given room for it (a steady gaze at their own typical offset is not looking away); a
    person with a small error keeps the config limits. Without usable validation numbers the acceptance limit is used, as for
    the layout."""
    acc = float(accuracy_deg) if accuracy_deg is not None and np.isfinite(accuracy_deg) else cfg.accuracy_threshold_deg
    sd = float(sd_deg) if sd_deg is not None and np.isfinite(sd_deg) else cfg.accuracy_threshold_deg / 2
    narrow = max(cfg.fixation_gate_deg, 1.5 * acc)
    wide = max(cfg.gate_wide_deg, acc + 2 * sd, narrow)
    stable = max(cfg.gate_stable_deg, 1.5 * sd)
    return {"accuracy_deg": acc, "sd_deg": sd, "narrow_deg": float(narrow), "wide_deg": float(wide), "stable_deg": float(stable),
            "first_wide_deg": float(max(cfg.gate_first_wide_deg, wide))}


GATE_MSG_Y_FRAC = 0.08       # the gate message is centred this far down the screen (share of its height) ...
GATE_MSG_SIZE = 40
GATE_CLEAR_DEG = 6.0         # ... and no text is drawn within this many degrees of the screen centre while the gate runs


def gate_message(disp, ppd: float) -> None:
    """The gaze-gate message at the TOP of the screen (about 8% down), never next to the target: text beside the target pulls the
    eyes away from it. If the line would come within GATE_CLEAR_DEG of the screen centre (very small screens) it is moved up."""
    W, H = disp.size
    h = disp.font(max(8, int(round(GATE_MSG_SIZE * disp.ui)))).get_height()
    cy = GATE_MSG_Y_FRAC * H
    limit = H / 2 - GATE_CLEAR_DEG * ppd                                  # the line's bottom edge must stay above this
    if cy + h / 2 > limit:
        cy = max(h / 2, limit - h / 2)
    disp.text(disp.tr("look.cross"), (W // 2, int(cy)), GATE_MSG_SIZE)


def drift_is_bad(cfg: Config, row: dict) -> bool:
    """A trial counts towards the automatic re-check when its drift check failed or its drift is above cfg.drift_recheck_deg."""
    deg = row.get("drift_deg", float("nan"))
    return bool(row.get("fixation_ok", 1) == 0 or (np.isfinite(deg) and deg > cfg.drift_recheck_deg))


class RecalNeeded(Exception):
    """Raised by the gaze gate after cfg.gate_max_timeouts timeouts in a row: the session loop runs a recalibration."""


def _gate_run(obs: list, cross: tuple, gate_px: float, drift: tuple, ear_thr: float, hold_ns: int, now_ns: int,
              wide_px: float | None = None, stable_px: float | None = None):
    """Start time (ns) of the latest unbroken run of valid on-cross frames that has lasted `hold_ns`, or None. A frame counts when
    a face is found, the eyes are open and the live gaze (plus the last trial's drift correction) is within `gate_px` of the
    cross; a missing frame (gap over 150 ms) breaks the run, and the newest frame must be recent.
    With `wide_px` and `stable_px`, a run within `wide_px` of the cross also passes when its last `hold_ns` is a steady fixation
    (radial SD of the gaze <= `stable_px`). A steady gaze a few degrees off the cross is the usual webcam offset after a posture
    change, not looking away; the drift correction taken from this fixation then removes it."""
    if not obs or now_ns - obs[-1].t_ns > 150_000_000:
        return None
    start, prev = None, None
    for o in reversed(obs):
        ok = (o.face and np.isfinite(o.gaze_x) and not (np.isfinite(ear_thr) and o.ear < ear_thr)
              and np.hypot(o.gaze_x + drift[0] - cross[0], o.gaze_y + drift[1] - cross[1]) <= (wide_px or gate_px))
        if not ok or (prev is not None and prev - o.t_ns > 150_000_000):
            break
        start, prev = o.t_ns, o.t_ns
    if start is None or obs[-1].t_ns - start < hold_ns:
        return None
    if not wide_px:
        return start
    last = [o for o in obs if o.t_ns >= obs[-1].t_ns - hold_ns]
    x = np.array([o.gaze_x + drift[0] for o in last]); y = np.array([o.gaze_y + drift[1] for o in last])
    near = bool((np.hypot(x - cross[0], y - cross[1]) <= gate_px).all())
    steady = stable_px is not None and float(np.sqrt(x.var() + y.var())) <= stable_px
    return start if near or steady else None


def _fixation(disp, pipe, cfg, rec, ppd: float, trial_no, ctx: dict):
    """Fixation target (Thaler et al. 2013 bullseye with cross-hair) with a GAZE GATE: the trial starts only after gaze has been on
    the target for cfg.gate_hold_seconds without a break. The limits come from the current segment's own validation
    (ctx["gate"], see gate_thresholds): within `narrow`, or within `wide` when the fixation is steady (radial SD <= `stable`).
    While no drift offset has been measured in this segment (ctx["anchored"] False: the first trials after a calibration, or after
    a trial whose drift check failed) the wide limit is cfg.gate_first_wide_deg, because a posture change since the calibration can
    shift the estimate by 10 deg or more; the trial's drift check then decides whether the offset is usable.
    After cfg.gate_timeout_seconds the message "Please look at the dot in the middle." is shown at the TOP of the screen (nothing
    else is ever drawn near the target) and the wait goes on; each further period is a timeout, and cfg.gate_max_timeouts in a row
    raise RecalNeeded. The target then stays for cfg.fixation_seconds counted from the start of the successful run, so the drift
    window (taken from the run) holds only on-target samples. Returns (target on, target position, run start, ms waited for the
    gate, timeouts in this wait)."""
    W, H = disp.size
    cross = (W / 2, H / 2)
    disp.hint = {"phase": "fixation", "target_px": cross, "trial": trial_no}
    disp.clear()
    disp.fixation_target(cross, ppd)
    t_on = disp.flip()
    hold, tmo = int(cfg.gate_hold_seconds * 1e9), int(cfg.gate_timeout_seconds * 1e9)
    g = ctx.get("gate") or gate_thresholds(cfg)
    wide_deg = g["wide_deg"] if ctx.get("anchored", False) else g["first_wide_deg"]
    gate_px, wide_px, stable_px = g["narrow_deg"] * ppd, wide_deg * ppd, g["stable_deg"] * ppd
    t_next, n_to, shown, t_start = t_on + tmo, 0, False, None
    while t_start is None:
        disp.poll()                                              # Esc aborts; other keys are ignored here
        now = CLOCK.now_ns()
        t_start = _gate_run(pipe.window(t_on, now), cross, gate_px, ctx["drift"], pipe.ear_threshold(), hold, now, wide_px, stable_px)
        if t_start is None:
            if now >= t_next:
                n_to += 1
                ctx["gate_timeouts"] += 1
                rec.event(CLOCK.ms(now), "gate_timeout", f"{trial_no}:{ctx['gate_timeouts']}")
                if ctx["gate_timeouts"] >= cfg.gate_max_timeouts:
                    raise RecalNeeded()
                disp.clear()
                disp.fixation_target(cross, ppd)
                gate_message(disp, ppd)
                disp.flip()
                shown, t_next = True, now + tmo
            disp.pg.time.wait(5)
    wait_ms = (CLOCK.now_ns() - t_on) / 1e6
    if n_to == 0:
        ctx["gate_timeouts"] = 0                                 # "in a row" ends with a clean gate
    if shown:
        disp.clear()
        disp.fixation_target(cross, ppd)
        disp.flip()
    _wait_until(disp, t_start + int(cfg.fixation_seconds * 1e9) - 17_000_000)       # leave one refresh for the stimulus flip
    return t_on, cross, t_start, wait_ms, n_to


def feedback_text(disp, answer: str, expected: str, tname: str, mode: str) -> str:
    """Practice feedback: Correct / Not quite: ... / Too slow (no answer). Mode "M" = the memory question of the face item sets."""
    if not answer:
        return disp.tr("fb.slow")
    if answer == expected:
        return disp.tr("fb.correct")
    if mode == "M":
        return disp.tr("fb.mem_shown" if expected == "yes" else "fb.mem_not_shown")
    if mode == "A":
        return disp.tr("fb.target_there" if expected == "yes" else "fb.target_not_there", target=tname)
    return disp.tr("fb.b_is" if expected == "yes" else "fb.b_is_not", target=tname)


def _message(disp, cfg, text: str, phase: str = "feedback") -> None:
    """Large centred message for cfg.feedback_seconds (practice feedback, look-away notice)."""
    disp.hint = {"phase": phase}
    disp.clear()
    disp.paragraph([text], int(disp.size[1] / 2 - 40 * disp.ui), 60)
    t = disp.flip()
    _wait_until(disp, t + int(cfg.feedback_seconds * 1e9))


def run_trial(disp, pipe, cfg, rec, layout, trial: dict, no: int, items: dict, ppd: float, ctx: dict | None = None,
              practice: bool = False) -> dict:
    """One trial. `ctx` carries the state shared between trials: last drift correction (px), gate timeouts in a row, the head-pitch
    reference from calibration, the calibration segment, the gate limits of the segment and whether a drift offset has been measured
    in it. Practice trials give feedback and return after the drift check, before the look-away check (they are saved in
    practice.csv and never scored; their drift anchors the gate and counts towards the automatic re-check)."""
    ctx = ctx if ctx is not None else {"drift": (0.0, 0.0), "gate_timeouts": 0, "pitch_ref": float("nan"), "segment": 0}
    t_fix_on, cross, t_run, gate_ms, n_to = _fixation(disp, pipe, cfg, rec, ppd, None if practice else no, ctx)
    row = {"trial": no, "mode": trial["mode"], "question_id": trial["question_id"], "fix_on_ms": CLOCK.ms(t_fix_on),
           "gate_wait_ms": gate_ms, "gate_timeouts": n_to, "segment": ctx["segment"]}
    if practice:
        row["practice"] = 1
    else:
        row.update({"is_repeat": int(bool(trial.get("is_repeat"))), "repeat_of": int(trial.get("repeat_of", -1))})
    keys_yes_no = {cfg.yes_key, cfg.no_key}
    answer_of = {cfg.yes_key: "yes", cfg.no_key: "no"}
    W, H = disp.size
    if trial["mode"] == "A":
        rects = layout["rects"]
        disp.hint = {"phase": "display", "rects": rects, "roles": trial["roles"], "trial": None if practice else no}
        disp.clear()
        for k in range(4):
            x0, y0, x1, y1 = rects[k]
            disp.draw_item(items[trial["items"][k]], (x0, y0, x1 - x0, y1 - y0))
        onset = disp.flip()
        early: list = []
        _wait_until(disp, onset + int(cfg.display_seconds * 1e9), ignored=early)
        for k, t in early:                   # keys pressed while the shapes were shown: logged, not taken as the answer
            rec.event(CLOCK.ms(t), "early_key", f"{no}:{k}")
        row["early_keys"] = " ".join(k for k, _ in early)
        # target question (yes / no on F / J, read by scancode): static screen, keys polled at ~1 kHz. The question is only a
        # cover task and an attention check; the measurement is the gaze during the display above.
        # Face item sets (memory task): the question is "was this test face one of the four?", the test face is drawn in the
        # centre at the size of one display card, the answer is yes exactly when it was in the display.
        memory = trial.get("task") == "memory"
        target = trial["target_pos"]
        expected = ("yes" if trial["test_in_display"] else "no") if memory else ("yes" if target >= 0 else "no")
        disp.hint = {"phase": "response", "expected": expected}
        disp.clear()
        # each label sits on the side of its key (F on the left, J on the right) whatever the reading direction
        kw = {"yes": cfg.yes_key.upper(), "no": cfg.no_key.upper(), "yes_ar": L.AR_KEY_LETTER.get(cfg.yes_key, ""),
              "no_ar": L.AR_KEY_LETTER.get(cfg.no_key, "")}
        if memory:
            fw, fh = rects[0][2] - rects[0][0], rects[0][3] - rects[0][1]
            face = (W / 2 - fw / 2, H / 2 - fh / 2, fw, fh)
            disp.text(trial["text"], (W // 2, int(face[1] - 30 * disp.ui)), 46, anchor="midbottom")
            disp.draw_item(items[trial["test_item"]], face)
            y_lab = int(face[1] + fh + 30 * disp.ui)
            disp.text(disp.tr("q.yes", **kw), (int(W * 0.30), y_lab), 58, anchor="midtop")
            disp.text(disp.tr("q.no", **kw), (int(W * 0.70), y_lab), 58, anchor="midtop")
            timeout = cfg.memory_timeout_seconds
        else:
            disp.text(trial["text"], (W // 2, int(H / 2 - 90 * disp.ui)), 46)
            disp.text(disp.tr("q.yes", **kw), (int(W * 0.30), int(H / 2 + 30 * disp.ui)), 58)
            disp.text(disp.tr("q.no", **kw), (int(W * 0.70), int(H / 2 + 30 * disp.ui)), 58)
            timeout = cfg.response_timeout_seconds
        t_q = disp.flip()
        r = _wait_until(disp, t_q + int(timeout * 1e9), keys_yes_no)
        key, rt = (r[0], (r[1] - t_q) / 1e6) if r else ("", float("nan"))
        resp = answer_of[key] if key else ""
        row.update({"onset_ms": CLOCK.ms(onset), "end_ms": CLOCK.ms(t_q), "response_key": key, "response": resp, "rt_ms": rt,
                    "expected": expected, "correct": float(resp == expected) if key else float("nan")})
        if memory:
            row.update({"test_item": trial["test_item"], "test_in_display": int(trial["test_in_display"]),
                        "test_is_probe": int(trial["test_is_probe"])})
            if "pool" in trial:                                      # which of the two test pools the display (and its probe) belongs to
                row["pool"] = int(trial["pool"])
        else:
            row["target_shown"] = int(target >= 0)
        for k in range(4):
            row[f"item{k}"], row[f"role{k}"] = trial["items"][k], trial["roles"][k]
        if practice:
            _measure_drift(pipe, cfg, row, ctx, cross, ppd, t_run, onset)
            _message(disp, cfg, feedback_text(disp, resp, expected, trial.get("target_name", ""), "M" if memory else "A"))
    else:
        it = items[trial["items"][0]] if trial["mode"] == "B" else {"id": trial["items"][0], "draw": {"kind": "text", "text": ""}}
        disp.hint = {"phase": "display", "rects": [], "roles": trial["roles"]}
        disp.clear()
        cx, cy = W / 2, H / 2
        if trial["mode"] == "B":
            # the text's bottom is at least the layout's required gap (px via ppd) above the item's top
            geo = mode_b_geometry(disp.size, ppd, layout, disp.font(max(8, int(round(32 * disp.ui)))).get_height())
            rec.meta["mode_b_geometry"] = geo
            disp.text(trial["text"], tuple(geo["text_mid_bottom_px"]), 32, anchor="midbottom")
            disp.draw_item(it, tuple(geo["item_rect_px"]))
        else:
            disp.paragraph([trial["text"]], int(cy - 20), 40)
        onset = disp.flip()
        r = _wait_until(disp, onset + int(cfg.seq_timeout_seconds * 1e9), keys_yes_no)
        key = answer_of[r[0]] if r else ""
        rt = (r[1] - onset) / 1e6 if r else float("nan")
        if practice:
            _measure_drift(pipe, cfg, row, ctx, cross, ppd, t_run, onset)
            _message(disp, cfg, feedback_text(disp, key, trial["expected"], trial.get("target_name", ""), "B"))
        disp.clear()
        t_end = disp.flip()
        _wait_until(disp, t_end + int(cfg.seq_iti_seconds * 1e9))
        row.update({"onset_ms": CLOCK.ms(onset), "end_ms": CLOCK.ms(t_end), "response_key": key, "rt_ms": rt,
                    "expected": trial["expected"], "correct": float(key == trial["expected"]) if key else float("nan"),
                    "response_item": key, "target_shown": int(trial["roles"][0] == "target"),
                    "item0": trial["items"][0], "role0": trial["roles"][0]})
    if practice:
        rec.event(row["onset_ms"], "practice_trial", f"{no}:{trial['question_id']}")
        return row
    # drift check from the fixation period, computed after the trial; the drift also corrects the gaze used by the gate of the next one
    dx, dy = _measure_drift(pipe, cfg, row, ctx, cross, ppd, t_run, onset)
    row.update({"valid_share": float("nan"), "lookaway": 0})
    if trial["mode"] == "A":
        # look-away guard: share of the display's frames with a valid on-screen sample (features.onscreen_valid_share)
        t_end_disp = onset + int(cfg.display_seconds * 1e9)
        pipe.wait_processed(t_end_disp, 0.3)
        fps = pipe.cam.achieved_fps()
        share = onscreen_valid_share(pipe.window(onset, t_end_disp), disp.size, cfg.lookaway_margin_deg * ppd, (dx, dy),
                                     pipe.ear_threshold(), ctx["pitch_ref"], cfg.pitch_down_max_deg,
                                     cfg.display_seconds * (fps if np.isfinite(fps) else cfg.cam_fps))
        row["valid_share"] = share
        row["lookaway"] = int(share < cfg.min_trial_valid_share)
        if row["lookaway"]:
            rec.event(row["onset_ms"], "lookaway", f"{no}:{share:.2f}")
            _message(disp, cfg, disp.tr("look.keep"), "lookaway")
    rec.event(row["onset_ms"], "trial", f"{no}:{trial['question_id']}")
    return row


def _measure_drift(pipe, cfg, row: dict, ctx: dict, cross: tuple, ppd: float, t_run: int, onset: int) -> tuple[float, float]:
    """Drift check of a trial from its fixation period (starts 40% into the on-target run for settling, last 0.1 s dropped): writes
    drift_dx_px, drift_dy_px, drift_deg and fixation_ok into `row`. An accepted offset corrects the gate of the next trial and marks
    the segment as anchored; a failed check leaves the old offset and un-anchors it (the next gate uses the first-trial wide limit,
    and the failure counts towards the automatic re-check). Returns (dx, dy)."""
    pipe.wait_processed(onset, 0.3)
    obs = pipe.window(t_run + int(0.4 * cfg.fixation_seconds * 1e9), onset - int(0.1e9))
    dx, dy, deg, ok = drift_from_frames(obs, cross, ppd, cfg.drift_max_deg, pipe.ear_threshold())
    row.update({"drift_dx_px": dx, "drift_dy_px": dy, "drift_deg": deg, "fixation_ok": int(ok)})
    if ok:
        ctx["drift"] = (dx, dy)
    ctx["anchored"] = bool(ok)
    return dx, dy


def _pick_probe(disp, items: dict, pick: dict, group: str, rng, layout) -> tuple[str, str]:
    """The participant privately picks one item of the pool (knowledgeable / countermeasure groups); the innocent group gets a
    random probe it never saw chosen. Returns (probe id, who chose it)."""
    pool = pick["pool"]
    if group == "innocent":
        return str(rng.choice(pool)), "random"
    n = len(pool)
    cols = min(n, 4)
    rows_ = int(np.ceil(n / cols))
    W, H = disp.size
    cw, ch = W / (cols + 1) * 0.8, H / (rows_ + 2) * 0.8
    sel = 0
    while True:
        disp.clear()
        disp.text(pick.get("prompt", disp.tr("pick.default")), (W / 2, 50), 28)
        for i, iid in enumerate(pool):
            c, r = i % cols, i // cols
            x = W / 2 + (c - (cols - 1) / 2) * (cw * 1.2) - cw / 2
            y = H / 2 + (r - (rows_ - 1) / 2) * (ch * 1.15) - ch / 2
            disp.draw_item(items[iid], (x, y, cw, ch))
            if i == sel:
                disp.pg.draw.rect(disp.screen, (30, 120, 220), (x - 6, y - 6, cw + 12, ch + 12), 4)
        disp.flip()
        k = disp.wait_key({"left", "right", "up", "down", "return"})
        if k == "return":
            break
        sel = (sel + {"left": -1, "right": 1, "up": -cols, "down": cols}[k]) % n
    disp.clear()
    disp.text(disp.tr("pick.remember"), (W / 2, H / 2), 30)
    disp.flip()
    disp.wait_key({"space"})
    return pool[sel], "participant"


def _scene(disp, cfg: Config, rec, faces: list, lines: list, remember: str) -> None:
    """Mock-crime scene of the face item sets (after calibration, before the instructions): one or two men (one per test pool), one after
    the other. Each one's story line and SMILING photo (large, centred) stay for cfg.scene_seconds; after the last one the "remember"
    line is added and SPACE continues, but only after the full time, so both groups see each face equally long. Knowledgeable /
    countermeasure participants see their partner (and their buyer: the probes), innocent ones the seller (and another customer: the
    innocent-scene faces, never in the test)."""
    W, H = disp.size
    rect = (W / 2 - 0.75 * 0.52 * H / 2, 0.22 * H, 0.75 * 0.52 * H, 0.52 * H)           # 3:4 photo

    def draw(k: int, with_remember: bool) -> None:
        disp.clear()
        disp.paragraph([lines[k]], int(0.05 * H), 40)
        disp.draw_item({"id": faces[k]["id"], "draw": {"kind": "image", "path": faces[k]["scene_path"]}}, rect)
        if with_remember:
            disp.paragraph([remember], int(0.78 * H), 40)
            disp.text(disp.tr("scene.continue"), (W // 2, int(0.93 * H)), 34)

    disp.hint = {"phase": "scene"}
    for k, item in enumerate(faces):
        draw(k, False)
        t = disp.flip()
        rec.event(CLOCK.ms(t), "scene", item["id"])
        _wait_until(disp, t + int(cfg.scene_seconds * 1e9))              # keys do nothing before the time is up
    draw(len(faces) - 1, True)
    rec.event(CLOCK.ms(disp.flip()), "scene_remember", faces[-1]["id"])
    disp.wait_key({"space"})
    rec.event(CLOCK.ms(), "scene_end", faces[-1]["id"])


LINEUP_KEYS = [str(k) for k in range(1, 9)]


def _lineup(disp, cfg: Config, rec, faces: list, items: dict, question: str, story: list) -> dict:
    """Story-recognition line-up (after the end validation, before the debrief): the 8 neutral photos in a 2 x 4 grid numbered 1-8.
    Keys 1-8 (read by scancode) toggle a selection, drawn as a bold black outline; SPACE confirms; after cfg.lineup_timeout_seconds
    the current selection is recorded. Returns the faces in their positions, the selected face ids (in the order selected), every
    toggle with its time (ms on the session clock and since the onset), how it ended and the duration. `story` goes only into the
    hint (the synthetic participant of the self-test reads it, as it reads `expected` on the answer screen)."""
    W, H = disp.size
    cols, rows_ = 4, 2
    top, label_h = 0.21 * H, 0.06 * H
    row_h = (0.96 * H - top) / rows_
    ch = row_h - label_h - 0.02 * H
    cw = min(0.75 * ch, 0.9 * W / cols - 0.03 * W)
    ch = cw / 0.75
    rects = [(W * (i % cols + 0.5) / cols - cw / 2, top + (i // cols) * row_h, cw, ch) for i in range(len(faces))]
    sel: list = []                                                       # face ids in the order they were selected (a deselected one drops out)
    toggles: list = []
    disp.hint = {"phase": "lineup", "faces": list(faces), "story": list(story)}

    def draw() -> None:
        disp.clear()
        disp.paragraph([question], int(0.03 * H), 34)
        for i, (fid, (x, y, w, h)) in enumerate(zip(faces, rects)):
            disp.draw_item(items[fid], (x, y, w, h))
            if fid in sel:
                disp.pg.draw.rect(disp.screen, (0, 0, 0), (x - 9, y - 9, w + 18, h + 18), 8)
            disp.text(str(i + 1), (x + w / 2, y + h + 0.012 * H), 40, anchor="midtop")

    draw()
    t0 = disp.flip()
    rec.event(CLOCK.ms(t0), "lineup_onset", ",".join(faces))
    t_end, how, t_done = t0 + int(cfg.lineup_timeout_seconds * 1e9), "timeout", None
    while t_done is None:
        changed = False
        for key, t in disp.poll():                                       # every key of a poll counts (two quick presses can arrive together)
            key = key.strip("[]") if len(key) == 3 and key.startswith("[") else key       # keypad digits
            if key == "space":
                how, t_done = "space", t
                break
            if key in LINEUP_KEYS:
                fid = faces[int(key) - 1]
                on = fid not in sel
                sel.append(fid) if on else sel.remove(fid)
                toggles.append({"t_ms": CLOCK.ms(t), "since_onset_ms": (t - t0) / 1e6, "position": int(key), "face": fid, "action": "on" if on else "off"})
                rec.event(CLOCK.ms(t), "lineup_toggle", f"{key}:{fid}:{'on' if on else 'off'}")
                changed = True
        if t_done is None and CLOCK.now_ns() >= t_end:
            t_done = CLOCK.now_ns()
        elif t_done is None:
            if changed:
                draw()
                disp.flip()
            time.sleep(0.001)
    rec.event(CLOCK.ms(t_done), "lineup_end", how)
    disp.hint = {}
    return {"faces": list(faces), "selected": list(sel), "selected_positions": [faces.index(f) + 1 for f in sel], "toggles": toggles,
            "confirmed_by": how, "onset_ms": CLOCK.ms(t0), "end_ms": CLOCK.ms(t_done), "duration_s": (t_done - t0) / 1e9,
            "timeout_seconds": cfg.lineup_timeout_seconds}


def _consent(disp) -> bool:
    """Consent screen (text in docs/consent_draft.md, in the chosen language): 1 = I agree, 2 = I do not agree, by key (scancode)
    or a click on the left / right button. Esc counts as not agreeing."""
    W, H = disp.size
    disp.clear()
    disp.text(disp.tr("consent.title"), (W // 2, int(60 * disp.ui)), 44)
    disp.paragraph(["• " + disp.tr(f"consent.p{i}") for i in range(1, 9)], int(130 * disp.ui), 30, gap=22, bottom=int(H * 0.8))
    for x0, key in ((0.06, "consent.agree"), (0.54, "consent.disagree")):
        disp.pg.draw.rect(disp.screen, (0, 0, 0), (W * x0, H * 0.84, W * 0.40, H * 0.11), 3, border_radius=12)
        disp.text(disp.tr(key), (W * (x0 + 0.20), H * 0.895), 40)
    disp.flip()
    try:
        return disp.wait_key({"1", "2"}, click_keys={"left": "1", "right": "2"}) == "1"
    except KeyboardInterrupt:
        return False


def _gate_for(cfg: Config, calinfo: dict) -> dict:
    """Gaze-gate limits of a calibration segment (gate_thresholds) from its chosen validation."""
    cv = calinfo["chosen"]["validation"]
    return gate_thresholds(cfg, cv.get("accuracy_deg"), cv.get("sd_deg"))


def _layout_for(disp, cfg: Config, calinfo: dict) -> dict:
    """Option layout sized from this person's validation accuracy (if the validation gave no number, the acceptance limit)."""
    cv = calinfo["chosen"]["validation"]
    acc_g = cv.get("accuracy_deg") if np.isfinite(cv.get("accuracy_deg", np.nan)) else cfg.accuracy_threshold_deg
    layout = make_layout(disp.size, calinfo["chosen"]["px_per_deg"], cfg.aoi_layout, cfg.aoi_deg, cfg, acc_g)
    if layout["layout_too_small"]:
        print(f"WARNING: layout too small for the required gap of {layout['required_gap_deg']:.1f} deg on this screen; "
              "the session is flagged.", flush=True)
    return layout


def run_session(cfg: Config, mode: str, items_path: str, group: str, session_id: str | None = None, repeats: int | None = None,
                seed: int | None = None, disp=None, pipe=None, fullscreen: bool = True, skip_consent: bool = False,
                lang: str | None = None, exam: int | None = None) -> Path | None:
    """Full session: (exam choice, faces task without a given item set) -> language -> consent -> camera check -> calibration/validation -> instructions -> practice -> trials -> end
    validation -> save. Returns the session folder, or None when the participant did not agree (nothing is saved).
    `disp` and `pipe` can be injected (used by the self-test with a synthetic camera)."""
    from calibrate import calibrate_until_ok, camera_check, run_validation
    from display import Display
    from pipeline import TrackingPipeline

    if group not in GROUP_LABEL:
        raise ValueError(f"group must be one of {sorted(GROUP_LABEL)}")
    ask_exam = str(items_path) == FACES_ALIAS and exam is None          # `--items faces` without `--exam`: ask on the very first screen
    if exam is not None and exam not in EXAM_ITEMS:
        raise ValueError("exam must be 1 (one person) or 2 (two persons)")
    if str(items_path) == FACES_ALIAS and exam is not None:
        items_path = str(exam_itemset_path(exam))
    iset_raw = {"task": "memory"} if ask_exam else load_itemset(items_path)
    memory = iset_raw.get("task") == "memory"
    if memory and mode != "A":
        raise ValueError("a memory item set (faces) runs in mode A only")
    sid = session_id or new_session_id(group)
    rng = np.random.default_rng(seed if seed is not None else int(time.time()) % (2 ** 31))
    own_disp, own_pipe = disp is None, pipe is None
    disp = disp or Display(fullscreen=fullscreen, text_scale=cfg.ui_text_scale)
    pipe = pipe or TrackingPipeline(cfg)
    disp.hint = {}
    CLOCK.start()
    started = False
    try:
        W, H = disp.size
        # --- exam choice (faces task): the very first screen, before the language; about the protocol, never the group
        if ask_exam:
            exam = L.choose_exam(disp)
            items_path = str(exam_itemset_path(exam))
            iset_raw = load_itemset(items_path)
        elif memory and exam is None:                       # an explicit item set file: the exam is the number of men in it
            exam = len(iset_raw["pools"]["test"])
        # --- language, then consent: nothing is recorded and no camera is opened before the participant agrees
        lang = lang or L.choose_language(disp)
        disp.set_lang(lang)
        iset = L.localise(iset_raw, lang)
        items = {i["id"]: i for i in iset["items"]}
        consent = {"skipped": True}
        if not skip_consent:
            if not _consent(disp):
                disp.clear()
                disp.paragraph([disp.tr("consent.declined")], H // 2 - 30, 36)
                disp.flip()
                time.sleep(1.5)
                return None
            consent = {"agreed": True, "agreed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "t_ms": CLOCK.ms(), "language": lang}
        folder = cfg.data_path / sid
        rec = SessionRecorder(folder, sid, group, mode)
        rec.meta["glasses"] = getattr(cfg, "glasses", "unknown")   # glasses degrade webcam gaze: record and check by group
        rec.meta["language"], rec.meta["consent"] = lang, consent
        pipe.start(str(folder / "video.avi") if cfg.save_video else None)
        started = True
        rec.event(CLOCK.ms(), "consent")

        def dot_log(name: str, t_ns: int, detail: str) -> None:
            """Calibration / validation dot onsets and offsets and the rest between the two calibration sets -> events.csv."""
            rec.event(CLOCK.ms(t_ns), name, detail)
        # --- camera check (face, distance, frame rate, lighting light); the lighting result goes into the meta
        rec.meta["lighting"] = camera_check(disp, pipe, cfg)
        # --- calibration + validation
        model, calinfo = calibrate_until_ok(disp, pipe, cfg, log=dot_log)
        ppd = calinfo["chosen"]["px_per_deg"]
        if not calinfo["accepted"]:
            disp.clear()
            disp.paragraph([disp.tr("cal.poor", acc=calinfo["chosen"]["validation"]["accuracy_deg"], thr=cfg.accuracy_threshold_deg)],
                           H // 2 - 40, 30)
            disp.flip()
            disp.wait_key({"space"})
        layout = layout0 = _layout_for(disp, cfg, calinfo)
        ppd0 = ppd
        # --- trials
        expected = {**DEFAULT_EXPECTED, **iset.get("expected", {})}
        reps = repeats or int(iset.get("default_repeats", 12))
        picked, picked_by = None, None
        iset["_pool"] = iset.get("pick", {}).get("pool", [])
        if memory:
            # pools, probe (random among the test pool), practice and the equal-exposure trial plan all come from the session's RNG
            reps = repeats or int(iset["n_trials"])
            mplan = build_memory_plan(iset, rng, reps, cfg.practice_trials)
            picked, picked_by, trials, practice = mplan["probes"], "random", mplan["trials"], mplan["practice"]
            knows = GROUP_LABEL[group] == 1
            story_faces = list(mplan["probes"] if knows else mplan["scene_faces"])        # the one or two men of this person's story
            mem_meta = {k: mplan[k] for k in ("test_pools", "test_pool", "probes", "irrelevants_by_pool", "irrelevants", "scene_faces",
                                              "practice_faces", "lineup_faces", "n_trials", "counts")}
            mem_meta.update(scene_faces_shown=story_faces, scene_faces_are="probes" if knows else "innocent_scene_faces",
                            scene_seconds=cfg.scene_seconds, memory_timeout_seconds=cfg.memory_timeout_seconds)
            rec.meta["memory_task"], rec.ground_truth_extra["memory_task"] = mem_meta, mem_meta   # the same dict: the line-up result is added to both
            scene = iset["scene"]["knowledgeable" if knows else "innocent"]
            _scene(disp, cfg, rec, [items[f] for f in story_faces], scene["lines"], scene["remember"])
        else:
            if mode in ("A", "B") and any(q["probe"] == "@pick" for q in iset.get("questions", [])):
                picked, picked_by = _pick_probe(disp, items, iset["pick"], group, rng, layout)
            if mode == "A":
                trials = build_trials_A(iset, iset["questions"], rng, picked, reps)
            elif mode == "B":
                trials = build_trials_B(iset, iset["questions"], rng, picked, reps, expected)
            else:
                if not iset.get("polygraph_questions"):
                    raise ValueError("mode C needs 'polygraph_questions' in the item set")
                trials = build_trials_C(iset["polygraph_questions"], rng, reps, expected)
            practice = build_practice(iset, items, iset.get("questions", []), rng, picked, mode, expected, lang, cfg.practice_trials)
        disp.clear()
        lines = [*iset.get("instructions", {}).get(mode, disp.tr(f"instr.{mode}")), "", disp.tr("posture.reminder"), ""]
        if practice:
            lines.append(disp.tr("instr.practice", k=len(practice)))
        lines.append(disp.tr("instr.count", n=len(trials)))
        disp.paragraph(lines, H // 3, 30)
        disp.flip()
        disp.wait_key({"space"})
        ctx = {"drift": (0.0, 0.0), "gate_timeouts": 0, "segment": 0, "anchored": False,
               "pitch_ref": calinfo["chosen"].get("pitch_median_deg", float("nan")),   # head pitch while looking at the dots
               "gate": _gate_for(cfg, calinfo)}
        cur = {"model": model, "ppd": ppd, "layout": layout, "acc": calinfo["chosen"]["validation"]["accuracy_deg"]}
        rec.meta["recalibrations"] = []
        rec.meta["rejected_rechecks"] = []                 # mid-session calibrations that were worse than the one in force (not segments)
        rec.meta["auto_rechecks"] = []
        # gaze-gate limits used in each calibration segment (from that segment's own validation), segment 0 = the first
        rec.meta["gate_thresholds"] = [{"segment": 0, **ctx["gate"]}]
        val_end = None
        n_rep, cap = 0, int(np.ceil(cfg.max_repeat_share * len(trials)))
        bad_run: list = []                                   # (trial number, drift deg, fixation_ok) of the current run of high-drift trials

        def recalibrate(kind: str, no: int, extra: dict | None = None) -> None:
            """Full recalibration in mid-session (the same calibrate_until_ok path for both kinds): a short screen that starts the
            dots by itself (SPACE / click: at once; Esc aborts), then new model, layout, pitch reference and gate limits in a new
            segment, logged in meta (`recalibrations` has both kinds, `kind` tells them apart). kind "gate": the gaze gate timed
            out repeatedly ("Tracking lost"); "auto_recheck": the drift check failed on consecutive trials ("Quick re-check").
            `no` = first trial of the new segment."""
            auto = kind == "auto_recheck"
            disp.hint = {"phase": "recheck"}
            disp.clear()
            disp.paragraph([disp.tr("recheck.quick" if auto else "track.lost")], H // 2 - 30, 40)
            disp.flip()
            _auto_continue(disp, cfg.recheck_auto_seconds if auto else cfg.tracking_lost_auto_seconds)
            m2, ci2 = calibrate_until_ok(disp, pipe, cfg, skip_intro=True, log=dot_log, recal=True)
            new_acc = ci2["chosen"]["validation"]["accuracy_deg"]
            if not ci2["accepted"] and np.isfinite(cur["acc"]) and not (np.isfinite(new_acc) and new_acc <= cur["acc"]):
                # The new calibration is above the acceptance limit AND worse than the one in force (e.g. the person looked away
                # during the dots): keep the old model, layout and segment; only the drift is re-anchored at the next gate.
                pipe.model = cur["model"]
                ctx.update(drift=(0.0, 0.0), gate_timeouts=0, anchored=False)
                rec.meta["rejected_rechecks"].append({"before_trial": no, "t_ms": CLOCK.ms(), "kind": kind, "new_accuracy_deg": new_acc,
                                                      "kept_accuracy_deg": cur["acc"], "calibration": ci2["chosen"], **(extra or {})})
                rec.event(CLOCK.ms(), "recheck_rejected", f"before:{no} new:{new_acc:.2f} kept:{cur['acc']:.2f}")
                bad_run.clear()
                print(f"re-check before trial {no} rejected: {new_acc:.2f} deg is worse than the calibration in force "
                      f"({cur['acc']:.2f} deg); kept it", flush=True)
                return
            cur.update(model=m2, ppd=ci2["chosen"]["px_per_deg"], layout=_layout_for(disp, cfg, ci2), acc=new_acc)
            ctx.update(drift=(0.0, 0.0), gate_timeouts=0, segment=ctx["segment"] + 1, anchored=False, gate=_gate_for(cfg, ci2),
                       pitch_ref=ci2["chosen"].get("pitch_median_deg", float("nan")))
            rec.meta["gate_thresholds"].append({"segment": ctx["segment"], **ctx["gate"]})
            rec.meta["recalibrations"].append({
                "after_trial": no, "t_ms": CLOCK.ms(), "kind": kind, "calibration": ci2["chosen"], "calibration_accepted": ci2["accepted"],
                "model": m2.to_dict(), "layout": cur["layout"], "px_per_deg": cur["ppd"], "gate_thresholds": ctx["gate"],
                "distance_mm": ci2["chosen"]["distance_mm"], "pitch_median_deg": ctx["pitch_ref"], **(extra or {})})
            bad_run.clear()
            what = "automatic re-check" if auto else "recalibrated"
            print(f"{what} before trial {no}: accuracy {ci2['chosen']['validation']['accuracy_deg']:.2f} deg", flush=True)

        def run_one(tr: dict, no: int, is_practice: bool) -> dict:
            """Run a trial; if the gaze gate times out repeatedly, recalibrate (new model, layout and pitch reference, logged in meta
            and as an event) and run the same trial again."""
            while True:
                try:
                    return run_trial(disp, pipe, cfg, rec, cur["layout"], tr, no, items, cur["ppd"], ctx, is_practice)
                except RecalNeeded:
                    recalibrate("gate", no)
                    rec.event(CLOCK.ms(), "recalibration", str(no))

        def after_trial(row: dict, no: int, next_no: int, more: bool, practice_phase: bool = False) -> None:
            """Automatic drift re-check: cfg.auto_recheck_trials trials in a row with a failed drift check or drift above
            cfg.drift_recheck_deg (practice trials count, look-away trials are skipped) trigger a recalibration before the next
            trial, at most cfg.max_auto_rechecks times per session. Logged as the event "auto_recheck" and in meta."""
            if row.get("lookaway"):
                return
            if not drift_is_bad(cfg, row):
                bad_run.clear()
                return
            bad_run.append((no, float(row.get("drift_deg", float("nan"))), int(row.get("fixation_ok", 0))))
            if len(bad_run) >= cfg.auto_recheck_trials and len(rec.meta["auto_rechecks"]) < cfg.max_auto_rechecks and more:
                info = {"trigger_trials": [b[0] for b in bad_run], "drift_deg": [b[1] for b in bad_run],
                        "fixation_ok": [b[2] for b in bad_run], "last_trial": no, "in_practice": practice_phase}
                rec.event(CLOCK.ms(), "auto_recheck", f"after:{no} drift:" + ",".join(f"{b[1]:.1f}" for b in bad_run))
                rec.meta["auto_rechecks"].append({"before_trial": next_no, "t_ms": CLOCK.ms(), **info})
                recalibrate("auto_recheck", next_no, info)

        try:
            for j, tr in enumerate(practice):                # unscored practice trials with feedback -> practice.csv
                prow = run_one(tr, j, True)
                rec.add_practice(prow)
                after_trial(prow, j, j + 1, True, True)
            if practice:
                disp.clear()
                disp.paragraph([disp.tr("practice.done"), "", disp.tr("posture.reminder")], H // 2 - 60, 40)
                disp.flip()
                disp.wait_key({"space"})
            rec.meta["t_trials_start_ms"] = CLOCK.ms()
            todo, i = list(trials), 0
            while i < len(todo):
                tr = todo[i]
                row = run_one(tr, i, False)
                rec.add_trial(row)
                # a look-away trial is repeated once (the repeat is queued at the end), up to `cap` extra trials in all
                if row["lookaway"] and not tr.get("is_repeat") and n_rep < cap:
                    todo.append({**tr, "is_repeat": True, "repeat_of": i})
                    n_rep += 1
                i += 1
                after_trial(row, i - 1, i, i < len(todo))
            rec.meta["t_trials_end_ms"] = CLOCK.ms()
            # --- end validation (drift / quality after the session)
            disp.clear()
            disp.paragraph([disp.tr("end.dots")], H // 2, 30)
            disp.flip()
            time.sleep(1.0)
            val_end = run_validation(disp, pipe, cfg, cur["model"], cur["ppd"], np.random.default_rng(), dot_log)  # random dot order
            if memory:                                       # story-recognition line-up: after the end validation, before the debrief
                res = _lineup(disp, cfg, rec, mplan["lineup_faces"], items, iset["lineup"]["question"], story_faces)
                mem_meta["lineup"] = {**res, "story_faces": story_faces, **lineup_score(res["selected"], story_faces, mplan["probes"])}
        except KeyboardInterrupt:                    # Esc: keep what was recorded, flag it
            rec.meta["aborted"] = True
            rec.meta["aborted_after_trials"] = len(rec.trials)
            print(f"Session aborted after {len(rec.trials)} of {len(trials)} trials; partial data are saved.")
        except RuntimeError as e:                    # a recalibration that failed in every attempt
            rec.meta["aborted"] = True
            rec.meta["aborted_after_trials"] = len(rec.trials)
            rec.meta["abort_reason"] = str(e)
            print(f"Session stopped after {len(rec.trials)} trials: {e}; partial data are saved.")
        rec.meta.update({"n_planned_trials": len(trials), "n_repeats_queued": n_rep, "repeat_cap": cap,
                         "lookaway_rule": {"min_trial_valid_share": cfg.min_trial_valid_share, "pitch_down_max_deg": cfg.pitch_down_max_deg,
                                           "lookaway_margin_deg": cfg.lookaway_margin_deg, "fixation_gate_deg": cfg.fixation_gate_deg,
                                           "gate_hold_seconds": cfg.gate_hold_seconds}})
        disp.clear()
        disp.paragraph([disp.tr("end.finished"), disp.tr("end.tell")], H // 2 - 30, 36)
        disp.flip()
        time.sleep(1.5)
        frames = pipe.snapshot_frames()
        rec.meta.update({
            "app_version": VERSION, "mode": mode, "screen_px": list(disp.size), "screen_mm": list(cfg.screen_mm),
            "calibration": calinfo["chosen"], "image_quality": calinfo["chosen"]["image_quality"], "calibration_attempts": calinfo["attempts"], "calibration_accepted": calinfo["accepted"],
            "validation_end": val_end, "layout": layout0, "px_per_deg": ppd0, "distance_mm": calinfo["chosen"]["distance_mm"],
            "camera": {**pipe.cam.actual, "index": cfg.camera_index, "achieved_fps": pipe.cam.achieved_fps(),
                       "frames_read": pipe.cam.n_read, "frames_dropped_backlog": pipe.cam.n_dropped,
                       "failed_reads": pipe.cam.n_failed_reads, "frames_processed": len(frames),
                       "focal_px": cfg.focal_length_px, "median_landmark_ms": float(np.nanmedian([f.proc_ms for f in frames]))},
            "flip_interval_ms_median": float(np.median(np.diff(disp.flip_times_ns)) / 1e6) if len(disp.flip_times_ns) > 2 else None,
            "itemset": {"name": iset.get("name"), "sha256": iset["_sha256"], "path": iset["_path"]},
            **({"exam_persons": int(exam)} if memory else {}),
            "probe_id": picked, "probe_chosen_by": picked_by, "repeats": reps, "seed": seed, "config": cfg.to_dict(),
            "n_practice_trials": len(practice),
            "frozen_model_sha256": _model_hash(), "clock": "time.perf_counter_ns; frames stamped when cap.read() returns"})
        rec.save(frames, CLOCK.t0_ns, model.to_dict())
        _debrief(disp, iset)                             # after saving, so nothing is lost if the person closes it early
        return folder
    finally:
        if own_pipe and started:
            pipe.stop()
        if own_disp:
            disp.close()


def _debrief(disp, iset: dict) -> None:
    """Debrief after the data are saved: what the story was and what the study measures (the item set's `debrief` lines in the
    session language, else the generic lines), then SPACE or a click closes it; Esc also closes it."""
    W, H = disp.size
    lines = iset.get("debrief") or [disp.tr(f"debrief.p{i}") for i in range(1, 5)]
    disp.hint = {"phase": "debrief"}
    disp.clear()
    disp.text(disp.tr("debrief.title"), (W // 2, int(70 * disp.ui)), 44)
    disp.paragraph(["• " + ln for ln in lines], int(150 * disp.ui), 30, gap=22, bottom=int(H * 0.86))
    disp.text(disp.tr("debrief.close"), (W // 2, int(H * 0.93)), 32)
    disp.flip()
    try:
        disp.wait_key({"space"})
    except KeyboardInterrupt:
        pass


def _model_hash() -> str | None:
    p = APP_DIR / "model" / "frozen_model.json"
    return json.loads(p.read_text()).get("sha256") if p.exists() else None
