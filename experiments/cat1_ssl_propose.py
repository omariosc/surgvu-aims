"""
STAGE 3 — run the proposers over the candidate frames and MEASURE, label-free,
whether presence-constrained pseudo-labelling can possibly work.  Writes GATE.json.

This stage spends ~20 GPU-min and can kill the whole lever before a single
training run is launched.  That is the point of it.

TWO PROPOSER-SIDE LEAKAGE RULES, both enforced here
---------------------------------------------------
1. PER-SPLIT PROPOSERS.  Pseudo-labels carry whatever the proposer knows.  If one
   proposer trained on all 7 annotated videos generated the labels, then every
   LOVO fold would train on labels that encode its own val video.  So fold K uses
   the LOVO_vK model (trained without video K) and the main split uses the
   main-split model (trained on videos 1-5 only).  All 8 already exist.
2. `last.pt`, NEVER `best.pt`.  `best.pt` is chosen by val mAP on the very videos
   the fold holds out — the checkpoint CHOICE is a channel from val into the
   pseudo-labels even though the training never saw those frames.  `last.pt` is
   unselected and closes it.

THREE MEASUREMENTS
------------------
A. AGREEMENT.  On singleton frames whose class the proposer already knows, what
   fraction of confident boxes are predicted as the installed class?  Compared
   against a SHUFFLED-PRESENCE control computed on the same boxes.  If the two
   match, presence is uninformative and the whole story is wrong -- this catches
   it for free, before any GPU is spent on training.
B. FIRING.  On singleton frames whose class the proposer has NEVER seen a box
   for, does it emit any box at all (class-agnostically)?  This is the ceiling on
   the only mechanism that can create the never-boxed classes.  A detector cannot
   invent a class it has never been trained on, so ASSIGN works only to the extent
   that a class-agnostic "this is an instrument" response transfers.
C. YIELD.  Surviving pseudo-boxes per class per confidence threshold under each
   rule, which is what actually determines whether train support moves.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cat1_ssl_common import ANN_POOL, CLASSES, CLS_IDX, SSL

MODELS = "/scratch/sc20osc/miccai-2026/SurgVU/models/cat1_sweep"
PROPOSERS = {"main": "S0_seed0", **{f"lovo{k}": f"LOVO_v{k}" for k in range(1, 8)}}
SPLIT_OF = {"main": "main", **{f"lovo{k}": f"lovo{k}" for k in range(1, 8)}}
TAUS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
TAU_REF = 0.25

# pre-registered gate thresholds
GATE_AGREE_ABS = 0.50     # confident boxes on singleton frames must mostly BE the installed class
GATE_AGREE_RATIO = 1.5    # ... and must beat the shuffled-presence control by this factor
GATE_FIRE = 0.30          # >=30% of never-seen-class singleton frames must yield a box


def load_candidates():
    import csv
    rows = list(csv.DictReader(open(f"{SSL}/candidates.csv")))
    paths = [f"{SSL}/images/{r['stem']}.jpg" for r in rows]
    return rows, paths


def run_proposer(weights, paths, batch=64):
    from ultralytics import YOLO
    model = YOLO(weights)
    idx, xyxy, conf, cls = [], [], [], []
    for s in range(0, len(paths), batch):
        chunk = paths[s:s + batch]
        for j, r in enumerate(model.predict(chunk, imgsz=640, conf=0.03, iou=0.7,
                                            max_det=20, device=0, verbose=False)):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            n = len(b)
            idx += [s + j] * n
            xyxy.append(b.xyxy.cpu().numpy())
            conf.append(b.conf.cpu().numpy())
            cls.append(b.cls.cpu().numpy().astype(np.int16))
    if not idx:
        return (np.zeros(0, np.int32), np.zeros((0, 4), np.float32),
                np.zeros(0, np.float32), np.zeros(0, np.int16))
    return (np.array(idx, np.int32), np.concatenate(xyxy).astype(np.float32),
            np.concatenate(conf).astype(np.float32), np.concatenate(cls))


def zero_support_classes(split):
    sup = json.load(open(f"{ANN_POOL}/class_support.json"))
    tr = sup["splits"][split]["train_counts"]
    return [c for c in CLASSES if tr.get(c, 0) == 0]


def diagnose(rows, det, split, tag, rng):
    idx, xyxy, conf, cls = det
    unseen = set(zero_support_classes(split))
    by_img = defaultdict(list)
    for k in range(len(idx)):
        by_img[int(idx[k])].append((float(conf[k]), int(cls[k]), k))

    sing = [(i, r["singleton"]) for i, r in enumerate(rows) if r["singleton"]]
    known_sing = [(i, c) for i, c in sing if c not in unseen]
    unseen_sing = [(i, c) for i, c in sing if c in unseen]

    # ---- A. agreement vs shuffled-presence control (on known-class singletons)
    shuf = [c for _, c in known_sing]
    rng.shuffle(shuf)
    agree = {}
    for tau in TAUS:
        hit = tot = shit = 0
        for (i, c), sc in zip(known_sing, shuf):
            for cf, cl, _ in by_img.get(i, []):
                if cf < tau:
                    continue
                tot += 1
                hit += (CLASSES[cl] == c)
                shit += (CLASSES[cl] == sc)
        agree[f"{tau:.2f}"] = {"boxes": tot, "observed": hit / tot if tot else 0.0,
                               "shuffled": shit / tot if tot else 0.0}

    # ---- B. class-agnostic firing on never-seen-class singleton frames
    fire = {}
    for tau in TAUS:
        per = Counter()
        den = Counter()
        for i, c in unseen_sing:
            den[c] += 1
            if any(cf >= tau for cf, _, _ in by_img.get(i, [])):
                per[c] += 1
        fire[f"{tau:.2f}"] = {
            "overall": (sum(per.values()) / sum(den.values())) if den else 0.0,
            "per_class": {c: (per[c] / den[c]) for c in sorted(den)},
            "frames_per_class": dict(den)}

    # ---- C. yield per rule
    yld = {}
    for tau in TAUS:
        f_cnt, a_cnt, n_cnt = Counter(), Counter(), Counter()
        for i, r in enumerate(rows):
            inst = set(r["installed"].split("|")) if r["installed"] else set()
            for cf, cl, _ in by_img.get(i, []):
                if cf < tau:
                    continue
                name = CLASSES[cl]
                n_cnt[name] += 1                       # NEG: unconstrained self-training
                if name in inst:
                    f_cnt[name] += 1                   # FILTER: keep iff class is installed
            if r["singleton"]:
                for cf, cl, _ in by_img.get(i, []):
                    if cf >= tau:
                        a_cnt[r["singleton"]] += 1     # ASSIGN: relabel to the one installed tool
        yld[f"{tau:.2f}"] = {"FILTER": dict(f_cnt), "ASSIGN": dict(a_cnt), "NEG": dict(n_cnt)}

    return {"proposer": tag, "split": split,
            "zero_support_classes_for_this_split": sorted(unseen),
            "n_singleton_frames": len(sing), "n_known_singleton": len(known_sing),
            "n_unseen_singleton": len(unseen_sing),
            "agreement": agree, "firing": fire, "yield": yld}


def montage(rows, det, split, tag, out, n_per=8):
    """Draw the top class-agnostic box on singleton frames of never-seen classes.
    Numbers cannot tell us whether a box is ON the instrument; this can."""
    idx, xyxy, conf, cls = det
    unseen = set(zero_support_classes(split))
    best = defaultdict(list)
    for k in range(len(idx)):
        r = rows[int(idx[k])]
        if r["singleton"] in unseen and conf[k] >= TAU_REF:
            best[r["singleton"]].append((float(conf[k]), int(idx[k]), k))
    os.makedirs(out, exist_ok=True)
    for c, lst in best.items():
        lst.sort(reverse=True)
        tiles = []
        for _, i, k in lst[:n_per]:
            im = cv2.imread(f"{SSL}/images/{rows[i]['stem']}.jpg")
            if im is None:
                continue
            x1, y1, x2, y2 = xyxy[k].astype(int)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im, f"{c} p={conf[k]:.2f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            tiles.append(cv2.resize(im, (320, 256)))
        if tiles:
            while len(tiles) % 4:
                tiles.append(np.zeros_like(tiles[0]))
            grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
            cv2.imwrite(f"{out}/{tag}_{c}.jpg", grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposers", default="main",
                    help="comma list from: main,lovo1..lovo7,all")
    args = ap.parse_args()
    names = list(PROPOSERS) if args.proposers == "all" else args.proposers.split(",")

    rows, paths = load_candidates()
    print(f"candidates: {len(paths)}", flush=True)
    os.makedirs(f"{SSL}/det", exist_ok=True)
    diags = {}
    rng = __import__("random").Random(0)

    for nm in names:
        w = f"{MODELS}/{PROPOSERS[nm]}/weights/last.pt"     # unselected, see header
        if not os.path.exists(w):
            print(f"!! missing proposer {nm}: {w}", flush=True)
            continue
        print(f"\n=== proposer {nm} ({PROPOSERS[nm]}/last.pt) ===", flush=True)
        det = run_proposer(w, paths)
        np.savez_compressed(f"{SSL}/det/{nm}.npz", idx=det[0], xyxy=det[1],
                            conf=det[2], cls=det[3])
        print(f"  boxes @conf>=0.03: {len(det[0])}", flush=True)
        d = diagnose(rows, det, SPLIT_OF[nm], nm, rng)
        diags[nm] = d
        a = d["agreement"][f"{TAU_REF:.2f}"]
        f = d["firing"][f"{TAU_REF:.2f}"]
        print(f"  A. agreement@{TAU_REF}: observed={a['observed']:.3f} "
              f"shuffled={a['shuffled']:.3f} over {a['boxes']} boxes")
        print(f"  B. firing@{TAU_REF} on never-seen-class singletons: {f['overall']:.3f} "
              f"({d['n_unseen_singleton']} frames)")
        for c, v in sorted(f["per_class"].items()):
            print(f"       {c:34s} {v:.3f}  (n={f['frames_per_class'][c]})")
        montage(rows, det, SPLIT_OF[nm], nm, f"{SSL}/montage")

    json.dump(diags, open(f"{SSL}/diagnostics.json", "w"), indent=2)

    # ---------------------------------------------------------------- the GATE
    m = diags.get("main")
    gate = {"evaluated_on": "main-split proposer", "tau_ref": TAU_REF}
    if m:
        a = m["agreement"][f"{TAU_REF:.2f}"]
        f = m["firing"][f"{TAU_REF:.2f}"]
        gate["agreement_observed"] = a["observed"]
        gate["agreement_shuffled"] = a["shuffled"]
        gate["firing_unseen"] = f["overall"]
        gate["FILTER_ok"] = bool(a["observed"] >= GATE_AGREE_ABS and
                                 a["observed"] >= GATE_AGREE_RATIO * max(a["shuffled"], 1e-9))
        gate["ASSIGN_ok"] = bool(f["overall"] >= GATE_FIRE)
        gate["PASS"] = bool(gate["FILTER_ok"] or gate["ASSIGN_ok"])
        gate["thresholds"] = {"agree_abs": GATE_AGREE_ABS, "agree_ratio": GATE_AGREE_RATIO,
                              "fire": GATE_FIRE}
    else:
        gate["PASS"] = False
        gate["reason"] = "main proposer not run"
    json.dump(gate, open(f"{SSL}/GATE.json", "w"), indent=2)
    print("\n=== GATE ===")
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
