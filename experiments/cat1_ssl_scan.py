"""
STAGE 1 — scan the 280-part / 754 GB corpus, do the LEAKAGE TEST, and emit
candidate frames for presence-constrained box proposal.

TWO JOBS IN ONE SEQUENTIAL DECODE PASS (measured at 142x realtime, so the whole
840 h corpus costs ~6 core-hours = ~13 min across a 28-way array):

  (A) VIDEO-DISJOINTNESS, MEASURED RATHER THAN ASSUMED.
      The 7 annotated Cat-1 videos are excerpts from the SAME acquisition system
      as the training corpus (identical da Vinci HUD, identical pillarbox).
      Whether they are excerpts OF the 155 training cases is an OPEN QUESTION
      that decides whether this entire lever is a gain or a leak, so we MEASURE
      it: every sampled corpus frame is dHash'd in the annotated geometry and
      compared against all 5,178 annotated frames.
        * frame-level: any candidate within LEAK_HAMMING of an annotated frame is
          dropped on the spot and never written.
        * case-level:  the per-part hit counts go to the reducer, which BANS the
          whole case (both parts) if it contains annotated content.
      Frame-level alone would not be enough -- a different frame of the same
      surgery is still the val video's content.

  (B) CANDIDATE SELECTION, driven by frame-exact presence from tools.csv.
      SINGLETON frames (exactly one in-vocabulary tool installed, no OOV tool)
      are the valuable ones: they are the only frames where a class-agnostic
      detection can be NAMED, which is the only mechanism that can create boxes
      for the 6 classes that have ZERO boxes anywhere in the annotated set
      (stapler, permanent_cautery_hook_spatula, tip_up_fenestrated_grasper,
      bipolar_dissector, prograsp_forceps, suction_irrigator).
      MULTI frames (>=1 installed) are kept at a lower rate for the FILTER rule.

Two-pass per part: pass 1 grab-skips the whole part (hash + candidate index),
pass 2 seeks only to the frames actually selected (~0.065 s each) and writes them.
Inode budget: hard per-part cap, so the pool is bounded by construction.
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cat1_ssl_common import (ANN_POOL, CORPUS, OOV, SSL, crop_resize, dhash,
                             hamming_to_set, installed_at, load_intervals, part_key)

STRIDE_S = 2.0          # sample the timeline every 2 s
# CALIBRATED, not guessed (2026-08-12, 256-bit dHash, 72 probe pairs):
#   NULL  = unrelated corpus frame -> min distance to the 5,178 annotated frames:
#           min 70, p1 70, median 86.
#   POS   = the same shot 0.5 s apart (the worst-case offset between our 2 s corpus
#           grid and the annotated 1 s grid): median 14.
# 48 sits in the empty band between them: no unrelated frame in the probe came
# within 22 bits of it, and the bulk of genuine near-duplicates sit 34 bits below.
LEAK_HAMMING = 48
CAP_PART_TOTAL = 340    # hard inode cap per video part
CAP_SINGLE_CLS = 110    # per class, per part
CAP_MULTI = 90          # non-singleton frames per part

# Classes with ZERO boxes anywhere in the annotated pool -> the only route to a
# box for them is the singleton-assign rule, so they get sampling priority.
ZERO_BOX_CLASSES = ['stapler', 'permanent_cautery_hook_spatula', 'prograsp_forceps',
                    'suction_irrigator', 'tip_up_fenestrated_grasper', 'bipolar_dissector']
# Zero TRAIN support on the main split (present only in val videos 6-7).
ZERO_SUPPORT_MAIN = ['clip_applier', 'vessel_sealer']
PRIORITY = {c: 0 for c in ZERO_BOX_CLASSES}
PRIORITY.update({c: 1 for c in ZERO_SUPPORT_MAIN})


def annotated_hashes():
    """dHash of all 5,178 annotated frames (they are already in the 640x512
    target geometry, so no crop is applied here)."""
    cache = f"{SSL}/ann_hashes.npy"
    if os.path.exists(cache):
        return np.load(cache)
    hs = []
    for p in sorted(glob.glob(f"{ANN_POOL}/images/*.jpg")):
        im = cv2.imread(p)
        if im is not None:
            hs.append(dhash(im))
    a = np.stack(hs).astype(np.uint8)      # (N, 32) packed 256-bit hashes
    os.makedirs(SSL, exist_ok=True)
    np.save(cache, a)
    return a


def scan_part(path, ivs, ann_h, imdir, rng):
    case, part = part_key(path)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or nfr <= 0:
        cap.release()
        return None
    step = max(1, int(round(STRIDE_S * fps)))

    # ---- pass 1: sequential grab-skip; hash everything, index candidates
    hashes, times, leak_hits = [], [], 0
    cand = []                      # (t, singleton_class or None, installed tuple)
    i = 0
    while i < nfr:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        h = dhash(crop_resize(frame))
        d = hamming_to_set(h, ann_h)
        hashes.append(h)
        times.append(t)
        if d <= LEAK_HAMMING:
            leak_hits += 1
        else:
            inst, oov = installed_at(ivs, t)
            if inst:
                single = next(iter(inst)) if (len(inst) == 1 and not oov) else None
                cand.append((t, single, tuple(sorted(inst))))
        # skip forward
        for _ in range(step - 1):
            if not cap.grab():
                i = nfr
                break
        i += step

    # ---- select, respecting caps and prioritising the starved classes
    by_single = defaultdict(list)
    multi = []
    for c in cand:
        (by_single[c[1]] if c[1] else multi).append(c)
    chosen, per_cls = [], Counter()
    for cls in sorted(by_single, key=lambda c: (PRIORITY.get(c, 2), c)):
        lst = by_single[cls]
        rng.shuffle(lst)
        take = lst[:min(CAP_SINGLE_CLS, CAP_PART_TOTAL - len(chosen))]
        chosen += take
        per_cls[cls] += len(take)
        if len(chosen) >= CAP_PART_TOTAL:
            break
    rng.shuffle(multi)
    chosen += multi[:max(0, min(CAP_MULTI, CAP_PART_TOTAL - len(chosen)))]
    chosen.sort(key=lambda c: c[0])

    # ---- pass 2: seek only to the selected timestamps and write them
    rows, written = [], 0
    for t, single, inst in chosen:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        img = crop_resize(frame)
        if hamming_to_set(dhash(img), ann_h) <= LEAK_HAMMING:   # belt and braces
            leak_hits += 1
            continue
        stem = f"{case}_p{part}_{int(round(t * 1000)):09d}"
        cv2.imwrite(f"{imdir}/{stem}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({"stem": stem, "case": case, "part": part, "t": f"{t:.3f}",
                     "singleton": single or "", "installed": "|".join(inst)})
        written += 1
    cap.release()
    return {"case": case, "part": part, "n_sampled": len(hashes), "leak_hits": leak_hits,
            "n_candidates": len(cand), "n_written": written,
            "per_class": dict(per_cls), "rows": rows,
            "hashes": (np.stack(hashes).astype(np.uint8) if hashes
                       else np.zeros((0, 32), np.uint8)),
            "times": np.array(times, dtype=np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    args = ap.parse_args()

    imdir, mdir, hdir = f"{SSL}/images", f"{SSL}/manifest", f"{SSL}/hashes"
    for d in (imdir, mdir, hdir):
        os.makedirs(d, exist_ok=True)

    parts = sorted(glob.glob(f"{CORPUS}/case_*/*.mp4"))
    mine = parts[args.shard::args.nshards]
    print(f"[shard {args.shard}/{args.nshards}] {len(mine)} parts", flush=True)

    durs = {}
    for p in parts:
        cap = cv2.VideoCapture(p)
        n, f = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        durs[part_key(p)] = (n / f) if f > 0 else 0.0
    ivstats = {}
    ivs_all = load_intervals(durs, stats=ivstats)
    print(f'[shard {args.shard}] interval defects: {ivstats}', flush=True)
    ann_h = annotated_hashes()
    print(f"[shard {args.shard}] annotated hashes: {ann_h.shape[0]}", flush=True)

    rng = random.Random(1234 + args.shard)
    summary = []
    fields = ["stem", "case", "part", "t", "singleton", "installed"]
    with open(f"{mdir}/shard{args.shard:03d}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for p in mine:
            k = part_key(p)
            r = scan_part(p, ivs_all.get(k, []), ann_h, imdir, rng)
            if r is None:
                print(f"  !! unreadable {p}", flush=True)
                continue
            for row in r["rows"]:
                w.writerow(row)
            fh.flush()
            np.savez_compressed(f"{hdir}/{k[0]}_p{k[1]}.npz",
                                hashes=r["hashes"], times=r["times"])
            summary.append({kk: r[kk] for kk in
                            ("case", "part", "n_sampled", "leak_hits", "n_candidates",
                             "n_written", "per_class")})
            print(f"  {k[0]} p{k[1]}: sampled={r['n_sampled']} LEAK={r['leak_hits']} "
                  f"cand={r['n_candidates']} written={r['n_written']} {r['per_class']}",
                  flush=True)

    with open(f"{mdir}/shard{args.shard:03d}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    tot_leak = sum(s["leak_hits"] for s in summary)
    print(f"[shard {args.shard}] DONE parts={len(summary)} "
          f"written={sum(s['n_written'] for s in summary)} LEAK_HITS={tot_leak}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
