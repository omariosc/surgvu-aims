"""
STAGE 2 — reduce the sharded scan into (a) the LEAK VERDICT and (b) a bounded,
class-balanced candidate set.

(a) THE LEAK VERDICT IS THE GATE ON EVERYTHING ELSE.
    Stage 1 hashed every sampled corpus frame against all 5,178 annotated frames.
    Here we roll those hits up to the CASE and ban any case that contains
    annotated content.  A ban is at case level, not part level and not frame
    level, because the unit of independence is the surgery: two parts of one case
    are the same patient, same anatomy, same session.  Pseudo-labels drawn from
    the same surgery as a val video would leak that video's content into training
    no matter how different the individual frames look.

    BAN_MIN_HITS guards against the hash's own false positives.  A genuine
    overlap is 5-19 contiguous minutes of the annotated video, which our 2 s grid
    samples 150-570 times, so a real overlap produces hits in bulk; an isolated
    hit is a collision.

(b) CANDIDATE SELECTION is capped per class so that one enormous case cannot
    dominate a class, and so the pool stays inode-bounded.  The training budget
    is applied later (in the gating step), on top of this.
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cat1_ssl_common import CLASSES, SSL

BAN_MIN_HITS = 3          # >=3 near-duplicate hits in a case => that case IS a val video
PER_CLASS_CAP = 6000      # singleton frames kept per class (training budget applied later)
PER_CASE_CLASS_CAP = 400  # stop one long case dominating a class
MULTI_CAP = 12000         # non-singleton frames kept in total (for the FILTER rule)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    mdir = f"{SSL}/manifest"

    # ---------------------------------------------------------------- leak roll-up
    per_case = defaultdict(lambda: {"sampled": 0, "hits": 0, "parts": []})
    for f in sorted(glob.glob(f"{mdir}/shard*_summary.json")):
        for s in json.load(open(f)):
            c = per_case[s["case"]]
            c["sampled"] += s["n_sampled"]
            c["hits"] += s["leak_hits"]
            c["parts"].append(s["part"])
    banned = sorted([c for c, v in per_case.items() if v["hits"] >= BAN_MIN_HITS])
    suspicious = sorted([c for c, v in per_case.items() if 0 < v["hits"] < BAN_MIN_HITS])

    total_sampled = sum(v["sampled"] for v in per_case.values())
    total_hits = sum(v["hits"] for v in per_case.values())
    leak = {
        "cases_scanned": len(per_case), "frames_sampled": total_sampled,
        "near_duplicate_hits": total_hits,
        "BANNED_cases": banned, "n_banned": len(banned),
        "suspicious_cases_below_threshold": suspicious,
        "ban_min_hits": BAN_MIN_HITS,
        "hits_by_banned_case": {c: per_case[c]["hits"] for c in banned},
        "VERDICT": ("OVERLAP FOUND — the annotated Cat-1 videos are excerpts of the "
                    "training corpus; those cases are excluded from pseudo-labelling"
                    if banned else
                    "NO OVERLAP — no training case contains annotated Cat-1 content"),
    }
    json.dump(leak, open(f"{SSL}/leak_report.json", "w"), indent=2)
    print("=== LEAK REPORT ===")
    print(json.dumps({k: v for k, v in leak.items()
                      if k != "suspicious_cases_below_threshold"}, indent=2))

    # ---------------------------------------------------------------- candidates
    rows = []
    for f in sorted(glob.glob(f"{mdir}/shard*.csv")):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    print(f"\nmanifest rows: {len(rows)}")
    rows = [r for r in rows if r["case"] not in banned]
    print(f"after case ban:  {len(rows)}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    by_cls = defaultdict(list)
    multi = []
    for r in rows:
        (by_cls[r["singleton"]] if r["singleton"] else multi).append(r)

    kept, per_cls_case = [], Counter()
    cls_tot = Counter()
    for cls in CLASSES:
        for r in by_cls.get(cls, []):
            if cls_tot[cls] >= PER_CLASS_CAP:
                break
            if per_cls_case[(cls, r["case"])] >= PER_CASE_CLASS_CAP:
                continue
            kept.append(r)
            cls_tot[cls] += 1
            per_cls_case[(cls, r["case"])] += 1
    kept += multi[:MULTI_CAP]

    with open(f"{SSL}/candidates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stem", "case", "part", "t", "singleton", "installed"])
        w.writeheader()
        for r in kept:
            w.writerow(r)
    with open(f"{SSL}/candidates.txt", "w") as fh:
        for r in kept:
            fh.write(f"{SSL}/images/{r['stem']}.jpg\n")

    print("\n=== CANDIDATE SET ===")
    print(f"{'class':34s} {'singleton frames':>17s} {'cases':>7s}")
    for c in CLASSES:
        ncase = len({r["case"] for r in kept if r["singleton"] == c})
        print(f"{c:34s} {cls_tot[c]:17d} {ncase:7d}")
    print(f"{'(multi / FILTER-only frames)':34s} {min(len(multi), MULTI_CAP):17d}")
    print(f"TOTAL candidates: {len(kept)}")
    json.dump({"per_class_singleton": dict(cls_tot),
               "multi": min(len(multi), MULTI_CAP), "total": len(kept),
               "banned_cases": banned},
              open(f"{SSL}/candidate_summary.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
