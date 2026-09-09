"""
Auto-diagnosis for the presence-constrained pseudo-labelling experiment.
Applies the PRE-REGISTERED criteria; it does not choose them after the fact.

PRE-REGISTERED 2026-08-12, BEFORE ANY ARM RAN
=============================================
GATE 0 (label-free, zero GPU, in cat1_ssl_propose.py -> GATE.json)
  FILTER_ok : agreement@0.25 >= 0.50 AND >= 1.5x the shuffled-presence control
  ASSIGN_ok : class-agnostic firing on never-seen-class singleton frames >= 0.30
  If neither holds, no training runs at all (the train script self-aborts).

GATE 1 -- MECHANISM, on the main split (the sharpest available instrument)
  clip_applier and vessel_sealer have ZERO train boxes on the main split, so their
  AP is 0.000 in ALL THREE baseline seeds BY CONSTRUCTION -- the seed variance of
  a structurally-zero quantity is itself zero, so no Delta_noise argument can
  explain away a non-zero value.
    CONFIRMED  if mean over 3 seeds of AP(clip_applier) >= 0.05 for ASSIGN or
               COMBINED, AND the same arm beats NEG_SHUFFLE by >= 2x on that AP.
               (vessel_sealer has only 13 val boxes -> reported, never gating.)
    REFUTED    if AP(clip_applier) stays < 0.05, or if NEG_SHUFFLE matches it --
               the latter meaning the boxes, not the names, did the work.

GATE 2 -- HEADLINE, on the 7-fold LOVO protocol, PAIRED per fold, on last.pt
  Baseline = LOVO seed 0/1/2 (jobs 7050543 + 7110751). Primary = unselected
  last.pt; best.pt reported alongside and never gating.
    CONFIRMED  if paired mean gain over the 7 folds > 2 * SE_paired, where
               SE_paired = sqrt(sum_f 2*s_f^2/3)/7 from the per-fold seed SDs s_f,
               AND >= 6 of 7 folds move in the same direction.
    REFUTED    otherwise.
  The main split's Delta_noise = 0.0365 is NOT reused: different estimator
  (best.pt max-of-60), different protocol, different ceiling.

NEGATIVE CONTROLS -- the mechanism claim dies if either fires
  NEG_UNCONSTRAINED ~= FILTER  => presence is decoration; plain self-training did it.
  NEG_SHUFFLE       ~= ASSIGN  => the extra boxes did it, not the correct names.
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

LOG_SWEEP = "/scratch/USERNAME/miccai-2026/SurgVU/logs/cat1_sweep"
LOG_SSL = "/scratch/USERNAME/miccai-2026/SurgVU/logs/cat1_ssl"
SSL = "/scratch/USERNAME/miccai-2026/SurgVU/data/cat1_ssl"


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def lovo_noise():
    """per-fold seed SD on last.pt + the pre-registered paired SE."""
    per = defaultdict(dict)
    for k in range(1, 8):
        for s, tag in [(0, f"LOVO_v{k}"), (1, f"LOVO_v{k}_s1"), (2, f"LOVO_v{k}_s2")]:
            d = load(f"{LOG_SWEEP}/{tag}.json")
            if d and "last" in d:
                per[k][s] = (d["last"]["map"], d.get("best", {}).get("map", float("nan")))
    print("\n=== LOVO baseline, 3 seeds (last.pt / best.pt) ===")
    print(f"{'fold':>5s} " + " ".join(f"{'seed'+str(s):>15s}" for s in (0, 1, 2)) +
          f"{'mean_last':>11s}{'sd_last':>9s}")
    sds, means = [], []
    for k in range(1, 8):
        row = per[k]
        cells = " ".join(f"{row[s][0]:.4f}/{row[s][1]:.4f}".rjust(15) if s in row
                         else " " * 15 for s in (0, 1, 2))
        v = [row[s][0] for s in sorted(row)]
        m = float(np.mean(v)) if v else float("nan")
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
        means.append(m)
        if len(v) > 1:
            sds.append(sd)
        print(f"{k:>5d} {cells}{m:11.4f}{sd:9.4f}")
    if sds:
        se = float(np.sqrt(sum(2 * s ** 2 / 3 for s in sds)) / 7)
        print(f"\nmean within-fold seed SD (last.pt) = {np.mean(sds):.4f}")
        print(f"PAIRED SE of the 7-fold mean       = {se:.4f}")
        print(f"PRE-REGISTERED BAR: paired mean gain > {2*se:.4f} AND >=6/7 folds same sign")
        print(f"baseline 7-fold mean last.pt       = {np.nanmean(means):.4f}")
        return se, means
    print("(seed replication not finished yet -- bar not computable)")
    return None, means


def mechanism():
    base = [load(f"{LOG_SWEEP}/S0_seed{s}.json") for s in range(3)]
    base = [b for b in base if b]
    arms = defaultdict(list)
    for f in sorted(glob.glob(f"{LOG_SSL}/*_main_*.json")):
        d = load(f)
        if d:
            arms[d["arm"]].append(d)
    if not arms:
        print("\n(no SSL arms finished yet)")
        return
    print("\n=== SUPPORT: train boxes before -> after (arm COMBINED, seed 0) ===")
    ex = next((d for d in arms.get("COMBINED", []) if d["seed"] == 0), None)
    if ex:
        print(f"{'class':34s} {'before':>8s} {'added':>8s} {'after':>8s}")
        for c, s in ex["support"].items():
            if s["before"] or s["added"]:
                flag = "  <-- WAS STRUCTURALLY ZERO" if s["before"] == 0 and s["added"] else ""
                print(f"{c:34s} {s['before']:8d} {s['added']:8d} {s['after']:8d}{flag}")

    print("\n=== MAIN SPLIT, 3-seed means (last.pt primary) ===")
    hdr = f"{'arm':20s}{'mAP_last':>10s}{'mAP_best':>10s}"
    for c in ("clip_applier", "vessel_sealer", "bipolar_forceps", "monopolar_curved_scissor"):
        hdr += f"{c[:14]:>16s}"
    print(hdr)

    def line(name, ds):
        if not ds:
            return None
        ml = np.mean([d["last"]["map"] for d in ds if "last" in d])
        mb = np.mean([d["best"]["map"] for d in ds if "best" in d])
        out = f"{name:20s}{ml:10.4f}{mb:10.4f}"
        aps = {}
        for c in ("clip_applier", "vessel_sealer", "bipolar_forceps", "monopolar_curved_scissor"):
            v = [d["last"]["per_class_ap"].get(c, 0.0) for d in ds if "last" in d]
            aps[c] = float(np.mean(v)) if v else 0.0
            out += f"{aps[c]:16.4f}"
        print(out)
        return aps

    b = line("BASELINE (S0)", base)
    res = {a: line(a, arms[a]) for a in
           ("FILTER", "ASSIGN", "COMBINED", "NEG_UNCONSTRAINED", "NEG_SHUFFLE") if a in arms}

    print("\n=== GATE 1 (mechanism) ===")
    shuf = res.get("NEG_SHUFFLE", {}).get("clip_applier", 0.0)
    for a in ("ASSIGN", "COMBINED"):
        if a not in res:
            continue
        v = res[a]["clip_applier"]
        ok = v >= 0.05 and v >= 2 * max(shuf, 1e-9)
        print(f"  {a:10s} clip_applier AP={v:.4f} (baseline 0.0000 by construction, "
              f"NEG_SHUFFLE {shuf:.4f}) -> {'CONFIRMED' if ok else 'REFUTED'}")
    if "FILTER" in res and "NEG_UNCONSTRAINED" in res:
        f_, n_ = res["FILTER"], res["NEG_UNCONSTRAINED"]
        print(f"  control: FILTER vs NEG_UNCONSTRAINED on known classes "
              f"(bipolar {f_['bipolar_forceps']:.4f} vs {n_['bipolar_forceps']:.4f}) "
              f"-- if equal, presence is decoration")


def main():
    g = load(f"{SSL}/GATE.json")
    if g:
        print("=== GATE 0 (label-free) ===")
        print(json.dumps(g, indent=2))
    lk = load(f"{SSL}/leak_report.json")
    if lk:
        print("\n=== LEAK / VIDEO-DISJOINTNESS ===")
        print(f"cases scanned={lk['cases_scanned']} frames={lk['frames_sampled']} "
              f"near-dup hits={lk['near_duplicate_hits']}")
        print(f"BANNED cases ({lk['n_banned']}): {lk['BANNED_cases']}")
        print(lk["VERDICT"])
    lovo_noise()
    mechanism()
    return 0


if __name__ == "__main__":
    sys.exit(main())
