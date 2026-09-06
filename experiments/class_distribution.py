#!/usr/bin/env python
"""SurgVU diagnostic STEP 1 (data-side, CPU-light, no GPU).

Reads the frame-label manifest and prints the per-TOOL / per-TASK class
distribution (train|val|#cases) + skew ratios, flagging the rare classes that
are DATA-FLOOR limited (few cases / zero val support) vs the head that holds
recoverable macro-F1.  Run this BEFORE / alongside every spine run so the
class skew that explains a low macro-mean is always on record.

Usage:
  python class_distribution.py [--manifest .../frame_manifest_full.csv]
Runtime: ~5 s on 1.07 M rows (pure csv, no torch).
"""
import argparse, csv, collections

TOOLS = ["needle_driver", "monopolar_curved_scissor", "force_bipolar", "clip_applier",
         "tip_up_fenestrated_grasper", "cadiere_forceps", "bipolar_forceps", "vessel_sealer",
         "suction_irrigator", "bipolar_dissector", "prograsp_forceps", "stapler",
         "permanent_cautery_hook_spatula", "grasping_retractor"]
TASKS = ["Suturing", "Uterine horn", "Suspensory ligaments", "Rectal artery/vein",
         "Skills application", "Range of motion", "Retraction and collision avoidance", "Other"]

# data-floor thresholds: a class with < MIN_CASES distinct cases OR 0 val frames
# cannot be fixed by loss/sampling — it needs MORE DATA (escalate to user).
MIN_CASES = 20
RARE_VAL = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="/users/sc20osc/SurgVU/data/frame_manifest_full.csv")
    args = ap.parse_args()

    tool_tr = collections.Counter(); tool_va = collections.Counter()
    task_tr = collections.Counter(); task_va = collections.Counter()
    cases_with_tool = collections.defaultdict(set)
    n = 0
    with open(args.manifest, newline="") as fh:
        for r in csv.DictReader(fh):
            n += 1; sp = r["split"]; mh = r["tools_multihot"]
            for j, c in enumerate(mh):
                if c == "1":
                    (tool_tr if sp == "train" else tool_va)[TOOLS[j]] += 1
                    cases_with_tool[TOOLS[j]].add(r["case"])
            t = TASKS[int(r["task_idx"])]
            (task_tr if sp == "train" else task_va)[t] += 1

    print(f"# manifest={args.manifest}\n# total frames={n}\n")
    print("=== TOOL distribution (sorted by train freq) ===")
    print(f"{'tool':34s} {'train':>9s} {'val':>8s} {'#cases':>7s}  flag")
    rare_tools, floor_tools = [], []
    for t, _ in sorted(tool_tr.items(), key=lambda x: -x[1]):
        nc = len(cases_with_tool[t]); va = tool_va.get(t, 0)
        flag = ""
        if va == 0 or nc < MIN_CASES:
            flag = "DATA-FLOOR (escalate: needs more cases)"; floor_tools.append(t)
        elif va < RARE_VAL:
            flag = "rare-tail (class-balanced loss)"; rare_tools.append(t)
        print(f"{t:34s} {tool_tr[t]:9d} {va:8d} {nc:7d}  {flag}")
    print("\n=== TASK distribution (sorted by train freq) ===")
    print(f"{'task':34s} {'train':>9s} {'val':>8s}  flag")
    for t, _ in sorted(task_tr.items(), key=lambda x: -x[1]):
        va = task_va.get(t, 0)
        flag = "rare-task (balanced sampling)" if task_tr[t] < 30000 else ""
        print(f"{t:34s} {task_tr[t]:9d} {va:8d}  {flag}")

    mx, mn = max(tool_tr.values()), min(tool_tr.values())
    mxk, mnk = max(task_tr.values()), min(task_tr.values())
    print(f"\nTOOL skew: {mx}/{mn} = {mx/mn:.0f}x   TASK skew: {mxk}/{mnk} = {mxk/mnk:.0f}x")
    print(f"DATA-FLOOR tools (loss CANNOT fix -> need data): {floor_tools}")
    print(f"RARE-TAIL tools (recoverable via balanced loss/sampling): {rare_tools}")


if __name__ == "__main__":
    main()
