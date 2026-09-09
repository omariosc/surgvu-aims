#!/usr/bin/env python
"""SurgVU diagnostic STEPS 1-3 (model-side) — auto-diagnoses any spine run.

Reloads a trained spine checkpoint (default = the FULL-SCALE job 5998329 out-dir
models/spine_full/spine_best.pt) and re-runs the val split to emit what the
trainer's aggregate macro-F1 hides:
  STEP 1 DECOMPOSE  : per-TOOL F1/precision/recall + support (sorted worst->best),
                      per-TASK F1 + support, per-CASE task accuracy (centre/proc shift).
  STEP 2 FAILURE    : tool confusion pairs (co-predicted / swapped), the task
                      confusion matrix (which task -> which), worst per-class.
  STEP 3 QUANTIFY   : splits each weak class into RECOVERABLE (has val support but
                      low F1 = method gap) vs DATA-FLOOR (support < floor = needs data),
                      and reports the macro-F1 *ceiling* recoverable by fixing the
                      non-floor classes (mean over classes with adequate support).

Outputs a JSON next to the checkpoint (per_class_diag.json) + a printed table.
Re-run on EVERY spine checkpoint; reads 5998329's spine_best.pt directly so the
full-scale macro-F1 auto-populates the playbook the moment it lands.

Usage (SLURM gpu node — decode + forward pass needs a GPU):
  python diagnose_spine.py \
      --ckpt /scratch/USERNAME/miccai-2026/SurgVU/models/spine_full/spine_best.pt \
      --manifest /scratch/USERNAME/miccai-2026/SurgVU/data/frame_manifest_full.csv \
      --max_val 40000
"""
import argparse, csv, json, os, sys, collections
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_spine import TOOLS, TASKS, FrameDS, Spine, read_manifest, NT, NK  # noqa

# data-floor: a val support below this is too thin for F1 to be a method signal.
FLOOR_VAL_SUPPORT = 300


@torch.no_grad()
def collect(model, dl, device):
    tl, tp, kk, kp = [], [], [], []
    for x, tools, task in dl:
        x = x.to(device, non_blocking=True)
        lt, lk = model(x)
        tl.append(tools); tp.append(torch.sigmoid(lt.float()).cpu())
        kk.append(task); kp.append(lk.float().cpu())
    return (torch.cat(tl).numpy(), torch.cat(tp).numpy(),
            torch.cat(kk).numpy(), torch.cat(kp).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/scratch/USERNAME/miccai-2026/SurgVU/models/spine_full/spine_best.pt")
    ap.add_argument("--manifest",
                    default="/scratch/USERNAME/miccai-2026/SurgVU/data/frame_manifest_full.csv")
    ap.add_argument("--max_val", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--img_size", type=int, default=384)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone = ck.get("args", {}).get("backbone", "tf_efficientnetv2_s.in21k_ft_in1k")
    thr = float(ck.get("tool_thr", 0.5))
    model = Spine(backbone=backbone).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"# ckpt={args.ckpt}\n# saved aggregate metrics={ck.get('metrics')}\n# tool_thr={thr}")

    rows = [r for r in read_manifest(args.manifest, require_video=True)
            if r["split"] == "val"]
    import random
    random.Random(1).shuffle(rows)
    rows = rows[:args.max_val]
    cases = [r["case"] for r in rows]
    dl = DataLoader(FrameDS(rows, img_size=args.img_size, train=False),
                    batch_size=args.bs, num_workers=args.workers, pin_memory=True)
    print(f"# val frames evaluated={len(rows)} device={device}")

    yt, pt, yk, pk = collect(model, dl, device)
    pt_bin = (pt > thr).astype(int)
    task_pred = pk.argmax(1)

    # ---------- STEP 1: per-class DECOMPOSE ----------
    tool_rows = []
    for j, name in enumerate(TOOLS):
        sup = int(yt[:, j].sum())
        f1 = f1_score(yt[:, j], pt_bin[:, j], zero_division=0)
        pr = precision_score(yt[:, j], pt_bin[:, j], zero_division=0)
        rc = recall_score(yt[:, j], pt_bin[:, j], zero_division=0)
        try:
            ap_j = average_precision_score(yt[:, j], pt[:, j]) if sup > 0 else float("nan")
        except ValueError:
            ap_j = float("nan")
        floor = sup < FLOOR_VAL_SUPPORT
        tool_rows.append(dict(tool=name, support=sup, f1=round(f1, 4),
                              precision=round(pr, 4), recall=round(rc, 4),
                              ap=round(float(ap_j), 4), data_floor=floor))
    tool_rows.sort(key=lambda d: d["f1"])  # worst -> best

    task_rows = []
    for i, name in enumerate(TASKS):
        m = yk == i; sup = int(m.sum())
        f1 = f1_score((yk == i).astype(int), (task_pred == i).astype(int), zero_division=0)
        acc = float((task_pred[m] == i).mean()) if sup else float("nan")
        task_rows.append(dict(task=name, support=sup, f1=round(f1, 4),
                              recall=round(acc, 4), data_floor=sup < FLOOR_VAL_SUPPORT))
    task_rows.sort(key=lambda d: d["f1"])

    # per-CASE task accuracy (centre / procedure shift)
    by_case = collections.defaultdict(lambda: [0, 0])
    for c, yp, yy in zip(cases, task_pred, yk):
        by_case[c][1] += 1; by_case[c][0] += int(yp == yy)
    case_acc = sorted(((c, v[0] / v[1], v[1]) for c, v in by_case.items()),
                      key=lambda x: x[1])

    # ---------- STEP 2: FAILURE modes ----------
    task_cm = confusion_matrix(yk, task_pred, labels=list(range(NK)))
    # tool confusion: among frames where true tool j absent but predicted, what
    # OTHER true tool was present (the look-alike that triggers the false positive).
    tool_conf = collections.Counter()
    for j in range(NT):
        fp = (pt_bin[:, j] == 1) & (yt[:, j] == 0)
        for k in range(NT):
            if k == j:
                continue
            tool_conf[(TOOLS[j], TOOLS[k])] += int((fp & (yt[:, k] == 1)).sum())
    top_conf = [(a, b, n) for (a, b), n in tool_conf.most_common(12) if n > 0]

    # ---------- STEP 3: QUANTIFY recoverable vs floor ----------
    recoverable = [d for d in tool_rows if not d["data_floor"]]
    floor = [d for d in tool_rows if d["data_floor"]]
    macro_now = float(np.mean([d["f1"] for d in tool_rows]))
    macro_no_floor = float(np.mean([d["f1"] for d in recoverable])) if recoverable else float("nan")
    # ceiling if every recoverable class hit the current best recoverable F1
    if recoverable:
        ceil = max(d["f1"] for d in recoverable)
        macro_ceiling_recoverable = float(
            np.mean([ceil if not d["data_floor"] else d["f1"] for d in tool_rows]))
    else:
        macro_ceiling_recoverable = float("nan")

    out = dict(ckpt=args.ckpt, n_val=len(rows), tool_thr=thr,
               tool_macro_f1=macro_now, tool_macro_f1_excl_floor=macro_no_floor,
               tool_macro_ceiling_if_recoverable_fixed=macro_ceiling_recoverable,
               per_tool=tool_rows, per_task=task_rows,
               worst_cases=[{"case": c, "task_acc": round(a, 3), "n": n} for c, a, n in case_acc[:10]],
               tool_confusion_top=[{"fp_tool": a, "true_present": b, "n": int(n)} for a, b, n in top_conf],
               task_confusion_matrix=task_cm.tolist(), tasks=TASKS,
               data_floor_tools=[d["tool"] for d in floor],
               recoverable_tools=[d["tool"] for d in recoverable])
    outpath = os.path.join(os.path.dirname(args.ckpt), "per_class_diag.json")
    with open(outpath, "w") as fh:
        json.dump(out, fh, indent=2)

    # ---------- printed report ----------
    print("\n=== STEP 1  per-TOOL F1 (worst -> best) ===")
    print(f"{'tool':34s} {'F1':>6s} {'P':>6s} {'R':>6s} {'AP':>6s} {'sup':>7s}  flag")
    for d in tool_rows:
        print(f"{d['tool']:34s} {d['f1']:6.3f} {d['precision']:6.3f} {d['recall']:6.3f} "
              f"{d['ap']:6.3f} {d['support']:7d}  {'DATA-FLOOR' if d['data_floor'] else ''}")
    print("\n=== STEP 1  per-TASK F1 (worst -> best) ===")
    for d in task_rows:
        print(f"{d['task']:34s} F1={d['f1']:.3f} recall={d['recall']:.3f} sup={d['support']:7d}"
              f"  {'DATA-FLOOR' if d['data_floor'] else ''}")
    print("\n=== STEP 1  worst 10 cases by task-acc (centre/procedure shift) ===")
    for c, a, n in case_acc[:10]:
        print(f"  {c:14s} task_acc={a:.3f} (n={n})")
    print("\n=== STEP 2  top tool false-positive confusion (fp_tool <- co-present true tool) ===")
    for a, b, n in top_conf:
        print(f"  predict {a:30s} when {b:30s} present : {n}")
    print("\n=== STEP 3  QUANTIFY bottleneck ===")
    print(f"  tool macro-F1 (all 14)            = {macro_now:.4f}")
    print(f"  tool macro-F1 (excl data-floor)   = {macro_no_floor:.4f}")
    print(f"  tool macro-F1 ceiling if recoverable fixed = {macro_ceiling_recoverable:.4f}")
    print(f"  DATA-FLOOR tools (need data): {out['data_floor_tools']}")
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
