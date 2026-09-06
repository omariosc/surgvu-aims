"""
STAGE 4 — build presence-gated pseudo-labels, retrain, and score.

THE FIVE ARMS.  Two are the lever; two are controls whose only job is to take the
lever away from us if the story is wrong; one is the ship candidate.

  FILTER   keep a proposal iff conf>=tau AND its predicted class is installed at
           that instant.  Raises support for classes the proposer already knows.
           Cannot create a class the proposer has never seen.

  ASSIGN   on SINGLETON frames only (exactly one in-vocabulary tool installed, no
           unknown instrument), take class-agnostic boxes and NAME them with that
           one installed tool.  This is the ONLY mechanism that can create boxes
           for a class with zero boxes anywhere -- and therefore the only one that
           can move clip_applier / vessel_sealer off their structural 0.000.

  COMBINED ASSIGN on singleton frames, FILTER elsewhere.  The ship candidate.

  NEG_UNCONSTRAINED  ordinary self-training: keep confident boxes with the
           detector's OWN predicted class, no presence check at all.  If this
           matches FILTER/COMBINED, the presence labels are decoration and the
           mechanism claim is false.

  NEG_SHUFFLE  ASSIGN with the singleton class permuted ACROSS CASES, preserving
           the class marginal but destroying the frame<->class correspondence.
           If this matches ASSIGN, the gain came from extra images and extra
           boxes, not from correct names.  This is the sharper of the two
           controls, because ASSIGN's whole claim is that the NAME is right.

WHY THE MAIN SPLIT IS THE RIGHT INSTRUMENT FOR THE MECHANISM.  It was correctly
diagnosed as a broken instrument for overall detector quality: 2 of its 4 val
classes have zero train boxes, so 50% of its denominator is pinned at 0.000 and
cannot move for any arm.  That property is exactly what makes it the SHARPEST
instrument for THIS lever.  clip_applier (258 val boxes) and vessel_sealer (13)
are 0.000 in all three baseline seeds BY CONSTRUCTION, so their seed noise is
identically zero and ANY non-zero AP is outside the noise band -- no Delta_noise
argument can explain it away.  The 7-fold LOVO protocol remains the headline.
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cat1_ssl_common import ANN_POOL, CLASSES, CLS_IDX, OUT_H, OUT_W, SSL

W, H = OUT_W, OUT_H
ARMS = ["FILTER", "ASSIGN", "COMBINED", "NEG_UNCONSTRAINED", "NEG_SHUFFLE"]


def nms_agnostic(boxes, scores, thr=0.6):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(a_i + a_r - inter, 1e-9)
        order = rest[iou <= thr]
    return keep


def build_pseudo(arm, proposer, tau, seed, max_boxes_assign=2):
    rows = list(csv.DictReader(open(f"{SSL}/candidates.csv")))
    d = np.load(f"{SSL}/det/{proposer}.npz")
    idx, xyxy, conf, cls = d["idx"], d["xyxy"], d["conf"], d["cls"]
    by_img = defaultdict(list)
    for k in range(len(idx)):
        if conf[k] >= tau:
            by_img[int(idx[k])].append(k)

    # NEG_SHUFFLE: permute the singleton class across CASES, so a whole case gets a
    # consistently wrong name.  Permuting per frame would let a case keep its class
    # on average and quietly leak the right answer back in.
    rng = random.Random(1000 + seed)
    case_relabel = {}
    if arm == "NEG_SHUFFLE":
        pairs = sorted({(r["case"], r["singleton"]) for r in rows if r["singleton"]})
        src = [c for _, c in pairs]
        rng.shuffle(src)
        case_relabel = {k: v for k, v in zip(pairs, src)}

    out = {}
    for i, r in enumerate(rows):
        ks = by_img.get(i, [])
        if not ks:
            continue
        inst = set(r["installed"].split("|")) if r["installed"] else set()
        single = r["singleton"]
        labs = []
        if arm == "NEG_UNCONSTRAINED":
            labs = [(int(cls[k]), xyxy[k]) for k in ks]
        elif arm == "FILTER":
            labs = [(int(cls[k]), xyxy[k]) for k in ks if CLASSES[int(cls[k])] in inst]
        elif arm in ("ASSIGN", "NEG_SHUFFLE"):
            if not single:
                continue
            name = case_relabel.get((r["case"], single), single) if arm == "NEG_SHUFFLE" else single
            b = np.stack([xyxy[k] for k in ks])
            s = np.array([conf[k] for k in ks])
            labs = [(CLS_IDX[name], b[j]) for j in nms_agnostic(b, s)[:max_boxes_assign]]
        elif arm == "COMBINED":
            if single:
                b = np.stack([xyxy[k] for k in ks])
                s = np.array([conf[k] for k in ks])
                labs = [(CLS_IDX[single], b[j]) for j in nms_agnostic(b, s)[:max_boxes_assign]]
            else:
                labs = [(int(cls[k]), xyxy[k]) for k in ks if CLASSES[int(cls[k])] in inst]
        if labs:
            out[r["stem"]] = labs
    return rows, out


def select(rows, pseudo, per_class, budget, seed, split):
    """Cap per primary class (so needle_driver cannot flood the pool) and overall.

    Starvation is measured against THIS SPLIT'S TRAIN COUNTS, not the corpus-wide
    totals.  On the main split clip_applier has 258 boxes overall but ZERO in
    training -- ranking by the overall count would push the one class this whole
    experiment exists to rescue to the back of the queue.
    """
    sup = json.load(open(f"{ANN_POOL}/class_support.json"))
    tr = sup["splits"][split]["train_counts"]
    prim = {stem: Counter(CLASSES[c] for c, _ in labs).most_common(1)[0][0]
            for stem, labs in pseudo.items()}
    rng = random.Random(2000 + seed)
    order = defaultdict(list)
    for stem in pseudo:
        order[prim[stem]].append(stem)
    for v in order.values():
        rng.shuffle(v)

    # rarest-in-training first; zero-support classes are served before everything
    ranked = sorted(order, key=lambda c: (tr.get(c, 0), c))
    keep, taken = [], Counter()
    hard_cap = budget + per_class * sum(1 for c in ranked if tr.get(c, 0) == 0)
    for c in ranked:
        starved = tr.get(c, 0) == 0
        for stem in order[c]:
            if taken[c] >= per_class:
                break
            if not starved and len(keep) >= budget:
                break
            if len(keep) >= hard_cap:
                break
            keep.append(stem)
            taken[c] += 1
    print("  selection per primary class: " +
          ", ".join(f"{c}={taken[c]}" for c in ranked if taken[c]))
    return keep, prim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--split", default="main")
    ap.add_argument("--proposer", default=None, help="defaults to the split's own proposer")
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--per_class", type=int, default=900)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--model", default="/scratch/sc20osc/AILET/from_pc/weights/yolo26m.pt")
    ap.add_argument("--project", default="/scratch/sc20osc/miccai-2026/SurgVU/models/cat1_ssl")
    ap.add_argument("--out_dir", default="/scratch/sc20osc/miccai-2026/SurgVU/logs/cat1_ssl")
    args = ap.parse_args()
    proposer = args.proposer or args.split

    gate = json.load(open(f"{SSL}/GATE.json"))
    if not gate.get("PASS"):
        print("GATE.json says the proposal mechanism FAILED its pre-registered "
              "label-free check; refusing to spend GPU on training.")
        print(json.dumps(gate, indent=2))
        return 0

    tag = f"{args.arm}_{args.split}_t{int(args.tau*100):02d}_s{args.seed}"
    print(f"=== {tag} (proposer={proposer}) ===", flush=True)

    rows, pseudo = build_pseudo(args.arm, proposer, args.tau, args.seed)
    keep, prim = select(rows, pseudo, args.per_class, args.budget, args.seed, args.split)
    print(f"pseudo frames available={len(pseudo)}  selected={len(keep)}", flush=True)

    armdir = f"{SSL}/arms/{tag}"
    lbdir = f"{armdir}/labels"
    os.makedirs(lbdir, exist_ok=True)
    link = f"{armdir}/images"
    if not os.path.islink(link):
        os.symlink(f"{SSL}/images", link)      # 1 inode, not one per image

    added = Counter()
    for stem in keep:
        lines = []
        for c, b in pseudo[stem]:
            x1, y1, x2, y2 = [float(v) for v in b]
            x1, x2 = max(0.0, x1), min(float(W), x2)
            y1, y2 = max(0.0, y1), min(float(H), y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            lines.append(f"{c} {((x1+x2)/2)/W:.6f} {((y1+y2)/2)/H:.6f} "
                         f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
            added[CLASSES[c]] += 1
        with open(f"{lbdir}/{stem}.txt", "w") as f:
            f.write("\n".join(lines))

    # ---- train list = annotated split train + selected pseudo frames
    ann_train = [l.strip() for l in open(f"{ANN_POOL}/splits/{args.split}_train.txt") if l.strip()]
    trlist = f"{armdir}/train.txt"
    with open(trlist, "w") as f:
        f.write("\n".join(ann_train + [f"{link}/{s}.jpg" for s in keep]) + "\n")

    # ---- SUPPORT BEFORE / AFTER (the number that says whether the mechanism fired)
    sup = json.load(open(f"{ANN_POOL}/class_support.json"))
    before = sup["splits"][args.split]["train_counts"]
    support = {c: {"before": before.get(c, 0), "added": added.get(c, 0),
                   "after": before.get(c, 0) + added.get(c, 0)} for c in CLASSES}
    print(f"\n{'class':34s} {'before':>8s} {'added':>8s} {'after':>8s}")
    for c in CLASSES:
        s = support[c]
        flag = "  <-- was ZERO" if s["before"] == 0 and s["added"] > 0 else ""
        print(f"{c:34s} {s['before']:8d} {s['added']:8d} {s['after']:8d}{flag}")

    yaml_path = f"{armdir}/data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {SSL}\ntrain: {trlist}\n"
                f"val: {ANN_POOL}/splits/{args.split}_val.txt\n"
                f"nc: {len(CLASSES)}\nnames: {json.dumps(CLASSES)}\n")

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=args.project, name=tag, exist_ok=True, pretrained=True,
                seed=args.seed, verbose=True, device=0)

    def evaluate(weights, t):
        mdl = YOLO(weights)
        m = mdl.val(data=yaml_path, imgsz=args.imgsz, device=0, split="val",
                    project=args.project, name=f"{tag}_val_{t}", exist_ok=True)
        present = [int(i) for i in m.box.ap_class_index]
        per_class = {CLASSES[c]: float(m.box.maps[c]) for c in present}
        return {"map": float(m.box.map), "map50": float(m.box.map50),
                "precision": float(m.box.mp), "recall": float(m.box.mr),
                "per_class_ap": per_class}

    res = {"tag": tag, "arm": args.arm, "split": args.split, "proposer": proposer,
           "tau": args.tau, "seed": args.seed,
           "n_pseudo_frames": len(keep), "n_pseudo_boxes": int(sum(added.values())),
           "support": support}
    for t in ("best", "last"):
        w = f"{args.project}/{tag}/weights/{t}.pt"
        if os.path.exists(w):
            res[t] = evaluate(w, t)
            ap_ = res[t]["per_class_ap"]
            print(f"[{tag}] {t}.pt mAP={res[t]['map']:.4f} P={res[t]['precision']:.3f} "
                  f"R={res[t]['recall']:.3f}")
            for c, v in sorted(ap_.items()):
                z = " (was structurally 0.000)" if before.get(c, 0) == 0 else ""
                print(f"      AP {c:34s} {v:.4f}{z}")

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(res, open(f"{args.out_dir}/{tag}.json", "w"), indent=2)
    print(f"\nSURGVU_SSL tag={tag} BEST={res.get('best',{}).get('map',float('nan')):.4f} "
          f"LAST={res.get('last',{}).get('map',float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
