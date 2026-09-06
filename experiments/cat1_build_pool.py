"""
SurgVU Cat-1 — build ONE shared frame/label pool + index files for every split.

WHY THIS EXISTS
---------------
`cat1_realbox_train.py` re-exports 5,178 jpgs + 5,178 label txts into a private
directory per run.  Running an 8-arm sweep + a 7-fold LOVO that way costs ~156k
inodes, and the scratch filesystem currently sits at 5.80M / 6.00M inodes
(see MEMORY: "SLURM inode-quota failure" — jobs die at 00:00:00 / ExitCode 0:53
with empty logs when this is exhausted).

So: export the 7 annotated videos ONCE into a flat pool (~10.4k inodes total) and
express every split as an ultralytics image-list `.txt`, which costs one inode each.

SPLITS PRODUCED
---------------
  main   : train = videos 1-5, val = videos 6,7   <- the EXACT split behind the
           logged mAP@[.5:.95] = 0.1470 baseline (job 6933641).  Unchanged, so
           every sweep arm stays directly comparable to that number.
  lovoK  : train = all videos except K, val = video K, for K = 1..7.
           Still strictly VIDEO-DISJOINT — this is an ADDITION to the main split,
           not a replacement, and it is reported as its own protocol.

It also writes `class_support.json`: per-video and per-split instance counts per
class.  That is load-bearing, because the main split's val set contains only 4 of
the 14 classes and TWO of them (clip_applier 258 boxes, vessel_sealer 13 boxes)
have ZERO instances anywhere in the training videos — their AP is forced to 0.000
by construction, which alone caps the main-split headline at 0.500.
"""

import json
import os
import sys
from collections import Counter

import cv2

EXT = "/scratch/sc20osc/miccai-2026/SurgVU/data/external/cat1_test_set"
POOL = "/scratch/sc20osc/miccai-2026/SurgVU/data/cat1_pool"
VIDEOS = [1, 2, 3, 4, 5, 6, 7]
MAIN_TRAIN = [1, 2, 3, 4, 5]
MAIN_VAL = [6, 7]


def norm(s):
    return s.lower().replace("-", " ").replace("/", " ").replace("_", " ").strip().replace(" ", "_")


def load_video_coco(v):
    return json.load(open(f"{EXT}/{v}_fps1_coco.json"))


def build_class_list():
    c = load_video_coco(VIDEOS[0])
    cats = sorted(c["categories"], key=lambda x: x["id"])
    return [norm(x["name"]) for x in cats], {x["id"]: i for i, x in enumerate(cats)}


def export_video(v, names, catid_to_idx, imdir, lbdir):
    """Decode video v at its annotated frame indices; write jpg + YOLO txt. Returns
    (list of image paths, Counter of class instance counts)."""
    c = load_video_coco(v)
    meta = {i["id"]: i for i in c["images"]}
    anns = {}
    for a in c["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    want = {int(os.path.splitext(meta[i]["file_name"])[0]): i for i in meta}

    paths, cnt = [], Counter()
    cap = cv2.VideoCapture(f"{EXT}/{v}_fps1.mp4")
    fidx, maxf = 0, (max(want) if want else -1)
    while True:
        ok, frame = cap.read()
        if not ok or fidx > maxf:
            break
        iid = want.get(fidx)
        if iid is not None:
            im = meta[iid]
            W, H = im["width"], im["height"]
            fr = frame if (frame.shape[1] == W and frame.shape[0] == H) else cv2.resize(frame, (W, H))
            stem = f"v{v}_{fidx:08d}"
            ip = os.path.join(imdir, stem + ".jpg")
            cv2.imwrite(ip, fr)
            lines = []
            for a in anns.get(iid, []):
                x, y, w, h = a["bbox"]
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                lines.append(f"{catid_to_idx[a['category_id']]} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
                cnt[names[catid_to_idx[a["category_id"]]]] += 1
            with open(os.path.join(lbdir, stem + ".txt"), "w") as f:
                f.write("\n".join(lines))
            paths.append(ip)
        fidx += 1
    cap.release()
    return paths, cnt


def main():
    names, catid_to_idx = build_class_list()
    imdir, lbdir, spdir = f"{POOL}/images", f"{POOL}/labels", f"{POOL}/splits"
    for d in (imdir, lbdir, spdir):
        os.makedirs(d, exist_ok=True)

    marker = f"{POOL}/POOL.done"
    if os.path.exists(marker):
        print(f"[pool] already built ({marker}); nothing to do", flush=True)
        return 0

    per_video_paths, per_video_cnt = {}, {}
    for v in VIDEOS:
        p, c = export_video(v, names, catid_to_idx, imdir, lbdir)
        per_video_paths[v], per_video_cnt[v] = p, c
        print(f"[pool] video {v}: {len(p)} imgs, {sum(c.values())} boxes, {len(c)} classes", flush=True)

    def write_split(tag, vids):
        fp = f"{spdir}/{tag}.txt"
        with open(fp, "w") as f:
            for v in vids:
                f.write("\n".join(per_video_paths[v]) + "\n")
        return fp

    splits = {}
    write_split("main_train", MAIN_TRAIN)
    write_split("main_val", MAIN_VAL)
    splits["main"] = {"train": MAIN_TRAIN, "val": MAIN_VAL}
    for k in VIDEOS:
        write_split(f"lovo{k}_train", [v for v in VIDEOS if v != k])
        write_split(f"lovo{k}_val", [k])
        splits[f"lovo{k}"] = {"train": [v for v in VIDEOS if v != k], "val": [k]}

    support = {"names": names, "per_video": {str(v): dict(per_video_cnt[v]) for v in VIDEOS},
               "splits": {}}
    for tag, sp in splits.items():
        tr, va = Counter(), Counter()
        for v in sp["train"]:
            tr += per_video_cnt[v]
        for v in sp["val"]:
            va += per_video_cnt[v]
        # classes present in val; those with ZERO train support have AP forced to 0.
        zero_support = sorted([c for c in va if tr.get(c, 0) == 0])
        support["splits"][tag] = {
            "train_videos": sp["train"], "val_videos": sp["val"],
            "train_counts": dict(tr), "val_counts": dict(va),
            "n_val_classes": len(va),
            "zero_train_support_classes": zero_support,
            "forced_zero_fraction": (len(zero_support) / len(va)) if va else 0.0,
            "attainable_map_ceiling": 1.0 - (len(zero_support) / len(va)) if va else 0.0,
        }
        print(f"[split {tag}] val classes={len(va)} zero-train-support={zero_support} "
              f"ceiling={support['splits'][tag]['attainable_map_ceiling']:.3f}", flush=True)

    with open(f"{POOL}/class_support.json", "w") as f:
        json.dump(support, f, indent=2)
    with open(marker, "w") as f:
        f.write("ok\n")
    print(f"[pool] DONE -> {POOL}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
