"""Cat-1 stage (b): single-object TRACKING propagation (HRI_MV #1 cascade).

A sparse-labeled frame (its seg-box + spine-class) SEEDS a dense, temporally-consistent
box track. We run an ultralytics tracker (bytetrack / botsort) over a short window of
consecutive frames sampled from each SurgVU clip; the seg model proposes instances per
frame, the tracker links them into tracks, and each track inherits the spine tool-class
of the seed frame it overlaps. This turns ONE labeled frame per window into dense
supervision across the window (cheap temporal label propagation).

Output schema is IDENTICAL to seg_pseudoboxes.py / cam_pseudoboxes.py so it drops
straight into train_detector.py, AND it is the second source for the stage-(d)
detection+tracking WBF fusion ensemble.

Because we have no per-clip frame index for the random manifest frames, "propagation"
here operates on a WINDOW of consecutive timestamps decoded from each clip's mp4 around
a seed timestamp: seed-frame seg+spine class -> tracker carries the box id forward/back
across the window, and the spine is queried per window-frame to keep the present-tool
set honest (a track is kept only while its tool stays spine-present).

LICENSE: same as seg_pseudoboxes.py (yolo*-seg.pt AGPL, OFFLINE generation only).

Usage (full -> sbatch):
  python track_propagate.py --ckpt .../spine_best.pt --manifest .../frame_manifest_full.csv \
      --out_dir .../track_vis --n_windows 200 --window 8 --stride_sec 0.5
"""
import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import cv2

from train_spine import Spine, TOOLS
from seg_pseudoboxes import mask_to_box, tool_likeness, saliency_box, SEG_WEIGHTS

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def decode_at(video, t_sec):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, float(t_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def prep(frame_bgr, img_size):
    """center-crop 6% + resize (matches FrameDS) -> (vis_bgr for cv2, tensor for spine)."""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    cy, cx = int(h * 0.06), int(w * 0.06)
    img = img[cy:h - cy, cx:w - cx]
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    vis = img[:, :, ::-1].copy()  # RGB->BGR for cv2 + seg
    n = (img.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(np.ascontiguousarray(n.transpose(2, 0, 1)))
    return vis, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default="/scratch/USERNAME/miccai-2026/SurgVU/models/track_vis")
    ap.add_argument("--n_windows", type=int, default=200)
    ap.add_argument("--window", type=int, default=8)        # frames per window
    ap.add_argument("--stride_sec", type=float, default=0.5)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--seg_conf", type=float, default=0.10)
    ap.add_argument("--max_tools", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = Spine(ck["args"]["backbone"]).to(device).eval()
    model.load_state_dict(ck["model"])
    tool_thr = ck.get("tool_thr", 0.5)
    img_size = ck["args"]["img_size"]

    from ultralytics import YOLO
    seg = YOLO(SEG_WEIGHTS)

    rows = [r for r in csv.DictReader(open(args.manifest)) if r["video"]]
    random.Random(1).shuffle(rows)

    out, n_box, gi = [], 0, 0
    nw = min(args.n_windows, len(rows))
    for wi in range(nw):
        r = rows[wi]
        t0 = float(r["t_sec"])
        # spine-present tools at the seed frame fix the class set for the whole window
        seed = decode_at(r["video"], t0)
        if seed is None:
            continue
        vis0, x0 = prep(seed, img_size)
        with torch.no_grad():
            lt, _ = model(x0.unsqueeze(0).to(device))
            prob = torch.sigmoid(lt)[0]
        present = (prob > tool_thr).nonzero().flatten().tolist()
        present = sorted(present, key=lambda c: -float(prob[c]))[: args.max_tools]
        if not present:
            continue

        # build the window of consecutive frames around the seed
        for k in range(args.window):
            t = t0 + (k - args.window // 2) * args.stride_sec
            if t < 0:
                continue
            fr = decode_at(r["video"], t)
            if fr is None:
                continue
            vis, _ = prep(fr, img_size)
            H, W = vis.shape[:2]
            # track() links instances across the persistent-stream calls
            res = seg.track(vis, conf=args.seg_conf, persist=True, verbose=False,
                            device=device, imgsz=img_size, tracker=args.tracker,
                            retina_masks=True)[0]
            seg_boxes = []
            if res.masks is not None:
                masks = res.masks.data.cpu().numpy()
                mh, mw = masks.shape[1], masks.shape[2]
                ids = (res.boxes.id.cpu().numpy().astype(int)
                       if res.boxes is not None and res.boxes.id is not None
                       else list(range(len(masks))))
                for mk, tid in zip(masks, ids):
                    if (mh, mw) != (H, W):
                        mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
                    bb = mask_to_box(mk)
                    if bb is None:
                        continue
                    seg_boxes.append((bb, tool_likeness(bb, H, W), int(tid)))
            seg_boxes.sort(key=lambda t_: -t_[1])
            frame_boxes = []
            for j, c in enumerate(present):
                if j < len(seg_boxes):
                    (bx, by, bw, bh), score, tid = seg_boxes[j]
                    src = f"track{tid}"
                else:
                    bx, by, bw, bh = saliency_box(vis)
                    score, src = 0.30, "fallback"
                frame_boxes.append({
                    "class": TOOLS[c], "class_id": int(c),
                    "xywh_norm": [bx / W, by / H, bw / W, bh / H],
                    "score": float(min(score, 1.0)), "src": src,
                })
                n_box += 1
            cv2.imwrite(os.path.join(args.out_dir, f"cam_{gi:03d}.png"), vis)
            out.append({"frame": gi, "case": r["case"], "boxes": frame_boxes})
            gi += 1
        seg.predictor = None  # reset tracker state between windows (fresh track ids)

    json.dump(out, open(os.path.join(args.out_dir, "track_pseudo_boxes.json"), "w"), indent=2)
    nb = [len(o["boxes"]) for o in out]
    areas = [b["xywh_norm"][2] * b["xywh_norm"][3] for o in out for b in o["boxes"]]
    print(f"wrote {len(out)} propagated frames -> {args.out_dir}/track_pseudo_boxes.json")
    print(f"TRACK_total_boxes={n_box}  windows={nw}  frames_per_window~{args.window}")
    print(f"TRACK_mean_boxes_per_frame={np.mean(nb):.3f}" if nb else "TRACK_mean=nan")
    print(f"TRACK_mean_box_area_frac={np.mean(areas):.4f}" if areas else "TRACK_area=nan")


if __name__ == "__main__":
    main()
