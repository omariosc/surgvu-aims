"""Cat-1 weakly-supervised detection scaffold (Phase 1, lever 3).

The prior SurgToolLoc/SurgVU detection winners bootstrapped a detector from a
tool-PRESENCE classifier with NO boxes:
    tool-presence classifier (our grounding spine)
      -> Grad-CAM per predicted tool class
      -> threshold + connected-components -> pseudo bounding boxes
      -> train a YOLO/DETR on the pseudo-boxes
      -> refine on the small 2026 bbox val set (when it lands).

There is NO 2024 bbox ground truth, so this module only produces + visualises
pseudo-boxes for qualitative sanity now. It reuses the trained Spine checkpoint
(train_spine.py) as the CAM source.

Usage (after a spine checkpoint exists and >=1 video is local):
  python cam_pseudoboxes.py --ckpt .../spine_best.pt --manifest .../frame_manifest_local.csv \
      --out_dir .../cam_vis --n 40
Outputs: per-frame overlay PNGs + a pseudo_boxes.json (YOLO-trainable) with
[class, x, y, w, h, score] in normalized coords, ready to feed ultralytics YOLO.
"""
import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from train_spine import Spine, FrameDS, TOOLS


class GradCAM:
    """Grad-CAM on the last conv block of the timm backbone."""
    def __init__(self, model):
        self.model = model.eval()
        self.acts = None
        self.grads = None
        # EfficientNetV2 final feature stage: hook the conv_head / last block
        target = None
        for name, mod in model.bb.named_modules():
            if name.endswith("conv_head") or name.endswith("bn2"):
                target = mod
        if target is None:  # fallback: last module
            target = list(model.bb.modules())[-1]
        target.register_forward_hook(self._fwd)
        target.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, cls):
        lt, _ = self.model(x)
        self.model.zero_grad()
        lt[0, cls].backward(retain_graph=True)
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1))[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        return cam.cpu().numpy()


def cam_to_boxes(cam, thr=0.4):
    """connected components on the thresholded CAM -> normalized xywh boxes."""
    H, W = cam.shape
    mask = (cam > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < 0.01 * H * W:
            continue
        score = float(cam[y:y + h, x:x + w].mean())
        boxes.append([x / W, y / H, w / W, h / H, score])
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default="/scratch/USERNAME/miccai-2026/SurgVU/models/cam_vis")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--thr", type=float, default=0.4)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = Spine(ck["args"]["backbone"]).to(device)
    model.load_state_dict(ck["model"])
    cam = GradCAM(model)
    tool_thr = ck.get("tool_thr", 0.5)

    rows = [r for r in csv.DictReader(open(args.manifest)) if r["video"]]
    random.Random(0).shuffle(rows)
    ds = FrameDS(rows, ck["args"]["img_size"], train=False)
    out = []
    for i in range(min(args.n, len(rows))):
        x, tools, _ = ds[i]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            lt, _ = model(x)
        present = (torch.sigmoid(lt)[0] > tool_thr).nonzero().flatten().tolist()
        vis = ((x[0].cpu().numpy().transpose(1, 2, 0) *
                np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1].copy()
        frame_boxes = []
        for c in present:
            heat = cam(x, c)
            heat = cv2.resize(heat, (vis.shape[1], vis.shape[0]))
            for b in cam_to_boxes(heat, args.thr):
                frame_boxes.append({"class": TOOLS[c], "class_id": c, "xywh_norm": b[:4], "score": b[4]})
                X, Y = int(b[0] * vis.shape[1]), int(b[1] * vis.shape[0])
                Wb, Hb = int(b[2] * vis.shape[1]), int(b[3] * vis.shape[0])
                cv2.rectangle(vis, (X, Y), (X + Wb, Y + Hb), (0, 255, 0), 2)
                cv2.putText(vis, TOOLS[c], (X, max(12, Y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(args.out_dir, f"cam_{i:03d}.png"), vis)
        out.append({"frame": i, "case": rows[i]["case"], "boxes": frame_boxes})
    json.dump(out, open(os.path.join(args.out_dir, "pseudo_boxes.json"), "w"), indent=2)
    print(f"wrote {len(out)} CAM overlays + pseudo_boxes.json -> {args.out_dir}")


if __name__ == "__main__":
    main()
