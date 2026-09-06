"""SurgVU grounding-spine classifier (Phase 1, P1.1).

EfficientNetV2-S (timm) with two heads:
  - tool head:  14-way multi-label (BCEWithLogits, <=3 tools/frame)
  - task head:  8-way single-label (CrossEntropy)

Trains from the frame manifest (build_frame_labels.py). Frames are decoded
on-the-fly from the mp4 by timestamp (OpenCV CAP_PROP_POS_MSEC seek), so no
frame pre-extraction is required. Only manifest rows with a local video are used.

Metrics: per-head macro-F1 on the case-disjoint val split (target >=0.95).
Saves best-macro-F1 checkpoint + a tool-threshold sweep for the Cat-2 spine.

This is the grounding spine feeding (a) the Cat-2 templated-answer router and
(b) the Cat-1 CAM->pseudo-box detector.
"""
import argparse
import csv
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2
import timm
from sklearn.metrics import f1_score, average_precision_score

TOOLS = [
    "needle_driver", "monopolar_curved_scissor", "force_bipolar", "clip_applier",
    "tip_up_fenestrated_grasper", "cadiere_forceps", "bipolar_forceps",
    "vessel_sealer", "suction_irrigator", "bipolar_dissector", "prograsp_forceps",
    "stapler", "permanent_cautery_hook_spatula", "grasping_retractor",
]
TASKS = [
    "Suturing", "Uterine horn", "Suspensory ligaments", "Rectal artery/vein",
    "Skills application", "Range of motion", "Retraction and collision avoidance",
    "Other",
]
NT, NK = len(TOOLS), len(TASKS)


# ----------------------------------------------------------------------------
def read_manifest(path, require_video=True):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if require_video and not r["video"]:
                continue
            rows.append(r)
    return rows


def balance_other(rows, other_keep_frac, seed=0):
    """Down-sample frames whose task==Other AND no tool present (pure-empty-ish)
    to keep the task head from collapsing onto the majority Other class."""
    rng = random.Random(seed)
    other_idx = TASKS.index("Other")
    keep = []
    for r in rows:
        is_other = int(r["task_idx"]) == other_idx
        has_tool = any(c == "1" for c in r["tools_multihot"])
        if is_other and not has_tool and rng.random() > other_keep_frac:
            continue
        keep.append(r)
    return keep


class FrameDS(Dataset):
    def __init__(self, rows, img_size=384, train=True):
        self.rows = rows
        self.img = img_size
        self.train = train
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def __len__(self):
        return len(self.rows)

    def _decode(self, video, t_sec):
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t_sec) * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __getitem__(self, i):
        r = self.rows[i]
        img = self._decode(r["video"], r["t_sec"])
        if img is None:
            img = np.zeros((self.img, self.img, 3), np.uint8)
        # SurgVU frames have black UI strips; center-crop to surgical FOV then resize
        h, w = img.shape[:2]
        # light center crop (remove ~6% border)
        cy, cx = int(h * 0.06), int(w * 0.06)
        img = img[cy:h - cy, cx:w - cx]
        img = cv2.resize(img, (self.img, self.img), interpolation=cv2.INTER_AREA)
        if self.train and random.random() < 0.5:
            img = img[:, ::-1, :]
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        tools = torch.tensor([float(c) for c in r["tools_multihot"]], dtype=torch.float32)
        task = torch.tensor(int(r["task_idx"]), dtype=torch.long)
        return x, tools, task


class Spine(nn.Module):
    def __init__(self, backbone="tf_efficientnetv2_s.in21k_ft_in1k", drop=0.2):
        super().__init__()
        self.bb = timm.create_model(backbone, pretrained=True, num_classes=0, drop_rate=drop)
        d = self.bb.num_features
        self.tool_head = nn.Linear(d, NT)
        self.task_head = nn.Linear(d, NK)

    def forward(self, x):
        f = self.bb(x)
        return self.tool_head(f), self.task_head(f)


def macro_f1_tools(logits, y, thr=0.5):
    p = (torch.sigmoid(logits) > thr).int().numpy()
    return f1_score(y.numpy(), p, average="macro", zero_division=0)


# tools with ~0 val support (data-floor) -- excluded from the balanced-sampling
# up-weighting per the brief ("excluding the 3 zero-val data-floor tools"). These
# are bipolar_dissector (0 val), suction_irrigator + tip_up_fenestrated_grasper
# (thin val); up-weighting them only over-fits noise, so we leave them at base wt.
DATA_FLOOR_TOOLS = ["bipolar_dissector", "suction_irrigator", "tip_up_fenestrated_grasper"]


def inverse_freq_sample_weights(rows, exclude_tools=DATA_FLOOR_TOOLS, task_alpha=0.5,
                                strength=1.0):
    """Per-frame sampling weight = max over the (non-floor) tools present of the
    tool's inverse frequency, combined with the frame task's inverse frequency.
    Rare-tool / rare-task frames are sampled more often so the head stops collapsing
    onto the head classes. Data-floor tools are excluded from the up-weighting (the
    brief: target the rare tools/tasks 'excluding the 3 zero-val data-floor tools').

    `strength` in [0,1+] tempers the inverse-freq weighting:
      0 -> uniform (no balancing); 1 -> full inverse-freq; >1 -> over-balance.
    Implemented as w = (inv_freq)^strength so the grid can sweep the balancing knob.
    """
    excl = {TOOLS.index(t) for t in exclude_tools if t in TOOLS}
    tool_count = np.ones(NT)
    task_count = np.ones(NK)
    for r in rows:
        for j, c in enumerate(r["tools_multihot"]):
            if c == "1":
                tool_count[j] += 1
        task_count[int(r["task_idx"])] += 1
    tool_inv = tool_count.sum() / tool_count        # inverse freq
    task_inv = task_count.sum() / task_count
    # cap so a 1-frame class doesn't dominate the whole epoch
    tool_inv = np.clip(tool_inv, 1.0, 50.0) ** strength
    task_inv = np.clip(task_inv, 1.0, 20.0) ** strength
    weights = []
    for r in rows:
        present = [j for j, c in enumerate(r["tools_multihot"]) if c == "1" and j not in excl]
        w_tool = max((tool_inv[j] for j in present), default=1.0)
        w_task = task_inv[int(r["task_idx"])] ** task_alpha
        weights.append(float(w_tool * w_task))
    return weights


class FocalBCE(nn.Module):
    """Multi-label focal loss (Lin et al.) for the tool head -- down-weights the
    easy head-class negatives so rare-tool gradients are not drowned out."""
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none", pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        pt = p * target + (1 - p) * (1 - target)
        return (((1 - pt) ** self.gamma) * bce).mean()


# ----------------------------------------------------------------------------
# Hard-pair confusion penalty (LEVER: the #1 spine confusion bipolar<-cadiere,
# n=1106 in per_class_diag; case124 fires bipolar_forceps not cadiere_forceps).
# For each look-alike pair (a, b): on frames where exactly ONE of the two is the
# ground-truth tool, the model must NOT prefer the absent member. We penalise the
# predicted prob of the ABSENT member relative to the PRESENT one with a margin --
# a contrastive hinge that sharpens the decision boundary between the confusables
# WITHOUT touching the other 12 tools' BCE. Default-OFF (weight 0) => byte-identical
# to the baseline tool loss; only --hardpair turns it on.
HARD_PAIRS = [
    ("bipolar_forceps", "cadiere_forceps"),   # #1 confusion (n=1106), drives case124
    ("cadiere_forceps", "needle_driver"),     # #2 forceps<-driver family confusion
    ("prograsp_forceps", "cadiere_forceps"),  # #3 grasper<->forceps look-alike
]


def build_hardpair_index(pairs=HARD_PAIRS):
    """Resolve the named pairs to TOOLS indices; drop any name not in the list."""
    idx = []
    for a, b in pairs:
        if a in TOOLS and b in TOOLS:
            idx.append((TOOLS.index(a), TOOLS.index(b)))
    return idx


class HardPairLoss(nn.Module):
    """Contrastive margin hinge on confusable tool pairs. On frames where exactly
    one member of a pair is the GT tool, push the present member's logit above the
    absent member's by `margin`:  L = mean over active rows of
        relu(margin - (logit_present - logit_absent)).
    Symmetric (handles both (a present,b absent) and (b present,a absent)); rows
    where both or neither are present contribute nothing (no gradient)."""
    def __init__(self, pair_idx, margin=2.0):
        super().__init__()
        self.pair_idx = pair_idx
        self.margin = margin

    def forward(self, logits, target):
        if not self.pair_idx:
            return logits.new_zeros(())
        terms = []
        for ia, ib in self.pair_idx:
            la, lb = logits[:, ia], logits[:, ib]
            ya, yb = target[:, ia], target[:, ib]
            # rows where a present & b absent  -> want la > lb + margin
            m_ab = (ya > 0.5) & (yb < 0.5)
            # rows where b present & a absent  -> want lb > la + margin
            m_ba = (yb > 0.5) & (ya < 0.5)
            if m_ab.any():
                terms.append(torch.relu(self.margin - (la[m_ab] - lb[m_ab])))
            if m_ba.any():
                terms.append(torch.relu(self.margin - (lb[m_ba] - la[m_ba])))
        if not terms:
            return logits.new_zeros(())
        return torch.cat(terms).mean()


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    tl, tp, kk, kp = [], [], [], []
    for x, tools, task in dl:
        x = x.to(device, non_blocking=True)
        lt, lk = model(x)
        tl.append(tools); tp.append(lt.float().cpu())
        kk.append(task); kp.append(lk.float().cpu())
    tl = torch.cat(tl); tp = torch.cat(tp); kk = torch.cat(kk); kp = torch.cat(kp)
    # tool threshold sweep
    best_thr, best_f1 = 0.5, -1
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        f = macro_f1_tools(tp, tl, thr)
        if f > best_f1:
            best_f1, best_thr = f, thr
    try:
        tool_map = average_precision_score(tl.numpy(), torch.sigmoid(tp).numpy(), average="macro")
    except ValueError:
        tool_map = float("nan")
    task_pred = kp.argmax(1)
    task_f1 = f1_score(kk.numpy(), task_pred.numpy(), average="macro", zero_division=0)
    task_acc = (task_pred == kk).float().mean().item()
    return dict(tool_f1=best_f1, tool_thr=best_thr, tool_map=tool_map,
                task_f1=task_f1, task_acc=task_acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/scratch/sc20osc/miccai-2026/SurgVU/data/frame_manifest.csv")
    ap.add_argument("--out_dir", default="/scratch/sc20osc/miccai-2026/SurgVU/models/spine")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_train", type=int, default=120000, help="cap train frames/epoch")
    ap.add_argument("--max_val", type=int, default=20000)
    ap.add_argument("--other_keep_frac", type=float, default=0.25)
    ap.add_argument("--backbone", default="tf_efficientnetv2_s.in21k_ft_in1k")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--balanced_sampling", action="store_true",
                    help="inverse-freq WeightedRandomSampler targeting rare tools/tasks "
                         "(excl. the 3 zero-val data-floor tools)")
    ap.add_argument("--focal", action="store_true",
                    help="use multi-label focal loss for the tool head")
    ap.add_argument("--gamma", type=float, default=2.0,
                    help="focal-loss gamma (only used with --focal); grid: 0.5/1/2/3")
    ap.add_argument("--sample_strength", type=float, default=1.0,
                    help="inverse-freq sampling temper (only with --balanced_sampling): "
                         "0=uniform, 1=full inv-freq, >1 over-balance")
    ap.add_argument("--hardpair", action="store_true",
                    help="add the confusable-pair contrastive hinge (targets the #1 "
                         "bipolar<-cadiere spine confusion); default OFF = baseline loss")
    ap.add_argument("--hardpair_w", type=float, default=0.5,
                    help="weight on the hard-pair hinge (only used with --hardpair)")
    ap.add_argument("--hardpair_margin", type=float, default=2.0,
                    help="logit margin the present member must beat the absent by")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    rows = read_manifest(args.manifest, require_video=True)
    if not rows:
        raise SystemExit("No manifest rows with a local video yet. Re-run "
                         "build_frame_labels.py --require_video once videos land.")
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    train_rows = balance_other(train_rows, args.other_keep_frac)
    random.Random(0).shuffle(train_rows)
    random.Random(1).shuffle(val_rows)
    if args.smoke:
        train_rows, val_rows = train_rows[:512], val_rows[:256]
        args.epochs = 1
    else:
        train_rows = train_rows[:args.max_train]
        val_rows = val_rows[:args.max_val]
    print(f"device={device} train={len(train_rows)} val={len(val_rows)} "
          f"backbone={args.backbone} img={args.img_size}")

    # class-balanced pos_weight for the tool BCE (rare tools up-weighted)
    pos = np.zeros(NT)
    for r in train_rows:
        for j, c in enumerate(r["tools_multihot"]):
            pos[j] += (c == "1")
    neg = len(train_rows) - pos
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 50.0), dtype=torch.float32).to(device)

    tr_ds = FrameDS(train_rows, args.img_size, train=True)
    va_ds = FrameDS(val_rows, args.img_size, train=False)
    if args.balanced_sampling:
        from torch.utils.data import WeightedRandomSampler
        w = inverse_freq_sample_weights(train_rows, strength=args.sample_strength)
        sampler = WeightedRandomSampler(w, num_samples=len(train_rows), replacement=True)
        print(f"balanced sampling ON (inverse-freq^{args.sample_strength}, excl data-floor "
              f"tools {DATA_FLOOR_TOOLS}); weight range [{min(w):.2f},{max(w):.2f}]")
        tr = DataLoader(tr_ds, batch_size=args.bs, sampler=sampler, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    else:
        tr = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    va = DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                    pin_memory=True, persistent_workers=args.workers > 0)

    model = Spine(args.backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                steps_per_epoch=max(1, len(tr)), epochs=args.epochs)
    bce = (FocalBCE(gamma=args.gamma, pos_weight=pos_weight) if args.focal
           else nn.BCEWithLogitsLoss(pos_weight=pos_weight))
    print(f"tool loss = {f'FocalBCE(gamma={args.gamma})' if args.focal else 'BCEWithLogits'} "
          f"(pos_weight clipped<=50x)")
    hp_loss = None
    if args.hardpair:
        pair_idx = build_hardpair_index()
        hp_loss = HardPairLoss(pair_idx, margin=args.hardpair_margin)
        named = [f"{TOOLS[a]}^{TOOLS[b]}" for a, b in pair_idx]
        print(f"hard-pair hinge ON (w={args.hardpair_w}, margin={args.hardpair_margin}); "
              f"pairs={named}")
    ce = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best = -1
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        run = 0.0
        for it, (x, tools, task) in enumerate(tr):
            x = x.to(device, non_blocking=True)
            tools = tools.to(device, non_blocking=True)
            task = task.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                lt, lk = model(x)
                loss = bce(lt, tools) + ce(lk, task)
                if hp_loss is not None:
                    loss = loss + args.hardpair_w * hp_loss(lt, tools)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
            if it % 50 == 0:
                print(f"  ep{ep} it{it}/{len(tr)} loss={run/(it+1):.4f} "
                      f"({(it+1)*args.bs/(time.time()-t0):.1f} img/s)", flush=True)
        m = evaluate(model, va, device)
        print(f"[ep{ep}] tool_macroF1={m['tool_f1']:.4f}@thr{m['tool_thr']} "
              f"tool_mAP={m['tool_map']:.4f} task_macroF1={m['task_f1']:.4f} "
              f"task_acc={m['task_acc']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        score = (m["tool_f1"] + m["task_f1"]) / 2
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "metrics": m, "tools": TOOLS, "tasks": TASKS,
                        "tool_thr": m["tool_thr"]},
                       os.path.join(args.out_dir, "spine_best.pt"))
            print(f"  saved best (avg macroF1={score:.4f})", flush=True)
    print(f"DONE best_avg_macroF1={best:.4f}")


if __name__ == "__main__":
    main()
