"""Container-side TASK predictor for Cat-2 (CPU, ~4 frames, seconds).

Design rule: this must DEGRADE GRACEFULLY to the already-shipped unconditional prior, which
scores 0.7546. Any failure -- missing checkpoint, unreadable video, torch import error -- returns
None and the caller falls back. A new path must never be able to regress a banked result.
"""
import os

def predict_task(video_path, ckpt_path, k=4, size=224):
    """-> canonical task string, or None if anything at all goes wrong."""
    try:
        import numpy as np, cv2, torch, torch.nn as nn
        import torchvision as tv
        if not (os.path.exists(ckpt_path) and os.path.exists(video_path)):
            return None
        ck = torch.load(ckpt_path, map_location="cpu")
        tasks = ck["tasks"]
        m = tv.models.resnet50()
        m.fc = nn.Linear(m.fc.in_features, len(tasks))
        m.load_state_dict(ck["state"]); m.eval()

        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n <= 0:
            cap.release(); return None
        idx = np.linspace(0, max(n - 1, 0), k + 2)[1:-1].astype(int)
        tf = tv.transforms.Compose([
            tv.transforms.Resize(int(size * 1.14)), tv.transforms.CenterCrop(size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        ims = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                continue
            ims.append(tf(tv.transforms.functional.to_pil_image(
                cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
        cap.release()
        if not ims:
            return None
        with torch.no_grad():
            logits = m(torch.stack(ims)).mean(0)
        return tasks[int(logits.argmax())]
    except Exception as exc:
        print("task head unavailable (%r); using the unconditional prior" % (exc,))
        return None
