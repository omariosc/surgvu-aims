"""SurgVU 2026 Category 1 (surgical tool detection) — AIMS submission.

WHAT SHIPS: a YOLO detector trained on the 5,178 human-annotated COCO boxes (video-disjoint
split), predicting all FOURTEEN challenge tool classes. Bundled at /opt/app/resources/detector.pt.

WHY SUBMIT AT ALL, GIVEN A WEAK LOCAL NUMBER — this is a MEASUREMENT, not a bid:
  * Our honest local mAP is 0.1828 on a 4-class val subset (the only classes with val boxes),
    against an organiser baseline of 0.5355 and a field top of 0.6040. We expect to score low.
  * But SIX of the fourteen classes have ZERO boxes anywhere in the annotated data, so no local
    split can score them at all. The prelim leaderboard is the ONLY instrument that can.
  * The Cat-1 prelim leaderboard is READABLE (unlike Cat-2's, which 403s), and 10 submissions
    remain, so the cost of finding out is ~zero.

⚠ ONE CONTRACT DETAIL IS GENUINELY UNRESOLVED AND IS THE MAIN RISK: what `slice_nr_<N>` indexes.
  The organisers' template shows a single example box named `slice_nr_0_needle_driver` and the
  README does not define the convention. We take the LITERAL reading -- N is the frame index
  within the clip -- and emit a detection set for EVERY decoded frame. A 30 s 60 fps clip is
  ~1800 frames, which YOLO clears well inside the 600 s/case budget, so emitting all frames is
  cheaper than guessing a sampling rate. If the evaluator expects a different indexing this
  submission will score ~0, and THAT IS ITSELF THE INFORMATION we are buying.

Box format is copied exactly from the template: 4 corners, each [x, y, 0.5], plus a probability.
"""
from pathlib import Path
import json
import os

INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/input"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/output"))
RESOURCE_PATH = Path(os.environ.get("RESOURCE_PATH", "/opt/app/resources"))

# MEASURED, not guessed (sweep 7334173 on the human-COCO val, paired):
#   conf 0.05 / max_det 10  -> mAP50-95 0.1686   <- what v1 shipped
#   conf 0.001 / max_det 100 -> mAP50-95 0.1828  <- +0.0142 (+8.4% relative)
#   conf 0.01  / max_det 30  -> mAP50-95 0.1754
#   coco-std + TTA           -> mAP50-95 0.1828  <- TTA adds NOTHING, and costs ~3x runtime
# mAP integrates precision across the FULL recall range, so suppressing low-confidence boxes
# caps recall that can never be recovered. v1's values were set by guess and cost 8.4%.
CONF = float(os.environ.get("SURGVU_CONF", "0.001"))
MAX_DET = int(os.environ.get("SURGVU_MAX_DET", "100"))
IMGSZ = int(os.environ.get("SURGVU_IMGSZ", "640"))


def log(*a): print("[cat1]", *a, flush=True)


def run():
    interface_key = get_interface_key()
    log("Inputs:", interface_key)
    return {("endoscopic-robotic-surgery-video",): interf0_handler}[interface_key]()


def interf0_handler():
    video = INPUT_PATH / "endoscopic-robotic-surgery-video.mp4"
    boxes = []
    try:
        boxes = detect(video)
    except Exception:
        import traceback; traceback.print_exc()
        # NEVER emit a malformed file: an unreadable output is an INVALID, which is strictly
        # worse than an empty-but-valid one. Ship zero boxes rather than nothing.
        log("detection failed; writing a well-formed EMPTY prediction")
    out = {"name": "surgical-tools",
           "type": "Multiple 2D bounding boxes",
           "boxes": boxes,
           "version": {"major": 1, "minor": 0}}
    write_json_file(location=OUTPUT_PATH / "surgical-tools.json", content=out)
    log("wrote %d boxes -> %s" % (len(boxes), OUTPUT_PATH / "surgical-tools.json"))
    return 0


def detect(video_path):
    import cv2
    from ultralytics import YOLO
    wt = RESOURCE_PATH / "detector.pt"
    log("loading detector:", wt, "exists:", wt.is_file())
    model = YOLO(str(wt))
    names = model.names
    log("classes:", names)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("cannot open %s" % video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    log("frames reported:", n, "fps:", cap.get(cv2.CAP_PROP_FPS))

    boxes, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=IMGSZ, conf=CONF, max_det=MAX_DET, verbose=False)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            cls = names[int(b.cls[0])]
            nm = "slice_nr_%d_%s" % (idx, cls)
            boxes.append({
                "name": nm,
                "corners": [[x1, y1, 0.5], [x2, y1, 0.5], [x2, y2, 0.5], [x1, y2, 0.5]],
                "probability": float(b.conf[0]),
            })
        idx += 1
    cap.release()
    log("decoded %d frames, produced %d boxes" % (idx, len(boxes)))
    return boxes


def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    return tuple(sorted(sv["interface"]["slug"] for sv in inputs))


def load_json_file(*, location):
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    Path(location).parent.mkdir(parents=True, exist_ok=True)
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


if __name__ == "__main__":
    raise SystemExit(run())
