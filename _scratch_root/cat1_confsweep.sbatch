#!/bin/bash
#SBATCH --job-name=cat1_sweep
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/cat1_sweep_%j.log
set -uo pipefail
echo "### Cat-1 inference-config sweep  $(date)"
echo "### PAIRED comparison on the human-COCO val. ⚠ That val has boxes for only 4 of 14 classes"
echo "### (bipolar_forceps 1594, monopolar_curved_scissor 1410, clip_applier 258, vessel_sealer 13;"
echo "### needle_driver -- 97% of our PLATFORM detections -- has ZERO), so the ABSOLUTE number is a"
echo "### biased estimator (local 0.1828 vs platform 0.4019). The ORDERING should still transfer"
echo "### because every arm carries the identical bias."
/scratch/sc20osc/surgvu_env/bin/python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
from ultralytics import YOLO
W="/scratch/sc20osc/miccai-2026/SurgVU/models/cat1_ssl/FILTER_main_t25_s0/weights/best.pt"
DATA="/scratch/sc20osc/miccai-2026/SurgVU/data/cat1_ssl/arms/FILTER_main_t25_s0/data.yaml"
# mAP integrates precision across the FULL recall range, so suppressing low-confidence boxes
# caps achievable recall. Standard detection eval uses conf~0.001 / max_det=100 for exactly
# this reason. A = what we SHIPPED; the rest test whether that choice cost us.
ARMS=[("A SHIPPED conf.05 det10",   dict(conf=0.05,  max_det=10)),
      ("B coco-std conf.001 det100",dict(conf=0.001, max_det=100)),
      ("C mid conf.01 det30",       dict(conf=0.01,  max_det=30)),
      ("D coco-std + TTA",          dict(conf=0.001, max_det=100, augment=True))]
print("%-30s %9s %9s %9s" % ("arm","mAP50-95","mAP50","mAR100"))
res={}
for name,kw in ARMS:
    m=YOLO(W)
    r=m.val(data=DATA, imgsz=640, device="cpu", verbose=False, plots=False,
            save_json=False, workers=8, **kw)
    b=r.box
    res[name]=(float(b.map), float(b.map50), float(b.mar100) if hasattr(b,"mar100") else float("nan"))
    print("%-30s %9.4f %9.4f %9.4f" % (name, *res[name]), flush=True)
print()
base=res["A SHIPPED conf.05 det10"][0]
print("DELTA vs the SHIPPED arm (mAP50-95):")
for n,(mp,_,_) in res.items():
    if n.startswith("A"): continue
    print("   %-30s %+0.4f  (%.1f%% relative)" % (n, mp-base, 100*(mp-base)/max(base,1e-9)))
PY
echo "### done $(date)"
