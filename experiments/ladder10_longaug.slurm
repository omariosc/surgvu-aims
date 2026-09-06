#!/bin/bash
#SBATCH --job-name=surgvu_l10aug
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder10_%j.out
#SBATCH --error=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder10_%j.err
# LEVER 2 (longer schedule + heavier augmentation, at the best existing CAM thr 0.5,
# yolo26m = best plain-detect cached). ladder8 trends were both monotone up (cleaner
# boxes + bigger YOLO) => detector not saturated. yolo26m is the LARGEST plain-detect
# cached weight, so the "bigger" axis here is LONGER schedule (120 ep vs ladder8's 40)
# + STRONGER augmentation (mosaic/mixup/hsv/degrees/translate/scale). Reuses the
# existing cam_thr0.5/ pseudo-boxes (1599 boxes, identical supervision to the 0.7746
# baseline) so this isolates the detector schedule/aug axis ONLY.
# KILL CRITERION: if longer+aug mAP <= the thr0.5_yolo26m baseline 0.7746 -> log
#   REFUTED: detector capacity/schedule is NOT the wall, localisation supervision
#   (Grad-CAM pseudo-boxes) is.
set -e
mkdir -p /scratch/sc20osc/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export OMP_NUM_THREADS=4
export YOLO_CONFIG_DIR=/scratch/sc20osc/hf_cache/ultralytics
PY=/scratch/sc20osc/surgvu_env/bin/python
EXP=/users/sc20osc/SurgVU/experiments
RESULTS=/users/sc20osc/AUTONOMOUS_RESULTS.md

THR=0.5
YOLO=/scratch/sc20osc/AILET/from_pc/weights/yolo26m.pt
YTAG=$(basename $YOLO .pt)
CELL="thr${THR}_${YTAG}_long120aug"
CAM=/scratch/sc20osc/miccai-2026/SurgVU/models/cam_thr${THR}
OUT=/scratch/sc20osc/miccai-2026/SurgVU/models/detector_$CELL
PARSE=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder10_${SLURM_JOB_ID}.parse

echo "=== node $(hostname)  cell=$CELL ==="; nvidia-smi -L
cd $EXP
mkdir -p $OUT

if [ ! -f $CAM/pseudo_boxes.json ]; then
  echo "ERROR: $CAM/pseudo_boxes.json missing (expected the existing thr0.5 cam dir)"; exit 1
fi
echo "=== reusing existing $CAM/pseudo_boxes.json (thr0.5, 1599 boxes) ==="

echo "=== train YOLO ($YTAG) 120 ep + heavy aug ==="
set +e
$PY $EXP/train_detector.py --pseudo $CAM/pseudo_boxes.json --img_dir $CAM \
    --out_root $OUT --weights $YOLO --epochs 120 --patience 40 \
    --mosaic 1.0 --mixup 0.15 --hsv_h 0.015 --hsv_s 0.7 --hsv_v 0.4 \
    --degrees 10.0 --translate 0.1 --scale 0.5 --fliplr 0.5 2>&1 | tee -a $PARSE
set -e

MAP=$(grep -oE 'DETECTOR_mAP50_95=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
MAP50=$(grep -oE 'DETECTOR_mAP50=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
NBOX=$(grep -oE '[0-9]+ pseudo-boxes' $PARSE | tail -1 | grep -oE '^[0-9]+')
[ -z "$MAP" ] && MAP="nan"; [ -z "$MAP50" ] && MAP50="nan"
VERD=""
if [ "$MAP" != "nan" ]; then
  CMP=$($PY -c "print('CONFIRMED' if $MAP > 0.7746 else 'REFUTED')" 2>/dev/null)
  VERD=" [$CMP vs thr0.5_yolo26m baseline 0.7746]"
fi
echo "[SurgVU][${SLURM_JOB_ID}] Cat-1 detector-variant ${CELL} (${NBOX:-1599} pseudo-boxes, case-disjoint self-consistency, NO human GT) -> mAP@[.5:.95]=${MAP} mAP50=${MAP50}${VERD}" >> $RESULTS
echo "=== appended to $RESULTS ==="; tail -1 $RESULTS
echo "=== $CELL DONE $(date) ==="
