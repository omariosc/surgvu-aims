#!/bin/bash
#SBATCH --job-name=surgvu_l8det
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder8_%A_%a.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder8_%A_%a.err
# NO mail directives (self-monitored). NEW LEVER D (brief lever 4) -- Cat-1 detector
# VARIANTS: pseudo-box confidence-threshold x YOLO size. Array of 4 cells:
#   thr in {0.3,0.5} (Grad-CAM CAM threshold for pseudo-box extraction)
#   yolo in {yolo11n (cached nano), yolo26m (cached medium)}  -- both cached, NO download.
# Each cell: cam_pseudoboxes at its thr -> train_detector with its YOLO weights ->
# COCO mAP@[.5:.95] on the case-disjoint pseudo-box val (self-consistency, NO human GT).
# Chained afterok off ladder3 (6015399) so the spine_full ckpt + base pseudo-boxes exist.
# KILL CRITERION: no cell's mAP exceeds the ladder3 single-config baseline meaningfully
#   -> localisation supervision (Grad-CAM pseudo-boxes) is the wall, not detector capacity.
set -e
mkdir -p /scratch/USERNAME/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/USERNAME/hf_cache
export TORCH_HOME=/scratch/USERNAME/hf_cache/torch
export OMP_NUM_THREADS=4
export YOLO_CONFIG_DIR=/scratch/USERNAME/hf_cache/ultralytics
PY=/scratch/USERNAME/surgvu_env/bin/python
EXP=/users/USERNAME/SurgVU/experiments
D=/scratch/USERNAME/miccai-2026/SurgVU/data
CKPT=/scratch/USERNAME/miccai-2026/SurgVU/models/spine_full/spine_best.pt
RESULTS=/users/USERNAME/AUTONOMOUS_RESULTS.md

THRS=(0.3 0.5)
YOLOS=(/scratch/USERNAME/AILET/from_pc/weights/yolo11n.pt /scratch/USERNAME/AILET/from_pc/weights/yolo26m.pt)
NY=${#YOLOS[@]}
TI=$(( SLURM_ARRAY_TASK_ID / NY ))
YI=$(( SLURM_ARRAY_TASK_ID % NY ))
THR=${THRS[$TI]}
YOLO=${YOLOS[$YI]}
YTAG=$(basename $YOLO .pt)
CELL="thr${THR}_${YTAG}"
CAM=/scratch/USERNAME/miccai-2026/SurgVU/models/cam_thr${THR}
OUT=/scratch/USERNAME/miccai-2026/SurgVU/models/detector_$CELL
PARSE=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder8_${SLURM_JOB_ID}.parse

echo "=== node $(hostname)  cell=$CELL (thr=$THR yolo=$YTAG) ==="; nvidia-smi -L
cd $EXP
mkdir -p $OUT

echo "=== (a) Grad-CAM pseudo-boxes @ thr=$THR (600 frames) ==="
# regenerate per-thr (idempotent); each thr dir reused if both yolo cells share a node order
if [ ! -f $CAM/pseudo_boxes.json ]; then
  $PY $EXP/cam_pseudoboxes.py --ckpt $CKPT --manifest $D/frame_manifest_full.csv \
      --out_dir $CAM --n 600 --thr $THR 2>&1 | tee $PARSE
else
  echo "reusing existing $CAM/pseudo_boxes.json"; touch $PARSE
fi

echo "=== (b) train YOLO ($YTAG) on thr=$THR pseudo-boxes ==="
set +e
$PY $EXP/train_detector.py --pseudo $CAM/pseudo_boxes.json --img_dir $CAM \
    --out_root $OUT --weights $YOLO --epochs 40 2>&1 | tee -a $PARSE
set -e

MAP=$(grep -oE 'DETECTOR_mAP50_95=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
MAP50=$(grep -oE 'DETECTOR_mAP50=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
NBOX=$(grep -oE '[0-9]+ pseudo-boxes' $PARSE | tail -1 | grep -oE '^[0-9]+')
[ -z "$MAP" ] && MAP="nan"; [ -z "$MAP50" ] && MAP50="nan"
echo "[SurgVU][${SLURM_JOB_ID}] Cat-1 detector-variant ${CELL} (${NBOX:-?} pseudo-boxes, case-disjoint self-consistency, NO human GT) -> mAP@[.5:.95]=${MAP} mAP50=${MAP50}" >> $RESULTS
echo "=== appended to $RESULTS ==="; tail -1 $RESULTS
echo "=== $CELL DONE $(date) ==="
