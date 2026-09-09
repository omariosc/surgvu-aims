#!/bin/bash
#SBATCH --job-name=surgvu_l9thr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder9_%A_%a.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder9_%A_%a.err
# LEVER 1 (push the winning axis: HIGHER Grad-CAM threshold). ladder8 showed CAM
# thr 0.3->0.5 monotonically improved mAP at yolo26m (0.6869 -> 0.7746). This sweeps
# thr in {0.6, 0.7} at yolo26m (the winning detector) to test whether even cleaner /
# tighter pseudo-boxes keep climbing or whether boxes get too SPARSE (yield drops).
# Each cell: cam_pseudoboxes @ its thr (fresh dir) -> train_detector yolo26m 40 ep ->
# COCO mAP@[.5:.95] on the case-disjoint pseudo-box val (self-consistency, NO human GT).
# KILL CRITERION: if 0.6/0.7 mAP < the thr0.5 baseline 0.7746 (boxes too sparse) ->
#   log REFUTED and confirm thr0.5 as the pseudo-box-threshold optimum.
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

THRS=(0.6 0.7)
THR=${THRS[$SLURM_ARRAY_TASK_ID]}
YOLO=/scratch/USERNAME/AILET/from_pc/weights/yolo26m.pt
YTAG=$(basename $YOLO .pt)
CELL="thr${THR}_${YTAG}"
CAM=/scratch/USERNAME/miccai-2026/SurgVU/models/cam_thr${THR}
OUT=/scratch/USERNAME/miccai-2026/SurgVU/models/detector_$CELL
PARSE=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder9_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.parse

echo "=== node $(hostname)  cell=$CELL (thr=$THR yolo=$YTAG) ==="; nvidia-smi -L
cd $EXP
mkdir -p $OUT

echo "=== (a) Grad-CAM pseudo-boxes @ thr=$THR (600 frames) ==="
if [ ! -f $CAM/pseudo_boxes.json ]; then
  $PY $EXP/cam_pseudoboxes.py --ckpt $CKPT --manifest $D/frame_manifest_full.csv \
      --out_dir $CAM --n 600 --thr $THR 2>&1 | tee $PARSE
else
  echo "reusing existing $CAM/pseudo_boxes.json"; touch $PARSE
fi

echo "=== (b) train YOLO ($YTAG) on thr=$THR pseudo-boxes (40 ep, ladder8-default) ==="
set +e
$PY $EXP/train_detector.py --pseudo $CAM/pseudo_boxes.json --img_dir $CAM \
    --out_root $OUT --weights $YOLO --epochs 40 2>&1 | tee -a $PARSE
set -e

MAP=$(grep -oE 'DETECTOR_mAP50_95=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
MAP50=$(grep -oE 'DETECTOR_mAP50=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
NBOX=$(grep -oE '[0-9]+ pseudo-boxes' $PARSE | tail -1 | grep -oE '^[0-9]+')
[ -z "$MAP" ] && MAP="nan"; [ -z "$MAP50" ] && MAP50="nan"
# verdict vs the thr0.5_yolo26m baseline 0.7746
VERD=""
if [ "$MAP" != "nan" ]; then
  CMP=$($PY -c "print('CONFIRMED' if $MAP > 0.7746 else 'REFUTED')" 2>/dev/null)
  VERD=" [$CMP vs thr0.5 baseline 0.7746]"
fi
echo "[SurgVU][${SLURM_JOB_ID}] Cat-1 detector-variant ${CELL} (${NBOX:-?} pseudo-boxes, case-disjoint self-consistency, NO human GT) -> mAP@[.5:.95]=${MAP} mAP50=${MAP50}${VERD}" >> $RESULTS
echo "=== appended to $RESULTS ==="; tail -1 $RESULTS
echo "=== $CELL DONE $(date) ==="
