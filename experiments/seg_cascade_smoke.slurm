#!/bin/bash
#SBATCH --job-name=surgvu_segsmoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/logs/segsmoke_%j.out
#SBATCH --error=/scratch/sc20osc/miccai-2026/SurgVU/logs/segsmoke_%j.err
# NO mail directives (self-monitored). SMOKE of the HRI_MV #1 seg-cascade (arXiv 2305.07152):
#   (a) seg_pseudoboxes  (b) track_propagate  (c) YOLO detector  (d) WBF fusion
# Small scale: 200 seg frames + 20 track windows + 12-ep YOLO -> proves the cascade runs
# on a gpu node before the full job. Substitution: Cascade-RCNN -> ultralytics YOLO
# (mmdet/mmcv has no prebuilt wheel for torch2.6+cu124; YOLO is the existing clean-licensed
# submission detector). All mAP = self-consistency vs the SEG pseudo-box GT (NO human bbox GT).
set -e
mkdir -p /scratch/sc20osc/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export OMP_NUM_THREADS=4
export YOLO_CONFIG_DIR=/scratch/sc20osc/hf_cache/ultralytics
PY=/scratch/sc20osc/surgvu_env/bin/python
EXP=/users/sc20osc/SurgVU/experiments
D=/scratch/sc20osc/miccai-2026/SurgVU/data
CKPT=/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/spine_best.pt
M=/scratch/sc20osc/miccai-2026/SurgVU/models
RESULTS=/users/sc20osc/AUTONOMOUS_RESULTS.md
SEGDIR=$M/seg_vis_smoke
TRKDIR=$M/track_vis_smoke
DETOUT=$M/detector_seg_smoke
PARSE=$M/logs/segsmoke_${SLURM_JOB_ID}.parse
mkdir -p $(dirname $PARSE) $SEGDIR $TRKDIR $DETOUT

echo "=== node $(hostname) ==="; nvidia-smi -L
cd $EXP

echo "=== (a) seg pseudo-boxes (200 frames) ==="
$PY $EXP/seg_pseudoboxes.py --ckpt $CKPT --manifest $D/frame_manifest_full.csv \
    --out_dir $SEGDIR --n 200 2>&1 | tee $PARSE

echo "=== (b) tracking propagation (20 windows x 8 frames) ==="
$PY $EXP/track_propagate.py --ckpt $CKPT --manifest $D/frame_manifest_full.csv \
    --out_dir $TRKDIR --n_windows 20 --window 8 --stride_sec 0.5 2>&1 | tee -a $PARSE

echo "=== (c) YOLO detector on seg pseudo-boxes (12 ep) ==="
set +e
$PY $EXP/train_detector.py --pseudo $SEGDIR/seg_pseudo_boxes.json --img_dir $SEGDIR \
    --out_root $DETOUT --weights /scratch/sc20osc/AILET/from_pc/weights/yolo26m.pt \
    --epochs 12 2>&1 | tee -a $PARSE
set -e

echo "=== (d) WBF detection+tracking fusion ==="
set +e
$PY $EXP/wbf_fuse.py --weights $DETOUT/train/weights/best.pt --yolo_ds $DETOUT/yolo_ds \
    --ckpt $CKPT --out $M/wbf_fusion_smoke.json 2>&1 | tee -a $PARSE
set -e

MAP=$(grep -oE 'DETECTOR_mAP50_95=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
SEGAREA=$(grep -oE 'SEG_mean_box_area_frac=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
WBFD=$(grep -oE 'WBF_DELTA=[+-]?[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
[ -z "$MAP" ] && MAP="nan"; [ -z "$SEGAREA" ] && SEGAREA="nan"; [ -z "$WBFD" ] && WBFD="nan"
echo "[SurgVU][${SLURM_JOB_ID}] Cat-1 SEG-cascade SMOKE (seg->track->YOLO->WBF, 200fr self-consistency, NO human GT; YOLO sub for Cascade-RCNN) -> seg-mAP=${MAP} seg_box_area=${SEGAREA} (GradCAM 0.1538) WBF_delta=${WBFD}" >> $RESULTS
echo "=== appended ==="; tail -1 $RESULTS
echo "=== SMOKE DONE $(date) ==="
