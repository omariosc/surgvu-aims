#!/bin/bash
#SBATCH --job-name=surgvu_segfull
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/logs/segfull_%j.out
#SBATCH --error=/scratch/sc20osc/miccai-2026/SurgVU/logs/segfull_%j.err
# NO mail directives (self-monitored). FULL HRI_MV #1 seg-cascade (arXiv 2305.07152):
#   (a) seg_pseudoboxes 1500 frames  (b) track_propagate 300 windows x 8
#   (c) YOLO detector 60 ep on the SEG pseudo-boxes  (d) WBF detection+tracking fusion
# Chained afterok off the smoke. Substitution: Cascade-RCNN -> ultralytics YOLO (mmdet/mmcv
# has no prebuilt wheel for torch2.6+cu124, an offline-Docker CUDA-build nightmare; YOLO is
# the existing clean-licensed submission detector). All mAP = self-consistency vs the SEG
# pseudo-box GT (NO human bbox GT for 2024) -> the deliverable is the PIPELINE + pseudo-box
# QUALITY + the WBF delta; REAL validation awaits the gated 2026 bbox-val set.
# KILL CRITERION: seg pseudo-boxes NOT tighter/cleaner than Grad-CAM (area >=0.1538,
#   not single-instance) AND WBF_delta <= 0 -> the seg-cascade lever is REFUTED.
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
SEGDIR=$M/seg_vis
TRKDIR=$M/track_vis
DETOUT=$M/detector_seg
PARSE=$M/logs/segfull_${SLURM_JOB_ID}.parse
mkdir -p $(dirname $PARSE) $SEGDIR $TRKDIR $DETOUT

echo "=== node $(hostname) ==="; nvidia-smi -L
cd $EXP

echo "=== (a) seg pseudo-boxes (1500 frames) ==="
$PY $EXP/seg_pseudoboxes.py --ckpt $CKPT --manifest $D/frame_manifest_full.csv \
    --out_dir $SEGDIR --n 1500 2>&1 | tee $PARSE

echo "=== (b) tracking propagation (300 windows x 8 frames) ==="
$PY $EXP/track_propagate.py --ckpt $CKPT --manifest $D/frame_manifest_full.csv \
    --out_dir $TRKDIR --n_windows 300 --window 8 --stride_sec 0.5 2>&1 | tee -a $PARSE

echo "=== (c) YOLO detector on seg pseudo-boxes (60 ep, heavy aug) ==="
set +e
$PY $EXP/train_detector.py --pseudo $SEGDIR/seg_pseudo_boxes.json --img_dir $SEGDIR \
    --out_root $DETOUT --weights /scratch/sc20osc/AILET/from_pc/weights/yolo26m.pt \
    --epochs 60 --mosaic 1.0 --mixup 0.1 --hsv_h 0.015 --hsv_s 0.7 --hsv_v 0.4 \
    --degrees 5.0 --translate 0.1 --scale 0.5 --fliplr 0.5 2>&1 | tee -a $PARSE
set -e

echo "=== (d) WBF detection+tracking fusion ==="
set +e
$PY $EXP/wbf_fuse.py --weights $DETOUT/train/weights/best.pt --yolo_ds $DETOUT/yolo_ds \
    --ckpt $CKPT --out $M/wbf_fusion.json 2>&1 | tee -a $PARSE
set -e

MAP=$(grep -oE 'DETECTOR_mAP50_95=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
SEGBPF=$(grep -oE 'SEG_mean_boxes_per_frame=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
SEGAREA=$(grep -oE 'SEG_mean_box_area_frac=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
DETONLY=$(grep -oE 'DET_ONLY_mAP=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
WBFM=$(grep -oE 'WBF_FUSED_mAP=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
WBFD=$(grep -oE 'WBF_DELTA=[+-]?[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
[ -z "$MAP" ] && MAP="nan"; [ -z "$SEGAREA" ] && SEGAREA="nan"; [ -z "$WBFD" ] && WBFD="nan"
echo "[SurgVU][${SLURM_JOB_ID}] Cat-1 SEG-cascade FULL (seg->track->YOLO->WBF, 1500fr self-consistency, NO human GT; YOLO sub for Cascade-RCNN) -> seg-mAP=${MAP} (GradCAM-ref 0.7746) seg_boxes/fr=${SEGBPF} seg_box_area=${SEGAREA} vs GradCAM 0.1538/2.71 | WBF det-only=${DETONLY} fused=${WBFM} delta=${WBFD} (HRI_MV margin +0.009)" >> $RESULTS
echo "=== appended ==="; tail -1 $RESULTS
echo "=== FULL DONE $(date) ==="
