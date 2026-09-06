#!/bin/bash
#SBATCH --job-name=surgvu_sam2full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/logs/sam2full_%j.out
#SBATCH --error=/scratch/sc20osc/miccai-2026/SurgVU/logs/sam2full_%j.err
# NO mail directives (self-monitored). FULL SAM2-prompted HRI_MV #1 seg-cascade
# (arXiv 2305.07152): (a) seg_pseudoboxes WITH SAM2 1500 frames
#   (b) track_propagate 300 windows x 8  (c) YOLO detector 60 ep on the SAM2 pseudo-boxes
#   (d) WBF detection+tracking fusion. Chained afterok off the SAM2 smoke.
# SAM2 (facebook/sam2.1-hiera-large, Apache-2.0) prompted from saliency peaks replaces the
# generic-COCO yolo-seg substitute (= the lever HRI_MV's SurgToolLoc winners used).
# Substitution kept: Cascade-RCNN -> ultralytics YOLO (no mmcv wheel for torch2.6+cu124).
# All mAP = self-consistency vs the SAM2 pseudo-box GT (NO human bbox GT for 2024) ->
# deliverable = PIPELINE + pseudo-box QUALITY + the WBF delta; REAL validation awaits the
# gated 2026 bbox-val set.
# KILL CRITERION: SAM2 pseudo-boxes NOT tighter than Grad-CAM (area >= 0.1538) AND
#   WBF_delta <= 0 -> the SAM2 seg-cascade lever is REFUTED.
# INODE-SAFE: extracted/temp artifacts routed to /tmp (cleaned by the EXIT trap below);
# only the bounded cam_*.png set (1500) is written to /scratch out_dirs, as before.
set -e
mkdir -p /scratch/sc20osc/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export OMP_NUM_THREADS=4
export YOLO_CONFIG_DIR=/tmp/ultralytics_${SLURM_JOB_ID}
mkdir -p /tmp/surgvu_sam_${SLURM_JOB_ID} /tmp/ultralytics_${SLURM_JOB_ID}
trap 'rm -rf /tmp/surgvu_sam_${SLURM_JOB_ID} /tmp/ultralytics_${SLURM_JOB_ID}' EXIT
PY=/scratch/sc20osc/surgvu_env/bin/python
EXP=/users/sc20osc/SurgVU/experiments
D=/scratch/sc20osc/miccai-2026/SurgVU/data
CKPT=/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/spine_best.pt
M=/scratch/sc20osc/miccai-2026/SurgVU/models
SAMW=/scratch/sc20osc/miccai-2026/SurgVU/models/sam_weights/sam2.1_l.pt
RESULTS=/users/sc20osc/AUTONOMOUS_RESULTS.md
SEGDIR=$M/seg_vis_sam2_full
TRKDIR=$M/track_vis_sam2_full
DETOUT=$M/detector_sam2_full
PARSE=$M/logs/sam2full_${SLURM_JOB_ID}.parse
mkdir -p $(dirname $PARSE) $SEGDIR $TRKDIR $DETOUT

echo "=== node $(hostname) ==="; nvidia-smi -L
cd $EXP

echo "=== (a) SAM2-prompted seg pseudo-boxes (1500 frames) ==="
$PY $EXP/seg_pseudoboxes.py --ckpt $CKPT --manifest $D/frame_manifest_full.csv \
    --out_dir $SEGDIR --sam_weights $SAMW --n 1500 2>&1 | tee $PARSE

echo "=== (b) tracking propagation (300 windows x 8 frames) ==="
$PY $EXP/track_propagate.py --ckpt $CKPT --manifest $D/frame_manifest_full.csv \
    --out_dir $TRKDIR --n_windows 300 --window 8 --stride_sec 0.5 2>&1 | tee -a $PARSE

echo "=== (c) YOLO detector on SAM2 seg pseudo-boxes (60 ep, heavy aug) ==="
set +e
$PY $EXP/train_detector.py --pseudo $SEGDIR/seg_pseudo_boxes.json --img_dir $SEGDIR \
    --out_root $DETOUT --weights /scratch/sc20osc/AILET/from_pc/weights/yolo26m.pt \
    --epochs 60 --mosaic 1.0 --mixup 0.1 --hsv_h 0.015 --hsv_s 0.7 --hsv_v 0.4 \
    --degrees 5.0 --translate 0.1 --scale 0.5 --fliplr 0.5 2>&1 | tee -a $PARSE
set -e

echo "=== (d) WBF detection+tracking fusion ==="
set +e
$PY $EXP/wbf_fuse.py --weights $DETOUT/train/weights/best.pt --yolo_ds $DETOUT/yolo_ds \
    --ckpt $CKPT --out $M/wbf_fusion_sam2.json 2>&1 | tee -a $PARSE
set -e

MAP=$(grep -oE 'DETECTOR_mAP50_95=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
SEGBPF=$(grep -oE 'SEG_mean_boxes_per_frame=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
SEGAREA=$(grep -oE 'SEG_mean_box_area_frac=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
DETONLY=$(grep -oE 'DET_ONLY_mAP=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
WBFM=$(grep -oE 'WBF_FUSED_mAP=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
WBFD=$(grep -oE 'WBF_DELTA=[+-]?[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
[ -z "$MAP" ] && MAP="nan"; [ -z "$SEGAREA" ] && SEGAREA="nan"; [ -z "$WBFD" ] && WBFD="nan"
echo "[SurgVU][${SLURM_JOB_ID}] Cat-1 SAM2 seg-cascade FULL (SAM2-prompted seg->track->YOLO->WBF, 1500fr self-consistency, NO human GT; YOLO sub for Cascade-RCNN) -> seg-mAP=${MAP} (GradCAM-ref 0.7746) seg_boxes/fr=${SEGBPF} seg_box_area=${SEGAREA} vs GradCAM 0.1538/genericCOCO 0.2180 | WBF det-only=${DETONLY} fused=${WBFM} delta=${WBFD} (HRI_MV margin +0.009)" >> $RESULTS
echo "=== appended ==="; tail -1 $RESULTS
echo "=== SAM2 FULL DONE $(date) ==="
