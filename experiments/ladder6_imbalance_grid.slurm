#!/bin/bash
#SBATCH --job-name=surgvu_l6grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder6_%A_%a.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder6_%A_%a.err
# NO mail directives (self-monitored). NEW LEVER B (brief lever 1) -- class-imbalance
# GRID: focal-gamma x sample-strength, each array task = 1 cell. Targets the recoverable
# (excl 3 data-floor tools) macro-F1 the diagnostic flags as the rare-tool bottleneck.
# 6 cells: gamma in {1,2} x strength in {0.5,1.0,1.5}. Shorter schedule (5 ep / 150k
# train frames) -- goal is the RELATIVE ranking of the imbalance knobs, not a final ckpt.
# Each cell trains -> diagnoses -> writes grid1/<cell>/per_class_diag.json. The aggregator
# job (ladder6_agg, afterok on this whole array) ranks them + picks the best.
set -e
mkdir -p /scratch/USERNAME/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/USERNAME/hf_cache
export TORCH_HOME=/scratch/USERNAME/hf_cache/torch
export OMP_NUM_THREADS=4
PY=/scratch/USERNAME/surgvu_env/bin/python
EXP=/users/USERNAME/SurgVU/experiments
D=/scratch/USERNAME/miccai-2026/SurgVU/data
MAN=$D/frame_manifest_full.csv

GAMMAS=(1 2)
STRENGTHS=(0.5 1.0 1.5)
NG=${#GAMMAS[@]}; NS=${#STRENGTHS[@]}
GI=$(( SLURM_ARRAY_TASK_ID / NS ))
SI=$(( SLURM_ARRAY_TASK_ID % NS ))
GAMMA=${GAMMAS[$GI]}
STRENGTH=${STRENGTHS[$SI]}
CELL="g${GAMMA}_s${STRENGTH}"
OUT=/scratch/USERNAME/miccai-2026/SurgVU/models/grid1/$CELL

echo "=== node $(hostname)  cell=$CELL (gamma=$GAMMA strength=$STRENGTH) ==="; nvidia-smi -L
cd $EXP
mkdir -p $OUT

echo "=== SMOKE ==="
$PY $EXP/train_spine.py --manifest $MAN --smoke --workers 6 --bs 32 --img_size 384 \
    --balanced_sampling --sample_strength $STRENGTH --focal --gamma $GAMMA --out_dir $OUT

echo "=== TRAIN cell $CELL ==="
$PY $EXP/train_spine.py --manifest $MAN --epochs 5 --bs 64 --img_size 384 \
    --workers 12 --max_train 150000 --max_val 40000 \
    --balanced_sampling --sample_strength $STRENGTH --focal --gamma $GAMMA --out_dir $OUT

echo "=== DIAGNOSE cell $CELL ==="
$PY $EXP/diagnose_spine.py --ckpt $OUT/spine_best.pt --manifest $MAN --max_val 40000
echo "=== cell $CELL DONE $(date) ==="
