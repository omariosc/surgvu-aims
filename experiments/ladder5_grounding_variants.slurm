#!/bin/bash
#SBATCH --job-name=surgvu_l5gv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder5_%j.out
#SBATCH --error=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder5_%j.err
# NO mail directives (self-monitored). NEW LEVER A (brief lever 3) -- Cat-2 grounding
# VARIANTS sweep: n_frames x presence-thr x pooling, holding the solved template router
# fixed, to find the entity-injection/temporal-grounding config that maximises Cat-2 BLEU
# on the 11 in-hand clips. Targets the #1 diagnostic bottleneck (Cat-2 vision grounding).
# Chained afterok off ladder2 (6015398) which confirms the spine_full ckpt exists.
# KILL CRITERION: best grounded-BLEU does not exceed the single-config ladder2 baseline
#   -> grounding-config is saturated; the gap is spine ACCURACY, not the injection recipe.
set -e
mkdir -p /scratch/sc20osc/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export OMP_NUM_THREADS=4
PY=/scratch/sc20osc/surgvu_env/bin/python
EXP=/users/sc20osc/SurgVU/experiments
CKPT=/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/spine_best.pt
RESULTS=/users/sc20osc/AUTONOMOUS_RESULTS.md
PARSE=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder5_${SLURM_JOB_ID}.parse

echo "=== node $(hostname) ==="; nvidia-smi -L
cd $EXP
$PY $EXP/grounding_variants_bleu.py --ckpt $CKPT 2>&1 | tee $PARSE

BEST=$(grep -E '^GROUNDVAR_BEST' $PARSE | tail -1 | sed 's/^GROUNDVAR_BEST //')
echo "[SurgVU][${SLURM_JOB_ID}] Cat-2 grounding-variant sweep (n_frames x thr x pool, n=11) -> ${BEST:-see log}" >> $RESULTS
echo "=== appended to $RESULTS ==="; tail -1 $RESULTS
echo "=== DONE $(date) ==="
