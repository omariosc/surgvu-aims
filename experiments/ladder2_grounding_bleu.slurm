#!/bin/bash
#SBATCH --job-name=surgvu_l2gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder2_%j.out
#SBATCH --error=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder2_%j.err
# NO mail directives (self-monitored). LADDER STEP 2 -- Cat-2 REAL visual grounding.
# Runs the trained spine over the 11 in-hand Cat-2 clips, injects its REAL tool/task
# predictions into the P0.2 template router, and re-scores SurgVU Cat-2 BLEU vs the
# 0.4215 SOTA / 0.9650 hand-set ceiling. Chained afterok off the full-scale spine.
set -e
mkdir -p /scratch/sc20osc/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export OMP_NUM_THREADS=4
PY=/scratch/sc20osc/surgvu_env/bin/python
EXP=/users/sc20osc/SurgVU/experiments
CKPT=/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/spine_best.pt
RESULTS=/users/sc20osc/AUTONOMOUS_RESULTS.md

echo "=== node $(hostname) ==="; nvidia-smi -L
cd $EXP
PARSE=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder2_${SLURM_JOB_ID}.parse
$PY $EXP/spine_grounding_bleu.py --ckpt $CKPT 2>&1 | tee $PARSE

# parse the machine-readable last lines from the captured stdout
BLEU=$(grep -oE 'GROUNDED_BLEU=[0-9.]+' $PARSE | tail -1 | cut -d= -f2)
VERD=$(grep -E '^# VERDICT:' $PARSE | tail -1 | sed 's/^# VERDICT: //')
[ -z "$BLEU" ] && BLEU="nan"
if awk "BEGIN{exit !($BLEU >= 0.4215)}" 2>/dev/null; then
  V="CONFIRMED >=0.4215 SOTA bar"
else
  V="below 0.4215 -- ${VERD:-see log}"
fi
echo "[SurgVU][${SLURM_JOB_ID}] Cat-2 REAL-grounded BLEU (n=11, spine_full) -> ${BLEU} ${V}" >> $RESULTS
echo "=== appended to $RESULTS ==="; tail -1 $RESULTS
echo "=== DONE $(date) ==="
