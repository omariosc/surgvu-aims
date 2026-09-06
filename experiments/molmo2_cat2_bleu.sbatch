#!/bin/bash
#SBATCH --job-name=surgvu_molmo2_cat2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/users/sc20osc/SurgVU/experiments/molmo2_cat2_bleu_%j.out
#SBATCH --error=/users/sc20osc/SurgVU/experiments/molmo2_cat2_bleu_%j.out
# NO #SBATCH mail (self-monitor)

set -euo pipefail

export HF_HOME=/scratch/sc20osc/hf_cache
source /scratch/sc20osc/miccai-2026/vqa_env/bin/activate

echo "=== SurgVU Cat-2 Molmo2-8B zero-shot grounded-BLEU (job $SLURM_JOB_ID) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PY=/scratch/sc20osc/miccai-2026/vqa_env/bin/python
RUNNER=/users/sc20osc/SurgVU/experiments/molmo2_cat2_bleu.py

# Capture stdout so we can extract the MOLMO2_CAT2_BLEU=<mean> line for the results log.
TMPLOG=$(mktemp)
$PY "$RUNNER" 2>&1 | tee "$TMPLOG"

MEAN=$(grep -oP 'MOLMO2_CAT2_BLEU=\K[0-9.]+' "$TMPLOG" | tail -1)
if [ -z "${MEAN:-}" ]; then MEAN="ERROR"; fi

echo "[SurgVU][$SLURM_JOB_ID] Molmo2-8B zero-shot Cat-2 grounded-BLEU=${MEAN} (n=11) vs spine 0.7428(optimistic)/winner 0.4215" \
  >> /users/sc20osc/AUTONOMOUS_RESULTS.md

rm -f "$TMPLOG"
echo "=== done (MOLMO2_CAT2_BLEU=${MEAN}) ==="
