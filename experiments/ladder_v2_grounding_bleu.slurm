#!/bin/bash
#SBATCH --job-name=surgvu_v2gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/SurgVU/logs/v2gb_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/SurgVU/logs/v2gb_%j.err
# NO mail directives (self-monitored). LEVER 1 -- Cat-2 V2 action-grounded BLEU.
# Runs the spine over the 11 in-hand Cat-2 clips with the V2 grounding (cut / suture-
# required action bridges) + V2 template router (action branches reachable + tense-aware
# "was utilized" carrier), at the best grounding config (nf4_thr0.3_mean), and re-scores
# SurgVU Cat-2 BLEU vs the 0.7428 v1-best / 0.4215 SOTA. GPU needed for decode+forward.
set -e
mkdir -p /scratch/USERNAME/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/USERNAME/hf_cache
export TORCH_HOME=/scratch/USERNAME/hf_cache/torch
export OMP_NUM_THREADS=4
PY=/scratch/USERNAME/surgvu_env/bin/python
EXP=/users/USERNAME/SurgVU/experiments
CKPT=${1:-/scratch/USERNAME/miccai-2026/SurgVU/models/spine_full/spine_best.pt}
TAG=${2:-spine_full}
OUT=$(dirname $CKPT)/grounded_bleu_v2.json
RESULTS=/users/USERNAME/AUTONOMOUS_RESULTS.md

echo "=== node $(hostname) ==="; nvidia-smi -L
cd $EXP
PARSE=/scratch/USERNAME/miccai-2026/SurgVU/logs/v2gb_${SLURM_JOB_ID}.parse
$PY $EXP/spine_grounding_bleu_v2.py --ckpt $CKPT --n_frames 4 --thr 0.3 --pool mean \
    --out $OUT 2>&1 | tee $PARSE

BLEU=$(grep -oE 'GROUNDED_BLEU_V2=[0-9.]+' $PARSE | tail -1 | cut -d= -f2)
VERD=$(grep -E '^# VERDICT:' $PARSE | tail -1 | sed 's/^# VERDICT: //')
[ -z "$BLEU" ] && BLEU="nan"
# verdict vs the 0.7428 saturated v1-best (the kill criterion)
if awk "BEGIN{exit !($BLEU > 0.7428)}" 2>/dev/null; then
  V="CONFIRMED >0.7428 v1-best (action/template fix HELPED)"
elif awk "BEGIN{exit !($BLEU >= 0.4215)}" 2>/dev/null; then
  V="REFUTED vs v1-best: ${BLEU} <= 0.7428 (still clears 0.4215 SOTA) -- ${VERD:-see log}"
else
  V="REGRESSION ${BLEU} < 0.4215 -- ${VERD:-see log}"
fi
echo "[SurgVU][${SLURM_JOB_ID}] Cat-2 V2 action-grounded BLEU (n=11, ${TAG}, nf4_thr0.3_mean) -> ${BLEU} ${V}" >> $RESULTS
echo "=== appended to $RESULTS ==="; tail -1 $RESULTS
echo "=== DONE $(date) ==="
