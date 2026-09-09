#!/bin/bash
#SBATCH --job-name=surgvu_l6agg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder6agg_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder6agg_%j.err
# NO mail directives. NEW LEVER B aggregator -- ranks the 6 imbalance-grid cells by
# recoverable (excl-floor) macro-F1, picks the best, compares vs the uniform baseline
# (ladder4 spine_full_bal, or spine_full if absent). afterok on the whole ladder6 array.
# KILL CRITERION (pre-registered): best cell's recoverable macro-F1 must beat the uniform
#   baseline by >=+0.05, else class-balancing is REFUTED -> data-floor, not loss -> MORE DATA.
set -e
mkdir -p /scratch/USERNAME/miccai-2026/SurgVU/logs
export OMP_NUM_THREADS=2
PY=/scratch/USERNAME/surgvu_env/bin/python
EXP=/users/USERNAME/SurgVU/experiments
ROOT=/scratch/USERNAME/miccai-2026/SurgVU/models/grid1
# baseline = the rebalanced ladder4 ckpt if present, else the uniform spine_full diag
BASE=/scratch/USERNAME/miccai-2026/SurgVU/models/spine_full_bal/per_class_diag.json
[ -f "$BASE" ] || BASE=/scratch/USERNAME/miccai-2026/SurgVU/models/spine_full/per_class_diag.json
RESULTS=/users/USERNAME/AUTONOMOUS_RESULTS.md
PARSE=/scratch/USERNAME/miccai-2026/SurgVU/logs/ladder6agg_${SLURM_JOB_ID}.parse

cd $EXP
$PY $EXP/grid_aggregate.py --root $ROOT --base_diag $BASE 2>&1 | tee $PARSE

BEST=$(grep -E '^GRID_BEST' $PARSE | tail -1 | sed 's/^GRID_BEST //')
echo "[SurgVU][${SLURM_JOB_ID}] Cat imbalance grid (focal-gamma x sample-strength, 6 cells) best -> ${BEST:-see log}" >> $RESULTS
echo "=== appended to $RESULTS ==="; tail -1 $RESULTS
echo "=== DONE $(date) ==="
