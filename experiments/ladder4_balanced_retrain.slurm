#!/bin/bash
#SBATCH --job-name=surgvu_l4bal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder4_%j.out
#SBATCH --error=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder4_%j.err
# NO mail directives (self-monitored). LADDER STEP 4 -- class-balanced retrain.
# Inverse-freq WeightedRandomSampler + multi-label focal loss, targeting the rare
# tools/tasks the diagnostic flagged, EXCLUDING the 3 zero-val data-floor tools
# (bipolar_dissector / suction_irrigator / tip_up_fenestrated_grasper). Then runs
# the per-class diagnostic on the rebalanced ckpt and compares the RECOVERABLE
# (excl-floor) macro-F1 vs the baseline spine_full diagnostic.
# KILL CRITERION (pre-registered): recoverable macro-F1 must rise >= +0.05 vs the
# uniform baseline, else balancing is REFUTED (it's data-floor, not loss).
set -e
mkdir -p /scratch/sc20osc/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export OMP_NUM_THREADS=4
PY=/scratch/sc20osc/surgvu_env/bin/python
EXP=/users/sc20osc/SurgVU/experiments
D=/scratch/sc20osc/miccai-2026/SurgVU/data
MAN=$D/frame_manifest_full.csv
OUT=/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full_bal
BASE_DIAG=/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/per_class_diag.json
RESULTS=/users/sc20osc/AUTONOMOUS_RESULTS.md
PARSE=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder4_${SLURM_JOB_ID}.parse

echo "=== node $(hostname) ==="; nvidia-smi -L
cd $EXP

echo "=== SMOKE (balanced + focal) ==="
$PY $EXP/train_spine.py --manifest $MAN --smoke --workers 6 --bs 32 --img_size 384 \
    --balanced_sampling --focal

echo "=== FULL balanced retrain ==="
$PY $EXP/train_spine.py --manifest $MAN --epochs 10 --bs 64 --img_size 384 \
    --workers 12 --max_train 300000 --max_val 40000 \
    --balanced_sampling --focal --out_dir $OUT

echo "=== per-class diagnosis of the rebalanced ckpt ==="
$PY $EXP/class_distribution.py --manifest $MAN
$PY $EXP/diagnose_spine.py --ckpt $OUT/spine_best.pt --manifest $MAN --max_val 40000

echo "=== compare recoverable macro-F1 vs baseline ==="
CMP=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder4_${SLURM_JOB_ID}_cmp.py
cat > $CMP <<'PYEOF'
import json, sys, os
new_p, base_p = sys.argv[1], sys.argv[2]
new = json.load(open(new_p))
nf  = new["tool_macro_f1_excl_floor"]
nall = new["tool_macro_f1"]
if os.path.exists(base_p):
    base = json.load(open(base_p))
    bf = base["tool_macro_f1_excl_floor"]; ball = base["tool_macro_f1"]
    d = nf - bf
    verdict = ("CONFIRMED balancing helps" if d >= 0.05 else
               "REFUTED (<+0.05 recoverable gain -> data-floor not loss)")
    print(f"BAL_RESULT recoverable_macroF1 base={bf:.4f} -> bal={nf:.4f} (delta {d:+.4f}); "
          f"all14 {ball:.4f}->{nall:.4f}; {verdict}")
else:
    print(f"BAL_RESULT recoverable_macroF1(bal)={nf:.4f} all14={nall:.4f} "
          f"(baseline diag missing -> no delta; CONFIRMED if standalone better than 5998329 agg)")
PYEOF
$PY $CMP "$OUT/per_class_diag.json" "$BASE_DIAG" 2>&1 | tee $PARSE

RES=$(grep -E '^BAL_RESULT' $PARSE | tail -1 | sed 's/^BAL_RESULT //')
echo "[SurgVU][${SLURM_JOB_ID}] Cat class-balanced retrain (inv-freq sampler+focal, excl 3 data-floor tools) -> ${RES:-see log}" >> $RESULTS
echo "=== appended to $RESULTS ==="; tail -1 $RESULTS
echo "=== DONE $(date) ==="
