#!/bin/bash
#SBATCH --job-name=surgvu_diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/logs/diag_%j.out
#SBATCH --error=/scratch/sc20osc/miccai-2026/SurgVU/logs/diag_%j.err
# NO mail directives (self-monitored). Per-class auto-diagnosis of a spine ckpt.
# GPU is needed for mp4 decode + forward pass over the val split -> run via sbatch,
# NEVER on the login node. Reads the FULL-scale job 5998329 ckpt by default; pass
# CKPT=... as the 1st arg to diagnose any other run.
set -e
mkdir -p /scratch/sc20osc/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export OMP_NUM_THREADS=4
PY=/scratch/sc20osc/surgvu_env/bin/python
EXP=/users/sc20osc/SurgVU/experiments
D=/scratch/sc20osc/miccai-2026/SurgVU/data
CKPT=${1:-/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/spine_best.pt}
MAN=${2:-$D/frame_manifest_full.csv}

RESULTS=/users/sc20osc/AUTONOMOUS_RESULTS.md
LOG=/scratch/sc20osc/miccai-2026/SurgVU/logs/diag_${SLURM_JOB_ID}.out

echo "=== node $(hostname) ==="; nvidia-smi -L
echo "=== STEP 1 (data) class distribution ==="
$PY $EXP/class_distribution.py --manifest $MAN
echo "=== STEPS 1-3 (model) per-class diagnosis on $CKPT ==="
$PY $EXP/diagnose_spine.py --ckpt $CKPT --manifest $MAN --max_val 40000

# append a one-line summary of the auto-diagnosis to the autonomous results log
DIAG=$(dirname $CKPT)/per_class_diag.json
$PY - "$DIAG" "$SLURM_JOB_ID" "$RESULTS" <<'PYEOF'
import json, sys
diag, jid, res = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(diag))
    floor = ",".join(d.get("data_floor_tools", []))
    worst = d["per_tool"][0] if d.get("per_tool") else {}
    line = (f"[SurgVU][{jid}] diagnose_spine (auto-diagnosis, n_val={d.get('n_val')}) -> "
            f"tool macroF1(all14)={d['tool_macro_f1']:.4f} / excl-floor={d['tool_macro_f1_excl_floor']:.4f} / "
            f"ceiling-if-recoverable={d['tool_macro_ceiling_if_recoverable_fixed']:.4f}; "
            f"worst tool={worst.get('tool')}(F1={worst.get('f1')},sup={worst.get('support')}); "
            f"data-floor={{{floor}}}")
except Exception as e:
    line = f"[SurgVU][{jid}] diagnose_spine -> FAILED to parse {diag}: {e}"
open(res, "a").write(line + "\n")
print("appended:", line)
PYEOF
echo "=== appended to $RESULTS ==="; tail -1 $RESULTS
echo "=== DONE $(date) ==="
