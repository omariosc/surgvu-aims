#!/bin/bash
#SBATCH --job-name=surgvu_l9rl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder9_%j.out
#SBATCH --error=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder9_%j.err
# NO mail directives (self-monitored). NEW RESEARCH-BACKED LEVERS (DEEP_RESEARCH_CAT2),
# distinct from the refill agent's variant/detector sweeps:
#   (1) qcond_retrieval_bleu  -- question-conditioned temporal retrieval (top-k by Q-relevance)
#                                vs the uniform np.linspace sampling everything else uses.
#   (2) canonical_answer_layer-- ontology -> single canonical surface form (the research's
#                                "biggest BLEU lever"; reduces variance vs the keyword router).
#   (3) real-grounding cache + leakage_check on the ACTUAL 0.7428 spine predictions
#                                (scorer-clone cross-check + text-only/polarity-shuffle ablations).
# Each appends ONE [SurgVU][jobid] line to AUTONOMOUS_RESULTS.md. Does NOT touch the spine,
# the experiment-log scripts the refill agent owns, or any aielt_*/jig_* job.
set -e
mkdir -p /scratch/sc20osc/miccai-2026/SurgVU/logs
mkdir -p /scratch/sc20osc/miccai-2026/SurgVU/models/logs   # FIX 2026-06-20: $MODELDIR/../logs tee target (6204346 died here AFTER step1 completed; steps 2+3 were skipped)
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export OMP_NUM_THREADS=4
PY=/scratch/sc20osc/surgvu_env/bin/python
EXP=/users/sc20osc/SurgVU/experiments
CKPT=/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/spine_best.pt
MODELDIR=/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full
RESULTS=/users/sc20osc/AUTONOMOUS_RESULTS.md
J=${SLURM_JOB_ID}

echo "=== node $(hostname) ==="; nvidia-smi -L
cd $EXP

# ---- (0) dump the REAL spine grounding (winning nf4_thr0.3_mean) for the leakage check ----
$PY - "$CKPT" "$MODELDIR/real_grounding.json" <<'PYEOF'
import sys, os, glob, json
sys.path.insert(0, "/users/sc20osc/SurgVU/experiments")
import spine_grounding_v2 as sg2
ckpt, out = sys.argv[1], sys.argv[2]
cache, g = {}, {}
for d in sorted(glob.glob("/users/sc20osc/SurgVU/data/case*")):
    cid = os.path.basename(d)
    qf, vf = os.path.join(d, f"{cid}_question.json"), os.path.join(d, f"{cid}.mp4")
    if not (os.path.exists(qf) and os.path.exists(vf)):
        continue
    q = json.load(open(qf))
    try:
        g[cid] = sg2.ground_clip_v2(vf, q, ckpt, 4, 0.3, "mean", cache)
    except Exception as e:
        print(f"[warn] {cid}: {e}"); g[cid] = {}
json.dump(g, open(out, "w"), indent=2)
print(f"[real-grounding] wrote {out}: {g}")
PYEOF

# ---- (1) question-conditioned temporal retrieval ----
P1=$MODELDIR/../logs/ladder9_${J}.qcond
$PY $EXP/qcond_retrieval_bleu.py --ckpt $CKPT 2>&1 | tee $P1
Q=$(grep -E '^QCOND_BEST' $P1 | tail -1 | sed 's/^QCOND_BEST //')
echo "[SurgVU][${J}] Cat-2 Q-conditioned temporal retrieval (top-k by Q-relevance vs uniform, n=11) -> ${Q:-see log}" >> $RESULTS

# ---- (2) canonical-answer layer ----
P2=$MODELDIR/../logs/ladder9_${J}.canon
$PY $EXP/canonical_answer_layer.py --ckpt $CKPT 2>&1 | tee $P2
C=$(grep -E '^CANON_BEST' $P2 | tail -1 | sed 's/^CANON_BEST //')
echo "[SurgVU][${J}] Cat-2 canonical-answer layer (ontology->canonical form vs router, n=11) -> ${C:-see log}" >> $RESULTS

# ---- (3) leakage check on the REAL 0.7428 grounding ----
P3=$MODELDIR/../logs/ladder9_${J}.leak
$PY $EXP/leakage_check.py 2>&1 | tee $P3
L=$(grep -E '^LEAKAGE_VERDICT' $P3 | tail -1 | sed 's/^LEAKAGE_VERDICT //')
echo "[SurgVU][${J}] Cat-2 adversarial/leakage check on real grounding (scorer-clone + text-only/polarity ablations) -> ${L:-see log}" >> $RESULTS

echo "=== appended to $RESULTS ==="; tail -3 $RESULTS
echo "=== DONE $(date) ==="
