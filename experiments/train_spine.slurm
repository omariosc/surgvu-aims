#!/bin/bash
#SBATCH --job-name=surgvu_spine
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/logs/spine_%j.out
#SBATCH --error=/scratch/sc20osc/miccai-2026/SurgVU/logs/spine_%j.err
# NO mail directives (self-monitored).

set -e
mkdir -p /scratch/sc20osc/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export OMP_NUM_THREADS=4
PY=/scratch/sc20osc/surgvu_env/bin/python
D=/scratch/sc20osc/miccai-2026/SurgVU/data
LBL=$D/surgvu24_labels/labels
VID=$D/surgvu24_videos
MAN=$D/frame_manifest_local.csv
EXP=/users/sc20osc/SurgVU/experiments

echo "=== node $(hostname) gpu ==="; nvidia-smi -L

# 1) Extract whatever mp4s have finished downloading by now (idempotent; the
#    320 GB zip is still streaming on the login node). Skips already-extracted.
echo "=== stream-extract complete mp4s ==="
$PY $EXP/stream_extract_zip.py --zip $D/surgvu24_videos_only.zip --out $VID --only_mp4 || true

# 2) (re)build a video-gated, part-aware manifest from whatever is local NOW.
$PY $EXP/build_frame_labels.py --labels_dir $LBL --videos_dir $VID \
    --out_manifest $MAN --stride_sec 2.0 --require_video

# 3) smoke first (1 epoch, 512 frames) — fail fast on decode/label bugs.
echo "=== SMOKE ==="
$PY $EXP/train_spine.py --manifest $MAN --smoke --workers 6 --bs 32 --img_size 384

# 4) full train.
echo "=== FULL ==="
$PY $EXP/train_spine.py --manifest $MAN --epochs 8 --bs 64 --img_size 384 \
    --workers 10 --max_train 120000 --max_val 20000 \
    --out_dir /scratch/sc20osc/miccai-2026/SurgVU/models/spine

echo "=== DONE $(date) ==="
