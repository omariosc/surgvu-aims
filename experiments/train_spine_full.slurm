#!/bin/bash
#SBATCH --job-name=surgvu_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/SurgVU/logs/spine_full_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/SurgVU/logs/spine_full_%j.err
# NO mail directives (self-monitored). FULL set = 155 cases / 280 mp4 parts;
# uses a DEDICATED manifest (frame_manifest_full.csv) so it never collides with
# the partial-set job 5997565's frame_manifest_local.csv.

set -e
mkdir -p /scratch/USERNAME/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/USERNAME/hf_cache
export TORCH_HOME=/scratch/USERNAME/hf_cache/torch
export OMP_NUM_THREADS=4
PY=/scratch/USERNAME/surgvu_env/bin/python
D=/scratch/USERNAME/miccai-2026/SurgVU/data
LBL=$D/surgvu24_labels/labels
VID=$D/surgvu24_videos
MAN=$D/frame_manifest_full.csv
EXP=/users/USERNAME/SurgVU/experiments

echo "=== node $(hostname) gpu ==="; nvidia-smi -L

# 0) Extraction already verified COMPLETE (280 decodable mp4 parts / 155 cases;
#    the 281st zip entry is a source-corrupt .gstmp = case_071 part_001, unusable).
#    Manifest already built at submit time (frame_manifest_full.csv, ~1.07M rows);
#    rebuild here only if it is missing, to keep the job self-contained.
echo "=== ensure FULL manifest exists ==="
if [ ! -s "$MAN" ]; then
  echo "  manifest absent -> rebuilding"
  $PY $EXP/build_frame_labels.py --labels_dir $LBL --videos_dir $VID \
      --out_manifest $MAN --stride_sec 2.0 --require_video
else
  echo "  using prebuilt manifest: $(wc -l < $MAN) rows"
fi

# 2) smoke first (1 epoch, 512 frames) — fail fast on decode/label bugs.
echo "=== SMOKE ==="
$PY $EXP/train_spine.py --manifest $MAN --smoke --workers 6 --bs 32 --img_size 384

# 3) FULL train — larger budget than the partial run (more cases now available).
echo "=== FULL ==="
$PY $EXP/train_spine.py --manifest $MAN --epochs 10 --bs 64 --img_size 384 \
    --workers 12 --max_train 300000 --max_val 40000 \
    --out_dir /scratch/USERNAME/miccai-2026/SurgVU/models/spine_full

echo "=== DONE $(date) ==="
