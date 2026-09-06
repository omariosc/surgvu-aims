#!/bin/bash
#SBATCH --job-name=surgvu_sslscan
#SBATCH --partition=nodes
#SBATCH --nice=90
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=20:00:00
#SBATCH --array=0-27
#SBATCH --output=/users/sc20osc/SurgVU/experiments/logs/sslscan_%A_%a.out
#SBATCH --error=/users/sc20osc/SurgVU/experiments/logs/sslscan_%A_%a.out

# STAGE 1 (CPU only -- no GPU is needed to decode video, and the gpu partition is
# the scarce resource).  Sharded 28 ways over the 280 corpus video parts.
# Decode throughput measured at 142x realtime on a login node with many cores;
# on a 2-CPU allocation the h264 decoder is several times slower, so this asks
# for 8 CPUs and 20 h.  (Tasks 0-1 of job 7110752 launched under the old 2-CPU /
# 8 h request -- if either hits the wall, resubmit just those shards with this file.)
set -uo pipefail
export OMP_NUM_THREADS=8
export OPENCV_FFMPEG_LOGLEVEL=-8
PY=/scratch/sc20osc/surgvu_env/bin/python
cd /users/sc20osc/SurgVU/experiments

echo "=== SurgVU Cat-1 SSL corpus scan shard ${SLURM_ARRAY_TASK_ID}/28 ==="
echo "host=$(hostname) job=${SLURM_JOB_ID}"; date

$PY cat1_ssl_scan.py --shard "${SLURM_ARRAY_TASK_ID}" --nshards 28

echo "DONE $(date)"
