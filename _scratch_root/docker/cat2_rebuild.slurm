#!/bin/bash
#SBATCH --job-name=cat2_rebuild
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/users/sc20osc/joblogs/cat2_rebuild_%j.out
#SBATCH --error=/users/sc20osc/joblogs/cat2_rebuild_%j.err
set -euo pipefail
SRC=/users/sc20osc/SurgVU/surgvu2025-category2-submission
OUT=/scratch/sc20osc/miccai-2026/SurgVU/docker
ls "$SRC" >/dev/null 2>&1 || { echo "[eio-guard] src unreadable -> abort"; exit 3; }
[ -f "$SRC/inference.py" ] || { echo "[eio-guard] inference.py missing -> abort"; exit 3; }

ST=/tmp/job.$SLURM_JOB_ID/cat2; mkdir -p "$ST/xdg" && chmod 700 "$ST/xdg"
export XDG_RUNTIME_DIR="$ST/xdg" TMPDIR="$ST/tmp"; mkdir -p "$TMPDIR"
export CONTAINERS_STORAGE_CONF="$ST/storage.conf"
cat > "$CONTAINERS_STORAGE_CONF" <<EOF
[storage]
driver = "vfs"
runroot = "$ST/run"
graphroot = "$ST/graph"
EOF
PF="--cgroup-manager=cgroupfs"

echo "=== source state (what we are baking in) ==="
grep -n "^USE_TASK_CONDITIONING" "$SRC/inference.py"

# netns cap is NODE-DEPENDENT (max_net_namespaces=0 on some nodes) -> probe, don't assume
BUILDNET=""
if ! podman $PF run --rm --network none docker.io/library/busybox true >/dev/null 2>&1; then
  BUILDNET="--network host"
  echo "  DOCUMENTED DIVERGENCE: --network none unusable here; building with --network host"
fi

TAG=localhost/aims-cat2:reverted
echo "=== build ==="
set +e
podman $PF build $BUILDNET --platform=linux/amd64 -t "$TAG" "$SRC"
RC=$?
set -e
[ $RC -eq 0 ] || { echo "RESULT: FAIL -- build rc=$RC"; exit $RC; }

echo "=== ASSERT the fix is IN the image (not merely in the source tree) ==="
set +e
podman $PF run --rm $BUILDNET "$TAG" python -c "
import re,sys
s=open('/opt/app/inference.py').read()
m=re.search(r'^USE_TASK_CONDITIONING\s*=\s*(\w+)', s, re.M)
print('  in-image USE_TASK_CONDITIONING =', m.group(1) if m else 'NOT FOUND')
sys.exit(0 if m and m.group(1)=='False' else 9)
"
A=$?
set -e
[ $A -eq 0 ] || { echo "RESULT: FAIL -- image does NOT carry the reverted flag (rc=$A)"; exit $A; }

echo "=== save ==="
TS=$(date '+%Y-%m-%d_%H-%M-%S')
F="$OUT/aims-cat2-reverted_${TS}.tar.gz"
podman $PF save "$TAG" | gzip -c > "$F"
ls -l "$F" | awk '{print "  saved:", $5, "bytes", $9}'
echo "  sha256: $(sha256sum "$F" | cut -d' ' -f1)"
echo "RESULT: PASS -- image built AND verified to contain USE_TASK_CONDITIONING=False"
echo "=== DONE at $(date '+%F %T') ==="
exit 0
