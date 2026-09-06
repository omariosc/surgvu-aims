#!/bin/bash
#SBATCH --job-name=cat2_rebuild2
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/users/sc20osc/joblogs/cat2_rebuild2_%j.out
#SBATCH --error=/users/sc20osc/joblogs/cat2_rebuild2_%j.err
#
# v1 (7434562) died at STEP 1 pulling python:3.11-slim:
#   "requested 0:42 for /etc/gshadow ... lchown: invalid argument"
# AIRE has NO /etc/subuid range, so rootless podman cannot map a non-zero GID. My v1
# storage.conf set graphroot/runroot but carried NO ignore_chown_errors at all.
# This v2 reuses the configuration PROVEN by the MultiBypass container gate (which pulled
# and built successfully today): CONTAINERS_STORAGE_CONF holding the driver AND its options,
# NO global --root/--runroot flags (those DISCARD the options), cgroupfs, buildah unshare.
set -uo pipefail
SRC=/users/sc20osc/SurgVU/surgvu2025-category2-submission
OUT=/scratch/sc20osc/miccai-2026/SurgVU/docker
ls "$SRC" >/dev/null 2>&1 || { echo "[eio-guard] src unreadable -> abort"; exit 3; }
[ -f "$SRC/inference.py" ] || { echo "[eio-guard] inference.py missing -> abort"; exit 3; }

export TMPDIR=${TMPDIR:-/tmp}
ST=$TMPDIR/cat2rb_${SLURM_JOB_ID:-$$}
mkdir -p "$ST"/{root,runroot,tmp,xdg}; chmod 700 "$ST/xdg"
export XDG_RUNTIME_DIR="$ST/xdg"
export PF="--cgroup-manager=cgroupfs --events-backend=file"
export NOISE='level=warning msg="(Failed to add pause process|Network file system detected|The storage .driver. option should be set)'

MOUNTPROG=""
[ -x /usr/bin/fuse-overlayfs ] && MOUNTPROG='mount_program = "/usr/bin/fuse-overlayfs"'
cat > "$ST/storage.conf" <<'EOF'
[storage]
driver = "overlay"
graphroot = "@@GRAPH@@"
runroot = "@@RUN@@"
[storage.options.overlay]
@@MOUNTPROG@@
ignore_chown_errors = "true"
EOF
sed -i "s|@@GRAPH@@|$ST/root|; s|@@RUN@@|$ST/runroot|; s|@@MOUNTPROG@@|$MOUNTPROG|" "$ST/storage.conf"
export CONTAINERS_STORAGE_CONF="$ST/storage.conf"

echo "=== source state (what we are baking in) ==="
grep -n "^USE_TASK_CONDITIONING" "$SRC/inference.py"
echo "=== storage.conf in use ==="; sed 's/^/  /' "$ST/storage.conf"

export TAG=localhost/aims-cat2:reverted
export SRC OUT ST
export OUTF="$OUT/aims-cat2-reverted_$(date '+%Y-%m-%d_%H-%M-%S').tar.gz"

# quoted heredoc; parameters come from the environment (unquoted heredocs with backticks
# produced a bogus error in ORENA 6909445)
cat > "$ST/build.sh" <<'EOF'
set -uo pipefail
BUILDNET=""
if ! podman $PF run --rm --network none docker.io/library/busybox true >/dev/null 2>&1; then
  BUILDNET="--network host"
  echo "DIVERGENCE: --network none unusable on this node; using --network host"
fi
podman $PF build $BUILDNET --platform=linux/amd64 -t "$TAG" "$SRC"; echo "BUILD_RC=$?"
podman $PF run --rm $BUILDNET --entrypoint grep "$TAG" -m1 '^USE_TASK_CONDITIONING' /opt/app/inference.py \
  '^USE_TASK_CONDITIONING' /opt/app/inference.py > "$ST/inimage.txt" 2>/dev/null; echo "ASSERT_RC=$?"
podman $PF save "$TAG" | gzip -c > "$OUTF"; echo "SAVE_RC=$?"
EOF
buildah unshare bash "$ST/build.sh" 2>&1 | grep -vE "$NOISE" | tail -30

BRC=$(grep -oP 'BUILD_RC=\K\d+' "$ST/../"* 2>/dev/null | head -1)
echo "=== in-image flag line ==="
IN=$(cat "$ST/inimage.txt" 2>/dev/null || true)
echo "  ${IN:-<none captured>}"
RC=0
case "$IN" in
  *"= False"*|*"=False"*) echo "  [PASS] the built IMAGE carries USE_TASK_CONDITIONING=False" ;;
  "" ) echo "  [FAIL] could not read the flag from inside the image"; RC=5 ;;
  *  ) echo "  [FAIL] image carries: $IN"; RC=6 ;;
esac
if [ -s "$OUTF" ] && [ $RC -eq 0 ]; then
  ls -l "$OUTF" | awk '{print "  saved:", $5, "bytes", $9}'
  echo "  sha256: $(sha256sum "$OUTF" | cut -d' ' -f1)"
  echo "RESULT: PASS -- image built AND verified in-image to carry the revert"
else
  [ -s "$OUTF" ] || { echo "  [FAIL] no tarball written"; RC=${RC:-7}; [ $RC -eq 0 ] && RC=7; }
  echo "RESULT: FAIL rc=$RC"
fi
echo "=== DONE at $(date '+%F %T') ==="
exit $RC
