#!/bin/bash
#SBATCH --job-name=cat1_filter
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/users/sc20osc/joblogs/cat1_filter_%j.out
#SBATCH --error=/users/sc20osc/joblogs/cat1_filter_%j.err
# Cat-1 FILTER detector image. podman config = the form PROVEN on AIRE 2026-08-22:
# CONTAINERS_STORAGE_CONF carrying driver AND options, no global --root, cgroupfs,
# inside buildah unshare. Asserts the FILTER weights are the ones baked in.
set -uo pipefail
SRC=/users/sc20osc/SurgVU/surgvu2025-category1-submission
OUT=/scratch/sc20osc/miccai-2026/SurgVU/docker
EXPECT_SHA=91ed80228a6bc2f96e014972bf75c3b8178ffda6efea745f603bd0655f922a66
ls "$SRC" >/dev/null 2>&1 || { echo "[eio-guard] src unreadable"; exit 3; }

export TMPDIR=${TMPDIR:-/tmp}
ST=$TMPDIR/cat1f_${SLURM_JOB_ID:-$$}
mkdir -p "$ST"/{root,runroot,tmp,xdg}; chmod 700 "$ST/xdg"
export XDG_RUNTIME_DIR="$ST/xdg"
export PF="--cgroup-manager=cgroupfs --events-backend=file"
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
export TAG=localhost/aims-cat1-filter:v1
export SRC ST EXPECT_SHA
export OUTF="$OUT/aims-cat1-filter_$(date '+%Y-%m-%d_%H-%M-%S').tar.gz"
echo "=== host-side detector sha: $(sha256sum "$SRC/resources/detector.pt" | cut -d' ' -f1)"
cat > "$ST/b.sh" <<'EOF'
set -uo pipefail
BN=""
podman $PF run --rm --network none docker.io/library/busybox true >/dev/null 2>&1 || BN="--network host"
podman $PF build $BN --platform=linux/amd64 -t "$TAG" "$SRC"; echo "BUILD_RC=$?"
podman $PF run --rm $BN --entrypoint sha256sum "$TAG" /opt/app/resources/detector.pt > "$ST/inimg.txt" 2>/dev/null
echo "ASSERT_RC=$?"
podman $PF save "$TAG" | gzip -c > "$OUTF"; echo "SAVE_RC=$?"
EOF
buildah unshare bash "$ST/b.sh" 2>&1 | tail -20
IN=$(cut -d' ' -f1 < "$ST/inimg.txt" 2>/dev/null)
echo "=== in-image detector sha: ${IN:-<none>}"
RC=0
if [ "$IN" = "$EXPECT_SHA" ]; then echo "  [PASS] the FILTER weights are the ones baked into the image"
else echo "  [FAIL] in-image detector sha $IN != expected $EXPECT_SHA"; RC=6; fi
if [ -s "$OUTF" ] && [ $RC -eq 0 ]; then
  ls -l "$OUTF" | awk '{print "  saved:",$5,"bytes",$9}'
  echo "  sha256: $(sha256sum "$OUTF" | cut -d' ' -f1)"
  echo "RESULT: PASS"
else echo "RESULT: FAIL rc=$RC"; [ $RC -eq 0 ] && RC=7; fi
echo "=== DONE at $(date '+%F %T') ==="
exit $RC
