#!/bin/bash
#SBATCH --job-name=cat1_build
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/cat1_build_%j.log
set -uo pipefail
D=/users/sc20osc/SurgVU/surgvu2025-category1-submission
OUT=/scratch/sc20osc/miccai-2026/SurgVU/docker
mkdir -p "$OUT"
export TMPDIR=${TMPDIR:-/tmp}
PODROOT=$TMPDIR/cat1build_${SLURM_JOB_ID:-$$}
mkdir -p "$PODROOT/storage" "$PODROOT/run"
POD="podman --root $PODROOT/storage --runroot $PODROOT/run --storage-driver overlay --storage-opt overlay.ignore_chown_errors=true"
TAG=localhost/surgvu26-cat1-aims:v1
TS=$(date +%Y-%m-%d_%H-%M-%S)
TAR="$OUT/surgvu26-cat1-aims_${TS}.tar.gz"
echo "### SurgVU Cat-1 container build | $(hostname) $(date)"
[ -s "$D/resources/detector.pt" ] || { echo "FATAL: detector.pt missing from the build context"; exit 1; }
echo "    detector.pt: $(stat -c%s "$D/resources/detector.pt") bytes"

# --network host is REQUIRED here: max_net_namespaces=0 makes the default per-RUN netns fail with
# a misleading "clone: No space left on device" (it is NOT disk).
timeout 5400 $POD build --network host --platform=linux/amd64 --tag "$TAG" "$D" 2>&1 \
  | grep -E "STEP|BUILD ASSERT|Error|error:|COMMIT|Successfully|Collecting torch|Installing" | tail -25
rc=${PIPESTATUS[0]}
$POD image exists "$TAG" || { echo "FATAL: image absent after build (rc=$rc)"; rm -rf "$PODROOT"; exit 1; }
echo "    image present."
echo "### CONFIG CHECK"
$POD inspect --format '    User={{.Config.User}} Entrypoint={{json .Config.Entrypoint}} WorkDir={{.Config.WorkingDir}}' "$TAG"
echo "### PERMISSION GATE (mode-based; --user <uid> cannot run here, no subuid ranges)"
$POD run --rm --network host --user 0 --entrypoint /bin/sh "$TAG" -c '
  bad=$(find /opt/app ! -perm -o+r -type f | head -3); baddir=$(find /opt/app ! -perm -o+rx -type d | head -3)
  if [ -n "$bad" ] || [ -n "$baddir" ]; then echo "    FAIL not world-readable"; echo "$bad$baddir"; exit 1; fi
  echo "    PASS: every file o+r, every dir o+rx under /opt/app"
  stat -c "    %n mode %a" /opt/app/resources/detector.pt /opt/app/inference.py' || { echo "FATAL: permission gate failed"; exit 1; }
echo "### SAVE"
$POD save --format docker-archive "$TAG" 2>/dev/null | gzip -c > "$TAR"
ls -l "$TAR" | awk '{print "    bytes:", $5}'
sha256sum "$TAR" | tee "$TAR.sha256"
python3 - "$TAR" <<'PY'
import sys, tarfile, gzip, json, io
with gzip.open(sys.argv[1],'rb') as g:
    with tarfile.open(fileobj=io.BytesIO(g.read()), mode='r') as t:
        m=json.load(t.extractfile('manifest.json'))
        print("    RepoTags:", m[0].get('RepoTags'), "| layers:", len(m[0].get('Layers',[])))
        print("    VERDICT: genuine docker-archive")
PY
rm -rf "$PODROOT"
echo "RESULT: PASS -- $TAR"
exit 0
