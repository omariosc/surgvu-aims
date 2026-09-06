#!/bin/bash
#SBATCH --job-name=cat1_smoke
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/cat1_smoke_%j.log
set -uo pipefail
D=/users/sc20osc/SurgVU/surgvu2025-category1-submission
T=/tmp/cat1_smoke_$$
mkdir -p "$T/input" "$T/output"
# a REAL 30s 720p clip from the Cat-2 sample data (same modality as the Cat-1 eval input)
cp /scratch/sc20osc/miccai-2026/SurgVU/data/case122/case122.mp4 "$T/input/endoscopic-robotic-surgery-video.mp4"
cat > "$T/input/inputs.json" <<'J'
[{"file":"x","image":null,"value":null,"interface":{"slug":"endoscopic-robotic-surgery-video","kind":"MP4 file","super_kind":"File","relative_path":"endoscopic-robotic-surgery-video.mp4","example_value":null}}]
J
echo "### Cat-1 entrypoint smoke on a REAL clip  $(date)"
ls -l "$T/input/"
cd "$D"
INPUT_PATH="$T/input" OUTPUT_PATH="$T/output" RESOURCE_PATH="$D/resources" \
  /scratch/sc20osc/surgvu_env/bin/python inference.py
echo "### OUTPUT ==="
ls -l "$T/output/"
/scratch/sc20osc/surgvu_env/bin/python - "$T/output/surgical-tools.json" <<'PY'
import json,sys,collections,re
d=json.load(open(sys.argv[1]))
print("  top-level keys:", list(d.keys()))
b=d["boxes"]; print("  boxes:", len(b))
if b:
    print("  example:", json.dumps(b[0])[:200])
    cls=collections.Counter(re.sub(r"^slice_nr_\d+_","",x["name"]) for x in b)
    print("  class histogram:", cls.most_common())
    fr=sorted({int(re.match(r"slice_nr_(\d+)_",x["name"]).group(1)) for x in b})
    print("  frames with detections: %d (range %d..%d)" % (len(fr), fr[0], fr[-1]))
    import numpy as np
    p=[x["probability"] for x in b]
    print("  probability: min %.3f median %.3f max %.3f" % (min(p), float(np.median(p)), max(p)))
    c=b[0]["corners"]
    assert len(c)==4 and all(len(pt)==3 for pt in c), "corner shape wrong"
    print("  corner shape OK (4 x [x,y,z])")
else:
    print("  !! ZERO boxes -- detector produced nothing on a real clip")
PY
rm -rf "$T"
echo "### done $(date)"
