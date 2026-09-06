#!/usr/bin/env python
"""Build a frame->task manifest for the Cat-2 grounding spine.

WHY: the Cat-2 reference answers were authored from `matched_description` + the tool list
(dossier Finding, 2026-07-30). Both are predictable from video:
  * tools  -> the 14-class Cat-1 detector we already ship;
  * task   -> 8 classes = surgical PHASE RECOGNITION, our core competence.
So: video -> (task, tools) -> a ~7-word declarative in the reference register (the measured
DOMINANT answer form). This targets ENTITY correctness, which the policy surface prices at
+0.207 of a ~0.36-wide competitive range.

Emits one row per labelled segment with its video path and time span; frame sampling happens
in the training job (this is CPU-only manifest construction, login-node safe).
"""
import csv, glob, json, os, collections

LAB="/scratch/sc20osc/miccai-2026/SurgVU/data/external/cat2_train_labels/SURGVU25_train_labels"
# the real corpus: surgvu24/case_XXX/case_XXX_video_part_NNN.mp4 (155 cases, matches the labels 1:1)
VIDDIR="/scratch/sc20osc/miccai-2026/SurgVU/data/surgvu24_videos/surgvu24"
OUT="/scratch/sc20osc/miccai-2026/SurgVU/data/cat2_task_manifest.csv"

def canon(t):
    return " ".join(t.strip().lower().split())

rows=[]
for f in sorted(glob.glob(os.path.join(LAB,"*","tasks.csv"))):
    case=os.path.basename(os.path.dirname(f))
    for r in csv.DictReader(f=open(f)):
        tn=r.get("groundtruth_taskname","").strip()
        if not tn: continue
        rows.append({"case":case,"task":canon(tn),
                     "start_part":r.get("start_part",""),"stop_part":r.get("stop_part",""),
                     "start":r.get("start_time",""),"stop":r.get("stop_time",""),
                     "duration":r.get("duration",""),
                     "desc":(r.get("matched_description") or "").strip()})

tasks=collections.Counter(r["task"] for r in rows)
print("segments: %d | canonical task classes: %d"%(len(rows),len(tasks)))
for k,v in tasks.most_common(): print("   %5d (%5.1f%%)  %s"%(v,100*v/len(rows),k))

# resolve the PART file per (case, start_part) -- segments are indexed by part, not by case
parts={}
for d in sorted(glob.glob(os.path.join(VIDDIR,"case_*"))):
    case=os.path.basename(d)
    for p in sorted(glob.glob(os.path.join(d,"*_video_part_*.mp4"))):
        n=os.path.splitext(p)[0].rsplit("_",1)[-1]
        try: parts[(case,int(n))]=p
        except ValueError: pass
print("\nparts indexed: %d over %d cases"%(len(parts),len({c for c,_ in parts})))
miss=0
for r in rows:
    try: key=(r["case"], int(float(r.get("start_part") or 0)))
    except Exception: key=None
    r["video"]=parts.get(key,"") if key else ""
    if not r["video"]: miss+=1
print("video resolution: %d/%d segments resolved (%d unresolved)"%(len(rows)-miss,len(rows),miss))

os.makedirs(os.path.dirname(OUT),exist_ok=True)
with open(OUT,"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=["case","task","start_part","stop_part","start","stop","duration","video","desc"])
    w.writeheader(); w.writerows(rows)
print("wrote", OUT)
