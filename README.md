# SurgVU 2026 — Categories 1 & 2 · MICCAI / EndoVis (Intuitive Surgical)

**We are doing BOTH categories** (updated 2026-06-16):
- **Category 1 — Surgical tool classification + localization.** Weakly/semi-supervised **detection**:
  output bounding boxes + tool class for each tool in each test frame, trained on **noisy tool-presence
  labels** (the 280-video train set) + **bbox labels in a small val set**. Metric = **COCO mAP@[.5:.95]**.
  Submission template cloned → `./surgvu2025-category1-submission/`. (Prior SurgToolLoc detection
  approaches were weakly-supervised, e.g. ResNet-CAM + YOLO.)
- **Category 2 — Surgical VQA.** Answer an open-ended question about a **30 s** da Vinci clip;
  metric = max-BLEU over 5 refs. Lever = classifier-grounded templated answers (details below).

Both categories share the same gated 280-video SurgToolLoc training set (tool-presence + bbox-val +
`matched_description` narratives) → **blocked on grand-challenge registration**.

- Challenge site: https://surgvu26.grand-challenge.org/surgvu26/
- Cat-2 submission template (cloned → `./surgvu2025-category2-submission/`): https://github.com/isi-challenges/surgvu2025-category2-submission
- Cat-1 template also cloned (`./surgvu2025-category1-submission/`) for reference.
- Joint 2022–2025 methods paper: https://arxiv.org/abs/2305.07152

## Metric (Cat 2)
**BLEU** (uniform weights 0.25×4, NLTK `SmoothingFunction().method1`). For each question there are
**5 reference answers**; your single predicted answer is scored as the **max BLEU** over the 5, then
**mean** over all Q&A pairs. → Short, natural, correct phrasing wins. Yes/No + concise answers.

## Sample data (PUBLIC — already downloaded ✅)
`data/` (→ scratch) holds the public Cat-2 sample set (`SURGVU25_cat_2_sample_set_public.zip`,
179 MB, 11 cases `case122…case132`). Per case:
- `caseXXX.mp4` — 30 s clip (~22 MB, 720p)
- `caseXXX_question.json` — the question string (e.g. *"Are there forceps being used here?"*)
- `caseXXX.json` — list of 5 ground-truth answers (e.g. `["No", "No, forceps are not mentioned.", …]`)
This mirrors the exact evaluation I/O format.

## Full training data (GATED — needs grand-challenge registration)
Expanded SurgToolLoc 2022/23 set: **280 videos / 155 sessions, 60 fps, 720p, >840 h, >18M frames**,
da Vinci training exercises. Labels:
- `tools.csv` — tool install start/stop; **primary label = `groundtruth_toolname`** (12 core tools,
  noisy — robot-derived; ≤3 tools/clip). `commercial_toolname` = extra context.
- `tasks.csv` — 8 mutually-exclusive task segments: Suturing, Uterine Horn, Suspensory Ligaments,
  Rectal Artery/Vein, Skills Application, Range of Motion, Retraction & Collision Avoidance, Other.
- **Cat-2 extra:** `matched_description` column — detailed per-task narrative (anatomy, behaviour,
  tools) — the main signal for training the VQA/VLM. Unannotated spans = "Other".
- ⚠ This is **hundreds of GB–TB** of video. Register on the grand-challenge page to get the download
  (prior years used AWS/Google-Cloud hosting). Send me the link and I'll pull to scratch.

## Submission (grand-challenge Docker)
- Preliminary phase: ≤10 tries to get a working container; Final phase: 2 submissions (best counts).
- Cat-2 output: one JSON of answers per video (format finalised in the Cat-2 template — "stay tuned").
- Also required: editor access to container for user `aneeqzia_isi`; methodology report (LaTeX
  template) + 3-min video to isi.challenges@intusurg.com.
- Prizes: tiered 2026 vs overall (must beat all prior years for "overall"). Cat-2 1st: $1000 (2026) /
  $3000 (overall).

## Candidate models (Cat-2 VQA)
- Surgical VLP for retrieval/grounding (SurgVLP/PeskaVLP/HecVL — being staged in the FM pool).
- Video/image VQA generators: a captioning→QA or VLM pipeline (e.g. LLaVA-style, Qwen2-VL, Video-LLaVA)
  fine-tuned on the `matched_description` narratives. Because scoring is BLEU vs short refs, a strong
  prior is templated/concise answers grounded in predicted tools+task.

## ⏭ What I need from you
1. **Register on surgvu26.grand-challenge.org** → send the full training-data download link; I'll
   pull the ~TB video set to `/scratch/sc20osc/miccai-2026/SurgVU/data/`.
2. Want me to stand up a baseline VQA pipeline (tool/task classifier on sampled frames → templated
   answer generator) against the 11-case sample set now? It's fully runnable offline already.
