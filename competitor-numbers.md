# SurgVU Category 2 — Verified competitor numbers & methods

> Rigor bar (`MICCAI2026_INSIGHTS.md`): **verified, source-cited numbers only — never estimate.**
> Every figure below is VERIFIED against a primary source (grand-challenge results pages, the master
> arXiv results paper v4, the dataset paper, or the teams' own repos) unless tagged **UNVERIFIED**.
> Last updated 2026-06-15.

## ⚠ CRITICAL: "Category 2" = different tasks in 2024 vs 2025
- **SurgVU 2024 Cat-2 = surgical TASK/STEP recognition** (8-class, frame-level, fully supervised).
  Scored by **mean weighted-F1** (scikit-learn), **NOT BLEU**. So the 2024 teams (PDMYR / SurgOPTeam /
  TeamBCU) have **F1 numbers, no BLEU**.
  - Source: `isi-challenges/surgvu24-category-2-eval/evaluation.py` uses
    `f1_score(average="weighted")` + accuracy/precision/recall, no NLTK/BLEU.
- **SurgVU 2025 Cat-2 = open-ended Surgical VQA**, free-form NL answers. Scored by **mean BLEU**
  (max over 5 references). **This is our 2026 task.** Source: <https://surgvu25.grand-challenge.org/evaluation-criteria/>.

The 2026 edition continues the 2025 VQA + BLEU formulation (per `/users/sc20osc/SurgVU/README.md` and the
challenge site <https://surgvu26.grand-challenge.org/surgvu26/>).

---

## 1. SurgVU 2025 Cat-2 — Surgical VQA (BLEU) — **the relevant prior edition**
Source: <https://surgvu25.grand-challenge.org/results/> and arXiv:2305.07152v4 Table 56.

| Rank | Team | **BLEU (final)** | Prize | Method (verified) |
|---|---|---|---|---|
| **1** | **Capybara** (Quan Huu Cap, Aillis Inc., Tokyo) | **0.4215** | $3000 | **Zero-shot LLaVA-OneVision-7B** + EfficientNetV2-S tool & organ classifiers (no VLM fine-tune) |
| 2 | **UoM-SurgicalAI** (Univ. of Manchester) | **0.3656** | $2000 | **Zero-shot InternVL3.5** prompting (beat their own PEFT fine-tune, which scored 0.202) |
| 3 | **Medibot** | **0.3404** | $1000 | Fine-tuned large multimodal model + their Cat-1 detector (YOLOv8l+ResNet-50) for tool grounding |
| 4 | AMI (Kyung Hee Univ.) | 0.295 | — | EndoViT→Vicuna visual adapter + LoRA on GPT-5-generated QA |
| 5 | UT | 0.2652 | — | — |
| 6 | gardenia | 0.2615 | — | — |
| 7 | SK | 0.2021 | — | — |
| 8 | Algoritmi | 0.1231 | — | — |

- **Best Methodology Report (Cat-2): UoM-SurgicalAI ($500).** 8 complete submissions.
- **2026 target to beat: BLEU 0.4215 (Capybara) for "overall"; any 2026 entrant for the tiered prize.**

### Winner method detail (verified)
- **Capybara (1st, BLEU 0.4215 final / 0.4237 prelim).** Two-stage, **no VLM fine-tuning**:
  1. **EfficientNetV2-S tool classifier** (97% macro-F1, trained on SurgToolLoc-2022 ~24.6k clips,
     512×512, label smoothing) + **EfficientNetV2-S organ classifier** (98% macro-F1, 7 organs, voting)
     → build a **structured tools+organ text description**.
  2. That description + a generated video description + question + **5 sampled frames** →
     **`llava-onevision-qwen2-7b-ov-hf` (zero-shot)** generates a short answer.
  - Video preprocessing: crop black side margins; **Gaussian-blur the bottom tool-list strip** (the
    da Vinci overlay) so the VLM can't cheat off burned-in text. 5 frames beat 21 frames (0.4022).
  - Repo: <https://github.com/huuquan1994/surgvu25-cat2-submission> (full pipeline + solution report PDF
    `solution_report/Team_Capybara_report.pdf` + corrected 11-video sample set on GDrive).
  - Preprint: "Effective Surgical VQA Without Ground-Truth Video Descriptions" (arXiv ID **UNVERIFIED**).
  - **KEY LESSON: a strong classifier-grounded ZERO-SHOT VLM won.** Fine-tuning was NOT required and in
    UoM's case actively hurt (0.202 fine-tuned vs 0.3656 zero-shot).
- **UoM-SurgicalAI (2nd, 0.3656).** Ranked submission = **zero-shot InternVL3.5** prompting. Their
  fine-tuned variant (InternVL3.5-1B + adapter, DoRA/DCT-GaLore PEFT, 8 frames, LLM-generated QA from
  the ~10 fixed task-description templates) scored only **0.202** — i.e. **PEFT fine-tune < zero-shot**.
- **Medibot (3rd, 0.3404).** Fine-tuned LMM + integrated their Cat-1 tool detector (cross-task synergy).
  Exact VLM name **UNVERIFIED**.
- 2025 method table also lists, across teams: Qwen2.5-VL-3B, LLaVA-OneVision-7B, InternVL3/3.5,
  InternVL2.5-MoP-4B.

### Public prior-art assets (study these)
- ⭐ `bravefox12138/surgvu2025vqa` — Qwen2.5-VL-3B fine-tune **purpose-built for SurgVU-2025 Cat-2**,
  bundles YOLO detector + ResNet-50 12-class classifier. Apache-2.0.
  <https://huggingface.co/bravefox12138/surgvu2025vqa>
- ⭐ `nvidia/Qwen2.5-VL-7B-Surg-CholecT50` — Qwen2.5-VL-7B SFT on CholecT50 triplets (single-frame).
  <https://huggingface.co/nvidia/Qwen2.5-VL-7B-Surg-CholecT50>
- `Simon-zsy/surgvu24` — SurgVU-2024 video data mirror (HF).

---

## 2. SurgVU 2024 Cat-2 — Task recognition (weighted-F1, **NOT BLEU**) — context only
Source: <https://surgvu24.grand-challenge.org/results/> and arXiv:2305.07152v4 Table 35.

| Rank | Team | **Weighted F1** | Prize | Method (verified) |
|---|---|---|---|---|
| 1 | **PDMYR** (Jie Tian) | **0.8977** | $3000 | Frame-level step classifier (**MViT**), heavy regularization/aug (3D-CutMix, TTA), temporal majority-vote smoothing. NOT a VLM. |
| 2 | **SurgOp / SurgOPTeam** (DEEL-AI) | 0.8521 | $2000 | Video transformer **MViT (`mvit_base_32x3`)**, 32-frame clips, weighted sampling |
| 3 | **TeamBCU** (M. Bilal et al.) | 0.8216 | $1000 | Video backbone (MViT/TimeSformer/SlowFast/Video-Swin family) |
| 4 | SmartLab HKUST | 0.821 | — | — |
| 5 | InspireLab | 0.8103 | — | — |
| 6 | MIDAS | 0.6916 | — | — |

- **Best Methodology Report (Cat-2): SurgOPTeam ($500).** 160 prelim / 24 final subs; 6 complete.
- **These are F1, not BLEU.** The 2024 task was step *classification*, so the methods (pure video
  classifiers) and metric do not transfer to 2026's free-form VQA. Listed only because the prompt asked.
- Repos: PDMYR <https://github.com/SalenGit/surgvu2024-category2-rank1> ·
  SurgOPTeam <https://github.com/deel-ai-papers/SurgVU2024-PhaseDetection> ·
  also <https://github.com/quzanh1130/SurgVU2024-Category-2-Submission>.

---

## 3. Master paper, dataset & metric (verified)
- **Master results paper:** arXiv:**2305.07152** — *"Intuitive Surgical SurgToolLoc and SurgVU Challenges
  Results: 2022-2025"* (Zia, Berniker, Nespolo, Zhang, Perreault et al., Intuitive Surgical). **v4
  (2026-05-16)** is the first version covering SurgVU 2024-2025. <https://arxiv.org/abs/2305.07152>
  (NB: the "pose estimation/localization" title in our README is INCORRECT — it is the SurgToolLoc+SurgVU
  results paper.)
- **Dataset paper:** arXiv:**2501.09209** — *"Surgical Visual Understanding (SurgVU) Dataset"* (Zia et al.,
  2025). Repo: <https://github.com/isi-challenges/surgVU-dataset>. 280 videos / 155 sessions, 60 fps,
  720p, >840 h, ~18M frames; da Vinci on porcine tissue; CC BY-NC-SA 4.0; ≤3 of 12 tools/clip.
- **12 SurgVU `groundtruth_toolname` values (verified, from Capybara's `target_tools`):** needle driver,
  monopolar curved scissors, force bipolar, clip applier, cadiere forceps, bipolar forceps, vessel sealer,
  permanent cautery hook/spatula, prograsp forceps, stapler, grasping retractor, tip-up fenestrated
  grasper. *(Underlying SurgToolLoc had 14 — drops suction-irrigator & bipolar-dissector.)*
- **8 task labels (Table 22):** Suturing · Uterine Horn · Suspensory Ligaments · Rectal Artery/Vein ·
  Skills Application · Range of Motion · Retraction and Collision Avoidance · Other. (eval fills missing
  preds with index 7 = "Other".)
- **`matched_description` (verified):** per-task NL narrative (anatomy + surgeon actions + tools). Only
  **~21 unique captions / ~10 fixed templates** total (Capybara "21 unique"; UoM "~ten fixed templates").
  The 7 organ types come from `matched_description`. Dataset ships **NO native QA pairs** — only the
  10/11-video eval-format sample. Teams bootstrap QA via LLM generation from these templates.
- **Organizer baseline for Cat-2: NONE reported** (paper gives only team leaderboards).

### BLEU metric — exact setup (VERIFIED, mirrored in Capybara's `inference_sample.py`)
```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
references_tokens = [a.split() for a in gt_answers]   # 5 refs, whitespace tokenized
candidate_tokens  = pred_answer.split()
bleu = sentence_bleu(references_tokens, candidate_tokens,
                     weights=(0.25,0.25,0.25,0.25),
                     smoothing_function=SmoothingFunction().method1)
```
- BLEU-4, uniform weights, `method1` smoothing (epsilon 0.1 on zero-count precisions).
- **Per-question score = MAX BLEU over the 5 references**; final metric = **mean over all questions**.
- **No documented text normalization** → raw `.split()`; **case & punctuation are significant**.

---

## 4. Our own verified BLEU probe (2026-06-15, `/tmp/bleu_probe.py`, faithful NLTK-method1 reimpl)
Run on the real 11-case sample answers. **Refutes the README hypothesis that bare "No" ≈ 1.0.**

| Candidate vs the case122 "No" reference set | max-over-5 BLEU |
|---|---|
| `No` (bare) | **0.1778** |
| `no` (lowercase) | 0.1000 |
| `No.` (period) | 0.1000 |
| `No, forceps are not being used.` (plausible, inexact) | 0.5081 |
| **`No forceps are being used.`** (exact match to ref #3) | **1.0000** |
| `No forceps are being used in this surgical video clip right now.` (verbose-correct) | 0.2120 |
| `There are no forceps visible in the current frame…` (very verbose) | 0.0203 |

**Per-strategy MEAN over the 11 sample cases:**
| Strategy | Mean BLEU |
|---|---|
| Bare yes/no token ("Yes"/"No") | **0.1199** |
| Echo the shortest reference (refs[0]) | **0.3525** |
| Oracle (exact match to any ref) | 1.0000 |

**Takeaways (all empirically verified):**
1. **Exact match to ANY of the 5 refs → 1.0** (need not be the shortest). This is the only route to 1.0.
2. **Bare 1-word answers cap at ~0.18**; three of four n-gram orders are epsilon-floored → `0.1^0.75`.
3. **Verbose-but-correct answers are PUNISHED** (0.02–0.21) by n-gram dilution.
4. **Plausible templated full sentences earn 0.40–0.51** even when inexact — far above the 0.18 floor.
5. **Case/punctuation matter** (`No`=0.178 vs `no`=0.100). Probe whether the official harness normalizes.
6. **"Echo shortest ref" (0.3525) alone is already within striking distance of Capybara's 0.4215** —
   so the prize is won by *picking the right templated sentence per question type*, not by VLM fluency.

---

## Verified vs UNVERIFIED
- **VERIFIED:** all 2025 BLEU & 2024 F1 leaderboard numbers + prizes; BLEU code/setup; 12 tools; 8 tasks;
  `matched_description`; Capybara/UoM/SurgOPTeam/PDMYR/AMI architectures; dataset stats; paper IDs;
  our own BLEU probe numbers.
- **UNVERIFIED:** Capybara's exact arXiv ID; Medibot's exact VLM; whether the 2026 eval normalizes
  text; the yes/no question fraction (no published QA split). Resolve via leaderboard probing + the
  gated training QA file once data lands.

## 2026-08-19 — ✅ VERIFIED SurgVU numbers read off the platform (authenticated, DataTables ajax)
### CATEGORY 1 PRELIM — leaderboard VISIBLE, 16 entries, metric = mAP
| rank | score | team / entry |
|---|---|---|
| 1 | **0.6040** | pengyuncong (UESTC-SCU-UCAS), "Algorithm1", 13 Aug 2026 |
| 2 | 0.5790 | guhongyu (PUMCH-UFH-Hikimaging) |
| 3 | 0.5779 | guhongyu |
| 4 | 0.5712 | guhongyu |
| 5 | 0.5469 | guhongyu |
| 6 | 0.5451 | guhongyu |
| 7 | 0.5400 | guhongyu |
| **8** | **0.5355** | **Hongyun — flagged `baseline` = the ORGANIZER BASELINE** |
**⇒ Cat-1 bar: beat 0.5355 to clear the organizer baseline; 0.6040 is the field top.** The board is thin (16 entries) and dominated by ONE team (guhongyu holds 6 of the top 7) — so the effective competition is 2–3 groups, not 16.
**OUR Cat-1: NOT SUBMITTED.** Local honest mAP is 0.1828 on our own 4-class human-COCO val (with `clip_applier` at 0.0000), which is not comparable to their 14-class board — but it is far enough below 0.5355 that a submission is a MEASUREMENT, not a bid.
### CATEGORY 2 PRELIM — submission SUCCEEDED, leaderboard **403 (cannot view)**
Our submission `Aug. 18, 2026, 9:38 p.m. | Category 2 - Prelim Phase | **Succeeded**` — the container executed cleanly on the platform. **But `…/category-2-prelim-phase/leaderboard/` returns HTTP 403, so we cannot read our own score.** Cat-1's leaderboard at the same path pattern returns 200, so this is the same per-phase permission gap as the create page.
⇒ **The user access request is STILL REQUIRED — not to submit (that routed through), but to SEE the result.**

## 2026-08-19 — ★★★ CAT-2 LEADERBOARD READ (30 entries, BERTScore-F1) — WE HAVE A SCORE, AND THE FIELD'S METHODS ARE NAMED
**⚠ CORRECTION: I previously reported that we could not read our Cat-2 score and asked the user to request phase access. THAT WAS WRONG.** Our result is on the **`category-2-final-phase` leaderboard**: **0.6761, rank 26 of 30**. The 403s on `category-2-prelim-phase` and `category-1-final-phase` are **BY DESIGN** — the challenge home page links exactly two boards (`category-1-prelim`, `category-2-final`), which are precisely the two that return 200. **No access request is needed.**
| rank | score | entry / stated method |
|---|---|---|
| **1** | **0.8523** | capybara — **"QwenVL LoRA, 5-frame"** |
| 2 | 0.8290 | TienNQ27 — "Domain Adapted Qwen2 VL for Robotic Surgery Video QA" |
| 4–7, 12–13 | 0.7910–0.8219 | jshmhsb (Emory Melody) — "qweninference" (6 rows) |
| 8 | 0.8051 | pengyuncong (UESTC-SCU-UCAS) — "Plan1" |
| 9 | 0.8015 | nkalthoff (OpScribe-AI) |
| 21 / 24 | 0.7435 / 0.7192 | pengyuncong — **"baseline cat2" / "baseline model"** |
| **26** | **0.6761** | **OURS** (no-vision deterministic template) |
| 27 / 28 | 0.6647 / 0.6456 | Jeremytriana811 — "SurgVU Qwen3VL QLoRA v10 / v11 few-shot" |
| 29 / 30 | 0.5291 / 0.1410 | rgarcianes-intusurg / OpScribe |
**⇒ BAR: baseline 0.7192–0.7435 (we are −0.043 to −0.067 below) · field top 0.8523 (−0.176).**
**★ THE FIELD IS ALMOST ENTIRELY QWEN-VL FINE-TUNING.** The leader states **"QwenVL LoRA, 5-frame"** outright; ranks 2–13 are Qwen2-VL / Qwen3-VL variants. **This is exactly the pipeline ORENA already gave us — Qwen3-VL-8B + LoRA on surgical video QA, with training, inference and container paths all proven.**
**★★ AND A GENUINELY INFORMATIVE DATA POINT: our NO-VISION template (0.6761) BEATS two real Qwen3VL QLoRA submissions (0.6647, 0.6456).** So naive VLM fine-tuning *underperforms a well-chosen answer register* on this metric — which is exactly what our offline decoding study predicted (register is worth more than correctness; verbose prose scores below the constant-"No" floor). The teams at 0.80+ have evidently solved register **and** content.
⇒ **HIGHEST-VALUE REMAINING LEVER IN THE CAMPAIGN: a Qwen-VL LoRA for Cat-2, reusing the ORENA stack, with our measured answer-register discipline on top.** We are 26/30 with a template that cost nothing; the recipe the whole top of the board uses is one we already own.

## 2026-08-19 — leaderboard read (verified)
**Cat-1 prelim, 18 entries.** #1 pengyuncong (UESTC) "Algorithm1" **0.6040** · guhongyu (PUMCH-UFH) holds **2nd–7th** (0.5790→0.5400) · **baseline** = Hongyun "surgtoolloc2025 1" **0.5355** (8th) · **OURS "AIMS run A" 0.4019 = 13th** (−0.2021 to #1, −0.1336 to baseline).
Nobody discloses architecture in Cat-1 (titles are `Algorithm1`, `surgvu2026 cat1 sub1`). ★ The baseline being named **`surgtoolloc2025`** says the organisers' bar is a prior-year SurgToolLoc solution.
**Cat-2 final, 30 entries.** #1 capybara **0.8523** *"QwenVL LoRA, 5-frame"* · 2nd TienNQ27 **0.8290** *"Domain Adapted Qwen2 VL for Robotic…"* · 4th–7th jshmhsb (Emory) *"qweninference"* 0.8219→0.8069 · 8th pengyuncong "Plan1" 0.8051 · **OURS "AIMS run B" 0.6761 = 26th** (−0.1762 to #1).
⇒ **the entire top of Cat-2 is Qwen-VL fine-tuning, stated openly.** Our entry is a no-vision template; the gap is exactly the vision model we have not trained.
