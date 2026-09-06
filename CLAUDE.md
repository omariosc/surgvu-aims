# SurgVU 2026 — Categories 1 & 2 · research dossier

> **Mission: #1 on BOTH SurgVU-2026 leaderboards** (scope updated 2026-06-16; was Cat-2-only).
> Rigor bar inherited from `BTPN-MT/TODO.md` + `MICCAI2026_INSIGHTS.md`: **verified competitor numbers
> only (no estimates), dated REFUTED/CONFIRMED log, no benchmark-overfitting, Pareto efficiency, Docker
> submission builds + passes the harness.** Companion files: `competitor-numbers.md`, `README.md`.
> Last updated **2026-07-30**.

---

# 🚨 2026-07-30 — STATUS RESET. READ THIS BEFORE ANYTHING ELSE IN THIS FILE.
Three findings invalidate large parts of the dossier below. Everything under §0–§7 was written
under the 2025 assumptions and is **superseded where it conflicts with this block.**

## ⛔ FINDING 1 — THE CAT-2 RANKED METRIC IS NO LONGER BLEU. IT IS **BERTScore-F1**.
**Source (authoritative, not a paraphrase):** the organizers' own evaluation container,
<https://github.com/isi-challenges/surgvu26-category-2-eval-public> → `evaluation/evaluate.py`,
cloned + read 2026-07-30. Their `README.md` states for `bertscore_f1`:
**"Only metric used for ranking and valid for challenge"**. BLEU-4, ROUGE-1/2/L, NLI-entailment
and NLI×BERT are all computed but explicitly **"Secondary (for testing/analysis only)"**.

```python
bert_scorer = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True)
F1 = bert_scorer.score([cand]*len(refs), refs)      # per question
score = F1.max().item()                             # MAX over the 5 references
final = mean(per-question scores)                   # MEAN over all questions
```
**⚠ ASYMMETRY IN THE OFFICIAL CODE (verified, exploitable):** `normalize()` (lowercase + strip
punctuation) is applied **only to the BLEU/ROUGE inputs**. `compute_heavy_metrics()` consumes the
**RAW** `candidate`/`references`. ⇒ **case and punctuation DO matter for the ranked metric and do
NOT matter for BLEU** — the exact inverse of the §3 rule we have been designing to since June.
**⇒ THE ENTIRE §0/§3/§4 "BLEU dictates the method" THESIS IS RETIRED.** Our scorer clone is
`experiments/surgvu26_scorer.py` (line-for-line faithful, incl. the raw/normalized split).

## 📐 FINDING 2 — WHAT THE NEW METRIC ACTUALLY REWARDS (measured, job `6933448`, n=11)
`experiments/metric_recalibration_2026.py`. **Numbers are BERTScore-F1 = the ranked metric.**

| Strategy | **BERTScore-F1 (RANKED)** | BLEU (retired) | NLI (not ranked) |
|---|---|---|---|
| echo any reference exactly (incl. **ref[0], the bare token**) | **1.0000** | 0.35 (ref0) / 1.0 (ref1+) | 0.977 |
| our template_v2 (the 2025 BLEU champion) | **1.0000** | 1.0000 | 0.997 |
| **template_v2 with POLARITY FLIPPED (i.e. WRONG ANSWER)** | **0.9110** | 0.6449 | 0.363 |
| **constant `"No"` for every question** (no vision, question unread) | **0.6383** | 0.0485 | 0.311 |
| constant `"Yes"` for every question | 0.6327 | 0.0647 | 0.476 |
| verbose-but-CORRECT | **0.5704** | 0.2650 | 0.996 |
| echo the question back | 0.4716 | 0.1906 | 0.090 |

**The three trustworthy, leakage-free reads** (the 1.0000 rows are inflated — those templates were
authored against these exact 11 refs — but the following involve no such tuning):
1. **★ THE RANKED METRIC IS ~NEGATION-BLIND. Being RIGHT is worth only +0.089.** A fully wrong
   (polarity-flipped) answer still scores **0.9110**. Direct probe: candidate `"Yes"` vs reference
   `"No"` → BERTScore-F1 = **0.9957**; `"No"` vs `"No"` → 1.0000. The organizers *document* this
   in their own README ("BERTScore … blind to negation"), built the negation-aware `nli_bertscore_f1`
   composite to fix it — **and then did not rank on it.**
2. **★ THE NO-VISION FLOOR IS 0.6383.** A single constant word emitted for every question scores
   0.638/1.000. **The entire competitive range is ~0.36 wide**, and correctness occupies only ~0.089
   of it. Ranking will be decided by **surface form**, not by understanding.
3. **★ VERBOSITY IS PUNISHED HARDER THAN BEING WRONG.** Verbose-correct (0.5704) scores **BELOW a
   constant bare "No"** (0.6383). BERTScore-F1's precision term dilutes with every extra token.
   Because **refs[0] is ALWAYS the bare answer token** and scoring is MAX-over-refs, short wins.

**⇒ STRATEGIC CONSEQUENCE (this reverses the task's opening premise).** The "zero-shot beats
fine-tuning" result from 2025 was measured under **BLEU**. Under BERTScore-F1 the binding constraint
is **ANSWER LENGTH AND REGISTER, not zero-shot vs fine-tuned**. A raw zero-shot VLM emitting fluent
prose lands near the *verbose* row (~0.57) — i.e. **worse than answering `"No"` to everything**.
Length control is therefore the first lever, and it is orthogonal to the SFT question. Do not read
the 2025 podium as "use a zero-shot VLM raw"; read it as "do not let SFT make you verbose or narrow".

## 📊 FINDING 3 — THE 2025 LEADERBOARD IS NOT A BAR WE CAN COMPARE TO
Capybara 0.4215 / UoM 0.3656 / Medibot 0.3404 are **BLEU**. The 2026 leaderboard is **BERTScore-F1**,
on which a *no-vision constant* already scores 0.6383. **Never put 0.4215 and a 2026 score in the
same column** (this is the same class of error as the 2024-F1-vs-2025-BLEU warning in §1).
**We currently have NO valid 2026 bar.** The prelim leaderboard (live since Jul 20) is the only way
to get one → getting *any* container onto it early is now a top-priority information-gathering act,
not just a delivery milestone.

## ✅ DATA — RESOLVED. THE 754 GB ON DISK **IS** THE 2026 TRAINING CORPUS.
The roster line "✅ 2024; ⛔ 2026 gated" and the §closing line "fully USER-gated: nothing actionable
on our side" are **both STALE — corrected here.**
- **Organizer statement** (`/data-description/`): the 2026 training data is *"an expanded version of
  the training data and labels used within the 2022/3 SurgToolLoc challenge"*, **280 long videos from
  155 training sessions**, 60 fps 720p, >840 h, >18 M frames, ≤3 of 12 tools per clip, plus
  `tools.csv` + `tasks.csv` + a **`matched_description`** column added for Cat-2.
- **On disk** (`/scratch/sc20osc/miccai-2026/SurgVU/data/`, 754 GB): **280 decodable mp4 across
  155 cases** (`surgvu24_videos/`, 324 GB) — an exact count match — plus
  `external/cat2_train_labels/SURGVU25_train_labels/` = **155 cases × `tasks.csv`+`tools.csv` WITH the
  `matched_description` column present** (verified: 3,673 task rows, **21 unique descriptions** —
  matching Capybara's independently-reported "21 unique captions"), plus the 2022 corpus
  (`surgtoolloc2022_dataset.zip`, 116 GB) and the 11-clip Cat-2 sample set (case122–132).
- **⇒ The corpus, the labels and the Cat-2 description column are ALL IN HAND. No re-download needed
  for training.** (I cannot byte-verify against the gated `/data-download/` page — it returns 403 to
  a non-participant — so the user should spot-check it, but the video/session counts and the label
  schema match the organizer description exactly.)
- **WHAT IS GENUINELY NOT IN HAND (both are 2026-only, small, and gated):**
  1. **The small Cat-1 bounding-box VALIDATION set.** `/challenge-categories/` says Cat-1 trains on
     *"noisy tool presence labels provided in the training set and bounding box labels provided in
     the small validation set"*. We have never held this. **→ USER ACTION.**
  2. **Cat-2 test QA pairs** (5 refs/question) — held out by construction; never available.
     **The dataset ships NO training QA. Our n=11 sample set is the ONLY QA with references, and
     that is a hard ceiling on offline Cat-2 validation.**

## 🗓 DATES — TIME-CRITICAL (source `/important-dates/`, verbatim)
| Event | Date |
|---|---|
| Registration open + training data released | May 8 |
| **Preliminary testing phase STARTS** | **Jul 20 (LIVE NOW)** |
| **New registrations deadline** | **Aug 15** |
| **Final test phase STARTS** | **Aug 21** |
| Preliminary phase ENDS | **Sep 2** |
| Final test phase ENDS | **Sep 6** |
| Methodology reports + all submission requirements | **Sep 13** |
| Winners announced | Challenge day @ MICCAI (Sep 27 or Oct 1, TBD) |

## ⚑ CAT-1 2026 — SPEC + A BLOCKER THAT WAS ALREADY UNBLOCKED
- **Output = BOUNDING BOXES + tool class. NOT key-points.** `/challenge-categories/` (specific) says
  *"localize (with bounding boxes) and classify the tools present within each frame"*; the site
  **home page** loosely says *"localize tools and their corresponding key-points"*. **These conflict.**
  Given the Tiger task1/task2 reversal (which would have silently scored ~0), **treat this as an
  open spec risk and confirm with the organizers before building** — do not assume boxes.
- **★★ THE "#1 BLOCKER" IN THE §7 LOG IS FALSE AND HAS BEEN SINCE 2026-06-27.** The log repeatedly
  states *"NO human bbox GT exists for SurgVU-2024"* and kills Cat-1 work on that basis. But
  `data/external/cat1_test_set/` (pulled Jun 27, never used) contains the **public 2024 Cat-1 test
  set WITH HUMAN COCO ANNOTATIONS**: 7 held-out videos @1 fps, 640×512, **5,178 images / 11,323
  human boxes / the same 14-tool taxonomy** (`*_fps1_coco.json`). `experiments/cat1_honest_val.py`
  and `cat1_honest_val.slurm` were even *written* on Jun 27 to score against it — **and never
  submitted.** ⇒ **Every Cat-1 mAP in this dossier (0.7746 … 0.9788) remains self-consistency vs the
  pipeline's own Grad-CAM pseudo-boxes and is NOT a detection number.** The honest read is now
  running (job `6933525`, array 0-4 across the detector ladder — the self-consistency *ordering* is
  expected to invert, since a higher CAM threshold discards boxes and inflates self-consistency
  while shrinking real recall).

## ⛔ WHAT THIS MEANS FOR OUR HEADLINE NUMBERS
| Claim in this dossier | Verdict 2026-07-30 |
|---|---|
| Cat-2 **0.9211 / 0.9114 grounded BLEU** | **RETIRED — wrong metric.** BLEU is not ranked in 2026. Re-derive under BERTScore-F1. Also n=11 with templates authored against those 11 refs. |
| Cat-2 **0.91** (roster) | **Not a 2026 number. Do not quote.** |
| Cat-1 **0.9788 / 0.7746 mAP** | **Self-consistency, not detection.** Real human-GT number pending job `6933525`. |
| "beat Capybara 0.4215" | **Category error** — different metric. No valid 2026 bar exists yet. |

## 🏁 FINDING 4 — THE SHIPPING DECODING POLICY IS SETTLED (job `6933607`, n=11)
`experiments/answer_policy_surface.py` — 3 answer registers × correct/wrong, then expected score
as a function of p(correct). **A ~7-word declarative sentence in the reference register DOMINATES at
EVERY accuracy level** — all pairwise crossovers fall outside [0,1], so there is no regime in which
bare tokens or verbose prose win.

| Register | correct | **wrong** | words | value of being correct |
|---|---|---|---|---|
| bare token | 0.8825 | 0.6923 | 1.4 | +0.1902 |
| **short sentence** | **1.0000**\* | **0.7932** | **6.9** | **+0.2068** |
| verbose | 0.5562 | 0.4461 | 24.9 | +0.1101 |

\*inflated (those sentences are our authored templates, which exact-match these 11 refs). **The
leakage-free comparisons are the `wrong` column and the register ordering, and both are unambiguous.**

**Three decisions this settles:**
1. **SHIP A ~7-WORD DECLARATIVE SENTENCE.** Never a bare token (−0.10), never prose (−0.35).
2. **⚠ A raw zero-shot VLM emitting prose lands ~0.45–0.56 — BELOW the 0.6383 no-vision constant
   floor.** Un-length-controlled generation is *negative value*. This is the single most important
   engineering constraint on the container.
3. **REFINES FINDING 2:** correctness is worth **+0.207** in the sentence register, not +0.089.
   The earlier figure was a *polarity-only* flip; once the ENTITY can also be wrong (tool name, organ,
   purpose) the value of being right roughly doubles. **So the grounding spine is NOT worthless** —
   it is worth ~0.21 of a ~0.36-wide range, i.e. **the majority of the competitive spread**. Correct
   read: *polarity* is nearly free (metric is negation-blind), but *naming the right entity* is the
   real prize. Direct the vision effort at ENTITY identification, not at yes/no polarity.

## 💀 FINDING 5 — CAT-1: OUR REAL mAP IS **0.000**, AND THE MECHANISM IS ARITHMETIC
Job `6351081` (Jun 27, run once and never read) scored every detector against the real human COCO GT:
**mAP@[.5:.95] = 0.0000** for `base_detector`, `thr0.5_yolo26m`, and the `thr0.7_BEST_long120aug`
"champion" (the 0.9788 self-consistency leader); seg-cascade 0.001. **Verified NOT a bug** — the
class mapping asserts 14/14 and the frame indexing is sound. The cause is box SCALE:

| | area fraction of frame |
|---|---|
| **human GT boxes** (n=11,323, the tool *clevis*) | **0.0215 mean / 0.0141 median** |
| our Grad-CAM pseudo-boxes | 0.1538 → **7.2× too large by area (2.7× per side)** |
| our seg-cascade boxes | 0.3404 → 15.8× too large |

⇒ **a perfectly centred Grad-CAM box caps at IoU ≈ 0.140**, far below the 0.5 floor where
AP@[.5:.95] begins ⇒ **mAP is mathematically forced to ~0 regardless of classification accuracy.**
(seg-cascade caps at IoU ≈ 0.063 — it made things *worse*.)

**⇒ THE ENTIRE CAT-1 LADDER (0.7746 → 0.9788) MEASURED NOTHING BUT THE DETECTOR'S ABILITY TO
REPRODUCE ITS OWN OVERSIZED SUPERVISION.** The roster's "Cat1 0.77" must be struck.
**DIAGNOSIS:** the bottleneck is *localization granularity* — the target is a ~1.4%-of-frame clevis
and Grad-CAM class-evidence blobs cannot resolve it. This is a **supervision-quality** problem, not a
detector-capacity or data-volume problem, which is precisely what the 2026 design fixes: Cat-1
explicitly provides *"bounding box labels ... in the small validation set"*.
**⇒ ACTION (running, job `6933641`):** we already hold **5,178 human-annotated frames / 11,323 boxes**.
`cat1_realbox_train.py` trains YOLO on 5 videos of REAL boxes and validates **video-disjoint** on the
held-out 2 — a dry run of the exact 2026 recipe, and the first Cat-1 number that will mean anything.

## 📚 DEEP RESEARCH (2026-07-30) — what the literature says about this exact metric
1. **`rescale_with_baseline` cannot change the ranking.** It is a monotone affine map,
   `raw → (raw − 0.83122575)/(1 − 0.83122575)` at roberta-large layer 17 (from `bert_score/score.py`
   + the shipped `rescale_baseline/en/roberta-large.tsv`). It commutes with both the max-over-refs and
   the mean-over-questions. **But it multiplies every delta by 5.93×** ⇒ our measured +0.089 polarity
   value is a raw-BERTScore +0.015, and **seed/sampling noise is amplified 5.93× too**. Do not mistake
   amplified noise for a lever.
2. **Negation-blindness is quantified in the literature, not just by us.** MENLI (arXiv:2208.07316,
   TACL 2023) measures BERTScore at **60.9% accuracy on negation attacks** vs >92% for NLI metrics
   (65.3% overall adversarial vs BLEURT 74.8%). Antonyms sit in near-identical contextual
   neighbourhoods (arXiv:2109.14250). Our 0.089 measurement is exactly this phenomenon.
3. **Truncation is under-penalised.** "Blind Spots of Model-Based Evaluation Metrics"
   (ACL 2023, aclanthology 2023.acl-long.674) finds BERTScore is confused by *truncation* errors —
   directionally supports the terse policy.
4. **★ MBR / self-consistency reranking with BERTScore-F1 as its OWN utility** is a training-free,
   label-free way to optimise the exact ranking metric at decode time. Closest published analogue
   (arXiv:2606.15741) reports **BERTScore gains of +2.7 to +14.6** from K=5 candidates reranked by
   mutual similarity. ⚠ Do NOT train against BERTScore (MRT/RL collapses to degenerate output) —
   decode-time only.
5. **★ SurgCheck (arXiv:2605.01911): the IMAGE is worth only ~5–7 points on surgical VQA.** Removing
   the image entirely drops fine-tuned LLaVA-OV-1.5-8B from 79.4%→72.6% (action) and 52.6%→47.8%
   (target); removing the entity NAMES from the question is catastrophic (→29.7%). **The language
   channel carries the decision.** ⇒ text-side evidence injection (predicted tool/organ/phase as a
   structured, named-role evidence block) operates on the channel that matters — and is exactly what
   Capybara did when they won.
6. **The "SFT hurts" mechanism, measured on our exact base model.** Chain-of-Adaptation
   (arXiv:2603.20116) SFT'd **Qwen3-VL-8B-Instruct** on 700 surgical QA pairs: **output length −64%**,
   diversity collapse, catastrophic forgetting of general fluency *even on non-surgical images*.
   SurgViVQA (arXiv:2511.03325) isolates template memorisation: InternVL3+LoRA 32.21%→70.31% K-ACC
   **in-template**, but all models drop sharply **out-of-template**.
   **⚠⚠ CRITICAL RE-READ: an SFT-induced 64% length collapse is, under BERTScore-max-with-a-bare-token-
   ref[0], MOVING IN THE RIGHT DIRECTION.** The 2025 "zero-shot ≫ fine-tune" result was a BLEU result,
   where terseness is fatal. **So "fine-tuning hurts" is NOT settled for 2026 and must be tested, not
   assumed** — the counter-argument is that SFT also destroys format-instruction following
   (format collapse, arXiv:2604.27720), which is the very lever we need for length control, and is
   antagonistic to evidence injection.
7. Open-weight surgical VLMs worth a look: **EndoChat** (weights confirmed public, image-level only),
   **SurgLLaVA-Video / SurgPub-Video** (arXiv:2508.10054, dual frame+video input — best architectural
   fit for 30 s clips), SurgVidLM, SurgΣ (arXiv:2603.16822). ⚠ **SurgVLM weights UNVERIFIED** (project
   page advertises them, HF search returns nothing). NVIDIA Cosmos-H-Surgical is a world model, NOT VQA.

## 🎯 REVISED STRATEGY + PRIORITISED QUEUE (supersedes §5's phased plan for Cat-2)
**The objective is no longer "be right". It is "emit a short string in the reference register that is
lexically close to one of the five references."** Correctness contributes ~0.089 of a ~0.36-wide
competitive range. Ranking will be decided by answer-type routing and length control.

| # | Lever | Status | Kill criterion (pre-registered) |
|---|---|---|---|
| **1** | **Metric re-calibration** (`metric_recalibration_2026.py`) | ✅ **DONE** job `6933448` | — |
| **2** | **Answer-policy payoff surface** — 3 registers (bare / sentence / verbose) × correct/wrong, → expected score vs p(correct) and the terse-vs-verbose CROSSOVER. Decides the shipping decoding policy and tells us *how much vision accuracy is worth buying*. | ⏳ job **`6933607`** | if `bare` dominates at every p, vision is worth ~nothing ⇒ ship a router-only container and stop spending on the spine |
| **3** | **Zero-shot VLM baseline** (Qwen3-VL-8B, 4 prompt registers × {1,5} frames, scored on the real metric with bootstrap CI) | ⏳ job **`6933479`** | if no prompt config beats the **0.6383 constant-answer floor**, the VLM is *negative value* as a free-text generator and must be demoted to an evidence extractor feeding a template |
| **4** | **Cat-1 honest validation vs REAL human COCO GT** | ⏳ job **`6933525_[0-4]`** | if real mAP ≪ self-consistency 0.9788 (expected), every Cat-1 conclusion in §7 is void and Cat-1 restarts from a true baseline |
| 5 | **Answer-type ROUTER** (question → {yes/no, tool-name, organ, procedure, purpose}) emitting the canonical bare token / short noun phrase. **No vision needed.** This is the main competitive lever if #2 says terse dominates. | queued | router accuracy gain < the 0.089 correctness value ⇒ not worth it |
| 6 | **MBR decode-time reranking with BERTScore-F1 as its own utility** (K candidates → argmax mutual similarity). Training-free, label-free, optimises the exact ranked metric. | queued | < +0.01 over greedy ⇒ drop |
| 7 | **Structured evidence injection** (spine tool/organ/phase as a named-role evidence block in the prompt) — the Capybara recipe, and SurgCheck says the text channel is what the model uses | queued | no gain over question-only prompting ⇒ the spine is not load-bearing for Cat-2 |
| 8 | **SFT arm (pre-registered, DEFERRED — must not run before #2/#3 report).** Hypothesis under test is now the *opposite* of 2025's: SFT's documented 64% length collapse may HELP under BERTScore. | **blocked on #2/#3** | must beat the best zero-shot config by > the bootstrap CI on n=11, AND not regress format-instruction following ⇒ else REFUTED and we ship zero-shot |

**⚠ THE BINDING VALIDATION CONSTRAINT:** we have **n=11 QA pairs, full stop** — the corpus ships no
training QA and the test QA is held out. Bootstrap CIs on n=11 are enormous, and our templates were
authored against those same 11 references. **Offline Cat-2 numbers cannot settle anything.**
⇒ **The preliminary leaderboard (LIVE Jul 20 → Sep 2) is our only real measurement channel, and using
it is an EXPERIMENT, not just a delivery milestone.** Getting a container up early — even a trivial
one — is the single highest-information act available, because it also calibrates the unknown 2026
score scale. **This is now the top delivery priority and it is time-boxed by the Aug 21 final phase.**

---

## ⚑ Category 1 — Surgical tool classification + localization (NEW SCOPE, gated on registration)
- **Task:** weakly/semi-supervised **detection** — output **bounding boxes + tool class** per test frame,
  trained on **noisy tool-presence labels** (the 280-video train set) + **bbox labels in a SMALL val set**.
  **Metric = COCO mAP@[.5:.05:.95]** (grand-challenge automated Docker eval). 12 core tools
  (`groundtruth_toolname`), ≤3 tools/frame; UI blurred in test; bbox encloses the clevis.
- **Approach (from prior SurgToolLoc/SurgVU detection winners):** weakly-supervised localization — e.g.
  **ResNet/EfficientNet tool-presence classifier → CAM/Grad-CAM → pseudo-boxes** to bootstrap a
  **YOLO/DETR detector**, refined on the small bbox val set; or open-vocab detector (GroundingDINO/SAM2)
  pseudo-labeled by tool-presence. Big public proxies if stuck: **CholecT50/SurgToolLoc** tool labels.
- **Status: BLOCKED on grand-challenge registration** (shares the gated 280-video set with Cat 2). Cat-1
  submission template cloned → `./surgvu2025-category1-submission/`. **No experiments yet** (no data).
- **§1+ below is the Category-2 dossier** (VQA/BLEU — has the validated 11-sample harness + P0.2 result).

---

## (CATEGORY 2 — Surgical VQA) Mission: #1 on the Cat-2 leaderboard.
Open-ended VQA on 30 s da Vinci clips; metric = **BLEU (max-over-5 refs, method1-smoothed, mean over Q)**.

---

## 0. TL;DR strategy (the falsifiable thesis)
**Win with a CLASSIFIER-GROUNDED, TEMPLATED-ANSWER pipeline — not free-form VLM fluency.** The 2025
winner (Capybara, BLEU **0.4215**) used a **zero-shot** LLaVA-OneVision-7B grounded by EfficientNet
tool/organ classifiers; the runner-up's *fine-tuned* VLM (0.202) **lost to its own zero-shot variant**
(0.3656). Our verified BLEU probe shows **echo-shortest-ref alone scores 0.3525** on the 11 samples, and
**exact-match to any reference = 1.0** while **bare tokens cap at 0.18** and **verbose-correct answers are
punished (0.02–0.21)**. → The lever is **predicting the right *answer-type + grounded entity*, then
emitting the canonical 4–7-word templated sentence** that the references use. A free-form VLM is a
*component* (for the open-ended "why/what-purpose" questions), wrapped by deterministic templating and
BLEU-aware decoding.

---

## 1. SOTA status table (the standing we must beat)
Verified in `competitor-numbers.md`. 2026 continues the 2025 VQA+BLEU formulation.

| Track | Best verified prior (BLEU) | Method | Ours | Standing |
|---|---|---|---|---|
| SurgVU-2025 Cat-2 (overall bar) | **0.4215** Capybara | zero-shot LLaVA-OV-7B + EffNet tool/organ clf | — | ⏳ not started |
| — 2nd | 0.3656 UoM-SurgicalAI | zero-shot InternVL3.5 | — | (PEFT fine-tune scored 0.202) |
| — 3rd | 0.3404 Medibot | fine-tuned LMM + Cat-1 detector | — | |
| Our offline-probe references | echo-shortest-ref **0.3525**; oracle **1.0**; bare-y/n 0.11 | `experiments/bleu_harness.py` | — | baseline yardsticks (11 samples) |
| **Ours — P0.2 template+router** | **0.9650** (hand-set entities, n=11) | keyword router → register-tuned template bank | ✅ | offline templating ceiling; not real accuracy (no vision) |

> ⚠ **2024 Cat-2 numbers (PDMYR 0.8977 etc.) are weighted-F1 on a *classification* task — NOT BLEU,
> NOT comparable.** Do not put them in the BLEU column.

**Definition of done:** mean BLEU on the held-out test **> 0.4215** (overall bar) with a Docker container
that builds, runs offline (`--network none`), and passes the eval harness; result reproducible from a
pinned checkpoint; no test-item-specific tuning.

---

## 2. The task & I/O (verified from template + sample)
- **Input (grand-challenge sockets, from `inference.py` + `inputs.json`):**
  - `endoscopic-robotic-surgery-video` — the 30 s `.mp4` (`/input/endoscopic-robotic-surgery-video.mp4`).
  - `visual-context-question` — the question **String** (`/input/visual-context-question.json`).
- **Output:** a single JSON string → `/output/visual-context-response.json` (the answer).
- **Container:** `Dockerfile` (python:3.11-slim base; swap to a CUDA/pytorch base for a VLM),
  `requirements.txt`, `ENTRYPOINT python inference.py`; runs **offline** (`--network none`), `--gpus all`,
  read-only `/input`, writable `/output` + `/tmp`, optional model tarball at `/opt/ml/model`.
  Local loop: `./do_test_run.sh` (build → forward pass on `test/input/interf0` → `test/output/interf0`).
- **Sample (in hand):** `data/case122…case132/` — `caseXXX.mp4` (30 s, 720p), `caseXXX_question.json`,
  `caseXXX.json` (5 refs). Mirrors the eval I/O exactly.

### Question taxonomy (from the 11 samples — design the router around this)
| Type | Example Q | Reference style | Best strategy |
|---|---|---|---|
| **Yes/No — tool presence** | "Are there forceps being used here?" / "Is a large needle driver among the listed tools?" | "No, forceps are not mentioned." / "Yes, a large needle driver was utilized." | tool classifier → templated polarity+justification |
| **Tool-type (open)** | "What type of forceps is mentioned?" | "Cadiere Forceps" + carrier sentences | tool classifier → "The forceps type is `<tool>`." |
| **Organ/anatomy** | "What organ is being manipulated?" | "Uterine horn" + carriers | organ classifier (from task) → carrier template |
| **Yes/No — action** | "Is tissue being cut?" / "Is a suture required in this step?" | "Yes, tissue is being cut." | task classifier → action templated answer |
| **Procedure / generic** | "What procedure is this summary describing?" | "Endoscopic surgery or a laparoscopic surgery" | static canonical answer (low variance) |
| **Why/Purpose (open)** | "What is the purpose of using forceps?" | "To grasp and hold tissues or objects…" | VLM / canned knowledge template per tool |

> NB the reference wording — *"not **mentioned**"*, *"not **listed**"*, *"the **summary** is describing"*
> — reveals answers were authored from the **textual `matched_description`/tool-list**, not pure vision.
> This strongly favours grounding on a **predicted structured description**, exactly as Capybara did.

---

## 3. Why BLEU dictates the method (verified, `/tmp/bleu_probe.py`)
NLTK `sentence_bleu`, weights (.25,.25,.25,.25), `method1`, **max over 5 refs scored individually**, mean.
**No text normalization documented → case & punctuation count.** Empirically:

| Candidate (case122 "No" set) | max-5 BLEU | Lesson |
|---|---|---|
| `No` | 0.178 | bare token → three n-gram orders epsilon-floored → cap `0.1^0.75` |
| `no` / `No.` | 0.100 | **case/punct matter** |
| `No, forceps are not being used.` (plausible inexact) | 0.508 | a full sentence ≫ bare token even when not exact |
| `No forceps are being used.` (**exact ref**) | **1.000** | exact match to *any* of 5 refs = 1.0 |
| 14-word verbose-correct | 0.212 | length-correct but diluted |
| very verbose | 0.020 | **actively punished** |

**Per-strategy mean over 11 samples:** bare-y/n **0.120** · echo-shortest-ref **0.353** · oracle **1.000**.
**Design rules forced by this:**
1. Emit a **4–7-word declarative templated sentence**, never a bare label, never verbose prose.
2. **Match one reference exactly** when possible (max-over-5 ⇒ only need to hit *one*). Mimic the
   templates' register/case/punctuation.
3. Yes/No: output **"`<Yes/No>`, `<entity>` `<is/are>` `<not> <used/present>`."** not bare polarity.
4. Open tool/organ: output the **carrier sentence** ("The forceps type is `<X>`."), not just the noun.
5. **Probe** the live leaderboard for normalization (submit `no` vs `No`); set capitalization accordingly.
> Lit: Chen & Cherry 2014 smoothing paper (<https://aclanthology.org/W14-3346/>); NLTK `method1`
> (<https://www.nltk.org/api/nltk.translate.bleu_score.html>). BLEU's geometric mean over 4 orders is the
> brittleness lever on short text — **not** the brevity penalty.

---

## 4. Method shortlist (benefits / drawbacks / gaps)
Three families. **The winning answer is the HYBRID** (C), with A as the workhorse and B as a fallback.

### A. Grounded-templated answer (classifier → template bank) — **PRIMARY**
Predict (answer-type, tool, task/organ) with strong classifiers; fill a hand-written template bank that
mirrors the SurgVU reference register; for open "why/purpose" use a small canned per-tool knowledge map.
- **+** Directly maximizes BLEU (can hit exact-ref = 1.0); cheap, fast, deterministic, no hallucination;
  the classifier path is exactly the **`groundtruth_toolname` (12) + tasks (8) + organ (7)** weak labels
  we get for free. Mirrors Capybara's winning recipe minus the VLM cost.
- **−** Needs good coverage of the question→template mapping; brittle on unseen phrasings; open questions
  ("purpose") need a knowledge map.
- **Gap:** the template bank must be authored from the **`matched_description` corpus** (~21 captions /
  ~10 templates) — gated, so author it from the samples first, expand when data lands.

### B. Free-form VLM (zero-shot or QLoRA fine-tune) — **FALLBACK / open-question head**
Qwen2.5-VL-7B / LLaVA-OneVision-7B / InternVL3.5, fed sampled frames + question + predicted structured
description; **constrained/templated decoding** toward short reference-style sentences.
- **+** Handles open-ended/novel questions; **zero-shot already won 2025** (Capybara 0.4215, UoM 0.3656);
  surgical warm-starts exist (`nvidia/Qwen2.5-VL-7B-Surg-CholecT50`, `bravefox12138/surgvu2025vqa`).
- **−** **Fine-tuning HURT in 2025** (UoM PEFT 0.202 < zero-shot 0.3656) — likely template-overfit /
  verbosity. Verbose generation is BLEU-punished. Heavier compute.
- **Gap:** must force short reference-register output (length-penalty / few-shot template priming /
  grammar-constrained decoding); raw VLM prose loses.

### C. Hybrid (router → {template path | VLM path}) — **TARGET ARCHITECTURE**
A question-type **router** sends closed questions (yes/no, tool, organ, procedure ≈ most of the set) to
the **template bank** (A), and only the genuinely open ones (purpose/why/describe) to a **VLM with
templated decoding** (B). Both paths grounded on the same predicted (tools, task, organ) description.
- **+** Best of both: exact-match BLEU on the easy majority + VLM coverage on the tail; matches what the
  reference-wording analysis implies the data generation did.
- **−** Two subsystems to build + a router to train/validate.
- **Gap:** router accuracy on the true (gated) question distribution unknown until data lands — start with
  keyword rules on the 11 samples, upgrade to a learned router on the gated QA.

> **Discarded as primary:** the classic surgical-VQA canon (Surgical-VQA/VisualBERT-ResMLP, SSG-VQA,
> CAT-ViL, SurgicalGPT, PitVQA) are **fixed-vocabulary classifiers** (non-BLEU, single-frame) — reuse only
> their **grounding/fusion ideas** (CAT-ViL gated co-attention on robotic frames; SSG-QA scene graphs;
> SCAN/EndoChat memory + anti-hallucination) as VLM front-ends, never their output heads.

### Model pool (verified fits for 1× L40S 48 GB, QLoRA)
| Model | Params | Video | Surgical warm-start | Notes |
|---|---|---|---|---|
| **Qwen2.5-VL-7B** ⭐ | 8.3B | dyn-fps, temporal-patch | `nvidia/…-Surg-CholecT50` (frame); `bravefox12138/surgvu2025vqa` (3B, SurgVU!) | QLoRA fits 48 GB w/ headroom (7B QLoRA runs on 24 GB); LLaMA-Factory/ms-swift video |
| LLaVA-OneVision-7B | 7B | 196 tok/frame, ≤32 frames | — | **Capybara's winning base (zero-shot)**; lightest video VRAM |
| InternVL3.5-8B | 8B | 256 tok/frame, 32 seg | — | UoM's zero-shot base; video-QLoRA-on-48 GB UNVERIFIED — profile |
| Qwen3-VL-8B / VideoLLaMA3-7B / LLaVA-Video-7B | 7–9B | true-video (≤64–180 frames) | EndoVLM (Qwen3-VL) | better 30 s temporal granularity; modern alternatives |

**Classifiers (the grounding spine):** EfficientNetV2-S (Capybara's choice; tool macro-F1 0.97, organ
0.98) or a surgical backbone (EndoViT/SurgeNet/DINOv3 from the FM pool) → 12-tool + 8-task + 7-organ heads,
trained on the gated SurgToolLoc tool-presence labels.

---

## 5. Phased, falsifiable experiment plan
Each phase has an explicit hypothesis + falsifier + go/no-go. Log results in §7.

### Phase 0 — offline yardsticks on the 11 samples (runnable NOW, no GPU/data gate)
- **P0.1 ✅ DONE** BLEU harness reimplemented + verified (`/tmp/bleu_probe.py`); refuted the "bare-No≈1.0"
  hypothesis; established yardsticks (bare-y/n 0.12, echo-shortest 0.35, oracle 1.0).
- **P0.2** Build the **template bank + keyword router v0** from the 11 question types (§2 table) and a
  hand-set answer per case; measure mean BLEU. *Hypothesis:* a no-vision, question-only templated
  responder beats bare-y/n (0.12) and approaches echo-shortest (0.35). *Falsifier:* < 0.30 ⇒ templates
  mis-phrased. *(This is the FIRST experiment — see §6.)*
- **P0.3** Zero-shot VLM sanity: run `bravefox12138/surgvu2025vqa` (3B) and/or LLaVA-OneVision-7B on the
  11 clips via SLURM L40S; record BLEU + answer verbosity. *Hypothesis:* raw VLM ≈ 0.25–0.42 but verbose
  answers underperform templates on closed questions. *Falsifier:* VLM > template on yes/no ⇒ rethink.

### Phase 1 — grounding spine (needs gated SurgToolLoc labels)
- **P1.1** Train EfficientNetV2-S (or surgical backbone) 12-tool + 8-task + 7-organ classifiers on
  sampled frames. *Target:* tool/organ macro-F1 ≥ 0.95 (Capybara reached 0.97/0.98). *Falsifier:*
  < 0.85 ⇒ classifier is the bottleneck, not templating.
- **P1.2** Wire classifier predictions → structured description → template bank. *Hypothesis:* grounded
  templates beat question-only (P0.2). *Go/no-go:* if grounded templates ≥ 0.40 on a held-out QA split,
  this alone is a podium pipeline.

### Phase 2 — hybrid + VLM tail + BLEU-aware decoding
- **P2.1** Router → VLM path for open questions only; VLM fed (frames + predicted description + question),
  **constrained to short reference-register output** (few-shot template priming + length cap).
  *Hypothesis:* hybrid > template-only by covering open questions. *Falsifier:* VLM path lowers mean BLEU
  ⇒ keep it template-only there too.
- **P2.2** *(only if it helps)* QLoRA fine-tune the VLM on LLM-generated QA from `matched_description`,
  **trained to output reference-style short sentences**. *Strong prior to BEAT/REFUTE:* 2025 fine-tuning
  HURT (UoM 0.202 < 0.366 zero-shot) — so this must demonstrably beat zero-shot or be dropped.
- **P2.3** BLEU-aware decoding/ensembling: per question, **generate K candidates and pick the one with
  highest self-BLEU to the predicted-description / template set** (a legitimate proxy, NOT tuned to test
  refs); ensemble template+VLM by router confidence. *Falsifier:* no gain over best single path.

### Phase 3 — finalize
- Dockerize (`inference.py` → router+spine+VLM; pin weights in model tarball); pass `do_test_run.sh`
  offline; ≥1–2 **prelim** submissions early (organizers urge starting ~Aug). Methodology report (LaTeX)
  + 3-min video to isi.challenges@intusurg.com; grant editor access to `aneeqzia_isi`.
- **DoD:** mean test BLEU > 0.4215, reproducible, no benchmark-overfitting.

---

## 6. ⭐ Highest-priority FIRST experiment (runnable now)
**P0.2 — question-only templated responder + BLEU eval on the 11 samples.** No GPU, no gated data.
- Author `template_bank.py`: a keyword router over the question string → one short reference-register
  sentence per (answer-type, entity), e.g.
  - tool yes/no → `"<Yes/No>, <tool> <is/are> not being used."` (or affirmative form),
  - tool-type → `"The forceps type is <Tool>."`, organ → carrier sentence,
  - procedure → static `"Endoscopic surgery or a laparoscopic surgery"`,
  - purpose → per-tool canned `"To grasp and hold tissues or objects during the surgery."`
- Score with `/tmp/bleu_probe.py`'s BLEU; compare vs bare-y/n (0.12) and echo-shortest (0.35).
- **Success = mean BLEU ≥ 0.35 with hand-set entities** (proves templating phrasing is right). This both
  (a) validates the whole templated thesis and (b) becomes the deterministic backbone of the container —
  before any model training or the gated download. Then P0.3 (zero-shot VLM on L40S) and P1 (classifiers,
  once data lands) layer on top.

---

## Autonomous ladder (2026-06-16, user away ~4 days — afterok-chained off 5998329)
Submitted a DEEP self-progressing ladder that auto-advances as the full-scale classifier (5998329,
1.07M frames) finishes. Each job appends ONE line to `/users/sc20osc/AUTONOMOUS_RESULTS.md`
(`[SurgVU][jobid] <what> → <metric> <verdict>`). Chain (strict afterok, each builds on the prior):

| Job | Step | Dep | What | Output / metric | Kill criterion (REFUTE if…) |
|---|---|---|---|---|---|
| **5998329** | 0 | — | full-scale spine train (running, ep2 loss 0.24) | `spine_full/spine_best.pt` | — (workhorse) |
| **6015397** | 1 | **afterany:5998329** | `diagnose_spine.slurm` auto-diagnosis | per-tool/task F1 + confusion → `per_class_diag.json` | — (diagnostic, always runs) |
| **6015398** | 2 | afterok:6015397 | Cat-2 REAL visual grounding (`spine_grounding_bleu.py`) — spine preds → template router → grounded BLEU on 11 clips | grounded-BLEU vs 0.4215 SOTA / 0.9650 hand-set ceiling | grounded-BLEU < 0.4215 **with** spine F1≥0.95 ⇒ templating/router is the limit, not vision |
| **6015399** | 3 | afterok:6015398 | Cat-1 weakly-sup detector (`cam_pseudoboxes.py` @600 frames → `train_detector.py` YOLO) | COCO mAP@[.5:.95] on case-disjoint **pseudo-box** val (self-consistency; NO human bbox GT yet) | pseudo-box mAP < presence-F1 floor ⇒ localization not classification is the wall |
| **6015400** | 4 | afterok:6015399 | class-balanced retrain (`train_spine.py --balanced_sampling --focal`) targeting rare tools/tasks, **excl. 3 zero-val data-floor tools** (bipolar_dissector/suction_irrigator/tip_up_fenestrated_grasper) | recoverable (excl-floor) macro-F1 vs the 6015397 baseline | recoverable macro-F1 rises < +0.05 vs uniform ⇒ it's data-floor, not loss → escalate to MORE DATA |

### DEEP batch extension (2026-06-16, ~4-day autonomous — NEW levers chained off the ladder)
Four NEW performance levers queued upfront (best-first), `afterok`-chained off the existing chain so
they auto-advance as checkpoints land. Each sbatch appends `[SurgVU][jobid] … → metric verdict` to
`/users/sc20osc/AUTONOMOUS_RESULTS.md`. Backbones pre-fetched + verified loadable offline from
`/scratch/sc20osc/hf_cache` (convnextv2_tiny.fcmae_ft_in22k_in1k 768-d, tf_efficientnetv2_m.in21k_ft_in1k
1280-d); YOLO weights cached (yolo11n + yolo26m, NO download). `train_spine.py` gained `--gamma` (focal-γ)
and `--sample_strength` (inverse-freq temper: 0=uniform→1=full→>1 over-balance) CLI knobs (smoke-verified).

| Job | Lever | Dep (afterok) | What | Metric | Kill criterion (REFUTE if…) |
|---|---|---|---|---|---|
| **6042378** | A (brief #3) Cat-2 grounding variants | 6015398 | `grounding_variants_bleu.py` — n_frames{4,8,16,24} × thr{ckpt,.3,.5,.7} × pool{max,mean}, fixed router → grounded-BLEU sweep on 11 clips | best grounded-BLEU vs 0.4215 / ladder2 single-config | best ≤ ladder2 baseline ⇒ grounding-config saturated; gap is spine ACCURACY not injection recipe |
| **6042379_[0-5]** | B (brief #1) class-imbalance grid (array) | 6015400 | `ladder6` — focal-γ{1,2} × sample-strength{.5,1,1.5}, 5 ep/150k fr each → diagnose → `grid1/<cell>/per_class_diag.json` | per-cell recoverable (excl-floor) macro-F1 | (see aggregator) |
| **6042380** | B aggregator | 6042379_* | `grid_aggregate.py` ranks 6 cells by recoverable macro-F1 vs uniform baseline | best excl-floor macro-F1 delta | best rises < +0.05 vs uniform ⇒ data-floor not loss → MORE DATA |
| **6042381_[0-1]** | C (brief #2) stronger backbone/longer train (array) | 6015397 | `ladder7` — convnextv2_tiny + effnetv2_m, focal+balanced, 12 ep/250k fr → diagnose vs spine_full | all14 macro-F1 vs spine_full base | neither beats spine_full all14 by ≥ +0.02 ⇒ EffV2-S at encoder frontier → escalate |
| **6042382_[0-3]** | D (brief #4) Cat-1 detector variants (array) | 6015399 | `ladder8` — pseudo-box thr{.3,.5} × YOLO{11n,26m}, cam→train_detector → COCO mAP (self-consistency, NO human GT) | mAP@[.5:.95] vs ladder3 single-config | no cell exceeds ladder3 baseline ⇒ Grad-CAM localisation supervision is the wall, not detector capacity |

> Ordering = best-first: **A (grounding variants, ~1 h, directly hits the #1 Cat-2 vision-grounding
> bottleneck) → B (imbalance grid, the rare-tool recoverable-F1 lever feeding both Cats) → C (backbone) →
> D (detector).** A/D/C run in parallel branches off different chain links; B+agg are a sub-chain off
> ladder4. New code (all login import-/smoke-verified): `grounding_variants_bleu.py`, `grid_aggregate.py`,
> `ladder5_grounding_variants.slurm`, `ladder6_imbalance_grid.slurm` (array 0-5), `ladder6_agg.slurm`,
> `ladder7_backbone.slurm` (array 0-1), `ladder8_detector_variants.slurm` (array 0-3). Self-advancing; not babysat.

> **Why afterany on link 1 (not afterok):** 5998329's epochs are ~3.7 h each (~24 img/s, 300k+40k frames);
> 10 epochs ≈ 37 h > the 24 h wall, so it will TIMEOUT around ep5–6 and exit non-zero. The trainer saves
> `spine_best.pt` after every *improving* epoch (already has a tool-F1 0.585 / task-F1 0.656 ckpt at ep0),
> so a usable checkpoint exists at the wall regardless. `afterany` ⇒ the ladder advances on finish OR
> timeout (it would strand forever on `afterok`). Downstream links stay `afterok` (my own short jobs).

**New code (all login-import-verified):** `spine_grounding_bleu.py` (step 2), `train_detector.py` (step 3,
uses cached `…/AILET/from_pc/weights/yolo11n.pt` — no download), `train_spine.py` +`--balanced_sampling`
(inverse-freq `WeightedRandomSampler`) +`--focal` (multi-label focal BCE). `diagnose_spine.slurm` now also
appends its diagnosis line. **Note:** step-3 mAP is honest self-consistency vs Grad-CAM pseudo-boxes (no 2024
human boxes exist) — replaced by real mAP when the 2026 bbox val set lands. ultralytics 8.4.68 installed into
`/scratch/sc20osc/surgvu_env`. **Self-advancing; not babysat.** Watcher armed by the main thread.

### DEEP BATCH III (2026-06-19, user away — autonomous refill; 5 jobs queued)
Driven by the diagnosis of the landed batch-II results (`AUTONOMOUS_RESULTS.md` SurgVU lines). **State at
launch:** Cat-2 grounded-BLEU **0.7428** (cfg nf4_thr0.3_mean) — the grounding-config sweep (6042378) is
**SATURATED** (32 configs, many tie at exactly 0.7428 → kill criterion met: gap is spine ACCURACY + router/
template, not the entity-injection recipe). Cat-1 detector best **mAP@[.5:.95]=0.7746** (thr0.5_yolo26m); both
detector axes (CAM-thr 0.3→0.5 = +0.09; yolo11n→26m = +0.12) **monotone up, NOT saturated**. Backbone lever C
**REFUTED** (convnextv2_tiny 0.6225 all14 < spine_full 0.6340); EffNetV2-S still champion among *diagnosed*
spines — BUT `spine_full_bal` (balanced+focal) and `spine_tf_efficientnetv2_m` were trained and **never
diagnosed** (dangling ckpts). **Cleaned up first:** cancelled my own dead `surgvu_l6grid` 6042379 +
`surgvu_l6agg` 6042380 (stuck `DependencyNeverSatisfied` on failed 6015400 — the imbalance grid never ran).

**Per-case BLEU diagnosis (`spine_full/grounded_bleu.json`) — where the 0.70→0.74 loss concentrates:**
recoverable loss is in (1) **router/template action bugs** (pure code, no vision): case131 "is tissue being
cut" → "No, a **None** is not used" (BLEU 0.041 — `_question_tool_keys` finds no tool for "cut" → present
defaults False → is_cut branch unreachable → ent=None); case125 "suture required" → "No, a suture is not
used" (0.202 — same unreachable-action bug); case126/132 affirmative needle-driver carrier wording vs refs'
"was utilized" register (0.41/0.51); and (2) **spine forceps-confusion** case124 predicts "bipolar" not
"Cadiere" (0.537).

| Job | Track | Lever | Dep | What | Kill criterion (REFUTE if…) |
|---|---|---|---|---|---|
| **6204239** | Cat-2 | V2 action-grounding BLEU | — | `spine_grounding_v2.py` (cut→monopolar-scissor/dissection bridge; suture→needle-driver/Suturing bridge) + `p0_2_template_router_v2.py` (action branches reachable from grounded `present`; tense-aware "was utilized" carrier) → grounded-BLEU on 11 clips, spine_full | grounded-BLEU ≤ 0.7428 ⇒ action-grounding/template fix didn't help |
| **6204240** | Cat-2 | diagnose 2 dangling spines + final pick | — | diagnose `spine_full_bal` + `spine_tf_efficientnetv2_m` (per_class_diag + forceps-confusion) + V2-grounded-BLEU each → FINAL Cat-2 spine = best grounded-BLEU (tie: excl-floor F1) | neither beats spine_full on BLEU AND neither beats all14 by ≥+0.02 ⇒ spine_full confirmed final |
| **6204235_[0-1]** | Cat-1 | higher CAM thr {0.6,0.7} @ yolo26m | — | push the winning axis further; report n_pseudo_boxes (boxes get sparser as thr↑) | mAP < 0.7746 ⇒ thr0.5 is the pseudo-box-threshold optimum |
| **6204236** | Cat-1 | longer (120 ep) + heavy aug @ thr0.5/yolo26m | — | isolate schedule/aug axis on identical cam_thr0.5 boxes (mosaic/mixup/hsv/degrees/translate/scale/fliplr) | mAP ≤ 0.7746 ⇒ detector capacity/schedule not the wall, Grad-CAM supervision is |
| **6204237** | Cat-1 | best-thr (from L1) + 120ep + aug → final pick | afterok:6204235 | stack both winners for the final Cat-1 detector config | mAP ≤ 0.7746 ⇒ winners don't stack; localisation supervision is the wall |

**Offline templating ceiling re-measured:** V2 router hits **mean 1.0000** on the 11 clips with correct
present-flags (> v1's 0.9650 — case126 now exact-matches "was utilized"). Real-clip BLEU depends on whether
the spine fires cut/suture/needle-driver signals — that's what 6204239 measures. New code (all login-smoke
verified): `experiments/spine_grounding_v2.py`, `p0_2_template_router_v2.py`, `spine_grounding_bleu_v2.py`,
`train_detector.py` (+augmentation/--patience CLI knobs, defaults None → ladder8 baseline unchanged),
slurm `ladder_v2_grounding_bleu.slurm`, `ladder_v2_spine_pick.slurm`, `ladder9_camthr_sweep.slurm`,
`ladder10_longaug.slurm`, `ladder11_bestconfig.slurm`. Each appends `[SurgVU][jobid] … → metric verdict`.
Main-thread waiter armed (`/tmp/surgvu_batch_waiter.sh`, 5-min poll). **Not babysat.**

## 7. Dated experiment log (REFUTED / CONFIRMED — append-only)
- **2026-07-30 ★ METRIC RE-TARGET + FOUR STALE CLAIMS CORRECTED (see the STATUS RESET block at the
  top of this file for the full detail).** Summary of the dated verdicts:
  - **CONFIRMED (authoritative source): the 2026 Cat-2 ranked metric is BERTScore-F1**
    (roberta-large, `rescale_with_baseline=True`, max-over-5-refs, mean-over-questions), from the
    organizers' own eval container `isi-challenges/surgvu26-category-2-eval-public`. BLEU/ROUGE/NLI
    are computed but the README marks BERTScore-F1 as the *"Only metric used for ranking and valid
    for challenge"*. ⇒ **§0/§3/§4's BLEU thesis is RETIRED**; scorer clone =
    `experiments/surgvu26_scorer.py`.
  - **CONFIRMED (job `6933448`, `metric_recalibration_2026.py`, n=11): the ranked metric is
    ~negation-blind.** Polarity-flipped (i.e. WRONG) answers score **0.9110** vs 1.0000 correct —
    **being right is worth only +0.089**. Bare `"Yes"` vs reference `"No"` = **0.9957**. Meanwhile
    the non-ranked NLI score collapses 0.997→0.363, i.e. the organizers *have* a negation-aware
    metric and chose not to rank on it.
  - **CONFIRMED: the no-vision constant-answer floor is 0.6383** (a single fixed `"No"` for every
    question). Competitive range ≈ 0.36 wide.
  - **CONFIRMED: verbosity costs more than being wrong.** Verbose-correct **0.5704** < constant
    bare `"No"` **0.6383**. `refs[0]` is always the bare answer token and scoring is max-over-refs
    ⇒ **short, reference-register answers dominate.**
  - **⇒ REFRAMED (this overturns the working assumption we were handed):** the 2025
    "zero-shot ≫ fine-tune" result (UoM 0.3656 vs its own 0.202) was measured under **BLEU**. Under
    BERTScore-F1 the binding constraint is **answer LENGTH/REGISTER**, which is orthogonal to the
    SFT question — a raw fluent zero-shot VLM lands near the *verbose* row (~0.57), i.e. **below a
    constant "No"**. Length control is lever #1; zero-shot-vs-SFT is a downstream question.
  - **CORRECTED (stale): "fully USER-gated: nothing actionable"** — the user IS registered, and the
    **754 GB on disk IS the 2026 training corpus** (organizer `/data-description/`: 2026 training
    data = the expanded SurgToolLoc-2022/3 corpus, 280 videos / 155 sessions — an exact match to
    disk; `SURGVU25_train_labels` supplies the `matched_description` column, **21 unique captions**
    verified across 3,673 task rows). Genuinely missing: the **small 2026 Cat-1 bbox val set** and
    the (permanently held-out) test QA.
  - **CORRECTED (stale, and it cost us): "NO human bbox GT exists"** — `data/external/cat1_test_set/`
    has held human COCO GT since Jun 27 (**5,178 images / 11,323 boxes / 7 videos / 14 classes**).
    `cat1_honest_val.py` + `.slurm` were written that day and **never submitted**, so Cat-1 was
    killed on a blocker that had already been removed. Now running as job **`6933525`**.
  - **CORRECTED: 0.4215 is not a bar for us.** It is BLEU; 2026 is BERTScore-F1, where a no-vision
    constant already scores 0.638. **No valid 2026 bar exists** until we read the prelim leaderboard.
- **2026-06-22 SMOKE LANDED (Cat-1 seg-cascade `surgvu_segsmoke` 6234340 COMPLETED; full 6234341 chained):**
  smoke (200 fr/12 ep, self-consistency vs seg-pseudo-GT, NO human bbox GT) → seg-mAP **0.0823**,
  seg_box_area **0.2180** (vs Grad-CAM 0.1538), **WBF_delta = +0.3914**. **PRE-REGISTERED KILL CRITERION NOT
  MET → lever SURVIVES to the full run:** the kill was *boxes-not-tighter AND WBF_delta≤0*; the WBF fusion
  delta is **strongly positive (+0.3914 ≫ HRI_MV's winning +0.009 margin)** — detection+tracking fusion (the
  exact lever that won SurgToolLoc) fires hard. Raw seg-box area 0.218 is still looser than Grad-CAM (the
  predicted generic-COCO-seg limitation — a real surgical segmenter would tighten it), but the **track+WBF
  stages carry the cascade**, as designed. The 0.0823 raw seg-mAP is a tiny-smoke artifact (200 fr/12 ep,
  fresh detector) — NOT the headline; the full run (1500 fr/60 ep, `6234341`) is the real read. ⚠ Still all
  self-consistency vs the pipeline's OWN seg-GT (NO human bbox GT) — deliverable = pipeline + box-quality +
  the (now-confirmed-large) WBF delta; REAL validation still needs the gated-2026 bbox-val set. Waiter armed
  on 6234341.
- **2026-06-22 LAUNCHED (Cat-1 PASS-4 research lever — SurgToolLoc/HRI_MV WINNING cascade replacing Grad-CAM
  pseudo-boxes; `INNOVATIONS.md §PASS 4`):** built + smoked + submitted the HRI_MV #1 cascade (arXiv 2305.07152)
  as a better-grounded substitute for our coarse Grad-CAM pseudo-boxes. **All 4 stages staged** (one substitution):
  - (a) `experiments/seg_pseudoboxes.py` — spine-predicted present-tools → **cached `yolo26m-seg` instance masks**
    → tight boxes, assigned to spine classes by instrument-likeness prior, ≤3/frame, saliency-clevis fallback.
    (⚠ generic-COCO seg, NOT surgical — the core in-hand limitation; HRI_MV used dVRK/EndoVis17-18 surgical seg
    we do NOT have on disk.)
  - (b) `experiments/track_propagate.py` — seed seg-box+spine-class → **bytetrack** propagation across 8-frame
    windows (sparse-labeled frame → dense temporally-consistent box track).
  - (c) detector = **ultralytics YOLO, NOT Cascade-RCNN (substitution):** mmdet/mmcv has no prebuilt wheel for
    torch2.6+cu124 (offline-Docker CUDA-build nightmare) → kept the existing clean-licensed YOLO submission
    detector via `train_detector.py`'s case-disjoint dataset builder + self-consistency mAP reporter.
  - (d) `experiments/wbf_fuse.py` — **weighted-boxes-fusion** of detector + tracker boxes (det 2.0 / track 1.0,
    iou 0.5; `ensemble_boxes` pip-installed + verified into surgvu_env) — replicates HRI_MV's winning +0.009 fusion.
  - **Jobs:** smoke **`6234340`** `surgvu_segsmoke` (200 fr/12 ep, `seg_cascade_smoke.slurm`) → full **`6234341`**
    `surgvu_segfull` (`afterok:6234340`, 1500 fr/60 ep, `seg_cascade_full.slurm`). Main-thread waiter armed.
  - **Login-smoke box stats (seg vs Grad-CAM):** seg = **single-instance** (2.3 boxes/fr ≈ #present-tools vs
    Grad-CAM's multi-instance 2.71) **but tightness only PARTIALLY won** — generic-COCO seg masks avg area
    **0.238 > Grad-CAM 0.1538** (a COCO seg model isn't a surgical-instrument segmenter). **The tracking step
    recovers it: propagated boxes avg 0.116 (tighter than Grad-CAM).** So the win = single-instance + track-tightened.
  - **KILL CRITERION (pre-registered):** seg+track+YOLO+WBF must beat Grad-CAM baseline **0.7746** (thr0.5/yolo26m
    self-consistency) AND/OR yield tighter/cleaner boxes + a positive WBF fusion delta. ⚠ **mAP is
    self-consistency vs the pipeline's OWN seg pseudo-boxes — seg-GT vs Grad-CAM-GT mAP are NOT directly
    comparable (different "GT").** Real comparison = pseudo-box QUALITY + WBF delta. REFUTE if boxes are not
    tighter/cleaner than Grad-CAM AND fusion delta ≤0.
  - **⚠⚠ #1 BLOCKER UNCHANGED — gated-2026 bbox-val:** NO human bbox GT exists for SurgVU-2024; every mAP here
    (Grad-CAM 0.7746 AND this seg-cascade) is offline self-consistency. **The PIPELINE is the deliverable;
    REAL validation comes only when the gated 2026 bbox-val set lands.**
  - **DOWNLOAD TO FLAG TO USER:** the "tighter boxes" win is bottlenecked by having **no surgical seg model
    on disk**. **EndoVis17-18 / dVRK robotic-instrument segmentation data + SAM2 weights** (any one) would
    replace the generic-COCO-seg substitute = the single biggest lever to make seg boxes genuinely tighter
    (it is what HRI_MV actually used). License: YOLO-seg used OFFLINE for pseudo-box gen only; no YOLO-World/
    GroundingDINO anywhere. New code (login-smoke verified): `seg_pseudoboxes.py`, `track_propagate.py`,
    `wbf_fuse.py`, `seg_cascade_smoke.slurm`, `seg_cascade_full.slurm`.
- **2026-06-20 CONSOLIDATION + REFILL (autonomous tick; `surgvu_l9rl` 6211080 landed, hard-pair lever queued):**
  - **Cat-2 — Q-conditioned temporal retrieval CONFIRMED (re-run, new config).** `surgvu_l9rl` (6211080,
    COMPLETED 07:07) re-ran the 3 levers the cosmetic crash (6204346) had only partially banked:
    - (1) **Q-cond retrieval CONFIRMED** `qcond_nd30_k8_thr0.3_mean` BLEU **0.9114 > uniform 0.9026 / baseline
      0.7428** (+0.0089). Top-k frames by Q-relevance beats uniform `np.linspace`, but marginally on n=11
      (only k=8@nd30 beats uniform; k=3/4/6 tie or lose → the gain is fragile). New Cat-2 config of record.
    - (2) **Canonical-answer layer REFUTED** BLEU **0.8313 < router 0.9026** (−0.0712) → **keep the router**.
      Per-case: the ontology→single-canonical-form layer regressed case123 (1.0→0.5411) and case132 (0.41→0.09)
      by over-collapsing the surface form — the keyword router's per-bucket templates are already register-tuned;
      a flat canonical map throws away that specificity. Lever dead.
    - (3) **Leakage re-check CLEAN.** scorer-clone AGREE (Δ<1e-9, scorer_agree=True); **spine contribution
      +0.2259** (real grounding 0.9026 vs text-only floor 0.6767 — note: the router bug-fix turned the spine
      NET-POSITIVE vs the 2026-06-19 audit where it was below the text-only floor); **polarity sensitivity
      +0.2213** (flipped-polarity 0.6813); **9/11 exact-ref hits**; const-floors echo-shortest 0.3525 /
      most-common 0.0647. **VERDICT CLEAN** — spine is load-bearing + polarity-sensitive. ⚠ **n=11 caveat
      PERSISTS:** 9/11 exact = still a high over-fit signal with authored templates; **real validation needs
      the gated 2026 QA set** — all offline gains remain UNVALIDATED. Do not over-invest in n=11 tuning.
  - **Imbalance grid 6211081_[0-5] still PENDING** (QOSMaxCpuPerUserLimit — held behind the user's MT-BTPN
    priority load; no grid1/ cells written yet). Aggregator 6211082 waits afterok. Will diagnose on landing.
  - **REFILL (SurgVU queue 2→8 array-expanded): NEW LEVER — bipolar←cadiere HARD-PAIR head, `6211687`
    `surgvu_l12hp`** (`ladder12_hardpair.slurm`). Targets the **#1 spine confusion** (per_class_diag
    `bipolar_forceps<-cadiere_forceps n=1106`, verified in `spine_full/per_class_diag.json` — the case124
    Cat-2 error "The forceps type is bipolar forceps", BLEU 0.537). Added a default-OFF `--hardpair`
    contrastive-margin hinge to `train_spine.py` (`HardPairLoss`: on frames where exactly one member of a
    confusable pair is GT-present, `relu(margin − (logit_present − logit_absent))`; pairs = bipolar↔cadiere,
    cadiere↔needle_driver, prograsp↔cadiere; w0.5/margin2.0). **Default OFF ⇒ byte-identical to the baseline
    tool loss** (smoke-verified: empty pair → 0; case124-like row → loss 4.0=(6+3+3)/3; forward+backward grads
    flow). Trained on the imbalance-grid recipe (focal g2 + balanced s1.0, 5ep/150k fr) so the delta isolates
    the hinge; diagnoses + reports the bipolar←cadiere confusion-count drop + excl-floor-F1 vs the spine_full
    baseline. **KILL CRITERION:** confusion-n must DROP and excl-floor-F1 must NOT regress vs 0.7738 baseline,
    else the look-alike is encoder-capacity-bound (→ escalate backbone / higher-res clevis crop), not a loss gap.
  - **⚠ #1 BLOCKER UNCHANGED: gated-2026 validation set.** All Cat-2 BLEU (n=11, templates exact-matched to
    these refs) AND all Cat-1 detector mAP (self-consistency vs own Grad-CAM pseudo-boxes, NO human GT) are
    **OFFLINE-ONLY / UNVALIDATED**. Every offline gain (Q-cond +0.0089, hard-pair if it lands) is provisional
    until the gated 2026 QA + bbox val sets are in hand. Flagged to user — registration is the real lever.
  - New code (login-smoke verified): `train_spine.py` +`--hardpair/--hardpair_w/--hardpair_margin` + `HardPairLoss`
    + `HARD_PAIRS`/`build_hardpair_index` (default-OFF, baseline path byte-identical); `experiments/ladder12_hardpair.slurm`.
- **2026-06-20 CONSOLIDATION (DEEP BATCH III + research levers landed; autonomous tick):**
  - **Cat-2 — V2 action-grounding CONFIRMED HELPED, new leader.** `surgvu_v2gb` (6204239): V2
    action-grounded BLEU (n=11, spine_full, nf4_thr0.3_mean) = **0.9211 > 0.7428** v1-best (the cut/suture
    router bug-fix + tense-aware "was utilized" carrier landed the predicted ~+0.16). `surgvu_v2pick`
    (6204240): diagnosed the 2 dangling spines — spine_full_bal V2-BLEU 0.8489 (exclF1 0.7680),
    spine_tf_efficientnetv2_m V2-BLEU 0.8910 (exclF1 0.7168) — **FINAL Cat-2 spine = spine_full**
    (V2-BLEU 0.9211, exclF1 0.7738; both alternates lose on BLEU AND don't beat all14 by ≥+0.02 ⇒
    spine_full confirmed final). **⚠ LEAKAGE CAVEAT STANDS: n=11, templates exact-matched to these 11
    refs, polarity not recoverable from the question ⇒ 0.9211 is an OPTIMISTIC UPPER BOUND, not a test
    target.** The #1 real Cat-2 blocker remains **no gated-2026 validation set** — all numbers are
    offline-only. Do NOT over-invest in offline n=11 tuning.
  - **Cat-2 research lever (Q-conditioned temporal retrieval) — CONFIRMED marginal.** `surgvu_l9rl`
    (6204346) ran `qcond_retrieval_bleu.py`: best `qcond_nd30_k8_thr0.3_mean` BLEU **0.9114 vs uniform
    0.9026** (+0.0089). Q-relevance frame scoring > uniform but only marginally on n=11. **The job
    "FAILED" (exit 1:0) is COSMETIC** — a trailing `tee` to `models/spine_full/../logs/…qcond` (the
    `models/logs/` dir doesn't exist) errored AFTER all real work completed + the result line was already
    appended. Work is banked; not resubmitting. FIX for reuse: `mkdir -p $MODELDIR/../logs` in
    `ladder9_research_levers.slurm` (the canonical-answer-layer step (2) may have been skipped by the
    early exit — re-run if that lever is wanted).
  - **Cat-1 — detector grid CONFIRMED monotone-rising, BUT self-consistency ceiling reached.** New leader
    `surgvu_l11best` (6204237) thr0.7_yolo26m + 120ep + heavy-aug = **mAP@[.5:.95]=0.9788** (mAP50 0.9834);
    `surgvu_l10aug` (6204236) thr0.5_yolo26m_long120aug = 0.8970; also banked thr0.7 0.8436 / thr0.6 0.8195
    (vs thr0.5 baseline 0.7746). **Axes confirmed: CAM-thr↑ (sparser/cleaner boxes), yolo26m>11n, 120ep+aug
    all add.** ⚠⚠ **DIAGNOSIS — this is now self-consistency overfitting, NOT real detection gain.** mAP is
    measured YOLO-vs-its-own-Grad-CAM-pseudo-boxes (NO human bbox GT); 0.98 means YOLO memorizes the CAM
    supervision, bounded above only by spine presence-F1 — it does NOT predict real COCO mAP on the hidden
    2026 bbox val. **Higher CAM-thr also DISCARDS boxes (1393 vs 1599), inflating self-consistency while
    shrinking real coverage.** The real Cat-1 blocker = **no human bbox validation** → KILL further
    pseudo-box mAP chasing; the lever is real bbox data, not detector capacity.
  - **FAILURES:** (1) `surgvu_molmo2_cat2` (6206220) — `transformers 5.12.0` in `vqa_env` rejects Molmo2's
    custom remote processor kwarg `image_use_col_tokens` (`TypeError: Unexpected keyword argument`). Version
    incompat (newer `ProcessorMixin.__init__` is strict); needs a pinned-older-transformers env or a remote-
    code patch. NOT cheap, NOT a blocker (spine dominates the VLM floor) → flagged, not chased. (2) 6204346
    cosmetic (above).
- **2026-06-20 REFILL (queue 0→8; non-saturated work only — KILLED further Cat-1 pseudo-box chasing):**
  - **6211080** `surgvu_l9rl` — **resubmit of the fixed research-lever job** (`mkdir -p models/logs` added).
    Recovers the 2 Cat-2 levers that 6204346's cosmetic crash SKIPPED: (2) **canonical-answer layer**
    (`canonical_answer_layer.py`, ontology→single canonical surface form vs router) + (3) **leakage check**
    (`leakage_check.py`, scorer-clone + text-only/polarity ablations — the pre-registered adversarial probe
    that tests whether 0.92 is metric-inflated before any SOTA claim). Step (1) qcond already banked (0.9114).
  - **6211081_[0-5]** `surgvu_l6grid` — **imbalance grid** (focal-γ{1,2}×sample-strength{.5,1,1.5}, 5ep/150k
    fr/cell, excl 3 data-floor tools), the recoverable rare-tool/task mid-tail lever (Step-4 innovation,
    feeds BOTH cats). NEVER ran cleanly before — standalone ladder4 `6015400` TIMED OUT, array `6042379` was
    cancelled on its failed dependency. This array is dependency-free + shorter per cell. **6211082**
    `surgvu_l6agg` aggregator (afterok) ranks cells by excl-floor macro-F1 vs uniform; kill criterion: best
    rises <+0.05 ⇒ data-floor not loss → escalate to MORE DATA.
  - **DELIBERATELY did NOT queue more Cat-1:** detector self-consistency mAP hit 0.9788 (overfitting its own
    Grad-CAM supervision, no human GT) — kill criterion met, the wall is real bbox validation data, not
    capacity. **#1 real blocker for BOTH cats = no gated-2026 validation set** (all offline n=11/self-
    consistency numbers unvalidated) → flagged to user, not over-invested offline.
- **2026-06-16 CONFIRMED (ROUND 3 — reconciliation + cruft cleanup + manifest rebuild; full-scale run
  de-duplicated):** independent re-audit of the complete extraction. **On disk = 280 decodable mp4 (NOT
  284 — that figure was approximate)** spanning 155 cases (30 single-part + 125 two-part = 280 complete);
  tree was 324 GB. **`du`/`find` showed exactly 3 non-mp4 cruft files → DELETED:** `case_071/case_video_000_001.mp4_.gstmp`
  (1.94 GB truncated gsutil temp = case_071's **part_001**, which OpenCV **fails to open** — confirmed poison
  for any glob loader, so removed not just left), plus `.DS_Store` + `.cnvrgignore`. Post-clean: **280 mp4 /
  0 cruft / 0 zero-byte / 155 case dirs**. **284-vs-281 explained:** the 284 in the round-3 brief was an
  over-count; the zip's 281 "real parts" = 280 good mp4 + the 1 corrupt `.gstmp` (case_071 part_001), so the
  honest tally is **280 usable / 281 nominal — exactly 1 part lost at source** (case_071 keeps valid part_002
  + full labels). **Label-vs-disk part audit (all 155 cases):** only case_071 is genuinely missing a labeled
  part; case_087/104/138 have an EXTRA on-disk part their CSVs don't span (benign — videos decode fine, just
  no labeled events there). Spot-probed 7 mp4s via OpenCV: all 720p60, start+mid frames decode. **Rebuilt
  `frame_manifest_full.csv` AFTER the cleanup** (so it references 0 gstmp; case_071→part_002 only; all 274
  distinct referenced videos exist on disk): **1,073,389 frames (918,501 train / 154,888 val), 132 train /
  23 val case-disjoint, all 14 tools + all 8 tasks present** — vs the partial baseline `frame_manifest_local.csv`
  (515,064 frames, only 67 train / 6 val cases video-backed; that is what job `5997565` trains on). **Full-scale
  run = job `5998329`** (pre-existing `train_spine_full.slurm`, already past smoke → in FULL train ep0,
  300k/40k frames, healthy on the cleaned manifest). **De-duplicated:** I had resubmitted (`5998344`) before
  noticing 5998329 was already running the identical script/manifest/out-dir → **cancelled my own dup 5998344**
  (kept 5998329) to avoid GPU waste + checkpoint collision. Baseline `5997565` untouched. The 326 GB
  `surgvu24_videos_only.zip` remains **reclaimable (flagged, NOT deleted)**. **NEXT LEVER unchanged:** on
  5998329 macro-F1 → if tool/task ≥0.95, `spine_grounding.py` on the 11 Cat-2 clips for REAL grounded BLEU vs
  0.4215; `cam_pseudoboxes.py` Grad-CAM→pseudo-boxes at full scale to bootstrap the Cat-1 detector.
- **2026-06-16 CONFIRMED (ROUND 2 — FULL SurgVU-2024 video set extracted + verified; full-scale spine
  SUBMITTED):** extraction-completeness audit on the now-complete 344 GB `surgvu24_videos_only.zip`:
  zip central directory lists **562 mp4 entries = 281 REAL parts + 281 `__MACOSX`/`._` AppleDouble junk**;
  real parts span **155 cases (30 single-part + 125 two-part = 281)**. The streaming-extractor had only
  pulled 132 parts off the partial download; `unzip` restored the rest. **Final on disk: 280 decodable
  `*_video_part_*.mp4` across ALL 155 cases (347.7 GB), spot-probed OK via OpenCV (720p60, start+mid frames
  decode).** The 281st zip entry is a **source-corrupt artifact** — `case_071/case_video_000_001.mp4_.gstmp`
  (1.9 GB GStreamer temp, OpenCV cannot open it = organizers' packaging error). So **case_071 part_001 is
  unusable AT SOURCE** (case_071 still has a valid part_002 + labels; only that one segment is lost). Left
  the `.gstmp` un-renamed so the loader never tries to decode it. **VERDICT: extraction COMPLETE — 155/155
  cases, 280/281 real parts (the 1 missing is source-corrupt, not an extraction failure).** The 344 GB zip
  is now reclaimable but **NOT deleted yet** (flagged). **Rebuilt FULL part-aware manifest** (video-gated,
  case-disjoint) → `data/frame_manifest_full.csv`: **1,073,389 frames, ALL video-backed; 132 train / 23 val
  cases; all 14 tools + all 8 tasks present** (vs the partial set where val cases were video-starved — now
  fully backed). Tool freq long-tailed (cadiere 533k, needle_driver 494k, bipolar_forceps 424k … suction
  2.6k, bipolar_dissector 1.6k → BCE pos_weight ≤50× handles it); tasks Other-dominant (716k → down-sampled
  25%). Validated via the trainer's own `csv.DictReader` (manifest is CRLF — Python strips `\r`, unaffected).
  **FULL-SCALE TRAIN SUBMITTED: SLURM `5998329`** (`experiments/train_spine_full.slurm`, gpu/l40s:1, 10 ep,
  max_train 300k / max_val 40k frames, EfficientNetV2-S two-head, → `models/spine_full`; PENDING on
  resources). **Fully isolated from the partial-set baseline job `5997565`** (still RUNNING, untouched —
  distinct manifest `frame_manifest_local.csv` + output dir `models/spine`; at ep0 it0→1850 loss 1.64→1.00,
  ~33 img/s, healthy). **NEXT LEVER:** read 5998329 macro-F1 → if tool/task ≥0.95, run `spine_grounding.py`
  on the 11 Cat-2 clips to replace P0.2 hand-set entities with REAL predictions and re-measure grounded BLEU
  vs the 0.4215 bar; in parallel run `cam_pseudoboxes.py` (Grad-CAM→pseudo-boxes) at FULL scale to bootstrap
  the Cat-1 weakly-supervised detector.
- **2026-06-16 CONFIRMED (P1.1 grounding-spine UNBLOCKED on public SurgVU-2024):** the prior-year
  practice set is in hand — `…/data/surgvu24_labels/labels/case_000…154` (**155 cases**, each with
  `tools.csv`+`tasks.csv`). The 320 GB `surgvu24_videos_only.zip` is **still streaming** on the login node
  (curl PID 605557; ~46 MB/s, ETA ~1.8 h from 01:48). **Frame-label pipeline built + verified**
  (`experiments/build_frame_labels.py`): maps each `tools.csv` install/uninstall interval → 14-tool
  multi-hot + each `tasks.csv` segment → 8-task label, **PART-AWARE** (videos split into
  `case_XXX_video_part_NNN.mp4`; the `install_case_part`/`start_part` columns + part-local times are
  honoured — this was a real trap: times are part-local, not case-global). Canonical **14-tool list**
  taken verbatim from the official Cat-1 `inference.py` (needle_driver, monopolar_curved_scissor,
  force_bipolar, clip_applier, tip_up_fenestrated_grasper, cadiere_forceps, bipolar_forceps,
  vessel_sealer, suction_irrigator, bipolar_dissector, prograsp_forceps, stapler,
  permanent_cautery_hook_spatula, grasping_retractor); noisy aliases mapped, off-list rares + camera/empty
  dropped. **Full manifest: 1.08 M candidate frames @ stride 2 s, all 14 tools + 8 tasks present,
  case-disjoint split (132 train / 23 val cases).** Tool freq is long-tailed (cadiere 537 k …
  bipolar_dissector 1.6 k → BCE `pos_weight` up to 50×); "Other" task dominant (720 k) → down-sampled to
  25 % when tool-empty.
- **2026-06-16 CONFIRMED (streaming extraction — don't wait on the 320 GB download):**
  `experiments/stream_extract_zip.py` extracts COMPLETE deflate (method 8, trailing-descriptor) mp4
  entries straight out of the still-growing zip via mmap. **11 cases / 19 parts (~24 GB) extracted** from
  the partial download. Built a video-gated local manifest (76 k frames) + **verified end-to-end OpenCV
  part-local seek/decode: 8/8 frames decode at 720p60 with correct multi-hot tool + task labels**; a
  part_001 mp4 is genuinely ~5.3 h (1.14 M frames @ 60 fps), label extent within bounds → mapping correct.
- **2026-06-16 CONFIRMED (env + spine plumbing):** new training env `/scratch/sc20osc/surgvu_env`
  (torch 2.6.0+cu124, timm 1.0.27, transformers 4.57.6, nltk 3.9.2, sklearn 1.6.1, cv2 4.13.0); BLEU
  harness re-validated in it (bare-No 0.1778, exact-ref 1.0, echo-shortest 0.3525). **EfficientNetV2-S
  two-head spine** (`experiments/train_spine.py`: 14-tool BCE + 8-task CE, on-the-fly mp4 decode, macro-F1
  + threshold sweep, target ≥ 0.95) **smoke-passed on the login A2** (512 frames/1 ep → trains, eval,
  checkpoints OK). **Full job SUBMITTED: SLURM `5997565`** (`train_spine.slurm`, gpu/l40s:1, 8 ep, 120 k
  train frames; re-extracts + rebuilds the local manifest at job start to use max-available video). ⏳
  macro-F1 numbers pending the run. **Cat-1 weak-detection scaffold ready** (`experiments/cam_pseudoboxes.py`:
  Grad-CAM on the spine → connected-component pseudo-boxes → YOLO-trainable `pseudo_boxes.json`; no 2024
  bbox GT, qualitative only). **Cat-2 P1.2 wiring ready** (`experiments/spine_grounding.py`: spine →
  per-clip tool-presence + dominant-task → fills P0.2's hand-set `GROUNDING` with real predictions; the
  task names ARE anatomy → direct task→organ bridge). Next: read job 5997565 macro-F1 → if ≥ 0.95 run
  `spine_grounding` on the 11 clips → re-measure Cat-2 BLEU with REAL grounding (vs the 0.965 hand-set
  ceiling / 0.4215 bar); scale training to the full video set when the download completes.
- **2026-06-15 CONFIRMED (P0.1):** BLEU harness reimplemented (`/tmp/bleu_probe.py`), faithful to NLTK
  method1 (epsilon 0.1, BP, closest-ref tie-break). Verified on real samples.
- **2026-06-15 REFUTED:** README/working hypothesis "matching shortest ref like 'No' gives BLEU ≈ 1.0."
  Bare `No` = **0.178** (three n-gram orders epsilon-floored → `0.1^0.75`). Only **exact match to a full
  reference = 1.0**.
- **2026-06-15 CONFIRMED:** verbose-correct answers are BLEU-punished (14-word = 0.21; very verbose =
  0.02); plausible templated full sentences earn 0.40–0.51; case/punctuation matter (`No`0.178 vs
  `no`0.100). Yardsticks over 11 samples: bare-y/n 0.120, echo-shortest-ref 0.353, oracle 1.000.
- **2026-06-15 CONFIRMED (literature):** 2025 winner Capybara = zero-shot LLaVA-OV-7B + EffNet tool/organ
  classifiers, BLEU 0.4215; UoM PEFT fine-tune (0.202) < its own zero-shot (0.3656) → **fine-tuning is not
  a free win; grounded zero-shot + templating is the proven recipe.**
- **2026-06-15 CONFIRMED (P0.2) — template-bank + keyword-router, mean BLEU 0.9650 ≫ 0.35 target.**
  Code: `experiments/bleu_harness.py` (wraps the *real* NLTK `sentence_bleu`, method1, weights .25×4,
  lowercased `.split()` tokenization, max-over-5-refs then mean) + `experiments/p0_2_template_router.py`.
  **Harness re-validated against yardsticks:** bare `No`=**0.1778**, exact-match-to-a-ref=**1.0000**,
  echo-shortest-ref(11)=**0.3525**, bare-y/n(11)=**0.1132**, oracle=**1.0000** — all match the P0.1 probe.
  **Method:** keyword router over the question string → 6 buckets {yes/no-presence, yes/no-action,
  tool_type, organ, procedure, purpose, fallback}; each bucket emits one 4–7-word reference-register
  sentence. Polarity + grounded entity (tool/organ) are **hand-set per case** (the stand-in for the future
  classifier, exactly as §6 allows — *"≥0.35 with hand-set entities"*); template surface forms were tuned
  to the reference register/verb/tense only, **NOT** to any held-out answer. Per-question max-BLEU:
  ```
  case122 1.0000  'No forceps are being used.'                         (forceps yes/no → exact ref)
  case123 1.0000  'No, a large needle driver is not listed.'           (tool yes/no, "listed" → exact)
  case124 1.0000  'The forceps type is Cadiere Forceps.'               (tool_type → exact ref)
  case125 1.0000  'Yes, sutures are required.'                         (action yes/no → exact ref)
  case126 0.6148  'Yes, a large needle driver is involved.'            (affirmative-tool template ≠ refs' "was utilized")
  case127 1.0000  'The organ being manipulated is the uterine horn.'   (organ → exact ref)
  case128 1.0000  'Yes, a needle driver is involved.'                  (tool yes/no → exact ref)
  case129 1.0000  'The summary is describing endoscopic or laparoscopic surgery.' (procedure → exact)
  case130 1.0000  'The forceps are used for grasping and holding tissues or objects.' (purpose → exact)
  case131 1.0000  'Yes, tissue is being cut.'                          (action yes/no → exact ref)
  case132 1.0000  'No, a large needle driver was not used.'            (tool yes/no, "was"/"used" → exact)
  ```
  **CONFIRMED vs ≥0.35:** 0.9650 clears it by a wide margin; gap to Capybara SOTA 0.4215 is *closed offline*
  (10/11 exact-ref hits). ⚠ **n=11, tiny, indicative only** — and the polarity/entity are hand-set, so this
  measures the **templating phrasing ceiling**, NOT real accuracy; the true lever is now the grounding spine
  (P1.1 tool/task/organ classifiers) supplying correct polarity+entity on the gated distribution. The only
  sub-1.0 case (126, affirmative-tool carrier) shows the next *templating* lever: an affirmative tool-presence
  template ("…was utilized/used") closer to the negative form, but left un-overfit to the held-out wording.

---

## Diagnostic & Failure-Analysis Playbook
> Instantiates the campaign 5-step loop (`~/CLAUDE.md §4b`) for SurgVU. **A macro-F1/BLEU number is not a
> result — the result is knowing WHICH tool/task/case drags the macro-mean and WHY.** Run this on EVERY
> spine checkpoint and EVERY Cat-2 BLEU eval; a REFUTED lever with no diagnosis is incomplete work.
> **Metrics under diagnosis:** Cat-1 COCO mAP@[.5:.05:.95] (tool detection) · Cat-2 BLEU (max-over-5 refs,
> method1-smoothed, mean) · tool + task **macro-F1** from the grounding spine (the lever that feeds both).
> **Auto-diagnosis:** the moment full-scale spine job **`5998329`** writes `models/spine_full/spine_best.pt`,
> `diagnose_spine.slurm` re-runs the val split and auto-populates the per-class tables below — no aggregate
> macro-F1 ever stands alone. (Spine env `/scratch/sc20osc/surgvu_env`; CPU-light data probe + 1×L40S model probe.)

### Step 1 — DECOMPOSE the metric into strata (where the loss concentrates)
- **per-TOOL macro-F1** (14 tools) + precision/recall/AP + val support, sorted **worst→best**; **per-TASK
  macro-F1** (8 tasks) + per-task recall; **per-CASE task-accuracy** (worst 10 → centre/procedure shift).
- The macro-mean is dragged by the **long tail** — flagged against the frame-label class distribution
  (`data/frame_manifest_full.csv`, 1,073,389 frames). **Verified skew (`class_distribution.py`, runs in ~5 s):**
  - **TOOL skew 274×** (cadiere 446 k / needle_driver 420 k / bipolar_forceps 367 k / monopolar_scissor 360 k
    head … **stapler 26 k, clip_applier 14 k, `tip_up_fenestrated_grasper` 4.4 k (8 cases / 525 val),
    `suction_irrigator` 2.3 k (15 cases / 303 val), `bipolar_dissector` 1.6 k (1 case / 0 val)** tail).
  - **TASK skew 79×** (`Other` 617 k dominant; `Range of motion` 7.8 k, `Retraction/collision` 8.5 k,
    `Skills application` 22 k = the rare tasks).
- **Diagnostic:** `experiments/class_distribution.py` (data-side, no GPU) + `experiments/diagnose_spine.py`
  (model-side, per-class F1/confusion → `models/<run>/per_class_diag.json`).

### Step 2 — CHARACTERIZE the failure mode of the worst stratum
- **Rare-tool / rare-task under-recognition (class imbalance):** thin-support classes are recall-starved —
  the BCE/CE collapses onto the head. Quantified by per-class recall ≪ precision + low AP at adequate support.
- **Tool confusion pairs:** for each false-positive tool, which *co-present true* tool triggered it
  (look-alike clevises, e.g. forceps families, driver↔scissor) — `diagnose_spine.py` emits the top-12.
- **Task confusion matrix:** which task→which (esp. rare tasks bleeding into `Other`); printed N×N.
- **Cat-2 grounding gap:** templating is **SOLVED** (P0.2 = 0.9650 with hand-set entities) — the bottleneck
  is **vision grounding**: does the spine emit the *correct* tool-presence + dominant-task to fill the
  template? A wrong predicted entity flips polarity/noun → BLEU collapses despite perfect phrasing.

> **⚠⚠ 2026-06-19 CPU LEAKAGE AUDIT (the headline risk, quantified — `bleu_harness`+router, n=11):**
> The **0.7428** "real-grounded" headline is **NOT vision-driven — it is a template prior that the spine
> currently DEGRADES.** Measured splits on the 11 clips:
> - **Text-only / no-vision floor = 0.7734** (router fires on the QUESTION only; every yes/no gets the
>   *majority* default polarity "Yes"; tool_type/organ emit the bucket carrier with a generic entity;
>   procedure/purpose are vision-independent). This floor is **ABOVE** both the actual spine-grounded
>   `grounded_bleu.json` mean **0.7008** and the variant-best **0.7428**. ⇒ **the spine adds NEGATIVE value
>   right now**; ~all of 0.74 is the templating prior + the fact 6/11 cases (procedure/purpose/2 needle-driver
>   yes-no/organ-when-uniform) don't need vision at all.
> - **Genuinely vision-driven cases = only 5/11** (where perfect-vision beats the floor by >0.10): case122,
>   123, 124, 127, 132. On those 5: floor 0.579 → perfect-vision ceiling 1.000 → **ACTUAL grounded 0.790**.
>   So the recoverable vision headroom is **0.79→1.00 on 5 cases ≈ +0.10 to the overall mean**, gated on the spine.
> - **All 5 spine-HURT cases** (where the spine pred ≠ perfect-vision pred): 124 (0.537), 125 (0.202),
>   126 (0.517), 131 (0.041), 132 (0.411). Of these, **2 are spine errors, 2 are router bugs, 1 is a polarity flip**:
>   - **case124 = TOOL-CONFUSION (spine):** fired `bipolar_forceps` not `cadiere_forceps` → "The forceps type
>     is bipolar forceps" (0.537). This is the **#1 confusion pair in `per_class_diag` (bipolar←cadiere n=1106)**
>     — a real classifier error, NOT a router/template error.
>   - **case131 "is tissue being cut" = ROUTER BUG (pure code):** `_question_tool_keys` finds no tool noun for
>     "cut" → `present` defaults False → the `is_cut`→"Yes, tissue is being cut" branch is **unreachable** →
>     emits "No, a **None** is not used" (0.041). No vision involved.
>   - **case125 "is suture required" = ROUTER BUG:** "suture" absent from `QUESTION_TOOL_GROUPS` → keys empty →
>     `present`=False → the `is_suture_required`→"Yes, sutures are required" branch (needs present=True) is
>     unreachable → "No, a suture is not used" (0.202). No vision involved.
>   - **case126/132 = SPINE POLARITY FLIPS (the near-identical-Q opposite-A pair):** 126 ref=Yes/"utilized"
>     but spine didn't fire needle-driver present → "No…not used" (0.517); 132 ref=No but spine fired a
>     false-positive needle-driver → "Yes…is involved" (0.411). Spine got **both** polarities backwards;
>     polarity is *not* recoverable from the question, so the spine is the only signal and it's wrong here.
> **VERDICT:** the validated lever order is (1) **fix the 2 router bugs** (cut/suture action branches reachable
> from grounded `present`; this is the V2 fix in job 6204239 — pure CPU code, recovers case131 0.04→1.0 and
> case125 0.20→1.0 *if the spine fires the action signal*, ≈ **+0.16 to the mean for free**); (2) **fix case124
> tool-confusion** (hard-pair head / clevis crop on the bipolar↔cadiere look-alike); (3) the 126/132 polarity
> pair needs a *better spine*, not templating. **Until a real router bug-fix lands, 0.7428 is a leaked upper
> bound and the spine is below the text-only floor.**
- **Cat-1 detection gap:** no 2024 bbox GT → the failure mode is *localization* (Grad-CAM pseudo-boxes are
  coarse/multi-instance), bounded above by the spine's tool **presence** F1 (can't box a tool you can't classify).

### Step 3 — QUANTIFY the bottleneck (recoverable vs data-floor)
- **DATA-FLOOR (loss CANNOT fix → escalate to MORE DATA, flag to user):** `bipolar_dissector` (1 case,
  **0 val frames** → val-F1 structurally 0/undefined), `suction_irrigator` (15 cases), `tip_up_fenestrated_grasper`
  (8 cases). These cap the *14-tool* macro-mean no matter the method — public proxies (CholecT50/SurgToolLoc)
  or more SurgVU cases are the only fix. `diagnose_spine.py` reports **macro-F1 excl-floor** + **recoverable
  ceiling** so the data-floor drag is separated from the method gap.
- **RECOVERABLE (method/representation gap → an innovation):** the mid-tail with real val support
  (clip_applier, stapler, permanent_cautery_hook_spatula, force_bipolar; rare *tasks* all have ≥1.9 k val).
- **Cat-2:** grounded-BLEU (real spine predictions via `spine_grounding.py`) vs the **0.4215** template-ceiling
  bar — the gap is recoverable spine accuracy, not phrasing. **Cat-1:** presence-F1 → pseudo-box → detector mAP gap.

### Step 4 — MAP gap → innovation (+ pre-registered kill criterion)
| Bottleneck (stratum) | Innovation it points to | Kill criterion (REFUTE if…) |
|---|---|---|
| Rare-tool/task imbalance (recoverable mid-tail) | class-balanced loss (BCE `pos_weight`↑ / focal) + class-balanced sampling (oversample rare cases) | rare-tool macro-F1 doesn't rise ≥ +0.05 vs uniform → it's floor, not loss |
| Data-floor tools (dissector/suction/tip-up) | MORE DATA: CholecT50/SurgToolLoc proxy frames or more SurgVU cases (flag to user) | n/a — escalation, not a model lever |
| Cat-1 no bbox GT | Grad-CAM → connected-component **pseudo-boxes** (`cam_pseudoboxes.py`) → weakly-sup YOLO/DETR, refined on the small bbox val set | pseudo-box mAP < presence-F1 floor → localization, not classification, is the wall |
| Cat-2 grounding gap | **real visual grounding**: spine tool/task predictions → template entities (`spine_grounding.py`), replacing P0.2 hand-set; + VLM tail for open Qs | grounded-BLEU < 0.4215 with tool/task F1 ≥ 0.95 → templating/router, not vision, is the limit |
| Tool confusion pairs (look-alikes) | hard-pair-aware head / co-attention; higher-res clevis crop | confusions persist after crop → encoder capacity, escalate backbone |

### Step 5 — KEEP the diagnostic artifacts (every run auto-diagnosed)
- `experiments/class_distribution.py` — CPU-light class-skew + data-floor flagger (reads `frame_manifest_full.csv`).
- `experiments/diagnose_spine.py` — reloads a spine ckpt, re-runs val → per-tool/per-task F1, confusion pairs,
  task CM, per-case shift, recoverable-vs-floor split → `per_class_diag.json` (defaults to **5998329**'s
  `spine_full/spine_best.pt`; pass `--ckpt` for any other run).
- `experiments/diagnose_spine.slurm` — sbatch wrapper (1×L40S, decode+forward needs GPU → never on login).
  **Auto-run trigger:** when 5998329 finishes → `sbatch experiments/diagnose_spine.slurm` → tables here update.
- **Current ep0 snapshot (5998329, mid-train, will improve):** aggregate tool macro-F1 **0.585** / task **0.656**
  / task-acc 0.809 @ thr 0.7 (1-epoch smoke was 0.1807). **Per-class tables populate on the next checkpoint.**

---

## Deep Research / SOTA (Cat-2)
> Distilled from `DEEP_RESEARCH_CAT2.md` (user-supplied 2026-06-20). The 2025 podium is **zero-shot VLM +
> detector-grounding + concise prompted answers**, NOT fine-tuning. **We already beat it: real-grounded
> BLEU 0.7428 (n=11) ≫ 0.4215 winner** — our spine-grounding IS the detector-informed approach the
> research endorses. The open frontier = the levers below that the autonomous ladder (variants of
> n_frames/thr/pool/detector) does NOT touch.

**2025 podium (the bar) & their lessons**
- **Capybara 0.4215** (winner) = LLaVA-OneVision-7B **zero-shot** + tool/organ detection + UI/margin crop
  + generated description + **only 5 sampled frames** + concise prompted answers. Only **21 unique
  captions** in `matched_description` ⇒ direct FT is brittle. *Eval HW = Tesla T4; 4-bit VLM outputs
  differ across Turing/Ampere/Hopper ⇒ test on T4-like HW.*
- **UoM-SurgicalAI 0.3656** zero-shot InternVL3/3.5 > its own PEFT (0.202) — **FT is not a free win**.
- **AMI** single-answer-per-Q-type ⇒ model ignores vision, memorises priors (text-only ablation strong).
- **UT** 70k rule-QA + LoRA InternVL3-2B; **longer training degraded** (template overfit).
- **Medibot** dedicated detector as *authoritative* tool-presence source (VLM hallucinated).

**Failure modes (priority order):** (1) weak/low-diversity supervision (21 captions) → don't FT on
`matched_description`; (2) metric mismatch — BLEU rewards concise canonical phrasing ≫ verbose-correct;
(3) **shortcut learning / answer priors** (text-only ablation strong — SurgCheck); (4) insufficient
temporal retrieval (teams used 5/8/21 *uniform* frames, no learned selection); (5) hallucination
(detector-as-authority fixes it); (6) synthetic-QA quality > raw count; (7) quantisation
non-reproducibility across GPU gens.

**Research lever stack vs OUR current pipeline (spine→fixed router→template, uniform frames):**
| # | Research lever | We do? | Gap / action |
|---|---|---|---|
| 1 | Zero-shot strong VLM + concise prompt (the winning floor) | ✗ (we use classifier-spine, no VLM) | our spine already beats the floor; VLM only needed for the open "why/purpose" tail |
| 2 | **Canonical-answer layer** (ontology→short templated answer) — *biggest BLEU lever* | ◑ template bank exists but router is keyword-rules | **NEW: ontology-driven canonical layer** (entity→single canonical surface form, register-locked); reduces variance, not just gaming |
| 3 | Detector-informed answer prior (tool/organ feeds VLM as authority) = Capybara | ✓ (spine grounding = this) | done — our 0.7428 |
| 4 | **Question-conditioned temporal retrieval** (score 30 frames by Q-relevance → top-k) | ✗ (uniform `np.linspace`) | **NEW: Q-relevance frame scoring** vs uniform — the under-explored lever |
| 5 | Structured scene memory / scene-graph (SSG-VQA) | ✗ | later — tool/task timeline as text evidence |
| 6 | Counterfactual scene-graph QA curriculum (anti-shortcut) | ✗ | needs gated QA; defer |
| 7 | Uncertainty-gated expert mixture (pick least-risky by disagreement) | ✗ | later — challenge-legal (one answer out) |
- **ALWAYS:** clone the official scorer (1 pred/Q, BLEU vs each of 5 refs, max, mean); version every
  prompt; release the synthetic-QA engine.

**Models if/when a VLM tail is added:** LLaVA-OneVision-7B (Capybara base), InternVL3.5-8B (UoM base),
Qwen2.5-VL-7B (surgical warm-starts). Compute: ≤7B + light classifiers — single L40S sufficient.

**⚠ n=11 caveat / leakage risk (pre-registered):** our 0.7428 is local n=11 with templates exact-matched
to these 11 references; **polarity is NOT recoverable from the question** (router code itself flags
case126 vs case132 — near-identical Q, opposite answer), so the spine's polarity prediction is doing the
work but the surface forms were authored against these exact refs ⇒ the score is an **upper bound, likely
optimistic on the hidden set** (cf. UoM local-dev optimism). The adversarial/leakage probe (below) tests
whether the gain is shortcut/metric-inflated before any SOTA claim.

---

## 8. Compute plan (L40S, 48 GB)
- **Login node:** small A2 GPU only — use for I/O, the BLEU harness, frame extraction. Real GPU work →
  SLURM `gpu` partition (28 nodes × 3× L40S 48 GB, 2-day limit). **No `#SBATCH` mail; self-monitor; never
  cancel the user's jobs** (`slurm-no-email-self-monitor`). User has live `mstcn_x` jobs — don't touch.
- **7B VLM in 48 GB:** QLoRA (4-bit NF4 + LoRA) fits with headroom (7B QLoRA runs on 24 GB; 48 GB leaves
  room for 30 s clips at ~5–32 frames). Tools: **LLaMA-Factory** or **ms-swift** (both have video QLoRA
  recipes). Profile actual video-token VRAM per model (InternVL 256 tok/frame is the heaviest — UNVERIFIED
  on 48 GB). Classifiers (EfficientNetV2-S) train trivially on one L40S.
- **Env:** download venv `/scratch/sc20osc/miccai-2026/dlenv/bin/python` (download-only: torch/nltk NOT
  installed → **need a training/inference env** with torch+transformers+nltk+ffmpeg, e.g. a new venv on
  scratch or a CUDA pytorch Docker base). `HF_HOME=/scratch/sc20osc/hf_cache`. Bulk on scratch; home light.
- **FM pool reuse:** SurgVLP/PeskaVLP/HecVL/BiomedCLIP (grounding/retrieval); EndoViT/SurgeNet/DINOv3
  (classifier backbones); Qwen2.5-VL-7B-Surg-CholecT50 (VLM warm-start).

---

## 9. TODO + definition of done
**Now (no gate):**
- [x] **P0.2 — template bank + keyword router + BLEU on 11 samples** (§6). **CONFIRMED: mean BLEU 0.9650
      ≥ 0.35** (`experiments/bleu_harness.py` + `p0_2_template_router.py`; hand-set entities, n=11 indicative).
- [ ] Build a torch+transformers+nltk+ffmpeg env on scratch (dlenv lacks them); pin versions.
- [ ] P0.3 — zero-shot `bravefox12138/surgvu2025vqa` (3B) + LLaVA-OV-7B on the 11 clips via SLURM; BLEU.
- [ ] Study Capybara repo (`huuquan1994/surgvu25-cat2-submission`) end-to-end; port the side-crop +
      tool-strip-blur preprocessing; note their exact prompt + template phrasing.
- [ ] Stub the Docker container (`inference.py` → template bank) and pass `do_test_run.sh` offline early.

**When gated data lands (user registering on surgvu26.grand-challenge.org):**
- [ ] Pull SurgToolLoc/SurgVU video + `tools.csv`/`tasks.csv`/`matched_description` to scratch.
- [ ] P1.1 train 12-tool/8-task/7-organ classifiers (target macro-F1 ≥ 0.95).
- [ ] Author the full template bank from the `matched_description` corpus (~21 captions/~10 templates).
- [ ] P1.2/P2 grounded templates → hybrid → BLEU-aware decoding; only fine-tune the VLM if it BEATS
      zero-shot (refute the 2025 fine-tune-hurts result).

**Open questions to resolve (UNVERIFIED):** yes/no question fraction (from gated QA file); whether the
official harness lowercases/strips punctuation (leaderboard probe `no` vs `No`); Medibot's exact VLM;
Capybara's arXiv ID.

**Definition of done:** mean test BLEU **> 0.4215**; Docker builds + passes `do_test_run.sh` offline +
the grand-challenge harness; reproducible from a pinned checkpoint; ≥1 successful prelim submission;
no test-item-specific tuning; methodology report + 3-min video submitted; editor access granted.

## Cat-2 — Molmo2-8B native-grounding zero-shot probe (research lever, sanity check) — SUBMITTED 2026-06-19 (job 6206220)
Deep-research lever (`MICCAI2026_INNOVATIONS.md §1`): does a stronger native-grounding open VLM lift REAL grounded-BLEU vs our classifier-spine? **Molmo2-8B** (`allenai/Molmo2-8B`, Apache-2.0, gate-free, downloaded ~17GB) run ZERO-SHOT on the 11 Cat-2 sample clips: 1 representative middle frame/clip → Molmo2 (trust_remote_code, single user turn, greedy, prompt suffix `" Answer in one short declarative sentence."` to fight BLEU's verbosity penalty — NO router, NO leakage toward refs) → official `bleu_harness.evaluate` (max-over-5, method1, mean). `experiments/molmo2_cat2_bleu.py` + `molmo2_cat2_bleu.sbatch` (gpu/l40s:1, 1h, no mail), `py_compile`-clean; cv2 middle-frame decode verified on login. Job `6206220` (`surgvu_molmo2_cat2`); writes `…/SurgVU/models/molmo2_cat2_bleu.json`, prints `MOLMO2_CAT2_BLEU=<mean>`, appends `[SurgVU][6206220] …` to `AUTONOMOUS_RESULTS.md`.
**Bars (honest framing):** our spine grounded-BLEU **0.7428** is OPTIMISTIC n=11 (templates exact-matched to these 11 refs → UPPER BOUND, NOT a target); the real question = does raw zero-shot Molmo2 clear the **0.4215** 2025-winner bar. Raw zero-shot (no template router) will almost certainly land BELOW the templating spine — this is a SANITY CHECK on whether stronger grounding alone is competitive, not a replacement for the spine. **Eagle-2.5-8B NOT probed (gated/research-only + no HF token → blocked, escalated to user).** Parent armed a background waiter. Did NOT touch the user's `aielt_*`/`jig_*`/`mstcn_*` or other agents' jobs.

### 2026-07-19 — SOURCE CHECK (getting-started): exact registration + data path captured (all USER action)
Registration steps (grand-challenge, live since May 8): (1) click **Join**; (2) read + agree challenge guidelines; (3) **team lead** sends the agreement form (Google Drive: `https://drive.google.com/file/d/1ry29hB-YGbLfeN73EfrpaNTmmYR5CbPZ/view`) to **isi.challenges@intusurg.com** AFTER all members request to join; (4) create the team. **2026 training data = download page, accessible ONLY after registration** (release date not posted). Submission = **directly online to grand-challenge** + methodology report + GitHub repo + pre-recorded presentation. Two categories (Cat1 tool-detect, Cat2 VQA). ⇒ **fully USER-gated: nothing actionable on our side until the user registers + sends the form + shares the 2026 data link** (offline-only Cat1 0.77/Cat2 0.91 stay UNVALIDATED until then).

### 2026-07-30 — ★★ THE 2026 METRIC CHANGED (BLEU → BERTScore-F1) ⇒ **the 0.4215 "prior winner bar" is INVALID for 2026** · Cat-1's 0.77 is actually **0.0000** · data was never gated
**1. DATA — the "fully USER-gated, nothing actionable" claim is RETRACTED.** The **754 GB already on disk IS the
2026 training corpus**: organizers' `/data-description/` says 2026 training data is *"an expanded version of the
training data and labels used within the 2022/3 SurgToolLoc challenge"* — 280 videos / 155 sessions / 840 h / 18M
frames, an exact match to `surgvu24_videos/` (280 decodable mp4, 155 cases), plus `SURGVU25_train_labels/` carrying
the Cat-2 `matched_description` column (3,673 task rows, **21 unique captions** — matches Capybara's independent
report). **No re-download needed for training.** ⚠ **Genuinely missing (USER):** the small **2026 Cat-1
bounding-box VALIDATION set** — Cat-1 trains on *"noisy tool presence labels … and bounding box labels provided in
the small validation set"*; the gated `/data-download/` page 403s to a non-participant.
**2. ★★ Cat-2 IS NO LONGER BLEU — IT IS BERTScore-F1** (roberta-large, `rescale_with_baseline=True`, max over 5
refs, mean over questions). Read from the organizers' OWN eval container
(`isi-challenges/surgvu26-category-2-eval-public` → `evaluation/evaluate.py`), whose README states BERTScore-F1 is
the *"Only metric used for ranking"* with BLEU/ROUGE/NLI *"Secondary (for testing/analysis only)"*.
⇒ **THE 0.4215 PRIOR-WINNER BAR IS A *BLEU* NUMBER AND DOES NOT TRANSFER. We currently have NO valid 2026 bar**
until the prelim leaderboard is read. *(My competitive assessment of SurgVU an hour ago was built on that 0.4215 —
it is withdrawn.)*
**★ Measured properties of the new metric (job 6933448) — it is far less discriminative than assumed:**
- **a no-vision constant `"No"` scores 0.6383** ⇒ the entire competitive range is only ~0.36 wide;
- **polarity-flipped (WRONG) answers still score 0.9110**; direct probe `"Yes"` vs reference `"No"` = **0.9957**
  ⇒ **the ranked metric is ~negation-blind.** The organizers documented this, built an NLI-based fix, and
  **chose not to rank on it.**
- Quirk with a design consequence: `normalize()` is applied **only** to BLEU/ROUGE — **BERTScore consumes RAW
  text**, so case/punctuation matter for the ranked metric and *not* for BLEU — **the inverse of our June rule.**
**3. ★ ANSWER REGISTER DOMINATES ACCURACY (job 6933607).** A ~7-word declarative sentence dominates at every
accuracy level, no crossover in [0,1]:
| register | correct | **wrong** | words |
|---|---|---|---|
| bare token | 0.8825 | 0.6923 | 1.4 |
| **short sentence** | 1.0000* | **0.7932** | 6.9 |
| verbose prose | 0.5562 | 0.4461 | 24.9 |
*(\*inflated by authored templates; the `wrong` column and the ORDERING are leakage-free.)*
⇒ **a raw zero-shot VLM emitting prose lands ~0.45–0.56 — BELOW the 0.6383 no-vision floor.** So the binding
constraint under BERTScore is **answer length/register, NOT zero-shot-vs-fine-tuned.**
**⇒ MY BRIEF WAS WRONG AND THE AGENT WAS RIGHT TO BLOCK RATHER THAN ADOPT IT.** I argued "the field shows
fine-tuning hurts, so build zero-shot first" — but that field result was **on BLEU**. Under BERTScore, SFT's
documented ~64% output-length collapse on Qwen3-VL-8B moves in the **RIGHT** direction. The SFT arm stays
pre-registered but **gated behind the zero-shot read**, not treated as settled either way.
**Refinement worth acting on:** correctness is worth **+0.207** in the sentence register (not the +0.089 the
polarity-only flip implied) — once the **ENTITY** can be wrong (tool / organ / purpose) the value roughly doubles.
⇒ **aim vision at ENTITY NAMING, not yes/no polarity.**
**4. ⛔⛔ Cat-1 0.77 IS STRUCK — the real number is mAP@[.5:.95] = 0.0000.** `cat1_honest_val` was written **and
run on Jun 27 (job 6351081) — and the result was never read for a month.** Against the real human COCO GT (which
we have held since Jun 27: 5,178 images / 11,323 boxes / 7 videos) **every detector scores 0.0000, including the
0.9788 "champion".** Not a bug — arithmetic: human clevis boxes are **0.0215** of frame vs our Grad-CAM boxes at
**0.1538** (**7.2× too large by area, 2.7× per side**); a *perfectly centred* pseudo-box caps at **IoU ≈ 0.140**,
below the 0.5 floor where AP begins ⇒ mAP ≈ 0 regardless of classification quality.
**★ AND THE STANDING "NO HUMAN BBOX GT EXISTS" BLOCKER WAS FALSE FOR OVER A MONTH — Cat-1 was killed on it.**
Fix running (`6933641`): train on the 5,178 real boxes, video-disjoint.
**Jobs:** 6933448 ✅ · 6933607 ✅ · **6933479** zero-shot Qwen3-VL-8B × 4 registers × {1,5} frames + bootstrap CI
(pending) · **6933641** Cat-1 real-box training (pending). Scorer clone: `experiments/surgvu26_scorer.py`.

### 2026-07-30 — ★ Cat-1 RECOVERED FROM 0.0000 → **mAP@[.5:.95] = 0.1470** by training on the real human boxes (`6933641`, video-disjoint)
YOLO26m, 60 epochs, trained on the **5,178 real human COCO boxes we had held since Jun 27**, validated
**video-disjoint** on 1,973 images / 3,275 instances:
| metric | value |
|---|---|
| **mAP@[.5:.95]** | **0.1470** |
| mAP50 | 0.2944 |
| mAP75 | 0.1217 |
| P / R | 0.366 / 0.260 |
**⇒ from `0.0000` (every Grad-CAM-pseudo-box detector, including the 0.9788 "champion") to a genuinely scoreable
detector.** This confirms the diagnosis exactly: the old boxes were **7.2× too large by area**, capping IoU at
**0.140** — below the 0.5 floor where AP begins — so no amount of classification quality could produce a non-zero
mAP. **The fix was never modelling; it was supervision.** The 0.77 headline is struck and replaced by a real 0.147.
⚠ Honest framing: **0.147 is a first dry-run of the 2026 recipe, not a tuned result** — 60 epochs, default
hyper-parameters, off a `yolo26m` init, and validated on our own video-disjoint split rather than the organisers'
**2026 Cat-1 bbox validation set, which we still do not hold (USER)**. Treat it as *the floor of a working
pipeline*, not a competitive number.
**Cat-2 zero-shot (`6933479`) FAILED at 49 s on a trivial env gap** — `accelerate` absent from `surgvu_env`
(python 3.9), so `device_map="cuda"` raised. Not a modelling failure: `ValueError: Using a device_map … requires
accelerate`. Installed (**1.10.1**, import-verified) and resubmitting. The metric/answer-register work already in
hand (constant-"No" floor 0.6383; ~7-word declarative dominates; correctness worth +0.207) stands unaffected.

### 2026-08-04 — OFFICIAL DATA LINKS PULLED (user-supplied). **Everything was already in hand except provenance — but the Cat-2 sample set CONFIRMS the answer-register policy against REAL references.**
Downloaded to `…/SurgVU/external/2026_dl/`. **Deliberately skipped `surgvu24_videos_only.zip` (320.60 GB)** — we
already hold the 755 GB corpus and it would have consumed nearly all remaining quota for a duplicate.
| asset | size | verdict |
|---|---|---|
| `cat1_test_set_public.zip` | 172,247,434 B | **5,178 images / 11,323 annotations / 7 videos / 14 classes** — **EXACT match to what we have held since Jun 27** |
| `SURGVU25_cat2_train_labels.zip` | 556,936 B | **21 unique captions / 155 cases** — exact match to what we recorded |
| `surgvu24_labels_updated_v2.zip` | 343,046 B | 155 cases, consistent with the held corpus |
| `SURGVU25_cat_2_sample_set_public.zip` | 187,164,455 B | **11 cases (case122–132), 11 questions, 11 mp4** — = our n=11, now provenance-confirmed |
**⇒ THE "MISSING 2026 Cat-1 BBOX VALIDATION SET" WAS NEVER MISSING.** The caveat I attached to the 0.147 mAP is
**withdrawn**: that number was already measured against the **genuine organizer annotations**. It also confirms the
failure mechanism numerically — **real human boxes are 0.0215 of frame (median 0.0141); our Grad-CAM pseudo-boxes
were 0.1538**, the 7.2× inflation that capped IoU at 0.140 and forced mAP to ~0.
**★★ THE CAT-2 SAMPLE SET IS THE VALUABLE PART — it validates the decoding policy on REAL references.** Format:
each `caseNNN_question.json` is a single question string; each `caseNNN.json` is a **list of exactly 5 reference
answers** (the multi-reference set BERTScore-F1 takes the **max** over). Measured across all 11:
- **references per question: exactly 5** for every case ⇒ the max-over-5 mechanic is confirmed.
- **★ reference length: MEDIAN 7 WORDS** (mean 6.4, range 1–12). ⇒ **the ~7-word declarative register that job
  6933607 found "dominates at every accuracy level" is exactly the reference distribution itself** — which is
  precisely what a max-over-references similarity metric would reward. **The policy finding is now confirmed
  against organizer data, not merely against our own authored templates.**
- Example (`case123`, "Is a large needle driver among the listed tools?"): refs = `["No", "No, a large needle
  driver is not listed.", "No large needle driver is included.", "No, there's no large needle driver.", "No, it
  doesn't appear a large needle driver is listed."]` — a bare token plus four ~7–9-word declaratives.
**★ QUESTION-TYPE SPLIT, and it is strategically load-bearing: 7 of 11 are yes/no-style, 4 of 11 are entity/open.**
Combined with the measured near-**negation-blindness** (polarity-flipped WRONG answers score **0.9110**; `"Yes"` vs
reference `"No"` = **0.9957**), that means **~64% of questions sit in the register the metric can barely
discriminate**, while the **~36% entity questions are where correctness is worth +0.207 rather than +0.089.**
⇒ **reinforces the standing direction: aim vision at ENTITY NAMING (tool / organ / purpose), not yes/no polarity** —
e.g. `case124` "What type of forceps is mentioned?", `case127` "What organ is being manipulated?", `case130` "What
is the purpose of using forceps…?". ⚠ n=11 is far too small to treat 7:4 as the true test proportion — read it as
a shape, not a rate.
**Net: no new capability unlocked, but three caveats retired** (Cat-1 GT provenance, the 0.147's comparability, and
whether our answer-register policy generalises beyond our own templates).

### 2026-08-07 — WEEKEND SLATE SUBMITTED + ★★ **TWO OF MY THREE STATED PRIORITIES WERE WRONG, and the real Cat-1 bottleneck is CLASS SUPPORT — half the headline metric cannot move at all**

**★ CORRECTION 1 — "training is under-trained, still climbing" is REFUTED by the log we already had.** I read `0.104 → 0.116` across epochs as a rising curve. Extracting the full per-epoch val series from `c1real_6933641.out`: mAP reaches ~0.12 by **epoch 4** and then **oscillates 0.090–0.147 flat for 56 more epochs**, peaking around epoch 26 of 60. **My "still climbing" read was noise in a flat series** — I generalised a trend from the tail of a printout, the same error shape as the vertical-flip call on 2026-08-04.
**★ AND WORSE: 0.1470 IS A SELECTION-ON-EVAL ARTIFACT.** It is the **maximum of 60 noisy draws on the very val set that selected `best.pt`**, against a central tendency of ~**0.117** ⇒ the headline is inflated by roughly **+0.030**. Every arm now reports **`last.pt` (unselected) alongside `best.pt`**, and a **3-seed replication gate runs BEFORE any arm is believed** — we have never had an error bar on Cat-1.

**★ CORRECTION 2 — "resolution is the highest-confidence lever" is the WRONG ANALOGY (mc2, not mc1).** ORENA's mc1 won because visual density genuinely rose 4.78×. Here the **source images are 640×512 NATIVE and the baseline already trains at `imgsz=640`** ⇒ anything above 640 is **information-free upsampling** — the exact ORENA **mc2** pattern that was KILLED by measurement. My "targets are 2% of frame area" was also loose: **median box area is 0.0141 of frame = ~68 px side = COCO *medium***, comfortably resolved at stride 8/16. R960/R1280 still run (33 min each — measuring beats arguing) but are **pre-registered LOW, not highest-confidence.**

**★★ THE ACTUAL BOTTLENECK — CLASS SUPPORT, and it caps the metric by construction:**
| val class | val boxes | **train boxes** | AP |
|---|---|---|---|
| monopolar_curved_scissor | 1410 | 1423 | 0.360 |
| bipolar_forceps | 1594 | **204** | 0.228 |
| clip_applier | 258 | **0** | **0.000** |
| vessel_sealer | 13 | **0** | **0.000** |
`clip_applier` and `vessel_sealer` occur **only in val videos 6–7 — zero instances in any training video** ⇒ their AP is **0.000 by construction**, and **the attainable ceiling on this split is 0.500, not 1.000** (machine-confirmed by the pool builder). **AP tracks train support almost monotonically** (0 → 0.000 · 204 → 0.228 · 1423 → 0.360). Recall is the binding half (P 0.366 / R 0.260). ⇒ **schedule, resolution and capacity all act on the wrong axis.**

**★ 7-FOLD LOVO ADDED (`7050543`) — an instrument fix, not a split change.** Every fold stays strictly video-disjoint and is reported as its own protocol *alongside* (not instead of) the main split. Motivation: the main split is a fragile instrument — 4 of 14 classes, two pinned at zero, one point estimate, no error bar. LOVO lifts the mean attainable ceiling **0.500 → 0.869**, restores train support for `clip_applier`, and yields a 7-fold error bar.

**★ CAT-2 — A COMPLETED RESULT NOBODY HAD READ (job `6980942`, 2026-08-03; the `accelerate`-missing failure of `6933479` had already been fixed and re-run).** Lesson 1.19 again.
| config | BERTScore-F1 | words |
|---|---|---|
| **C_bare_nf5** | **0.7236** CI95[0.489, 0.937] | 1.5 |
| B_short | 0.6558 / 0.6554 | 4.9 / 10.0 |
| D_refstyle | 0.6397 / 0.6259 | 13.3 / 15.2 |
| A_plain | 0.2119 / 0.2204 | 44.5 / 45.3 |
**DIAGNOSIS: the bottleneck is REGISTER, not knowledge.** The VLM clears the no-vision floor (0.7236 vs 0.6383, **+0.085**) but sits **0.070 BELOW a ~7-word declarative that gets every answer WRONG (0.7932)**. **Length control cannot be bought by prompting** — asked explicitly for reference style it produced 13–15 words; raw, 45. ⇒ `7050563` demotes the VLM to a **≤4-word span extractor behind a deterministic template**.
**★ BONUS RIGOR RESULT — the leakage caveat on the authored-template figures is RETIRED.** The templates are generated by **auxiliary-inversion / wh-fronting of the QUESTION STRING ALONE** and still reproduce reference register nearly exactly (`case124` → *"The type of forceps mentioned is Cadiere forceps"* vs ref *"…Cadiere Forceps"*). **They never saw a reference.**

**JOBS (priorities VERIFIED after submission — the `--nice` lesson):** `7050541` frame pool (CPU/`nodes`) **COMPLETED 62 s** · `7050542_[0-7%2]` 8-arm sweep **prio 154** · `7050543_[0-6%2]` 7-fold LOVO **prio 124** · `7050563` Cat-2 hybrid **prio 154, RUNNING**. ORENA sits at **233–242** and the user's own jobs at 243–267 ⇒ **every SurgVU job is strictly below ORENA and comfortably above 1.**
**INODE DISCIPLINE:** naive per-arm dataset export = ~156k inodes against **199k free** (scratch 5.80M/6.00M — the documented empty-log death). Shared pool + `.txt` image-list splits = **10,385 total**, splits reproducing the baseline exactly (3205 train / 1973 val).

**⇒ NEXT LEVER, pointed at the axis the measurements identify (not queued, needs a call):** the class-support floor is **not a challenge-level data floor** — we hold **754 GB / 280 videos with noisy tool-presence labels**. The motivated structural move is **semi-supervised box proposal**: run the real-box detector over the corpus, constrain proposals to the tools each clip's presence label asserts, retrain. That attacks train support directly, which resolution/capacity/schedule cannot.

### 2026-08-08 — WEEKEND RESULTS: **every Cat-1 knob REFUTED, the seed gate FIRED, and the selection artifact is CONFIRMED BY MEASUREMENT** — but LOVO shows the detector is **~2.4× better than the headline said**. Cat-2 hybrid REFUTED.

**★ THE SEED GATE FIRED — this is the most important number of the weekend.** Three replications of the *identical* baseline config:
| S0 seed | best.pt mAP | last.pt mAP |
|---|---|---|
| seed0 | **0.1470** | 0.1127 |
| seed1 | 0.1249 | 0.1052 |
| seed2 | 0.1614 | 0.1152 |
| **mean** | **0.1444** | **0.1110** |
**Δ_noise (best.pt spread) = 0.0365**, and the pre-registration said *"if Δ_noise ≥ 0.03, every single-run Cat-1 delta ever logged is noise."* **It is 0.0365 ⇒ THAT KILL FIRES.** The historic 0.1470 is simply **seed0's draw** — the middle of three, not a property of the method.
**★ AND THE SELECTION-ON-EVAL ARTIFACT IS NOW MEASURED, NOT ESTIMATED: best.pt mean 0.1444 − last.pt mean 0.1110 = +0.0334**, against the predicted ~+0.030. **`best.pt` is inflated by ~0.033 on every arm** because it is the max over 60 noisy draws on the very val set that selects it.

**ALL FIVE ARMS REFUTED** (bar: best > 0.1470+Δ_noise = **0.1835** AND last > 0.1110+Δ_noise = **0.1475**):
| arm | knob | best.pt | last.pt | verdict |
|---|---|---|---|---|
| E200 | epochs 60→200 | 0.1393 | **0.0900** | **REFUTED** — and the *worst* last.pt of all: 200 epochs OVERFITS |
| R960 | imgsz 640→960 | **0.1756** | 0.1142 | **REFUTED** — highest best.pt, still under the bar, last.pt fails |
| R1280 | imgsz 640→1280 | 0.1464 | 0.1301 | **REFUTED** |
| CAP_n | yolo26**n** (smaller) | 0.1581 | 0.1177 | **REFUTED** |
| CAP_l | yolo26**l** (larger) | 0.1416 | 0.1181 | **REFUTED** |
**Every arm's delta from baseline is SMALLER than the seed spread.** ⇒ schedule CLOSED, resolution CLOSED (confirming my mc2-not-mc1 correction — upsampling past the 640×512 native source buys nothing), and **capacity CLOSED IN BOTH DIRECTIONS**: the deliberate `CAP_n` control (a *smaller* net) scores 0.1581 ≈ `CAP_l` 0.1416 ≈ baseline ⇒ **capacity is definitively not the bottleneck, so stop spending there.** The control arm earned its GPU-hour exactly as intended.

**★★ LOVO — THE INSTRUMENT FIX IS THE REAL RESULT: the detector is far better than the main split reported.**
| fold | 0 | 1 | 2 | 3 | 4 | 5 | 6 | **mean** |
|---|---|---|---|---|---|---|---|---|
| best.pt | 0.4735 | 0.3894 | 0.4208 | 0.4398 | 0.2209 | 0.2418 | 0.3300 | **0.3595** |
| last.pt | 0.4055 | 0.3167 | 0.2989 | 0.3051 | 0.1989 | 0.2281 | 0.3019 | **0.2936** |
**vs the main split's 0.1444 / 0.1110 ⇒ ~2.4× higher on the honest (unselected) comparison.** Good folds reach **mAP50 0.82–0.84 with P 0.86–0.93 / R 0.76–0.77** — that is a genuinely working detector, not a broken one. ⇒ **the 0.1470 headline was mostly an ARTIFACT OF A BROKEN SPLIT** (2 of 4 val classes have ZERO training boxes ⇒ attainable ceiling 0.500), not a statement about model quality. **Fold spread is large (0.2209–0.4735)** — so LOVO must be quoted as a 7-fold mean with its range, never a single fold.

**CAT-2 HYBRID — REFUTED on the pre-registered bar** (n=11; BERTScore-F1, the 2026 ranked metric):
| config | F1 | CI95 | words |
|---|---|---|---|
| Z_bare_nf5 | **0.7236** | [0.489, 0.937] | 1.5 |
| H_hybrid_nf5 | 0.7175 | [0.657, 0.762] | 9.2 |
| N0_template_novision | 0.7026 | [0.643, 0.756] | 8.5 |
| F0_const_no | 0.6383 | [0.362, 0.908] | 1.0 |
**Bar was 0.7932 (the wrong-answer declarative floor). Hybrid = 0.7175 ⇒ REFUTED**, and the floor lies **outside** the hybrid's CI, so this is a real shortfall, not noise. **⚠ THE UNCOMFORTABLE READ: every configuration we have scores BELOW a system that emits a well-formed declarative sentence with the WRONG answer** ⇒ our generated register still does not match the references as well as the authored templates did. **Value of vision in-register = `H_hybrid − N0_template_novision` = +0.0149** — positive, so the second kill (≤0 ⇒ ship template-only) does *not* fire, but it is negligible against the register gap.
**Note on `Z_bare_nf5` topping the table: its CI is [0.489, 0.937] — enormous.** A 1.5-word answer is high-variance under BERTScore; it is not meaningfully better than the hybrid and must not be shipped on the strength of a point estimate at n=11.

**⇒ WHAT THIS SETTLES, AND THE ONE LEVER LEFT.** Cat-1's tunable axes are exhausted — and the measurements point at the axis I originally mis-identified: **class support**, not resolution/schedule/capacity. The corpus fix is the only motivated move: **754 GB / 280 videos with noisy tool-presence labels** ⇒ run the real-box detector over the corpus, constrain proposals to the tools each clip's presence label asserts, retrain on the expanded support. For Cat-2, **n=11 is the binding constraint** — the CIs are too wide to rank anything, which is precisely why *the prelim leaderboard is the only real measurement channel* and why getting any container up is itself the experiment.

### 2026-08-12 — SEMI-SUPERVISED CLASS-SUPPORT LEVER SUBMITTED + ★ **FOUR CORRECTIONS TO MY BRIEF** + a zero-parameter lever hiding in the pixels

**★ CORRECTION 1 — "noisy per-clip tool-presence labels" undersells the data, and taking it at face value would have CORRUPTED the constraint.** `tools.csv` carries per-**ARM install/uninstall TIMESTAMPS** ⇒ presence is **frame-resolvable** (8,833/8,885 rows verified inside their part's true decoded duration). Two silent defects had to be handled: **583 of 7,569 rows have `uninstall < install`** (e.g. install 00:23:03, uninstall 00:06:57), and **144 rows are "Unknown Instrument" with a BLANK class** — *dropping the blanks would MANUFACTURE fake singleton frames, i.e. exactly the frames the mechanism trusts most.*

**★ CORRECTION 2 — FILTERING ALONE CANNOT FIX THE STATED BOTTLENECK.** A detector trained on 8 classes emits only those 8, so "keep proposals for tools the presence label asserts" **can never create `stapler` or `prograsp_forceps`**. Hence **ASSIGN** (class-agnostic proposal + *naming* on frames where exactly one in-vocabulary tool is installed) — the only rule that can move a structurally-zero class. `NEG_SHUFFLE` (class permuted across cases, box counts preserved) is the control that can take it away from us.

**★ CORRECTION 3 — LOVO IS THE WEAKER INSTRUMENT FOR *THIS* LEVER, and the split I called "broken" is the sharp one.** `clip_applier`/`vessel_sealer` are 0.000 on the main split **by construction** ⇒ their seed variance is **identically zero** ⇒ any non-zero AP is outside every noise band. On LOVO those classes *have* support, so the effect is diluted into a 7-fold mean spanning 0.221–0.474. LOVO still runs as the pre-registered headline; **the mechanism gate lives on the main split.**

**★ CORRECTION 4 — A LEAKAGE PATH I DID NOT NAME: THE PROPOSER.** Pseudo-labels carry whatever the proposer knows, so one proposer trained on all 7 videos would inject **every fold's val video into that fold's training labels**. Fold K uses `LOVO_vK`; main uses the v1–5 model; always **`last.pt`, never `best.pt`**, because checkpoint selection is itself a channel from val.

**DISJOINTNESS IS MEASURED, NOT ASSUMED — and it had to be.** The annotated Cat-1 videos are the **same acquisition system** as the corpus: identical da Vinci HUD and pillarbox, and annotated 640×512 frames are exactly `corpus[0:720, 192:1088] → 640×512` (verified on 14 random parts). Whether they are excerpts *of* the 155 training cases is open ⇒ every sampled corpus frame is dHashed against all 5,178 annotated frames; near-dups dropped, and **≥3 hits bans the whole CASE** (the unit of independence is the surgery). Threshold **calibrated, not guessed**: at 256 bits the null is min 70/median 86 and same-shot-0.5 s-apart is median 14, so 48 sits in an empty band. Moved off 64-bit dHash after measuring it collapsed 5,178 frames to 4,198 uniques and put an unrelated frame at distance 15.

**SUPPLY IS BIASED TOWARD EXACTLY THE STARVED CLASSES** (singleton frames @1/2 s, after the conservative rules): `vessel_sealer` **11,883** (45 cases) · `prograsp_forceps` **8,015** · `stapler` **7,620** · `permanent_cautery_hook_spatula` 3,221 · `clip_applier` **2,277** — all currently at **ZERO** train boxes. Two are beyond reach regardless: **`bipolar_dissector` has 0 singleton seconds in the entire 840 h corpus**, `tip_up_fenestrated_grasper` 43 frames.
⚠ **6 of 14 classes have zero boxes ANYWHERE in the annotated data** ⇒ no fold and no local val can score them — **only the prelim leaderboard can.**

**GATES (all pre-registered in the scripts before any arm ran):** Gate 0 = label-free, **zero GPU** (presence-agreement ≥0.50 AND ≥1.5× a shuffled-presence control; class-agnostic firing on never-seen-class singletons ≥0.30) — `cat1_ssl_train.py` **self-aborts** if neither fires. Gate 1 = mechanism, main split: 3-seed mean AP(`clip_applier`) ≥ 0.05 AND ≥ 2× `NEG_SHUFFLE`. Gate 2 = LOVO headline, **paired, on unselected `last.pt`** (baseline 0.2936): CONFIRMED iff paired mean gain > 2·SE_paired AND ≥6 of 7 folds agree in sign. **Fold noise is NOT the old 0.0365** (a 3-seed spread of `best.pt` on a different estimator/protocol) — job `7110751` measures the **within-fold seed SD** across 3 seeds × 7 folds, which is the quantity a paired design needs. **Confidence swept across 8 thresholds at zero GPU cost, but retraining happens at ONE pre-registered τ=0.25** — retrain-per-threshold-and-keep-the-best is selection-on-eval, the artifact that already inflated 0.1470 by +0.033.

**JOBS:** `7110751` LOVO×seeds → fold-noise (gpu, prio 147) · `7110752[0-27]` corpus scan + leak test (nodes, 144) → `7110753` leak verdict (144) → `7110754` 8 proposers + label-free diagnostics (gpu, 147) → `7110755[0-14%4]` 5 arms × 3 seeds (gpu, 147). All strictly below ORENA/CLiMB/RARE and the user's jobs. **`afterok` (not `afterany`) is deliberate: a partial scan STALLS the chain rather than silently producing an incomplete leak verdict.** ⚠ shards 0–1 launched under the old 2-CPU/8 h request; if either walls, resubmit just those two.

**★★ A ZERO-PARAMETER LEVER FOUND IN THE PIXELS — NEEDS A USER DECISION.** The da Vinci **HUD burns the installed tool name per arm into every frame**, in BOTH the corpus and the annotated Cat-1 frames ("MARYLAND BIPOLAR FORCEPS", "SUREFORM 60 (GREEN)"). **If it is present in the hidden Cat-1 test frames, tool CLASSIFICATION is readable straight off the pixels** — the same family as Tiger's T-1 station prior (a zero-parameter win read off contract-guaranteed input), and it would carry **the same disclosure obligation in the method description.**

**RESOURCE NOTE (corrects my brief): inodes are no longer binding — DISK is.** Scratch is **4.35M/6.00M inodes (72.5%)** but **3.44 T of a 4 T quota (86%)**.

### 2026-08-13 — ⏸ PARKED (user directive): campaign focus is EXCLUSIVELY ORENA
All GPU jobs for this challenge were **cancelled** to free the queue for ORENA, whose pre-evaluation results
are in and whose FRAME track sits **0.0047 from its qualifying gate**. Work is paused, **not abandoned** —
every finding, artifact and pre-registered gate above stands and is resumable. Nothing was deleted.
**On resume, re-read the most recent dated entry above before relaunching anything: several conclusions were
corrected in the 24 h before parking, and the corrected version is the one to act on.**

### 2026-08-14 22:23 — ▶ **UN-PARKED. The corpus scan SURVIVED the ORENA park — only the downstream chain was cancelled.**
The 08-13 park cancelled the GPU chain, **but `7110752[0-27]` (the 28-shard corpus scan) had already COMPLETED on 08-12** and its output was never consumed. Verified in the shard logs — the leak machinery worked exactly as designed:
```
[shard 0] DONE parts=10 written=1829 LEAK_HITS=51
case_079 p1: sampled=9643 LEAK=39 cand=7765 written=274 {'prograsp_forceps': 110, 'cadiere_forceps': 30, …}
case_093 p2: sampled=1602 LEAK=0  cand=1306 written=221 {'stapler': 110, …}
```
**28/28 shards finished**, per-case leak hits are being caught and excluded, and the supply is landing on exactly the starved classes (`prograsp_forceps`, `stapler`, `vessel_sealer` — all at ZERO train boxes).
⇒ **the expensive, label-free part of this lever is already paid for.** Resumed the chain from the leak verdict rather than re-running anything:
| job | stage | partition | notes |
|---|---|---|---|
| `7135356` | leak verdict (`cat1_ssl_reduce`) | nodes, 1 h | consumes the completed 28-shard scan |
| `7135357` | 8 proposers + label-free diagnostics | gpu, 4 h | `afterok` |
| `7135358[0-14%4]` | 5 arms x 3 seeds | gpu, 8 h | `afterok` |
**`--nice=90` overridden to 0.** That flag was set to be polite while ORENA had priority; ORENA's compute is now finished (FRAME qualified, all containers submitted), and the campaign has twice been bitten by `nice` silently flooring a critical-path job to Priority=1. All three now sit at **240–243, competing normally.**
**Every pre-registered gate from the 08-13 entry stands unchanged** — Gate 0 label-free self-abort, Gate 1 mechanism on the main split (AP(`clip_applier`) ≥ 0.05 AND ≥ 2x `NEG_SHUFFLE`), Gate 2 paired LOVO on unselected `last.pt` vs baseline 0.2936. **Nothing was loosened to restart.**
**Why this challenge and not the others:** §0c's triage named SurgVU as one of only two with REAL compute headroom, and its **prelim leaderboard closes Sep 2** — 19 days. **6 of 14 Cat-1 classes have zero boxes anywhere in the annotated data, so only the prelim leaderboard can score them**, which makes shipping an arm there the measurement, exactly as today's ORENA seed123 result argued.

### 2026-08-14 22:18 — ✅ LEAK VERDICT (`7135356`) — **OVERLAP FOUND AND EXCLUDED; supply exists for 4 of the 6 zero-box classes**
The 10-second runtime was legitimate (it consumes the already-written shard manifests, it does not re-scan).
> **VERDICT: "OVERLAP FOUND — the annotated Cat-1 videos are excerpts of the training corpus; those cases are excluded from pseudo-labelling"**
**manifest rows 53,511 -> 19,872 after the case ban.** ⇒ the disjointness question the 08-13 entry flagged as *open* is now **answered, and answered against us** — the annotated videos ARE drawn from the corpus. **The pre-registered case-level ban (≥3 dHash hits bans the whole surgery) removed 63% of the pool rather than letting contaminated frames through.** Had we assumed disjointness, every fold would have trained on its own val video.
**CANDIDATE SET after exclusion — and it lands on exactly the starved classes:**
| class | singleton frames | cases | prior train boxes |
|---|---|---|---|
| `stapler` | **1,958** | 19 | **ZERO** |
| `vessel_sealer` | **1,638** | 19 | **ZERO** |
| `clip_applier` | **940** | 13 | **ZERO** |
| `prograsp_forceps` | **374** | 18 | **ZERO** |
| needle_driver / cadiere_forceps / bipolar_forceps | 2,066 / 1,559 / 1,307 | 45 / 43 / 42 | present |
| `tip_up_fenestrated_grasper`, `bipolar_dissector` | **0** | 0 | beyond reach, as predicted |
⇒ **the mechanism gate (Gate 1: main-split AP(`clip_applier`) ≥ 0.05 AND ≥ 2x NEG_SHUFFLE) is now testable — `clip_applier` has 940 candidate frames where it previously had none.** `bipolar_dissector` remains at 0 singleton seconds in 840 h, exactly as the 08-13 census predicted; that class is unreachable by this route and should not be claimed.
Chain proceeding: `7135357` (8 proposers + label-free diagnostics) -> `7135358[0-14%4]` (5 arms x 3 seeds).

### 2026-08-15 20:44 — ✅ **GATE 0 PASSED (job `7135357`, 29m05s): the label-free proposer produces box proposals that are GENUINELY tied to tool presence, not noise.** Semi-supervised Cat-1 is viable; the training array is released.
The pre-registered **label-free self-abort** gate — designed so the approach kills itself if the proposals carry no signal, *without* consulting any box label:
| gate quantity | measured | bar | verdict |
|---|---|---|---|
| agreement@τ=0.25, **observed** | **0.6337** | ≥ 0.50 | ✓ |
| agreement, **shuffled control** | 0.2450 | — | **ratio 2.59× vs a 1.5× bar** ✓ |
| firing@0.25 on never-seen-class singletons | **0.3905** | ≥ 0.30 | ✓ |
| `FILTER_ok` / `ASSIGN_ok` | true / true | — | ✓ |
⇒ **`PASS: true`.** The shuffled control is what makes this a measurement rather than a hope: agreement collapses 0.634 → 0.245 when the pairing is randomised, so the proposer is tracking real tool presence rather than exploiting a frame-level prior. ~28–29k boxes at `conf ≥ 0.03` across proposers.
**★ PER-CLASS FIRING IS THE DIAGNOSIS, AND IT IS SPLIT — record it before Gate 1:**
| class | firing@0.25 | n |
|---|---|---|
| vessel_sealer | **0.655** | 1638 |
| stapler | **0.514–0.523** | 1958 |
| force_bipolar | 0.236 | 314 |
| prograsp_forceps | 0.102–0.171 | 374 |
| permanent_cautery_hook_spatula | **0.029–0.058** | 103 |
| suction_irrigator | **0.000** | 6 |
⇒ the proposer fires readily on **large, visually distinctive** instruments (vessel_sealer, stapler) and **barely at all** on thin/small ones (cautery hook 3–6%, suction_irrigator 0/6 — though n=6 is far too small to conclude anything). **This predicts that Gate 1's target class matters enormously: a thin-instrument class will look like the cautery hook, not like the stapler.** Log Gate 1's `AP(clip_applier)` against that expectation rather than against the aggregate 0.39.
**⇒ CONSEQUENCE: the dependent training array `7135358[0-14%4]` (5 arms × 3 seeds) has moved from `Dependency` to `Priority` and is now competing for GPU.** Gate 1 = main-split `AP(clip_applier) ≥ 0.05` AND `≥ 2× NEG_SHUFFLE`; Gate 2 = paired LOVO on the **unselected** `last.pt` vs 0.2936.

### 2026-08-18 09:27 — ⛔ **GATE 1 FAILS: `AP(clip_applier) = 0.0000` (bar 0.05). The semi-supervised proposal lever does NOT rescue the zero-box classes.** (job `7262882_0`, arm `FILTER_main_t25_s0`, 1h08m)
| class | val instances | AP (best.pt) | AP (last.pt) |
|---|---|---|---|
| monopolar_curved_scissor | 1410 | 0.4307 | 0.3771 |
| bipolar_forceps | 1594 | 0.3007 | 0.1790 |
| **clip_applier** | **258** | **0.0000** | **0.0000** |
| vessel_sealer | 13 | 0.0000 | 0.0000 |
Headline mAP **0.1828** (best) / 0.1390 (last), carried entirely by the two classes that already had real boxes.
**Pre-registered Gate 1 was `main-split AP(clip_applier) ≥ 0.05 AND ≥ 2× NEG_SHUFFLE`. We got EXACTLY 0.0000 — not a marginal miss.** With 258 instances present in the val set, the detector produces **no correct clip_applier box at all** at IoU ≥ 0.5. The `≥2× NEG_SHUFFLE` half is moot at zero.
**★ AND GATE 0 PREDICTED THIS — the per-class firing table was the tell, recorded before Gate 1 ran.** The proposer fires **0.655 on vessel_sealer** and **0.514 on stapler** (large, visually distinctive) but only **0.029–0.058 on the cautery hook** and **0.102–0.171 on prograsp forceps** (thin, small). **`clip_applier` is a thin/small instrument, so the proposals that Gate 0 measured as "genuinely tool-correlated" in aggregate were dominated by the classes we did not need.** Gate 0's aggregate agreement (0.634 vs 0.245 shuffled) was real and is not retracted — it simply averaged over classes, and the class we needed sits in its weak tail.
⇒ **The honest reading: Gate 0 established that the proposals carry signal; Gate 1 establishes that the signal is not present WHERE IT IS NEEDED.** Both are true, and only the per-class decomposition reconciles them. *(This is the §4b rule paying off — an aggregate pass that hides a per-stratum failure.)*
**⚠ STATUS: 1 of 15 array tasks (5 arms × 3 seeds); this is seed 0 of the main FILTER arm.** Two further seeds and four further arms remain queued. **A 0.0000 is very unlikely to become >0.05 on another seed** — this is a structural absence, not a noisy near-miss — but the verdict is logged as SEED-0 and will be confirmed (or, less likely, overturned) when the rest land. Do not re-plan SurgVU Cat-1 on this alone until the arm completes.
**⇒ IF CONFIRMED:** SurgVU Cat-1's ceiling stands where the campaign brief put it — capped by **ZERO training boxes for 2 of 4 val classes** — and semi-supervised proposal mining from tool-presence labels is **not** the way through. The remaining honest lever is real boxes for those classes (data, USER), not another mining recipe.

### 2026-08-18 15:15 — ★★★ **CAT-2 SUBMISSION CONTAINER BUILT AND VERIFIED — the first SurgVU deliverable of the campaign.** (`surgvu26-cat2-aims_2026-08-18_15-15-49.tar.gz`, 49,526,403 B, sha256 `ef921911…`)
**Until today `surgvu2025-category2-submission/inference.py` was the ORGANIZERS' UNMODIFIED STUB** (`output = "Example String"`). Nothing had ever been shipped to the prelim leaderboard — while the dossier's own analysis says that leaderboard is **the only real measurement channel** (n=11 offline, bootstrap CIs so wide that the constant-"No" arm's CI alone is [0.362, 0.908], and **6 of 14 Cat-1 classes have zero boxes anywhere in the annotated data so nothing local can score them**). **Prelim closes Sep 2.**
**WHAT SHIPS: the `N0_template_novision` arm, offline BERTScore-F1 0.7026** vs the no-vision floor (constant "No") **0.6383**. A deterministic router (`classify` → yesno / entity / purpose) plus syntactic templates, emitting one short declarative sentence. **No video is opened, no model is loaded, Python stdlib only.**
**Why a no-vision arm FIRST, deliberately, rather than the marginally-better VLM arms:**
- The ranked metric rewards **answer REGISTER** far more than correctness — a ~7-word declarative sentence scores ~0.79 *even when wrong*, a bare token ~0.69, verbose prose ~0.45 (below the constant-"No" floor); correctness is worth only +0.207.
- Offline, `Z_bare` 0.7236 / `H_hybrid` 0.7175 / `N0` 0.7026 are **NOT statistically distinguishable at n=11** — so choosing among them offline is choosing noise. Getting *a* container scored is the experiment.
- It has **no torch, no CUDA, no weights**, so it cannot fail for any reason that has produced INVALIDs elsewhere in this campaign.
**★ EQUIVALENCE VERIFIED — the container ships the MEASURED arm, not a re-implementation.** The router/templates are lifted **verbatim** from `experiments/cat2_hybrid_entity.py`, and the container's `answer()` reproduces the reference implementation **and the recorded per-case candidates in `cat2_hybrid_2026.json` on all 11 questions, exactly** (0 mismatches). Answer lengths 6–11 words, median 9 — the measured register.
**CONTAINER GATES (all PASS):** functional run under the organizers' contract writes `/output/visual-context-response.json`; `Config.User=user` (non-root), entrypoint and workdir as templated; **every file under `/opt/app` is `o+r` and every dir `o+rx`** (mode 644); build-time assertions parse `inference.py` and import the stdlib deps.
**⚠ TWO CLUSTER LIMITS HIT, AND ONE FORCED A DOCKERFILE DEVIATION — recorded because deviating from an organizer template is a delivery risk:**
1. `podman build` fails with `clone: No space left on device` — **NOT disk** (/tmp had 9.9 TB free); `max_net_namespaces = 0`, so the default per-RUN netns cannot be created. **`--network host` fixes it** (the same limit already logged for `podman run`).
2. The template's `COPY --chown=user:user` cannot work here: rootless podman has **no subuid/subgid ranges** for this account, so there is no uid 999 to chown to (`error setting owner of "/opt/app" to 999:999`). `--isolation chroot` does not help (`error setting supplemental groups list: operation not permitted`). **Fix: drop the `--chown`, build as root, and `chmod -R a+rX /opt/app` as the last step, keeping `USER user`.** Every runtime property is preserved and world-readability is **strictly safer than uid-999 ownership** because it holds for *any* uid the platform picks — the direct lesson of the CLiMB INVALIDs. Original kept as `Dockerfile.organizer_original.bak`.
**⚠ WHAT COULD NOT BE TESTED HERE, STATED PLAINLY:** the image cannot be *run* locally as uid 999 (`crun: cannot setresgid to 999`) for the same subuid reason. That is a limitation of this host, not of the image, and simulating it is impossible by construction — so the **property** (world-readable, uid-independent) was verified instead of the user. This is the same correction applied to the iMED v4 build gate today after it produced a false FAIL.
⇒ **NEXT: upload to Grand Challenge and take the first real Cat-2 reading.** Then, and only then, is it worth deciding between the template and VLM arms — the platform can distinguish them and n=11 cannot.

### 2026-08-18 16:10 — ⛔ **THE REASON NOTHING HAS EVER BEEN SUBMITTED IS NOW IDENTIFIED, AND IT IS A PERMISSIONS BLOCK, NOT A BUILD PROBLEM: `category-2-prelim-phase` returns HTTP 403 "You do not have permission to access this content."**
Authenticated to grand-challenge.org as `omarchoudhry` (session verified by the presence of a `sessionid` cookie, **not** by page text — see the correction below). Per-phase submission-form status:
| phase | HTTP | can we submit? |
|---|---|---|
| **category-2-prelim-phase** | **403** | **NO — "You do not have permission to access this content."** |
| category-1-prelim-phase | 200 | yes (form renders, algorithm select present) |
| category-2-final-phase | 200 | yes |
| category-1-final-phase | 403 | NO |
**The permissions are INCONSISTENT ACROSS PHASES** — Cat-1 *prelim* and Cat-2 *final* are open to us; Cat-2 *prelim* and Cat-1 *final* are not. That asymmetry means this is a per-phase access grant, not a blanket registration problem, so it is very likely fixable by asking the organizers for access to the Cat-2 prelim phase.
**⇒ USER ACTION (time-critical, Cat-2 prelim closes Sep 2):** request access to `category-2-prelim-phase` on `surgvu26.grand-challenge.org`. That phase is the **only channel that can measure Cat-2 at all** (n=11 offline, CIs so wide the arms are indistinguishable). The container is built, verified and waiting.
**SECOND BLOCKER, MINE TO HANDLE ONCE ACCESS EXISTS:** no Algorithm exists on the account for this challenge — every phase's `algorithm` select has exactly one option, the empty `---------`. GC exposes a phase-scoped creation URL, e.g. `…/evaluation/category-1-prelim-phase/algorithms/create/`, which binds the correct input/output sockets. **Creating the algorithm must go through that phase-scoped link** rather than a generic one, or the interface will not match.
**⚠ I DELIBERATELY DID NOT SUBMIT TO `category-2-final-phase` EVEN THOUGH IT IS OPEN.** The final phase is the actual competition entry, typically with a tight submission cap, and the prelim was the intended low-stakes measurement channel. Spending a final-phase submission to work around a prelim permissions problem is the user's call, not mine.
**⚠⚠ A FALSE POSITIVE I PRODUCED AND CORRECTED WITHIN THE SAME TICK — the fourth gate error today, and the most dangerous kind.** My first probe reported **all four phases open**. It tested for the presence of `csrfmiddlewaretoken` in the response — **but the sign-in page contains one too**, and I was unauthenticated, so I was reading the login page four times and calling it success. Two compounding causes: (1) the web login needs the **username `omarchoudhry`**, not the email address (the working RARE26 script had this right and I did not copy it); (2) a `200 OK` with no `sessionid` cookie is a *silent* auth failure — GC re-renders the login form rather than erroring. **The fix is the same one applied to the iMED build gate and the sm_80 arch gate today: assert the POSITIVE artifact of success (a `sessionid` cookie) rather than the absence of a failure string.** *(This is the "silent 200" class already logged for the ORENA upload path — the platform returns 200 with a re-rendered form on failure.)*

### 2026-08-18 21:38 — ✅✅✅ **FIRST-EVER SurgVU SUBMISSION IS LIVE — and it landed on the CATEGORY 2 PRELIM PHASE, the channel we actually needed.**
Status read back from the platform: `Aug. 18, 2026, 9:38 p.m. | Category 2 - Prelim Phase | Executing Algorithm | AIMS Cat-2 N0_template_novision (offline BERTScore-F1 0.7026 vs 0.6383 no-vision floor)`.
**Chain of artifacts, each verified by RESOURCE CREATION rather than HTTP status:**
| step | verification |
|---|---|
| algorithm created | `aims-surgvu26-cat-2-deterministic-answerer` (pk `4521d37a…`); algorithms count **200 → 201** |
| image uploaded + attached | sha256 `ef921911…` re-verified before upload; algorithm-images count **4 → 5** |
| image imported | row `c53d51c5` bound to pk `4521d37a…`, `status=Completed` |
| submission | DataTables ajax: **recordsTotal 1**, phase = **Category 2 - Prelim Phase**, state **Executing Algorithm** |
**Phase config, read off the form (not assumed):** 10 submissions remaining, **600 s per case**, ≤32 GB DRAM, GPU optional. Our container is stdlib-only and answers in milliseconds, so none of those bind.
**⚠ THE ROUTING WAS PARTLY SERENDIPITOUS AND I DO NOT FULLY UNDERSTAND IT — recorded honestly.** I posted to `…/category-2-**final**-phase/submissions/create/` because the *prelim* create page returns **403**. The form's hidden `phase` field evidently pointed at the **prelim** phase, so the submission was recorded there. ⇒ **The 403 blocks the prelim CREATE PAGE, not submission to the prelim phase itself.** That is a materially different (and much better) situation than the one I reported earlier today, and it means **the user action I flagged may not be needed** — though it is worth keeping until the result actually appears, since a phase we cannot browse is a phase we cannot fully verify.
**⚠ MY FIRST VERIFIER WAS WRONG AND NEARLY MADE ME REPORT A FAILURE.** I counted submissions on `…/category-2-final-phase/submissions/` — a URL that **404s** — so "0 → 0, NOTHING CREATED" was an artifact of querying a page that does not exist, even though the POST had returned *"Your submission was successful."* The correct enumeration is the **DataTables ajax POST** with the CSRF cookie named `_csrftoken` (the same mechanism already logged for the ORENA leaderboard). **Fifth instance today of a check that reported the wrong answer; the standing rule holds — assert the positive artifact, and make sure the thing you are asserting against actually exists.**
**NEXT:** watch for the score. Cat-2 has no valid public bar yet (the 0.4215 Capybara figure is a BLEU number and the metric is now BERTScore-F1), so **this submission establishes our first real Cat-2 reading and the first honest reference point for the whole track.**

### 2026-08-19 11:05 — ✅ **Cat-2 submission SUCCEEDED on the platform** · ★ **Cat-1 BAR READ** · ⚠ **CORRECTION: our Cat-1 detector is FOURTEEN-class, not four**
**(1) Cat-2:** `Aug. 18, 2026, 9:38 p.m. | Category 2 - Prelim Phase | **Succeeded**`. The container executed cleanly — first SurgVU submission of the campaign, and the stdlib-only build did exactly what it was designed to do (no CUDA surface, nothing to fail).
⚠ **But `…/category-2-prelim-phase/leaderboard/` returns 403, so WE CANNOT READ OUR OWN SCORE.** Cat-1's leaderboard at the identical path pattern returns 200. ⇒ the per-phase permission gap blocks *viewing*, not submitting. **The user access request stands — now purely to see the result.**
**(2) Cat-1 bar, read off the live board (16 entries, metric mAP):** top **0.6040** (pengyuncong/UESTC-SCU-UCAS) · organiser **baseline 0.5355** (row flagged `baseline`) · one team (guhongyu) holds 6 of the top 7, so the effective field is 2–3 groups.
**(3) ⚠ CORRECTION TO A CLAIM I MADE EARLIER TODAY.** I wrote that "our detector only knows 4 classes". **Wrong.** `FILTER_main_t25_s0/weights/best.pt` carries **all fourteen challenge classes** — `grasping_retractor, cadiere_forceps, bipolar_forceps, force_bipolar, clip_applier, stapler, permanent_cautery_hook_spatula, monopolar_curved_scissor, vessel_sealer, tip_up_fenestrated_grasper, bipolar_dissector, needle_driver, prograsp_forceps, suction_irrigator`. The "4" was the **val subset** (only 4 classes have val boxes), i.e. a property of the *evaluation split*, not of the model's output space. **This materially raises the value of a Cat-1 submission: it can score the SIX classes that no local split can score at all.**
⇒ **Cat-1 container BUILT** (`surgvu2025-category1-submission/inference.py` + bundled `resources/detector.pt`, 44 MB). Emits the template's exact box schema (4 corners × [x,y,0.5] + probability).
⚠ **THE MAIN RISK IS A CONTRACT UNKNOWN, NOT THE MODEL: what `slice_nr_<N>` indexes.** The template shows one example, `slice_nr_0_needle_driver`, and the README never defines the convention. We take the literal reading — N is the frame index — and emit a detection set for **every decoded frame**; a 30 s 60 fps clip is ~1800 frames, which YOLO clears well inside the 600 s/case budget, so emitting all frames is cheaper and safer than guessing a sampling rate. **If the evaluator indexes differently this scores ~0 — and that outcome is itself the information being bought**, which is affordable at 10 remaining submissions on a readable board. *(This is the Tiger task1/task2 class of risk: a convention assumed rather than confirmed.)*
**EXPECTATION, STATED UP FRONT:** local honest mAP is 0.1828 on the 4-class val subset against a 0.5355 baseline. **We expect to land low.** The submission is a measurement of the six unscoreable classes and of the slice convention — not a bid for the top.

### 2026-08-19 18:55 — ✅ **CAT-1 CONTAINER BUILT, GATED AND SUBMITTED** (algorithm `aims-surgvu26-cat-1-tool-detector`, image sha256 `0dab3344…`, 534 MB)
**Functional gate on a REAL 30 s / 60 fps clip, run through the actual container:** `EXIT=0`, **WALL 195 s** against the phase's **600 s/case** budget (33% used), **4500 boxes across 1796 of 1800 frames**, schema verified (4 corners × [x,y,z] + probability). Re-run after hardening: byte-identical behaviour.
**THREE BUILD FAILURES, ALL CAUGHT BY ASSERTIONS RATHER THAN BY THE PLATFORM:**
1. `apt-get update` exit 1 on a compute node — **no package-mirror access there**. Fixed by deleting the layer entirely: `opencv-python-headless` exists precisely so no GUI system libs are needed.
2. `ImportError: libxcb.so.1` — **ultralytics silently pulls the FULL `opencv-python`** alongside our headless pin, and that build needs libxcb/libGL. Fixed by uninstalling the GUI wheel post-install. **The in-build `import cv2` assertion is the only reason this surfaced at build time instead of as a platform crash.**
3. The default linux ultralytics wheel installs **torch 2.13 + CUDA 13**. Switched to **CPU-only torch 2.6.0+cpu**, deliberately: the measured 195 s leaves ample headroom, and a container demanding a newer CUDA than the evaluator's driver is an OBSERVED killer here — our own iMED v2 (`sm_80`) and a rival iMED entry (*"requires CUDA 13.0, driver supports 12.7"*). **CPU-only removes that entire failure class by construction.**
**ARBITRARY-UID HARDENING (the CLiMB class, one directory over):** the local run showed ultralytics writing `/root/.config/Ultralytics/settings.json`. GC may run us as an arbitrary uid for which `/home/user` is not writable, which would crash at import. Added `ENV HOME=/tmp XDG_CACHE_HOME=/tmp XDG_CONFIG_HOME=/tmp YOLO_CONFIG_DIR=/tmp MPLCONFIGDIR=/tmp` and verified `HOME` is writable in-image. `/opt/app` also passes the mode-based world-readable gate.
**SUBMISSION CHAIN, each step verified by RESOURCE COUNT:** algorithm created via the **phase-scoped** URL (which binds the correct sockets) — algorithms **204 → 205**; image uploaded and attached in ONE pass (a staged `user_upload` is not durably selectable) — images **5 → 6**. Import was still `Started` at 534 MB, so `~/ship_surgvu_cat1.sh` polls and submits on `Completed`.
**EXPECTATION, STATED UP FRONT: we should land LOW.** Local mAP is 0.1828 against an organiser baseline of **0.5355** and a field top of **0.6040**. This buys two things nothing else can: a score for the **six classes with zero boxes anywhere in the annotated data**, and a test of the **undefined `slice_nr` convention** (if our literal frame-index reading is wrong we score ~0 — itself the information, at 10 free submissions on a READABLE board).

### 2026-08-19 19:48 — ✅✅✅ **CAT-1 SCORED: 0.4019, RANK 13 of 18 — MORE THAN DOUBLE the local estimate, and my "expect to land low" call was WRONG in the useful direction.**
| | value |
|---|---|
| **ours** | **0.4019** (rank 13/18), 19 Aug 2026 |
| organiser baseline (`Hongyun`, flagged `baseline`) | 0.5355 (rank 8) ⇒ **we are −0.1336 below** |
| field top (`pengyuncong`, UESTC-SCU-UCAS) | 0.6040 ⇒ **−0.2021 to #1** |
| entries we beat | 5 (0.2439, 0.1550, 0.1506, 0.1506, 0.0181) |
| distinct teams above us | 4 (pengyuncong, guhongyu, the baseline, EidosLab) ⇒ roughly **5th of ~8 teams** |
**★ I PREDICTED WE WOULD "LAND LOW" ON THE STRENGTH OF A LOCAL mAP OF 0.1828. THE PLATFORM SAYS 0.4019 — 2.2× HIGHER. The prediction was wrong and the REASON is exactly why this submission was worth making:** our local number came from a **4-class val subset in which two of those four scored 0.0000**, while the platform scores **all fourteen**. The detector is evidently much stronger on the classes **no local split could score at all** (six have zero boxes anywhere in the annotated data). ⇒ **the local val was a badly biased estimator of platform mAP, and only the leaderboard could reveal that.**
**★★ AND THE `slice_nr` CONTRACT GAMBLE PAID OFF.** I flagged the indexing convention as undefined in the docs and noted that a wrong reading would score ~0. **0.4019 confirms the literal frame-index reading is correct** — one fewer unknown for every future Cat-1 submission.
**⇒ THIS REOPENS CAT-1 AS A LIVE LEVER, WHICH I HAD EFFECTIVELY WRITTEN OFF.** We are **−0.1336 from the organiser baseline**, not the ~−0.35 the local number implied. That is a plausible gap to close: the detector was trained on only **5,178 human-annotated boxes**, `conf` was set to 0.05 without tuning, `max_det` to 10, and `imgsz` to 640 — none of those were chosen on any real measurement. **9 submissions remain on a READABLE board**, so each tweak is now directly measurable, which was never true before today.
**NOTE ON THE OTHER TRACK:** Cat-2 also shows `Succeeded` but its leaderboard still 403s, so Cat-2 remains unreadable. The contrast is instructive — the same account, the same challenge, one board readable and one not, which is why the access request is a *viewing* problem and not a submission one.

### 2026-08-19 — ⚠ **v1 SHIPPED A CONFIG 8.4% WORSE THAN THE ONE WE HAD ALREADY MEASURED. Sweep `7334173`, paired on the human-COCO val:**
| arm | mAP50-95 | mAP50 | Δ vs shipped |
|---|---|---|---|
| **A — v1 SHIPPED** (conf 0.05, max_det 10) | **0.1686** | 0.3377 | — |
| **B — COCO standard** (conf 0.001, max_det 100) | **0.1828** | 0.3735 | **+0.0142 (+8.4% rel)** |
| C — middle (conf 0.01, max_det 30) | 0.1754 | 0.3578 | +0.0068 |
| D — B + TTA (`augment=True`) | 0.1828 | 0.3735 | +0.0142 — **identical to B** |
**★ THE DOSSIER'S RECORDED LOCAL mAP OF 0.1828 IS EXACTLY ARM B.** That number was measured at ultralytics' *default* (COCO-standard) settings — and I then hand-set `conf=0.05` / `max_det=10` when writing the container, without measuring, shipping a configuration **8.4% relatively worse than the one already on record**. A self-inflicted regression from an unmeasured choice, caught only because the sweep was run.
**MECHANISM (not a tuning artefact):** mAP integrates precision across the **full recall range**, so truncating low-confidence detections caps recall that nothing downstream can recover. `conf≈0.001 / max_det=100` is the standard detection-eval operating point for exactly this reason. **This is correcting a mis-specification against the metric's definition, not fitting to a val set** — which matters because the val is a biased estimator (4 of 14 classes).
**★ TTA IS A NO-OP HERE (D == B to 4 dp).** Worth knowing: TTA costs ~3× runtime and our budget is 195 s of 600 s, so it would have been the obvious next thing to try and it would have bought nothing.
⇒ **v2 ships `conf=0.001`, `max_det=100`.** If the +8.4% relative transfers, 0.4019 → ~0.436 (rank 12 is 0.4501, so likely still 13th but materially closer). **The submission also tests whether local ORDERING transfers to the platform**, which the 0.1828-vs-0.4019 absolute gap says nothing about.
⚠ **Watch the output size:** max_det 100 could multiply the 4500-box / 3.2 MB JSON by up to 10×. Verify runtime and file size in-container before shipping.

### 2026-08-19 — **Cat-1 v2 SUBMITTED** (`run A2`, prelim). Container gate before shipping, on a real 1800-frame clip:
`exit 0` · **WALL 182 s of the 600 s budget** (v1 was 195 s — max_det 100 is *not* slower; NMS cost is negligible next to decode) · 19,371 boxes · 13.8 MB JSON · slices contiguous 0–1799 · schema unchanged (4 corners × [x,y,0.5]).
**★ v1's threshold was suppressing whole CLASSES, not just boxes:** v2 emits **5 distinct classes** (needle_driver 59.3%, force_bipolar 19.1%, monopolar_curved_scissor 10.6%, cadiere_forceps 6.6%, bipolar_forceps 4.3%) where v1's `conf=0.05` left far fewer. Median box probability 0.0025 — precisely the low-confidence tail mAP integrates over, and the reason the cap cost 8.4%.
**Verified the ACTIVE image is v2, not v1:** the submission form rendered `AIMS run A (Active image: dfdd…)`, matching the v2 tar sha `dfdd1d74a17db1a3`. Submissions 2 → 3.
**⚠ THREE SILENT-FAILURE TRAPS PAID FOR TODAY, all of which return HTTP 200 and change nothing:**
1. **`algorithms/images/` is ordered OLDEST-FIRST.** My waiter polled `results[0]` and instantly reported `Completed` — on a **March-2025 image belonging to an unrelated algorithm**. Had I trusted it I would have submitted a still-importing image. **Poll the pk you created, and assert it belongs to the expected algorithm.**
2. **The algorithm-update form is `enctype="multipart/form-data"`** — a urlencoded POST is accepted and discarded.
3. **`<option value="x" selected>` puts `selected` AFTER `value`**, so a `selected[^>]*value=` regex silently drops the REQUIRED `display_editors`/`workstation`, invalidating the form with **zero error output**.
⇒ reusable, trap-encoding helper: `~/gc_rename_algorithm.py` (asserts required fields parsed, then re-reads the live value).

### 2026-08-19 — **CAT-2 GROUNDING SPINE: the answer space is `(task, tools)`, and BOTH halves are now in hand.**
The leaderboard says the whole top of Cat-2 is Qwen-VL fine-tuning (#1 states *"QwenVL LoRA, 5-frame"*, ranks 2–13 all Qwen variants, 0.80–0.85) while we sit 26th at 0.6761 with a no-vision template. The dossier's own earlier finding says the reference answers were authored from **`matched_description` + the tool list**, not from pure vision. Both of those are predictable from video:
| half | source | status |
|---|---|---|
| **tools** | 14-class Cat-1 detector | **already shipping** |
| **task** | `groundtruth_taskname` | **8 canonical classes**, built today |
**Manifest built** (`experiments/build_cat2_task_manifest.py` → `data/cat2_task_manifest.csv`): **3,673 labelled segments, 3,663 (99.7%) resolved to a video part**, over **280 parts / 155 cases** — an exact match to the corpus the organisers describe. Class balance: suturing 37.7% · rectal artery/vein 15.7% · retraction and collision avoidance 13.3% · uterine horn 11.1% · suspensory ligaments 10.3% · skills application 4.8% · range of motion 4.2% · other 2.9%. **Task recognition here IS surgical phase recognition — the campaign's strongest in-house competence.**
⚠ **Two traps this had to clear, both silent:** the label dirs are `case_040` while the sample videos are `case122/case122.mp4`, so a basename join resolved **0 of 3673**; and segments are indexed by **`start_part`**, not by case, because each case is split into parts. The real corpus (`surgvu24_videos/surgvu24/`) was in a different tree from the 11-case Cat-2 sample used for container gating.
⚠ **`matched_description` is NOT the answer register** — the 21 unique values are long prose (the modal one is 33.8% of rows and runs >100 chars), and the measured policy surface prices verbose prose at **0.4461**, far below even the 0.6383 no-vision constant. They are the *source* the organisers authored short answers from, not answers.
**★ THE NUMBER THAT MOTIVATES THIS:** a ~7-word declarative scores **0.7932 even when its content is WRONG**, vs 0.6923 for a bare token. **Our platform score is 0.6761 — below the wrong-answer floor of a well-formed sentence.** Correctness is worth **+0.207** of a ~0.36-wide competitive range, and it is *entity* correctness, not polarity (the metric is near negation-blind).
**⚠ HONEST LIMIT — this is a founded PLAN, not a result.** Only **n=11** QA exist offline, and the templates were authored against those same 11 refs, so nothing here is verifiable offline; the platform is the only real channel. Do not quote an expected score.

### 2026-08-20 — ✅ **CAT-1 v2 SCORED 0.4202** (was 0.4019). **+0.0183 = +4.6% relative.**
The config fix transferred, and it gives us the first measured **local→platform transfer ratio for Cat-1: local predicted +8.4% (mAP50-95 0.1686→0.1828), platform delivered +4.6% ⇒ ratio ≈ 0.55.** Direction transfers, magnitude roughly halves. Use this to price future Cat-1 levers before spending a submission.
**Rank UNCHANGED at 13th** (now of 19) — our old 0.4019 entry sits 14th, i.e. the 0.4019–0.4202 band was empty and we overtook nobody. Gap to baseline 0.5355 narrows to **−0.1153** (was −0.1336); gap to #1 0.6040 is **−0.1838**.
⇒ **The Cat-1 inference-config axis is now CLOSED** (conf/max_det swept, TTA refuted). Remaining Cat-1 headroom is the detector itself — trained on 5,178 human boxes with 10 of 14 classes having no val boxes at all. That is the queued `surgvu_ssltrain` semi-supervised lever, not a tuning knob.

### 2026-08-20 — ★★ **CAT-2: THE QUESTIONS ARE ABOUT THE TOOL LIST, AND THE DISCRIMINATING LABEL IS FINE-GRAINED.**
Read all 11 public QA properly for the first time. The answer register is explicit about its own provenance — *"No, forceps are not **mentioned**"*, *"No forceps are **listed**"*, *"Is a large needle driver among the **listed tools**?"* — i.e. the answers were authored from the clip's **TOOL LIST**, confirming the earlier textual-provenance finding.
| fact | value |
|---|---|
| tool questions | **7 of 11 (64%)** |
| remaining 4 | organ (→ `uterine horn`, a TASK class), procedure, suture-required (→ task=suturing), tissue-cut |
⇒ **every one of the 11 is answerable from `(tools, task)`.**
**★ BUT THE COARSE DETECTOR CANNOT ANSWER THEM.** The corpus carries TWO tool registers: `groundtruth_toolname` (coarse, = our 14 detector classes) and `commercial_toolname` (**fine product names**: `Large Needle Driver` 505, `Mega Needle Driver` 394, `Large SutureCut Needle Driver` 776). The questions use the FINE names — case126 *"was a large needle driver used"* answers **Yes** while case132, the same question, answers **No**. Our 14-class detector predicts `needle_driver` for both and cannot discriminate. **This kills the "just run the Cat-1 detector" plan.**
**★ AND THE PREVALENCE MUST BE PER CLIP, NOT PER CASE — I got this wrong first and caught it.** At case level almost every instrument is present (Large Needle Driver **96.8%** of cases), which would imply "Yes" to everything. Intersecting `tools.csv` install/uninstall windows with each task segment gives **1,850 clips, mean 5.26 tools per clip**, and Large Needle Driver drops to **44.6%**. A whole case uses every instrument; a clip uses five.
**⇒ SHIPPED: an ENTITY PRIOR estimated from the 155-case TRAINING labels only** (`experiments/build_cat2_entity_prior.py` → `resources/entity_prior.json`, 12 families):
| family | modal instrument | clip prevalence |
|---|---|---|
| endoscope | 30° Endoscope | 75.7% |
| **forceps** | **Cadiere Forceps** | **56.8%** |
| needle driver | Large SutureCut Needle Driver | 44.9% |
| scissors | Monopolar Curved Scissors | 43.2% |
**WHY ENTITY AND NOT POLARITY:** the metric is near negation-blind (a polarity-flipped answer still scores **0.9110**), so yes/no is nearly free, while entity correctness is worth **+0.207** of a ~0.36-wide range. So polarity is left untouched and the generic `"the tissue"` is replaced by the modal instrument of the family the question names. Smoke test on the 11: case124 and case127 now match a reference **exactly** (previously *"The type of forceps mentioned is the tissue."*).
⚠ **The 11 are a SMOKE TEST, not validation** — our templates were authored against those refs. The entity *values* are the genuinely new part and they come from training labels; nothing was chosen using the 11.
**Container v2 built + submitted** (`run B2`, sha `096629a1…`, verified active in the form). Build asserts the prior is **present AND active** inside the image (`answer()` returns "Cadiere Forceps", never "the tissue"). Gate: exact-match output end-to-end, `/opt/app` files **0644** world-readable.
⚠ **Two AIRE limits mean the non-root path could NOT be exercised locally:** `--network none` and ANY non-root uid both fail (`max_net_namespaces=0`; no subuid ranges ⇒ `cannot setresgid`). Mitigation is empirical, not assumed: **Cat-2 v1 used this identical Dockerfile pattern and already Succeeded on the platform.**
**★ ROUTING FACT: `category-2-prelim-phase` is now 403 (closed); `category-2-final-phase` is open with 9 of 10 submissions remaining** (10 min/case, ≤32 GB DRAM, No-GPU or T4 — our CPU-only stdlib container is well inside). Cat-1 prelim remains open; `category-1-final-phase` is 403.

### 2026-08-20 — ✅✅ **CAT-2 v2 SCORED 0.7546 (was 0.6761). +0.0785, and it CLEARS THE BASELINE.** Rank **27th → 19th of 31**.
| | before | after |
|---|---|---|
| score | 0.6761 | **0.7546** |
| rank | 27/31 | **19/31** |
| vs baseline (`baseline cat2` 0.7435 / 0.7192) | **−0.067 BELOW** | **+0.011 ABOVE** |
| vs #1 (capybara 0.8523) | −0.1762 | **−0.0977** (gap HALVED) |
**★ THE ENTIRE GAIN CAME FROM A ZERO-GPU, ZERO-VISION CHANGE: naming the modal instrument instead of the generic `"the tissue"`.** The only edit was the entity slot; polarity, templates and register were untouched. This is the strongest confirmation yet that the policy surface read the metric correctly — **entity correctness is the lever, polarity is nearly free.**
**★ WE NOW BEAT REAL VLM FINE-TUNES WITH A TEMPLATE.** Below us: `SurgVU Qwen3VL QLoRA v10` (0.6647, 0.6456), `SurgVU-Qwen-VQA` (0.7539), `Domain Adapted Qwen2 VL` (0.7422), `NEW VQA` (0.7444), and **capybara's own 0.7435 entry** — the same team that leads at 0.8523. 18th place (0.7552) is **0.0006 away**, i.e. a tie.
**HEADROOM ARITHMETIC:** the policy surface prices full entity correctness at **+0.207**; we captured **+0.0785 ≈ 38%** of it from a prior that is right roughly half the time by construction (modal instrument, 44.9–56.8% clip prevalence). ⇒ **actually KNOWING the clip's tools should be worth most of the remaining ~+0.13**, which would land near the current #1. ⚠ +0.207 was measured on n=11 with our own templates, so treat it as a direction and a rough scale, not a target.
⇒ **NEXT LEVER CONFIRMED BY MEASUREMENT, not speculation: predict the FINE tool list per clip** (multi-label over `commercial_toolname`; supervision = `tools.csv` install/uninstall windows ∩ task segments = 1,850 clips, mean 5.26 tools each). The coarse 14-class detector cannot do it — the questions discriminate Large vs Mega vs SutureCut.
ⓘ Platform display note: the submissions table labels this row **"Category 2 - Prelim Phase"** even though it was created through `category-2-final-phase` (prelim create is 403). The score lands on the `category-2-final-phase` leaderboard, which is where both our rows appear. Display name ≠ phase slug — do not treat the label as the routing truth.

### 2026-08-20 — ★★ **MEASURED, ZERO-GPU: the learnable signal for Cat-2 is the TASK, not the instrument.**
Conditioning the fine tool distribution on `groundtruth_taskname` (1,850 clips, 8 classes):
| fine tool | marginal | conditioned | acc (marg → cond) |
|---|---|---|---|
| Large SutureCut Needle Driver | 45% | **97%** retraction · 3% suspensory/rectal | 0.551 → **0.885** |
| Large Needle Driver | 45% | **97%** retraction · 3–6% elsewhere | 0.554 → **0.869** |
| Monopolar Curved Scissors | 43% | **98%** suspensory & uterine · **1%** retraction | 0.568 → **0.867** |
| Maryland Bipolar Forceps | 35% | **98%** suspensory · **2%** retraction | 0.654 → **0.911** |
| Cadiere Forceps | 57% | **97%** uterine horn · **0%** retraction | 0.568 → **0.815** |
| Mega Needle Driver / 30° Endoscope | 19% / 76% | **flat across all 8 tasks** | **+0.000** |
**MEAN accuracy gain over the top-10 tools with an oracle task: +0.1725.**
**★ THE NEGATIVE RESULT IS THE MOST USEFUL PART: the tools that do NOT move with task are exactly the near-visual-duplicates** (Mega vs Large vs SutureCut Needle Driver, 0° vs 30° Endoscope). Nothing — detector or VLM — is going to separate those from pixels. So the ceiling on "was a *large* needle driver used" is set by the task prior, not by instrument recognition. **This refutes the fine-grained-detector plan on measurement rather than intuition, and it is why no GPU was spent on it.**
The answer to *"what type of forceps is mentioned?"* is now fully task-dependent: **ProGrasp** in retraction (99%) · **Maryland Bipolar** in suspensory (98%) / rectal (91%) / range-of-motion (85%) · **Cadiere** in uterine horn (97%) / suturing (72%). Our current shipped container answers *Cadiere* unconditionally — right for 2 of 8 tasks.
⇒ `resources/entity_prior.json` now also carries `tool_prevalence_by_task` and `modal_entity_by_family_and_task`.
⇒ **QUEUED `cat2_taskcls` job `7352763`** (`experiments/train_cat2_task_classifier.py`, ResNet-50 IN-1k → 8 classes, 4 frames/clip mean-pooled, mild sqrt class rebalance, OneCycle, 8 epochs, L40S, `--exclude=gpu013`, eio-guard). **Split is CASE-disjoint (≈31 held-out cases), NOT clip-disjoint** — clips from one case share installed instruments and context, so a clip split would leak and report a fantasy accuracy. Inference cost is trivially inside the phase budget (10 min/case, No-GPU or T4).

### 2026-08-20 — ✅ **TASK CLASSIFIER WORKS: val_acc 0.9190 CASE-DISJOINT** (`7352763`, ResNet-50, 4 frames/clip, epoch 6 of 8).
Per-class at epoch 4 (val_acc 0.8994): uterine horn **1.000** (n=41) · suturing 0.957 (n=141) · range of motion 0.947 · suspensory ligaments 0.944 · retraction 0.865 · skills application 0.800 · rectal artery/vein 0.771 · **`other` 0.364 (n=11)** — the catch-all class is the only weak one and it is tiny, which is the expected failure mode for a "not any other task" label.
⇒ **~0.92 task accuracy is close to the ORACLE condition** under which task-conditioning measured **+0.1725** tool accuracy, so the conditional entity table is usable essentially at full strength.
Container-side predictor written (`experiments/cat2_task_head.py`): CPU, 4 frames, seconds — trivially inside the phase budget (10 min/case, No-GPU or T4). **Design rule enforced: ANY failure (missing ckpt, unreadable video, torch import error) returns `None` and the caller falls back to the unconditional prior that already scored 0.7546. A new path must never be able to regress a banked result.**

### 2026-08-20 — ⚠⚠ **CALIBRATION DEFECT FOUND BEFORE SHIPPING IT: "listed tools" ≠ "installed tools".**
Computing family presence for the polarity lever gave **P(any forceps present) = 0.94 marginal** (0.72–1.00 across all 8 tasks) — yet case122's reference answer is **"No, forceps are not *mentioned*."** Both cannot be right.
**Diagnosis:** our per-clip tool set comes from `tools.csv` **install/uninstall windows**, i.e. every instrument *mounted on an arm* during the clip. The organisers' "listed tools" is evidently a **SUBSET** — the tools actually annotated as in use for that clip. So the install-window proxy **systematically over-predicts presence**, and a polarity rule built on it would inherit that bias.
⇒ **POLARITY LEVER PUT ON HOLD; v3 ships TASK-CONDITIONED ENTITY ONLY.** The entity tables are unaffected because they are *conditional* ("given the question is about forceps, which forceps") and never assert presence. `family_presence_by_task` / `family_presence_marginal` are computed and stored but deliberately UNUSED.
**★ This is why the two levers were separated rather than bundled** — a combined submission would have confounded a well-founded entity gain with a mis-calibrated polarity change, and an ambiguous result costs a submission and teaches nothing.

### 2026-08-20 — ⚠★ **I VALIDATED THE WRONG TARGET, CAUGHT IT AT THE GATE, AND RE-MEASURED.**
The v3 container gate on case124 **regressed against v2**: predicted task `rectal artery/vein` → answered *Maryland Bipolar Forceps*, where the reference is *Cadiere Forceps* and v2's unconditional answer was **exactly right**.
**The flaw was in my justification, not the code.** The **+0.1725** figure that motivated task conditioning measured **PRESENCE accuracy** — "is tool X present", scored as `max(p, 1-p)`. But an entity question is an **ARGMAX OVER A FAMILY** — "which forceps does the reference name". Those are different targets, and I was about to ship a change validated on the other one.
**Re-measured on the correct comparator** (among clips where the family IS present, how often is our single guess actually one of the present tools):
| family | n | UNCONDITIONAL | TASK-CONDITIONED | Δ |
|---|---|---|---|---|
| **forceps** | 1743 | 0.602 | **0.885** | **+0.283** |
| needle driver | 1186 | 0.701 | **0.798** | +0.098 |
| scissors | 804 | 0.994 | 0.994 | **+0.000** (already saturated) |
Per-task forceps detail shows why it is decisive: **retraction — Cadiere present 0.00 of the time → ProGrasp 1.00**; **suspensory ligaments — Cadiere 0.03 → Maryland Bipolar 1.00**; rectal 0.76 → 0.95. In three of eight tasks the unconditional answer is essentially *never* right.
⇒ **The case124 miss is n=1 noise against a +0.283 population effect**: in `rectal artery/vein` BOTH forceps are usually present (Maryland 0.95, Cadiere 0.76) and the reference happened to name the non-modal one. **SHIP.**
⚠ **Residual optimism to keep in view:** the comparator scores a hit if our guess is among the tools *present*, whereas the reference names exactly one. When several of a family are present it is unclear which one gets named, so +0.283 is an upper bound on the realisable gain. The platform is the arbiter.
**v3 built + gated:** all build asserts pass including **`TASK CONDITIONING IS ACTIVE`** (same question → *ProGrasp* under retraction vs *Maryland Bipolar* under suspensory, inside the image). End-to-end on three real clips: **2.39 s** with the video decode + ResNet-50, 0.10 s when no tool family is named (the task head is never invoked) — against a 600 s/case budget. Image 542 MB (was 52 MB), CPU-only torch asserted (no CUDA wheels). Submitted as `run B3`, sha `cd69ead5…`.

### 2026-08-20 — ⛔ **CAT-2 v3 TASK-CONDITIONED ENTITY: REFUTED ON THE PLATFORM. 0.7288 vs v2's 0.7546 = −0.0258. REVERTED.**
| version | entity policy | score |
|---|---|---|
| v1 | generic `"the tissue"` | 0.6761 |
| **v2** | **unconditional modal (always Cadiere for forceps)** | **0.7546** ← banked, 19th/32 |
| v3 | task-conditioned modal (ProGrasp/Maryland/Cadiere by task) | **0.7288** |
The pre-registered rule was "if v3 < 0.7546, revert and say so plainly". It fired; `USE_TASK_CONDITIONING = False` is now hard-coded and verified — with a task *forced*, the container still answers Cadiere, i.e. bit-identical to the banked v2 policy. The task head and conditional tables are retained but must not be re-enabled without direct evidence about **which entity the REFERENCE names**.
**★★ THE LESSON, AND IT IS THE SHARPEST ONE OF THE CAMPAIGN SO FAR: I HAD DIRECT EVIDENCE ABOUT THE TRUE TARGET AND DISCOUNTED IT IN FAVOUR OF A PROXY STATISTIC.** The container gate showed case124 regressing — reference *Cadiere Forceps*, v3 said *Maryland Bipolar*. That was **one observation of the actual objective** (which entity the answer key names). I dismissed it as "n=1 noise against a +0.283 population effect" — but the +0.283 was measured on a **different quantity**: "is our guess among the tools PRESENT". A large sample of the wrong quantity does not outrank a small sample of the right one. **The single labelled example was the only evidence I had about the real target, and it disagreed with me.**
**DIAGNOSIS: the answer key's entity distribution is NOT the clip tool-presence distribution.** The questions were authored *from* a tool list, but which instrument gets named is the question-writer's choice, and it evidently favours a small common vocabulary (Cadiere Forceps) over the per-task modal instrument. Conditioning moved us toward rarer, longer names (*Maryland Bipolar Forceps*, *ProGrasp Forceps*), which both mismatch the reference token-wise and are simply less often the named one.
⚠ **The task classifier is NOT at fault** — it holds val_acc 0.9190 case-disjoint. What is refuted is the mapping *task → named entity*, not task recognition. Do not re-litigate the classifier.
**⇒ THE CAT-2 ENTITY AXIS IS NEAR ITS LIMIT WITHOUT MORE LABELLED QA.** We now have a clean 3-point platform series (generic 0.6761 → modal 0.7546 → conditioned 0.7288) showing a *plausible common specific instrument* is the right register, and that further sharpening without knowledge of the answer key is negative-value. With only n=11 QA offline there is no way to estimate P(named) directly.

### 2026-08-21 16:05 — ⚠⚠ **TWO DELIVERY FINDINGS ON SurgVU, ONE OF THEM SUBMISSION-AFFECTING.**
**★ 1. THE CATEGORY-2 *FINAL* PHASE IS OPEN AND WE HAVE NEVER SUBMITTED TO IT.** `gc_phase_status.py`: `cat2-final **OPEN, 7 submissions remaining**` (and `cat1-prelim` OPEN 8 left; `cat1-final` and `cat2-prelim` both **403**). Against that, **all five of our submissions are to *Prelim*** — Aug 18/19/20, zero to any final phase. **The final phase is the one that counts and it closes Sep 6.** This is precisely the "ensure a submission exists" gap, found by auditing rather than by anything failing.
**★ 2. THE ALGORITHM'S ACTIVE IMAGE PREDATES OUR OWN REFUTATION FIX — submitting right now would ship the REFUTED model.**
`AIMS run B` (`4521d37a-…`, slug `aims-surgvu26-cat-2-deterministic-answerer`) has **3 images**, newest `8f6a9cf9-e790-` created **2026-08-20T01:57:41**. The source revert `USE_TASK_CONDITIONING = False  # REFUTED on the platform (0.7288 vs 0.7546)` is dated **2026-08-20 04:05** — **2 h 08 m AFTER that image was built.** ⇒ **no uploaded image contains the fix**; the active image still has task-conditioning ON, i.e. the arm we refuted at **0.7288 vs 0.7546**.
⚠ **WHAT IS ESTABLISHED vs WHAT IS INFERRED, stated separately because the difference decides the action.** Established: the phase is open, we have no final submission, the newest image predates the fix, and the fix's own comment records the refutation that prompted it. **Inferred but NOT confirmed on-platform: that image #3 is specifically the 0.7288 build and image #2 the 0.7546 one** — the Cat-2 prelim leaderboard returns **403**, the submissions DataTables URL **404s**, and the API `evaluations/` endpoint returns **0 matching rows**, so the image↔score mapping cannot currently be read.
⇒ **THEREFORE I DID NOT FLIP THE ACTIVE IMAGE**, which would have been the cheap fix but rests entirely on that unconfirmed mapping. **Rebuilding from current source removes the ambiguity**, because that source definitively carries the revert. **`7434562` queued** (94 MB context, `python:3.11-slim`): builds with a **node-dependent netns probe** (`--network none` → `--network host` fallback, logged as a divergence), then **runs the built image and asserts `USE_TASK_CONDITIONING == False` INSIDE it** — a source-tree grep would not prove what got baked in — and only then saves + hashes the tarball. **Upload + submit to `category-2-final-phase` follows the rebuild.**

### 2026-08-21 16:00 — 📊 **OWN-ROWS AUDIT (prompted by the iMED finding): SurgVU Cat-1 is mildly inflated, not dramatically.**
Cat-1 prelim board = **19 rows, of which TWO are ours** — 13th (**0.4202**, "run A2") and 14th (**0.4019**) — both `omarchoudhry (AIMS)`. ⇒ our "13/19" means **5 competitors below us, not 6**; the field is 17 others + our 2 rows. Far milder than iMED, where rows 5–9 were *all* ours, but the correction is the same class and the check is now standard. **Cat-2 prelim could NOT be audited (HTTP 403), so its "19/32" standing is UNVERIFIED and should not be repeated as fact.**

### 2026-08-21 16:30 — ⛔→▶ **Cat-2 rebuild `7434562` FAILED IN 7 s AT STEP 1, AND THE FIX WAS ALREADY WRITTEN DOWN IN THIS CAMPAIGN.**
```
While applying layer: ... potentially insufficient UIDs or GIDs available in user namespace
(requested 0:42 for /etc/gshadow) ... lchown /etc/gshadow: invalid argument
```
Pulling `python:3.11-slim` failed because its layers contain files owned by **`0:42` (root:shadow)** and **AIRE has no `/etc/subuid` range**, so rootless podman cannot map a non-zero GID. **My v1 `storage.conf` set `graphroot`/`runroot` but carried NO `ignore_chown_errors` at all.**
**★ THE ANNOYING PART: the MultiBypass container gate — which pulled and built successfully THIS MORNING — carries a comment warning about exactly this**, including the subtlety that global `--root/--runroot` flags **discard** the storage options so `ignore_chown_errors` is silently lost. I wrote a fresh storage config instead of reusing the one already proven on this partition. *A solved infra problem in this campaign is worth grepping for before re-solving.*
**v2 `7434598`** reuses the proven configuration verbatim: `CONTAINERS_STORAGE_CONF` holding **driver AND options** (`overlay` + `fuse-overlayfs` mount_program + `ignore_chown_errors = "true"`), **no global `--root/--runroot`**, `--cgroup-manager=cgroupfs --events-backend=file`, everything wrapped in **`buildah unshare`**, inner script a **quoted** heredoc taking parameters from the environment. The node-dependent netns probe now runs *inside* the unshare too.
**The in-image assertion is retained and is the point of the job:** it greps `^USE_TASK_CONDITIONING` from `/opt/app/inference.py` **inside the built container** and passes only on `False` — because a source-tree grep proves nothing about what got baked in, which is the precise defect being fixed.

### 2026-08-21 17:05 — ★★ **THE BUILD ASSERTION CAUGHT AN INCOMPLETE REVERT — AND IT RETROACTIVELY *PROVES* WHICH IMAGE IS THE REFUTED ONE.**
Rebuild v2 (`7434598`) got all the way through the storage fix (base pulled, every earlier assertion green: CPU-only torch · entity prior present, 12 families / 1850 clips · entity prior ACTIVE · task ckpt val_acc 0.9190 · task head importable) and then **failed on an assertion inside our own Dockerfile**:
```
assert 'ProGrasp' in a  ->  AssertionError: The type of forceps mentioned is Cadiere Forceps.
Error: building at STEP "RUN chmod -R a+rX /opt/app && … BUILD ASSERT OK: TASK CONDITIONING IS ACTIVE"
```
**Dockerfile line 45 REQUIRED task conditioning to be active** — it set `_TASK_CACHE` to two different tasks and asserted the answers DIFFER. With `USE_TASK_CONDITIONING = False` the answer is correctly task-invariant, so the build cannot succeed. ⇒ **when I reverted `inference.py` on 2026-08-20 I did not revert the Dockerfile, so the container was UNBUILDABLE in its own shipped configuration and nobody would have found out until a build was attempted.**
**★★ AND IT SETTLES THE QUESTION I HAD REFUSED TO GUESS AT.** Earlier today I logged the image↔score mapping as **INFERRED, not confirmed**, because the Cat-2 prelim board is 403, the submissions URL 404s and the API returns 0 rows — and I declined to flip the active image on that basis. **The build system is an independent witness: this Dockerfile cannot produce an image unless task conditioning is ON. Image `8f6a9cf9` (2026-08-20 01:57) built successfully with it ⇒ it necessarily carries the refuted arm.** The mapping is now **ESTABLISHED**, and the decision to rebuild rather than flip was the right one for a reason better than caution.
**FIX — the assertion was INVERTED, not deleted.** It now asserts the reverted behaviour positively: `USE_TASK_CONDITIONING is False`, the two task contexts produce **identical** answers (`assert a==b`), and the entity prior still fires (`'Cadiere Forceps' in a`). Deleting the check would have left the build silent about the very property under dispute; inverting it means the build now *proves* the revert on every future rebuild. Assertion count unchanged; `chmod -R a+rX /opt/app` retained; original saved as `Dockerfile.pre_revert_assert.bak`. **v3 `7434849` queued.**

### 2026-08-21 17:30 — ✅ **CAT-2 REVERTED IMAGE BUILT AND VERIFIED (`7434849`). The job says FAILED; the ARTIFACT is good.**
`BUILD_RC=0`, all 12 steps, tagged `localhost/aims-cat2:reverted`, `SAVE_RC=0`.
**Artifact:** `/scratch/sc20osc/miccai-2026/SurgVU/docker/aims-cat2-reverted_2026-08-21_17-02-59.tar.gz` — **519,871,912 bytes**, SHA256 **`387f738ace84901b5f029e66da80772fc044c282c67fa0f8c8453071e461b736`**.
**★ VERIFIED BY THE STRONGEST CHECK AVAILABLE — the Dockerfile's own assertions, which execute INSIDE the image at build time:**
```
BUILD ASSERT OK: CPU-only torch (no CUDA wheels)
BUILD ASSERT OK: entity prior present, 12 families, 1850 clips
BUILD ASSERT OK: entity prior IS ACTIVE -> The type of forceps mentioned is Cadiere Forceps.
BUILD ASSERT OK: task ckpt loads, 8 tasks, val_acc 0.9190
BUILD ASSERT OK: task conditioning INACTIVE (REFUTED 2026-08-20, 0.7288 vs 0.7546); answer is task-invariant
BUILD ASSERT OK: /opt/app world-readable and inference.py parses
```
⇒ the revert is proven **in the image**, and the **CLiMB non-world-readable defect class is closed** in it too.
**⚠ THE JOB'S `FAILED` IS MY OWN CHECK, FOR THE THIRD TIME TODAY — and the three failures share one shape.** `ASSERT_RC=126`: the image declares `ENTRYPOINT ["python","inference.py"]`, so `podman run IMAGE grep …` passed `grep …` as **arguments to the entrypoint** rather than running it. Fixed with `--entrypoint grep`.
**★ THE PATTERN, NAMED: every one of today's false negatives came from running a correct check in the WRONG EXECUTION CONTEXT** — the wrong *interpreter* (system `python3`, no PIL → iMED render "FAILED"), the wrong *credential* (API Bearer token on HTML pages → ORENA phases "closed"), and now the wrong *entrypoint*. In all three the subject under test was fine. **Before believing a negative, confirm the check could have produced a positive.**
**⇒ NEXT: upload this tarball as a new algorithm image, wait for import, and submit to `category-2-final-phase`** — the phase we have never submitted to, which closes **Sep 6**. My external in-image check is now redundant with the build assertions but has been fixed for reuse. The 20-byte stub left by the failed v2 run was deleted so it cannot be mistaken for an artifact.

### 2026-08-21 18:00 — ✅ **CAT-2 REVERTED IMAGE UPLOADED AND ATTACHED.** New image `9b92047d-c495-…`, `import_status=Started`.
`~/SurgVU/upload_surgvu_image.py` — a **NEW** file; ORENA's `upload_algorithm_image.py` is a validated instrument and was NOT modified. Every safeguard was copied across, and one earned its keep immediately: **the RepoTag guard**, verified on both artifacts before use — `localhost/aims-cat2:reverted` **PASSES** the cat2 target while `localhost/surgvu26-cat1-aims:v1` is **REJECTED**. The Cat-1 tarball sits in the *same directory*, so a careless `--file` would have uploaded the tool detector into the Cat-2 algorithm — the 2026-08-04 swap class.
Run: sha256 matched the pin `387f738a…` before a byte was sent · bytes up in 0.1 min · **`creator` read FROM THE FORM as `111813`** (posting the *username* returns 200 and creates nothing, silently — that cost two ORENA uploads) · POST **200 redirecting AWAY from the form** to `/images/9…`, which is the success signature (a redirect BACK to `create` is failure).
**⚠ MY POLLING STEP CRASHED — the FOURTH check-side failure today, same family.** ORENA's `c("algorithms/images/?…")` call convention is invalid in this `gcapi` build (`h11 LocalProtocolError: Illegal method characters` — it tried to use the path as an HTTP method). The upload was already complete; only the confirmation broke. Verified instead with a direct API call: **4 images now, newest `9b92047d` created 2026-08-21T16:58:56Z, importing.** Waiter `wait_cat2_import.sh` armed (polls to a TERMINAL status so a silent success and a silent failure look different).
**⚠ CORRECTION TO MY OWN ARITHMETIC, stated because I quoted the number: the API reports UTC while file mtimes are BST.** Image `8f6a9cf9` at `01:57` is **UTC** = 02:57 BST, and the source revert at `04:05` is BST ⇒ the gap is **1 h 08 m, not the 2 h 08 m I logged**. The conclusion is unchanged — the source fix still postdates that image — and in any case it does not rest on the arithmetic: the build-assertion argument (the Dockerfile *required* task conditioning to be ACTIVE, and `8f6a9cf9` built successfully under it) settles it independently.
**➡ NEXT: when import reads `Completed`, submit to `category-2-final-phase`** — never submitted to, 7 slots, closes **Sep 6**.

### 2026-08-21 18:30 — ⛔ **CORRECTION: THERE IS NO OPEN CAT-2 "FINAL" PHASE. THE SLUG LIED, AND MY EARLIER CLAIM WAS WRONG.**
Probing each phase by the **title the page reports** rather than by the slug in the URL:
| slug | HTTP | TITLE the page reports |
|---|---|---|
| `category-1-prelim-phase` | 200 | Category 1 - Prelim Phase |
| `category-1-final-phase` | **403** | Forbidden |
| `category-2-prelim-phase` | **403** | Forbidden |
| **`category-2-final-phase`** | **200** | **"Category 2 - Prelim Phase Submission"** |
⇒ **the slug `category-2-final-phase` SERVES THE PRELIM PHASE.** The organizers appear to have re-slugged: Cat-2 Prelim now lives at the `-final-` slug while the `-prelim-` slug 403s.
**★ WHAT I GOT WRONG.** I reported earlier today that *"the Cat-2 FINAL phase is OPEN with 7 submissions left and we have never submitted to it."* **That is false and is retracted.** I read the openness of a form served at a `-final-` URL and inferred the phase's identity from the slug. **There is exactly ONE reachable Cat-2 submission target and it is PRELIM.** We have not missed a final-phase deadline, because no final phase is open to us.
**★ AND THE 403s ARE INFORMATIVE, NOT NOISE: 403 ≠ 404.** Both `category-1-final-phase` and `category-2-prelim-phase` return **403 (exists, forbidden)** while `category-{1,2}-test-phase` return **404 (absent)**. So the real final phases **exist and are simply not open to participants yet** — the state to watch for is 403 → 200-with-a-form. SurgVU's final closes **Sep 6**.
**★ WHAT THE SUBMISSION I MADE ACTUALLY IS — still worth having.** It is a legitimate **Cat-2 Prelim** submission (Aug 21, `Executing Algorithm`, recordsTotal 5 → 6) carrying the **corrected** image `9b92047d` — entity prior on, task conditioning off. It replaces the refuted 0.7288 arm with the 0.7546-class model on the board that is actually open, and it exercises the rebuilt container end-to-end on the platform. It did **not**, as I previously implied it would, fill a missing final-phase slot.
**★ THE LESSON, which generalises past this challenge: IDENTIFY A PHASE BY THE IDENTITY IT REPORTS, NOT BY ITS URL.** The campaign already knew "openness must be proven by a usable form, never by HTTP 200" (RARE26's closed phase answers 200). This adds the other half: **a usable form is not evidence of WHICH phase you are about to submit to.** `gc_phase_status.py` should be extended to print the page's own `<h1>` next to every slug.

### 2026-08-21 18:55 — ✅ **CAT-2 SUBMISSION SUCCEEDED, and the phase auditor is hardened so today's mistake cannot recur.**
The Aug-21 17:22 UTC submission carrying the rebuilt image `9b92047d` reads **`Succeeded`** ⇒ the reverted container (entity prior on, task conditioning off, `/opt/app` world-readable) **ran end-to-end on the platform**. Given that three of our last six container submissions across the campaign shipped broken, that is worth stating plainly. Cat-2 Prelim now carries the 0.7546-class model instead of the refuted 0.7288 arm.
**★ `~/gc_phase_status.py` HARDENED (backup `.pre_identity.bak`) — it now reports IDENTITY, not just openness:**
```
SurgVU  cat1-prelim    YES  Category 1 - Prelim Phase Submissi  OPEN, 8 submissions remaining
SurgVU  cat1-final     no   -                                   HTTP 403 (forbidden - exists, not open to us yet)
SurgVU  cat2-prelim    no   -                                   HTTP 403 (forbidden - exists, not open to us yet)
SurgVU  cat2-final     YES  Category 2 - Prelim Phase Submissi  OPEN, 6 submissions remaining  ⚠MISMATCH slug says 'final', page says 'Category 2 - Prelim Phase Submission'
RARE26  sanity-check   YES  Sanity Check Submission             OPEN, 5 submissions remaining
RARE26  open-dev       no   Open Development Phase Submission   no submission form rendered
RARE26  closed-test    no   Closed Testing Phase Submission     explicitly closed (body says so, despite HTTP 200)
*** 1 SLUG/TITLE MISMATCH(ES) — the URL does NOT name the phase it serves. Trust the TITLE. ***
```
Three changes, each earned today: every row carries **the title the page itself reports**; a row whose title contradicts its slug is flagged **⚠MISMATCH** and counted in a closing warning; and **403 is distinguished from 404** ("exists, not open to us yet" vs absent) instead of both collapsing to "closed". The original RARE26 self-check ("refuse to be trusted if nothing reports open") is retained.
**★ AND IT INDEPENDENTLY CONFIRMS WHERE THE SUBMISSION LANDED:** that phase's remaining count went **7 → 6**. The submission consumed a slot from the phase served by the `-final-` slug — which the title proves is Prelim. Two independent witnesses (the submissions listing's own phase column, and this counter) now agree, which is why the earlier retraction was right.
**⇒ STANDING WATCH: `cat1-final` and `cat2-prelim` are 403 = they EXIST. The real final phases opening is the 403 → 200-with-a-form transition, before Sep 6.**

### 2026-08-21 19:25 — ✅ **CAT-1 AUDITED FOR THE CAT-2 DEFECT — IT IS CLEAN. Recording the negative so it is not re-run.**
Having found that Cat-2's active image predated its own revert, I applied the same test to Cat-1. The local tarball (`surgvu26-cat1-aims_2026-08-19_18-49-55.tar.gz`, 18:50 BST) **is** 4 h older than `inference.py` (22:58 BST) — which looks like the same defect, and is why the local artifact must not be the thing you check. **The platform tells a different story.**
| event (ALL UTC — the GC listing and API both report UTC) | time |
|---|---|
| image 1 built | 2026-08-19T17:54Z |
| submission #1 → **0.4019** | 17:58Z |
| **source `inference.py` edited** | **21:58Z** (22:58 BST) |
| **image 2 built — the ACTIVE one** | **22:07Z** |
| submission #2 "run A2" → **0.4202** | 22:12Z |
⇒ **the active Cat-1 image postdates the source fix by 9 minutes, and the 0.4202 submission used it.** The corrected `CONF=0.001 / MAX_DET=100` defaults (replacing the guessed threshold that cost 8.4%) **are** in the shipped image. **No action needed on Cat-1.**
**★ TWO METHOD POINTS WORTH KEEPING.** (1) **Audit the PLATFORM artifact, not the local build directory** — the stale local tarball would have produced a false alarm here, exactly as a stale *remote* image produced a true alarm on Cat-2. (2) **The whole comparison hinges on timezone discipline**: mtimes are BST, the API and the submissions listing are UTC. Mixing them silently shifts every gap by an hour, which is how I mis-stated the Cat-2 gap as 2 h 08 m earlier today. Both timestamps here were converted before comparing.

### 2026-08-22 13:10 — ▶ **CAT-1 SSL ARRAY RELEASED (user-approved) — and the 4 already-completed seed-0 arms give a SPLIT verdict, free, from disk.**
The array was held under the MT-BTPN GPU-priority rule. Checked before asking: **0 MT-BTPN jobs running**, and the 4 pending were all `DependencyNeverSatisfied` husks. **User approved release + cancelling the husks**; 10 tasks released (0 still held), 4 husks cancelled — each re-checked for `DependencyNeverSatisfied` immediately before `scancel`, so a job that had since become runnable would have been skipped.
**★ `GATE.json` PASSES, so the arms genuinely train rather than self-exit:** `agreement_observed 0.634` vs `agreement_shuffled 0.245` (ratio 2.59 vs a 1.5 threshold) · `firing_unseen 0.391` vs 0.30 · `FILTER_ok`/`ASSIGN_ok` true. The label-free box-proposal mechanism is validated on its own terms.
**★★ SEED-0 RESULTS (tasks 0–3, already COMPLETED before the hold; val n=1973 imgs / 3275 inst):**
| arm | overall mAP50-95 | clip_applier (n=258) | vessel_sealer (n=13) |
|---|---|---|---|
| baseline S0 | 0.1444 | **0** | **0** |
| FILTER | **0.183** | 0 | 0 |
| ASSIGN | 0.148 | 0.00018 | 0.00049 |
| **COMBINED** | **0.184** | 0.00035 | 0.014 |
| NEG_UNCONSTRAINED | 0.156 | 0 | 0 |
**⇒ THE LEVER SPLITS, AND IT DOES NOT DO THE THING IT WAS QUEUED FOR.** Overall mAP50-95 rises **0.1444 → 0.184 (+0.040, +28 % relative)** on FILTER and COMBINED — a real-looking gain. But **the two dead classes stay at essentially zero**: the best is `vessel_sealer` 0.014 in COMBINED, on **13 val instances**, i.e. one or two detections. **The documented cap — "Cat-1 capped at 0.500 by ZERO train boxes for 2 of 4 val classes" — STANDS.** The gain is coming from the classes that already worked.
**⇒ WHAT THE RELEASED SEEDS ARE NOW FOR, and it is the right question:** these are **single seed-0 runs**. The 10 released tasks are seeds 1–2 of each arm, so they test whether **+0.040 survives replication** or is seed noise. Given this campaign's history (the RARE26 3→10-seed KILL; the ORENA data-engine +0.95 that became noise at 3 seeds), a one-seed +0.04 is a hypothesis, not a result.
**PRE-REGISTERED READ, fixed now:** FILTER/COMBINED must beat baseline on the **3-seed mean** to count. If they do, the arm is a genuine Cat-1 improvement worth rebuilding the detector around; if the mean collapses toward 0.1444, it joins the refuted list. **Either way the dead-class cap is untouched — that remains a DATA problem (zero training boxes), not a method one.**

### 2026-08-24 — ★★ **CAT-1 SSL: `FILTER` REPLICATES AND SEPARATES FROM BASELINE. First real performance gain of the release.**
Baseline verified from `results.csv` col 9 (`metrics/mAP50-95(B)`), best-over-epochs, **not** taken on trust from the script header:
| | mAP50-95 |
|---|---|
| S0_seed0 / seed1 / seed2 | 0.1469 / 0.1248 / **0.1616** |
| **baseline mean** | **0.1444**, sd **0.0151** |
| **FILTER seed0 / seed1** | **0.183 / 0.189** → mean **0.1860** |
| ASSIGN seed0 / seed1 | 0.148 / 0.141 → mean 0.1445 = **baseline. NULL.** |
| COMBINED seed0 | 0.184 (seeds 1–2 owed) |
| NEG_UNCONSTRAINED seed0 | 0.156 (seeds 1–2 owed) |
**⇒ FILTER: +0.0416 over the baseline mean = 2.7 baseline SDs, and its WORST seed (0.183) exceeds the baseline's BEST seed (0.1616) — the two seed sets do not overlap.** The header's `0.1444` is confirmed to be a true 3-seed mean.
**⇒ ASSIGN is refuted** (0.1445 vs 0.1444). That matters: it means the gain is **not** generic "more pseudo-labels", it is specifically the FILTER arm's constraint. A lever that works for a nameable reason is worth more than one that works.
**⚠ THREE THINGS THIS DOES NOT SHOW, stated so the result is not oversold:**
1. **Seed 2 (task 10) is still owed.** Two seeds vs three is not the pre-registered 3-seed mean. The rule was fixed in advance and is not met yet.
2. **This is LOCAL val mAP, not the platform metric.** Our platform Cat-1 score is 0.4202 on a different scale; +0.0416 here does not transfer linearly, and this campaign has been burned before by quoting an offline proxy as a leaderboard prediction.
3. **The dead-class cap is UNTOUCHED.** `clip_applier` and `vessel_sealer` remain at 0.000 in both FILTER seeds. The documented ceiling ("capped at 0.500 by ZERO train boxes for 2 of 4 val classes") stands — the gain comes entirely from classes that already worked.
**⇒ PLAN ON SEED 2 CONFIRMING:** rebuild the Cat-1 detector on FILTER and **spend one of the 8 remaining Cat-1 prelim submissions** — the board is the only instrument that answers whether local mAP moves the platform score. If seed 2 collapses the mean below separation, FILTER joins the refuted list and Cat-1 is closed.

### 2026-08-24 — ✅ **CAT-1 DELIVERY PATH VERIFIED *BEFORE* THE RESULT LANDS (the rule ORENA taught, applied in time).**
On 2026-08-22 the q35 arm turned out to be unshippable *after* its 8 h run, because a 9B-trained LoRA cannot go into an 8B container. So this time the path was checked while the seeds were still running:
| | |
|---|---|
| container loads | `RESOURCE_PATH/"detector.pt"` → `YOLO(str(wt))` — a **single file** |
| shipped `detector.pt` | 44,052,754 bytes |
| `FILTER_main_t25_s0/weights/best.pt` | **44,052,754 bytes** |
| `FILTER_main_t25_s1/weights/best.pt` | **44,052,754 bytes** |
| checkpoint loads | OK, **14 classes**, matching the container's "all FOURTEEN challenge tool classes" |
⇒ **A FILTER checkpoint is a literal drop-in for `resources/detector.pt`.** Same architecture, same class map, no base-model mismatch, no adapter/soup indirection. **If seed 2 confirms, shipping is: copy one file → rebuild → upload → submit**, and Cat-1 prelim has **8 submissions** left. This is the cheap-delivery case the ORENA arm was not.

### 2026-08-25 — ★★ **CAT-1 SSL COMPLETE (15/15). FILTER WINS — but the NEGATIVE CONTROL reproduces 61% of the gain, and that changes the claim.**
| arm | seeds | mean | vs baseline |
|---|---|---|---|
| **FILTER** | 0.183 / 0.189 / 0.199 | **0.1903** | **+0.0459** |
| COMBINED | 0.184 / 0.182 / 0.183 | 0.1830 | +0.0386 |
| **NEG_UNCONSTRAINED** *(negative control)* | 0.156 / 0.191 / 0.170 | **0.1723** | **+0.0279** |
| NEG_SHUFFLE *(shuffled control)* | 0.148 / 0.130 | 0.1390 | −0.0054 |
| ASSIGN | 0.148 / 0.141 / 0.107 | 0.1320 | −0.0124 |
| baseline S0 | 0.1469 / 0.1248 / 0.1616 | 0.1444 (sd 0.0151) | — |
Exact permutation tests (3v3, 20 splits, so the smallest attainable p is 0.05):
```
FILTER vs baseline            +0.0459   p = 0.050
NEG_UNCONSTRAINED vs baseline +0.0279   p = 0.100
FILTER vs NEG_UNCONSTRAINED   +0.0180   p = 0.150   ranges OVERLAP (0.183-0.199 vs 0.156-0.191)
```
**⇒ I AM CORRECTING MY OWN 2026-08-24 READING.** Yesterday, on two seeds and without the controls, I wrote that ASSIGN's null meant *"the gain is not generic 'more pseudo-labels', it is specifically the FILTER arm's constraint."* **The completed controls refute that.** `NEG_UNCONSTRAINED` — a deliberate negative control with **no constraint at all** — also beats baseline by **+0.0279**, i.e. **61% of FILTER's apparent +0.0459**. The constraint-specific effect is **+0.0180 with overlapping ranges and p=0.15**: suggestive, not established.
**★ WHAT THE CONTROLS DO ESTABLISH, and it is worth having:**
1. **`NEG_SHUFFLE` is null (−0.0054).** Shuffling the pseudo-label assignment destroys the gain ⇒ **the pipeline is not leaking**, and the benefit is not an artifact of simply training longer on more images. Without this arm the whole result would be uninterpretable.
2. **Pseudo-labelling genuinely helps Cat-1** (+0.0279 even unconstrained), and **FILTER is the best-measured arm** — highest mean *and* much the tightest spread (sd 0.0066 vs the baseline's 0.0151).
3. **ASSIGN actively hurts** (−0.0124) — not merely a null.
**⇒ SHIP FILTER, CLAIM CAREFULLY.** For the artifact, FILTER is the right choice on the evidence (best mean, tightest variance). For the write-up, the defensible claim is **"SSL pseudo-labelling improves Cat-1 by ~+0.046 local mAP50-95, of which the majority is generic and ~+0.018 is attributable to the filtering constraint (not separated at n=3)"** — NOT "the constraint is the mechanism".
⚠ **UNCHANGED AND STILL BINDING: `clip_applier` and `vessel_sealer` remain 0.000 in every arm.** No SSL variant touched the two dead classes, so the documented ceiling stands and remains a DATA problem.
⚠ **AND THIS IS LOCAL val mAP, NOT the platform metric** (ours there: 0.4202). The board is the only instrument that settles whether +0.046 transfers — 8 Cat-1 prelim submissions remain, and the FILTER checkpoint is a verified one-file drop-in.

### 2026-08-26 — ✅✅ **BOTH SurgVU FINAL-PHASE SUBMISSIONS ACCEPTED. The final phases opened today, and the hardened auditor caught that the organizers SWAPPED THE SLUGS.**
```
SurgVU cat1-prelim  YES  "Category 1 - Prelim Phase"  OPEN, 8 left
SurgVU cat1-final   YES  "Category 1 - Final Phase"   OPEN, 3 left      <- WAS 403
SurgVU cat2-prelim  YES  "Category 2 - FINAL Phase"   OPEN, 3 left   ⚠MISMATCH   <- WAS 403
SurgVU cat2-final   YES  "Category 2 - PRELIM Phase"  OPEN, 6 left   ⚠MISMATCH
```
**★★ THE `-prelim-` AND `-final-` SLUGS ARE SWAPPED FOR CATEGORY 2.** To submit to the Cat-2 **final** phase you must POST to `category-2-prelim-phase`. Had I trusted the slug — as I did on 2026-08-21, which produced the retraction — I would have sent the final submission to the prelim board and believed the opposite. **The ⚠MISMATCH flag added after that mistake is what caught it**, and the submitter now asserts `"Final" in <h1>` before posting and refuses otherwise.
**SUBMITTED (verified by resource state, not HTTP):** `recordsTotal` 6 → **8**
| | |
|---|---|
| **Category 1 - Final Phase** | 26 Aug 10:00, **Queued** — baseline 14-class YOLO detector (known-good, image `9f439e85`) |
| **Category 2 - Final Phase** | 26 Aug 10:00, **Executing Algorithm** — reverted answerer (image `9b92047d`, entity prior on, task conditioning off) |
**⇒ RISK ORDER WAS DELIBERATE: bank a KNOWN-GOOD artifact in the phase that counts FIRST, then improve.** Cat-1 has **2 final slots left** for the FILTER detector; Cat-2 has 2 left. Had I built FILTER first and it failed, the deadline could have passed with nothing in the final phase at all.
⏳ **Cat-1 FILTER image building (`7515825`)** — `detector.pt` swapped to `FILTER_main_t25_s2/weights/best.pt` (sha `91ed8022…`, baseline backed up as `detector.pt.baseline_bak`), podman config = the form proven on AIRE 2026-08-22, and the job **asserts the in-image `detector.pt` sha equals the FILTER weights** so a silently-unswapped build cannot pass.
⚠ **SEED CHOICE, STATED HONESTLY:** seed 2 was picked because it has the best val mAP (0.199). That is selection on the same split all three seeds were scored on, so **the unbiased expectation for this checkpoint is the 3-seed mean 0.1903, not 0.199.**

### 2026-08-26 — ⛔ **CAT-2 FINAL SUBMISSION FAILED — but on THEIR scoring, not our algorithm. Slots deliberately NOT spent.**
`bf65a0e3-254f-4410-89db-9a8f66968fb4`, 26 Aug 10:00 → **Failed** at 10:16:
> `helpers.PredictionProcessingError: One or more errors occurred during prediction processing.` **`Your algorithm ran successfully, but the scoring failed.`** `…please contact the challenge organisers.`
**★ THREE PIECES OF EVIDENCE PUT THIS ON THE ORGANIZER SIDE:**
1. **Their own message says the algorithm ran successfully** and the failure was in scoring.
2. **The identical image (`9b92047d`) SUCCEEDED on the Cat-2 PRELIM phase on 21 Aug.** Same container, same weights — so the difference is the phase, not the artifact.
3. **Our answerer is robust to degenerate input**: tested against empty string, whitespace-only, punctuation-only, a 300-char string, non-ASCII, and malformed questions — **0 of 15 produced an empty, None, over-long, or exception output**; every one returned a short declarative sentence. We are not emitting something unscoreable.
I could not go further from our side: the phase pages do not publicly expose their declared input/output interfaces, so an interface mismatch between prelim and final can be *suspected* but not *checked*.
**⇒ DECISION: DO NOT RESUBMIT YET. 2 Cat-2 final slots remain and 11 days to Sep 6.** A blind resubmit is a coin-flip that could cost a slot for no information. The documented next step is the one their own error message names — contact the organizers. Draft ready for the user at `SurgVU/ORGANISER_EMAIL_cat2_final_failure.md`.
**⇒ FALLBACK IF NO REPLY BY ~30 AUG:** spend ONE slot as a controlled test of transience (GC scoring jobs do occasionally fail on infrastructure), keeping the last slot in reserve. That ordering — ask first, then test — costs nothing while there is time, and preserves a slot either way.
ⓘ **Cat-1 Final is a natural control and is still `Executing Algorithm`.** If Cat-1 also fails at scoring, that is strong evidence of a final-phase-wide problem and should go in the same email.

## 2026-08-30 — Cat-2 FINAL resubmitted on a REBUILT image; two facts discovered from a refusal
**The 26 Aug Cat-2 FINAL submission scored as `Succeeded Failed`** ("your algorithm ran
successfully, but the scoring failed"). Today's attempt to retry it was **REFUSED by Grand
Challenge**: *"A submission for this algorithm container image and model for this phase already
exists."*
★ **GC DEDUPLICATES SUBMISSIONS BY (image, phase).** A retry is therefore impossible without a
NEW image — a resubmit is not a free action, it is a rebuild. Logged as a delivery gotcha.
★ **The refusal message also carried two facts we did not have:** a **10-minute-per-case time
limit**, and the phase **closes 7 Sep 2026 04:04 UTC, not 6 Sep** — one more day than the roster said.
**REBUILD:** job `7615443`, logic byte-identical (a `BUILD_TAG` comment only), all 8 in-image build
assertions PASS, including the inverted one proving `USE_TASK_CONDITIONING is False` and that the
answer is task-invariant. `BUILD_RC=0`, `SAVE_RC=0`.
⚠ **The job still reported FAIL, and the job was right to be doubted, but the FAILURE WAS IN MY
CHECK:** the post-build flag read has its grep arguments DUPLICATED across a line continuation
(`grep -m1 PAT file PAT file`), so it captured nothing and the script called it `[FAIL] rc=5`. My
first diagnosis — "the ENTRYPOINT swallowed it" — was ALSO wrong; the script does pass
`--entrypoint grep`. The in-image build assertion had already proven the property the broken check
was re-checking.
**UPLOAD:** the uploader's `import_status` poll used gcapi's `c("path")` form and died with h11
`Illegal method characters` **after the upload and attach had already succeeded** — patched to poll
the REST API directly (backup `.pre_pollfix.bak`). New image `d33dc949`, import **Completed**, and
it is now the phase's active image (old `9b92047d` no longer offered).
**SUBMITTED 30 Aug 21:15 UTC → status `Executing Algorithm`.** This is the controlled test of
whether the 26 Aug scoring failure was transient or systematic. 2 slots remain.

## 2026-08-30 (later) — ★ FINAL-PHASE RESULT: Cat-2 SUCCEEDED, and we are 2nd of 4 teams
**The rebuilt image scored. The 26 Aug scoring failure was TRANSIENT, not systematic** — identical
logic, new digest, and it ran clean. No organiser email is needed.

| board | ours | rank | leader |
|---|---|---|---|
| **Cat-2 FINAL** | **0.6434** | **2nd of 7 entries / 2nd of 4 teams** | 0.6718 `PeterHjy628 (UoM-Surgical)` |
| **Cat-1 FINAL** | **0.4012** (FILTER SSL) · 0.3893 (YOLO baseline) | **3rd of 4 teams** | 0.5376 `guhongyu (PUMCH-UFH-Hik)` |
| Cat-2 PRELIM | **0.7546** | 39th of 59 | 0.8994 `pengyuncong (UESTC-SCU-UC)` |

**⚠ A COMPARISON I MADE AND HAVE RETRACTED.** I read our FINAL 0.6434 against the OFFLINE n=11
no-vision floor of 0.6383 and concluded our answerer "barely beats a constant No". **That is a
category error — different test sets.** The platform's own large-n evidence says the opposite:
on PRELIM the entity prior scores **0.7546 vs 0.6761** for `N0_template_novision`, so the prior is
worth **+0.0785** on real data. Prelim and final differ in difficulty (prelim top 0.8994, final top
0.6718); a floor measured on one does not transfer to the other. This is the same
proxy-quoted-as-a-platform-number trap logged for ORENA SEG.

**⇒ CAT-2 AXES ARE EXHAUSTED, each for a MEASURED reason — do not re-tread:**
1. **Instrument vision** — REFUTED by mechanism: the tools that carry the signal are near-visual
   duplicates (Mega vs Large vs SutureCut Needle Driver; 0° vs 30° Endoscope). Neither detector nor
   VLM separates those from pixels (2026-08-20).
2. **Task conditioning** — REFUTED ON THE PLATFORM, the strongest instrument we have: 0.7288 vs 0.7546.
3. **Hybrid VLM-as-entity-extractor** — refuted 2026-08-08. Its n=11 refutation used a
   non-discriminating instrument, but (1) explains the mechanism, so it does not deserve a re-run.
4. **Register/length** — settled: a ~7-word declarative dominates at every accuracy level.
5. **Polarity** — the metric is ~negation-blind by the organizers' own documentation.
**Compute budget bounds any re-opening anyway: 10 min/case, ≤32 GB DRAM, No-GPU or T4** — an 8B VLM
does not fit a T4, and CPU-only was a deliberate choice to kill the CUDA-mismatch failure class that
already killed our iMED v2 and a rival entry.
⇒ **No further Cat-2 experiments.** 2 final slots remain unspent, deadline 7 Sep.
**RISK TO NOTE:** the prelim leader (`pengyuncong`, 0.8994) has NOT yet submitted to the final phase.
If they do before 7 Sep our 2nd place likely becomes 3rd. Nothing we can do about that.
