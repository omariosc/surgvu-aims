# SurgVU26 Category-2 (Surgical VQA) — Deep-Research Briefing (user-supplied 2026-06-20)

## Official task (Cat-2 = video-conditioned open-ended surgical VQA)
- Input: **30-second clip**, **open-ended NL question**. Output: **one answer string per question, JSON** (one JSON/video).
- Metric: **mean over questions of (max BLEU over the 5 reference answers)**. → lexical normalisation, concise/canonical answers, synonym handling are FIRST-CLASS design choices.
- Test data **sub-sampled to 1 FPS** (~30 frames/clip); **UI region blurred**; hidden test on Grand Challenge (Type-2 container).
- Corpus: SurgVU 280 vids/155 sessions, 840h, 18M frames, 60fps/720p; weak labels = noisy **tool presence**, **8 steps**, `matched_description`; public VQA sample = **10–11 clips** (sources disagree; some public answers wrong).
- BLEU implementation details (n-gram order, tokeniser, smoothing) NOT officially exposed → verify vs live docs/organiser code.

## 2025 results — the central lesson
- **Winner Capybara = BLEU 0.4215**: NOT heavy fine-tuning. = LLaVA-OneVision-7B **zero-shot** + **tool/organ detection** + crop UI/black-margins + generated description + **only 5 sampled frames** + concise prompted answers. Only **21 unique captions** in `matched_description` → direct FT is brittle.
- **UoM-SurgicalAI**: zero-shot InternVL3/3.5 **generalised better than PEFT** despite PEFT winning local val (local dev from same template engine is optimistic).
- **AMI**: single-answer-per-Q-type → model ignores vision, memorises priors; balanced contrasting answers help.
- **UT**: 70k rule-QA + LoRA InternVL3-2B; longer training degraded (overfit to templates).
- **Medibot**: dedicated detector as "authoritative" tool-presence source (VLM hallucinated); Qwen2.5-VL-3B under 16GB.
- **Capybara**: 4-bit VLM outputs differ across Turing/Ampere/Hopper; eval platform = **Tesla T4** → test on T4-like HW.

## Gaps / failure modes (priority)
1. Weak/low-diversity supervision (21 captions) → don't FT on `matched_description`.
2. Metric mismatch: BLEU rewards concise canonical phrasing > verbose-correct.
3. Shortcut learning / answer priors (text-only ablation strong; SurgCheck).
4. Insufficient temporal retrieval (teams used 5/8/21 frames, not learned selection).
5. Visual grounding failures / hallucination (detector-as-authority fixes it).
6. Synthetic-QA quality (counterfactual/paraphrase diversity > raw count).
7. Quantisation non-reproducibility across GPU gens.

## Top research-backed levers (build sequence)
1. **Zero-shot strong VLM** (LLaVA-OneVision-7B / InternVL3.5 / Qwen2.5-VL) + concise prompt — reproduce the winning floor.
2. **Canonical answer layer** (ontology: tool/organ/task/action/yes-no → short templated answer) — likely the single biggest BLEU lever; not just gaming, reduces variance.
3. **Detector-informed answer prior** (tool/organ/task perception feeds the VLM as authoritative) — = the winning Capybara strategy.
4. **Question-conditioned temporal retrieval** (score all 30 frames by Q-relevance → top-k=4–8 to the VLM) — beats uniform sampling; the under-explored lever. Headline paper claim.
5. **Structured scene memory** (tool/organ/action timeline as text evidence) + scene-graph (SSG-VQA).
6. **Counterfactual scene-graph QA curriculum** (perturb one variable: tool/organ/action/task; paraphrases AFTER labels fixed) — reduces shortcut learning.
7. **Uncertainty-gated expert mixture** (zero-shot VLM + retrieval+memory + detector/rule expert; pick least-risky by inter-expert disagreement / semantic entropy) — challenge-legal (one answer out), strong paper-evaluation.
- ALWAYS: clone the official scorer (one-pred/Q, BLEU vs each of 5 refs, max, mean); version every prompt; release the synthetic-QA engine.

## OUR STATUS vs this research (autonomous-run wins)
- **We ALREADY beat the 0.4215 winner: Cat-2 real-grounded BLEU 0.7428** (cfg nf4_thr0.3_mean) — our spine-grounding = the detector-informed approach the research endorses. **But** local n=11 → must validate it's not metric-inflated / shortcut.
- **Cat-1 detector mAP 0.7746** (thr0.5_yolo26m, self-consistency, no human GT).
- Next research-backed levers to try: canonical answer layer (BLEU lever), Q-conditioned temporal retrieval (vs our current uniform frames), counterfactual-QA anti-shortcut, T4-quantisation repro check, uncertainty-gated ensemble.

Models recommended: LLaVA-OneVision-7B, InternVL3.5, Qwen2.5-VL. Compute: low-to-mid single node sufficient (winners used ≤7B + light classifiers).
