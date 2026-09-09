"""Cat-2 NEW lever: QUESTION-CONDITIONED TEMPORAL RETRIEVAL (vs uniform sampling).

DEEP_RESEARCH_CAT2 lever #4 (the under-explored one): instead of aggregating the
spine over UNIFORMLY-sampled frames (`np.linspace`, as spine_grounding /
grounding_variants / spine_grounding_v2 all do), DENSELY sample the 30 s clip
(N_DENSE frames @ ~1 fps, matching the test 1-fps sub-sampling), then SCORE each
frame by its RELEVANCE to the question's entity and aggregate only the TOP-K most
relevant frames.

Why this can beat uniform: the questioned tool/action is present in only part of a
30 s clip; uniform mean/max-pool either dilutes a brief appearance (mean) or fires
on a single noisy frame (max). Q-relevance top-k focuses the evidence on the frames
that actually concern the question -> cleaner polarity for the template router.

Relevance score per frame f for question q:
  * tool/presence question (forceps / needle driver / scissors / ...):
        rel(f) = max over the queried tool keys of sigmoid(tool_logit)[key]
  * forceps-type question:  rel(f) = max over the forceps-family tool probs
  * organ question:         rel(f) = 1 - softmax(task)[Other]  (task-informative frames)
  * action (cut / suture-required): rel(f) = max prob of the action's evidence tools
  * fallback:               rel(f) = max tool prob (most tool-active frames)

We then aggregate the spine over the top-k frames exactly as the WINNING config
(thr=0.3, pooling=mean) does, feed the v2 router (with the action bridges), and
score bleu_harness.evaluate over the 11 clips. We sweep k and N_DENSE and compare
to the uniform baseline (re-run here under the SAME N_DENSE for a fair A/B).

KILL CRITERION (pre-registered): if best Q-cond BLEU <= the uniform baseline
(0.7428) by < +0.005 at any k -> retrieval doesn't help on n=11; uniform pooling
is already sufficient at 1-fps density; lever REFUTED (the gap is spine accuracy /
phrasing, not frame selection).

Run (SLURM gpu node, surgvu_env):
  python qcond_retrieval_bleu.py --ckpt .../spine_full/spine_best.pt
Appends a [SurgVU][jobid] line to AUTONOMOUS_RESULTS.md from the slurm wrapper.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import bleu_harness
import spine_grounding as sg
import spine_grounding_v2 as sg2

# forceps family (for the forceps-type relevance + queried-tool relevance)
FORCEPS_KEYS = ("cadiere_forceps", "prograsp_forceps", "bipolar_forceps",
                "tip_up_fenestrated_grasper")


def _query_keys_for_relevance(question, TOOLS):
    """Return tool-INDICES whose probability is the relevance signal for this Q."""
    ql = question.lower()
    idx = {t: i for i, t in enumerate(TOOLS)}
    # action questions -> evidence tools
    if "cut" in ql or "cutting" in ql:
        keys = sg2.CUT_TOOL_KEYS
    elif "sutur" in ql and "requir" in ql:
        keys = sg2.SUTURE_TOOL_KEYS
    elif "what type" in ql and "forceps" in ql:
        keys = FORCEPS_KEYS
    else:
        _, keys = sg._question_tool_keys(question)
    return [idx[k] for k in keys if k in idx]


def _frame_relevance(tp, kp, question, TOOLS, TASKS):
    """tp:[F,14] sigmoid, kp:[F,8] softmax -> rel:[F] in [0,1]."""
    ql = question.lower()
    if "organ" in ql:
        # task-informative frames (not the dominant 'Other' filler)
        other_i = TASKS.index("Other") if "Other" in TASKS else -1
        if other_i >= 0:
            return 1.0 - kp[:, other_i]
        return kp.max(axis=1)
    qidx = _query_keys_for_relevance(question, TOOLS)
    if qidx:
        return tp[:, qidx].max(axis=1)
    # fallback: most tool-active frames
    return tp.max(axis=1)


def ground_clip_qcond(video_path, question, ckpt_path, n_dense, k, thr, pooling,
                      cache):
    key = (video_path, n_dense)
    if key not in cache:
        cache[key] = sg2.extract_probs(video_path, ckpt_path, n_dense)
    tp, kp, ck_thr, TOOLS, TASKS = cache[key]
    if tp is None:
        return {}, 0
    F = tp.shape[0]
    rel = _frame_relevance(tp, kp, question, TOOLS, TASKS)
    kk = min(k, F)
    top = np.argsort(-rel)[:kk]
    tp_k, kp_k = tp[top], kp[top]
    thr_eff = ck_thr if thr is None else thr
    present_keys, dom_task = sg2._aggregate(tp_k, kp_k, thr_eff, pooling, TOOLS, TASKS)
    return sg2._grounding_from(question, present_keys, dom_task), kk


def predict_factory(ckpt, data_dir, n_dense, k, thr, pooling, cache, uniform=False):
    import p0_2_template_router as tr

    def predict(question, case_id):
        vf = os.path.join(data_dir, case_id, f"{case_id}.mp4")
        g = {}
        if os.path.exists(vf):
            try:
                if uniform:
                    # uniform A/B baseline at the SAME density: take all N_DENSE frames
                    g, _ = ground_clip_qcond(vf, question, ckpt, n_dense, n_dense,
                                             thr, pooling, cache)
                else:
                    g, _ = ground_clip_qcond(vf, question, ckpt, n_dense, k, thr,
                                             pooling, cache)
            except Exception as e:
                print(f"  [warn] {case_id} qcond failed: {e}", flush=True)
        tr.GROUNDING[case_id] = g
        return tr.predict(question, case_id)

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/scratch/USERNAME/miccai-2026/SurgVU/models/spine_full/spine_best.pt")
    ap.add_argument("--data_dir", default="/users/USERNAME/SurgVU/data")
    ap.add_argument("--out",
                    default="/scratch/USERNAME/miccai-2026/SurgVU/models/spine_full/qcond_retrieval.json")
    args = ap.parse_args()

    BASELINE = 0.7428  # uniform nf4_thr0.3_mean (job 6042378)
    N_DENSE = [30, 16]            # 1-fps-like dense sampling, + a coarser comparator
    KS = [3, 4, 6, 8]            # top-k Q-relevant frames
    THR, POOL = 0.3, "mean"     # the WINNING grounding config, held fixed

    cache = {}
    results = []
    print(f"# Q-conditioned temporal retrieval  ckpt={args.ckpt}")
    print(f"# baseline (uniform nf4_thr0.3_mean) = {BASELINE}")

    # uniform A/B at each density (fair comparator: same decoded frames, no top-k)
    uni = {}
    for nd in N_DENSE:
        pf = predict_factory(args.ckpt, args.data_dir, nd, nd, THR, POOL, cache, uniform=True)
        m, _ = bleu_harness.evaluate(pf, data_dir=args.data_dir, verbose=False)
        uni[nd] = m
        print(f"#   UNIFORM  n_dense={nd:3d}            BLEU={m:.4f}", flush=True)

    for nd in N_DENSE:
        for k in KS:
            if k >= nd:
                continue
            pf = predict_factory(args.ckpt, args.data_dir, nd, k, THR, POOL, cache)
            m, _ = bleu_harness.evaluate(pf, data_dir=args.data_dir, verbose=False)
            tag = f"qcond_nd{nd}_k{k}_thr{THR}_{POOL}"
            results.append(dict(cfg=tag, n_dense=nd, k=k, bleu=m,
                                uniform_same_density=uni[nd]))
            print(f"#   {tag:26s} BLEU={m:.4f}  (uniform@{nd}={uni[nd]:.4f})", flush=True)

    results.sort(key=lambda r: r["bleu"], reverse=True)
    best = results[0] if results else {"cfg": "none", "bleu": 0.0}
    best_uni = max(uni.values()) if uni else 0.0
    gain = best["bleu"] - max(BASELINE, best_uni)
    verdict = ("CONFIRMED Q-cond > uniform (+%.4f)" % gain if gain >= 0.005
               else "REFUTED Q-cond <= uniform (+%.4f) -> frame-selection not the lever" % gain)
    json.dump(dict(baseline=BASELINE, uniform=uni, results=results,
                   best=best, verdict=verdict), open(args.out, "w"), indent=2)
    print(f"\n# wrote {args.out}")
    print(f"QCOND_BEST cfg={best['cfg']} BLEU={best['bleu']:.4f} "
          f"vs uniform_best={best_uni:.4f} / baseline={BASELINE} -> {verdict}")


if __name__ == "__main__":
    main()
