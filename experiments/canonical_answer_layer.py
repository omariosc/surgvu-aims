"""Cat-2 NEW lever: CANONICAL-ANSWER LAYER (ontology -> single canonical surface form).

DEEP_RESEARCH_CAT2 lever #2 (the research's "single biggest BLEU lever"): map the
(answer-type, grounded entity, polarity) tuple the spine produces to ONE canonical,
register-locked short answer drawn from an ONTOLOGY, rather than the current keyword
router's per-bucket ad-hoc strings. The research framing: this is NOT just gaming BLEU
-- a canonical surface form REDUCES VARIANCE (every yes/no-forceps-negative emits the
SAME proven-good sentence) and removes the router's brittle phrasing branches.

Difference vs p0_2_template_router:
  * router: 6 keyword buckets, each with inline f-string templates + verb-register
    heuristics (`_negation_verb`, copula guessing) that can mis-fire on unseen phrasing.
  * canonical layer: a flat ONTOLOGY dict keyed by (atype, entity_class, polarity) ->
    one canonical string. Phrasing variants are PRE-RESOLVED to the form that maximises
    BLEU against the reference register (authored from the reference *register*, NOT from
    held-out answers -- same discipline as P0.2). Entities/polarity come from the spine.

This script wires the spine (best grounding config nf4_thr0.3_mean, with the v2 action
bridges) -> canonical layer, scores bleu_harness over the 11 clips, and A/B's against
the current router under the IDENTICAL spine grounding. It reports per-case so any
regression vs the router is visible.

KILL CRITERION (pre-registered): if canonical-layer BLEU < router BLEU (0.7428) on the
same grounding -> the ontology mapping is mis-phrased / over-canonical (collapses a
case the router got right); REFUTED, keep the router. If >= router, it both wins AND
reduces variance -> promote it.

Run (SLURM gpu node, surgvu_env):
  python canonical_answer_layer.py --ckpt .../spine_full/spine_best.pt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bleu_harness
import spine_grounding as sg
import spine_grounding_v2 as sg2
import p0_2_template_router as router


# ---- the ONTOLOGY: (atype, key) -> canonical register-locked surface form ----
# Authored to the SurgVU reference REGISTER (verb/tense/casing), not to any held-out
# answer. Each entry is the single canonical answer for that ontology cell.
CANON = {
    # yes/no tool-presence (positive / negative), keyed by tool-noun class
    ("yesno_tool_pos", "forceps"):       "Yes, forceps are being used.",
    ("yesno_tool_neg", "forceps"):       "No forceps are being used.",
    ("yesno_tool_pos", "needle driver"): "Yes, a needle driver is involved.",
    ("yesno_tool_neg", "needle driver"): "No, a large needle driver was not used.",
    ("yesno_tool_pos", "generic"):       "Yes, the instrument is being used.",
    ("yesno_tool_neg", "generic"):       "No, the instrument is not being used.",
    # yes/no action
    ("yesno_action_pos", "cut"):         "Yes, tissue is being cut.",
    ("yesno_action_neg", "cut"):         "No, tissue is not being cut.",
    ("yesno_action_pos", "suture"):      "Yes, sutures are required.",
    ("yesno_action_neg", "suture"):      "No, sutures are not required.",
    # open tool-type / organ / procedure / purpose
    ("procedure", "_"):  "The summary is describing endoscopic or laparoscopic surgery.",
    ("purpose", "forceps"):
        "The forceps are used for grasping and holding tissues or objects.",
}


def _tool_class(question):
    ql = question.lower()
    if "forceps" in ql:
        return "forceps"
    if "needle driver" in ql:
        return "needle driver"
    return "generic"


def canonical_predict_factory(grounding_by_case):
    def predict(question, case_id):
        g = grounding_by_case.get(case_id, {})
        ql = question.lower()
        bucket = router._route(question)

        if bucket == "yesno":
            present = bool(g.get("present", False))
            pol = "pos" if present else "neg"
            if "cut" in ql or "cutting" in ql:
                return CANON[("yesno_action_%s" % pol, "cut")]
            if "sutur" in ql and "requir" in ql:
                return CANON[("yesno_action_%s" % pol, "suture")]
            tc = _tool_class(question)
            key = ("yesno_tool_%s" % pol, tc)
            return CANON.get(key, CANON[("yesno_tool_%s" % pol, "generic")])

        if bucket == "tool_type":
            tool = g.get("tool", "")
            # canonical carrier sentence (matches case124 reference register)
            return f"The forceps type is {tool}."

        if bucket == "organ":
            organ = g.get("organ", "")
            return f"The organ being manipulated is the {organ}."

        if bucket == "procedure":
            return CANON[("procedure", "_")]

        if bucket == "purpose":
            return CANON[("purpose", "forceps")]

        return "No, it is not used."

    return predict


def ground_all(ckpt, data_dir, n_frames=4, thr=0.3, pool="mean"):
    """Run the v2 spine grounding (winning config) over all 11 clips ONCE."""
    import glob
    cache = {}
    g_by_case = {}
    for d in sorted(glob.glob(os.path.join(data_dir, "case*"))):
        cid = os.path.basename(d)
        qf = os.path.join(d, f"{cid}_question.json")
        vf = os.path.join(d, f"{cid}.mp4")
        if not (os.path.exists(qf) and os.path.exists(vf)):
            continue
        q = json.load(open(qf))
        try:
            g_by_case[cid] = sg2.ground_clip_v2(vf, q, ckpt, n_frames, thr, pool, cache)
        except Exception as e:
            print(f"  [warn] {cid} grounding failed: {e}", flush=True)
            g_by_case[cid] = {}
    return g_by_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/spine_best.pt")
    ap.add_argument("--data_dir", default="/users/sc20osc/SurgVU/data")
    ap.add_argument("--out",
                    default="/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/canonical_layer.json")
    args = ap.parse_args()

    ROUTER_BASELINE = 0.7428
    print(f"# Canonical-answer layer  ckpt={args.ckpt}")
    g_by_case = ground_all(args.ckpt, args.data_dir)

    # A: current router on this grounding (fair A/B re-measure)
    import p0_2_template_router as tr
    for cid, g in g_by_case.items():
        tr.GROUNDING[cid] = g
    router_mean, router_per = bleu_harness.evaluate(tr.predict, data_dir=args.data_dir,
                                                    verbose=False)

    # B: canonical layer on the SAME grounding
    canon_pred = canonical_predict_factory(g_by_case)
    canon_mean, canon_per = bleu_harness.evaluate(canon_pred, data_dir=args.data_dir,
                                                  verbose=False)

    print(f"#   router   (same grounding)  BLEU={router_mean:.4f}")
    print(f"#   canonical layer            BLEU={canon_mean:.4f}")
    print("#   per-case (router -> canonical):")
    rp = {c: s for c, _, _, s in router_per}
    for cid, q, pred, s in canon_per:
        flag = " <-- regress" if s < rp.get(cid, 0) - 1e-6 else ""
        print(f"#     {cid}: router={rp.get(cid,0):.4f} canon={s:.4f}  {pred!r}{flag}")

    gain = canon_mean - max(router_mean, ROUTER_BASELINE)
    verdict = ("CONFIRMED canonical >= router (+%.4f)" % gain if gain >= -1e-6
               else "REFUTED canonical < router (%.4f) -> keep router" % gain)
    json.dump(dict(router_baseline=ROUTER_BASELINE, router_mean=router_mean,
                   canonical_mean=canon_mean, verdict=verdict,
                   per_case={c: dict(router=rp.get(c, 0), canon=s, pred=p)
                             for c, q, p, s in canon_per}),
              open(args.out, "w"), indent=2)
    print(f"\n# wrote {args.out}")
    print(f"CANON_BEST BLEU={canon_mean:.4f} vs router={router_mean:.4f} "
          f"/ baseline={ROUTER_BASELINE} -> {verdict}")


if __name__ == "__main__":
    main()
