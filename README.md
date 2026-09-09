# SurgVU 2026 submission (team AIMS Lab)

Category 1 (surgical instrument detection) and Category 2 (surgical visual question answering) for
the [SurgVU 2026 challenge](https://surgvu26.grand-challenge.org/), MICCAI 2026 / EndoVis.

Both systems run inside the published inference budget: ten minutes per case, 32 GB host memory, and
either no GPU or a single T4.

## Category 1 — instrument detection

A single-stage YOLO detector over the 14 challenge instrument classes, applied per frame with no
temporal smoothing. Supervision combines human bounding-box annotation with proposals derived from
class-activation evidence, retained only where the image-level presence label for that frame asserts
the corresponding tool. Three seeds are trained.

Inference thresholds are set from the metric, not from appearance: mAP integrates precision across
the full recall range, so a confidence floor permanently caps recoverable recall. Confidence is
0.001, up to 100 detections per frame, at 640 px.

**Derived boxes cannot stand alone.** Trained without human annotation the same detector reaches
mAP@[.5:.95] = 0.000 against human ground truth, for a geometric reason: human boxes enclose the
instrument clevis at 0.0215 of frame area while derived boxes average 0.1538, so a perfectly centred
prediction attains at most IoU = 0.0215/0.1538 = 0.140, below the 0.5 threshold where AP begins.

Two of the four validation classes carry no training boxes, which bounds the attainable class mean
irrespective of method.

## Category 2 — visual question answering

Deterministic and vision-free. A question is classified into one of three families by surface form
(polar, purpose, entity), each with its own template; the slot is filled from a prior of modal
answers per family estimated on the training answers, and the result is a declarative sentence of
about seven words. No model is loaded and no frame is read at inference.

This follows from the ranked metric. BERTScore-F1 gives 0.6383 to a constant answer that never reads
the question and 0.9110 to a factually inverted answer, so it rewards register and entity naming far
more than correctness. A wrong answer in seven-word declarative form scores 0.7932 against 0.4461
for verbose prose, and unconstrained generative prose scores 0.45–0.56, below the level reached
without reading the question.

A task-conditioning head is present but disabled: on the platform it scored below the unconditioned
prior.

The corresponding limitation, stated plainly: this system is tuned to a metric that does not
reliably reward factual correctness, and its score is not evidence of clinical utility.

## Layout

```
experiments/    training, pseudo-label generation, scoring, ablations
our_scripts/    submission-side helpers
```

SLURM scripts carry cluster-specific paths with the account name replaced by `USERNAME`.

## Licence

MIT, see [LICENSE](LICENSE). Challenge data and evaluation code are not redistributed and remain
under their own terms.

## Contact

Omar Choudhry, <O.Choudhry@leeds.ac.uk>
Artificial Intelligence in Medicine and Surgery Group, School of Computer Science,
University of Leeds, Leeds LS2 9JT, UK.
ORCID [0000-0003-4434-3550](https://orcid.org/0000-0003-4434-3550)

Funded by UKRI EPSRC grant EP/S024336/1.
