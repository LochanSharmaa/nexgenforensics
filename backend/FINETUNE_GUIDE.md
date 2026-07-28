# Fine-tuning guide

The shipped engine uses pretrained ArcFace r100 weights from the InsightFace
`buffalo_l` pack. **You almost certainly do not need to fine-tune.** Read this
section before the rest.

## Try these first

Fine-tuning a face recognition backbone is expensive, easy to get wrong, and
usually solves a problem that was not the actual problem. In order of
effort-to-benefit:

1. **Calibrate your threshold.** Most "the model is inaccurate" reports are a
   threshold mismatch. Run `scripts/calibrate_threshold.py` on your own imagery.
2. **Enrol more images per subject.** Three to five images spanning pose,
   lighting, and age moves recall far more than fine-tuning does.
3. **Fix enrolment quality.** One poor enrolment image degrades every future
   search against that subject. The quality gate already refuses the worst, but
   sitting just above the gate is still weak.
4. **Check alignment.** If `detector.produces_landmarks` is false you are
   running box-crop alignment, and accuracy is materially below normal. Install
   the InsightFace pack.

Fine-tuning is justified when your imagery is genuinely out of distribution for
a web-photo-trained model — heavy infrared, extreme low resolution CCTV, or a
demographic distribution far from the training set — **and** you have measured
that gap rather than assumed it.

## If you are going ahead

Utilities live in `nexgen_engine/training/` (curriculum, scheduler, hard-negative
mining) and `nexgen_engine/losses/` (ArcFace, CosFace, AdaFace, triplet). The
datasets already extracted under `src_extracted/` (WebFace, MS1M-RetinaFace,
AgeDB, TinyFace) are suitable starting points.

1. **Establish a baseline first.** Run `tests/test_recognition_accuracy.py` and
   `scripts/calibrate_threshold.py` on a held-out split of *your* data. Without
   this number you cannot tell whether fine-tuning helped.
2. **Confirm the lawful basis** for every image you train on. Training data is
   biometric data, and consent for verification is not consent for training.
3. **Hold out by identity, not by image.** Splitting images of the same person
   across train and test leaks identity and produces an accuracy figure that
   collapses in deployment.
4. **Fine-tune the backbone**, keeping the 512-d output and the 112×112 aligned
   input so the rest of the pipeline is unchanged.
5. **Re-measure on the same held-out split**, and report genuine/impostor
   separation and rank-1 accuracy — not training loss.
6. **Measure demographic performance separately.** An aggregate improvement can
   hide a regression for a subgroup. This is the most commonly skipped step and
   the most consequential.
7. **Export to ONNX** and place it where `NEXGEN_MODEL_ROOT` points.
8. **Recalibrate thresholds.** A new model has a new score distribution; the old
   threshold is meaningless against it.
9. **Keep the previous checkpoint** and a rollback path.

## Re-enrolment is mandatory

Templates from different models are not comparable. Changing the recognition
model invalidates every stored template, and comparing an old template against a
new one produces a meaningless score rather than an error.

Plan the migration before you swap models: keep the original enrolment images,
re-enrol everyone against the new model, and only then cut over.

## Do not publish accuracy claims

Not until an independent party has evaluated the model on data neither you nor
they selected. A self-reported benchmark on a self-selected split is a
development signal, not evidence.
