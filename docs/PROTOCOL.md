# MammoDiffusion v1.1 consolidated protocol

This document describes the final v1.1.0 whole-image study. Schema values named
`protocol_version = 2` and historical references to “V2” identify the second internal methodological
phase retained for provenance; they do not change the public release version.

The repository encodes a fixed comparison family and deterministic analysis, but the study was not
formally preregistered. “Primary” below means primary within the frozen repository protocol, not
prospectively registered before every earlier project decision.

## 1. Cohort, endpoint, and preprocessing

The downstream endpoint is the RSNA competition `cancer` label: cancer-positive (`1`) versus
non-cancer (`0`). A non-cancer label must not be interpreted as proof that an image contains no
benign or suspicious finding.

The source is the derived Kaggle release
[RSNA Breast Cancer 512 PNGs](https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-512-pngs),
which distributes the RSNA Screening Mammography Breast Cancer Detection images already converted
to PNG with the competition metadata. The preprocessing notebook requires a user-supplied archive;
it does not perform an opaque download or read DICOM. No notebook fetches the RSNA source archive.
The generator notebooks can retrieve the already processed cohort from the author's Drive when it is
absent, which is consistent with their status as explicit real-run tools; the augmentation notebook
keeps the same retrieval behind `ALLOW_PROCESSED_DOWNLOAD = False` and no longer deletes an existing
`data/real_augmented/` by default.

Preprocessing keeps MLO views, derives patient status as the maximum image-level cancer label,
excludes negative images from positive patients, and selects at most one image per patient. It then
caps the non-cancer-to-cancer ratio at 5:1 before patient-level stratified splitting. The resulting
2,916-patient cohort contains 486 positives and 2,430 negatives; this constructed 16.7% prevalence
is not source or screening prevalence. Train/validation/test contain 2,041/437/438 patients and
340/73/73 positive patients. Images are converted to single-channel grayscale, visually oriented,
intensity-normalized, and resized with aspect-preserving padding to 512×512.

`processed_dataset_reuse.audit_patient_split_disjointness` rejects patient overlap. Reuse mode
checks the stored manifests and images but cannot reconstruct source-population counts that were
discarded before the final 5:1 cohort was written.

## 2. Generator benchmark and selection

`01_Unified_Generator_Benchmark.ipynb` defaults to `RUN_REAL_BENCHMARK = False`. Review mode reads
the protocol, registry, and saved small reports; it does not inspect image pools, checkpoints,
encoder caches, CUDA, or model repositories and performs no writes. Real mode is an explicit,
asset-dependent action and does not download weights.

The benchmark uses:

- training and traditional-training-augmentation images as the train-reference memorization pool;
- validation positives for distribution quality and descriptive similarity;
- candidate synthetic pools for diversity and duplicate analysis;
- no classifier-test image, metadata, embedding, or metric.

Each eligible FILTERED pool contains 1,361 unique readable images. Full-pool RAD-DINO KID is the
primary ranking metric and FID is descriptive because the real reference is small. Balanced PRDC
uses all available validation positives against an equal-size deterministic synthetic subset.
Repeated KID/PRDC subsampling is without replacement. RAW results preserve generator failures;
FILTERED is the official ranking representation.

Eligibility requires the configured valid-image count, exact-duplicate and detected train-reference
memorization limits, zero corruption, complete metrics, no test access, and an eligible registry
role. Perceptual-hash duplicate rate and RAD-DINO coverage use descriptive reference values rather
than binary gates. Passing any gate means only that a candidate can enter this project's ranking;
it does not certify cancer content, label adherence, radiological validity, or clinical realism.

The deterministic ranking order is:

1. RAD-DINO KID, lower is better;
2. RAD-DINO coverage, higher is better;
3. RAD-DINO precision, higher is better;
4. descriptive RAD-DINO FID, lower is better;
5. Inception KID, lower is better;
6. RAD-DINO KID stability SD, lower is better;
7. generator ID as a deterministic tie-break.

Efficiency is not a ranking field. Every ranking field must be present for every eligible
candidate. G02 is selected for the fine-tuned family and G07 for the from-scratch family.

## 3. Training metadata, generation records, and provenance

The memorization pool is assembled in memory from `data/processed/metadata/train.csv` and the
traditional-augmentation metadata when present. Rows must resolve to train samples, labels must
agree, paths and sample IDs must be unique, and train patients must not overlap validation.
Synthetic evaluation candidates are excluded and test metadata is not opened.

Generation manifests are operational resume records. They retain generator/checkpoint identity,
sampling parameters, class, seed, expected count, image size, output directory, and relevant
decoder/model details. Content checks prevent mixing incompatible partial outputs. These records do
not make Git state, benchmark execution ID, or GPU identity a scientific compatibility gate.

Historical metadata and results contain absolute paths from the original workstation. They remain
unchanged as execution provenance. Active loaders reroot recognized repository-relative suffixes
when possible, so an absolute legacy string is not itself a portable asset reference. New records
should prefer relative paths.

## 4. Classifier design and realized training

The configured matrix is:

- architectures: MaxViT-512 and Mammo-FM;
- conditions: real only, real plus traditional augmentation, real plus G02 positives, and real plus
  G07 positives;
- seeds: 17, 42, and 73;
- outputs: 24 completed seed runs and 8 mean-probability ensembles.

Within each architecture the maximum optimizer-update budget, effective batch size, loss,
scheduler, early-stopping rule, and validation criterion are fixed. Validation PR-AUC selects
checkpoints. The maximum is 6,400 optimizer updates, but that is an upper budget rather than equal
realized compute: 23 runs stopped early and completed jobs range from 2,750 to 6,400 updates.

Dataset composition is also not dose-matched across every condition. Real only contains 2,041
images. Traditional augmentation adds 1,020 transformed positives, for 3,061 images and a
1,701/1,360 negative/positive balance. Each synthetic condition adds 1,361 positives, for 3,402
images and a 1,701/1,701 balance. Thus real-only and traditional comparisons combine augmentation
method with dataset size, class balance, sample exposure, and potentially early-stopping time. The
G02-versus-G07 comparison has the same nominal synthetic dose.

Synthetic conditions consume only the selected FILTERED positive pool; real-only never resolves a
synthetic directory. Checkpoint content and dataset fingerprints are local resume-safety mechanisms.
GPU auto/index/UUID selection only configures the visible device.

## 5. Frozen operating points and report regeneration

Validation chooses two operating points for each seed and ensemble:

- the decision threshold maximizing Youden's J;
- the threshold with the best validation sensitivity among thresholds achieving nominal validation
  specificity of at least 0.90.

Test mode requires both values and cannot call Youden, F1 optimization, or specificity search. The
decision threshold drives confusion-matrix metrics. The nominal-0.90 threshold is applied unchanged
to test, where both sensitivity and **achieved test specificity** are reported. Achieved test
specificity is not guaranteed to equal or exceed 0.90. Every patient-bootstrap replica uses the
same frozen thresholds.

`notebooks/utility/rebuild_classifier_reports.py` is the canonical dependency-ordered rebuild: it
first rebuilds validation ensembles and comparisons, then rebuilds 24 seed reports and 8 held-out
ensembles from existing validation/test prediction CSVs without inference. The lower-level
`regenerate_classifier_metrics.py` performs the seed/test stage. Corrected reports are written to
their canonical paths. The historical field name `pr_auc` is implemented as **average precision**
(stepwise precision multiplied by each increase in recall), not trapezoidal area under an
interpolated precision-recall curve. All observations with equal scores are treated as one
threshold, so average precision is invariant to row order within tied scores.

## 6. Comparisons and uncertainty

The primary metric is patient-level PR-AUC. Within each architecture the configured contrasts are:

1. real only versus traditional augmentation;
2. real only versus G02 positives;
3. real only versus G07 positives;
4. G02 positives versus G07 positives.

The eight contrasts form one Holm-adjustment family. Paired bootstrap resampling uses the same
patient indices for each condition and samples positive and negative strata separately with
replacement for 2,000 iterations at seed 20260714. Reports include the bootstrap mean difference
and 2.5th/97.5th percentile interval.

The field `p_value_two_sided` is an empirical tail area: twice the proportion of bootstrap
differences on the opposite side of zero from the observed mean direction, capped at one. It is not
a permutation p-value or a null-centered bootstrap-test p-value. Holm is applied consistently to
these stored tail-area values, but the adjusted values should remain exploratory. No adjusted
comparison is below 0.05 in the final report.

The largest favorable result is MaxViT G07 versus real only: point PR-AUC 0.5230 versus 0.4128,
bootstrap mean difference +0.1052, percentile interval [+0.0247, +0.1827], tail area 0.011, and
Holm-adjusted value 0.088. Mammo-FM does not reproduce the pattern: 0.3161 versus 0.3241, bootstrap
mean difference −0.0083, interval [−0.0529, +0.0342], tail area 0.708, adjusted value 1.000. This is
an architecture-dependent descriptive pattern; no formal interaction test was run.

## 7. Interpretability boundary

The classifier workflows save Grad-CAM and Integrated Gradients panels for deterministically
selected validation cases. The study has no lesion masks and does not compute attribution overlap,
lesion localization, or causal feature use. These panels are qualitative diagnostics only; they
cannot demonstrate that either classifier attends to true or synthetic pathology.

## 8. Final-evaluation guard and test-history caveat

The safe notebook defaults are:

```python
RUN_FINAL_EVALUATION = False
OVERWRITE_TEST_PREDICTIONS = False
```

A future inference run needs explicit `RUN_FINAL_EVALUATION = True`. Before test data or models are
opened, code requires all seed and ensemble threshold pairs from canonical validation results.
Existing prediction files are protected unless overwrite is separately enabled. Review mode loads
only saved reports and does not build adapters, load models, touch CUDA, or open test images.

The current implementation isolates test data from generator selection and threshold choice.
However, the 438-patient test cohort is the same project split used across the historical internal
V1 and V2 phases, and the earlier phase had less strict development/test separation. It is not a
new independent or external confirmation cohort.

## 9. Outputs and reproduction boundary

Canonical benchmark reports live under `results/2_diffusers/benchmark/`; seed reports and
predictions under `results/3_classifiers/seed_runs/`; validation ensembles under
`results/3_classifiers/validation_ensembles/`; and held-out ensemble reports under
`results/4_final_evaluation/`.

`rebuild_generator_ranking.py` regenerates the ranking from the committed generator summary.
`rebuild_classifier_reports.py` regenerates the full validation-to-test classifier report chain
from saved predictions. Neither loads images or models. A full generator rerun remains a manual
explicit action requiring external assets.

The Git release intentionally excludes datasets, checkpoints, local encoders, and model weights.
The full archive requirements and active artifact consumers are listed in
[ARCHIVE_AND_RELEASE.md](ARCHIVE_AND_RELEASE.md). Scientific claim limits are listed in
[LIMITATIONS.md](LIMITATIONS.md).
