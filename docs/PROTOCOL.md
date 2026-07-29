# MammoDiffusion consolidated protocol

## 1. Cohort and preprocessing

The binary task is malignant finding versus no lesion on 512×512 grayscale MLO mammograms.
Analysis can start from the processed PNG cohort; original DICOMs are needed only for a full rebuild.
See the two workflows in the main README. Splits are patient-level and
`processed_dataset_reuse.audit_patient_split_disjointness` rejects train/validation/test overlap.
The final held-out cohort has 438 patients.

The original source is the official
[RSNA Screening Mammography Breast Cancer Detection dataset](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data).
The preprocessing notebook requires a user-supplied archive, keeps MLO views, selects one image per
patient, normalizes laterality/visual tissue side and intensity, handles corrupt inputs, converts to
PNG, and resizes to 512×512. No opaque download is part of the default path.

## 2. Generator benchmark

The default of `01_Unified_Generator_Benchmark.ipynb` is
`RUN_REAL_BENCHMARK = False`. Review mode reads the protocol, registry, and saved small
`generator_summary.csv`/`generator_ranking.csv`; it does not inspect `data/`, `experiments/`, image
pools, checkpoints, encoder caches, GPU, or model repositories, and it performs no writes.

Real mode must be enabled explicitly. It requires local train/validation metadata, declared
synthetic pools, local InceptionV3 and RAD-DINO encoders, and CUDA. It
does not download weights. Missing or incompatible embedding-cache inputs trigger cache
invalidation; encoder and image hashes have no downstream role beyond that cache decision.

The benchmark uses:

- train and traditional train augmentation for memorization;
- validation positives for distribution quality and descriptive similarity;
- the candidate synthetic pools for diversity and duplicate analysis;
- no classifier-test image, metadata, embedding, or metric.

Each eligible FILTERED pool contains 1,361 unique readable images. Full-pool KID is primary and FID
is descriptive because the real reference is small. Balanced PRDC uses every validation positive
against an equal-size deterministic synthetic subset. Repeated KID/PRDC subsampling is without
replacement. RAW results preserve generator failures; FILTERED is the official ranking
representation.

Eligibility gates are minimum valid image count, maximum exact-duplicate rate, maximum train
memorization rate, zero corrupted-file rate, complete metrics, no test access, and an eligible
registry role. Perceptual-hash duplicate rate and RAD-DINO coverage are descriptive reference values.
The ranking order is:

1. RAD-DINO KID, lower is better;
2. RAD-DINO coverage, higher is better;
3. RAD-DINO precision, higher is better;
4. descriptive RAD-DINO FID, lower is better;
5. Inception KID, lower is better;
6. RAD-DINO KID stability SD, lower is better;
7. generator ID as deterministic tie-break.

Every ranking field must be present for every eligible candidate. The result remains G02 for the
fine-tuned family and G07 for the from-scratch family.

## 3. Training metadata and generation records

The memorization pool is assembled in memory from `data/processed/metadata/train.csv` and the
existing traditional-augmentation metadata, when present. Rows must resolve to train samples,
labels must agree, paths and sample IDs must be unique, and train patients must not overlap the
validation cohort. Synthetic evaluation candidates are excluded, and test metadata is not opened.

Generation manifests are operational resume records. They store the generator/checkpoint,
sampling parameters, class, seed, expected count, image size, output directory, and decoder/model
details needed to avoid mixing incompatible outputs. Content checks verify expected names, readable
files, counts, partial outputs, and directly relevant checkpoint changes. They do not bind a
scientific result to Git state, a benchmark execution ID, or a GPU identity.

## 4. Classifier design

The fixed matrix is:

- architectures: MaxViT-512 and Mammo-FM;
- conditions: real only, real plus traditional augmentation, real plus G02 positives, real plus
  G07 positives;
- seeds: 17, 42, and 73.

Within each architecture the optimizer-update budget, effective batch size, loss, scheduler,
early-stopping rule, and validation criterion are fixed. Validation PR-AUC selects checkpoints.
Synthetic conditions consume only the selected FILTERED positive pool; real-only never resolves a
synthetic directory. Checkpoint content and dataset fingerprints are used locally for safe resume,
not as cross-stage scientific gates. GPU auto/index/UUID selection only configures the visible
device.

## 5. Frozen thresholds and corrected reports

Validation chooses two operating points for each seed and ensemble:

- the decision threshold maximizing Youden's J;
- the threshold meeting target specificity 0.90 with the best validation sensitivity.

Test mode requires both values. It cannot invoke Youden, an F1 optimizer, or a specificity search.
The decision threshold drives the confusion matrix, sensitivity/recall, specificity, precision,
NPV, F1, accuracy, balanced accuracy, and MCC. The frozen target-specificity threshold is applied
to test and the achieved test specificity is reported. Every patient-bootstrap replica uses these
same fixed values.

`notebooks/utility/regenerate_classifier_metrics.py` rebuilds 24 seed reports and 8 ensemble reports
from the existing validation/test prediction CSVs, without inference. Corrected metrics are written
directly to their canonical result paths; no prior metric copies are kept in the repository.

## 6. Final-evaluation guard

The safe notebook defaults are:

```python
RUN_FINAL_EVALUATION = False
OVERWRITE_TEST_PREDICTIONS = False
```

A future inference run needs explicit `RUN_FINAL_EVALUATION=True`. Before test data or models are
opened, code requires all 24 seed and 8 ensemble threshold pairs from canonical validation results.
Existing prediction files are protected unless the separate overwrite option is explicitly enabled.
Review mode loads only small saved reports and does not construct an adapter, load a model,
touch CUDA, or open the test dataset.

In the current pipeline, generator benchmarking and selection use only training and validation
data. The test split is used for final classifier evaluation.

## 7. Outputs and reproduction

Canonical benchmark reports are small files under `results/2_diffusers/benchmark/`. Runtime
embeddings, per-image tables, model caches, and diagnostic panels are ignored. Classifier seed results are under
`results/3_classifiers/seed_runs/`; validation ensembles under
`results/3_classifiers/validation_ensembles/`; corrected held-out reports under
`results/4_final_evaluation/`.

Use `rebuild_generator_ranking.py` to regenerate the ranking from the required canonical
`generator_summary.csv`; it does not reconstruct or modify the summary. Use
`regenerate_classifier_metrics.py` for classifier reports from saved predictions. Neither script
loads images or models. A full generator rerun remains a manual, explicit action.

Hashes are used only for duplicate detection, cache validation, and resume safety.
