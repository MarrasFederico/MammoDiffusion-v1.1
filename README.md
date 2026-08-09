<p align="center">
  <img src="assets/logo_MammoDiffusion.png" alt="MammoDiffusion Logo" width="180"/>
</p>

# MammoDiffusion v1.1

MammoDiffusion v1.1.0 is the final release of the project's **whole-image mammography synthesis**
study. It provides a reproducible notebook-first pipeline for generating 512×512 grayscale MLO
mammograms, selecting synthetic pools, and measuring their downstream effect in a fixed
2-architecture × 4-condition × 3-seed cancer-classification experiment.

The central result is promising but not conclusive. Adding positives from the selected
from-scratch generator (G07) produced a favorable PR-AUC signal with MaxViT-512, but the signal did
not survive adjustment across the eight configured primary comparisons and did not reappear with
Mammo-FM. The evidence therefore does **not** show that whole-image synthetic mammograms provide
classifier-independent, generalizable pathological information.

> MammoDiffusion v1.1 demonstrates a reproducible whole-image mammography synthesis and downstream
> evaluation pipeline and identifies a promising but architecture-dependent augmentation signal.
> The lack of robust cross-classifier replication motivates a lesion-aware approach designed to
> preserve real target-domain anatomy while synthesizing only pathological content.

## Release and naming

Historical notebooks, artifacts, and `protocol_version = 2` refer to an internal methodological
phase called **V2**, which followed the original internal V1 experiments. Those identifiers are
preserved because they are part of the provenance. The public release that freezes the complete
whole-image study is **MammoDiffusion v1.1.0**; the historical internal V2 label must not be confused
with a separate lesion-aware successor project.

The history of the two internal phases is documented in
[docs/FROM_V1_TO_V2.md](docs/FROM_V1_TO_V2.md). Release and archive boundaries are documented in
[docs/ARCHIVE_AND_RELEASE.md](docs/ARCHIVE_AND_RELEASE.md).

## Task and cohort

The downstream endpoint is the RSNA competition `cancer` label: **cancer-positive versus
non-cancer**, not “malignant finding versus no lesion.” A `cancer = 0` image is not guaranteed to be
free of every benign or suspicious finding.

The analytical cohort contains 2,916 patients and one selected MLO image per patient: 486
cancer-positive and 2,430 non-cancer cases. This 16.7% positive fraction is **constructed**, not an
estimate of screening prevalence: preprocessing retains the selected positive patients and caps
the non-cancer-to-cancer ratio at 5:1. The patient-level splits contain 2,041/437/438 patients for
train/validation/test, including 340/73/73 positive patients.

The same canonical split was retained across the historical internal V1 and V2 phases. The current
pipeline prevents generator selection and threshold optimization from using test data, but the test
cohort is a historically reused project test set, not an independent external confirmation cohort.

## Canonical downstream results

The point estimates below are patient-level mean-probability ensembles of seeds 17, 42, and 73 on
the 438-patient test cohort. The historical field `pr_auc` is average precision (stepwise precision
times the increase in recall), not trapezoidal PR-curve area. G02 is the selected fine-tuned
generator and G07 the selected from-scratch generator.

| architecture | real only | traditional augmentation | real + G02 positives | real + G07 positives |
|---|---:|---:|---:|---:|
| MaxViT-512 | 0.4128 | 0.4497 | 0.4682 | **0.5230** |
| Mammo-FM | 0.3241 | 0.3129 | 0.3232 | 0.3161 |

For MaxViT-512, the G07-minus-real-only point difference is +0.1102. The paired stratified
bootstrap mean difference is +0.1052, with a 95% percentile interval of [+0.0247, +0.1827], an
empirical two-sided tail area of 0.011, and a Holm-adjusted value of 0.088. For Mammo-FM, the
corresponding point difference is −0.0079 and the bootstrap mean difference is −0.0083, with a 95%
interval of [−0.0529, +0.0342], tail area 0.708, and Holm-adjusted value 1.000. No comparison in the
eight-comparison family is rejected at 0.05 after Holm adjustment.

These empirical tail areas are computed as twice the fraction of paired, class-stratified bootstrap
differences on the opposite side of zero from the observed direction. They are useful descriptive
uncertainty summaries, but they are not permutation p-values or bootstrap-null test p-values. The
study was not formally preregistered; “primary” denotes the comparison family encoded in the frozen
repository protocol. See [docs/DISCUSSION.md](docs/DISCUSSION.md) for the full interpretation.

## Research questions

**RQ1 — Generator quality and selection.** Which eligible fine-tuned generator and which eligible
from-scratch generator best satisfy the declared distribution, diversity, duplicate, and detected
memorization criteria? `notebooks/3_generator_benchmark/` applies a RAD-DINO-KID-first ranking and
selects G02 and G07. Eligibility and ranking are project-level technical/descriptive rules; they do
not certify lesion presence, radiological validity, or clinical realism.

**RQ2 — Downstream impact.** Does adding selected synthetic positives change cancer classification
relative to real-only training and traditional augmentation? The favorable MaxViT-specific G07
signal is not sufficient to establish a general benefit after multiplicity adjustment and
cross-architecture comparison.

**RQ3 — Generator family.** Within each architecture, do the selected fine-tuned and from-scratch
conditions differ in downstream usefulness? The G07-minus-G02 bootstrap mean difference is +0.0530
for MaxViT-512 (95% interval [−0.0067, +0.1149], tail area 0.089, Holm-adjusted 0.623) and −0.0087
for Mammo-FM (95% interval [−0.0646, +0.0408], tail area 0.755, Holm-adjusted 1.000). Neither
comparison supports a clear family-level distinction.

**RQ4 — Cross-architecture consistency.** Does an augmentation signal observed with one classifier
representation appear with the other? G07 improves the MaxViT point estimate but not the Mammo-FM
point estimate. This is an architecture-dependent pattern on the same cohort; no formal
architecture-by-condition interaction test or independent replication study was performed.

**RQ5 — Computational cost.** What estimated energy did the generator workflows require? The
secondary analysis in `notebooks/5_sustainability/` uses elapsed time from a frozen event registry
and a 0.170 kW single-GPU assumption. It contains no classifier-training events, is not a
wall-socket or carbon measurement, and does not enter generator selection.

## Dataset workflows

There are two distinct ways to reproduce the data setup.

### A. Reuse the processed cohort

This is the normal analysis workflow and does not require the original source archive. Provide:

```text
data/processed/
├── train/{0,1}/*.png
├── val/{0,1}/*.png
├── test/{0,1}/*.png
└── metadata/
    ├── all_processed.csv
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Each metadata row needs at least `patient_id`, `image_id`, `label`, `split`, and
`processed_path`. `notebooks/utility/processed_dataset_reuse.py` checks the schema, image presence,
binary labels, unique output paths, reconciliation of split CSVs, and patient separation.
Historical manifests and result records may contain absolute paths from the original workstation;
the reuse utilities reroot recognized repository-relative suffixes, but the old strings remain in
frozen artifacts as provenance.

### B. Rebuild from original RSNA data

The source is the derived Kaggle release
[RSNA Breast Cancer 512 PNGs](https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-512-pngs),
which distributes the RSNA Screening Mammography Breast Cancer Detection images as PNG files with
the competition `train.csv` metadata. Obtain it under the source terms and place the user-supplied
archive where the preprocessing notebook requests it. The repository does not perform an opaque
automatic download and does not read DICOM.

`notebooks/1_preprocessing/01_Preprocessing_RSNA_512_gray_MLO.ipynb` keeps MLO images, derives
patient status, excludes nominally negative images from positive patients, selects at most one
image per patient, applies the 5:1 cap, normalizes visual tissue side and intensity, creates
single-channel 512×512 images with aspect-preserving padding, and writes patient-level splits.

## Reproduction workflow

1. Validate or reconstruct the processed cohort.
2. Run generator notebooks only when local checkpoints or pools must genuinely be rebuilt.
3. Review `notebooks/3_generator_benchmark/01_Unified_Generator_Benchmark.ipynb` with its safe
   default `RUN_REAL_BENCHMARK = False`.
4. Enable the real benchmark only when data, pools, local encoders, and a compatible GPU are present.
5. Review the G02/G07 selection in `02_Generator_Selection.ipynb` and
   `configs/selected_generators.json`.
6. Run MaxViT-512 and Mammo-FM for four conditions and seeds 17, 42, and 73 when retraining is
   intentionally requested.
7. Build validation ensembles and freeze the Youden threshold and the threshold chosen against a
   nominal validation specificity target of 0.90.
8. Keep `RUN_FINAL_EVALUATION = False` while reviewing saved outputs. Replacing test predictions
   requires both explicit inference opt-in and `OVERWRITE_TEST_PREDICTIONS = True`.

No test-mode function may choose a threshold. The nominal 0.90 operating-point threshold is frozen
from validation; its achieved specificity on test is reported and is not guaranteed to equal or
exceed 0.90 after transfer to the test cohort.

## Generator benchmark and selection

The canonical synthetic target is 1,361 images. RAW and FILTERED representations remain separate.
The benchmark reports KID, descriptive FID, PRDC, diversity, train-reference memorization,
validation similarity, exact duplication, and perceptual-duplicate diagnostics. RAD-DINO KID is
the primary ranking metric. RAD-DINO coverage and perceptual-hash duplicate rate are descriptive
reference values rather than eligibility gates.

The authoritative selection is:

- G02 (`02_sd21_filtered_100steps`), fine-tuned family;
- G07 (`07_ldm_sdvae_extra1361`), from-scratch family.

Generator selection uses training and validation data only. Passing the technical gates means only
that a candidate can enter this project's ranking under its declared thresholds; it is not evidence
that a positive image contains a valid localized cancer finding or that a non-cancer image is
lesion-free. Efficiency is reported separately and is not a ranking field.

Small, versionable reports live in `results/2_diffusers/benchmark/`. The deterministic
`notebooks/utility/rebuild_generator_ranking.py` rebuilds the ranking from the canonical
`generator_summary.csv` without opening images or models.

## Repository and archive boundary

- `configs/`: executable study protocols, generator registry, and G02/G07 selection;
- `notebooks/`: preprocessing, generators, benchmark, classifiers, and reusable utilities;
- `results/`: versioned scientific reports, saved predictions, and final figures;
- `tests/`: model-free fixtures and static protocol checks;
- `data/`: local real and synthetic image pools, excluded from Git;
- `experiments/`: local checkpoints, latents, resume state, and heavy artifacts, excluded from Git.

A Git clone is sufficient to inspect the protocol, run the lightweight tests, and regenerate
classifier reports from committed predictions. It is not sufficient for full training or image
generation: the image data, checkpoints, local encoders, shared SD2.1 assets, and authorized
Mammo-FM weights must be restored separately. Mammo-FM weights and derivatives must not be
redistributed; see [docs/mammo_fm_license_note.md](docs/mammo_fm_license_note.md).

Rebuild all derived validation and test classifier reports, in dependency order, without inference:

```bash
python notebooks/utility/rebuild_classifier_reports.py --root .
```

## Validation

Run from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q notebooks/utility tests assets/mammodiffusion_gradio
python -m unittest discover -s tests -p 'test_*.py' -v
python -m pytest -q
```

The lightweight suite does not train, generate, perform classifier inference, open the scientific
image cohort, or require a GPU. Full scientific limitations are listed in
[docs/LIMITATIONS.md](docs/LIMITATIONS.md).

Further details: [protocol](docs/PROTOCOL.md), [discussion](docs/DISCUSSION.md),
[generator status](docs/GENERATOR_STATUS.md), [sustainability analysis](docs/SUSTAINABILITY_ANALYSIS.md),
[test suite](docs/TESTS.md), [shared assets](docs/SHARED_ASSETS.md), and
[archive/release policy](docs/ARCHIVE_AND_RELEASE.md).
