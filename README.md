<p align="center">
  <img src="assets/logo_MammoDiffusion.png" alt="MammoDiffusion Logo" width="180"/>
</p>

# MammoDiffusion v2

MammoDiffusion is a notebook-first study of synthetic 512×512 grayscale MLO mammograms and
their downstream use for binary classification: malignant finding versus no lesion. The fixed
classifier design is 2 architectures × 4 training conditions × 3 seeds, for 24 seed runs and
8 mean-probability ensembles.

The active generator benchmark uses real training data for memorization, validation data for
quality and selection, and synthetic pools for synthetic metrics. It never uses the classifier
test split.

## Dataset workflows

There are two distinct ways to reproduce the data setup.

### A. Reuse the processed cohort

This is the normal analysis workflow and does not require the original DICOM archive. Provide:

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
`processed_path`. `notebooks/utility/processed_dataset_reuse.py` checks the schema, image
presence, binary labels, unique output paths, reconciliation of the split CSVs, and patient
separation before reuse. The canonical test cohort contains 438 patients.

### B. Rebuild from original RSNA data

Obtain the data under the terms of the official
[RSNA Screening Mammography Breast Cancer Detection competition](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data)
and place the user-supplied archive where the preprocessing notebook requests it. The repository
does not perform an opaque automatic download.

`notebooks/1_preprocessing/01_Preprocessing_RSNA_512_gray_MLO.ipynb` filters to MLO views,
selects one image per patient, handles laterality and visual-side normalization, rescales the
tissue intensity, converts DICOM to grayscale PNG, resizes to 512×512, and writes patient-level
train/validation/test splits. Corrupt inputs are reported rather than silently accepted. A patient
ID may occur in exactly one split.

## Workflow

1. Validate or reconstruct the processed cohort.
2. Run generator notebooks only when their local checkpoints or pools must genuinely be rebuilt.
3. Review `notebooks/3_generator_benchmark/01_Unified_Generator_Benchmark.ipynb`. Its default is
   `RUN_REAL_BENCHMARK = False`, which reads only the saved small CSV/JSON reports.
4. Enable the real benchmark explicitly only when data, pools, local encoders, and GPU are present.
5. Review the G02/G07 selection in `02_Generator_Selection.ipynb` and
   `configs/selected_generators.json`.
6. Run the MaxViT-512 and Mammo-FM notebooks for the four conditions and seeds 17, 42, and 73.
7. Build validation ensembles and freeze both the decision threshold and the operating point for
   target specificity 0.90.
8. Keep `RUN_FINAL_EVALUATION = False` while reviewing saved outputs. A future inference run needs
   explicit opt-in and a separate `OVERWRITE_TEST_PREDICTIONS=True` confirmation before replacing
   existing predictions.

No test-mode function may choose a threshold. Seed test reports read their own validation
thresholds; test ensembles read the matching validation-ensemble thresholds. The same frozen
operating points are used in every patient-bootstrap replica.

## Generator benchmark and selection

The canonical synthetic target is 1,361 images. RAW and FILTERED representations stay separate.
The benchmark reports KID, descriptive FID, PRDC, diversity, train memorization, validation
similarity, exact duplication, and perceptual-duplicate diagnostics. RAD-DINO KID is the primary
ranking metric; coverage and perceptual-hash duplicate rate are descriptive, not eligibility gates.
All ranking metrics must be available for every eligible candidate, so a missing optional value
cannot become a hidden worst-value penalty.

The authoritative selection remains:

- G02 (`02_sd21_filtered_100steps`), fine-tuned family;
- G07 (`07_ldm_sdvae_extra1361`), from-scratch family.

Memorization builds its reference list in memory from `data/processed/metadata/train.csv` and, when
present, `data/real_augmented/metadata.csv`. It validates train paths, labels, unique samples, and
patient separation from validation without creating another manifest.

Small, versionable benchmark outputs live in `results/2_diffusers/benchmark/`:
`candidate_audit.csv`, `generator_summary.csv`, `generator_ranking.csv`,
`resampling_plan.json`, `paired_generator_differences.csv`, `selection_summary.json`, and the
summary figure. Embeddings, per-image tables, diagnostic panels, and execution records are local
runtime outputs. `notebooks/utility/rebuild_generator_ranking.py` deterministically regenerates the
ranking from the required canonical `generator_summary.csv` without modifying the summary or opening
images or models.

In the current pipeline, generator benchmarking and selection use only training and validation
data. The test split is used for final classifier evaluation.

## Repository layout

- `configs/`: plain protocols, registry, and the G02/G07 selection;
- `data/`: local datasets and synthetic pools;
- `experiments/`: checkpoints, resume state, latents, and other heavy execution artifacts;
- `notebooks/`: preprocessing, generators, benchmark, classifiers, and reusable utilities;
- `results/`: canonical scientific reports for preprocessing, generators, classifiers, final
  evaluation, and sustainability;
- `tests/`: model-free fixture tests and static notebook checks.

Checkpoints and data remain outside Git. Mammo-FM weights and derivatives must not be
redistributed; see [docs/mammo_fm_license_note.md](docs/mammo_fm_license_note.md).

## Validation and reproducibility

Run from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q notebooks/utility tests assets/mammodiffusion_gradio
python -m unittest discover -s tests -p 'test_*.py' -v
python -m pytest -q
```

`pytest.ini` restricts collection to `tests/` and excludes data, experiments, results, caches, and
the vendored Diffusers repository. Hashes are used only for duplicate detection, cache validation,
and resume safety. GPU index/UUID selection is an operational
convenience and is not a scientific compatibility condition.

Further details: [consolidated protocol](docs/PROTOCOL.md),
[generator status](docs/GENERATOR_STATUS.md), [test suite](docs/TESTS.md), and
[shared Diffusers assets](docs/SHARED_ASSETS.md).
