# MammoDiffusion — consolidated protocol

Single reference for the experimental design, the generator benchmark, the G02/G07 selection, the
downstream 2 × 4 × 3 protocol, the held-out final evaluation, and manual execution. It consolidates
the previously fragmented notes under `docs/`.

The processed corpus is the starting point of this repository: preprocessing begins from the
already-converted **512×512 grayscale MLO PNG** dataset under `data/processed/`. The original RSNA
DICOM archive is **not** part of this project and is not required to reproduce it; the preprocessing
notebook verifies the processed corpus (schema, splits, patient separation, image presence) rather
than re-decoding DICOMs.
The Mammo-FM academic-license terms are kept separately in
[`docs/mammo_fm_license_note.md`](mammo_fm_license_note.md); shared SD2.1/Diffusers asset identities in
[`docs/SHARED_ASSETS.md`](SHARED_ASSETS.md); the sustainability event schema in
[`docs/SUSTAINABILITY_ANALYSIS.md`](SUSTAINABILITY_ANALYSIS.md); the per-generator status table in
[`docs/GENERATOR_STATUS.md`](GENERATOR_STATUS.md).

## 1. Research questions

- **RQ1** compares eligible generators on validation-only fidelity, diversity, coverage, efficiency,
  duplication and train memorization.
- **RQ2** compares real-only training, traditional augmentation, and positive synthetic augmentation.
- **RQ3** tests whether the downstream effect is consistent across MaxViT-512 and Mammo-FM.

Generator choice is validation-only; downstream test performance cannot change the choice.

## 2. Generator benchmark

Open and run `notebooks/3_generator_benchmark/01_Unified_Generator_Benchmark.ipynb`; it is the
canonical interface and no CLI wrapper is required.

### 2.1 Notebook 01 execution modes

Notebook 01 has two intentionally different modes, controlled by two explicit Boolean flags in its
first code cell. The repository is currently prepared for the final real rerun:
`RUN_REAL_BENCHMARK = True` and `REFRESH_CANDIDATE_AUDIT = True`. Run All therefore refreshes the
candidate audit, opens the declared validation and synthetic images, loads or content-validates the
feature cache, recomputes the metrics and overwrites the canonical output tables.

For an audit-only inspection, temporarily set `RUN_REAL_BENCHMARK = False` and
`REFRESH_CANDIDATE_AUDIT = False`: the notebook then loads protocol, registry and roles, but does not
open validation images, load feature encoders, recompute embeddings or overwrite metric tables. A
clean audit-only execution does **not** mean that the numerical benchmark was recomputed.

Real mode additionally requires a local torchvision `Inception_V3_Weights.IMAGENET1K_V1` checkpoint,
a complete local `microsoft/rad-dino` snapshot and CUDA. The checked-in execution cell resolves encoder
paths from portable cache/project defaults (with explicit environment overrides) and selects the GPU at
runtime through `MAMMODIFFUSION_BENCHMARK_GPU` or the automatic maximum-memory policy before any framework
import. Downloads are forbidden inside the notebook. Encoder identities are hashed only to invalidate the
embedding cache, and each extractor must return identical finite features on two preflight passes before
the benchmark starts.

### 2.2 Notebook 01 cell flow

| Section | What it does | Main input/output |
|---|---|---|
| Protocol configuration | Freezes pool size, sampling, feature spaces, metric definitions, gates and ranking hierarchy. | `configs/generator_benchmark_protocol.json` |
| Candidate discovery | Resolves generator family and role and checks that the registered pools exist with the target image count and no forbidden paths. | registry → candidate audit |
| Candidate eligibility | Separates primary candidates from sampling ablations, descriptive baselines and generation-pool ablations. | candidate audit + role policy |
| RAW/FILTERED counts | Verifies each declared pool directory and keeps the two representations separate. | registered pool directories |
| Real reference set | Loads all positive validation records only; test and historical-test paths are forbidden. | `data/processed/metadata/val.csv` |
| Technical validity | Measures readability, expected shape, finite range, near-black/constant images, uniqueness and exact duplication. | `technical_validity.csv` |
| Feature extraction | Extracts or content-validates cached InceptionV3 and RAD-DINO embeddings. | `embedding_cache/` |
| Distribution metrics | Computes full-pool KID/FID, balanced PRDC and repeated-subsampling stability. | repetitions + summary CSVs |
| Diversity/duplication | Computes MS-SSIM diversity, synthetic nearest neighbours and exact/confirmed duplicate evidence. | diversity and duplication CSVs |
| Train memorization | Compares every synthetic image with the complete training corpus declared by that generator. This is the only similarity analysis allowed to gate memorization. | `train_memorization.csv` |
| Validation similarity | Finds the nearest real validation positive for descriptive inspection; it is never called memorization. | `validation_similarity.csv` |
| Summary/ranking | Joins integrity, metrics, safety and valid efficiency evidence, then ranks FILTERED primary candidates within each family. | generator summary/ranking + figures |
| Diagnostic panels | Renders deterministic closest, median and farthest train/validation/synthetic neighbours. | `diagnostic_panels/` |

### 2.3 Metric design and interpretation

- `synthetic_pool_target = 1361` is the minimum valid synthetic pool per eligible candidate; each
  candidate provides a uniform 1,361 positive images.
- `real_reference_count` is every positive image in `data/processed/metadata/val.csv`; real images are
  never duplicated to reach 1,361.
- `kid_full_pool` (primary) and descriptive `fid_full_pool` compare every validation positive against
  the canonical 1,361-image synthetic pool. Unequal group sizes are intentional; the small real pool
  is an explicit FID caveat, so FID is secondary and descriptive (one repetition by default).
- `precision/recall/density/coverage_balanced_point` use every validation positive against a
  deterministic synthetic subset of the same size.
- Stability uses `floor(0.8 × min(real_reference_count, synthetic_pool_count))`, which must exceed the
  PRDC neighbour count; a shared plan is recorded in `resampling_plan.json`. KID uses 200 repeated
  balanced subsets, PRDC 100, both `replace=False`. Repeated balanced results are explicitly called
  **repeated-subsampling stability intervals**.

Outputs are separated into `full_pool_distribution_estimates`, `balanced_prdc_point_estimates` and
`stability_estimates`. RAW and FILTERED results are kept separate. Frozen InceptionV3 and RAD-DINO
embeddings are cached once per generator × representation × extractor, keyed on ordered image IDs,
path/size/SHA-256 fingerprints, full encoder identity, preprocessing, feature dimension, code version,
metadata-CSV hash and source-manifest hash. Missing local weights defer execution; no download occurs.

Metric directions are explicit: KID and FID are lower-is-better; precision measures how much
synthetic content lies in real neighbourhoods; recall and coverage measure how much of the real
validation distribution is represented; density measures synthetic concentration in real
neighbourhoods. RAD-DINO KID is primary. FID is secondary because the validation-positive reference
pool is small. `ms_ssim_diversity = 1 - mean_ms_ssim`, so larger values indicate less pairwise visual
similarity under that diagnostic.

**Representation-aware technical validity.** Feature extractability (readable, expected shape,
non-empty finite numeric array) and image quality (additionally not near-black or constant-range) are
reported separately. RAW quality defects are retained as warnings so RAW metrics reflect authentic
generator failures; FILTERED is the official-ranking representation and its eligibility is strict
(corrupt/wrong-shape/near-black/constant-range fatal; ≥ 1,361 quality-valid unique images and the
duplicate-rate gate required). A non-finite feature blocks only the affected representation, with
sample ID, path, extractor and cause; the image is never silently removed. The benchmark stops before
feature loading only when no official FILTERED candidate remains in an entire scientific family.

**Similarity analyses.** Train memorization (synthetic → the complete declared training corpus,
including negatives/augmentations) is the *only* analysis that can gate memorization. Validation
similarity (synthetic → nearest real validation positive) is descriptive and never called
memorization. Synthetic duplication (synthetic → nearest other synthetic) reports nearest distance and
exact/perceptual duplicate rates. Deterministic panels show closest/median/farthest examples.

**Ranking hierarchy** (KID primary): `raddino_kid` → `raddino_coverage` → `raddino_precision` →
descriptive `raddino_fid` → `inception_kid` → `raddino_kid_std` → `generator_id`.

**Selection roles.** The 50-step SD variant is a `sampling_ablation` of G02 (same model identity,
different generation identity) and is not automatically eligible; the canonical 100-step variant may
be. G05 is a `descriptive_baseline`. G06 has the same U-Net/VAE/latents/architecture as G05 and is a
non-eligible `generation_pool_ablation` (its only difference is the larger RAW/filtering pool); its
invalid per-image mapping is documented, not a blocked primary candidate. The generator-benchmark
selection notebook (`02_Generator_Selection.ipynb`) performs a transparent manual selection and saves
only `configs/selected_generators.json`.

Complete registered pool directories are accepted as the owner-generated inputs after count,
readability, uniqueness, and technical-validity checks; there is no signed per-image manifest, SHA
chain, lineage record or provenance gate. Runtime efficiency fields are imported only when explicitly
recorded with verified duration semantics; otherwise they remain `unavailable`.

### 2.4 Notebook 01 outputs

The canonical root is `results/2_diffusers/benchmark/`. `candidate_audit.csv` records roles and any
blocking reasons; `technical_validity.csv` records RAW/FILTERED integrity;
`distribution_metrics_repetitions.csv` and `distribution_metrics_summary.csv` contain the shared
resampling results; `diversity_metrics.csv`, `synthetic_duplication.csv`, `train_memorization.csv` and
`validation_similarity.csv` keep the four distinct similarity questions separate. The publication
summary is `generator_summary_corrected.csv`, which preserves invalid legacy timing evidence as
`unavailable_invalid_duration_semantics`.

## 3. Eligibility policy

Eligibility depends only on technical/scientific safety gates computed by the benchmark itself:
minimum valid positive images (1361); maximum exact-duplicate rate (0.01); maximum train-memorization
rate (0.01); maximum corrupted-file rate (0.0); complete metrics; forbidden test access; and the
registry role. **Perceptual-hash-only rate and RAD-DINO coverage are descriptive ranking metrics, not
binary gates**: pHash-only is order-dependent and does not by itself establish duplication, and a
single 73-image balanced-point coverage value is unstable. There are **no** provenance, lineage,
filter-mapping-manifest or signed-identity gates, and no protocol amendment mechanism.

Under this policy the five official candidates (G02, G03, G04, G07, G08 — FILTERED) pass the safety
gates, and the preregistered KID-primary hierarchy gives the descriptive ranking fine-tuned
G02 → G03 → G04 and from-scratch G07 → G08. The internal historical test was **not** opened at any
point. Efficiency: the G02/G03/G04 manifests imply physically impossible microsecond-per-image
durations with no `duration_semantics`, so `generation_seconds_per_image` is
`unavailable_invalid_duration_semantics` and unverified energy/VRAM are dropped (checkpoint size kept).

## 4. Selection

`notebooks/3_generator_benchmark/02_Generator_Selection.ipynb` is deliberately lightweight. It loads
no encoder and reads no image pixels. It prefers `generator_summary_corrected.csv`, ranks each family
by the KID-primary hierarchy under the technical safety gates, and shows the paired KID differences.
The proposed top-ranked rows are displayed beside the two explicit manual constants. With
`SAVE_SELECTION=True`, the notebook validates family, registry eligibility, metric completeness, image
count, test access, and the technical gates before writing the downstream contract; the manual
decision remains explicit.

- **Fine-tuned:** G02 — `02_sd21_filtered_100steps`.
- **From-scratch:** G07 — `07_ldm_sdvae_extra1361`.

`configs/selected_generators.json` is the simple, authoritative record of the choice: the two selected
ids, their family, descriptive rank and primary-metric value, and `test_access = false`. It contains
no benchmark git-HEAD, run id, amendment reference, selection-evidence pointer or SHA chain. Classifier
synthetic conditions read the selected generator's canonical FILTERED positive pool directory
(`data/synthetic/<generator_id>/positive`) and verify it holds exactly 1,361 unique, readable,
non-test images — directly, without a signed manifest.

## 5. Classifier design (2 × 4 × 3)

Fixed primary design: architectures **MaxViT-512** and **Mammo-FM**; conditions `real_only`,
`real_augmented`, `real_plus_best_finetuned_positive`, `real_plus_best_fromscratch_positive`; seeds
17, 42, 73 — exactly 24 logical experiments and 8 three-seed validation ensembles. RAD-DINO is not a
downstream classifier; it is only a feature extractor in the generative benchmark.

Within an architecture, every condition shares at most 6,400 optimizer updates, the same effective
batch size, loss, class weighting, online real augmentation, validation manifest and frequency,
scheduler, early stopping and checkpoint criterion. The primary checkpoint metric is validation
PR-AUC; an equal PR-AUC uses lower validation loss, then the earlier epoch; ROC-AUC never selects the
checkpoint. No hidden oversampling; samples seen per source, optimizer updates and effective epochs
are reported. Following the project layout, model files are separated from result tables. Each run
writes its small tables and plots under
`results/3_classifiers/seed_runs/<architecture>/<condition>/seed_<seed>/` (`configuration.json`,
`dataset_summary.json`, `model_summary.json`, `source_accounting.json`, `training_history.csv`,
`validation_predictions.csv`, `validation_metrics.json`, `run_complete.json`), while the model
checkpoints (`checkpoint_best.pt`, the resume `checkpoint_latest`/`checkpoint_previous`/`checkpoint_best`
pickles) and the intermediate interpretability maps go under
`experiments/classifiers/<architecture>/<condition>/seed_<seed>/`. The corresponding presentation-ready
Grad-CAM and Integrated Gradients panels are persisted as PNG files under
`results/3_classifiers/figures/interpretability/<architecture>/<condition>/seed_<seed>/`, rather than
remaining available only as notebook output.

**Inference and statistics.** The validation notebook reports patient-level PR-AUC, ROC-AUC, Brier,
ECE, sensitivity, specificity, balanced accuracy, bootstrap intervals, seed mean ± SD and
mean-probability ensembles (all three seeds required, identical patient/image keys and labels, no
duplicates/missing, finite probabilities, same validation manifest). The eight declared PR-AUC
comparisons form one Holm family; any additional comparison is exploratory.

## 6. Final evaluation on the held-out test set

Final evaluation runs once on the held-out test set (`data/processed/test/`). Every decision — the
selected checkpoints, the decision thresholds and the eight preregistered comparisons — is fixed on
validation before any test access, so the test set contributes no model selection. Primary inference
uses patient-level bootstrap and Holm correction over the eight declared comparisons; any additional
analysis is exploratory.

Final evaluation uses a visible opt-in flag (`RUN_TEST_INFERENCE`) — no cryptographic lock, Git
revision gate or one-shot enforcement. No model or threshold selection may occur after final
evaluation begins.

## 7. Manual execution

No script launches or replaces a scientific notebook.

### 7.1 Final benchmark reproduction

This refers specifically to
`notebooks/3_generator_benchmark/01_Unified_Generator_Benchmark.ipynb`, not to the Diffusers or
classifier notebook that also has the local number `01`. Its first code cell is already configured for
the final run with a verified local InceptionV3 checkpoint and complete RAD-DINO snapshot. It selects
the largest-memory visible host GPU at runtime; no physical GPU UUID is stored in notebook source.
Override the automatic choice with `MAMMODIFFUSION_BENCHMARK_GPU=auto|<physical-index>|GPU-...` when
needed. The Inception checkpoint follows `TORCH_HOME`/`XDG_CACHE_HOME` discovery and can be overridden
with `MAMMODIFFUSION_INCEPTION_CHECKPOINT`; the RAD-DINO snapshot can be overridden with
`MAMMODIFFUSION_RAD_DINO_SNAPSHOT`.

Restart the Jupyter kernel, then Run All from the first cell. The restart matters because the portable
selector resolves and verifies `CUDA_VISIBLE_DEVICES` before importing PyTorch. A content-validated embedding
cache hit is a valid real execution: the notebook still opens the declared real/synthetic inputs,
validates identities and records the cache decision. Do not delete a valid cache merely to force model
loading.

After Notebook 01 completes without deferred cells or errors, refresh the corrected efficiency summary:

```bash
python notebooks/utility/correct_efficiency_summary.py
```

Then Run All
`notebooks/3_generator_benchmark/02_Generator_Selection.ipynb` with `SAVE_SELECTION=True`, and
optionally run the metadata-only hand-off audit:

```bash
python notebooks/utility/classifier_preflight.py
```

1. Run generator notebooks only when candidate outputs are missing.
2. Execute `notebooks/3_generator_benchmark/01_Unified_Generator_Benchmark.ipynb`; review RAW/FILTERED tables, repeated-subsampling
   intervals, duplication, train memorization, validation similarity and efficiency.
3. In `notebooks/3_generator_benchmark/02_Generator_Selection.ipynb`, review the family rankings and save the selection.
4. Run all cells in `notebooks/04_classifiers/01_MaxViT512.ipynb` once. The notebook executes all
   four conditions and seeds 17, 42 and 73 sequentially, with an independent model, optimizer,
   output directory and resumable checkpoint for each of its 12 jobs. Before importing a GPU
   framework, the shared utility
   discovers the physical NVIDIA inventory and deterministically selects the device with the most
   total VRAM (the RTX 5060 Ti on the reference workstation), then masks and verifies its runtime
   UUID. No device index, UUID or shell environment variable is required. Model construction,
   DataLoaders, optimizer, focal loss, AMP, gradient accumulation, training/validation loops,
   scheduler, early stopping and checkpoint selection are explicit notebook cells; Python modules
   are imported there only for reusable architecture, data, metric and atomic-I/O primitives.
5. Run `notebooks/04_classifiers/02_MammoFM.ipynb` once. It executes its 12 jobs with the same
   condition/seed isolation and uses the same portable automatic GPU
   selection and resolves the official Mammo-FM checkpoint strictly from the local Hugging Face
   cache; no shell environment variable or runtime download is required.
6. Execute `notebooks/04_classifiers/03_Validation_Comparison.ipynb` after all 24 prediction files exist.
7. Freeze selections, thresholds and comparisons on validation; identify an honest final-evaluation
   dataset and execute `notebooks/04_classifiers/04_Final_Evaluation_and_Report.ipynb` only when ready.

The 24 logical experiments are the two architectures × the four conditions × seeds 17/42/73. There
are two manual classifier executions because each architecture notebook cycles over all 12 of its
condition/seed jobs; synthetic conditions require `configs/selected_generators.json`. Do not tune
one condition differently and do not consult a final-evaluation split while choosing models or thresholds.

## 8. Mammo-FM license

Mammo-FM weights are governed by a **Custom Academic License for Model Weights** (non-commercial
academic use only; no redistribution of weights or derivatives; no distillation). The repository must
not contain Mammo-FM checkpoints. See [`docs/mammo_fm_license_note.md`](mammo_fm_license_note.md) for
the full terms, the required acknowledgment and the citation.
