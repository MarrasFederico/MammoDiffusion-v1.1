# MammoDiffusion — consolidated protocol

Single reference for the experimental design, the generator benchmark and its post-benchmark
amendment, the G02/G07 selection, the downstream 2 × 4 × 3 protocol, the held-out final evaluation,
and manual execution. It consolidates the previously fragmented notes under `docs/`.
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

Notebook 01 has two intentionally different modes, controlled by the three explicit Boolean flags in
its first code cell. The repository is currently prepared for the final real rerun:
`RUN_REAL_BENCHMARK = True`, `REFRESH_CANDIDATE_AUDIT = True` and
`BUILD_CANONICAL_PROVENANCE = False`. Run All therefore refreshes the candidate audit, opens the
declared validation and synthetic images, loads or content-validates the feature cache, recomputes the
metrics and overwrites the canonical output tables. The provenance rebuild remains disabled because
the signed runtime identities have already been rebuilt and verified.

For an audit-only inspection, temporarily set `RUN_REAL_BENCHMARK = False` and
`REFRESH_CANDIDATE_AUDIT = False`: the notebook then loads protocol, registry, roles and provenance,
but does not open validation images, load feature encoders, recompute embeddings or overwrite metric
tables. A clean audit-only execution does **not** mean that the numerical benchmark was recomputed.

Real mode additionally requires a local torchvision `Inception_V3_Weights.IMAGENET1K_V1` checkpoint,
a complete local `microsoft/rad-dino` snapshot and CUDA. The checked-in execution cell currently names
the verified local files and the physical RTX 5060 Ti UUID before any framework import. Downloads are
forbidden inside the notebook. Encoder files, preprocessing and identities are hashed, and each
extractor must return identical finite features on two preflight passes before the benchmark starts.
Set `BUILD_CANONICAL_PROVENANCE = True` only when a declared checkpoint, latent manifest, generation
manifest, sample pool or filter mapping has changed; rebuilding changes signed identities and requires
a fresh selection save.

### 2.2 Notebook 01 cell flow

| Section | What it does | Main input/output |
|---|---|---|
| Protocol configuration | Freezes pool size, sampling, feature spaces, metric definitions, gates and ranking hierarchy. | `configs/generator_benchmark_protocol.json` |
| Candidate discovery | Resolves generator family and role, hashes provenance and model/generation identity, checks manifests and forbidden paths. | registry + provenance manifests → candidate audit |
| Candidate eligibility | Separates primary candidates from sampling ablations, descriptive baselines and generation-pool ablations. | candidate audit + role policy |
| RAW/FILTERED counts | Verifies each declared pool and keeps the two representations separate. | signed per-image manifests |
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

Canonical publication-v2 identity is `(sample_id, project-relative path, SHA-256)` — basenames are
never sufficient. Per-image provenance CSVs are local, git-ignored, regenerable runtime artifacts under
`results/2_diffusers/provenance/runtime/` (the shared 3,061-row train corpus once under
`runtime/shared/`); the repository publishes only the schema, compact v2 index, project-relative
records, G06 refusal diagnostic and documentary candidate audit. Runtime efficiency fields are
imported only when explicitly recorded with verified duration semantics; otherwise `unavailable`.

### 2.4 Notebook 01 outputs

The canonical root is `results/2_diffusers/benchmark/`. `candidate_audit.csv` records
roles and provenance blockers; `technical_validity.csv` records RAW/FILTERED integrity;
`distribution_metrics_repetitions.csv` and `distribution_metrics_summary.csv` contain the shared
resampling results; `diversity_metrics.csv`, `synthetic_duplication.csv`, `train_memorization.csv` and
`validation_similarity.csv` keep the four distinct similarity questions separate. The publication
summary is `generator_summary_corrected.csv`, which preserves invalid legacy timing evidence as
`unavailable_invalid_duration_semantics`. Gate calibration, amendment evidence and corrected rankings
are under `gate_audit/`.

## 3. Post-benchmark gate calibration audit

A methodological audit (`notebooks/utility/gate_audit.py`, `run_gate_audit.py`,
`run_gate_amendment.py`; runtime artifacts git-ignored under
`results/2_diffusers/benchmark/gate_audit/`) ran **after** the benchmark. It never loads
an encoder, re-extracts embeddings, re-runs KID/FID/PRDC, selects a generator, trains a classifier, or
reads the test split; it only measures, freezes and proposes.

- **Order-independent perceptual hash:** `perceptual_hash_cluster_diagnostics`
  (`phash_any_neighbour_rate`, `phash_component_excess_rate`) and `confirmed_duplicate_analysis` using
  the preregistered rule `exact | (pHash ≤ 2 AND SSIM ≥ 0.98)`. The deprecated order-dependent rate is
  kept only for comparison.
- **Real-vs-real baselines:** `repeated_prdc_baseline` / `split_half_prdc_baseline` from cached
  embeddings.
- **Strict efficiency:** durations are trusted only with verified semantics.

**Findings (frozen run at HEAD `cd05886c`):** five official candidates (G02, G03, G04, G07, G08 —
FILTERED) measured; **zero eligible** under the original gates. `confirmed_duplicate_rate` is 0.0 for
all five — the flagged pairs are pHash-near pairs not confirmed as duplicates (low SSIM, large RAD-DINO
distance), a descriptive signal of structural similarity, not mere hash collisions. Perceptual-hash
any-neighbour/component-excess rates depend on pool size and must be read size-matched (matched to
n=73 the candidates fall to ≈ 0.004–0.018 vs full-pool 0.06–0.17; real validation ≈ 0.0, real train
n=340 ≈ 0.006). RAD-DINO coverage from a single 73-image balanced point is unstable; real-vs-real
baselines (≈ 0.96 train-vs-validation, ≈ 0.98 validation split-half) are an upper/reference benchmark,
**not a directly transferable synthetic eligibility threshold** — best candidate G07 ≈ 0.42 (mean
≈ 0.458, fraction ≥ 0.5 ≈ 0.375). Gate-independent descriptive ranking: fine-tuned G02 → G03 → G04;
from-scratch G07 → G08. Efficiency: the G02/G03/G04 manifests imply physically impossible
microsecond-per-image durations with no `duration_semantics`, so `generation_seconds_per_image` is
`unavailable_invalid_duration_semantics` and unverified energy/VRAM are dropped (checkpoint size kept).

## 4. Amendment v1 (Option B), approved post-benchmark

- **Status:** approved (human), transparent post-benchmark amendment. **Approval date:** 2026-07-15.
- **Machine-readable record:** [`configs/generator_benchmark_protocol_amendment_v1.json`](../configs/generator_benchmark_protocol_amendment_v1.json);
  portable evidence in [`configs/generator_selection_evidence_v1.json`](../configs/generator_selection_evidence_v1.json).

The amendment is **not** presented as preregistered. The original gates
(`maximum_perceptual_duplicate_rate = 0.02`, `minimum_rad_dino_coverage = 0.5`) and the original
**zero-eligible** outcome are preserved verbatim (frozen `original_protocol.json`; retained in the
protocol's `eligibility_gates`). pHash-only is order-dependent and does not establish duplication, and
73-image balanced-point coverage is unstable and was not derived from data; both become
**descriptive/ranking metrics, not binary gates**, and **no new coverage threshold is introduced.**

**Option B blocking safety gates:** minimum valid positive images (1361); maximum exact duplicate rate
(0.01); maximum confirmed duplicate rate (0.02); maximum train memorization rate (0.01); maximum
corrupted file rate (0.0); valid filter manifest; complete RAW/FILTERED mapping; complete metrics;
lineage; provenance; test access forbidden. The KID-primary selection hierarchy is unchanged. Under
Option B all five official candidates pass the safety gates and the descriptive ranking is fine-tuned
G02 → G03 → G04 and from-scratch G07 → G08. The internal historical test was **not** opened at any
point. Any downstream result must therefore be interpreted with this stated limitation: the generators
were selected under the approved amendment, not the original preregistered thresholds.

## 5. Selection

`notebooks/3_generator_benchmark/02_Generator_Selection.ipynb` is deliberately lightweight. It loads
no encoder and reads no image pixels. It prefers `generator_summary_corrected.csv`, displays the
original zero-eligible outcome beside the approved Option B outcome, shows the unchanged KID-primary
family rankings and paired KID differences, and checks that the two explicit manual constants equal
the amended top-ranked candidates. With `SAVE_SELECTION=True`, it refuses an inconsistent choice and
writes the downstream contract.

- **Fine-tuned:** G02 — `02_sd21_filtered_100steps`.
- **From-scratch:** G07 — `07_ldm_sdvae_extra1361`.

`configs/selected_generators.json` is content-aware (schema 2): it binds the benchmark identity, the
active amendment, the committed selection evidence, and each generator's model/generation identity and
FILTERED manifest (path, SHA-256, 1361 records) so silent edits are detected. Classifier synthetic
conditions consume exactly those signed 1,361 FILTERED records (no directory scan or resampling).
The file also records that the amendment is post-benchmark and that the test split was not accessed.

## 6. Classifier design (2 × 4 × 3)

Fixed primary design: architectures **MaxViT-512** and **Mammo-FM**; conditions `real_only`,
`real_augmented`, `real_plus_best_finetuned_positive`, `real_plus_best_fromscratch_positive`; seeds
17, 42, 73 — exactly 24 logical experiments and 8 three-seed validation ensembles. RAD-DINO is not a
downstream classifier; it is only a feature extractor in the generative benchmark.

Within an architecture, every condition shares at most 6,400 optimizer updates, the same effective
batch size, loss, class weighting, online real augmentation, validation manifest and frequency,
scheduler, early stopping and checkpoint criterion. The primary checkpoint metric is validation
PR-AUC; an equal PR-AUC uses lower validation loss, then the earlier epoch; ROC-AUC never selects the
checkpoint. No hidden oversampling; samples seen per source, optimizer updates and effective epochs
are reported. Each run writes under
`results/3_classifiers/seed_runs/<architecture>/<condition>/seed_<seed>/` (`configuration.json`,
`dataset_summary.json`, checkpoint/resume, `training_history.csv`, `validation_predictions.csv`,
`validation_metrics.json`, `interpretability/` when produced).

**Inference and statistics.** The validation notebook reports patient-level PR-AUC, ROC-AUC, Brier,
ECE, sensitivity, specificity, balanced accuracy, bootstrap intervals, seed mean ± SD and
mean-probability ensembles (all three seeds required, identical patient/image keys and labels, no
duplicates/missing, finite probabilities, same validation manifest). The eight declared PR-AUC
comparisons form one Holm family; any additional comparison is exploratory.

## 7. Final evaluation on the held-out test set

Final evaluation runs once on the held-out test set (`data/processed/test/`). Every decision — the
selected checkpoints, the decision thresholds and the eight preregistered comparisons — is fixed on
validation before any test access, so the test set contributes no model selection. Primary inference
uses patient-level bootstrap and Holm correction over the eight declared comparisons; any additional
analysis is exploratory.

Final evaluation uses a visible opt-in flag (`RUN_TEST_INFERENCE`) — no cryptographic lock, Git
revision gate or one-shot enforcement. No model or threshold selection may occur after final
evaluation begins.

## 8. Manual execution

No script launches or replaces a scientific notebook.

### 8.1 Final benchmark reproduction

This refers specifically to
`notebooks/3_generator_benchmark/01_Unified_Generator_Benchmark.ipynb`, not to the Diffusers or
classifier notebook that also has the local number `01`. Its first code cell is already configured for
the final run with the verified local InceptionV3 checkpoint, complete RAD-DINO snapshot and physical
RTX 5060 Ti UUID. No shell exports are required on this workstation.

Restart the Jupyter kernel, then Run All from the first cell. The restart matters because
`CUDA_VISIBLE_DEVICES` is assigned before importing PyTorch. A content-validated embedding
cache hit is a valid real execution: the notebook still opens the declared real/synthetic inputs,
validates identities and records the cache decision. Do not delete a valid cache merely to force model
loading. Keep `BUILD_CANONICAL_PROVENANCE = False` for this rerun; set it to `True` only when the
candidate audit reports a changed runtime component. Rebuilding provenance changes signed identities
and must be followed by a fresh selection save.

After Notebook 01 completes without deferred cells or errors, refresh the derived post-benchmark
artifacts in this order:

```bash
python notebooks/utility/run_gate_audit.py
python notebooks/utility/run_gate_amendment.py
python notebooks/utility/correct_efficiency_summary.py
```

Then Run All
`notebooks/3_generator_benchmark/02_Generator_Selection.ipynb` with `SAVE_SELECTION=True`, and require
the final hand-off audit to pass:

```bash
python notebooks/utility/classifier_preflight.py
```

If the real rerun changes a frozen benchmark/evidence hash, selection must stop for explicit review;
the mismatch must not be bypassed or silently replaced.

1. Run generator notebooks only when candidate outputs are missing.
2. Execute `notebooks/3_generator_benchmark/01_Unified_Generator_Benchmark.ipynb`; review RAW/FILTERED tables, repeated-subsampling
   intervals, duplication, train memorization, validation similarity and efficiency.
3. In `notebooks/3_generator_benchmark/02_Generator_Selection.ipynb`, review the original and amended outcomes and save the selection.
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

## 9. Mammo-FM license

Mammo-FM weights are governed by a **Custom Academic License for Model Weights** (non-commercial
academic use only; no redistribution of weights or derivatives; no distillation). The repository must
not contain Mammo-FM checkpoints. See [`docs/mammo_fm_license_note.md`](mammo_fm_license_note.md) for
the full terms, the required acknowledgment and the citation.
