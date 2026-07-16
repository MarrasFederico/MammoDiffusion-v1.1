# MammoDiffusion — consolidated protocol

Single reference for the experimental design, the generator benchmark and its post-benchmark
amendment, the G02/G07 selection, the downstream 2 × 4 × 3 protocol, the historical-test-reuse
limitation, and manual execution. It consolidates the previously fragmented notes under `docs/`.
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

Open and run `notebooks/3_generator_benchmark/05_Unified_Generator_Benchmark.ipynb`; it is the
canonical interface and no CLI wrapper is required.

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
invalid per-image mapping is documented, not a blocked primary candidate. Notebook 06 performs a
transparent manual selection and saves only `configs/selected_generators.json`.

Canonical publication-v2 identity is `(sample_id, project-relative path, SHA-256)` — basenames are
never sufficient. Per-image provenance CSVs are local, git-ignored, regenerable runtime artifacts under
`results/publication_v2/generator_provenance/runtime/` (the shared 3,061-row train corpus once under
`runtime/shared/`); the repository publishes only the schema, compact v2 index, project-relative
records, G06 refusal diagnostic and documentary candidate audit. Runtime efficiency fields are
imported only when explicitly recorded with verified duration semantics; otherwise `unavailable`.

## 3. Post-benchmark gate calibration audit

A methodological audit (`notebooks/utility/gate_audit.py`, `run_gate_audit.py`,
`run_gate_amendment.py`; runtime artifacts git-ignored under
`results/publication_v2/generator_benchmark/gate_audit/`) ran **after** the benchmark. It never loads
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

- **Fine-tuned:** G02 — `02_sd21_filtered_100steps`.
- **From-scratch:** G07 — `07_ldm_sdvae_extra1361`.

`configs/selected_generators.json` is content-aware (schema 2): it binds the benchmark identity, the
active amendment, the committed selection evidence, and each generator's model/generation identity and
FILTERED manifest (path, SHA-256, 1361 records) so silent edits are detected. Downstream synthetic
conditions consume exactly those signed 1,361 FILTERED records (no directory scan or resampling).

## 6. Downstream design (2 × 4 × 3)

Fixed primary design: architectures **MaxViT-512** and **Mammo-FM**; conditions `real_only`,
`real_augmented`, `real_plus_best_finetuned_positive`, `real_plus_best_fromscratch_positive`; seeds
17, 42, 73 — exactly 24 logical experiments and 8 three-seed validation ensembles. RAD-DINO is not a
downstream classifier; ResNet-50 remains historical V1 material.

Within an architecture, every condition shares at most 6,400 optimizer updates, the same effective
batch size, loss, class weighting, online real augmentation, validation manifest and frequency,
scheduler, early stopping and checkpoint criterion. The primary checkpoint metric is validation
PR-AUC; an equal PR-AUC uses lower validation loss, then the earlier epoch; ROC-AUC never selects the
checkpoint. No hidden oversampling; samples seen per source, optimizer updates and effective epochs
are reported. Each run writes under
`results/publication_v2/downstream/<architecture>/<condition>/seed_<seed>/` (`configuration.json`,
`dataset_summary.json`, checkpoint/resume, `training_history.csv`, `validation_predictions.csv`,
`validation_metrics.json`, `interpretability/` when produced).

**Inference and statistics.** The validation notebook reports patient-level PR-AUC, ROC-AUC, Brier,
ECE, sensitivity, specificity, balanced accuracy, bootstrap intervals, seed mean ± SD and
mean-probability ensembles (all three seeds required, identical patient/image keys and labels, no
duplicates/missing, finite probabilities, same validation manifest). The eight declared PR-AUC
comparisons form one Holm family; any additional comparison is exploratory.

## 7. Historical internal test — status and limitation

The previous internal test is a **historically reused internal evaluation set**, not an untouched
internal holdout: the repository contains V1 test metric files, historical MaxViT/ResNet test outputs,
final-test prediction paths and prior coverage tables, which is sufficient evidence that the split has
informed previous analyses. It must be described as a historical internal test / reused internal
evaluation set; results are internal and exploratory and **not an independent external confirmation**,
and it must not be called unopened, untouched, or pristine. This audit did not open new test images,
run inference, or modify any split. For publication, prefer an external dataset or a new untouched
holdout created through an explicit scientific decision.

Final evaluation uses a visible opt-in Boolean, a readiness checklist and a plain JSON protocol
snapshot — no cryptographic lock, Git revision gate or one-shot enforcement. No model or threshold
selection may occur after final evaluation begins.

## 8. Manual execution

No script launches or replaces a scientific notebook.

1. Run generator notebooks only when candidate outputs are missing.
2. Execute `05_Unified_Generator_Benchmark.ipynb`; review RAW/FILTERED tables, repeated-subsampling
   intervals, duplication, train memorization, validation similarity and efficiency.
3. In `06_Generator_Selection.ipynb`, review the original and amended outcomes and save the selection.
4. In `07_MaxViT512_Downstream.ipynb`, set `CONDITION`, `SEED`, `GPU`, `RESUME`, run all cells; repeat
   its 12 combinations. `GPU` accepts a physical nvidia-smi index or a `GPU-…` UUID (resolved to a
   UUID before any framework import; verified by physical identity).
5. Repeat with `08_MammoFM_Downstream.ipynb`.
6. Execute `09_Downstream_Validation_Comparison.ipynb` after all 24 prediction files exist.
7. Freeze selections, thresholds and comparisons on validation; identify an honest final-evaluation
   dataset and execute `10_Final_Evaluation_and_Report.ipynb` only when ready.

The 24 manual combinations are the two architectures × the four conditions × seeds 17/42/73; synthetic
conditions require `configs/selected_generators.json`. Do not tune one condition differently and do not
consult a final-evaluation split while choosing models or thresholds.

## 9. Historical V1 classifiers

ResNet-50 was the central classifier in MammoDiffusion V1. Tracked artifacts under
`experiments/classifiers/resnet50/` are retained as historical evidence, not deleted or rewritten.
They belong to a previous version/protocol, motivated the move to MaxViT-512, are not part of the
24-job V2 protocol or the eight primary comparisons, and must not be merged with publication-v2
prediction tables or any future final evaluation. The full pre-simplification pipeline is recoverable
from Git history and the tag `classifier-matrix-v2-full`.

## 10. Mammo-FM license

Mammo-FM weights are governed by a **Custom Academic License for Model Weights** (non-commercial
academic use only; no redistribution of weights or derivatives; no distillation). The repository must
not contain Mammo-FM checkpoints. See [`docs/mammo_fm_license_note.md`](mammo_fm_license_note.md) for
the full terms, the required acknowledgment and the citation.
