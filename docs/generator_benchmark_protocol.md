# Unified generator benchmark protocol

Open and run `notebooks/3_generator_benchmark/05_Unified_Generator_Benchmark.ipynb`. It is the canonical interface; no CLI wrapper is required.

## Counts and references

- `synthetic_pool_target = 1361` is the minimum valid synthetic pool per eligible candidate.
- `real_reference_count` is every positive image available in `data/processed/metadata/val.csv`.
- `kid_full_pool` (primary) and descriptive `fid_full_pool` use every validation positive against the canonical 1,361-image synthetic pool. Unequal group sizes are intentional; the small real pool is an explicit FID caveat.
- `precision_balanced_point`, `recall_balanced_point`, `density_balanced_point`, and `coverage_balanced_point` use every validation positive against a deterministic synthetic subset of exactly the same size.
- Stability uses `floor(0.8 × min(real_reference_count, synthetic_pool_count))`, which must exceed the PRDC neighbour count.

For example, 1,361 synthetic images and 73 validation positives produce stability repetitions with 58 real and 58 synthetic features. Both subsets therefore vary. Sampling is deterministic and without replacement, and a shared plan is recorded in `resampling_plan.json`.

## Metrics

KID full-pool is primary. Outputs are separated into `full_pool_distribution_estimates`, `balanced_prdc_point_estimates`, and `stability_estimates`; repeated balanced KID/PRDC results are explicitly called repeated-subsampling stability intervals. FID remains descriptive. RAW and FILTERED results remain separate. Top candidates are compared with paired per-repetition KID differences, and practical equivalence uses the protocol-configured margin.

Frozen InceptionV3 and RAD-DINO embeddings are cached only when ordered image IDs, path/size/SHA-256 fingerprints, the complete encoder identity, preprocessing, feature dimension, relevant code version, metadata CSV hash and source-manifest content hash all match. Inception identity includes torchvision version, weights enum, checkpoint filename/hash and transforms. RAD-DINO identity includes repository, resolved local snapshot/commit, config hash, weight-shard composite hash, processor hash and preprocessing. Missing local weights defer execution; no download is performed.

## Similarity analyses

- Train memorization: synthetic → the complete training corpus declared by each generator, including negative and augmented sources where applicable. It reports nearest ID/label/source, embedding distance, SSIM, perceptual hash and an exact-hash check against the entire corpus. Only this analysis can gate memorization.
- Validation similarity: synthetic → nearest real validation positive; descriptive distribution similarity, never called memorization.
- Synthetic duplication: synthetic → nearest other synthetic; reports nearest distance and exact/perceptual duplicate rates.

Panels show closest, median and farthest examples deterministically for all three relations.

## Selection roles

The 50-step Stable Diffusion variant is a `sampling_ablation` of G02 and excluded from automatic downstream eligibility. G01 and G02 have the same complete model identity but different generation identities. The canonical 100-step variant may be eligible. G05 is a `descriptive_baseline`. Full-file hashes establish that G06 has the same U-Net, custom VAE encoder/decoder, latent statistics, latent arrays, architecture and parameterization as G05; it is therefore a non-eligible `generation_pool_ablation` whose only scientific difference is the larger RAW/filtering pool. Its invalid per-image mapping remains documented, but is not a blocked primary candidate. Notebook 06 performs a transparent manual selection and saves only `configs/selected_generators.json`.

Technical validity and filtering acceptance are independent. Acceptance rate is a descriptive selection-pressure/efficiency measure, not a universal exclusion gate and not a direct comparison across deliberately different RAW pool sizes. Eligibility instead requires a valid canonical filter manifest, complete RAW→FILTERED mapping, at least 1,361 valid unique FILTERED images, no corrupt files, complete provenance and a canonical CSV training-corpus manifest. A protocol-specific top-K run passes when its declared target is reached even when acceptance is about 1,361/4,083.

Canonical publication-v2 identity is `(sample_id, project-relative path, SHA-256)`. Basenames are never sufficient. The canonical flat `generator_summary.csv` ranking fields are, in order, `raddino_kid` (RAD-DINO KID full-pool), `raddino_coverage` (balanced point), `raddino_precision` (balanced point), descriptive `raddino_fid` (full-pool), `inception_kid` (full-pool), `raddino_kid_std` (stability standard deviation), and `generator_id`.

Per-image provenance CSVs are local, regenerable runtime artifacts under `results/publication_v2/generator_provenance/runtime/`; the common 3,061-row train corpus is stored once under `runtime/shared/`. They are deliberately excluded from Git because they contain dataset identifiers and thousands of paths. Schema v2 records every model component with a project-relative path and real SHA-256, plus canonical model and generation signatures. The model signature excludes storage paths, so byte-identical copies are deduplicated; the generation signature includes sampling, conditioning, RAW/FILTERED configuration and relevant code signature, but never Git HEAD.

The repository publishes the schema, compact v2 index, project-relative records, G06 refusal diagnostic, and documentary candidate audit. Source-only validation reports `provenance_record_schema_valid`, `provenance_index_consistent`, and whether runtime hashes are declared; it always reports `runtime_manifest_contents_verified = false` and `runtime_assets_verified = false`. The host runtime audit verifies manifest contents and assets before benchmark execution. Legacy reports are evidence inputs only. Runtime efficiency fields are imported only when explicitly recorded; otherwise they remain `unavailable`.
