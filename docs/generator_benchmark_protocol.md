# Unified generator benchmark protocol

Open and run `notebooks/3_generator_benchmark/05_Unified_Generator_Benchmark.ipynb`. It is the canonical interface; no CLI wrapper is required.

## Counts and references

- `synthetic_pool_target = 1361` is the minimum valid synthetic pool per eligible candidate.
- `real_reference_count` is every positive image available in `data/processed/metadata/val.csv`.
- The main result is a full-reference point estimate using all validation positives and a deterministic balanced synthetic subset.
- Stability uses `floor(0.8 × min(real_reference_count, synthetic_pool_count))`, which must exceed the PRDC neighbour count.

For example, 1,361 synthetic images and 73 validation positives produce stability repetitions with 58 real and 58 synthetic features. Both subsets therefore vary. Sampling is deterministic and without replacement, and a shared plan is recorded in `resampling_plan.json`.

## Metrics

KID is primary. Full-reference values are point estimates; repeated balanced KID/PRDC results are explicitly called repeated-subsampling stability intervals. FID remains descriptive. RAW and FILTERED results remain separate. Top candidates are compared with paired per-repetition KID differences, and practical equivalence uses the protocol-configured margin.

Frozen InceptionV3 and RAD-DINO embeddings are cached only when ordered image IDs, path/size/SHA-256 fingerprints, model and weights identifiers, preprocessing, feature dimension, relevant code version, metadata CSV hash and source-manifest content hash all match.

## Similarity analyses

- Train memorization: synthetic → the complete training corpus declared by each generator, including negative and augmented sources where applicable. It reports nearest ID/label/source, embedding distance, SSIM, perceptual hash and an exact-hash check against the entire corpus. Only this analysis can gate memorization.
- Validation similarity: synthetic → nearest real validation positive; descriptive distribution similarity, never called memorization.
- Synthetic duplication: synthetic → nearest other synthetic; reports nearest distance and exact/perceptual duplicate rates.

Panels show closest, median and farthest examples deterministically for all three relations.

## Selection roles

The 50-step Stable Diffusion variant is a `sampling_ablation` and excluded from automatic downstream eligibility. The canonical 100-step variant may be eligible. The first LDM is a `descriptive_baseline` until lineage is demonstrated. Notebook 06 performs a transparent manual selection and saves only `configs/selected_generators.json`.

Technical validity and filtering acceptance are independent. Filtering acceptance is read from a filter manifest (`accepted / raw submitted`), never inferred from a filtered directory. Primary candidates require readable content-aware provenance, matching sample sets, coherent RAW/FILTERED lineage, a filter manifest when filtering was applied, and a declared training-corpus manifest. Runtime efficiency fields are imported only when a referenced manifest contains them; otherwise their status is `unavailable`.
