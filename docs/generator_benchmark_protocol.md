# Unified generator benchmark protocol

Open and run `notebooks/3_generator_benchmark/05_Unified_Generator_Benchmark.ipynb`. It is the canonical interface; no CLI wrapper is required.

## Counts and references

- `synthetic_pool_target = 1361` is the minimum valid synthetic pool per eligible candidate.
- `real_reference_count` is every positive image available in `data/processed/metadata/val.csv`.
- `evaluation_subset_size = min(real_reference_count, synthetic_pool_count)`.

For example, 1,361 synthetic images and 73 validation positives produce valid repetitions with 73 real and 73 synthetic features. Sampling is deterministic and without replacement, and sampled IDs are recorded.

## Metrics

KID is primary and uses repeated balanced subsampling. PRDC uses its own repeated balanced subsampling with `replace=False`; a subset no larger than the nearest-neighbour `k` is rejected. FID has an independent, small repetition count and is descriptive. RAW and FILTERED results remain separate.

Frozen InceptionV3 and RAD-DINO embeddings are extracted once and cached with image IDs, paths, extractor, preprocessing, dimension, code version and source manifest. Resampling reuses these arrays.

## Similarity analyses

- Train memorization: synthetic → nearest real training positive; reports embedding distance, SSIM, perceptual hash, exact match and memorization rate. Only this analysis can gate memorization.
- Validation similarity: synthetic → nearest real validation positive; descriptive distribution similarity, never called memorization.
- Synthetic duplication: synthetic → nearest other synthetic; reports nearest distance and exact/perceptual duplicate rates.

Panels show closest, median and farthest examples deterministically for all three relations.

## Selection roles

The 50-step Stable Diffusion variant is a `sampling_ablation` and excluded from automatic downstream eligibility. The canonical 100-step variant may be eligible. The first LDM is a `descriptive_baseline` until lineage is demonstrated. Notebook 06 performs a transparent manual selection and saves only `configs/selected_generators.json`.
