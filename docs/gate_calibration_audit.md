# Post-benchmark gate calibration audit

This document describes the methodological audit of the generator-benchmark eligibility gates that
runs **after** the benchmark (`notebooks/3_generator_benchmark/05_Unified_Generator_Benchmark.ipynb`)
has produced its results. The audit **does not** change the active gate policy and **does not** modify
`configs/generator_benchmark_protocol.json`. It only measures, freezes, and proposes.

## What it does not do

The audit never loads a feature encoder, never re-extracts embeddings, never re-runs KID/FID/PRDC on
the synthetic pools, never selects a generator, never trains a downstream classifier, and never reads
the test split. It consumes only already-computed CSVs, cached RAD-DINO/Inception embeddings, and the
image files required for perceptual-hash / SSIM diagnostics.

## Code

* `notebooks/utility/gate_audit.py` — pure, unit-tested helpers:
  * `perceptual_hash_cluster_diagnostics` — **order-independent** perceptual-hash cluster metrics
    (`phash_any_neighbour_rate`, `phash_component_excess_rate`, connected components, largest cluster);
  * `confirmed_duplicate_analysis` — the preregistered `exact | (pHash<=2 AND SSIM>=0.98)` rule,
    reported as `confirmed_duplicate_rate` with inspectable per-pair evidence;
  * `legacy_order_dependent_phash_rate` — the deprecated, order-dependent rate, kept only for
    comparison;
  * `repeated_prdc_baseline` / `split_half_prdc_baseline` — real-vs-real RAD-DINO PRDC baselines from
    cached embeddings;
  * `descriptive_generator_ranking` — the preregistered metric hierarchy applied **ignoring** the gates;
  * `efficiency_from_manifest_strict` — refuses to infer seconds-per-image from ambiguous durations.
* `notebooks/utility/run_gate_audit.py` — regenerates all runtime artifacts under
  `results/publication_v2/generator_benchmark/gate_audit/` (git-ignored). Run from the repo root:
  `python notebooks/utility/run_gate_audit.py`.
* `tests/test_gate_audit.py` — behaviour tests (order independence, chain clustering, confirmed vs
  exact rates, deterministic baselines, strict efficiency).

## Runtime artifacts (not committed)

```
gate_audit/
  original_protocol.json                 frozen protocol snapshot
  original_generator_summary.csv         frozen benchmark summary
  original_generator_ranking.csv         frozen benchmark ranking
  original_gate_results.json             gates + per-candidate failures (never overwritten)
  perceptual_hash_diagnostics.csv        order-independent pHash metrics per candidate
  perceptual_hash_pairs.csv              inspectable pHash<=2 / exact pairs (+ SSIM, rad_dino_distance)
  phash_panels/                          deterministic closest-pair panels per candidate
  real_phash_baselines.csv               real train/validation positives + labelled augmentation
  real_real_prdc_baselines.csv           per-repetition real-vs-real PRDC
  real_real_prdc_summary.csv             summarised real-vs-real PRDC (coverage/precision/recall/density)
  descriptive_generator_ranking.csv      gate-independent ranking of the 5 official candidates
  paired_generator_differences.csv       paired KID stability differences within each family
  generator_summary.csv                  benchmark summary with corrected efficiency columns
  generator_ranking.csv                  copy of the ranking (efficiency is not part of the key)
  gate_policy_review.md                  human-readable review of the three options
  gate_policy_proposal.json              machine-readable proposal (no threshold asserted)
```

## Findings (from the frozen run at HEAD `cd05886c`)

* **Original outcome:** five official candidates (G02, G03, G04, G07, G08 — FILTERED) measured; **zero**
  eligible. Every candidate fails both `maximum_perceptual_duplicate_rate` (0.02) and
  `minimum_rad_dino_coverage` (0.5).
* **Perceptual hash:** the active gate uses an order-dependent count (flag an image if any *earlier*
  image is within pHash distance 2). Under the order-independent audit and the preregistered
  confirmed-duplicate rule, **`confirmed_duplicate_rate` is 0.0 for all five candidates** — every
  flagged pair is a perceptual-hash collision between visually distinct images (low SSIM, large
  RAD-DINO distance), not a true duplicate. Real train positives have `phash_any_neighbour_rate` ≈ 0.006
  and validation positives ≈ 0.0; the deliberately similar augmentation pool (labelled separately) is
  ≈ 0.98 and is excluded from the primary baseline.
* **RAD-DINO coverage:** the gate uses a single balanced-point coverage from one 73-image subset. The
  real-vs-real baselines put coverage at ≈ 0.96 (real train vs validation) and ≈ 0.98 (validation
  split-half), while the best candidate (G07) reaches ≈ 0.42. The 0.5 threshold is therefore neither
  calibrated to real-real behaviour nor achievable by any candidate.
* **Descriptive ranking (gate-independent):** fine-tuned G02 → G03 → G04; from-scratch G07 → G08.
* **Efficiency:** the G02/G03/G04 manifests record an `elapsed_seconds` that implies physically
  impossible microsecond-per-image generation and carry no `duration_semantics`. The corrected parser
  marks `generation_seconds_per_image` as `unavailable_invalid_duration_semantics`; unverified
  `energy_kwh`/`peak_vram_mb` are dropped; `checkpoint_size_bytes` (verifiable) is retained.

## Options (decision required — none is applied automatically)

* **A — original protocol:** pHash-only ≤ 0.02 and coverage-point ≥ 0.5 as hard gates. Outcome: no
  candidate selectable; downstream cannot run.
* **B — safety gates + coverage as ranking:** block on exact/confirmed duplicates, train memorization,
  corruption, FILTERED validity, provenance and lineage; treat pHash-only, coverage, precision, recall
  and density as descriptive/ranking metrics. No new coverage threshold asserted.
* **C — calibrated thresholds:** propose a coverage threshold only if derived explicitly from the
  real-real baselines (state formula, baseline, percentile, resulting pass list, and the post-benchmark
  nature of the amendment). This audit does not assert one.

Changing the active gates requires human approval and an explicit edit to
`configs/generator_benchmark_protocol.json`.
