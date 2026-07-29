# Sustainability analysis (v2)

The current analysis covers log-scale absolute energy/CO2, performance-versus-consumption
trade-offs, per-phase cost decomposition, and augmentation-versus-diffusion comparisons. It avoids
double-counting resumed segments and keeps validation and test accounting separate.

## Schema

`notebooks/utility/sustainability_registry.py` defines the canonical event
(`results/5_sustainability/canonical_events.jsonl`, one JSON object per line): `run_id, experiment_id,
dataset_variant_id, architecture, seed, phase, status, parent_run_id, canonical, reused_artifact,
start_time, end_time, elapsed_seconds, energy_kwh, co2_kg, peak_ram_mb, peak_vram_mb, gpu_uuid,
gpu_name, num_images, optimizer_updates, epochs, source_log, signature, value_precision`. `phase`
is restricted to the fixed pipeline-stage vocabulary (`preprocessing` ... `metrics`);
`value_precision` must be one of `measured, estimated, reconstructed, legacy_unverified,
missing`. NaN energy/CO2/elapsed values are rejected at write time, not silently coerced.

Deduplication (`deduplicate_canonical_events`): only `canonical=true`, non-`reused`,
`status=completed` events count toward the reproducible pipeline; a duplicate `run_id` keeps only
its latest entry. `actual_vs_canonical` reports two numbers side by side, always:
`actual_project_energy` (every real attempt, including failures, deduplicated only by exact
repeated log lines) and `canonical_pipeline_energy` (the reproducible cost) — the code asserts
canonical never exceeds actual via test coverage, and the difference is reported as
`retry_and_failure_overhead_kwh`.

## Publication workflow

`notebooks/3_generator_benchmark/01_Unified_Generator_Benchmark.ipynb` includes the descriptive
generator efficiency table. `notebooks/5_sustainability/01_Sustainability_Comparison.ipynb` reads the
canonical registry (never raw EcoTracker logs directly) and compares the generators by **time-based
energy**: because CodeCarbon does not model the RTX 5060 Ti correctly, energy is estimated as
`wall-clock hours × 0.170 kW` (the measured mean draw of that GPU under load), and only real runs
longer than 60 s are kept — which discards CodeCarbon's empty restart appends. It plots per-generator
energy, the per-phase decomposition and the generation cost. The `02_sd21_filtered` generation cost is
a controlled 100-image, 100-step measurement (its historical log conflated generation into filtering);
`05_ldm_basic` is the positive-only baseline and is omitted from the selected-generator
comparison. Efficiency is never a primary generator-selection metric.

**CodeCarbon is kept only as a source of wall-clock time.** `notebooks/utility/eco_tracker.py` wraps
CodeCarbon's `EmissionsTracker` and RAM-peak sampling to produce the raw logs, but its `energy_kwh`
and `co2_kg` are unreliable on the RTX 5060 Ti (CodeCarbon has no power model for that GPU) and are
**discarded**. Only the recorded `elapsed_seconds` is trusted; energy is always recomputed as
`hours × 0.170 kW`. The `energy_kwh`/`co2_kg` fields still exist in the event schema for
backward compatibility with the historical logs, but no analysis reads them.

## Frozen canonical registry

`results/5_sustainability/canonical_events.jsonl` is the versioned publication snapshot and contains
193 events. The comparison notebook consumes this file directly and does not rediscover or rewrite
events from machine-local logs. Reports must continue to distinguish actual and canonical energy.
