# Sustainability analysis (v2)

Modernizes the ideas in the legacy `../Versione vecchia/notebooks/12_Valutazione_Sostenibilità.ipynb`
(log-scale absolute energy/CO2, performance-vs-consumption trade-off, per-phase cost
decomposition, augmentation-vs-diffusion comparison) while fixing its known limitations: naive
summation of every JSON/JSONL on disk, duplicated resume segments, no separation between
validation and test, only two configurations compared, and no provenance on where a number came
from.

## Schema

`notebooks/utility/sustainability_registry.py` defines the canonical event
(`results/5_sustainability/events.jsonl`, one JSON object per line): `run_id, experiment_id,
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
canonical registry (never raw EcoTracker logs directly) and plots absolute energy/CO2 per generator
experiment, the per-phase decomposition, generation energy per 1000 images, and the actual-vs-
canonical breakdown. Efficiency is never a primary generator-selection metric.

Existing `notebooks/utility/eco_tracker.py` provides CodeCarbon measurement and RAM-peak
sampling. Runtime events may be recorded by later real executions; this refactoring produced no
training or scientific sustainability result.

CodeCarbon figures are estimates, not wall-socket measurements — repeated here because the
original notebook's summary made the same disclaimer and it remains true.
## Legacy canonical import

`python scripts/import_legacy_sustainability_logs.py` idempotently normalizes the EcoTracker
JSON/JSONL logs already present under `experiments/` and `results/`, deduplicates them by content
signature and writes `results/5_sustainability/canonical_events.jsonl`. The current dataset contains
193 imported events. The publication-oriented report may use this registry and must report actual
and canonical energy separately.
