# Sustainability analysis (v2)

Modernizes the ideas in the legacy `../Versione vecchia/notebooks/12_Valutazione_Sostenibilità.ipynb`
(log-scale absolute energy/CO2, performance-vs-consumption trade-off, per-phase cost
decomposition, augmentation-vs-diffusion comparison) while fixing its known limitations: naive
summation of every JSON/JSONL on disk, duplicated resume segments, no separation between
validation and test, only two configurations compared, and no provenance on where a number came
from.

## Schema

`notebooks/utility/sustainability_registry.py` defines the canonical event
(`results/sustainability/events.jsonl`, one JSON object per line): `run_id, experiment_id,
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
generator efficiency table. The final report may also use the registry (never raw EcoTracker
logs directly) for absolute energy/CO2 per phase, PR-AUC-vs-kWh trade-offs, and actual-vs-
canonical breakdowns. Efficiency is never a primary generator-selection metric.

Existing `notebooks/utility/eco_tracker.py` provides CodeCarbon measurement and RAM-peak
sampling. Runtime events may be recorded by later real executions; this refactoring produced no
training or scientific sustainability result.

CodeCarbon figures are estimates, not wall-socket measurements — repeated here because the
original notebook's summary made the same disclaimer and it remains true.
# Import legacy canonico

`python scripts/import_legacy_sustainability_logs.py` normalizza in modo idempotente i JSON/JSONL
EcoTracker già presenti in `experiments/` e `results/`, deduplica per firma del contenuto e scrive
`results/sustainability/canonical_events.jsonl`. Il dataset corrente contiene 193 eventi importati.
Il report publication-oriented può usare questa registry e deve riportare separatamente actual
e canonical energy.
