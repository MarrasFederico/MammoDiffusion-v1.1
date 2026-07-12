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

## Notebook

`notebooks/4_comparisons_and_test/00y_Analisi_Consumi_e_Sostenibilita.ipynb` reads only this
registry (never a raw eco_tracker log directly) and produces: absolute energy/CO2 per phase
(log-scale bar), a PR-AUC-vs-kWh trade-off scatter per completed job, a stacked per-phase cost
decomposition, the actual-vs-canonical breakdown, and `results/sustainability/summary_by_run.csv`
/ `summary_by_experiment.csv`. It degrades gracefully and says so explicitly when
`events.jsonl` is empty (verified this session: 0 events currently logged, since no classifier
training has run yet) rather than plotting misleading empty/zero charts silently.

Existing `notebooks/utility/eco_tracker.py` (measurement, via CodeCarbon + a RAM-peak sampling
thread) is unchanged; producing `results/sustainability/events.jsonl` entries from its
`SustainabilityMetrics` output at the point each classifier run completes is future wiring work,
not part of this session (no training ran, so there is nothing yet to log).

CodeCarbon figures are estimates, not wall-socket measurements — repeated here because the
original notebook's summary made the same disclaimer and it remains true.
# Import legacy canonico

`python scripts/import_legacy_sustainability_logs.py` normalizza in modo idempotente i JSON/JSONL
EcoTracker già presenti in `experiments/` e `results/`, deduplica per firma del contenuto e scrive
`results/sustainability/canonical_events.jsonl`. Il dataset corrente contiene 193 eventi importati.
Il notebook `00y` usa questa registry e riporta separatamente actual e canonical energy.
