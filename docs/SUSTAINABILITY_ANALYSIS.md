# Sustainability analysis in MammoDiffusion v1.1

This is a secondary, descriptive analysis of generator-workflow elapsed time. Its only supported
energy estimate is:

```text
estimated_energy_kwh = elapsed_seconds / 3600 × 0.170 kW
```

The 0.170 kW value is the measured mean draw used for the RTX 5060 Ti under the relevant workload.
The estimate is not a wall-socket measurement, a carbon estimate, or a lifecycle assessment.

## Scope

The retained event registry covers generator-related phases such as preprocessing, generator
training, generation, filtering, and validation. It contains no classifier-training events.
Consequently, this analysis does not compare end-to-end classifier performance against consumption
and does not repeat the historical internal-V1 comparison of traditional versus diffusion
augmentation. Efficiency is not a generator eligibility gate or ranking field.

## Authoritative inputs

`results/5_sustainability/canonical_events.jsonl` is a frozen 193-event snapshot retained for
timing provenance. `notebooks/5_sustainability/01_Sustainability_Comparison.ipynb` loads canonical,
completed, non-reused events through `deduplicate_canonical_events`, converts
`elapsed_seconds` to numeric values, retains events longer than 60 seconds, and removes exact
duplicate-duration entries within generator and phase. The duration filter excludes empty restart
appends.

The event schema still contains legacy `energy_kwh`, `co2_kg`, and `value_precision` fields because
the registry also preserves historical records. **The supported v1.1 analysis does not read the
stored energy or CO2 values.** CodeCarbon did not model the RTX 5060 Ti reliably; it is used only as
a historical source of elapsed duration.

`results/5_sustainability/g02_generation_timing.json` provides the controlled 100-image,
100-step timing for G02. The notebook scales its `seconds_per_image` to the canonical 1,361-image
pool because the historical event log had conflated that generation segment with filtering. For
other retained generators, the analysis sums valid real-run durations by phase. G06 reuses G05
training, so its shared training event is excluded rather than charged twice.

## Supported outputs

The current notebook produces exactly these elapsed-time-based figures:

- `results/5_sustainability/figures/generator_energy_total.png`;
- `results/5_sustainability/figures/generator_energy_by_phase.png`;
- `results/5_sustainability/figures/generation_efficiency.png`.

All use duration multiplied by 0.170 kW. The notebook tables shown in its outputs are derived from
the same in-memory calculation.

Legacy actual-versus-canonical, CodeCarbon-energy, and CO2 summaries are not part of release
evidence. In particular, the removed files `actual_vs_canonical.json`,
`sustainability_summary.md`, `summary_by_run.csv`, `summary_by_experiment.csv`,
`actual_vs_canonical.png`, `energy_by_phase_log.png`, `generator_energy_co2.png`,
`phase_decomposition_stacked.png`, and
`results/2_diffusers/06_ldm_extra1361_fromscratch/plots/ecotracker_summary_per_stage.png` must not
be used or regenerated as current v1.1 results.

The last of those was removed by the independent post-release audit: it rendered the untrusted
CodeCarbon `energy_kwh` and `co2_kg` fields as per-phase Wh and gram-CO2 panels, it had no producing
code left in the repository, and it therefore contradicted the boundary stated above. The underlying
`results/2_diffusers/*/ecotracker/*.jsonl` records are unchanged: they remain frozen execution
provenance, and their energy and CO2 fields stay unsupported wherever they appear.

## Interpretation boundary

The figures compare estimated generator-workflow energy under one hardware/power assumption. They
do not include classifier training, idle-system draw, CPU/storage/network energy, embodied impact,
or location-dependent carbon intensity. Differences also inherit the completeness and historical
quality of elapsed-time logging. They are descriptive resource-accounting estimates and do not
alter the scientific generator selection or downstream conclusions.
