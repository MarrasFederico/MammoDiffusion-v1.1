# Generator status

The machine-readable source of truth is `configs/generator_registry.json`; the selected family
representatives and their immutable benchmark evidence are recorded in
`configs/selected_generators.json` and `configs/generator_selection_evidence_v1.json`. Generator
choice is validation-only; downstream test performance cannot change the choice.

| ID | Status | Classes | Scientific role |
|---|---|---|---|
| 01 | complete, verified | negative, positive | SD2.1 50-step sampling ablation of experiment 02 |
| 02 | complete | negative, positive | final comparison |
| 03 | complete | negative, positive | final comparison, fine-tuned VAE/full tuning |
| 04 | complete | negative, positive | final comparison, LoRA |
| 05 | complete | positive only | from-scratch baseline |
| 06 | model identity complete; per-image mapping refused | negative, positive | generation-pool ablation/extension of G05; not a distinct generator |
| 07 | complete | negative, positive | final comparison, SD-VAE |
| 08 | complete | negative, positive | final comparison plus 25/50/75/100-step ablation |

LoRA selected `checkpoint-4500` on validation. The final manifest records 2,722 images per class,
100 steps, CFG 7.5 and `per_image_seed_v2`; both final class directories and filter reports exist.
The adapter is separate from the immutable SD2.1 base. Its CodeCarbon-style logs are estimates,
not direct wall-socket measurements. Failed/resumed/worker records must be grouped by run, phase,
class and worker before totals are computed; summary records must not be added to their children.

`notebooks/3_generator_benchmark/01_Unified_Generator_Benchmark.ipynb` is the analysis entry
point; `02_Generator_Selection.ipynb` records the family-level selection. Positive-only and
two-class rankings remain separate, while sampling variants and RAW/FILTERED results remain
explicit ablations. The registry never claims a missing file or metric.

The reproducible benchmark artifacts live under `results/2_diffusers/benchmark/`;
the per-generator lineage records live under `results/2_diffusers/provenance/`.
