# Generator status

The machine-readable source of truth is `configs/final_generator_registry.json`. Generator choice
is validation-only; downstream test performance cannot change the choice.

| ID | Status | Classes | Scientific role |
|---|---|---|---|
| 01 | legacy/incomplete | not canonically evidenced | SD2.1 50-step baseline |
| 02 | complete | negative, positive | final comparison |
| 03 | complete | negative, positive | final comparison, fine-tuned VAE/full tuning |
| 04 | complete | negative, positive | final comparison, LoRA |
| 05 | complete | negative, positive | from-scratch baseline |
| 06 | complete | negative, positive | extra-data ablation |
| 07 | complete | negative, positive | final comparison, SD-VAE |
| 08 | complete | negative, positive | final comparison plus 25/50/75/100-step ablation |

LoRA selected `checkpoint-4500` on validation. The final manifest records 2,722 images per class,
100 steps, CFG 7.5 and `per_image_seed_v2`; both final class directories and filter reports exist.
The adapter is separate from the immutable SD2.1 base. Its CodeCarbon-style logs are estimates,
not direct wall-socket measurements. Failed/resumed/worker records must be grouped by run, phase,
class and worker before totals are computed; summary records must not be added to their children.

`00z_Confronto_Diffusori.ipynb` remains the analysis entry point. Positive-only and two-class
rankings must remain separate; sampling variants and RAW/FILTERED results are ablations. The
registry never claims a missing file or metric.
