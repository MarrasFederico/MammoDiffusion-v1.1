# Manual notebook execution guide

No script launches or replaces a scientific notebook.

1. Run generator notebooks only when candidate outputs are missing.
2. Open and execute `05_Unified_Generator_Benchmark.ipynb`.
3. Review RAW/FILTERED tables, repeated-subsampling intervals, duplication, train memorization, validation similarity and efficiency.
4. Open `06_Generator_Selection.ipynb`, enter one eligible ID per family, validate and save the selection.
5. Open `07_MaxViT512_Downstream.ipynb`, set `CONDITION`, `SEED`, `GPU`, and `RESUME`, then run all cells. Repeat its 12 combinations.
6. Repeat the same procedure with `08_MammoFM_Downstream.ipynb`.
7. Execute `09_Downstream_Validation_Comparison.ipynb` after all 24 prediction files exist.
8. Freeze selections, thresholds and comparisons on validation.
9. Identify an honest final-evaluation dataset and execute `10_Final_Evaluation_and_Report.ipynb` only when ready.

## The 24 manual combinations

| Architecture | Condition | Seeds |
|---|---|---|
| MaxViT-512 | `real_only` | 17, 42, 73 |
| MaxViT-512 | `real_augmented` | 17, 42, 73 |
| MaxViT-512 | `real_plus_best_finetuned_positive` | 17, 42, 73 |
| MaxViT-512 | `real_plus_best_fromscratch_positive` | 17, 42, 73 |
| Mammo-FM | `real_only` | 17, 42, 73 |
| Mammo-FM | `real_augmented` | 17, 42, 73 |
| Mammo-FM | `real_plus_best_finetuned_positive` | 17, 42, 73 |
| Mammo-FM | `real_plus_best_fromscratch_positive` | 17, 42, 73 |

Each row represents three manual runs. Synthetic conditions require `configs/selected_generators.json`. Do not tune one condition differently, and do not consult a final-evaluation split while choosing models or thresholds.
