# Final locked test protocol (v2)

This is a second, independent lock alongside the existing one: `results/final_evaluation/
FINALISTS_LOCKED` (from `notebooks/utility/lock_validation_finalists.py`) already freezes the
validation-selected finalists of the original 22-experiment registry and is untouched by
anything in this document. The v2 lock covers the expanded matrix and lives entirely under
`results/final_evaluation_v2/`.

Per the existing `docs/LEGACY_OPERATIONAL_MIGRATION.md` and `docs/CLASSIFIER_EXECUTION_PLAN.md`,
the original pipeline's `04y_Final_Test_Locked.ipynb` real-run has also not happened yet
(`final_aggregation_complete=False`); the v2 lock does not depend on that changing first.

## What gets frozen

`scripts/finalize_locked_test_stage.py --confirm-locked-test` computes and writes, only after
its preconditions pass (a non-empty signed `SELECTED_GENERATOR_UNION`, a Stage 2 primary
finalists manifest, and a real `data/processed/metadata/test.csv`):

- `experiment_matrix_manifest.json` — SHA-256 of `classifier_experiment_matrix.json`,
  `dataset_variant_registry.json`, `classifier_training_protocols.json`,
  `final_generator_registry.json`.
- `test_dataset_manifest.json` — SHA-256 of `test.csv`, row/patient counts, and a hash of the
  sorted patient-ID set (so a same-size but different-composition swap is still detected).
- `primary_finalists_manifest.json` / `secondary_panel_manifest.json` — which configurations are
  in scope, produced by `scripts/finalize_validation_stage.py --stage 2` and never edited by
  the lock script itself.
- `primary_finalists_checkpoints.json` — per-experiment checkpoint SHA-256 at lock time.
- `EXPERIMENT_MATRIX_LOCKED` — the permanent marker, containing a single `lock_signature` hash
  over everything above.

The script never reads a single test-set prediction or metric. It refuses to run
(`preconditions()`, exit code 1, nothing written) whenever `SELECTED_GENERATOR_UNION` is
missing/empty, the Stage 2 manifest is missing, or `test.csv` is missing — verified this session
against the real (currently unready) repository state.

## Re-verification before any real-run test read

`verify_lock_still_valid(root)` re-derives every hash above from current disk state and refuses
on any mismatch: a changed checkpoint, a changed `dataset_variant_registry.json` or
`classifier_experiment_matrix.json`, or a changed `test.csv` (by content hash or by patient-ID
set). Both v2 notebooks (`04y_v2_Final_Test_Locked_Matrix.ipynb`,
`04z_v2_Final_Statistical_Comparison_Matrix.ipynb`) call this as their first executable cell.

**Important nuance found and fixed this session:** inside a Jupyter kernel, `SystemExit` raised
in one cell does not stop later cells in the same run (unlike a plain script) — nbconvert
happily continues to the next cell. Both v2 notebooks therefore re-check `is_valid` explicitly
in every subsequent cell that would touch test-adjacent data, rather than relying on an earlier
cell's exit to have stopped execution. This is worth checking in the original (non-v2)
`04y_Final_Test_Locked.ipynb`/`04z_Final_Statistical_Comparison.ipynb` too, since they were not
modified this session and may share the same single-point-of-refusal pattern.

## Statistics, once real predictions exist

`notebooks/utility/classifier_statistics.py`: paired stratified patient-level bootstrap, DeLong
(cross-validated this session against an independent ROC-AUC implementation), exact-binomial/
chi-square McNemar at the locked threshold, and Holm-Bonferroni correction applied strictly
within one family at a time (`primary_roc_auc, primary_pr_auc, primary_mcnemar, secondary_roc_auc,
secondary_pr_auc, secondary_mcnemar` — spec 17.4; families are never pooled).

## What this session did not do

No lock exists yet (preconditions correctly refuse), no checkpoint has been trained, and no cell
in either v2 notebook has read `test.csv`. The exact commands to reach a real lock and a real
test run are in the top-level report delivered with this change.
