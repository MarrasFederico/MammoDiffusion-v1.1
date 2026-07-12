# Classifier execution plan

Selection and thresholds are frozen from validation before any locked test is inspected. The eight
inference-ready finalists are MaxViT 02a/02c/02j, Mammo-FM 03a/03b/03d and RAD-DINO 04a/04b.
ResNet 01a/01b remain excluded from execution until their exact `.keras` checkpoints are recovered
and validated. This does not alter the locked ten-finalist policy.

Execute in this order:

1. Run validation-only notebook `04x_Leaderboard_Validation_All_Classifiers.ipynb` and verify the
   existing lock/threshold artefacts; do not recompute choices from test data.
2. For each inference-ready registry entry, run its family locked notebook: `02k`, then `03e`, then
   `04c`. Dry-run first and confirm checkpoint, dataset manifest, positive class and threshold.
3. Run `04y_Final_Test_Locked.ipynb` once all intended locked predictions are present.
4. Run `04z_Final_Statistical_Comparison.ipynb` only after 04y real-run completes.

Do not run ResNet `01y` with substitute weights. If both exact checkpoints are later recovered,
load with `compile=False`, validate ResNet50 input/output/activation and provenance, regenerate
validation predictions and full-precision thresholds, then lock them before their one test pass.

Current terminal state is intentionally not “complete”: `final_aggregation_complete=False`; 04y
and 04z real-runs remain pending. Training is not required for the eight ready finalists, while any
new dataset/classifier combination must first be justified as validation screening rather than an
automatic Cartesian sweep.
