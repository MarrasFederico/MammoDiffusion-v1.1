# Classifier-matrix v2 execution runbook

This is the canonical operational sequence for the v2 classifier pipeline. Run every command
from the repository root. The historical pipeline under `results/final_evaluation/` is separate;
v2 uses `experiments/classifiers_matrix/` and `results/final_evaluation_v2/` exclusively.

The static/unit-test environment needs Python plus the dependencies in `requirements.txt`; CUDA
is not required. The execution host additionally needs working NVIDIA drivers/CUDA and locally
available model assets. The registered frameworks are TensorFlow 2.15+ and PyTorch 2.0+ with
`timm` 1.0+; use the versions installed and certified by the signed GPU profile/smoke artifacts.

## 1. Clone or update the repository

```bash
git clone <repository-url> MammoDiffusion
cd MammoDiffusion
git switch main
git pull --ff-only origin main
```

Prerequisite/input: repository access. Success means `main` is fast-forwarded and `git status
--short` contains no unexplained source changes. This is rerunnable. Keep all committed files;
do not delete runtime manifests/checkpoints after execution begins.

## 2. Activate the environment

```bash
python -m pip install -r requirements.txt
python scripts/check_classifier_runtime_environment.py
```

Success means required imports and registered local assets resolve. A missing mandatory package or
model asset blocks profiling/execution. Installation is rerunnable; package caches are disposable.

## 3. Dataset preflight

```bash
python scripts/preflight_classifier_pipeline.py --json
```

Input: registered train/validation manifests and files. Output: read-only JSON. Success at this
point is `READY_TO_PROFILE_GPU`, `READY_TO_DRY_RUN`, or `READY_TO_RUN_STAGE1`; `BLOCKED` must be
resolved without changing split/lineage rules. The command is idempotent, resumeless, and writes
nothing. It never reads the locked test split.

## 4. Static preflight and unit checks

```bash
python -m compileall notebooks/utility scripts
python scripts/run_classifier_static_tests.py
python scripts/status_classifier_pipeline.py
```

Input: source/configuration only. Success means compilation/tests pass and status reports exactly
112 Stage-1 notebooks, 100 READY, 12 BLOCKED, and 300 Stage-1 jobs. These commands are rerunnable
and do not create scientific results. Test/cache output is disposable.

## 5. Profile GPU memory

Use the stable UUID of the intended profiling GPU, normally the RTX 5060 Ti. Run once per
architecture without changing code/environment between commands:

```bash
CUDA_VISIBLE_DEVICES=<GPU_UUID> python scripts/profile_classifier_vram.py --architecture resnet50 --gpu-uuid <GPU_UUID>
CUDA_VISIBLE_DEVICES=<GPU_UUID> python scripts/profile_classifier_vram.py --architecture maxvit512 --gpu-uuid <GPU_UUID>
CUDA_VISIBLE_DEVICES=<GPU_UUID> python scripts/profile_classifier_vram.py --architecture mammofm --gpu-uuid <GPU_UUID>
CUDA_VISIBLE_DEVICES=<GPU_UUID> python scripts/profile_classifier_vram.py --architecture raddino --gpu-uuid <GPU_UUID>
```

Output: `results/runtime_profiles/classifier_vram_profiles.json`. Success requires four signed,
fresh records with measured peaks. Missing/error records block real launches but not dry-runs.
Rerun replaces only the matching architecture record. Keep the signed bundle; temporary framework
caches may be removed.

## 6. Certify the four architecture GPU smokes

```bash
CUDA_VISIBLE_DEVICES=<GPU_UUID> python scripts/run_classifier_gpu_smokes.py --architecture resnet50 --gpu-uuid <GPU_UUID>
CUDA_VISIBLE_DEVICES=<GPU_UUID> python scripts/run_classifier_gpu_smokes.py --architecture maxvit512 --gpu-uuid <GPU_UUID>
CUDA_VISIBLE_DEVICES=<GPU_UUID> python scripts/run_classifier_gpu_smokes.py --architecture mammofm --gpu-uuid <GPU_UUID>
CUDA_VISIBLE_DEVICES=<GPU_UUID> python scripts/run_classifier_gpu_smokes.py --architecture raddino --gpu-uuid <GPU_UUID>
```

Input: the unchanged environment/revision/profile policy. Output:
`results/runtime_profiles/classifier_gpu_smoke_results.json`. All forward, backward and checkpoint
round-trip fields must pass and match the profile records. A failure blocks execution. Failed
architecture records may be rerun after a documented environment fix; keep the final signed bundle.

## 7. Scheduler dry-run

```bash
python scripts/run_classifier_experiment_matrix.py --stage 1 --mode plan --dry-run
```

Success means every eligible pending job has a reasoned admission/wait decision. This read-only
command is freely rerunnable and does not claim jobs, load models or write runtime results.

## 8. Launch Stage 1

```bash
python scripts/run_classifier_experiment_matrix.py --stage 1 --mode auto --target-5060-jobs 3 --target-3060-jobs 2
```

Input: valid signed GPU certifications and the 300-job matrix. Outputs are isolated run/checkpoint
and result directories. Success means all Stage-1 logical configurations reach COMPLETE. Missing
certification, live scheduler ownership, incompatible artifacts, or scientific blockers stop the
launch. This command is resumable and may be rerun; never delete `run_manifest.json`, resume
checkpoints, checkpoint metadata, validation predictions, or ensemble manifests.

## 9. Monitor Stage 1

```bash
python scripts/status_classifier_pipeline.py
python scripts/status_classifier_pipeline.py --json
```

The command validates schemas/signatures and never reads the locked test. It is read-only and
rerunnable. Success means Stage 1 reports 100/100 logical ensembles complete.

## 10. Interrupt and resume

Send Ctrl-C once to the scheduler, wait for children to stop, then rerun the Stage-1 launch command.
SIGTERM-compatible workers publish resumable state. Stale PID locks are reclaimed atomically;
verified completed jobs are skipped. Do not manually edit state files or delete checkpoint rotation
files. Only `.tmp.*` files left by a confirmed-dead process are disposable after incident review.
`FAILED_FINAL` never resumes automatically. After correcting and documenting its root cause, use:

```bash
python scripts/resume_classifier_experiment_matrix.py --reset-failed-final <EXPERIMENT_ID> --reason "<INCIDENT_AND_FIX>"
```

This explicit audited reset returns only that job to PENDING; it does not delete artifacts.

## 11. Finalize Stage 1

```bash
python scripts/finalize_validation_stage.py --stage 1
```

Input: every signed three-seed validation ensemble. Outputs: Stage-1 completion manifest,
validation leaderboard, rationale, and signed union under `results/generator_comparison/`. The
command refuses incomplete/seed-level inputs and never reads test data. Identical reruns are
byte-idempotent; a different pre-existing union requires incident review.

## 12. Verify the selected generator union

```bash
python scripts/status_classifier_pipeline.py --json
```

Success means `selected_generator_union.status` is `VALID` and Stage-1 scientific completion is
true. Preserve the union, leaderboard and ensemble manifests together.

## 13. Generate Stage 2 notebooks and matrix

```bash
python scripts/create_classifier_stage2_notebooks.py --selected-union results/generator_comparison/selected_generator_union.json
python scripts/build_classifier_experiment_matrix.py --stage 2 --selected-union results/generator_comparison/selected_generator_union.json
```

Inputs: the signed, revision-bound, complete union and scientifically available generator files.
Outputs: deterministic Stage-2 notebooks/inventory, variants, and matrix jobs. Missing images,
lineage, signatures or test-leakage checks block generation. Repeating with identical inputs is
idempotent; do not hand-edit generated notebooks or the matrix.

## 14. Stage 2 dry-run

```bash
python scripts/run_classifier_experiment_matrix.py --stage 2 --mode plan --dry-run
```

The same read-only admission rules and signed GPU profiles apply. Resolve every unexpected refusal
before launching.

## 15. Launch Stage 2

```bash
python scripts/run_classifier_experiment_matrix.py --stage 2 --mode auto --target-5060-jobs 3 --target-3060-jobs 2
```

Outputs use the same atomic checkpoints, state machine, validation and ensemble contracts as
Stage 1. The launch is resumable and rerunnable; COMPLETE jobs are not overwritten.

## 16. Finalize Stage 2

```bash
python scripts/finalize_validation_stage.py --stage 2
```

Input: all signed Stage-2 ensembles plus preregistered baselines. Output:
`results/final_evaluation_v2/primary_finalists_manifest.json`, including deterministic primary and
secondary logical panels and exact logical-to-seed provenance. Incomplete Stage 2 is rejected.

## 17. Validate panel selection

```bash
python scripts/status_classifier_pipeline.py --json
```

Success means `panel_selection.status` is `VALID`. The panel is validation-selected only. Keep its
manifest and all referenced checkpoint/ensemble artifacts immutable.

## 18. Create the scientific lock

First run the read-only readiness check, then explicitly create the lock:

```bash
python scripts/finalize_locked_test_stage.py
python scripts/finalize_locked_test_stage.py --confirm-locked-test
```

The confirmed command validates Stage completions, panels, checkpoint signatures and the locked
dataset manifest/content identities, then writes the immutable, revision-bound v2 lock. It does not
run model inference or calculate test metrics. A missing/duplicate/out-of-root test entry or prior
locked attempt blocks creation. An identical existing valid lock is a no-op; a divergent lock is
never overwritten.

## 19. Run one-shot locked inference

```bash
python scripts/run_locked_classifier_inference.py --confirm-locked-inference
```

This is the first model evaluation on the locked test and may be run only after the lock. It writes
one atomic canonical table per logical finalist and a signed completion manifest. A valid result is
never overwritten. After a technical failure, record an incident and retry exactly once through:

```bash
python scripts/run_locked_classifier_inference.py --authorize-retry <INCIDENT_ID>
python scripts/run_locked_classifier_inference.py --confirm-locked-inference --incident-token <INCIDENT_ID>
```

The retry reuses complete atomic tables and cannot alter panels or thresholds. Preserve all start,
failure, authorization, prediction and completion artifacts.

## 20. Aggregate final statistics

```bash
python scripts/finalize_classifier_report.py --bootstrap 2000
```

Input: the signed completed locked prediction manifest. Output: patient-aggregated tables, ROC/PR
AUC with patient bootstrap CIs, confusion/sensitivity/specificity, calibration/Brier, paired
DeLong/bootstrap/McNemar comparisons, within-family Holm corrections and provenance. Missing or
changed prediction files block aggregation. Identical inputs yield deterministic statistical JSON.

## 21. Inspect the final report

```bash
python scripts/status_classifier_pipeline.py
```

Success means `FINAL_AGGREGATION_COMPLETE` validates and the final Markdown/JSON/CSV/figures exist
under `results/final_evaluation_v2/`. Scientific selection and final aggregation remain separate
states. Published artifacts must be retained; transient caches and `.tmp.*` files from dead
processes may be removed only after confirming no owner is active.

## Commands that must never be run before the scientific lock

Never run either command before Step 18 succeeds:

```bash
python scripts/run_locked_classifier_inference.py --confirm-locked-inference
python scripts/finalize_classifier_report.py
```

Do not open `data/processed/metadata/test.csv`, inspect locked images, create alternate test panels,
tune thresholds on test predictions, or run legacy final-evaluation commands as substitutes for v2.
