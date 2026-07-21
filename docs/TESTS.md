# Test suite

The `tests/` directory is a fast, model-free regression suite that protects the validated logic of
the project: the generator selection, the classifier protocol, the exact-count/readability/isolation
invariants of the synthetic data, and the reproducibility invariants of the notebooks. The tests use
fixtures and static inspection only — they never train a model, load model weights, or open the real
image datasets — so the whole suite runs in well under a minute.

Run everything from the repository root:

```bash
conda run -n tf-gpu python3 -m unittest discover -s tests -p "test_*.py"
```

The tests caught real regressions during the reorganization of this repository (a machine GPU UUID
leaking into committed notebook outputs, and a broken content-hash chain after a path refactor), so
they are kept green as a gate before every commit.

## Generator benchmark and selection

| File | Tests | What it verifies |
|---|---|---|
| `test_generator_benchmark_protocol.py` | 22 | The benchmark protocol: synthetic pool target, feature spaces, gates, ranking hierarchy. |
| `test_benchmark_metrics_guard.py` | 2 | The distribution-metrics guard in benchmark notebook 01. |
| `test_generator_benchmark_local_encoders.py` | 6 | Notebook 01 only extracts through the configured local InceptionV3/RAD-DINO encoders; no downloads. |
| `test_generator_representation_policy.py` | 9 | RAW vs FILTERED representations are kept separate; FILTERED is the official ranking. |
| `test_scientific_integrity_patch.py` | 9 | Scientific-integrity invariants: filter-acceptance independence and descriptive-baseline ranking ineligibility. |

## Classifier pipeline

| File | Tests | What it verifies |
|---|---|---|
| `test_classifier_protocol.py` | 7 | The 2 × 4 × 3 protocol, job resolution and the selection-decision content bindings. |
| `test_classifier_preflight.py` | 5 | Metadata-only downstream preflight integrity checks. |
| `test_classifier_resume_safety.py` | 10 | Checkpoint/resume boundaries and the no-terminal-environment-variable rule for the notebooks. |
| `test_classifier_analysis_and_final_evaluation.py` | 6 | Seed ensembles, patient-level aggregation, the Holm comparison and the final-evaluation guard. |
| `test_final_matrix_statistics.py` | 20 | Patient-level bootstrap, Holm correction and the underlying metric math. |

## Generation (diffusers)

| File | Tests | What it verifies |
|---|---|---|
| `test_parallel_generation.py` | 99 | Multi-GPU generation planning, locking, resume and GPU-by-UUID selection (no model loading). |
| `test_artifact_phase_planner.py` | 19 | The phase planner that schedules generation/evaluation artifacts. |
| `test_shared_diffusers_assets.py` | 11 | Shared SD2.1/Diffusers asset identities and the pinned `diffusers_repo` commit. |

## Repository, notebooks and utilities

| File | Tests | What it verifies |
|---|---|---|
| `test_notebook_completion.py` | 28 | Notebooks compile, carry no legacy paths or terminal environment variables, and raise no unimplemented errors. |
| `test_publication_repository.py` | 6 | Required files exist, the final-evaluation guard is present, and no heavy model/archive artifacts are tracked. |
| `test_gradio_selected_generators.py` | 4 | The Gradio demo reads the current `configs/selected_generators.json` selection correctly. |
| `test_sustainability_registry.py` | 21 | The canonical sustainability event schema, deduplication and the actual-vs-canonical accounting. |
