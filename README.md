<p align="center">
  <img src="assets/logo_MammoDiffusion.png" alt="MammoDiffusion Logo" width="180"/>
</p>

# MammoDiffusion v2

MammoDiffusion is a notebook-first study of synthetic mammography and its downstream
usefulness. The workflow is modular, readable and reproducible: the notebooks show the
configuration, the data, the audits, the utility calls, the metrics, the plots and the
artifacts. There is no mandatory automatic pipeline.

The project starts from the already-processed **512×512 grayscale MLO PNG** corpus under
`data/processed/`; the original RSNA DICOM archive is not part of this repository and is not
needed to reproduce it. The preprocessing notebook therefore verifies the processed corpus
(schema, splits, patient separation, image presence) rather than re-decoding DICOMs.

## Research questions

- **RQ1:** which fine-tuned and which from-scratch generator best balance fidelity,
  diversity, coverage, efficiency and absence of train memorization?
- **RQ2:** does adding synthetic positive mammograms improve classification over real-only
  data and traditional augmentation?
- **RQ3:** is the effect consistent across MaxViT-512 and Mammo-FM?

## Notebook-first workflow

1. Run the generator notebooks if their outputs are missing.
2. Run `notebooks/3_generator_benchmark/01_Unified_Generator_Benchmark.ipynb`.
3. Inspect the RAW/FILTERED metrics and the diagnostic panels.
4. In `notebooks/3_generator_benchmark/02_Generator_Selection.ipynb`, manually choose one
   fine-tuned and one from-scratch generator.
5. Run `notebooks/04_classifiers/01_MaxViT512.ipynb` once; it loops 4 conditions × 3 seeds.
6. Run `notebooks/04_classifiers/02_MammoFM.ipynb` once; it loops 4 conditions × 3 seeds.
7. Run `notebooks/04_classifiers/03_Validation_Comparison.ipynb` and freeze the checkpoints
   and thresholds on validation.
8. Run `notebooks/04_classifiers/04_Final_Evaluation_and_Report.ipynb` to score the frozen
   decisions once on the held-out test set.

The protocol keeps exactly **2 architectures × 4 conditions × 3 seeds = 24 experiments**.
The architectures are MaxViT-512 and Mammo-FM; RAD-DINO is only a medical feature extractor
used inside the generative benchmark.

## Scientific rigor

The benchmark targets a synthetic pool of 1,361 images but uses every real positive available
in validation and evaluates balanced subsets of size `min(real_reference_count,
synthetic_pool_count)`. KID and PRDC use repeated subsampling without replacement; FID is
secondary and uses a single repetition by default. Train memorization, validation similarity
and synthetic duplication are reported as distinct results.

Checkpoints, early stopping and the downstream scheduler all monitor validation PR-AUC. The
maximum optimizer-update budget is fixed within each architecture. Validation and bootstrap are
patient-level; the eight primary comparisons use Holm correction. Checkpoints and thresholds are
fixed on validation before any test access, so the test set contributes no model selection.

## Results and checkpoints

The active code writes or consumes these canonical roots under `results/`:

- `preprocessing/` — preprocessing and augmentation summaries;
- `diffusers/` — generator metrics, plots, energy tracking and sampling sweeps (one folder per
  experiment; the step-count sweep for experiment 08 lives under
  `08_ldm_v3_sdvae_fromscratch/sampling/`);
- `generator_benchmark/` — generator benchmark metrics, rankings and diagnostics;
- `classifier_seed_runs/` — per-seed classifier training and validation outputs;
- `classifier_validation_ensembles/` — the eight three-seed validation ensembles;
- `final_evaluation/` — the test-set ensembles (`test_ensembles/`) and `results.json`;
- `sustainability/` — cross-cutting energy analysis.

Classifier checkpoints under `results/3_classifiers/seed_runs/` are resume and evaluation state:
`checkpoint_latest`, `checkpoint_previous` and every representation of the best checkpoint must
not be pruned.

The earlier scripted pipeline is archived in the `publication-pipeline-scripted-v1` tag; the
300-job matrix in the `classifier-matrix-v2-full` tag.

## Gradio demo

`assets/mammodiffusion_gradio/app.py` reads the current selection from
`configs/selected_generators.json` and serves the two family winners with their best checkpoints:

- G02, Stable Diffusion 2.1 fine-tuned, `checkpoint-3000`, canonical 100-step sampling;
- G07, from-scratch LDM with SD-VAE, `ldm_unet_best_eval.keras` selected at step 130000,
  100-step sampling.

The demo has its own README because it is a separately launchable application:
[Gradio instructions](assets/mammodiffusion_gradio/README.md).

## Portable hand-off and Google Drive

For a complete hand-off, upload `notebooks/`, `configs/`, `experiments/`, `results/` and the
project-permitted `data/` material. In particular include:

- `notebooks/utility/diffusers_repo`, together with its `.git`, to verify the pinned commit;
- `notebooks/pretrained_model/stable-diffusion-2-1-base`;
- the filtered synthetic datasets used by the benchmark and the classifiers.

Keep checkpoints, latents, checkpoint-validation caches, evaluation outputs and embedding
caches. Exclude only regenerable Hugging Face caches/compositions, `__pycache__`, `*.pyc`
files and empty work queues. `experiments/diffusers/` can be uploaded as a first block, but on
its own it is not enough to resume execution on another machine.

## Documentation

- [Consolidated protocol](docs/PROTOCOL.md) — experimental design, generative benchmark,
  Option B amendment, G02/G07 selection, the 2 × 4 × 3 downstream protocol, final evaluation
  and manual execution.
- [Generator status](docs/GENERATOR_STATUS.md)
- [Shared SD2.1/Diffusers assets](docs/SHARED_ASSETS.md)
- [Sustainability analysis](docs/SUSTAINABILITY_ANALYSIS.md) — the energy registry, the legacy-log
  import script and the comparison notebook.
- [Test suite](docs/TESTS.md) — what each regression test protects and how to run them.
- [Mammo-FM academic license](docs/mammo_fm_license_note.md)

Datasets, synthetic images, embeddings and weights stay local and must not be committed.
Mammo-FM weights are subject to their academic license and are not redistributed.
