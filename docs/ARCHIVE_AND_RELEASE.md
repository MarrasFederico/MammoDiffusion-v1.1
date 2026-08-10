# MammoDiffusion v1.1 archive inventory

## Project and archive identity

- **Repository name:** `MammoDiffusion-v1.1`
- **Project:** MammoDiffusion v1.1
- **Authoritative branch:** `main`
- **Scientific scope:** frozen whole-image mammography synthesis and downstream-classification study
- **Historical identifiers:** earlier phase labels remain in notebooks, schemas, and provenance;
  they are development phases of this frozen study, not project versions

`origin/main` is the authoritative source of the code and of the lightweight scientific artifacts
tracked by Git. Frozen result JSON/CSV files remain the authoritative numerical record; notebook
outputs are presentation and execution records, not a separate source of truth.

## Two different archives

The project has two legitimate archive boundaries:

1. **Git source tree.** Code, configurations, documentation, tests, versionable predictions,
   metrics, and figures. It excludes datasets, checkpoints, model weights, and runtime caches.
2. **Full scientific archive.** The source tree plus permitted datasets, synthetic image pools,
   checkpoints, latent caches, shared model assets, and execution state needed for expensive reruns.
   This archive is too large and, in some cases, not legally suitable for Git. It is kept on shared
   storage for collaboration.

The shared archive is a sanitized copy: it carries no Git metadata (`.git/` anywhere in the tree),
no agent or editor state directories, and no regenerable runtime junk such as `__pycache__/`,
`*.pyc`, or tool caches. Tracked files such as `.gitignore`, `.gitattributes`, and `.github/` are
ordinary repository content and are kept. Where dropping `.git` would lose provenance -- notably the
pinned Diffusers checkout -- the archive carries a manifest under `archive_manifests/` recording the
upstream URL, the pinned commit, and a deterministic content hash of the source tree, so the
checkout remains verifiable from the uploaded files alone.

A source clone supports protocol review, lightweight tests, deterministic generator-ranking rebuild,
and classifier-report regeneration from committed predictions. It does not by itself support full
training, generation, or image-level benchmark reconstruction.

## KEEP

| category | paths | reason |
|---|---|---|
| source and interfaces | `notebooks/`, `assets/mammodiffusion_gradio/` | executable pipeline and review UI |
| scientific configuration | `configs/` | matrix, generator registry, benchmark rules, and frozen selection |
| documentation | `README.md`, `docs/` | protocol, limitations, interpretation, licenses, and archive boundary |
| tests and environments | `tests/`, `requirements*.txt`, `pytest.ini` | executable scientific contracts and dependency declarations |
| versioned evidence | `results/` entries tracked by Git | preprocessing summaries, benchmark reports, saved predictions, ensemble reports, final evaluation, attribution panels, and supported sustainability outputs |
| data manifests and image pools | local `data/` in the full archive | required for image-level or training reproduction; intentionally outside Git |
| experiment state | local `experiments/` in the full archive | checkpoints, histories, latents, and resume/evaluation state that may be expensive or impossible to reconstruct exactly |
| shared generator assets | `notebooks/utility/diffusers_repo`, `notebooks/pretrained_model/stable-diffusion-2-1-base` in the full archive | pinned code/base assets resolved by generator notebooks; see `SHARED_ASSETS.md` |
| canonical timing provenance | `results/5_sustainability/canonical_events.jsonl` | frozen event snapshot; supported analysis reads only `elapsed_seconds` |

Mammo-FM weights and derived checkpoints must never be placed in the public Git repository. In any
shared archive they must be handled strictly under their own licence and made accessible only to
parties authorized under the applicable terms. Authorized users restore them separately; the terms
are summarized in [mammo_fm_license_note.md](mammo_fm_license_note.md), which is a pointer to the
licence and not a reinterpretation of it.

## REMOVE

Only clearly regenerable or misleading artifacts belong in this category:

- `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, and tool/model caches;
- incomplete downloads, editor swap files, temporary work queues, and disposable composed-model
  views under `.cache/mammodiffusion/`;
- duplicated archives created only for transfer after their validated destination copy exists;
- obsolete sustainability summaries and figures that use the untrusted CodeCarbon energy/CO2
  values and are not consumed by the supported notebook: `actual_vs_canonical.json`,
  `sustainability_summary.md`, `summary_by_run.csv`, `summary_by_experiment.csv`,
  `actual_vs_canonical.png`, `energy_by_phase_log.png`, `generator_energy_co2.png`,
  `phase_decomposition_stacked.png`, and
  `results/2_diffusers/06_ldm_extra1361_fromscratch/plots/ecotracker_summary_per_stage.png`.

The rule covers rendered figures anywhere in `results/`, not only the sustainability tree. The
`*/ecotracker/*.jsonl` source records those figures were drawn from stay in place as frozen
provenance.

Removing caches does not remove scientific evidence. The sustainability deletions prevent stale,
unsupported energy values from competing with the elapsed-time-based analysis.

## REVIEW, do not remove automatically

- intermediate generator checkpoints, optimizer states, and latent archives;
- embedding caches and per-image benchmark tables needed for an exact expensive rerun;
- old execution logs that may be the only timing or failure provenance for a run;
- large image panels or local exports not referenced by the current publication workflow;
- remote ZIP files or duplicate directories whose equivalence has not been verified by checksum;
- any data or model artifact whose redistribution terms are unclear.

These objects can be large, but size alone is not evidence that they are dead. They should be
removed only after a consumer/provenance audit and, for duplicates, a content comparison.

## Artifact-to-consumer map

| authoritative input/artifact | active consumer |
|---|---|
| `data/processed/metadata/*.csv` | preprocessing reuse audit, generator references, classifier dataset builders |
| `configs/generator_registry.json` and `configs/generator_benchmark_protocol.json` | unified benchmark, ranking rebuild, selection validation |
| `results/2_diffusers/benchmark/generator_summary.csv` | `notebooks/utility/rebuild_generator_ranking.py` |
| `configs/selected_generators.json` | classifier condition resolution and dataset construction |
| per-seed validation/test prediction CSVs | `notebooks/utility/rebuild_classifier_reports.py` (validation first, then held-out reports) |
| validation ensemble thresholds | guarded test ensemble reports and final evaluation |
| `results/4_final_evaluation/results.json` | final review notebook, README, and discussion |
| `results/3_classifiers/figures/interpretability/` | qualitative Grad-CAM and Integrated Gradients evidence |
| `results/5_sustainability/canonical_events.jsonl` | sustainability comparison notebook, using `elapsed_seconds` only |
| shared Diffusers checkout and SD2.1 base | generator training/inference notebooks in explicit real-run mode |
| classifier matrix, train/validation metadata, and selected FILTERED pools | manual `notebooks/utility/classifier_preflight.py` audit |
| shared Diffusers checkout/base and possible duplicate copies | manual `notebooks/utility/audit_shared_diffusers_assets.py` audit |

This map identifies the principal edges; tests additionally validate many schemas and guard paths.
Historical artifacts that are retained solely for provenance need not have an active runtime
consumer, but they must be identified as historical rather than presented as current outputs.

## Manual preflight and asset-audit CLIs

Two supported utilities are intentionally operator-run rather than notebook-driven. From the
repository root, audit classifier inputs before an expensive training run with:

```bash
python notebooks/utility/classifier_preflight.py
```

This reads train/validation metadata, resolves all four conditions, checks train/validation patient
separation, and audits the selected FILTERED positive pools and constructed file lists. It does not
load a model, train, run inference, or read the test split. It requires the local metadata and
selected synthetic pools, so it is a full-archive preflight rather than a source-clone smoke test.

Audit the pinned shared Diffusers checkout, SD2.1 base signature, and possible duplicate copies with:

```bash
python notebooks/utility/audit_shared_diffusers_assets.py --dry-run
```

Dry-run auditing is the normal check. The script also exposes explicit duplicate-removal
flags, but those are destructive maintenance actions and are not part of ordinary reproduction or
archive verification.

## Legacy paths and portability

Some frozen CSV/JSON records contain absolute paths such as the original project mount. They are
historical provenance and must not be mass-rewritten because that would alter signed or interpreted
execution evidence. Active loaders normalize separators and reroot recognized project-relative
suffixes when a project is restored elsewhere. New artifacts should prefer repository-relative
paths, and a full archive must preserve enough metadata to resolve any remaining external assets.

## Supported sustainability record

`canonical_events.jsonl` is retained as the frozen event/provenance snapshot. CodeCarbon did not
model the RTX 5060 Ti reliably, so its stored `energy_kwh` and `co2_kg` fields are not supported
measurements. The current analysis uses only valid events longer than 60 seconds and computes an
estimate as `elapsed_seconds / 3600 × 0.170 kW`. Legacy actual-versus-canonical and CO2 summaries are
not supported v1.1 evidence.

## Remaining risks

- The repository has no project-wide license file. Public visibility does not itself grant a reuse
  license; the owner should make a deliberate licensing decision without overriding third-party
  dataset/model terms.
- A full archive may contain assets governed by RSNA/Kaggle, Hugging Face, Mammo-FM, or other
  third-party terms. Verify redistribution rights per asset.
- The source dependency files constrain versions but are not a complete cross-platform lockfile;
  hardware-specific full reruns may still require environment reconstruction.
- Git cannot verify ignored heavy assets. A full archive should be checked independently for
  presence, counts, and checksums before it is treated as a reproduction package.

## Environment declaration versus verification

`requirements.txt` and `requirements-dev.txt` contain compatible-version bounds; they are not a
fully resolved scientific lock and must not be presented as an exact historical training
environment. The v1.1 source tree and lightweight suite were verified with the following
environment:

- Python 3.11.15;
- NumPy 1.26.4;
- TensorFlow 2.15.0;
- PyTorch 2.12.0+cu130;
- Gradio 6.17.3.

This is a verification environment, not proof that every historical training run used the same
package builds. Generator code also relies on the separately archived local Diffusers checkout
at commit `3759fab56d3170a04d747e918a13e55fda6681e2`, documented in
[SHARED_ASSETS.md](SHARED_ASSETS.md). Full training requires the appropriate separately restored
artifacts and a compatible local GPU environment.

## Current reproducibility guarantees

These describe the state of the repository, and each is covered by the test suite.

- **Safe review mode.** Every notebook activates the contract in
  `notebooks/utility/review_mode.py` in its bootstrap cell. Outbound sockets and name resolution are
  blocked, package managers and repository clones are refused, and cohort or shared-asset downloads
  raise an explicit error naming the flag to set. Opening a notebook and running every cell cannot
  train, generate, download, install, clone, or delete anything.
- **Explicit opt-in for anything heavy.** Network access, dependency installation, and data
  downloads are separate flags that ship disabled, and the scientific phases stay gated by their own
  `RUN_*_PHASE` flags. Granting network access does not by itself start training or generation.
- **Project-root isolation.** Root discovery requires a repository marker rather than a directory
  name, so a checkout resolves to itself regardless of where it is cloned or what neighbouring
  directories are called, and resolution from outside any checkout fails rather than guessing.
- **Legacy path rerooting.** `notebooks/utility/project_paths.py` is the single authority for
  interpreting a path recorded by an earlier run. Identity is the repository-relative suffix, so a
  historical absolute prefix is never followed; resolution lands inside the current checkout or
  fails, and a symlinked `data/` or `experiments/` still counts as inside the project.
- **Content-based artifact identity.** Cache and manifest compatibility compares content, not the
  location a file occupied when the record was written, so relocating the checkout does not
  invalidate a still-valid frozen artifact.
- **Deterministic report regeneration.** `rebuild_classifier_reports.py` and
  `rebuild_generator_ranking.py` reproduce the tracked report tree from the committed predictions
  and the committed generator summary, without inference and without opening an image or a model.
- **Frozen scientific artifacts.** Reusing an existing cohort never rewrites the canonical split
  manifests, the frozen preprocessing record, or the traditional-augmentation pool.

## Freeze rule

After validation and synchronization of `main`, the v1.1 scientific study is frozen. Documentation,
tests, and reproducibility fixes may still be corrected on `main`; the cohort, predictions, metrics,
generator selection, and reported results are not to be modified.
