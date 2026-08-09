# MammoDiffusion v1.1 archive and release inventory

## Release identity

- **Repository name:** `MammoDiffusion-v1.1`
- **Release version:** `v1.1.0`
- **Scientific scope:** final whole-image mammography synthesis and downstream-classification study
- **Historical identifiers:** internal V1 and internal V2 remain in notebooks, schemas, and
  provenance; they are phases of this frozen study, not release-major versions

The release tag identifies the authoritative source snapshot. Frozen result JSON/CSV files remain
the authoritative numerical record; notebook outputs are presentation and execution records, not a
separate source of truth.

## Two different archives

The project has two legitimate archive boundaries:

1. **Git/source release.** Code, configurations, documentation, tests, versionable predictions,
   metrics, and figures. It excludes datasets, checkpoints, model weights, and runtime caches.
2. **Full local scientific archive.** The source release plus permitted datasets, synthetic image
   pools, checkpoints, latent caches, shared model assets, and execution state needed for expensive
   reruns. This archive is too large and, in some cases, not legally suitable for Git.

A source clone supports protocol audit, lightweight tests, deterministic generator-ranking rebuild,
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

Mammo-FM weights and derived checkpoints must not be placed in the public Git release or a shared
archive that violates their license. Authorized users restore them separately under the terms in
[mammo_fm_license_note.md](mammo_fm_license_note.md).

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
`ecotracker_summary_per_stage.png` entry was found by the independent post-release audit; the
`*/ecotracker/*.jsonl` source records it was drawn from stay in place as frozen provenance.

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

Dry-run auditing is the normal release check. The script also exposes explicit duplicate-removal
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
not release evidence.

## Remaining release risks

- The repository has no project-wide license file. Public visibility does not itself grant a reuse
  license; the owner should make a deliberate licensing decision without overriding third-party
  dataset/model terms.
- A full archive may contain assets governed by RSNA/Kaggle, Hugging Face, Mammo-FM, or other
  third-party terms. Verify redistribution rights per asset.
- The source dependency files constrain versions but are not a complete cross-platform lockfile;
  hardware-specific full reruns may still require environment reconstruction.
- Git cannot verify ignored heavy assets. A full archive should be checked independently for
  presence, counts, and checksums before it is treated as a reproduction package.

## Environment declaration versus release verification

`requirements.txt` and `requirements-dev.txt` contain compatible-version bounds; they are not a
fully resolved scientific lock and must not be presented as an exact historical training
environment. The v1.1 source and lightweight suite were verified at release time with
`/home/fede/miniforge3/envs/tf-gpu/bin/python` and the following current environment:

- Python 3.11.15;
- NumPy 1.26.4;
- TensorFlow 2.15.0;
- PyTorch 2.12.0+cu130;
- Gradio 6.17.3.

This is a release-verification environment, not proof that every historical training run used the
same package builds. Generator code also relies on the separately archived local Diffusers checkout
at commit `3759fab56d3170a04d747e918a13e55fda6681e2`, documented in
[SHARED_ASSETS.md](SHARED_ASSETS.md). Full training requires the appropriate separately restored
artifacts and a compatible local GPU environment.

## Freeze rule

After tests, documentation checks, cleanup, repository rename, `v1.1.0` tag, remote push, and full
archive verification, this tree is frozen. Lesion-aware development belongs in a separate
`MammoDiffusion-v2` repository and must not mutate the v1.1 release artifacts.

## Post-release independent audit

An independent audit was run against the `v1.1.0` tag after publication. It re-derived every
released figure from the committed prediction CSV files with scikit-learn as an external reference
and found **no change to any scientific result**: point metrics, bootstrap intervals, empirical tail
areas, and Holm-adjusted values all reproduce to within floating-point noise, and
`rebuild_classifier_reports.py` regenerates the whole tracked report tree byte-identically from a
clean checkout.

`main` therefore carries release-hygiene corrections only, and no result artifact was edited:

- project-root resolution in the notebook bootstrap, in two notebook-level resolvers, and in
  `ldm_project_paths` now prefers a repository marker (`.git`, `configs/`, `data/`) over a directory
  that merely carries the project name. The old order escaped the checkout whenever `data/` was
  absent — the state a fresh clone starts in — and on the original workstation it resolved to the
  parent directory that now also holds the separate successor project;
- `results/2_diffusers/06_ldm_extra1361_fromscratch/plots/ecotracker_summary_per_stage.png` was
  removed under the REMOVE rule above;
- the root-marker heuristic in `shared_diffusers_assets` no longer keys on a configuration file
  this release does not ship;
- regression tests were added for reference-checked average precision and ROC-AUC, paired-bootstrap
  semantics, Holm, and the consistency of the published numbers with the frozen predictions.

The `v1.1.0` tag is unchanged and remains a complete, self-consistent scientific snapshot.
