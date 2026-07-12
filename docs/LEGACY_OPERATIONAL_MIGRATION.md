# Operational legacy migration and idempotent Run All

The read-only source is `../Versione vecchia`. The complete machine-readable mapping is
`configs/legacy_experiment_migration.json`; local heavy artefacts are ignored by Git.

| Current ID | Legacy experiment | Selected checkpoint | Reused output | Scientific difference |
|---|---|---|---|---|
| 01 | `20260607_sd21_rsna_mlo_512` | SD UNet checkpoint 3000; training evidence at 8000 | 1,361 final images/class, validation cache/logs | 50 sampling steps |
| 02 | `20260611_sd21_rsna_mlo_512_inference_100_steps` | same UNet bytes as 01 | 2,722 final + 1,361 RAW-matched/class | 100-step inference; final and matched sets remain distinct |
| 05 | `20260617_ldm_basic` | LDM step 70,000; training evidence at 80,000 | 2,722 positive RAW, 1,361 filtered, VAE/latents/logs | positive-only historical generation |
| 06 | `20260619_ldm_extra1361` | same LDM/VAE bytes as 05 | 4,083 RAW/class, 1,361 positive filtered | first 2,722 positives inherit 05; additional 1,361 are separately indexed |

Every copied file was transferred individually through the migration allow-list, checked with
SHA-256 before and after, and refused on a non-identical collision. The migration excluded embedded
Diffusers repositories, SD2.1 bases, Hugging Face caches, optimizers and temporary files. Historical
energy logs are copied once and never opened for overwrite by the planner.

All diffuser notebooks expose `TRAIN_MODE`, `GENERATION_MODE`, `EVALUATION_MODE`, `FILTER_MODE`
and `PLAN_ONLY`. `auto` skips only content-verified checkpoint/image sets; an incomplete index set
causes generation/resume; `run` explicitly executes; `skip` fails for incomplete evidence; and
`recompute` is restricted to evaluation/filter products. Heavy subprocess cells are guarded by the
computed plan and print their decision. Classifier training notebooks expose `TRAIN_MODE`,
`VALIDATION_MODE`, and a mandatory `LOCKED_TEST_MODE="manual"`.

Local ZIPs, audit packages, snapshots and the exhaustive 43,075-row inventory live under
`../MammoDiffusion_local_archive/{packages,audits,patches,inventories,backups}`. Permanent runtime
and reproducibility material remains in `configs/`, `docs/`, `scripts/`, and `tests/`.

## Verified execution matrix

| Notebook | Checkpoint | Training required | Generation required | Metrics recomputable | Run All safe |
|---|---|---:|---:|---:|---:|
| Diffuser 01 | selected 3000; terminal 8000 | no | no | yes | yes |
| Diffuser 02 | selected 3000; terminal 8000 | no | no | yes | yes |
| Diffuser 03 | selected 4000; terminal 8000 | no | no | yes | yes |
| Diffuser 04 | LoRA 4500; final adapter | no | no | yes | yes |
| Diffuser 05 | selected 70000; terminal 80000 | no | no | yes | yes |
| Diffuser 06 | selected 70000; terminal 80000 | no | no | yes | yes |
| Diffuser 07 | best eval; terminal 150000 | no | no | yes | yes |
| Diffuser 08 | best eval; terminal 150000 | no | no | yes | yes |
| Classifier training notebooks | registry-specific | only where checkpoint is absent/unverified | n/a | yes | locked test remains manual |

“Safe” means that the content-aware local manifests validate. Moving or changing an artefact makes
`auto` run/resume or fail explicitly; directory existence alone is never sufficient.
