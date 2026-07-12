# Cleanup inventory before push

| Path/pattern | Tracked | Referenced | Action | Reason / regenerability |
|---|---|---|---|---|
| `experiments/diffusers/**` | no | yes | keep local | checkpoints, generated datasets and model assets are scientifically required |
| `notebooks/pretrained_model/**` | no | yes | keep canonical symlink only | immutable shared SD2.1 base |
| experiment 04 `diffusers_repo` | no | no | archive after verification | clean duplicate of pinned checkout |
| experiment 03/04 base copies | no | no | archive after signature verification | byte-identical to canonical SD2.1 base |
| experiment 03/08 derived pipelines | no | yes | keep local | distinct signature; derived VAE content |
| `results/**/*.json`, `results/**/*.csv`, plots | mixed | yes | keep | compact scientific evidence, not blanket-ignored |
| `__pycache__`, `*.pyc`, runtime locks/claims | no | no | remove | regenerated automatically |
| `recovered_legacy_candidates/` | no | no | ignore | collision staging only |
| `../MammoDiffusion_local_archive/` | no | recovery only | keep outside Git | packages, audits, patches, inventories and reversible backups |

The pre-audit working tree already contained extensive notebook, utility, metric and classifier
pipeline changes. They are preserved as user work and reviewed/tested together; unrelated content
is not discarded.
