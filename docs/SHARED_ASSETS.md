# Shared Diffusers and SD2.1 assets

The only canonical clean Diffusers checkout is
`notebooks/utility/diffusers_repo`, pinned at
`3759fab56d3170a04d747e918a13e55fda6681e2` with remote
`https://github.com/huggingface/diffusers`. Its working tree was clean at the recovery audit.
All code resolves it through `notebooks/utility/shared_diffusers_assets.py`; arbitrary experiment
fallbacks are forbidden.

The project-facing immutable SD2.1 base is the single physical directory
`notebooks/pretrained_model/stable-diffusion-2-1-base`. Its content-aware SHA-256 is
`f15a779ba591c53aff05b66a9e94fdb09c6c4ba63709344b8b02fcd557f47351` (18 files,
5,161,725,454 bytes). Disposable Hugging Face cache metadata is intentionally excluded.
Override only with `MAMMODIFFUSION_SD21_BASE`.

The fine-tuned VAE is retained once as an experiment-03 output:

- `experiments/diffusers/03_sd21_vae_finetuned/vae_finetuning/vae_finetuned`.

Notebook 03 creates a lightweight relative-symlink view under `.cache/mammodiffusion/` when the
full SD pipeline with the fine-tuned VAE is needed. LDM notebooks load a standalone VAE directly,
so experiments never contain another physical SD2.1 base copy.

The LoRA is stored as an adapter and points to the canonical base; it does not require a complete
base copy. `tests.test_shared_diffusers_assets` verifies root discovery, pinning, signatures and
notebook resolution. Heavy assets are ignored by Git and must be restored locally before real runs.

## Google Drive transfer policy

For a complete hand-off, upload `notebooks/`, `configs/`, `experiments/`, `results/` and the data
material allowed by the project. In particular, include the physical directories
`notebooks/utility/diffusers_repo` (including its nested `.git`, needed to verify the pinned commit)
and `notebooks/pretrained_model/stable-diffusion-2-1-base`. Git-ignore rules only prevent accidental
commits of these heavy assets; they do not mean that the assets should be omitted from Drive.

Uploading `experiments/diffusers/` first is safe, but it is only the experiment-state tranche:
the notebooks resolve the shared code checkout and SD2.1 base from the two `notebooks/` paths
above, and filtered synthetic sets from `data/synthetic/`. Upload those paths in a later tranche
before another machine attempts to run or evaluate the project.

Keep scientific restart/evaluation state: canonical checkpoint histories, latent archives,
checkpoint-validation caches, evaluation outputs and embedding caches. Do not upload disposable
runtime state: `.cache/huggingface`, `.cache/mammodiffusion`, `__pycache__`, `*.pyc`, smoke-test
outputs or empty work queues. The VAE-composed SD pipeline under `.cache/mammodiffusion` is rebuilt
automatically from the shared SD2.1 base and the standalone experiment-03 VAE.

Checkpoint ownership is intentionally unique:

- experiment 01 reads the full SD checkpoint history from experiment 02;
- experiment 06 reads the full LDM checkpoint history from experiment 05;
- every other experiment retains its own checkpoint history.
