# Shared Diffusers and SD2.1 assets

The only canonical clean Diffusers checkout is
`experiments/diffusers/03_sd21_vae_finetuned/diffusers_repo`, pinned at
`3759fab56d3170a04d747e918a13e55fda6681e2` with remote
`https://github.com/huggingface/diffusers`. Its working tree was clean at the recovery audit.
All code resolves it through `notebooks/utility/shared_diffusers_assets.py`; arbitrary experiment
fallbacks are forbidden.

The project-facing immutable SD2.1 base is
`notebooks/pretrained_model/stable-diffusion-2-1-base`. It is a compatibility symlink to the one
physical canonical copy under experiment 07. Its content-aware SHA-256 is
`b7cf9e437c92e45c0a6556cf790da6eabec7506a5f2d9a8aa9ea99af7f0040c0` (44 files,
5,161,728,353 bytes). Override only with `MAMMODIFFUSION_SD21_BASE`.

Two different signatures are deliberately retained as derived models, not bases:

- experiment 03 `pretrained_model_vaeft/...`: VAE-fine-tuned pipeline,
  SHA-256 `c4b64d590937690a9eb4c065d3c9b4e67b21b5fc7d04d9ba09e9477b5a2050f0`;
- experiment 08 `pretrained_model/...`: the same derived SD-VAE layout used by the LDM v3
  experiment, with the same signature. It remains local pending extraction of its unique VAE.

The LoRA is stored as an adapter and points to the canonical base; it does not require a complete
base copy. `tests.test_shared_diffusers_assets` verifies root discovery, pinning, signatures and
notebook resolution. Heavy assets are ignored by Git and must be restored locally before real runs.
