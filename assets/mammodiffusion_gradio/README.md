# MammoDiffusion Studio

Local Gradio interface, inspired by Fooocus's minimal layout, for generating
MLO mammograms with two MammoDiffusion generators selectable from the GUI.

The app reads the current selection from `configs/selected_generators.json` and uses:

- G02, fine-tuned Stable Diffusion 2.1: `checkpoint-3000`;
- G07, an SD-VAE LDM trained from scratch: `ldm_unet_best_eval.keras`, selected
  by the sweep at `step_130000`;
- the positive and negative prompts defined for SD2.1 fine-tuning;
- `100` inference steps and guidance scale `7.5` as the G02 defaults;
- `100` sampling steps and guidance scale `1.5` as the G07 defaults.

Images are generated sequentially to limit VRAM use and saved under
`assets/mammodiffusion_gradio/outputs/`. This directory is separate from
`data/synthetic/02_sd21_filtered_100steps`, so using the demo does not modify the
canonical dataset or final results.

Each generation request runs in a dedicated subprocess. When it exits, the
system releases the VRAM used by the selected model. This allows switching
between G02 and G07 without restarting Gradio. Worker diagnostics are stored
in `worker.log` inside the request's output directory.

## Prerequisites

Weights are not published on GitHub. The following files must exist locally
before the app starts:

- `notebooks/pretrained_model/stable-diffusion-2-1-base`;
- `experiments/diffusers/02_sd21_filtered_100steps/model/checkpoint-3000/unet`;
- `experiments/diffusers/07_ldm_sdvae_extra1361/checkpoints_ldm/ldm_unet_best_eval.keras`;
- `experiments/diffusers/07_ldm_sdvae_extra1361/checkpoints_ldm/ldm_step130000.keras`;
- `experiments/diffusers/07_ldm_sdvae_extra1361/latents/latent_stats.npz`;
- `results/2_diffusers/07_ldm_sdvae_extra1361/metrics/best_checkpoint.json`.

## Launch

From the project root, using an environment with PyTorch/Diffusers and
TensorFlow/Keras installed:

```bash
python -m pip install -r assets/mammodiffusion_gradio/requirements.txt
python assets/mammodiffusion_gradio/app.py --open-browser
```

The interface will be available at <http://127.0.0.1:7860>.

To expose it on the local network:

```bash
python assets/mammodiffusion_gradio/app.py --host 0.0.0.0
```

The `--share` option creates a temporary public Gradio link. Do not use it with
sensitive data.

## Controls

- **Model** selects the best G02 or G07 checkpoint.
- **Label** selects the standard positive or negative prompt.
- **Number of images** generates between 1 and 12 images.
- **Initial seed** uses `-1` for a random seed; an explicit integer makes the
  generation reproducible.
- Advanced settings change inference steps and guidance scale without altering
  the experiment defaults.

The demo is intended only for research and presentation, not for clinical use.

## Examples

Reproducible historical outputs generated with the positive label, G02
`checkpoint-3000` weights, and consecutive seeds starting at `42` are available
under [`examples/`](examples/). They may come from the earlier 50-step ablation
and do not represent the current canonical 100-step default.

| Seed 42 | Seed 43 |
|---|---|
| ![Positive output, seed 42](examples/positive_seed_42.png) | ![Positive output, seed 43](examples/positive_seed_43.png) |

| Seed 44 | Seed 45 |
|---|---|
| ![Positive output, seed 44](examples/positive_seed_44.png) | ![Positive output, seed 45](examples/positive_seed_45.png) |
