from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "assets/mammodiffusion_gradio/app.py"


class GradioSelectedGeneratorsTests(unittest.TestCase):
    def test_app_source_compiles_without_loading_gpu_frameworks(self):
        source = APP_PATH.read_text(encoding="utf-8")
        compile(source, str(APP_PATH), "exec")

    def test_app_uses_current_selected_generators_and_best_checkpoints(self):
        selected = json.loads((ROOT / "configs/selected_generators.json").read_text())
        registry_payload = json.loads((ROOT / "configs/generator_registry.json").read_text())
        registry = {entry["id"]: entry for entry in registry_payload["generators"]}

        self.assertEqual(selected["finetuned"], "02_sd21_filtered_100steps")
        self.assertEqual(selected["from_scratch"], "07_ldm_sdvae_extra1361")
        self.assertIn("checkpoint-3000/unet/diffusion_pytorch_model.safetensors", registry[selected["finetuned"]]["checkpoint"])
        self.assertTrue(registry[selected["from_scratch"]]["checkpoint"].endswith("ldm_unet_best_eval.keras"))

        best = json.loads(
            (
                ROOT
                / "results/2_diffusers/07_ldm_sdvae_extra1361/metrics/best_checkpoint.json"
            ).read_text()
        )
        self.assertEqual(best["best_checkpoint_id"], "step_130000")

    def test_app_uses_sd_vae_latent_sampling_for_g07(self):
        source = APP_PATH.read_text(encoding="utf-8")
        for required in (
            'CONFIGS_DIR / "selected_generators.json"',
            'CONFIGS_DIR / "generator_registry.json"',
            "make_compiled_latent_sampler",
            "decode_sd_latents_to_grayscale",
            'parameterization="eps"',
        ):
            self.assertIn(required, source)
        for retired in (
            "06_ldm_extra1361_fromscratch",
            "ldm_step070000.keras",
            "LDM_VAE_DECODER_PATH",
        ):
            self.assertNotIn(retired, source)

    def test_only_main_and_gradio_project_readmes_remain(self):
        allowed = {
            Path("README.md"),
            Path("assets/mammodiffusion_gradio/README.md"),
        }
        found = set()
        for path in ROOT.rglob("README*"):
            relative = path.relative_to(ROOT)
            relative_text = relative.as_posix()
            if relative_text.startswith("notebooks/utility/diffusers_repo/"):
                continue
            if relative_text.startswith("notebooks/pretrained_model/"):
                continue
            found.add(relative)
        self.assertEqual(found, allowed)


if __name__ == "__main__":
    unittest.main()
