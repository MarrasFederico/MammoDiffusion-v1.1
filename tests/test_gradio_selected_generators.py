from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "assets/mammodiffusion_gradio/app.py"
RESULT_CATEGORY_PREFIXES = ("1_", "2_", "3_", "4_", "5_")


class _GradioStubValue:
    """Accepts any attribute access or call, so module-level theme setup runs."""

    def __getattr__(self, name):
        return _GradioStubValue()

    def __call__(self, *args, **kwargs):
        return _GradioStubValue()


def import_app_configuration():
    """Import app.py far enough to expose its module-level paths.

    The module level only reads the three tracked JSON records; it never loads
    weights, CUDA, or the image cohort. Gradio is stubbed because the demo
    dependency is not part of the light test environment.
    """
    stub = types.ModuleType("gradio")
    stub.__getattr__ = lambda name: _GradioStubValue()
    spec = importlib.util.spec_from_file_location("mammodiffusion_gradio_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("gradio")
    sys.modules["gradio"] = stub
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
        if previous is None:
            sys.modules.pop("gradio", None)
        else:
            sys.modules["gradio"] = previous
    return module


def gradio_is_installed() -> bool:
    return importlib.util.find_spec("gradio") is not None


class GradioProjectRootTests(unittest.TestCase):
    """The demo must anchor on this checkout, not on a directory name.

    ``app.py`` derives every asset path from ``PROJECT_ROOT``. On the original
    workstation a sibling symlink called ``MammoDiffusion`` points at a separate
    successor project, so a name-based root would make the demo read another
    repository's configuration.
    """

    def test_project_root_is_this_checkout_and_not_a_successor_project(self):
        module = import_app_configuration()
        project_root = Path(module.PROJECT_ROOT).resolve()
        self.assertEqual(project_root, ROOT.resolve())
        self.assertNotIn("MammoDiffusion-v2", str(project_root))
        self.assertTrue((project_root / "configs" / "selected_generators.json").is_file())

    def test_every_absolute_module_path_stays_inside_the_checkout(self):
        module = import_app_configuration()
        runtime_prefix = Path(sys.prefix).resolve()
        escaping = []
        for name, value in vars(module).items():
            if not isinstance(value, Path) or not value.is_absolute():
                continue
            resolved = value.resolve()
            if resolved.is_relative_to(ROOT.resolve()):
                continue
            # The interpreter prefix is a legitimate runtime location, not an asset.
            if resolved == runtime_prefix or resolved.is_relative_to(runtime_prefix):
                continue
            escaping.append((name, str(resolved)))
        self.assertEqual(escaping, [], f"app paths escape the checkout: {escaping}")


@unittest.skipUnless(gradio_is_installed(), "gradio is not part of the light test environment")
class GradioInterfaceSmokeTests(unittest.TestCase):
    """Build the real UI once. No model, no CUDA, no server."""

    def test_interface_builds_and_is_named_for_this_release(self):
        spec = importlib.util.spec_from_file_location("mammodiffusion_gradio_app_real", APP_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            demo = module.build_demo()
        finally:
            sys.modules.pop(spec.name, None)
        self.assertIsNotNone(demo)
        self.assertIn("MammoDiffusion", str(getattr(demo, "title", "")))
        # Building must not have started anything.
        self.assertFalse(getattr(demo, "is_running", False))


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

    def test_demo_uses_memory_safe_cross_framework_defaults(self):
        module = import_app_configuration()
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertEqual(module.LDM_VAE_DEVICE, "cpu")
        self.assertTrue(module.SD_MODEL_CPU_OFFLOAD)
        self.assertIn("pipeline.enable_model_cpu_offload()", source)
        self.assertIn("device=LDM_VAE_DEVICE", source)
        self.assertIn('torch.Generator(device="cpu")', source)

    def test_app_configuration_imports_and_resolves_the_g07_selection_record(self):
        module = import_app_configuration()
        expected = (
            ROOT
            / "results"
            / "2_diffusers"
            / "07_ldm_sdvae_extra1361"
            / "metrics"
            / "best_checkpoint.json"
        )
        self.assertEqual(module.LDM_BEST_SELECTION_PATH, expected)
        self.assertTrue(expected.is_file(), expected)
        self.assertEqual(module.LDM_BEST_CHECKPOINT_ID, "step_130000")
        self.assertEqual(module.LDM_GENERATOR_ID, "07_ldm_sdvae_extra1361")
        self.assertEqual(module.SD_GENERATOR_ID, "02_sd21_filtered_100steps")

    def test_app_result_paths_stay_inside_numbered_result_categories(self):
        module = import_app_configuration()
        results_root = ROOT / "results"
        checked = []
        for name, value in vars(module).items():
            if not isinstance(value, Path):
                continue
            relative = value.relative_to(results_root) if value.is_relative_to(results_root) else None
            if relative is None or not relative.parts:
                continue
            category = relative.parts[0]
            checked.append(name)
            self.assertTrue(
                category.startswith(RESULT_CATEGORY_PREFIXES),
                f"{name} resolves to the retired results/{category}/ namespace: {value}",
            )
        self.assertTrue(checked, "no module-level results path was inspected")

    def test_readme_scan_ignores_results_and_runtime_namespaces(self):
        allowed = {
            Path("README.md"),
            Path("assets/mammodiffusion_gradio/README.md"),
        }
        found = set()
        for path in ROOT.rglob("README*"):
            relative = path.relative_to(ROOT)
            relative_text = relative.as_posix()
            if relative_text.startswith((".cache/", ".pytest_cache/")):
                continue
            if relative_text.startswith("notebooks/utility/diffusers_repo/"):
                continue
            if relative_text.startswith("notebooks/pretrained_model/"):
                continue
            if relative_text.startswith("results/"):
                continue
            found.add(relative)
        self.assertEqual(found, allowed)


if __name__ == "__main__":
    unittest.main()
