from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "notebooks").rglob("*.ipynb"))
PROJECT_NOTEBOOKS = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "notebooks").glob("*/*.ipynb"))


class PublicationRepositoryTests(unittest.TestCase):
    def test_notebooks_validate_and_code_cells_compile(self):
        transformer = TransformerManager()
        for relative in NOTEBOOKS:
            notebook = nbformat.read(ROOT / relative, as_version=4); nbformat.validate(notebook)
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    compile(transformer.transform_cell(cell.source), f"{relative}:cell-{index}", "exec")

    def test_main_publication_files_exist(self):
        required = (
            "notebooks/04_classifiers/01_MaxViT512.ipynb",
            "notebooks/04_classifiers/02_MammoFM.ipynb",
            "notebooks/04_classifiers/03_Validation_Comparison.ipynb",
            "notebooks/04_classifiers/04_Final_Evaluation_and_Report.ipynb",
            "notebooks/utility/classifier_experiment.py",
            "notebooks/utility/classifier_architecture_adapters.py",
        )
        self.assertTrue(all((ROOT / path).is_file() for path in required))

    def test_project_notebook_code_has_no_workstation_specific_gpu_or_home_path(self):
        literal_gpu_uuid = re.compile(r"GPU-[0-9a-fA-F]{8,}(?:-[0-9a-fA-F]+)*")
        fixed_training_ordinal = re.compile(
            r"TRAIN_GPU_VISIBLE_DEVICES\s*=\s*['\"]\d+['\"]"
        )
        literal_user_home = re.compile(r"(?:/home|/Users)/[^/\s'\"]+")
        for relative in PROJECT_NOTEBOOKS:
            notebook_path = ROOT / relative
            notebook = nbformat.read(notebook_path, as_version=4)
            code = "\n".join(
                cell.source for cell in notebook.cells if cell.cell_type == "code"
            )
            self.assertIsNone(literal_gpu_uuid.search(code), relative)
            self.assertIsNone(literal_gpu_uuid.search(notebook_path.read_text()), relative)
            self.assertIsNone(fixed_training_ordinal.search(code), relative)
            self.assertIsNone(literal_user_home.search(code), relative)
            self.assertNotIn('"python3.11"', code, relative)
            self.assertNotIn('"cu13"', code, relative)

    def test_final_evaluation_guard_exists(self):
        final = (ROOT / "notebooks/04_classifiers/04_Final_Evaluation_and_Report.ipynb").read_text()
        self.assertIn("RUN_TEST_INFERENCE", final)
        self.assertIn("split='test'", final)

    def test_selection_records_g02_g07(self):
        # configs/selected_generators.json is the committed authoritative record of the selection.
        import json
        selection = ROOT / "configs/selected_generators.json"
        self.assertTrue(selection.exists())
        payload = json.loads(selection.read_text())
        self.assertEqual(payload["finetuned"], "02_sd21_filtered_100steps")
        self.assertEqual(payload["from_scratch"], "07_ldm_sdvae_extra1361")
        self.assertFalse(payload["test_access"])

    def test_no_tracked_model_cache_or_archive_artifacts(self):
        if not (ROOT / ".git").exists():
            self.skipTest("Git metadata unavailable in source archive")
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
        forbidden = (".pt", ".pth", ".ckpt", ".keras", ".h5", ".safetensors", ".zip", ".tar.gz", ".tgz", ".pyc", ".pyo")
        self.assertEqual([path for path in tracked if path.endswith(forbidden)], [])
        self.assertFalse(any("__pycache__" in path for path in tracked))

    def test_ldm_results_default_uses_canonical_diffusers_namespace(self):
        utility_dir = ROOT / "notebooks/utility"
        paths_source = (utility_dir / "ldm_project_paths.py").read_text()
        self.assertIn('RESULTS_STAGE_NAME = "2_diffusers/', paths_source)
        self.assertNotIn("keras_v2", paths_source)
        for filename in (
            "02_SD21_Filtered_100steps.ipynb",
            "03_SD21_VAE_FineTuned.ipynb",
            "04_SD21_LoRA.ipynb",
        ):
            notebook = nbformat.read(ROOT / "notebooks/2_diffusers" / filename, as_version=4)
            source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
            self.assertIn('RESULTS_DIR = PROJECT_ROOT / "results" / "2_diffusers"', source, filename)
            self.assertNotIn('RESULTS_DIR / "diffusers/', source, filename)
        for filename in (
            "train_vae.py",
            "train_ldm.py",
            "evaluate_ldm.py",
            "generate_ldm.py",
            "evaluate_filtered_ldm.py",
        ):
            source = (utility_dir / filename).read_text()
            self.assertIn("RESULTS_STAGE_NAME", source, filename)
            self.assertNotIn("keras_v2", source, filename)

    def test_ldm_filtered_pool_paths_preserve_registered_generator_identity(self):
        from notebooks.utility.ldm_project_paths import (
            get_class_image_dirs,
            get_experiment_paths,
        )

        g05_experiment = ROOT / "experiments/diffusers/05_ldm_basic_fromscratch"
        g05_paths = get_experiment_paths(ROOT, g05_experiment, create=False)
        _, g05_positive = get_class_image_dirs(g05_paths, 1)
        _, g05_negative = get_class_image_dirs(g05_paths, 0)
        self.assertEqual(g05_positive, g05_experiment / "synthetic_filtered")
        self.assertEqual(
            g05_negative,
            ROOT / "data/synthetic/05_ldm_basic_fromscratch/negative",
        )

        g06_experiment = ROOT / "experiments/diffusers/06_ldm_extra1361_fromscratch"
        g06_paths = get_experiment_paths(ROOT, g06_experiment, create=False)
        _, g06_positive = get_class_image_dirs(g06_paths, 1)
        self.assertEqual(g06_positive, g06_experiment / "synthetic_filtered")

        g07_experiment = ROOT / "experiments/diffusers/07_ldm_sdvae_extra1361"
        g07_paths = get_experiment_paths(ROOT, g07_experiment, create=False)
        _, g07_positive = get_class_image_dirs(g07_paths, 1)
        self.assertEqual(
            g07_positive,
            ROOT / "data/synthetic/07_ldm_sdvae_extra1361/positive",
        )

    def test_ldm_notebooks_recheck_metric_caches_independently(self):
        notebook_dir = ROOT / "notebooks/2_diffusers"
        g05 = (notebook_dir / "05_LDM_Basic_FromScratch.ipynb").read_text()
        g06 = (notebook_dir / "06_LDM_Extra1361_FromScratch.ipynb").read_text()
        g07 = (notebook_dir / "07_LDM_SDVAE_Extra1361.ipynb").read_text()
        g08 = (notebook_dir / "08_LDM_v3_SDVAE_FromScratch.ipynb").read_text()

        self.assertIn("ACTIVE_G05_FILTERED_DIR", g05)
        self.assertIn("ACTIVE_G06_FILTERED_DIR", g06)
        self.assertIn('GEN_MODE = \\"both\\" if RUN_GENERATION_PHASE else \\"filter\\"', g05)
        self.assertIn('GEN_MODE = \\"both\\" if RUN_GENERATION_PHASE else \\"filter\\"', g06)
        self.assertIn("verify_g07_metric_cache", g07)
        self.assertIn("verify_g08_metric_cache", g08)
        self.assertIn('for mode in (\\"validate\\", \\"test\\")', g07)
        self.assertIn('for mode in (\\"validate\\", \\"test\\")', g08)
        self.assertIn("finally:", g08)

if __name__ == "__main__": unittest.main()
