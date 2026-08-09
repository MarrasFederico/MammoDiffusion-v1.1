from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
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
            "assets/logo_MammoDiffusion.png",
            "assets/mammodiffusion_gradio/app.py",
            "assets/mammodiffusion_gradio/README.md",
            "requirements-dev.txt",
            "results/2_diffusers/benchmark/candidate_audit.csv",
            "results/2_diffusers/benchmark/generator_summary.csv",
            "results/2_diffusers/benchmark/generator_ranking.csv",
            "results/2_diffusers/benchmark/resampling_plan.json",
            "results/2_diffusers/benchmark/paired_generator_differences.csv",
            "results/2_diffusers/benchmark/selection_summary.json",
            "results/2_diffusers/benchmark/figures/generator_summary.png",
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
        self.assertIn("RUN_FINAL_EVALUATION", final)
        self.assertIn("OVERWRITE_TEST_PREDICTIONS", final)
        self.assertNotIn("RUN_TEST_INFERENCE", final)

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
        self.assertEqual(g05_positive, g05_experiment / "synthetic_filtered_positive")
        self.assertEqual(
            g05_negative,
            ROOT / "data/synthetic/05_ldm_basic_fromscratch/negative",
        )

        g06_experiment = ROOT / "experiments/diffusers/06_ldm_extra1361_fromscratch"
        g06_paths = get_experiment_paths(ROOT, g06_experiment, create=False)
        _, g06_positive = get_class_image_dirs(g06_paths, 1)
        self.assertEqual(g06_positive, g06_experiment / "synthetic_filtered_positive")

        g07_experiment = ROOT / "experiments/diffusers/07_ldm_sdvae_extra1361"
        g07_paths = get_experiment_paths(ROOT, g07_experiment, create=False)
        _, g07_positive = get_class_image_dirs(g07_paths, 1)
        self.assertEqual(
            g07_positive,
            ROOT / "data/synthetic/07_ldm_sdvae_extra1361/positive",
        )

    def test_both_ldm_raw_pools_are_first_class_experiment_paths(self):
        """Neither class may be reachable only through an inline literal.

        The negative RAW pool used to be rebuilt as a string inside
        ``get_class_image_dirs``, which is how ``reset_downstream_artifacts``
        came to wipe the positive pool and forget the negative one.
        """
        from notebooks.utility.ldm_project_paths import (
            get_class_image_dirs,
            get_experiment_paths,
        )

        experiment = ROOT / "experiments/diffusers/07_ldm_sdvae_extra1361"
        paths = get_experiment_paths(ROOT, experiment, create=False)
        positive_raw, _ = get_class_image_dirs(paths, 1)
        negative_raw, _ = get_class_image_dirs(paths, 0)
        self.assertEqual(positive_raw, paths.synthetic_raw_positive_dir)
        self.assertEqual(negative_raw, paths.synthetic_raw_negative_dir)
        self.assertEqual(positive_raw, experiment / "synthetic_raw_positive")
        self.assertEqual(negative_raw, experiment / "synthetic_raw_negative")

    def test_experiment_skeleton_does_not_fabricate_a_negative_pool(self):
        """G05 declares only the positive class; it must not gain an empty negative pool."""
        from notebooks.utility.ldm_project_paths import get_experiment_paths

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notebooks").mkdir()
            (root / "data").mkdir()
            experiment = root / "experiments/diffusers/05_ldm_basic_fromscratch"
            paths = get_experiment_paths(root, experiment, create=True)
            self.assertTrue(paths.synthetic_raw_positive_dir.is_dir())
            self.assertTrue(paths.synthetic_filtered_positive_dir.is_dir())
            self.assertFalse(paths.synthetic_raw_negative_dir.exists())

    def test_vae_reset_clears_every_pool_derived_from_the_old_decoder(self):
        """A retrained VAE invalidates both classes, RAW and filtered alike.

        Filtering selects from the RAW pool rather than regenerating, so a
        filtered pool that outlives its RAW pool is output of a decoder that no
        longer exists while still being the registered pool for the generator.
        """
        utility_dir = str(ROOT / "notebooks" / "utility")
        if utility_dir not in sys.path:
            sys.path.insert(0, utility_dir)
        import train_vae
        from ldm_project_paths import get_experiment_paths

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notebooks").mkdir()
            (root / "data").mkdir()
            experiment = root / "experiments/diffusers/06_ldm_extra1361_fromscratch"
            paths = get_experiment_paths(root, experiment, create=True)
            paths.synthetic_raw_negative_dir.mkdir(parents=True)
            negative_filtered = root / "data/synthetic/06_ldm_extra1361_fromscratch/negative"
            negative_filtered.mkdir(parents=True)
            for class_name in ("positive", "negative"):
                (paths.logs_dir / class_name).mkdir()
                (paths.logs_dir / class_name / "generation_raw_state.json").write_text("{}")
            (paths.logs_dir / "generation_raw_state.json").write_text("{}")
            (paths.logs_dir / "generation_summary.jsonl").write_text('{"phase": "x"}\n')
            stale = {
                "raw_positive": paths.synthetic_raw_positive_dir / "synth_00000.png",
                "raw_negative": paths.synthetic_raw_negative_dir / "synth_00000.png",
                "filtered_positive": paths.synthetic_filtered_positive_dir / "synth_filtered_0000.png",
                "filtered_negative": negative_filtered / "synth_filtered_0000.png",
                "latents": paths.latents_dir / "latent_stats.npz",
                "evaluation": paths.evaluation_dir / "best_checkpoint.json",
            }
            for path in stale.values():
                path.write_bytes(b"x")

            train_vae.reset_downstream_artifacts(paths)

            for name, path in stale.items():
                self.assertFalse(path.exists(), name)
            for name in ("synthetic_raw_positive_dir", "synthetic_raw_negative_dir",
                         "synthetic_filtered_positive_dir"):
                self.assertTrue(getattr(paths, name).is_dir(), name)
            self.assertTrue(negative_filtered.is_dir())
            for class_name in ("positive", "negative"):
                self.assertFalse((paths.logs_dir / class_name / "generation_raw_state.json").exists())
            self.assertFalse((paths.logs_dir / "generation_raw_state.json").exists())
            self.assertTrue((paths.logs_dir / "generation_summary.jsonl").exists())

    def test_retired_sibling_evaluation_layout_is_not_referenced(self):
        """Per-class evaluation lives under ``evaluation/<class>/``.

        The pre-unification layout wrote the negative class to a sibling
        ``evaluation_negative/`` directory; nothing may address it again.
        """
        for relative in PROJECT_NOTEBOOKS:
            notebook = nbformat.read(ROOT / relative, as_version=4)
            source = "\n".join(cell.source for cell in notebook.cells)
            self.assertNotIn("evaluation_negative", source, relative)
        for path in sorted((ROOT / "notebooks" / "utility").glob("*.py")):
            self.assertNotIn("evaluation_negative", path.read_text(), path.name)

    def test_diffuser_notebooks_declare_explicit_phase_flags_defaulting_to_off(self):
        """Phases are plain booleans, off until the operator flips one.

        The mode machinery (``artifact_phase_planner``) is retired. A notebook
        that reintroduces it, or that ships a phase already enabled, would
        retrain or regenerate on an ordinary Run All.
        """
        self.assertFalse((ROOT / "notebooks/utility/artifact_phase_planner.py").exists())
        retired = ("artifact_phase_planner", "phase_should_run", "PHASE_PLAN", "PLAN_ONLY",
                   "TRAIN_MODE", "GENERATION_MODE", "EVALUATION_MODE", "FILTER_MODE")
        for relative in PROJECT_NOTEBOOKS:
            if "2_diffusers" not in relative:
                continue
            notebook = nbformat.read(ROOT / relative, as_version=4)
            code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
            self.assertIn("EXPLICIT_PHASE_FLAGS_V1", code, relative)
            for token in retired:
                self.assertNotIn(token, code, f"{relative}: {token}")
            for phase in ("TRAINING", "GENERATION", "EVALUATION", "FILTER"):
                self.assertIn(f"RUN_{phase}_PHASE = False", code, f"{relative}: {phase}")
            for name, value in re.findall(r"^(RUN_\w+_PHASE) = (\w+)", code, re.M):
                self.assertEqual(value, "False", f"{relative}: {name}")

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
        self.assertIn("verify_g07_validation_cache", g07)
        self.assertIn("verify_g08_validation_cache", g08)
        self.assertNotIn('\\"test\\"', g07)
        self.assertNotIn('\\"test\\"', g08)
        self.assertIn("finally:", g08)

    def test_tracked_result_records_reference_existing_notebooks(self):
        """Path-shaped `notebook` fields must resolve.

        Records written by the preprocessing and LDM stages store a bare
        notebook identifier instead of a path; only entries that spell out a
        repository-relative location are checked here.
        """
        if not (ROOT / ".git").exists():
            self.skipTest("Git metadata unavailable in source archive")
        tracked = subprocess.run(
            ["git", "ls-files", "results"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.splitlines()
        checked = []
        for relative in tracked:
            if not relative.endswith(".json"):
                continue
            try:
                payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.fail(f"{relative} is not readable JSON")
            if not isinstance(payload, dict):
                continue
            reference = payload.get("notebook")
            if not isinstance(reference, str) or "/" not in reference:
                continue
            checked.append(relative)
            self.assertTrue(
                (ROOT / reference).is_file(),
                f"{relative} points at a missing notebook: {reference}",
            )
        self.assertTrue(checked, "no path-shaped notebook reference was inspected")

if __name__ == "__main__": unittest.main()
