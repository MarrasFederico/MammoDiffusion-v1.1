from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import matplotlib
import nbformat
import numpy as np
from PIL import Image

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import classifier_checkpoint_io as checkpoint_io  # noqa: E402
import downstream_analysis as da  # noqa: E402
import downstream_experiment as de  # noqa: E402
import downstream_protocol as dp  # noqa: E402
import final_evaluation as fe  # noqa: E402
import generator_benchmark as gb  # noqa: E402


class GeneratorCompletionTests(unittest.TestCase):
    def _images(self, directory: Path) -> list[Path]:
        paths = []
        for index, value in enumerate((20, 80, 160, 220)):
            array = np.full((64, 64), value, dtype=np.uint8)
            array[index * 4:(index + 1) * 4, :] = 255 - value
            path = directory / f"image_{index}.png"
            Image.fromarray(array).save(path)
            paths.append(path)
        return paths

    def test_notebook_05_has_full_wiring_and_no_permanent_empty_placeholders(self):
        text = (ROOT / "notebooks/3_generator_benchmark/05_Unified_Generator_Benchmark.ipynb").read_text()
        self.assertIn("RUN_REAL_BENCHMARK = False", text)
        for forbidden in ("diversity_results = {}", "train_memorization_rows = []",
                          "validation_similarity_rows = []", "results_table = []"):
            self.assertNotIn(forbidden, text)
        for token in ("technical_validity_row", "get_or_extract_embeddings", "repeated_distribution_metrics",
                      "diversity_metrics", "build_train_memorization_rows", "build_validation_similarity_rows",
                      "build_synthetic_duplication_rows", "generator_summary.csv", "generator_ranking.csv",
                      "plot_generator_summary", "render_similarity_panel"):
            self.assertIn(token, text)

    def test_notebook_05_false_mode_executes_without_mutation_or_real_references(self):
        output = ROOT / gb.BENCHMARK_ROOT
        before = {(path.relative_to(output).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
                  for path in output.rglob("*") if path.is_file()} if output.exists() else set()
        namespace = {}
        notebook = nbformat.read(ROOT / "notebooks/3_generator_benchmark/05_Unified_Generator_Benchmark.ipynb", 4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                exec(compile(cell.source, f"notebook05:{index}", "exec"), namespace)
        after = {(path.relative_to(output).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
                 for path in output.rglob("*") if path.is_file()} if output.exists() else set()
        self.assertFalse(namespace["RUN_REAL_BENCHMARK"])
        self.assertEqual(namespace["reference_status"]["status"], "Deferred: no real dataset was opened")
        self.assertIsNone(namespace["generator_summary"])
        self.assertEqual(before, after)

    def test_technical_validity_schema_is_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._images(Path(temporary))
            row = gb.technical_validity_row("g", "raw", paths, minimum_unique=4, expected_size=(64, 64))
        expected = {"generator_id", "condition", "n_discovered", "n_readable", "n_corrupt", "n_wrong_shape",
                    "n_near_black", "n_invalid_range", "n_technically_valid", "n_technically_invalid",
                    "n_unique_valid_content", "n_exact_duplicates_among_valid", "technical_validity_rate",
                    "eligible_for_distribution_metrics", "failure_reason"}
        self.assertEqual(set(row), expected)
        self.assertTrue(row["eligible_for_distribution_metrics"])

    def test_ms_ssim_pair_sampling_and_diversity_are_deterministic(self):
        self.assertEqual(gb.deterministic_pair_indices(20, 15, 73), gb.deterministic_pair_indices(20, 15, 73))
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._images(Path(temporary))
            features = np.arange(16, dtype=float).reshape(4, 4)
            first = gb.diversity_metrics(paths, features, pair_count=4, seed=17)
            second = gb.diversity_metrics(paths, features, pair_count=4, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first["evaluated_pairs"], 4)
        self.assertEqual(first["lpips_status"], "optional_not_evaluated")

    def test_similarity_outputs_are_separate_and_train_only_has_memorization_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._images(Path(temporary))
            ids = [path.name for path in paths]
            mapping = dict(zip(ids, paths))
            features = np.arange(16, dtype=float).reshape(4, 4)
            train = gb.build_train_memorization_rows(features, features, ids, ids, mapping, mapping,
                                                     {"ssim_gte": .98, "perceptual_hash_distance_lte": 2})
            validation = gb.build_validation_similarity_rows(features, features, ids, ids, mapping, mapping)
            synthetic = gb.build_synthetic_duplication_rows(features, ids, mapping,
                                                             {"ssim_gte": .98, "perceptual_hash_distance_lte": 2})
        self.assertIn("memorization_flag", train[0])
        self.assertNotIn("memorization_flag", validation[0])
        self.assertIn("nearest_validation_id", validation[0])
        self.assertIn("duplicate_flag", synthetic[0])

    def _ranking_row(self, generator_id: str, family: str, eligible: bool, kid: float):
        return {"generator_id": generator_id, "family": family, "eligible_for_selection": eligible,
                "valid_positive_images": 1361, "synthetic_exact_duplicate_rate": 0,
                "perceptual_hash_duplicate_rate": 0, "train_memorization_rate": 0,
                "raddino_coverage": .8, "filter_acceptance_rate": .9, "metrics_complete": True,
                "lineage_complete": True, "provenance_manifest_valid": True,
                "filter_manifest_valid": True, "filter_provenance_complete": True, "n_corrupt": 0,
                "training_corpus_manifest_valid": True,
                "test_access": False, "raddino_kid": kid, "raddino_precision": .8,
                "raddino_fid": 3, "inception_kid": .2, "raddino_kid_std": .01}

    def test_ranking_is_deterministic_and_never_proposes_ineligible_roles(self):
        protocol = gb.load_protocol(ROOT)
        rows = [self._ranking_row("02_primary", "finetuned", True, .1),
                self._ranking_row("01_sd21_baseline_50steps", "finetuned", False, .01),
                self._ranking_row("06_primary", "from_scratch", True, .2),
                self._ranking_row("05_ldm_basic_fromscratch", "from_scratch", False, .001)]
        first = gb.rank_generator_family(rows, "finetuned", protocol["eligibility_gates"])
        second = gb.rank_generator_family(list(reversed(rows)), "finetuned", protocol["eligibility_gates"])
        self.assertEqual([row["generator_id"] for row in first], [row["generator_id"] for row in second])
        self.assertEqual(next(row["generator_id"] for row in first if row["eligible"]), "02_primary")
        fs = gb.rank_generator_family(rows, "from_scratch", protocol["eligibility_gates"])
        self.assertEqual(next(row["generator_id"] for row in fs if row["eligible"]), "06_primary")

    def test_selection_notebook_has_proposal_override_and_explicit_save_guard(self):
        text = (ROOT / "notebooks/3_generator_benchmark/06_Generator_Selection.ipynb").read_text()
        for token in ("PROPOSED_FINETUNED_GENERATOR", "PROPOSED_FROM_SCRATCH_GENERATOR",
                      "SELECTED_FINETUNED_GENERATOR = PROPOSED_FINETUNED_GENERATOR",
                      "SELECTED_FROM_SCRATCH_GENERATOR = PROPOSED_FROM_SCRATCH_GENERATOR",
                      "SAVE_SELECTION = False", "manual_override"):
            self.assertIn(token, text)
        self.assertIn("sampling_ablations", text)
        self.assertIn("descriptive_baselines", text)


class GPUResumeAndVisualizationTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def test_requested_gpu_matches_visible_gpu(self):
        result = de.configure_visible_gpu(2, probe=lambda: {"visible_count": 1, "local_index": 0,
                                                            "physical_index": 2, "name": "fixture", "memory_bytes": 1024})
        self.assertEqual((result["requested_physical_index"], result["local_index"]), (2, 0))

    def test_gpu_mismatch_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "Requested physical GPU 2"):
            de.configure_visible_gpu(2, probe=lambda: {"visible_count": 1, "local_index": 0, "physical_index": 1})

    def test_initialized_framework_requests_kernel_restart(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "Restart the kernel"):
                de.configure_visible_gpu(1, probe=lambda: {})

    def _configuration(self, resume: bool, confirm: bool = False):
        return {"architecture": "maxvit512", "condition": "real_only", "seed": 17,
                "experiment_id": "maxvit512__real_only__seed17", "training_policy_name": "maxvit512_fixed_protocol",
                "resume": resume, "confirm_existing_output": confirm}

    def _checkpoint(self, root: Path):
        run = de.experiment_dir(root, "maxvit512", "real_only", 17)
        expected = {"architecture": "maxvit512", "experiment_id": "maxvit512__real_only__seed17",
                    "dataset_variant_id": "real_only", "training_policy": "maxvit512_fixed_protocol",
                    "config_signature": "informational", "dataset_signature": "informational", "seed": 17}
        checkpoint_io.save_resume_checkpoint(run, {**expected, "epoch": 3, "global_step": 400})

    def test_resume_true_loads_compatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._checkpoint(root)
            status = de.resume_status(root, self._configuration(True))
        self.assertTrue(status["compatible"])
        self.assertEqual((status["resume_epoch"], status["resume_step"]), (3, 400))

    def test_resume_false_never_loads_and_refuses_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._checkpoint(root)
            with self.assertRaisesRegex(RuntimeError, "RESUME=False"):
                de.resume_status(root, self._configuration(False))
            with mock.patch.object(checkpoint_io, "load_resume_checkpoint", side_effect=AssertionError("must not load")):
                status = de.resume_status(root, self._configuration(False, confirm=True))
            self.assertFalse(status["resume_requested"])
            self.assertIsNone(status["resumed_from"])

    def test_visualization_helpers_return_figures_and_dataframe(self):
        history = [{"epoch": 1, "loss": .8, "val_loss": .9, "val_pr_auc": .4, "val_auc": .6, "learning_rate": 1e-4, "optimizer_steps": 10},
                   {"epoch": 2, "loss": .6, "val_loss": .7, "val_pr_auc": .5, "val_auc": .7, "learning_rate": 5e-5, "optimizer_steps": 20}]
        rows = [{"patient_id": f"p{i}", "image_id": f"i{i}", "label": i % 2,
                 "probability": (.8 if i % 2 else .2), "processed_path": f"x/{i}.png"} for i in range(8)]
        self.assertIsNotNone(de.plot_training_history(history))
        self.assertIsNotNone(de.plot_validation_curves(rows))
        self.assertIsNotNone(de.plot_calibration(rows))
        table = de.build_error_case_table(rows)
        self.assertEqual(list(table.columns), ["patient_id", "image_id", "label", "probability", "error_type", "source/path"])

    def test_pr_auc_epoch_four_wins_for_both_adapters(self):
        history = [{"epoch": 3, "val_auc": .82, "val_pr_auc": .41, "val_loss": .4},
                   {"epoch": 4, "val_auc": .80, "val_pr_auc": .46, "val_loss": .5}]
        for architecture in ("maxvit512", "mammofm"):
            with self.subTest(architecture=architecture):
                self.assertEqual(de.select_best_epoch(history)["epoch"], 4)


class EnsembleFinalAndArchiveTests(unittest.TestCase):
    def _rows(self):
        return [{"patient_id": "p0", "image_id": "i0", "label": 0, "probability": .2},
                {"patient_id": "p1", "image_id": "i1", "label": 1, "probability": .8}]

    def test_eight_logical_ensembles_and_three_seeds(self):
        self.assertEqual(len(dp.ARCHITECTURES) * len(dp.CONDITIONS), 8)
        self.assertEqual(dp.SEEDS, (17, 42, 73))

    def test_missing_duplicate_misaligned_and_nonfinite_rows_are_rejected(self):
        good = {seed: self._rows() for seed in dp.SEEDS}
        duplicate = {seed: self._rows() for seed in dp.SEEDS}
        duplicate[73].append(dict(duplicate[73][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            da.align_seed_predictions(duplicate)
        missing = {seed: self._rows() for seed in dp.SEEDS}
        missing[42] = missing[42][:-1]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            da.align_seed_predictions(missing)
        nonfinite = {seed: self._rows() for seed in dp.SEEDS}
        nonfinite[17][0]["probability"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            da.align_seed_predictions(nonfinite)
        self.assertEqual(len(da.align_seed_predictions(good)), 2)

    def test_final_false_does_not_access_adapter_and_missing_adapter_is_clear(self):
        class Trap(fe.FinalEvaluationDatasetAdapter):
            def load_manifest(self, root):
                raise AssertionError("final data accessed")
        with self.assertRaises(PermissionError):
            fe.run_final_evaluation(ROOT, run_final_evaluation=False, checklist={}, adapter=Trap())
        complete = {item: True for item in fe.REQUIRED_CHECKLIST}
        with self.assertRaisesRegex(RuntimeError, "No final evaluation dataset adapter"):
            fe.run_final_evaluation(ROOT, run_final_evaluation=True, checklist=complete, adapter=None)

    def test_report_uses_markdown_tables_and_missing_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs/generator_benchmark_protocol.json").write_text(json.dumps({"study_question": "RQ1 fixture"}))
            (root / "configs/downstream_classifier_protocol.json").write_text(json.dumps({"research_questions": {"RQ2": "fixture", "RQ3": "fixture"}}))
            path = fe.generate_publication_report(root)
            text = path.read_text()
        self.assertIn("## 12. Conclusions", text)
        self.assertIn("| question | text |", text)
        self.assertIn("Not yet evaluated", text)
        self.assertNotIn("```json", text)

    def test_final_paths_have_no_unimplemented_error(self):
        paths = [ROOT / "notebooks/3_generator_benchmark", ROOT / "notebooks/4_downstream_classifiers", ROOT / "notebooks/utility"]
        matches = [path for directory in paths for path in directory.rglob("*")
                   if path.is_file() and "NotImplementedError" in path.read_text(errors="ignore")]
        self.assertEqual(matches, [])

    def test_source_archive_contract_files_exist(self):
        self.assertTrue((ROOT / ".gitignore").is_file())
        self.assertTrue((ROOT / "assets/logo_MammoDiffusion.png").is_file())


if __name__ == "__main__":
    unittest.main()
