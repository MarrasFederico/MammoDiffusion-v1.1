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
import classifier_architecture_adapters as adapters  # noqa: E402
import classifier_analysis as da  # noqa: E402
import classifier_experiment as de  # noqa: E402
import classifier_protocol as dp  # noqa: E402
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

    def test_benchmark_notebook_01_has_full_wiring_and_no_permanent_empty_placeholders(self):
        text = (ROOT / "notebooks/3_generator_benchmark/01_Unified_Generator_Benchmark.ipynb").read_text()
        self.assertIn("RUN_REAL_BENCHMARK = True", text)
        self.assertIn("REFRESH_CANDIDATE_AUDIT = True", text)
        self.assertIn("BUILD_CANONICAL_PROVENANCE = False", text)
        for forbidden in ("diversity_results = {}", "train_memorization_rows = []",
                          "validation_similarity_rows = []", "results_table = []"):
            self.assertNotIn(forbidden, text)
        for token in ("technical_validity_row", "get_or_extract_embeddings", "repeated_distribution_metrics",
                      "diversity_metrics", "build_train_memorization_rows", "build_validation_similarity_rows",
                      "build_synthetic_duplication_rows", "generator_summary.csv", "generator_ranking.csv",
                      "plot_generator_summary", "render_similarity_panel"):
            self.assertIn(token, text)

    def test_benchmark_notebook_01_false_mode_executes_without_mutation_or_real_references(self):
        output = ROOT / gb.BENCHMARK_ROOT
        before = {(path.relative_to(output).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
                  for path in output.rglob("*") if path.is_file()} if output.exists() else set()
        namespace = {}
        notebook = nbformat.read(ROOT / "notebooks/3_generator_benchmark/01_Unified_Generator_Benchmark.ipynb", 4)
        with mock.patch.dict(os.environ, {}, clear=False):
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    # Exercise review mode without editing or launching the final-run notebook on disk.
                    source = cell.source.replace("REFRESH_CANDIDATE_AUDIT = True", "REFRESH_CANDIDATE_AUDIT = False")
                    source = source.replace("RUN_REAL_BENCHMARK = True", "RUN_REAL_BENCHMARK = False")
                    exec(compile(source, f"benchmark_notebook01:{index}", "exec"), namespace)
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
                    "n_feature_extractable", "n_feature_nonextractable", "feature_extractable_rate",
                    "n_unique_feature_extractable_content", "n_exact_duplicates_among_feature_extractable",
                    "n_near_black", "n_invalid_range", "n_quality_valid", "n_quality_invalid",
                    "quality_validity_rate", "n_unique_quality_valid_content",
                    "n_exact_duplicates_among_quality_valid", "eligible_for_distribution_metrics",
                    "eligible_for_official_ranking", "quality_warning", "warning_reasons",
                    "fatal_failure_reasons"}
        self.assertTrue(expected.issubset(row))
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

    def test_selection_notebook_records_amended_selection(self):
        # After the benchmark and the human-approved post-benchmark amendment (Option B), notebook 02
        # reads the active amendment, shows the original zero-eligible outcome alongside the amended
        # safety-gate outcome, and records the explicit G02/G07 selection.
        text = (ROOT / "notebooks/3_generator_benchmark/02_Generator_Selection.ipynb").read_text()
        for token in ("SELECTED_FINETUNED_GENERATOR", "SELECTED_FROM_SCRATCH_GENERATOR",
                      "02_sd21_filtered_100steps", "07_ldm_sdvae_extra1361",
                      "load_active_amendment", "amended_family_ranking", "save_amended_selection",
                      "original_finetuned", "amended_finetuned", "SAVE_SELECTION"):
            self.assertIn(token, text)


class GPUResumeAndVisualizationTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    INVENTORY = [{"index": 0, "uuid": "GPU-0000", "name": "RTX 3060", "memory_total": "12288"},
                 {"index": 1, "uuid": "GPU-right", "name": "RTX 5060 Ti", "memory_total": "16384"}]

    def _inventory(self):
        return [dict(row) for row in self.INVENTORY]

    def _observe(self, uuid="GPU-right", name="RTX 5060 Ti", visible=1, local=0):
        return lambda: {"visible_count": visible, "local_index": local, "uuid": uuid,
                        "name": name, "memory_bytes": 16 * 1024 ** 3}

    def test_physical_index_is_resolved_to_uuid(self):
        result = de.configure_visible_gpu(1, inventory=self._inventory, observe=self._observe())
        self.assertEqual(result["resolved_uuid"], "GPU-right")
        self.assertEqual(result["resolved_physical_index"], 1)
        self.assertEqual(result["local_index"], 0)
        self.assertTrue(result["physical_identity_verified"])
        # CUDA_VISIBLE_DEVICES must be the UUID, never the numeric index.
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-right")

    def test_automatic_selection_chooses_largest_memory_gpu_portably(self):
        for inventory in (self._inventory(), list(reversed(self._inventory()))):
            with self.subTest(order=[row["index"] for row in inventory]):
                result = de.configure_visible_gpu(
                    "auto", inventory=lambda: inventory, observe=self._observe()
                )
                self.assertEqual(result["resolved_uuid"], "GPU-right")
                self.assertEqual(result["observed_name"], "RTX 5060 Ti")
                self.assertTrue(result["automatic_selection"])
                self.assertEqual(
                    result["selection_policy"],
                    "maximum_total_memory_then_lowest_physical_index",
                )
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-right")
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def test_default_experiment_configuration_uses_automatic_gpu_selection(self):
        configuration = de.experiment_configuration(
            ROOT, "maxvit512", "real_only", 17
        )
        self.assertEqual(configuration["gpu"], "auto")

    def test_direct_uuid_selector_passes(self):
        result = de.configure_visible_gpu("GPU-right", inventory=self._inventory, observe=self._observe())
        self.assertEqual(result["resolved_uuid"], "GPU-right")
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-right")
        self.assertTrue(result["physical_identity_verified"])

    def test_manual_configuration_preserves_gpu_selector_and_standard_path(self):
        for selector in (1, "GPU-right"):
            configuration = de.experiment_configuration(
                ROOT, "maxvit512", "real_only", 17, gpu=selector)
            self.assertEqual(configuration["gpu"], selector)
            self.assertEqual(
                Path(configuration["results_dir"]),
                ROOT / "results/3_classifiers/seed_runs/maxvit512/real_only/seed_17")

    def test_nonexistent_uuid_fails(self):
        with self.assertRaisesRegex(RuntimeError, "not present as exactly one device"):
            de.configure_visible_gpu("GPU-missing", inventory=self._inventory, observe=self._observe())

    def test_unresolvable_index_fails_without_cuda_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "not uniquely resolvable"):
            de.configure_visible_gpu(9, inventory=self._inventory, observe=self._observe())

    def test_empty_inventory_fails(self):
        with self.assertRaisesRegex(RuntimeError, "nvidia-smi inventory is unavailable"):
            de.configure_visible_gpu(0, inventory=lambda: [], observe=self._observe())

    def test_identity_mismatch_fails(self):
        with self.assertRaisesRegex(RuntimeError, "reports UUID GPU-wrong"):
            de.configure_visible_gpu(1, inventory=self._inventory, observe=self._observe(uuid="GPU-wrong"))

    def test_missing_observed_uuid_fails(self):
        with self.assertRaisesRegex(RuntimeError, "could not read the visible device UUID"):
            de.configure_visible_gpu(1, inventory=self._inventory, observe=self._observe(uuid=""))

    def test_initialized_framework_requests_kernel_restart(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-0000"
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "Restart the kernel"):
                de.configure_visible_gpu(1, inventory=self._inventory, observe=self._observe())

    def test_dataset_guard_rejects_historical_test_paths(self):
        for path in ("data/processed/train/../test/1/x.png",
                     "results/4_final_evaluation/x.png",
                     "data/historical_test/x.png"):
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                de.assert_no_forbidden_data_paths(ROOT, [{"path": path, "label": 1}])

    def test_resume_records_portable_gpu_change(self):
        provenance = adapters._gpu_resume_provenance(
            {"gpu_uuid": "GPU-old"}, "GPU-right"
        )
        self.assertEqual(provenance, {
            "checkpoint_gpu_uuid": "GPU-old",
            "runtime_gpu_uuid": "GPU-right",
            "gpu_changed": True,
        })

    def _checkpoint(self, root: Path):
        run = de.experiment_dir(root, "maxvit512", "real_only", 17)
        expected = {"architecture": "maxvit512", "experiment_id": "maxvit512__real_only__seed17",
                    "dataset_variant_id": "real_only", "training_policy": "maxvit512_fixed_protocol",
                    "config_signature": "informational", "dataset_signature": "informational", "seed": 17}
        checkpoint_io.save_resume_checkpoint(run, {**expected, "epoch": 3, "global_step": 400})

    def test_compatible_resume_checkpoint_loads_automatically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._checkpoint(root)
            expected = {"architecture": "maxvit512", "experiment_id": "maxvit512__real_only__seed17",
                        "dataset_variant_id": "real_only", "training_policy": "maxvit512_fixed_protocol",
                        "config_signature": "informational", "dataset_signature": "informational", "seed": 17}
            payload, source = checkpoint_io.load_resume_checkpoint(
                de.experiment_dir(root, "maxvit512", "real_only", 17), expected
            )
        self.assertEqual(source, "checkpoint_latest")
        self.assertEqual((payload["epoch"], payload["global_step"]), (3, 400))

    def test_corrupt_and_incompatible_resume_checkpoints_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = de.experiment_dir(root, "maxvit512", "real_only", 17)
            run.mkdir(parents=True)
            checkpoint_io.resume_checkpoint_path(run).write_bytes(b"not a pickle")
            payload, reason = checkpoint_io.load_resume_checkpoint(run, {"architecture": "maxvit512"})
            self.assertIsNone(payload)
            self.assertIn("UnpicklingError", reason)

            checkpoint_io.save_resume_checkpoint(run, {
                "architecture": "mammofm", "epoch": 1, "global_step": 1
            })
            payload, reason = checkpoint_io.load_resume_checkpoint(run, {"architecture": "maxvit512"})
            self.assertIsNone(payload)
            self.assertIn("incompatible", reason)

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


    def test_final_paths_have_no_unimplemented_error(self):
        paths = [ROOT / "notebooks/3_generator_benchmark", ROOT / "notebooks/04_classifiers", ROOT / "notebooks/utility"]
        matches = [path for directory in paths for path in directory.rglob("*")
                   if path.is_file()
                   and "diffusers_repo" not in path.parts
                   and "NotImplementedError" in path.read_text(errors="ignore")]
        self.assertEqual(matches, [])

    def test_source_archive_contract_files_exist(self):
        self.assertTrue((ROOT / ".gitignore").is_file())
        self.assertTrue((ROOT / "assets/logo_MammoDiffusion.png").is_file())


if __name__ == "__main__":
    unittest.main()
