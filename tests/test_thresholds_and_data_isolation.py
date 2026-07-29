from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import artifact_phase_planner as planner  # noqa: E402
import classifier_analysis as analysis  # noqa: E402
import classifier_dataset_builder as datasets  # noqa: E402
import classifier_experiment as experiment  # noqa: E402
import classifier_metrics as metrics  # noqa: E402
import final_evaluation  # noqa: E402
import generator_benchmark as benchmark  # noqa: E402
from processed_dataset_reuse import audit_patient_split_disjointness  # noqa: E402
from rebuild_generator_ranking import rebuild_ranking  # noqa: E402
from regenerate_classifier_metrics import regenerate as regenerate_classifier_metrics  # noqa: E402


class FrozenThresholdTests(unittest.TestCase):
    labels = [0, 0, 1, 1]
    probabilities = [.1, .6, .4, .9]

    def test_test_report_requires_both_validation_operating_points(self):
        with self.assertRaisesRegex(ValueError, "threshold frozen on validation"):
            metrics.full_report(self.labels, self.probabilities, split="test")
        with self.assertRaisesRegex(ValueError, "specificity threshold frozen on validation"):
            metrics.full_report(self.labels, self.probabilities, .5, split="test")

    def test_test_report_never_calls_an_optimizer(self):
        with mock.patch.object(metrics, "youden_threshold", side_effect=AssertionError), \
             mock.patch.object(metrics, "sensitivity_at_fixed_specificity", side_effect=AssertionError):
            report = metrics.full_report(
                self.labels, self.probabilities, .55, split="test", specificity_threshold=.7
            )
        self.assertEqual(report["threshold"], .55)
        self.assertEqual(report["sensitivity_at_specificity_0_90"]["threshold"], .7)

    def test_threshold_independent_metrics_do_not_depend_on_operating_point(self):
        low = metrics.full_report(
            self.labels, self.probabilities, .2, split="test", specificity_threshold=.3
        )
        high = metrics.full_report(
            self.labels, self.probabilities, .8, split="test", specificity_threshold=.9
        )
        for name in ("roc_auc", "pr_auc", "brier_score", "ece"):
            self.assertEqual(low[name], high[name])

    def test_patient_bootstrap_reuses_fixed_thresholds_every_time(self):
        rows = [
            {"patient_id": f"n{i}", "label": 0, "probability": value}
            for i, value in enumerate((.1, .2, .7))
        ] + [
            {"patient_id": f"p{i}", "label": 1, "probability": value}
            for i, value in enumerate((.3, .8, .9))
        ]
        calls = []
        original = metrics.full_report

        def recorded(labels, probabilities, threshold=None, **kwargs):
            calls.append((threshold, kwargs["split"], kwargs["specificity_threshold"]))
            return original(labels, probabilities, threshold, **kwargs)

        with mock.patch.object(analysis.metrics, "full_report", side_effect=recorded):
            intervals = analysis.patient_bootstrap_intervals(
                rows, iterations=12, seed=17, threshold=.61, split="test",
                specificity_threshold=.77,
            )
        self.assertEqual(calls, [(.61, "test", .77)] * 12)
        self.assertIn("sensitivity_at_specificity_0_90", intervals)

    def test_all_saved_seed_test_thresholds_equal_validation(self):
        checked = 0
        for validation_path in sorted((ROOT / "results/3_classifiers/seed_runs").glob("*/*/seed_*/validation_metrics.json")):
            test_path = validation_path.with_name("test_metrics.json")
            validation = json.loads(validation_path.read_text())
            test = json.loads(test_path.read_text())
            self.assertEqual(test["threshold"], validation["threshold"])
            self.assertEqual(
                test["sensitivity_at_specificity_0_90"]["threshold"],
                validation["sensitivity_at_specificity_0_90"]["threshold"],
            )
            checked += 1
        self.assertEqual(checked, 24)

    def test_all_saved_ensemble_test_thresholds_equal_validation(self):
        checked = 0
        for validation_path in sorted((ROOT / "results/3_classifiers/validation_ensembles").glob("*/*/ensemble_metrics.json")):
            relative = validation_path.relative_to(ROOT / "results/3_classifiers/validation_ensembles")
            test_path = ROOT / "results/4_final_evaluation/test_ensembles" / relative
            validation = json.loads(validation_path.read_text())["metrics"]
            test = json.loads(test_path.read_text())["metrics"]
            self.assertEqual(test["threshold"], validation["threshold"])
            self.assertEqual(
                test["sensitivity_at_specificity_0_90"]["threshold"],
                validation["sensitivity_at_specificity_0_90"]["threshold"],
            )
            checked += 1
        self.assertEqual(checked, 8)

    def test_csv_only_metric_regeneration_is_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_summary = regenerate_classifier_metrics(ROOT, Path(first), bootstrap_iterations=10)
            second_summary = regenerate_classifier_metrics(ROOT, Path(second), bootstrap_iterations=10)
            self.assertEqual(first_summary, second_summary)
            self.assertEqual((first_summary["seed_reports_written"], first_summary["ensemble_reports_written"]),
                             (24, 8))
            first_root, second_root = Path(first), Path(second)
            first_files = sorted(path.relative_to(first_root) for path in first_root.rglob("*") if path.is_file())
            second_files = sorted(path.relative_to(second_root) for path in second_root.rglob("*") if path.is_file())
            self.assertEqual(first_files, second_files)
            self.assertTrue(first_files)
            for relative in first_files:
                self.assertEqual((first_root / relative).read_bytes(), (second_root / relative).read_bytes(), relative)


class GeneratorIsolationTests(unittest.TestCase):
    def test_generator_notebook_code_has_no_test_data_path(self):
        forbidden = ("test.csv", "processed/test", "final_test_metrics", "final_filtered_vs_test", "TEST_METADATA_PATH")
        for path in sorted((ROOT / "notebooks/2_diffusers").glob("*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
            for token in forbidden:
                self.assertNotIn(token.lower(), code.lower(), path.name)

    def test_simple_selection_resolves_g02_and_g07(self):
        payload = json.loads((ROOT / "configs/selected_generators.json").read_text())
        self.assertEqual(payload["schema_version"], 3)
        self.assertEqual((payload["finetuned"], payload["from_scratch"]),
                         ("02_sd21_filtered_100steps", "07_ldm_sdvae_extra1361"))
        self.assertEqual({row["expected_count"] for row in payload["generators"]}, {1361})

    def test_pool_target_and_unique_selected_pool_records(self):
        self.assertEqual(benchmark.evaluation_subset_size(1361, 73, 1361, .8), 58)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool = root / "data/synthetic/g/positive"; pool.mkdir(parents=True)
            (pool / "a.png").write_bytes(b"a"); (pool / "b.png").write_bytes(b"b")
            configs = root / "configs"; configs.mkdir()
            (configs / "generator_registry.json").write_text(json.dumps({"generators": [{
                "id": "g", "samples": {"filtered_positive": "data/synthetic/g/positive"}
            }]}))
            records = datasets.load_selected_pool_records(root, "g", "finetuned", expected_count=2)
            self.assertEqual(len(records), 2)
            self.assertEqual(len({row["path"] for row in records}), 2)

    def test_training_metadata_rejects_non_train_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "data/processed/metadata"; metadata.mkdir(parents=True)
            (metadata / "val.csv").write_text("patient_id\np-val\n")
            (metadata / "train.csv").write_text(
                "patient_id,image_id,label,split,processed_path\n"
                "p,x,1,validation,data/processed/val/1/x.png\n"
            )
            with self.assertRaises((ValueError, PermissionError)):
                benchmark.training_corpus_from_metadata(root, verify_files=False)

    def test_training_metadata_rejects_validation_patient(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "data/processed/metadata"; metadata.mkdir(parents=True)
            (metadata / "val.csv").write_text("patient_id\np-held\n")
            (metadata / "train.csv").write_text(
                "patient_id,image_id,label,split,processed_path\n"
                "p-held,x,1,train,data/processed/train/1/x.png\n"
            )
            image = root / "data/processed/train/1/x.png"; image.parent.mkdir(parents=True); image.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "validation"):
                benchmark.training_corpus_from_metadata(root)

    def test_training_metadata_does_not_open_test_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "data/processed/metadata"; metadata.mkdir(parents=True)
            (metadata / "val.csv").write_text("patient_id\np-val\n")
            (metadata / "test.csv").write_text("patient_id\np-train\n")
            (metadata / "train.csv").write_text(
                "patient_id,image_id,label,split,processed_path\n"
                "p-train,x,1,train,data/processed/train/1/x.png\n"
            )
            image = root / "data/processed/train/1/x.png"; image.parent.mkdir(parents=True); image.write_bytes(b"x")
            paths, ids, labels, sources = benchmark.training_corpus_from_metadata(root)
            self.assertEqual((len(paths), ids, labels[ids[0]], sources[ids[0]]),
                             (1, ["real::p-train::x"], 1, "real_train"))

    def test_real_only_file_list_never_loads_synthetic_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); metadata = root / "data/processed/metadata"; metadata.mkdir(parents=True)
            (metadata / "train.csv").write_text("patient_id,image_id,label,processed_path\np,i,0,data/processed/train/0/i.png\n")
            files = datasets.build_file_list(root, {
                "dataset_variant_id": "real_only", "real_source": True,
                "augmentation_source": False, "synthetic_count_by_class": {},
            })
            self.assertTrue(files["negative"])
            self.assertTrue(all(row["source"] == "real" for rows in files.values() for row in rows))

    def test_ranking_rejects_missing_metric_instead_of_hidden_penalty(self):
        row = {"generator_id": "g", "family": "finetuned", "eligible_for_selection": True,
               "valid_positive_images": 1361, "synthetic_exact_duplicate_rate": 0,
               "train_memorization_rate": 0, "n_corrupt": 0, "metrics_complete": True,
               "test_access": False, "raddino_kid": .1, "raddino_coverage": .8,
               "raddino_precision": .8, "inception_kid": .2, "raddino_kid_std": .01}
        gates = benchmark.load_protocol(ROOT)["eligibility_gates"]
        with self.assertRaisesRegex(ValueError, "every eligible candidate"):
            benchmark.rank_generator_family([row], "finetuned", gates)

    def test_generator_ranking_rebuild_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            first = rebuild_ranking(ROOT, out)
            before = first.read_bytes()
            second = rebuild_ranking(ROOT, out)
            self.assertEqual(first, second)
            self.assertEqual(before, second.read_bytes())


class ProtocolAndRepositoryTests(unittest.TestCase):
    def test_patient_overlap_audit(self):
        overlaps = audit_patient_split_disjointness([
            {"patient_id": "p", "split": "train"},
            {"patient_id": "p", "split": "validation"},
            {"patient_id": "q", "split": "test"},
        ])
        self.assertEqual(overlaps["train_val"], ["p"])
        self.assertEqual(overlaps["train_test"], [])

    def test_runtime_manifest_requires_only_declared_phase_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "x.bin").write_bytes(b"x")
            (root / "runtime_manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "phases": {"training": {"files": [{"path": "x.bin", "size_bytes": 1}]}}
            }))
            self.assertTrue(planner.load_runtime_manifest(root)["valid"])

    def test_final_evaluation_default_is_disabled_and_overwrite_is_separate(self):
        notebook = nbformat.read(ROOT / "notebooks/04_classifiers/04_Final_Evaluation_and_Report.ipynb", 4)
        code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        self.assertIn("RUN_FINAL_EVALUATION = False", code)
        self.assertIn("OVERWRITE_TEST_PREDICTIONS = False", code)
        with self.assertRaises(FileExistsError):
            final_evaluation.require_final_evaluation_opt_in(ROOT, run_final_evaluation=True)
        payload = final_evaluation.require_final_evaluation_opt_in(
            ROOT, run_final_evaluation=True, overwrite_test_predictions=True
        )
        self.assertEqual(payload["expected_patient_count"], 438)

    def test_run_test_refuses_existing_prediction_before_adapter_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = experiment.experiment_dir(root, "maxvit512", "real_only", 17)
            output.mkdir(parents=True); (output / "test_predictions.csv").write_text("existing")
            configuration = {"architecture": "maxvit512", "condition": "real_only", "seed": 17, "policy": {}}
            with self.assertRaises(FileExistsError):
                experiment.run_test(root, configuration, "checkpoint.pt", [])

    def test_eligibility_config_matches_implementation(self):
        protocol = benchmark.load_protocol(ROOT)
        gates = protocol["eligibility_gates"]
        self.assertNotIn("maximum_perceptual_duplicate_rate", gates)
        self.assertNotIn("minimum_rad_dino_coverage", gates)
        descriptive = protocol["descriptive_reference_values"]
        self.assertIn("maximum_perceptual_duplicate_rate", descriptive)
        self.assertIn("minimum_rad_dino_coverage", descriptive)

    def test_pytest_collection_excludes_vendor_and_artifact_trees(self):
        expected = """[pytest]
testpaths = tests
norecursedirs =
    notebooks/utility/diffusers_repo
    .git
    .pytest_cache
    __pycache__
    data
    experiments
    results
"""
        self.assertEqual((ROOT / "pytest.ini").read_text(), expected)

if __name__ == "__main__":
    unittest.main()
