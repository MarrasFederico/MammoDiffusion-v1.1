from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import generator_comparison_analysis as gca  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


GEN_NO_METRICS = "01_sd21_baseline_50steps"
GEN_SCHEMA_A = "02_sd21_filtered_100steps"
GEN_POSITIVE_ONLY = "05_ldm_basic_fromscratch"
GEN_SCHEMA_B_SINGLE = "07_ldm_sdvae_extra1361"
GEN_SCHEMA_B_SPLIT = "08_ldm_v3_sdvae_fromscratch"


def build_fixture(root: Path) -> dict:
    registry = {"generators": [
        {"id": GEN_NO_METRICS, "family": "sd21", "role": "baseline", "classes": ["negative", "positive"]},
        {"id": GEN_SCHEMA_A, "family": "sd21", "role": "final_comparison", "classes": ["negative", "positive"],
         "metrics": f"results/diffusers/{GEN_SCHEMA_A}/metrics/final_test_metrics.json"},
        {"id": GEN_POSITIVE_ONLY, "family": "ldm", "role": "baseline", "classes": ["positive"],
         "metrics": f"results/diffusers/{GEN_POSITIVE_ONLY}/metrics/final_filtered_vs_test.json"},
        {"id": GEN_SCHEMA_B_SINGLE, "family": "ldm_sdvae", "role": "final_comparison", "classes": ["negative", "positive"],
         "metrics": f"results/diffusers/{GEN_SCHEMA_B_SINGLE}/metrics/final_filtered_vs_test.json"},
        {"id": GEN_SCHEMA_B_SPLIT, "family": "ldm_sdvae_v3", "role": "final_comparison", "classes": ["negative", "positive"],
         "metrics": f"results/diffusers/{GEN_SCHEMA_B_SPLIT}/metrics/final_filtered_vs_test.json",
         "sampling_ablations": [25, 50]},
    ]}
    write_json(root / "configs/final_generator_registry.json", registry)

    write_json(root / f"results/diffusers/{GEN_SCHEMA_A}/metrics/final_test_metrics.json", {
        "per_class": {
            "negative": {"FID": 100.0, "IS_mean": 2.0, "precision": 0.5, "recall": 0.3, "n_generated": 1361},
            "positive": {"FID": 110.0, "IS_mean": 2.2, "precision": 0.6, "recall": 0.4, "n_generated": 1361},
        }
    })
    write_json(root / f"results/diffusers/{GEN_POSITIVE_ONLY}/metrics/final_filtered_vs_test.json", {
        "metrics": {"FID": 150.0, "IS_mean": 1.8, "target_label": 1, "n_synthetic_filtered": 1361},
    })
    write_json(root / f"results/diffusers/{GEN_SCHEMA_B_SINGLE}/metrics/final_filtered_vs_test.json", {
        "metrics": {"FID": 120.0, "IS_mean": 2.1, "target_label": 1, "n_synthetic_filtered": 1361},
    })
    write_json(root / f"results/diffusers/{GEN_SCHEMA_B_SPLIT}/metrics/positive/final_filtered_vs_test.json", {
        "metrics": {"FID": 90.0, "IS_mean": 2.5, "target_label": 1, "n_synthetic_filtered": 1361},
    })
    write_json(root / f"results/diffusers/{GEN_SCHEMA_B_SPLIT}/metrics/negative/final_filtered_vs_test.json", {
        "metrics": {"FID": 95.0, "IS_mean": 2.4, "target_label": 0, "n_synthetic_filtered": 1361},
    })
    return registry


class MetricsSchemaNormalizationTests(unittest.TestCase):
    def test_generator_with_no_metrics_field_returns_none_not_zero(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); registry = build_fixture(root)
            entry = next(g for g in registry["generators"] if g["id"] == GEN_NO_METRICS)
            result = gca.generator_metrics_by_class(root, entry)
            self.assertIsNone(result["negative"])
            self.assertIsNone(result["positive"])

    def test_schema_a_per_class_dict_resolves_both_classes(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); registry = build_fixture(root)
            entry = next(g for g in registry["generators"] if g["id"] == GEN_SCHEMA_A)
            result = gca.generator_metrics_by_class(root, entry)
            self.assertEqual(result["negative"]["FID"], 100.0)
            self.assertEqual(result["positive"]["FID"], 110.0)

    def test_schema_b_flat_single_class_resolves_only_that_class(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); registry = build_fixture(root)
            entry = next(g for g in registry["generators"] if g["id"] == GEN_SCHEMA_B_SINGLE)
            result = gca.generator_metrics_by_class(root, entry)
            self.assertEqual(result["positive"]["FID"], 120.0)
            self.assertIsNone(result["negative"])  # honestly missing, never fabricated

    def test_schema_b_split_by_class_subdirectory_resolves_both(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); registry = build_fixture(root)
            entry = next(g for g in registry["generators"] if g["id"] == GEN_SCHEMA_B_SPLIT)
            result = gca.generator_metrics_by_class(root, entry)
            self.assertEqual(result["positive"]["FID"], 90.0)
            self.assertEqual(result["negative"]["FID"], 95.0)


class PositiveClassComparisonTests(unittest.TestCase):
    def test_includes_every_generator_with_a_positive_class(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture(root)
            rows = gca.positive_class_comparison(root)
            self.assertEqual(len(rows), 5)  # all 5 fixture generators have positive

    def test_positive_only_generator_is_included(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture(root)
            rows = gca.positive_class_comparison(root)
            ids = [r["generator_id"] for r in rows]
            self.assertIn(GEN_POSITIVE_ONLY, ids)


class TwoClassComparisonTests(unittest.TestCase):
    def test_positive_only_generator_excluded(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture(root)
            rows = gca.two_class_comparison(root)
            ids = [r["generator_id"] for r in rows]
            self.assertNotIn(GEN_POSITIVE_ONLY, ids)

    def test_never_averages_a_generator_missing_one_class(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture(root)
            rows = gca.two_class_comparison(root)
            g07_row = next(r for r in rows if r["generator_id"] == GEN_SCHEMA_B_SINGLE)
            self.assertEqual(g07_row["status"], "incomplete_metrics")
            self.assertNotIn("FID", g07_row)

    def test_complete_generator_gets_averaged_fid(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture(root)
            rows = gca.two_class_comparison(root)
            g02_row = next(r for r in rows if r["generator_id"] == GEN_SCHEMA_A)
            self.assertEqual(g02_row["status"], "ok")
            self.assertAlmostEqual(g02_row["FID"], 105.0)  # mean(100, 110)


class AblationComparisonTests(unittest.TestCase):
    def test_ablation_pair_present_but_empty_when_one_generator_lacks_metrics(self):
        # Both generator IDs exist in the registry (01 and 02), so the ablation slot appears;
        # 01 has no metrics file, so no delta_* fields get fabricated for it.
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture(root)
            rows = gca.ablation_comparison(root)
            row = next(r for r in rows if r["ablation"] == "sampling_steps_50_vs_100")
            self.assertFalse(any(k.startswith("positive_delta_") or k.startswith("negative_delta_") for k in row))

    def test_delta_direction_is_b_minus_a(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture(root)
            rows = gca.ablation_comparison(root)
            row = next(r for r in rows if r["ablation"] == "ldm_v2_vs_v3")
            # g07 FID=120 (positive) -> g08 FID=90 (positive): delta = 90-120 = -30
            self.assertAlmostEqual(row["positive_delta_FID_b_minus_a"], -30.0)


class SamplingAblationTests(unittest.TestCase):
    def test_missing_sweep_directory_reported_not_found_not_skipped_silently(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture(root)
            rows = gca.sampling_ablation(root, generator_id=GEN_SCHEMA_B_SPLIT)
            statuses = {r["sampling_steps"]: r["status"] for r in rows}
            self.assertEqual(statuses, {25: "not_found", 50: "not_found"})  # no sweep dirs created in this fixture

    def test_generator_without_sampling_ablations_field_returns_empty(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture(root)
            rows = gca.sampling_ablation(root, generator_id=GEN_SCHEMA_A)
            self.assertEqual(rows, [])


class FinalComparisonTableTests(unittest.TestCase):
    def test_only_final_comparison_role_generators_included(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture(root)
            rows = gca.final_comparison_table(root)
            ids = {r["generator_id"] for r in rows}
            self.assertNotIn(GEN_NO_METRICS, ids)  # role=baseline
            self.assertNotIn(GEN_POSITIVE_ONLY, ids)  # positive-only, excluded from two-class table entirely
            self.assertIn(GEN_SCHEMA_A, ids)
            self.assertIn(GEN_SCHEMA_B_SPLIT, ids)


class WriteOutputsTests(unittest.TestCase):
    def test_write_outputs_creates_all_canonical_table_files(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture(root)
            gca.write_outputs(root)
            tables_dir = root / "results/generator_comparison/tables"
            for name in ("positive_class_comparison", "two_class_comparison", "ablation_comparison",
                         "sampling_ablation", "final_comparison"):
                self.assertTrue((tables_dir / f"{name}.json").is_file(), name)

    def test_write_outputs_creates_summary_markdown(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture(root)
            gca.write_outputs(root)
            summary = root / "results/generator_comparison/generator_comparison_summary.md"
            self.assertTrue(summary.is_file())
            self.assertIn("Nessun vincitore downstream", summary.read_text())


if __name__ == "__main__":
    unittest.main()
