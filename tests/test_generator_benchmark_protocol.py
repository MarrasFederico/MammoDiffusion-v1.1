from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import generator_benchmark as gb  # noqa: E402


class GeneratorBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.protocol = gb.load_protocol(ROOT)
        self.registry = gb.load_registry(ROOT)

    def test_1361_synthetic_and_73_real_is_valid_with_subset_73(self):
        self.assertEqual(gb.evaluation_subset_size(1361, 73, 1361), 73)

    def test_real_reference_may_be_smaller_than_synthetic_pool(self):
        self.assertEqual(gb.evaluation_subset_size(2000, 41, 1361), 41)

    def test_synthetic_pool_below_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "below target"):
            gb.evaluation_subset_size(1360, 73, 1361)

    def test_protocol_forbids_test_reference(self):
        broken = copy.deepcopy(self.protocol)
        broken["reference_sets"]["distribution_metrics"] = "data/test.csv"
        with self.assertRaises(ValueError): gb.validate_protocol(broken)

    def test_prdc_balancing_is_without_replacement_and_deterministic(self):
        first = gb.balanced_subsample_indices(73, 1361, 73, 4, 17, nearest_neighbour_k=5)
        second = gb.balanced_subsample_indices(73, 1361, 73, 4, 17, nearest_neighbour_k=5)
        self.assertEqual(first, second)
        for row in first:
            self.assertFalse(row["replace"])
            self.assertEqual(len(row["real_indices"]), len(row["synthetic_indices"]))
            self.assertEqual(len(set(row["synthetic_indices"])), 73)

    def test_prdc_k_must_be_smaller_than_subset(self):
        with self.assertRaisesRegex(ValueError, "exceed"):
            gb.balanced_subsample_indices(5, 10, 5, 1, 17, nearest_neighbour_k=5)

    def test_fid_repetitions_are_independent_and_small(self):
        cfg = self.protocol["resampling"]
        self.assertNotEqual(cfg["fid_repetitions"], cfg["kid_repetitions"])
        self.assertLessEqual(cfg["fid_repetitions"], 10)
        ranking = [row["metric"] for row in self.protocol["selection"]["ranking"]]
        self.assertEqual(ranking, ["raddino_kid", "raddino_coverage", "raddino_precision", "raddino_fid",
                                   "inception_kid", "raddino_kid_std", "generator_id"])
        fid_rows = [row for row in self.protocol["selection"]["ranking"] if "fid" in row["metric"]]
        self.assertTrue(fid_rows)
        self.assertTrue(all(row.get("role") == "descriptive_tiebreak" for row in fid_rows))

    def test_train_memorization_reference_can_never_return_to_positive_only(self):
        value = self.protocol["reference_sets"]["train_memorization"]
        self.assertEqual(value, "generator_specific_declared_complete_training_corpus")
        self.assertNotIn("positive_only", value)
        notebook = (ROOT / "notebooks/3_generator_benchmark/05_Unified_Generator_Benchmark.ipynb").read_text()
        self.assertIn("generator-specific complete declared training corpus", notebook)

    def test_repeated_metrics_use_shared_stability_plan_and_full_estimate(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["synthetic_pool_target"] = 20
        protocol["resampling"].update(stability_repetitions=3, nearest_neighbour_k=3)
        rng = np.random.default_rng(9)
        rows, summary = gb.repeated_distribution_metrics(rng.normal(size=(10, 4)), rng.normal(size=(20, 4)), protocol)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["metric_group"] == "kid_prdc_stability" for row in rows))
        self.assertEqual(summary["stability_subset_size"], 8)
        self.assertEqual(summary["full_pool_real_count"], 10)
        self.assertEqual(summary["full_pool_synthetic_count"], 20)
        self.assertEqual(summary["balanced_prdc_point_real_count"], summary["balanced_prdc_point_synthetic_count"])
        self.assertIn("kid_full_pool", summary["full_pool_distribution_estimates"])
        self.assertIn("precision_balanced_point", summary["balanced_prdc_point_estimates"])
        self.assertIn("kid", summary["stability_estimates"])
        self.assertEqual(summary["stability_interval_type"], "repeated-subsampling stability interval")

    def test_similarity_categories_are_separate(self):
        result = gb.similarity_summaries(
            [{"embedding_distance": .1, "memorization_flag": True, "exact_hash_match": False}],
            [{"embedding_distance": .01, "ssim": .999, "similarity_flag": True}],
            [{"embedding_distance": .2, "duplicate_flag": False, "perceptual_hash_duplicate": True}],
        )
        self.assertEqual(result["train_memorization_rate"], 1.0)
        self.assertEqual(result["validation_similarity_rate"], 1.0)
        self.assertEqual(result["synthetic_duplicate_rate"], 0.0)

    def test_flat_generator_summary_uses_every_preregistered_tiebreak_in_order(self):
        base = {"family": "finetuned", "eligible_for_selection": True, "valid_positive_images": 1361,
                "synthetic_exact_duplicate_rate": 0, "perceptual_hash_duplicate_rate": 0,
                "train_memorization_rate": 0, "filter_manifest_valid": True,
                "filter_provenance_complete": True, "n_corrupt": 0, "metrics_complete": True,
                "test_access": False, "lineage_complete": True, "provenance_manifest_valid": True,
                "training_corpus_manifest_valid": True}
        def row(generator_id, kid, coverage, precision, fid, inception, stability):
            return {**base, "generator_id": generator_id, "raddino_kid": kid,
                    "raddino_coverage": coverage, "raddino_precision": precision,
                    "raddino_fid": fid, "inception_kid": inception, "raddino_kid_std": stability}
        summary = [
            row("a_kid", .1, .5, .1, 9, .9, .9),
            row("b_coverage", .2, .9, .1, 9, .9, .9),
            row("c_precision", .2, .8, .9, 9, .9, .9),
            row("d_fid", .2, .8, .8, 1, .9, .9),
            row("e_inception", .2, .8, .8, 2, .1, .9),
            row("f_stability", .2, .8, .8, 2, .2, .01),
            row("g_generator_id", .2, .8, .8, 2, .2, .02),
            row("h_generator_id", .2, .8, .8, 2, .2, .02),
        ]
        ranked = gb.rank_generator_family(list(reversed(summary)), "finetuned", self.protocol["eligibility_gates"])
        self.assertEqual([row["generator_id"] for row in ranked], [row["generator_id"] for row in summary])

    def test_registry_roles_keep_ablation_and_descriptive_baseline_visible_but_ineligible(self):
        by_id = {entry["id"]: entry for entry in self.registry["generators"]}
        sd50 = by_id["01_sd21_baseline_50steps"]
        ldm = by_id["05_ldm_basic_fromscratch"]
        self.assertEqual((sd50["candidate_role"], sd50["sampling_steps"]), ("sampling_ablation", 50))
        self.assertFalse(sd50["eligible_for_downstream_selection"])
        self.assertEqual(ldm["candidate_role"], "descriptive_baseline")
        self.assertFalse(ldm["eligible_for_downstream_selection"])
        self.assertTrue(sd50["benchmark"]["enabled"] and ldm["benchmark"]["enabled"])

    def _result(self, generator_id, count=1361):
        return {"generator_id": generator_id, "metrics_complete": True, "valid_positive_images": count,
                "test_access": False, "technical_gates_passed": True}

    def test_manual_selection_accepts_simple_json_shape(self):
        selected = gb.validate_selected_generators("02_sd21_filtered_100steps", "06_ldm_extra1361_fromscratch",
            self.registry, [self._result("02_sd21_filtered_100steps"), self._result("06_ldm_extra1361_fromscratch")])
        self.assertEqual(selected, {"finetuned": "02_sd21_filtered_100steps", "from_scratch": "06_ldm_extra1361_fromscratch"})

    def test_wrong_family_and_ineligible_selection_are_rejected(self):
        rows = [self._result("02_sd21_filtered_100steps"), self._result("06_ldm_extra1361_fromscratch"),
                self._result("01_sd21_baseline_50steps")]
        with self.assertRaisesRegex(ValueError, "wrong family"):
            gb.validate_selected_generators("06_ldm_extra1361_fromscratch", "02_sd21_filtered_100steps", self.registry, rows)
        with self.assertRaisesRegex(ValueError, "not selection-eligible"):
            gb.validate_selected_generators("01_sd21_baseline_50steps", "06_ldm_extra1361_fromscratch", self.registry, rows)

    def test_embedding_cache_requires_and_round_trips_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.npy"
            metadata = {"schema_version": 2, "image_ids": ["a", "b"], "image_paths": ["a.png", "b.png"],
                        "image_fingerprints": [{"image_id": "a"}, {"image_id": "b"}], "extractor": "rad_dino",
                        "extractor_model_id": "microsoft/rad-dino", "extractor_weights_identifier": "w",
                        "extractor_identity": {"weight_sha256": "w"},
                        "extractor_identity_sha256": gb._identity_digest({"weight_sha256": "w"}),
                        "preprocessing_signature": "x", "feature_dimension": 3, "code_version": "abc",
                        "source_manifest_path": "m.json", "source_manifest_sha256": "123"}
            gb.write_embedding_cache(path, np.ones((2, 3)), metadata)
            features, restored = gb.load_embedding_cache(path)
            np.testing.assert_array_equal(features, np.ones((2, 3)))
            self.assertEqual(restored, metadata)


if __name__ == "__main__": unittest.main()
