"""Tests for the size-matched perceptual-hash audit, the Option B amendment, and the selection."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import gate_audit as ga  # noqa: E402
import generator_benchmark as gb  # noqa: E402

AMENDMENT = json.loads((ROOT / "configs/generator_benchmark_protocol_amendment_v1.json").read_text())
SELECTION_PATH = ROOT / "configs/selected_generators.json"
AUDIT = ROOT / gb.BENCHMARK_ROOT / "gate_audit"


def _distinct_image(directory: Path, name: str, seed: int) -> Path:
    pixels = np.random.default_rng(seed).integers(0, 255, (48, 48), dtype=np.uint8)
    path = directory / name
    Image.fromarray(pixels).save(path)
    return path


class PerceptualHashCacheTests(unittest.TestCase):
    def test_cache_is_content_aware_and_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            image = _distinct_image(tmp, "a.png", 1)
            cache = ga.PerceptualHashCache(tmp / "cache")
            first = cache.phash_int(image)
            cache.save()
            # A fresh cache instance reuses the persisted, content-keyed entry.
            reopened = ga.PerceptualHashCache(tmp / "cache")
            self.assertEqual(reopened.phash_int(image), first)
            entry = next(iter(reopened.index.values()))
            for field in ("path", "file_size", "sha256", "phash_hex", "phash_version"):
                self.assertIn(field, entry)


class SizeMatchedPhashTests(unittest.TestCase):
    def _evidence(self, n: int) -> dict:
        # A deterministic synthetic evidence graph: a few near pairs, one confirmed, one exact.
        return {"n": n,
                "neighbour_pairs": {(0, 1), (1, 2), (5, 6)},
                "confirmed_pairs": {(5, 6)},
                "exact_pairs": {(5, 6)}}

    def test_deterministic_and_size_dependent(self):
        evidence = self._evidence(400)
        a = ga.repeated_size_matched_phash(evidence, subset_size=73, repetitions=50, seed=20260714)
        b = ga.repeated_size_matched_phash(evidence, subset_size=73, repetitions=50, seed=20260714)
        self.assertEqual([r["phash_any_neighbour_rate"] for r in a],
                         [r["phash_any_neighbour_rate"] for r in b])
        at340 = ga.repeated_size_matched_phash(evidence, subset_size=340, repetitions=50, seed=20260714)
        self.assertTrue(all(r["sample_size"] == 73 for r in a))
        self.assertTrue(all(r["sample_size"] == 340 for r in at340))

    def test_sampling_is_without_replacement(self):
        evidence = self._evidence(80)
        rows = ga.repeated_size_matched_phash(evidence, subset_size=80, repetitions=1, seed=1)
        # Full-pool subset must include every index exactly once: the two confirmed nodes appear.
        self.assertEqual(rows[0]["sample_size"], 80)
        self.assertGreaterEqual(rows[0]["confirmed_duplicate_rate"], 2 / 80)

    def test_subset_larger_than_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            ga.repeated_size_matched_phash(self._evidence(50), subset_size=73, repetitions=1, seed=1)

    def test_metrics_are_kept_separate(self):
        for metric in ("phash_any_neighbour_rate", "phash_component_excess_rate",
                       "confirmed_duplicate_rate", "exact_duplicate_rate", "largest_component_size"):
            self.assertIn(metric, ga.SIZE_MATCHED_METRICS)


class NoTestAccessTests(unittest.TestCase):
    def test_test_split_paths_are_rejected(self):
        for candidate in ("data/processed/test/1/x.png", "data/historical_internal_test/y.png"):
            with self.assertRaises(Exception):
                gb._reject_test_path(candidate)


class AmendedSafetyGateTests(unittest.TestCase):
    def _row(self, **overrides):
        row = {"generator_id": "02_sd21_filtered_100steps", "valid_positive_images": 1361,
               "synthetic_exact_duplicate_rate": 0.0, "train_memorization_rate": 0.0, "n_corrupt": 0,
               "filter_manifest_valid": True, "filter_provenance_complete": True, "metrics_complete": True,
               "test_access": False, "lineage_complete": True, "provenance_manifest_valid": True,
               "training_corpus_manifest_valid": True, "eligible_for_selection": True,
               # descriptive, must NOT gate:
               "raddino_coverage": 0.05, "perceptual_hash_duplicate_rate": 0.11}
        row.update(overrides)
        return row

    def test_safety_gates_pass_and_ignore_coverage_and_phash_only(self):
        failures = ga.amended_safety_gate_failures(self._row(), AMENDMENT["new_blocking_gates"],
                                                   confirmed_duplicate_rate=0.0)
        self.assertEqual(failures, [])

    def test_confirmed_duplicates_still_block(self):
        failures = ga.amended_safety_gate_failures(self._row(), AMENDMENT["new_blocking_gates"],
                                                   confirmed_duplicate_rate=0.5)
        self.assertIn("confirmed_duplicate_rate", failures)

    def test_missing_confirmed_rate_is_conservative(self):
        failures = ga.amended_safety_gate_failures(self._row(), AMENDMENT["new_blocking_gates"])
        self.assertIn("confirmed_duplicate_rate", failures)

    def test_amendment_removes_coverage_and_phash_only_gates(self):
        self.assertNotIn("minimum_rad_dino_coverage", AMENDMENT["new_blocking_gates"])
        self.assertNotIn("maximum_perceptual_duplicate_rate", AMENDMENT["new_blocking_gates"])
        for removed in ("minimum_rad_dino_coverage", "maximum_perceptual_duplicate_rate"):
            self.assertIn(removed, AMENDMENT["removed_blocking_gates"])


class AmendmentRecordTests(unittest.TestCase):
    def test_amendment_json_shape(self):
        self.assertEqual(AMENDMENT["status"], "approved_post_benchmark")
        self.assertEqual(AMENDMENT["approved_by"], "human")
        self.assertEqual(AMENDMENT["selected_policy"], "B")
        self.assertFalse(AMENDMENT["test_access"])
        self.assertEqual(AMENDMENT["original_outcome"]["eligible_under_original_gates"], 0)
        self.assertEqual(AMENDMENT["original_outcome"]["official_candidates_measured"], 5)
        self.assertEqual(AMENDMENT["selection_hierarchy"][0], "raddino_kid")

    def test_active_protocol_points_to_amendment_and_preserves_original_gates(self):
        protocol = gb.load_protocol(ROOT)
        self.assertEqual(protocol["active_amendment"], "configs/generator_benchmark_protocol_amendment_v1.json")
        self.assertIn("protocol_version", protocol)
        # The original preregistered gates remain in the historical record, unchanged.
        self.assertEqual(protocol["eligibility_gates"]["minimum_rad_dino_coverage"], 0.5)
        self.assertEqual(protocol["eligibility_gates"]["maximum_perceptual_duplicate_rate"], 0.02)

    def test_no_new_coverage_threshold_is_introduced(self):
        blob = json.dumps(AMENDMENT)
        self.assertNotIn("minimum_rad_dino_coverage", AMENDMENT["new_blocking_gates"])
        # No coverage gate should appear anywhere in the new blocking gates.
        self.assertFalse(any("coverage" in key for key in AMENDMENT["new_blocking_gates"]))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.registry = gb.load_registry(ROOT)
        summary = ROOT / gb.BENCHMARK_ROOT / "generator_summary.csv"
        if not summary.is_file():
            self.skipTest("benchmark generator_summary.csv not present")
        import csv
        with summary.open(newline="") as stream:
            self.rows = list(csv.DictReader(stream))
        self.confirmed = ga.confirmed_duplicate_rates(ROOT)

    def test_selected_generators_file_shape(self):
        payload = json.loads(SELECTION_PATH.read_text())
        self.assertEqual(payload["finetuned"], "02_sd21_filtered_100steps")
        self.assertEqual(payload["from_scratch"], "07_ldm_sdvae_extra1361")
        self.assertTrue(payload["post_benchmark_amendment"])
        self.assertFalse(payload["test_access"])
        for key in ("primary_metric", "benchmark_run_id", "original_protocol_result",
                    "active_amendment", "selection_notes"):
            self.assertIn(key, payload)

    def test_validate_accepts_g02_g07(self):
        selected = ga.validate_amended_selection(ROOT, "02_sd21_filtered_100steps", "07_ldm_sdvae_extra1361",
                                                 self.registry, self.rows, AMENDMENT, self.confirmed)
        self.assertEqual(selected["finetuned"], "02_sd21_filtered_100steps")

    def test_ineligible_generators_are_rejected(self):
        # G01 (ablation), G05 (descriptive baseline) and G06 (pool ablation) are not selection-eligible.
        for finetuned, from_scratch in (("01_sd21_baseline_50steps", "07_ldm_sdvae_extra1361"),
                                        ("02_sd21_filtered_100steps", "05_ldm_basic_fromscratch"),
                                        ("02_sd21_filtered_100steps", "06_ldm_extra1361_fromscratch")):
            with self.assertRaises(ValueError):
                ga.validate_amended_selection(ROOT, finetuned, from_scratch, self.registry, self.rows,
                                              AMENDMENT, self.confirmed)

    def test_wrong_family_is_rejected(self):
        with self.assertRaises(ValueError):
            ga.validate_amended_selection(ROOT, "07_ldm_sdvae_extra1361", "02_sd21_filtered_100steps",
                                          self.registry, self.rows, AMENDMENT, self.confirmed)


class CoverageReportingTests(unittest.TestCase):
    def test_coverage_point_and_stability_and_pass_fraction_are_distinct(self):
        path = AUDIT / "coverage_stability_summary.csv"
        if not path.is_file():
            self.skipTest("coverage_stability_summary.csv not present (run run_gate_amendment.py)")
        import csv
        with path.open(newline="") as stream:
            rows = {row["generator_id"]: row for row in csv.DictReader(stream)}
        for column in ("coverage_balanced_point", "coverage_stability_mean", "coverage_stability_median",
                       "coverage_stability_interval_low", "coverage_stability_interval_high",
                       "fraction_of_repetitions_above_0_5"):
            self.assertIn(column, rows["G07"])
        # Point, mean and pass-fraction are genuinely different quantities.
        self.assertNotEqual(rows["G07"]["coverage_balanced_point"], rows["G07"]["fraction_of_repetitions_above_0_5"])


class StrictEfficiencyDirectFieldTests(unittest.TestCase):
    def _entry(self, tmp: Path, payload: dict):
        (tmp / "m.json").write_text(json.dumps(payload))
        (tmp / "ckpt.bin").write_bytes(b"x" * 10)
        return tmp, {"efficiency_manifest": "m.json", "checkpoint": "ckpt.bin"}

    def test_direct_seconds_per_image_without_semantics_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root, entry = self._entry(tmp, {"seconds_per_image": 1.5})
            result = ga.efficiency_from_manifest_strict(root, entry)
        self.assertIsNone(result["generation_seconds_per_image"])
        self.assertEqual(result["generation_efficiency_status"], ga.INVALID_DURATION_STATUS)

    def test_direct_seconds_per_image_with_verified_semantics_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root, entry = self._entry(tmp, {"seconds_per_image": 1.5,
                                            "duration_semantics": "verified_seconds_per_image",
                                            "duration_unit": "seconds", "measurement_complete": True})
            result = ga.efficiency_from_manifest_strict(root, entry)
        self.assertEqual(result["generation_seconds_per_image"], 1.5)
        self.assertEqual(result["generation_efficiency_status"], "available")


if __name__ == "__main__":
    unittest.main()
