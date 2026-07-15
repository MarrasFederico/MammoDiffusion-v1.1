"""Tests for the post-benchmark gate calibration audit utilities."""
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


class PopcountAndPairsTests(unittest.TestCase):
    def test_popcount_matches_python(self):
        values = np.array([0, 1, 3, 7, 0xFFFFFFFFFFFFFFFF, 0x0101010101010101], dtype=np.uint64)
        expected = [int(v).bit_count() for v in values]
        self.assertEqual(list(ga._popcount64(values)), expected)

    def test_neighbour_pairs_threshold(self):
        # A=0, B=3 (dist 2 from A), C=7 (dist 1 from B, dist 3 from A)
        pairs = ga.phash_neighbour_pairs([0, 3, 7], max_hamming_distance=2)
        self.assertEqual(set(pairs), {(0, 1), (1, 2)})  # A-B and B-C, not A-C


class PhashClusterOrderIndependenceTests(unittest.TestCase):
    def _chain(self):
        # A near B, B near C, A NOT near C (transitive single component).
        return [0, 3, 7]

    def test_chain_forms_single_component(self):
        diag = ga.perceptual_hash_cluster_diagnostics([f"img{i}" for i in range(3)],
                                                       hash_ints=self._chain())
        self.assertEqual(diag["n_phash_pairs"], 2)
        self.assertEqual(diag["n_phash_connected_components"], 1)
        self.assertEqual(diag["n_nontrivial_phash_components"], 1)
        self.assertEqual(diag["largest_phash_component_size"], 3)
        self.assertEqual(diag["n_images_with_any_phash_neighbour"], 3)
        self.assertAlmostEqual(diag["phash_any_neighbour_rate"], 1.0)
        self.assertAlmostEqual(diag["phash_component_excess_rate"], 2 / 3)

    def test_order_independent(self):
        hashes = [0, 3, 7, 1024, 2048]  # last two isolated
        base = ga.perceptual_hash_cluster_diagnostics([str(i) for i in range(5)], hash_ints=hashes)
        for permutation in ([2, 0, 4, 1, 3], [4, 3, 2, 1, 0]):
            shuffled = [hashes[i] for i in permutation]
            other = ga.perceptual_hash_cluster_diagnostics([str(i) for i in range(5)], hash_ints=shuffled)
            for field in ("phash_any_neighbour_rate", "phash_component_excess_rate",
                          "largest_phash_component_size", "n_phash_connected_components",
                          "n_nontrivial_phash_components", "n_phash_pairs"):
                self.assertEqual(base[field], other[field], f"{field} changed under reordering")

    def test_components_identity_under_reordering(self):
        hashes = [0, 3, 7, 1024, 2048]
        base = ga.connected_components(5, ga.phash_neighbour_pairs(hashes))
        base_sizes = sorted(len(c) for c in base)
        for permutation in ([2, 0, 4, 1, 3], [1, 4, 0, 3, 2]):
            shuffled = [hashes[i] for i in permutation]
            comp = ga.connected_components(5, ga.phash_neighbour_pairs(shuffled))
            self.assertEqual(sorted(len(c) for c in comp), base_sizes)

    def test_legacy_rate_is_order_dependent_but_available(self):
        # Deprecated legacy metric: exposed for comparison, not for decisions.
        self.assertAlmostEqual(ga.legacy_order_dependent_phash_rate([0, 3, 7]), 2 / 3)
        self.assertAlmostEqual(ga.legacy_order_dependent_phash_rate([7, 3, 0]), 2 / 3)


class ConfirmedDuplicateTests(unittest.TestCase):
    def _make(self, directory: Path):
        base = np.full((64, 64), 60, dtype=np.uint8)
        base[:20, :20] = 200
        Image.fromarray(base).save(directory / "a.png")
        Image.fromarray(base).save(directory / "b.png")  # byte-identical -> exact duplicate
        near = base.copy()
        near[63, 63] = 61  # near identical: high SSIM, tiny phash change
        Image.fromarray(near).save(directory / "c.png")
        distinct = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
        Image.fromarray(distinct).save(directory / "d.png")
        return [directory / name for name in ("a.png", "b.png", "c.png", "d.png")]

    def test_confirmed_and_exact_rates_are_distinct_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._make(Path(tmp))
            result = ga.confirmed_duplicate_analysis(paths)
        # a and b are exact duplicates -> both flagged.
        self.assertGreaterEqual(result["exact_duplicate_rate"], 0.5)
        self.assertGreaterEqual(result["confirmed_duplicate_rate"], result["exact_duplicate_rate"])
        # The audit must not collapse different metrics onto one "synthetic_duplicate_rate" name.
        self.assertIn("confirmed_duplicate_rate", result)
        self.assertNotIn("synthetic_duplicate_rate", result)

    def test_distinct_image_is_not_confirmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._make(Path(tmp))
            result = ga.confirmed_duplicate_analysis(paths)
        # The random image d must not appear in any confirmed pair.
        d_index = 3
        for row in result["pairs"]:
            if row["confirmed_duplicate"]:
                self.assertNotIn(d_index, (row["left_index"], row["right_index"]))


class PrdcBaselineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.reference = rng.normal(size=(73, 16))
        self.candidate = rng.normal(size=(340, 16))

    def test_repeated_baseline_is_deterministic(self):
        first = ga.repeated_prdc_baseline(self.reference, self.candidate, subset_reference=73,
                                          subset_candidate=73, repetitions=5, seed=99, nearest_k=5)
        second = ga.repeated_prdc_baseline(self.reference, self.candidate, subset_reference=73,
                                           subset_candidate=73, repetitions=5, seed=99, nearest_k=5)
        self.assertEqual([r["coverage"] for r in first], [r["coverage"] for r in second])
        self.assertEqual(len(first), 5)

    def test_split_half_is_deterministic_and_smaller(self):
        first = ga.split_half_prdc_baseline(self.reference, repetitions=4, seed=7, nearest_k=5)
        second = ga.split_half_prdc_baseline(self.reference, repetitions=4, seed=7, nearest_k=5)
        self.assertEqual([r["coverage"] for r in first], [r["coverage"] for r in second])
        self.assertEqual(first[0]["reference_subset"] + first[0]["candidate_subset"], 73)
        self.assertLess(first[0]["reference_subset"], 73)

    def test_summarize_metric_prefixes_fields(self):
        rows = ga.repeated_prdc_baseline(self.reference, self.candidate, subset_reference=73,
                                         subset_candidate=73, repetitions=10, seed=3)
        summary = ga.summarize_metric(rows, "coverage")
        self.assertIn("coverage_mean", summary)
        self.assertIn("coverage_percentile_2_5", summary)


class DescriptiveRankingTests(unittest.TestCase):
    def test_ranks_ignore_eligibility(self):
        rows = [
            {"generator_id": "G02", "family": "finetuned", "raddino_kid": 0.20, "raddino_coverage": 0.14},
            {"generator_id": "G03", "family": "finetuned", "raddino_kid": 0.28, "raddino_coverage": 0.08},
            {"generator_id": "G04", "family": "finetuned", "raddino_kid": 0.29, "raddino_coverage": 0.05},
        ]
        ranked = ga.descriptive_generator_ranking(rows, "finetuned")
        order = [r["generator_id"] for r in ranked]
        self.assertEqual(order, ["G02", "G03", "G04"])  # by ascending KID
        self.assertEqual([r["descriptive_family_rank"] for r in ranked], [1, 2, 3])

    def test_family_filter(self):
        rows = [
            {"generator_id": "G07", "family": "from_scratch", "raddino_kid": 0.087, "raddino_coverage": 0.42},
            {"generator_id": "G08", "family": "from_scratch", "raddino_kid": 0.138, "raddino_coverage": 0.41},
            {"generator_id": "G02", "family": "finetuned", "raddino_kid": 0.20, "raddino_coverage": 0.14},
        ]
        ranked = ga.descriptive_generator_ranking(rows, "from_scratch")
        self.assertEqual([r["generator_id"] for r in ranked], ["G07", "G08"])


class StrictEfficiencyTests(unittest.TestCase):
    def _entry(self, tmp: Path, payload: dict) -> tuple[Path, dict]:
        (tmp / "manifest.json").write_text(json.dumps(payload))
        checkpoint = tmp / "model.bin"
        checkpoint.write_bytes(b"x" * 100)
        return tmp, {"efficiency_manifest": "manifest.json", "checkpoint": "model.bin"}

    def test_elapsed_without_semantics_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root, entry = self._entry(tmp, {"elapsed_seconds": 0.0067, "n_per_class": 2722,
                                            "generated_classes": ["0", "1"]})
            result = ga.efficiency_from_manifest_strict(root, entry)
        self.assertIsNone(result["generation_seconds_per_image"])
        self.assertEqual(result["efficiency_status"], ga.INVALID_DURATION_STATUS)
        # checkpoint size is still a legitimate, verifiable quantity.
        self.assertEqual(result["checkpoint_size_bytes"], 100)

    def test_declared_wall_clock_is_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root, entry = self._entry(tmp, {"elapsed_seconds": 5444.0, "n_per_class": 2722,
                                            "generated_classes": ["0", "1"],
                                            "duration_semantics": "wall_clock_full_generation",
                                            "duration_unit": "seconds", "measurement_complete": True})
            result = ga.efficiency_from_manifest_strict(root, entry)
        self.assertAlmostEqual(result["generation_seconds_per_image"], 5444.0 / 5444)
        self.assertEqual(result["efficiency_status"], "available")

    def test_energy_and_vram_require_verified_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root, entry = self._entry(tmp, {"elapsed_seconds": 0.0067, "n_per_class": 10,
                                            "generated_classes": ["1"], "energy_kwh": 2.7e-5,
                                            "peak_vram_mb": 1498.7})
            result = ga.efficiency_from_manifest_strict(root, entry)
        self.assertIsNone(result["energy_kwh"])
        self.assertIsNone(result["peak_vram_mb"])


if __name__ == "__main__":
    unittest.main()
