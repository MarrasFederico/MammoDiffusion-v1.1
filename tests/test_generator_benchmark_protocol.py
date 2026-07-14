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

import generator_benchmark as gb  # noqa: E402


def metric_row(generator_id: str, family: str, kid: float = 0.1) -> dict:
    summary = lambda value: {"mean": value, "median": value, "standard_deviation": 0.01,
                             "percentile_2_5": value - 0.01, "percentile_97_5": value + 0.01}
    return {
        "generator_id": generator_id, "scientific_family": family,
        "exact_duplicate_rate": 0.0, "perceptual_hash_duplicate_rate": 0.0,
        "memorization_flag_rate": 0.0, "filter_acceptance_rate": 0.9,
        "corrupted_file_rate": 0.0, "valid_positive_images": 1361,
        "test_access": False, "lineage_verified": True, "provenance_verified": True,
        "metrics_complete": True, "bootstrap_stability": 0.99,
        "rad_dino": {"filtered": {"kid": summary(kid), "coverage": summary(0.8),
                                    "precision": summary(0.7), "fid": summary(4.0)}},
        "inception_v3": {"filtered": {"kid": summary(0.2)}},
    }


class GeneratorBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.protocol = gb.load_protocol(ROOT)

    def test_protocol_forbids_test_reference(self):
        broken = json.loads(json.dumps(self.protocol))
        broken["reference_sets"]["distribution_metrics"] = "data/test.csv"
        with self.assertRaises(ValueError):
            gb.validate_protocol(broken)

    def test_raw_and_filtered_are_separate_registered_representations(self):
        self.assertEqual(tuple(self.protocol["representations"]), ("raw", "filtered"))
        self.assertIn("never mix", self.protocol["comparison_rule"].lower())

    def test_deterministic_sampling_is_order_independent(self):
        paths = [f"x/{index}.png" for index in range(20)]
        self.assertEqual(gb.deterministic_sample(paths, 8, 17), gb.deterministic_sample(reversed(paths), 8, 17))
        self.assertEqual(len(gb.deterministic_sample(paths, 8, 17)), 8)

    def test_sampling_never_duplicates_to_fill_shortfall(self):
        with self.assertRaises(ValueError):
            gb.deterministic_sample(["a.png"], 2, 17)

    def test_duplicate_detection_exact_and_perceptual(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            array = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
            Image.fromarray(array).save(root / "a.png")
            Image.fromarray(array).save(root / "b.png")
            result = gb.duplicate_diagnostics([root / "a.png", root / "b.png"])
            self.assertEqual(result["exact_duplicate_items"], 1)
            self.assertGreater(result["perceptual_hash_duplicate_rate"], 0)

    def test_bootstrap_is_deterministic(self):
        rng = np.random.default_rng(3)
        real, synthetic = rng.normal(size=(20, 4)), rng.normal(size=(20, 4))
        first = gb.bootstrap_distribution_metrics(real, synthetic, iterations=4, sample_size=12, seed=9)
        second = gb.bootstrap_distribution_metrics(real, synthetic, iterations=4, sample_size=12, seed=9)
        self.assertEqual(first, second)

    def test_nearest_neighbour_rejects_test_pool(self):
        values = np.eye(2)
        with self.assertRaises(PermissionError):
            gb.nearest_neighbours(values, values, ["a", "b"], ["c", "d"], "real_test_positive")

    def test_family_specific_selection(self):
        rows = [metric_row("ft_b", "finetuned", 0.2), metric_row("ft_a", "finetuned", 0.1),
                metric_row("fs_a", "from_scratch", 0.3)]
        result = gb.select_generators(rows, self.protocol)
        self.assertEqual(result["selected"], {"finetuned": "ft_a", "from_scratch": "fs_a"})

    def test_eligibility_gate_excludes_missing_metrics(self):
        row = metric_row("ft", "finetuned"); row["metrics_complete"] = False
        self.assertIn("metrics_complete", gb.eligibility_failures(row, self.protocol["eligibility_gates"]))

    def test_tie_break_is_generator_id_deterministic(self):
        rows = [metric_row("z", "finetuned"), metric_row("a", "finetuned"), metric_row("fs", "from_scratch")]
        result = gb.select_generators(rows, self.protocol)
        self.assertEqual(result["selected"]["finetuned"], "a")


if __name__ == "__main__":
    unittest.main()
