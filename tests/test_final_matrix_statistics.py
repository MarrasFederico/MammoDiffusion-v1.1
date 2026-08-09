from __future__ import annotations
import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import classifier_statistics as cs  # noqa: E402
import classifier_metrics as cm  # noqa: E402
import numpy as np  # noqa: E402


class AveragePrecisionTests(unittest.TestCase):
    def test_tied_scores_are_one_threshold_and_order_invariant(self):
        labels_a = [1, 0, 1, 0]
        labels_b = [0, 1, 0, 1]
        probabilities = [0.8, 0.8, 0.2, 0.2]

        self.assertAlmostEqual(cm.pr_auc(labels_a, probabilities), 0.5)
        self.assertAlmostEqual(cm.pr_auc(labels_b, probabilities), 0.5)

    def test_metric_inputs_reject_malformed_arrays(self):
        invalid_cases = (
            ([0, 1], [0.2]),
            ([0, 2], [0.2, 0.8]),
            ([0, 1], [0.2, float("nan")]),
            ([0, 1], [0.2, 1.1]),
            ([], []),
        )
        for labels, probabilities in invalid_cases:
            with self.subTest(labels=labels, probabilities=probabilities):
                with self.assertRaises(ValueError):
                    cm.pr_auc(labels, probabilities)


class ActiveStatisticsApiTests(unittest.TestCase):
    def test_unused_legacy_tests_are_not_exposed_as_v11_protocol(self):
        self.assertFalse(hasattr(cs, "delong_test"))
        self.assertFalse(hasattr(cs, "mcnemar_test"))


class HolmCorrectionTests(unittest.TestCase):
    def test_smallest_p_value_adjusted_by_full_family_size(self):
        result = cs.holm_correction({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5})
        self.assertAlmostEqual(result["adjusted_p_values"]["a"], 0.004)  # 0.001 * 4

    def test_step_down_stops_rejecting_once_a_hypothesis_fails(self):
        result = cs.holm_correction({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5})
        self.assertTrue(result["reject_null"]["a"])
        self.assertFalse(result["reject_null"]["b"])
        self.assertFalse(result["reject_null"]["c"])
        self.assertFalse(result["reject_null"]["d"])

    def test_adjusted_p_values_are_monotonically_non_decreasing_by_rank(self):
        result = cs.holm_correction({"a": 0.01, "b": 0.011, "c": 0.5})
        ordered = sorted(result["adjusted_p_values"].items(), key=lambda kv: kv[1])
        values = [v for _, v in ordered]
        self.assertEqual(values, sorted(values))

    def test_all_significant_when_all_p_values_tiny(self):
        result = cs.holm_correction({"a": 0.0001, "b": 0.0002, "c": 0.0003})
        self.assertTrue(all(result["reject_null"].values()))

    def test_single_hypothesis_family_equals_unadjusted(self):
        result = cs.holm_correction({"only": 0.03})
        self.assertAlmostEqual(result["adjusted_p_values"]["only"], 0.03)


class BootstrapTests(unittest.TestCase):
    def test_identical_predictions_give_zero_mean_difference(self):
        rng = np.random.RandomState(5)
        labels = rng.randint(0, 2, 60).tolist()
        probs = rng.rand(60).tolist()
        result = cs.paired_stratified_bootstrap(labels, probs, probs, cm.roc_auc, n_bootstrap=200, seed=1)
        self.assertAlmostEqual(result["mean_diff_b_minus_a"], 0.0, places=6)

    def test_clearly_better_model_b_has_positive_ci_excluding_zero(self):
        labels = [0] * 100 + [1] * 100
        probs_a = np.random.RandomState(9).rand(200).tolist()
        probs_b = [0.05] * 100 + [0.95] * 100
        result = cs.paired_stratified_bootstrap(labels, probs_a, probs_b, cm.roc_auc, n_bootstrap=300, seed=2)
        self.assertGreater(result["ci_95_low"], 0.0)

    def test_bootstrap_is_reproducible_with_fixed_seed(self):
        labels = [0] * 30 + [1] * 30
        probs_a = np.random.RandomState(4).rand(60).tolist()
        probs_b = np.random.RandomState(5).rand(60).tolist()
        r1 = cs.paired_stratified_bootstrap(labels, probs_a, probs_b, cm.roc_auc, n_bootstrap=100, seed=42)
        r2 = cs.paired_stratified_bootstrap(labels, probs_a, probs_b, cm.roc_auc, n_bootstrap=100, seed=42)
        self.assertEqual(r1["mean_diff_b_minus_a"], r2["mean_diff_b_minus_a"])

    def test_bootstrap_requires_both_classes(self):
        with self.assertRaises(ValueError):
            cs.paired_stratified_bootstrap([1, 1, 1], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], cm.roc_auc)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            cs.paired_stratified_bootstrap([0, 1], [0.1, 0.2], [0.1], cm.roc_auc)


if __name__ == "__main__":
    unittest.main()
