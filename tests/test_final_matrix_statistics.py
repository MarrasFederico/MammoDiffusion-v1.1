from __future__ import annotations
import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import classifier_statistics as cs  # noqa: E402
import classifier_metrics as cm  # noqa: E402
import numpy as np  # noqa: E402


class DeLongTests(unittest.TestCase):
    def test_identical_models_have_zero_difference_and_p_one(self):
        rng = np.random.RandomState(0)
        labels = rng.randint(0, 2, 100).tolist()
        probs = rng.rand(100).tolist()
        result = cs.delong_test(labels, probs, probs)
        self.assertEqual(result["diff_b_minus_a"], 0.0)
        self.assertEqual(result["p_value"], 1.0)

    def test_auc_values_match_independent_roc_auc_implementation(self):
        rng = np.random.RandomState(2)
        labels = ([0] * 40 + [1] * 40)
        probs_a = rng.rand(80).tolist()
        probs_b = rng.rand(80).tolist()
        result = cs.delong_test(labels, probs_a, probs_b)
        self.assertAlmostEqual(result["auc_a"], cm.roc_auc(labels, probs_a), places=9)
        self.assertAlmostEqual(result["auc_b"], cm.roc_auc(labels, probs_b), places=9)

    def test_clearly_superior_model_yields_small_p_value(self):
        labels = [0] * 200 + [1] * 200
        probs_a = np.random.RandomState(1).rand(400).tolist()
        probs_b = [0.1] * 200 + [0.9] * 200
        result = cs.delong_test(labels, probs_a, probs_b)
        self.assertLess(result["p_value"], 0.001)
        self.assertGreater(result["auc_b"], result["auc_a"])

    def test_mismatched_lengths_raise_instead_of_silently_truncating(self):
        with self.assertRaises(ValueError):
            cs.delong_test([0, 1, 0], [0.1, 0.2], [0.1, 0.2, 0.3])

    def test_requires_both_classes_present(self):
        with self.assertRaises(ValueError):
            cs.delong_test([1, 1, 1], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6])


class McNemarTests(unittest.TestCase):
    def test_no_discordant_pairs_is_degenerate_p_one(self):
        result = cs.mcnemar_test([0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1])
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["n_discordant"], 0)

    def test_uses_exact_binomial_below_threshold(self):
        labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 2
        preds_a = [0] * 20
        preds_b = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 2
        result = cs.mcnemar_test(labels, preds_a, preds_b)
        self.assertEqual(result["method"], "exact_binomial")
        self.assertLess(result["p_value"], 0.01)

    def test_uses_chi_square_above_threshold(self):
        rng = np.random.RandomState(3)
        n = 200
        labels = rng.randint(0, 2, n)
        preds_a = labels.copy()
        preds_b = labels.copy()
        # force > exact_threshold discordant pairs in a fixed, reproducible pattern
        preds_a[:30] = 1 - preds_a[:30]
        result = cs.mcnemar_test(labels.tolist(), preds_a.tolist(), preds_b.tolist())
        self.assertEqual(result["method"], "chi_square_continuity_corrected")

    def test_symmetric_disagreement_is_not_significant(self):
        labels = [0, 1] * 20
        preds_a = [0, 1] * 10 + [1, 0] * 10  # 10 wrong one way
        preds_b = [1, 0] * 10 + [0, 1] * 10  # roughly symmetric discordance
        result = cs.mcnemar_test(labels, preds_a, preds_b)
        self.assertGreater(result["p_value"], 0.05)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            cs.mcnemar_test([0, 1], [0, 1, 1], [0, 1])


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
