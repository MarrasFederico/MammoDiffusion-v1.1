from __future__ import annotations
import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import classifier_metrics as cm  # noqa: E402


class MetricsCorrectnessTests(unittest.TestCase):
    def test_roc_auc_perfect_and_inverted_and_tied(self):
        labels = [0, 0, 0, 1, 1, 1]
        probs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        self.assertEqual(cm.roc_auc(labels, probs), 1.0)
        self.assertEqual(cm.roc_auc(labels, probs[::-1]), 0.0)
        self.assertEqual(cm.roc_auc([0, 1], [0.5, 0.5]), 0.5)

    def test_roc_auc_matches_brute_force_pairwise_definition(self):
        import random
        rng = random.Random(0)
        labels = [rng.randint(0, 1) for _ in range(150)]
        probs = [rng.random() for _ in range(150)]
        pos = [p for p, y in zip(probs, labels) if y == 1]
        neg = [p for p, y in zip(probs, labels) if y == 0]
        wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
        expected = wins / (len(pos) * len(neg))
        self.assertAlmostEqual(cm.roc_auc(labels, probs), expected, places=9)

    def test_roc_auc_requires_both_classes(self):
        with self.assertRaises(ValueError):
            cm.roc_auc([1, 1, 1], [0.1, 0.2, 0.3])

    def test_pr_auc_perfect_separation(self):
        labels = [0, 0, 1, 1]
        probs = [0.1, 0.2, 0.8, 0.9]
        self.assertEqual(cm.pr_auc(labels, probs), 1.0)

    def test_youden_threshold_on_separable_data(self):
        labels = [0, 0, 0, 1, 1, 1]
        probs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        result = cm.youden_threshold(labels, probs)
        self.assertEqual(result["youden_j"], 1.0)
        self.assertEqual(result["threshold"], 0.7)

    def test_metrics_at_threshold_confusion_derived_values(self):
        labels = [0, 0, 1, 1]
        probs = [0.1, 0.6, 0.4, 0.9]  # one FP (0.6) one FN (0.4) at threshold 0.5
        report = cm.metrics_at_threshold(labels, probs, 0.5)
        self.assertEqual((report["tp"], report["tn"], report["fp"], report["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(report["sensitivity_recall"], 0.5)
        self.assertAlmostEqual(report["specificity"], 0.5)
        self.assertAlmostEqual(report["accuracy"], 0.5)
        self.assertAlmostEqual(report["mcc"], 0.0)

    def test_brier_score_and_ece_perfect_predictions_are_zero(self):
        labels = [0, 1, 0, 1]
        probs = [0.0, 1.0, 0.0, 1.0]
        self.assertEqual(cm.brier_score(labels, probs), 0.0)
        self.assertEqual(cm.expected_calibration_error(labels, probs), 0.0)

    def test_full_report_contains_all_required_fields(self):
        labels = [0, 0, 1, 1, 0, 1]
        probs = [0.2, 0.4, 0.6, 0.9, 0.1, 0.7]
        report = cm.full_report(labels, probs)
        required = {"roc_auc", "pr_auc", "f1", "sensitivity_recall", "specificity", "precision_ppv",
                    "npv", "balanced_accuracy", "mcc", "accuracy", "brier_score", "ece", "threshold",
                    "tp", "tn", "fp", "fn"}
        self.assertTrue(required.issubset(report.keys()), required - report.keys())

    def test_full_report_uses_youden_threshold_when_not_given(self):
        labels = [0, 0, 0, 1, 1, 1]
        probs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        report = cm.full_report(labels, probs)
        self.assertEqual(report["threshold"], 0.7)

    def test_full_report_honors_explicit_locked_threshold(self):
        labels = [0, 0, 0, 1, 1, 1]
        probs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        report = cm.full_report(labels, probs, threshold=0.5)
        self.assertEqual(report["threshold"], 0.5)


class EnsembleTests(unittest.TestCase):
    def test_ensemble_probabilities_is_elementwise_mean_of_three_seeds(self):
        per_seed = [[0.1, 0.9], [0.3, 0.7], [0.2, 0.8]]
        result = cm.ensemble_probabilities(per_seed)
        self.assertAlmostEqual(result[0], 0.2, places=9)
        self.assertAlmostEqual(result[1], 0.8, places=9)

    def test_ensemble_probabilities_order_of_seeds_does_not_matter(self):
        a = cm.ensemble_probabilities([[0.1, 0.9], [0.3, 0.7], [0.2, 0.8]])
        b = cm.ensemble_probabilities([[0.2, 0.8], [0.1, 0.9], [0.3, 0.7]])
        self.assertEqual(a, b)

    def test_ensemble_probabilities_rejects_ragged_input(self):
        with self.assertRaises(ValueError):
            cm.ensemble_probabilities([[0.1, 0.9], [0.3]])

    def test_seed_stability_reports_mean_std_and_range(self):
        stability = cm.seed_stability([0.80, 0.82, 0.78])
        self.assertAlmostEqual(stability["mean"], 0.80, places=9)
        self.assertAlmostEqual(stability["range"], 0.04, places=9)
        self.assertGreaterEqual(stability["std"], 0.0)

    def test_seed_stability_zero_variance_for_identical_seeds(self):
        stability = cm.seed_stability([0.9, 0.9, 0.9])
        self.assertEqual(stability["std"], 0.0)
        self.assertEqual(stability["range"], 0.0)

    def test_ensemble_never_picks_a_single_best_seed(self):
        # Regression guard for spec 6: "Non scegliere il seed migliore sulla validation per il
        # test principale" — the ensemble must be the mean, never max/argmax across seeds.
        per_seed = [[0.10], [0.50], [0.99]]
        result = cm.ensemble_probabilities(per_seed)
        self.assertNotEqual(result[0], max(0.10, 0.50, 0.99))
        self.assertAlmostEqual(result[0], (0.10 + 0.50 + 0.99) / 3, places=9)


if __name__ == "__main__":
    unittest.main()
