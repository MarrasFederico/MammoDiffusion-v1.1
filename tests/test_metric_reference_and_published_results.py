"""Independent guards for the v1.1 statistics and for the published numbers.

Two gaps motivated this file.

1. ``test_final_matrix_statistics`` checks average precision only against
   hand-written expected values, so it validates the implementation with
   itself. The historical tied-score defect belongs to exactly that class, and
   ROC-AUC tie handling had no direct coverage at all. The reference tests here
   recompute both metrics with a deliberately different, brute-force algorithm
   (and with scikit-learn when it is installed) on tie-heavy random inputs.
2. Nothing bound the published numbers to the frozen prediction CSV files. A
   hand-edited ``ensemble_metrics.json``, ``results.json``, or README table
   would have left the suite green. The consistency tests recompute the release
   figures from the committed patient-level predictions.

Everything here reads committed artifacts only: no model, image, GPU, or
network access.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import classifier_metrics as cm  # noqa: E402
import classifier_statistics as cs  # noqa: E402

try:  # scikit-learn is in requirements.txt but not in the light requirements-dev.txt.
    from sklearn.metrics import average_precision_score, roc_auc_score

    SKLEARN = True
except ImportError:  # pragma: no cover - depends on the installed environment
    SKLEARN = False

ARCHITECTURES = ("maxvit512", "mammofm")
CONDITIONS = (
    "real_only",
    "real_augmented",
    "real_plus_best_finetuned_positive",
    "real_plus_best_fromscratch_positive",
)
TEST_ENSEMBLES = ROOT / "results/4_final_evaluation/test_ensembles"
VALIDATION_ENSEMBLES = ROOT / "results/3_classifiers/validation_ensembles"
FINAL_RESULTS = ROOT / "results/4_final_evaluation/results.json"


# --- deliberately naive references -------------------------------------------------------------

def brute_force_average_precision(labels, probabilities) -> float:
    """Average precision straight from its definition.

    Sweeps every distinct score as a ``>=`` cut point and accumulates
    ``precision * delta_recall``. No cumulative sums and no tie-group
    bookkeeping, so it shares no code path with ``classifier_metrics.pr_auc``.
    """
    y = [int(value) for value in labels]
    p = [float(value) for value in probabilities]
    positives = sum(y)
    if positives == 0:
        raise ValueError("no positive labels")
    total, previous_recall = 0.0, 0.0
    for cut in sorted(set(p), reverse=True):
        true_positive = sum(1 for label, score in zip(y, p) if score >= cut and label == 1)
        false_positive = sum(1 for label, score in zip(y, p) if score >= cut and label == 0)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / positives
        total += precision * (recall - previous_recall)
        previous_recall = recall
    return total


def brute_force_roc_auc(labels, probabilities) -> float:
    """Mann-Whitney statistic by explicit pair counting, ties worth one half."""
    y = [int(value) for value in labels]
    p = [float(value) for value in probabilities]
    positives = [score for label, score in zip(y, p) if label == 1]
    negatives = [score for label, score in zip(y, p) if label == 0]
    if not positives or not negatives:
        raise ValueError("need both classes")
    concordant = 0.0
    for high in positives:
        for low in negatives:
            concordant += 1.0 if high > low else 0.5 if high == low else 0.0
    return concordant / (len(positives) * len(negatives))


def reference_holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down with the standard monotonicity enforcement."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    size = len(ordered)
    adjusted, running_maximum = {}, 0.0
    for rank, (name, value) in enumerate(ordered, start=1):
        running_maximum = max(running_maximum, min(1.0, value * (size - rank + 1)))
        adjusted[name] = running_maximum
    return adjusted


def read_patient_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    labels = np.array([int(row["label"]) for row in rows])
    probabilities = np.array([float(row["probability"]) for row in rows])
    return [row["patient_id"] for row in rows], labels, probabilities


# --- reference-implementation differential tests -----------------------------------------------

class MetricReferenceTests(unittest.TestCase):
    def _random_cases(self, count: int = 300):
        generator = random.Random(20260809)
        produced = 0
        while produced < count:
            size = generator.randint(2, 40)
            mode = produced % 4
            if mode == 0:
                scores = [generator.random() for _ in range(size)]
            elif mode == 1:  # heavy ties
                scores = [round(generator.random(), 1) for _ in range(size)]
            elif mode == 2:  # every score identical
                scores = [0.5] * size
            else:  # only the two extremes
                scores = [generator.choice([0.0, 1.0]) for _ in range(size)]
            prevalence = generator.choice([0.05, 0.2, 0.5, 0.9])
            labels = [1 if generator.random() < prevalence else 0 for _ in range(size)]
            if 0 < sum(labels) < size:
                produced += 1
                yield labels, scores

    def test_average_precision_matches_a_brute_force_reference(self):
        for labels, scores in self._random_cases():
            with self.subTest(n=len(labels)):
                self.assertAlmostEqual(
                    cm.pr_auc(labels, scores),
                    brute_force_average_precision(labels, scores),
                    places=12,
                )

    def test_roc_auc_matches_a_pair_counting_reference(self):
        for labels, scores in self._random_cases():
            with self.subTest(n=len(labels)):
                self.assertAlmostEqual(
                    cm.roc_auc(labels, scores),
                    brute_force_roc_auc(labels, scores),
                    places=12,
                )

    @unittest.skipUnless(SKLEARN, "scikit-learn is not installed in this environment")
    def test_metrics_match_scikit_learn(self):
        for labels, scores in self._random_cases():
            with self.subTest(n=len(labels)):
                self.assertAlmostEqual(cm.pr_auc(labels, scores),
                                       float(average_precision_score(labels, scores)), places=12)
                self.assertAlmostEqual(cm.roc_auc(labels, scores),
                                       float(roc_auc_score(labels, scores)), places=12)

    def test_tied_scores_keep_both_metrics_row_order_invariant(self):
        generator = random.Random(7)
        for _ in range(100):
            size = generator.randint(4, 30)
            scores = [round(generator.random(), 1) for _ in range(size)]
            labels = [1 if generator.random() < 0.3 else 0 for _ in range(size)]
            if not 0 < sum(labels) < size:
                continue
            order = list(range(size))
            generator.shuffle(order)
            shuffled_labels = [labels[index] for index in order]
            shuffled_scores = [scores[index] for index in order]
            self.assertAlmostEqual(cm.pr_auc(labels, scores),
                                   cm.pr_auc(shuffled_labels, shuffled_scores), places=12)
            self.assertAlmostEqual(cm.roc_auc(labels, scores),
                                   cm.roc_auc(shuffled_labels, shuffled_scores), places=12)

    def test_roc_auc_rejects_one_class_input_instead_of_returning_a_number(self):
        for labels in ([1, 1, 1], [0, 0, 0]):
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    cm.roc_auc(labels, [0.1, 0.5, 0.9])

    def test_non_finite_values_fail_loudly_in_both_metrics(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    cm.pr_auc([1, 0], [0.5, bad])
                with self.assertRaises(ValueError):
                    cm.roc_auc([1, 0], [0.5, bad])


# --- bootstrap semantics ------------------------------------------------------------------------

class PairedBootstrapSemanticsTests(unittest.TestCase):
    """The declared method is a *paired* bootstrap; prove the pairing is real."""

    def _correlated_pair(self):
        generator = np.random.RandomState(3)
        labels = np.array([1] * 60 + [0] * 240)
        first = np.clip(generator.rand(300) * 0.5 + labels * 0.25, 0.0, 1.0)
        second = np.clip(first + generator.normal(0.0, 0.02, 300), 0.0, 1.0)
        return labels, first, second

    def test_identical_inputs_give_an_exactly_degenerate_difference_distribution(self):
        """The sharpest pairing invariant: A == B must leave no residual spread.

        A paired bootstrap scores both models on the *same* resampled patients,
        so every replicate difference is exactly zero. An implementation that
        resampled the two arms independently would produce a non-zero interval.
        """
        labels, first, _ = self._correlated_pair()
        result = cs.paired_stratified_bootstrap(labels, first, first, cm.pr_auc,
                                                n_bootstrap=200, seed=17)
        self.assertEqual(result["ci_95_low"], 0.0)
        self.assertEqual(result["ci_95_high"], 0.0)
        self.assertEqual(result["mean_diff_b_minus_a"], 0.0)

    def test_breaking_the_pairing_widens_the_interval(self):
        labels, first, second = self._correlated_pair()
        paired = cs.paired_stratified_bootstrap(labels, first, second, cm.pr_auc,
                                                n_bootstrap=300, seed=11)
        # Permute B within each class: same labels, same marginal score
        # distribution, but the patient-to-patient correspondence is destroyed.
        generator = np.random.RandomState(101)
        unpaired_second = second.copy()
        for value in (0, 1):
            index = np.where(labels == value)[0]
            unpaired_second[index] = generator.permutation(second[index])
        unpaired = cs.paired_stratified_bootstrap(labels, first, unpaired_second, cm.pr_auc,
                                                  n_bootstrap=300, seed=11)
        paired_width = paired["ci_95_high"] - paired["ci_95_low"]
        unpaired_width = unpaired["ci_95_high"] - unpaired["ci_95_low"]
        self.assertLess(paired_width, unpaired_width,
                        "an unpaired implementation would not narrow the interval for paired inputs")

    def test_every_resample_preserves_the_class_counts(self):
        labels = np.array([1] * 25 + [0] * 75)
        observed = []

        def recording_metric(sampled_labels, sampled_scores):
            observed.append((int(np.sum(sampled_labels == 1)), int(len(sampled_labels))))
            return cm.pr_auc(sampled_labels, sampled_scores)

        scores = np.linspace(0.0, 1.0, 100)
        cs.paired_stratified_bootstrap(labels, scores, scores, recording_metric,
                                       n_bootstrap=40, seed=5)
        self.assertEqual(set(observed), {(25, 100)})

    def test_the_seed_controls_the_resampling(self):
        labels, first, second = self._correlated_pair()
        kwargs = dict(metric_fn=cm.pr_auc, n_bootstrap=80)
        same = [cs.paired_stratified_bootstrap(labels, first, second, seed=9, **kwargs)
                for _ in range(2)]
        self.assertEqual(same[0], same[1])
        other = cs.paired_stratified_bootstrap(labels, first, second, seed=10, **kwargs)
        self.assertNotEqual(same[0]["mean_diff_b_minus_a"], other["mean_diff_b_minus_a"])

    def test_a_degenerate_single_class_input_is_refused(self):
        with self.assertRaises(ValueError):
            cs.paired_stratified_bootstrap([1, 1], [0.2, 0.4], [0.3, 0.5], cm.pr_auc, n_bootstrap=4)


class HolmReferenceTests(unittest.TestCase):
    CASES = (
        {"a": 0.011, "b": 0.089, "c": 0.178, "d": 0.382,
         "e": 0.687, "f": 0.708, "g": 0.755, "h": 1.0},
        {"a": 0.04, "b": 0.04, "c": 0.04},          # exact ties
        {"a": 0.0, "b": 0.5},                       # a zero tail area
        {"a": 1.0, "b": 1.0},
        {"only": 0.03},                             # single-member family
        {"a": 0.006, "b": 0.0079, "c": 0.008, "d": 0.9},
    )

    def test_adjusted_values_match_an_independent_step_down_reference(self):
        for index, case in enumerate(self.CASES):
            adjusted = cs.holm_correction(case)["adjusted_p_values"]
            expected = reference_holm(case)
            for name in case:
                with self.subTest(case=index, name=name):
                    self.assertAlmostEqual(adjusted[name], expected[name], places=12)

    def test_adjusted_values_are_monotone_and_never_below_the_raw_value(self):
        for index, case in enumerate(self.CASES):
            adjusted = cs.holm_correction(case)["adjusted_p_values"]
            ordered = [adjusted[name] for name, _ in sorted(case.items(), key=lambda kv: kv[1])]
            with self.subTest(case=index):
                self.assertEqual(ordered, sorted(ordered))
                for name, raw in case.items():
                    self.assertGreaterEqual(adjusted[name] + 1e-15, raw)


# --- published-number consistency ---------------------------------------------------------------

class PublishedResultConsistencyTests(unittest.TestCase):
    """Bind every released figure to the committed prediction CSV files."""

    def test_saved_ensemble_metrics_are_recomputable_from_their_predictions(self):
        for split, base in (("validation", VALIDATION_ENSEMBLES), ("test", TEST_ENSEMBLES)):
            for architecture in ARCHITECTURES:
                for condition in CONDITIONS:
                    directory = base / architecture / condition
                    _, labels, probabilities = read_patient_rows(
                        directory / "patient_level_predictions.csv"
                    )
                    saved = json.loads((directory / "ensemble_metrics.json").read_text())["metrics"]
                    with self.subTest(split=split, architecture=architecture, condition=condition):
                        self.assertAlmostEqual(cm.pr_auc(labels, probabilities),
                                               saved["pr_auc"], places=12)
                        self.assertAlmostEqual(cm.roc_auc(labels, probabilities),
                                               saved["roc_auc"], places=12)
                        self.assertAlmostEqual(cm.brier_score(labels, probabilities),
                                               saved["brier_score"], places=12)

    def test_patient_ensembles_are_the_mean_of_the_three_seed_predictions(self):
        for architecture in ARCHITECTURES:
            for condition in CONDITIONS:
                per_seed = {}
                for seed in (17, 42, 73):
                    path = (ROOT / "results/3_classifiers/seed_runs" / architecture / condition
                            / f"seed_{seed}/test_predictions.csv")
                    with path.open(newline="", encoding="utf-8") as stream:
                        per_seed[seed] = {
                            (row["patient_id"], row["image_id"]): float(row["probability"])
                            for row in csv.DictReader(stream)
                        }
                keys = sorted(per_seed[17])
                rebuilt = {}
                for key in keys:
                    mean = sum(per_seed[seed][key] for seed in (17, 42, 73)) / 3.0
                    rebuilt.setdefault(key[0], []).append(mean)
                identifiers, _, probabilities = read_patient_rows(
                    TEST_ENSEMBLES / architecture / condition / "patient_level_predictions.csv"
                )
                with self.subTest(architecture=architecture, condition=condition):
                    self.assertEqual(identifiers, sorted(rebuilt))
                    for identifier, saved in zip(identifiers, probabilities):
                        values = rebuilt[identifier]
                        self.assertAlmostEqual(sum(values) / len(values), saved, places=12)

    def test_every_reported_comparison_uses_the_same_patients_and_labels(self):
        reference = None
        for architecture in ARCHITECTURES:
            for condition in CONDITIONS:
                identifiers, labels, _ = read_patient_rows(
                    TEST_ENSEMBLES / architecture / condition / "patient_level_predictions.csv"
                )
                current = (tuple(identifiers), tuple(labels.tolist()))
                if reference is None:
                    reference = current
                self.assertEqual(current, reference,
                                 f"{architecture}/{condition} is not evaluated on the shared cohort")
        self.assertEqual(len(reference[0]), 438)

    def test_holm_block_is_recomputable_from_the_stored_tail_areas(self):
        payload = json.loads(FINAL_RESULTS.read_text())
        stored = {row["comparison_id"]: row["p_value_two_sided"] for row in payload["comparisons"]}
        self.assertEqual(len(stored), 8)
        expected = reference_holm(stored)
        adjusted = payload["holm_correction"]["adjusted_p_values"]
        self.assertEqual(payload["holm_correction"]["n_comparisons"], 8)
        for name, value in expected.items():
            self.assertAlmostEqual(adjusted[name], value, places=12)
        self.assertFalse(any(payload["holm_correction"]["reject_null"].values()),
                         "no comparison should be rejected after Holm adjustment")

    def test_headline_comparison_reproduces_the_stored_bootstrap(self):
        """Recompute the one comparison the README leads with, end to end."""
        payload = json.loads(FINAL_RESULTS.read_text())
        stored = next(row for row in payload["comparisons"]
                      if row["comparison_id"]
                      == "maxvit512:real_only_vs_real_plus_best_fromscratch_positive")
        _, labels, baseline = read_patient_rows(
            TEST_ENSEMBLES / "maxvit512/real_only/patient_level_predictions.csv")
        _, _, augmented = read_patient_rows(
            TEST_ENSEMBLES
            / "maxvit512/real_plus_best_fromscratch_positive/patient_level_predictions.csv")
        recomputed = cs.paired_stratified_bootstrap(
            labels, baseline, augmented, cm.pr_auc,
            n_bootstrap=stored["n_bootstrap"], seed=stored["seed"],
        )
        for field in ("mean_a", "mean_b", "mean_diff_b_minus_a",
                      "ci_95_low", "ci_95_high", "p_value_two_sided"):
            self.assertAlmostEqual(recomputed[field], stored[field], places=12, msg=field)

    def test_readme_result_table_matches_the_saved_predictions(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        row_pattern = re.compile(r"^\|\s*(MaxViT-512|Mammo-FM)\s*\|(.+)\|\s*$", re.M)
        rows = row_pattern.findall(readme)
        self.assertEqual(len(rows), 2, "the README results table must keep exactly two rows")
        architecture_by_label = {"MaxViT-512": "maxvit512", "Mammo-FM": "mammofm"}
        for label, cells in rows:
            values = [float(re.sub(r"[^0-9.]", "", cell)) for cell in cells.split("|")]
            self.assertEqual(len(values), 4, f"{label}: expected four condition columns")
            for condition, claimed in zip(CONDITIONS, values):
                _, labels_array, probabilities = read_patient_rows(
                    TEST_ENSEMBLES / architecture_by_label[label] / condition
                    / "patient_level_predictions.csv"
                )
                with self.subTest(architecture=label, condition=condition):
                    self.assertAlmostEqual(round(cm.pr_auc(labels_array, probabilities), 4),
                                           claimed, places=6)


if __name__ == "__main__":
    unittest.main()
