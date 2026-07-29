from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
from classifier_analysis import aggregate_patient, align_seed_predictions  # noqa: E402
from classifier_interpretability import save_attribution_figure  # noqa: E402
from final_evaluation import frozen_validation_thresholds, require_final_evaluation_opt_in  # noqa: E402


class EnsembleAndFinalEvaluationTests(unittest.TestCase):
    def rows(self, seed):
        return [{"patient_id": "p0", "image_id": "i0", "label": 0, "probability": .2 + seed * 1e-6},
                {"patient_id": "p1", "image_id": "i1", "label": 1, "probability": .8 + seed * 1e-6}]

    def test_ensemble_requires_exact_three_seeds(self):
        with self.assertRaisesRegex(ValueError, "exactly seeds"):
            align_seed_predictions({17: self.rows(17)})

    def test_ensemble_checks_alignment_and_averages(self):
        per_seed = {seed: self.rows(seed) for seed in (17, 42, 73)}
        result = align_seed_predictions(per_seed)
        self.assertAlmostEqual(result[0]["probability"], sum(.2 + seed * 1e-6 for seed in (17, 42, 73)) / 3)
        per_seed[73][1]["image_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            align_seed_predictions(per_seed)

    def test_patient_aggregation_is_patient_level(self):
        result = aggregate_patient([{"patient_id": "p", "image_id": "a", "label": 1, "probability": .2},
                                    {"patient_id": "p", "image_id": "b", "label": 1, "probability": .8}])
        self.assertEqual(result, [{"patient_id": "p", "label": 1, "probability": .5, "n_images": 2}])

    def test_false_guard_prevents_final_evaluation(self):
        with self.assertRaisesRegex(PermissionError, "RUN_FINAL_EVALUATION"):
            require_final_evaluation_opt_in(ROOT, run_final_evaluation=False)

    def test_final_evaluation_loads_all_canonical_validation_thresholds(self):
        thresholds = frozen_validation_thresholds(ROOT)
        self.assertEqual(len(thresholds), 32)
        self.assertTrue(all(set(row) == {"decision_threshold", "specificity_0_90_threshold"}
                            for row in thresholds.values()))

    def test_attribution_figure_is_persisted_under_results_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = {}

            class Figure:
                def savefig(self, path, **kwargs):
                    calls.update(path=Path(path), kwargs=kwargs)
                    Path(path).write_bytes(b"png")

            root = Path(temporary) / "results/3_classifiers"
            path = save_attribution_figure(
                Figure(), root, architecture="maxvit512", condition="real_only",
                seed=17, method="gradcam",
            )
            self.assertEqual(
                path,
                root / "figures/interpretability/maxvit512/real_only/seed_17"
                / "gradcam_validation_cases.png",
            )
            self.assertTrue(path.is_file())
            self.assertEqual(calls["kwargs"]["dpi"], 180)


if __name__ == "__main__": unittest.main()
