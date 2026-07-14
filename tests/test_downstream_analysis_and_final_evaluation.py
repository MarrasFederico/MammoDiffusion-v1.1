from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
from downstream_analysis import aggregate_patient, align_seed_predictions, discover_experiments  # noqa: E402
from final_evaluation import require_final_evaluation_opt_in, save_protocol_snapshot  # noqa: E402


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

    def test_publication_discovery_ignores_legacy_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "results/legacy_classifier_matrix/maxvit512/real_only/seed_17"
            legacy.mkdir(parents=True); (legacy / "validation_metrics.json").write_text("{}")
            discovered = discover_experiments(root)
            self.assertEqual(len(discovered), 24)
            self.assertFalse(any(row["complete"] for row in discovered))
            self.assertTrue(all("publication_v2" in row["directory"] for row in discovered))

    def test_false_guard_prevents_final_evaluation(self):
        with self.assertRaisesRegex(PermissionError, "RUN_FINAL_EVALUATION"):
            require_final_evaluation_opt_in(False, {})

    def test_protocol_snapshot_is_plain_and_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = save_protocol_snapshot(root, selected_generators={"finetuned": "ft", "from_scratch": "fs"},
                seed_checkpoints={"job": "checkpoint"}, validation_thresholds={"job": .4},
                planned_comparisons=[{"id": "c"}], final_evaluation_dataset_identifier="external-v1")
            payload = json.loads(path.read_text())
            self.assertIn("ensemble_definitions", payload)
            self.assertIn("planned_statistical_comparisons", payload)
            self.assertNotIn("signature", payload)
            self.assertNotIn("lock", payload)


if __name__ == "__main__": unittest.main()
