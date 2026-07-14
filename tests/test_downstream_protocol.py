from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import downstream_protocol as dp  # noqa: E402


class DownstreamProtocolTests(unittest.TestCase):
    def test_exact_primary_matrix(self):
        jobs = dp.load_jobs(ROOT)["jobs"]
        self.assertEqual(len(jobs), 24)
        self.assertEqual({job["architecture"] for job in jobs}, set(dp.ARCHITECTURES))
        self.assertEqual({job["condition"] for job in jobs}, set(dp.CONDITIONS))
        self.assertEqual({job["seed"] for job in jobs}, set(dp.SEEDS))
        self.assertEqual(len({job["experiment_id"] for job in jobs}), 24)

    def test_only_two_downstream_architectures(self):
        protocol = dp.load_protocol(ROOT)
        self.assertEqual(set(protocol["architectures"]), {"maxvit512", "mammofm"})
        self.assertNotIn("resnet50", protocol["architectures"])
        self.assertNotIn("raddino", protocol["architectures"])

    def test_optional_ablations_disabled(self):
        payload = json.loads((ROOT / "configs/optional_downstream_ablations.json").read_text())
        self.assertFalse(payload["enabled"])

    def test_real_condition_resolves_without_approval(self):
        variant = dp.resolve_condition(ROOT, "real_only")
        self.assertTrue(variant["real_source"])
        self.assertFalse(variant["synthetic_count_by_class"])

    def test_synthetic_condition_requires_approval(self):
        self.assertFalse((ROOT / "configs/approved_generators.json").exists())
        with self.assertRaises(FileNotFoundError):
            dp.resolve_condition(ROOT, "real_plus_best_finetuned_positive")

    def test_fairness_is_constant_within_architecture(self):
        protocol = dp.load_protocol(ROOT)
        self.assertEqual(protocol["fairness"]["training_budget_policy"], "fixed_maximum_optimizer_updates")
        for architecture in dp.ARCHITECTURES:
            self.assertEqual(protocol["architectures"][architecture]["max_optimizer_updates"],
                             protocol["fairness"]["maximum_optimizer_updates"])

    def test_same_validation_and_checkpoint_policy(self):
        protocol = dp.load_protocol(ROOT)
        self.assertEqual(protocol["fairness"]["validation_manifest"], "data/processed/metadata/val.csv")
        for architecture in dp.ARCHITECTURES:
            self.assertTrue(protocol["architectures"][architecture]["checkpoint_criterion"])

    def test_experiment_id_round_trip(self):
        value = dp.experiment_id("maxvit512", "real_only", 17)
        self.assertEqual(dp.parse_experiment_id(value), {"architecture": "maxvit512", "condition": "real_only", "seed": 17})

    def test_duplicate_job_is_rejected(self):
        payload = json.loads((ROOT / "configs/downstream_classifier_jobs.json").read_text())
        payload["jobs"][-1] = dict(payload["jobs"][0])
        with self.assertRaises(ValueError):
            dp.validate_jobs(payload)

    def test_approval_signature_detects_tampering(self):
        payload = dp.signed_payload({"artifact_type": "x", "value": 1})
        payload["value"] = 2
        with self.assertRaises(ValueError):
            dp.verify_signed_payload(payload)


if __name__ == "__main__":
    unittest.main()
