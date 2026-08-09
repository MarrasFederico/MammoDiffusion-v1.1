from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import classifier_protocol as dp  # noqa: E402


class ClassifierProtocolTests(unittest.TestCase):
    def test_exact_primary_matrix(self):
        jobs = dp.load_jobs(ROOT)["jobs"]
        self.assertEqual(len(jobs), 24)
        self.assertEqual({job["architecture"] for job in jobs}, set(dp.ARCHITECTURES))
        self.assertEqual({job["condition"] for job in jobs}, set(dp.CONDITIONS))
        self.assertEqual({job["seed"] for job in jobs}, set(dp.SEEDS))
        self.assertFalse(any("suggested_command" in job for job in jobs))

    def test_only_maxvit_and_mammofm(self):
        self.assertEqual(set(dp.load_protocol(ROOT)["architectures"]), {"maxvit512", "mammofm"})

    def test_real_condition_does_not_require_selection(self):
        self.assertFalse(dp.resolve_condition(ROOT, "real_only")["synthetic_count_by_class"])

    def test_selection_file_records_g02_g07(self):
        # configs/selected_generators.json is the committed authoritative record of the selection.
        selection = ROOT / "configs/selected_generators.json"
        self.assertTrue(selection.exists())
        payload = json.loads(selection.read_text())
        self.assertEqual(payload["finetuned"], "02_sd21_filtered_100steps")
        self.assertEqual(payload["from_scratch"], "07_ldm_sdvae_extra1361")
        self.assertFalse(payload["test_access"])
        # The selection loader accepts it against the registry (family + eligibility, no SHA binding).
        loaded = dp.load_selected_generators(ROOT)
        self.assertEqual(loaded["finetuned"], "02_sd21_filtered_100steps")

    def test_synthetic_condition_resolves_to_selected_generator(self):
        resolved = dp.resolve_condition(ROOT, "real_plus_best_finetuned_positive")
        self.assertEqual(resolved["synthetic_generator_id"], "02_sd21_filtered_100steps")
        self.assertEqual(resolved["synthetic_count_by_class"], {"positive": 1361})

    def test_pr_auc_controls_checkpoint_early_stopping_and_scheduler(self):
        protocol = dp.load_protocol(ROOT)
        for policy in protocol["architectures"].values():
            self.assertEqual(policy["scheduler_params"]["monitor"], "val_pr_auc")
            self.assertEqual(policy["early_stopping"]["monitor"], "val_pr_auc")
            self.assertTrue(policy["checkpoint_criterion"].startswith("val_pr_auc_max"))

    def test_duplicate_job_is_rejected(self):
        payload = json.loads((ROOT / "configs/classifier_jobs.json").read_text())
        payload["jobs"][-1] = dict(payload["jobs"][0])
        with self.assertRaises(ValueError): dp.validate_jobs(payload)


if __name__ == "__main__": unittest.main()
