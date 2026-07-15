from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import downstream_protocol as dp  # noqa: E402
from downstream_experiment import select_best_epoch, training_budget  # noqa: E402


class DownstreamProtocolTests(unittest.TestCase):
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

    def test_synthetic_condition_resolves_after_amended_selection(self):
        # The generator benchmark ran and the post-benchmark amendment (Option B) selected G02/G07;
        # configs/selected_generators.json now exists and drives the synthetic downstream conditions.
        selection = ROOT / "configs/selected_generators.json"
        self.assertTrue(selection.exists())
        payload = json.loads(selection.read_text())
        self.assertEqual(payload["finetuned"], "02_sd21_filtered_100steps")
        self.assertEqual(payload["from_scratch"], "07_ldm_sdvae_extra1361")
        resolved = dp.resolve_condition(ROOT, "real_plus_best_finetuned_positive")
        self.assertEqual(resolved["synthetic_generator_id"], "02_sd21_filtered_100steps")

    def test_pr_auc_controls_checkpoint_early_stopping_and_scheduler(self):
        protocol = dp.load_protocol(ROOT)
        for architecture, policy in protocol["architectures"].items():
            self.assertEqual(policy["scheduler_params"]["monitor"], "val_pr_auc")
            self.assertEqual(policy["early_stopping"]["monitor"], "val_pr_auc")
            self.assertTrue(policy["checkpoint_criterion"].startswith("val_pr_auc_max"))

    def test_pr_auc_beats_better_roc_auc(self):
        history = [{"epoch": 1, "val_pr_auc": .40, "val_auc": .95, "val_loss": .5},
                   {"epoch": 2, "val_pr_auc": .55, "val_auc": .80, "val_loss": .6}]
        self.assertEqual(select_best_epoch(history)["epoch"], 2)

    def test_pr_auc_tie_uses_loss_then_earliest_epoch(self):
        history = [{"epoch": 1, "val_pr_auc": .5, "val_auc": .8, "val_loss": .4},
                   {"epoch": 2, "val_pr_auc": .5, "val_auc": .9, "val_loss": .3},
                   {"epoch": 3, "val_pr_auc": .5, "val_auc": .95, "val_loss": .3}]
        self.assertEqual(select_best_epoch(history)["epoch"], 2)

    def test_four_conditions_share_budget_and_validation(self):
        protocol = dp.load_protocol(ROOT)
        for architecture in dp.ARCHITECTURES:
            configuration = {"policy": protocol["architectures"][architecture]}
            budget = training_budget(configuration)
            self.assertEqual(budget["max_optimizer_updates"], protocol["fairness"]["maximum_optimizer_updates"])
            self.assertEqual(budget["validation_interval"], protocol["fairness"]["validation_interval_optimizer_updates"])
            self.assertEqual(budget["checkpoint_metric"], protocol["architectures"][architecture]["checkpoint_criterion"])
            self.assertEqual(budget["validation_manifest"], protocol["fairness"]["validation_manifest"])
            self.assertEqual(budget["effective_batch_size"], 16)
            self.assertEqual(budget["lr_schedule"], "ReduceLROnPlateau")
            self.assertEqual(budget["early_stopping_policy"]["monitor"], "val_pr_auc")

    def test_duplicate_job_is_rejected(self):
        payload = json.loads((ROOT / "configs/downstream_classifier_jobs.json").read_text())
        payload["jobs"][-1] = dict(payload["jobs"][0])
        with self.assertRaises(ValueError): dp.validate_jobs(payload)


if __name__ == "__main__": unittest.main()
