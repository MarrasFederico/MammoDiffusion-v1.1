"""Smoke-mode machinery for the classifier notebooks (no GPU or model is loaded)."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import downstream_experiment as de  # noqa: E402


def _rows(n_neg=20, n_pos=20, n_syn=0):
    rows = []
    for i in range(n_neg):
        rows.append({"patient_id": f"n{i}", "image_id": f"ni{i}",
                     "path": f"data/processed/train/0/n{i}.png", "label": 0, "source": "real"})
    for i in range(n_pos):
        rows.append({"patient_id": f"p{i}", "image_id": f"pi{i}",
                     "path": f"data/processed/train/1/p{i}.png", "label": 1, "source": "real"})
    for i in range(n_syn):
        rows.append({"patient_id": None, "image_id": f"s{i}",
                     "relative_path": f"data/synthetic/g/positive/s{i}.png", "label": 1,
                     "source": "synthetic", "sha256": f"sha{i}"})
    return rows


class NotebookRenameTests(unittest.TestCase):
    def test_new_dir_exists_and_old_absent(self):
        self.assertTrue((ROOT / "notebooks/04_classifiers/07_MaxViT512_Downstream.ipynb").is_file())
        self.assertTrue((ROOT / "notebooks/04_classifiers/08_MammoFM_Downstream.ipynb").is_file())
        self.assertFalse((ROOT / "notebooks/4_downstream_classifiers").exists())

    def test_no_reference_to_old_notebook_dir(self):
        result = subprocess.run(["git", "grep", "-l", "notebooks/4_downstream_classifiers"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "")


class RunModeTests(unittest.TestCase):
    def test_standard_and_smoke_paths(self):
        std = de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu=1)
        self.assertIn("results/publication_v2/downstream/maxvit512/real_only/seed_17", std["results_dir"])
        self.assertEqual(std["run_mode"], "standard")
        smoke = de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu=1, run_mode="smoke")
        self.assertIn("results/smoke/maxvit512/real_only/seed_17", smoke["results_dir"])
        self.assertNotIn("publication_v2", smoke["results_dir"])

    def test_unknown_run_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, run_mode="production")
        with self.assertRaises(ValueError):
            de.experiment_dir(ROOT, "maxvit512", "real_only", 17, run_mode="bogus")

    def test_gpu_selector_index_or_uuid_is_preserved(self):
        self.assertEqual(de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu=1)["gpu"], 1)
        cfg = de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu="GPU-abc")
        self.assertEqual(cfg["gpu"], "GPU-abc")


class BudgetTests(unittest.TestCase):
    def test_smoke_budget_two_updates(self):
        cfg = de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu=1, run_mode="smoke")
        self.assertEqual(cfg["policy"]["max_optimizer_updates"], 2)
        self.assertEqual(de.training_budget(cfg)["max_optimizer_updates"], 2)

    def test_smoke_budget_guard_rejects_over_two(self):
        cfg = de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu=1)  # standard 6400 updates
        cfg["run_mode"] = "smoke"  # inconsistent: smoke mode but standard policy budget
        with self.assertRaisesRegex(RuntimeError, "two optimizer updates"):
            de.training_budget(cfg)

    def test_standard_policy_is_not_mutated(self):
        std = de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu=1)
        self.assertGreater(int(std["policy"]["max_optimizer_updates"]), 2)
        # A smoke config must not change the standard config built afterwards.
        de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu=1, run_mode="smoke")
        std2 = de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu=1)
        self.assertEqual(std["policy"]["max_optimizer_updates"], std2["policy"]["max_optimizer_updates"])


class SubsetTests(unittest.TestCase):
    def test_deterministic_and_both_classes(self):
        rows = _rows()
        a = de.deterministic_smoke_subset(rows)
        b = de.deterministic_smoke_subset(list(reversed(rows)))
        self.assertEqual([r["image_id"] for r in a], [r["image_id"] for r in b])
        self.assertEqual(sorted({r["label"] for r in a}), [0, 1])
        self.assertEqual(sum(1 for r in a if r["label"] == 0), 8)
        self.assertEqual(sum(1 for r in a if r["label"] == 1), 8)

    def test_synthetic_included_when_requested(self):
        rows = _rows(n_syn=12)
        without = de.deterministic_smoke_subset(rows, include_synthetic=False)
        with_syn = de.deterministic_smoke_subset(rows, include_synthetic=True)
        self.assertEqual(sum(1 for r in without if r["source"] == "synthetic"), 0)
        self.assertEqual(sum(1 for r in with_syn if r["source"] == "synthetic"), 8)

    def test_insufficient_class_is_rejected(self):
        with self.assertRaises(RuntimeError):
            de.deterministic_smoke_subset(_rows(n_neg=3))

    def test_zero_train_validation_patient_overlap(self):
        train = de.deterministic_smoke_subset(_rows())
        # Validation rows use distinct patient ids.
        val = de.deterministic_smoke_subset([{**r, "patient_id": f"v{r['patient_id']}"} for r in _rows()])
        audit = de.audit_dataset(train, val)
        self.assertEqual(audit["train_validation_patient_overlap"], [])


class ForbiddenPathTests(unittest.TestCase):
    def test_rejects_test_and_final_evaluation_paths(self):
        for bad in ("data/processed/test/1/x.png", "results/final_evaluation/x.png",
                    "data/locked_test/x.png", "data/historical_test/x.png"):
            with self.assertRaises(RuntimeError):
                de.assert_no_forbidden_data_paths([{"path": bad, "label": 1}])

    def test_allows_train_and_val_paths(self):
        de.assert_no_forbidden_data_paths([
            {"path": "data/processed/train/1/x.png", "label": 1},
            {"processed_path": "data/processed/val/0/y.png", "label": 0},
            {"relative_path": "data/synthetic/g/positive/s.png", "label": 1}])


class ResumeTests(unittest.TestCase):
    def _payload(self, step=1, uuid="GPU-abc"):
        return {"global_step": step, "model_state": {"w": 1}, "optimizer_state": {"m": 1}, "gpu_uuid": uuid}

    def test_resume_step_one_continues_to_two(self):
        result = de.verify_resume_continuity(self._payload(1), current_gpu_uuid="GPU-abc")
        self.assertEqual((result["resume_step"], result["next_step"], result["complete"]), (1, 2, False))
        done = de.verify_resume_continuity(self._payload(2), current_gpu_uuid="GPU-abc")
        self.assertTrue(done["complete"])

    def test_missing_state_or_zero_step_is_rejected(self):
        with self.assertRaises(RuntimeError):
            de.verify_resume_continuity({"global_step": 1, "gpu_uuid": "GPU-abc"}, current_gpu_uuid="GPU-abc")
        with self.assertRaises(RuntimeError):
            de.verify_resume_continuity(self._payload(0), current_gpu_uuid="GPU-abc")

    def test_gpu_identity_mismatch_is_rejected(self):
        with self.assertRaises(RuntimeError):
            de.verify_resume_continuity(self._payload(1, "GPU-abc"), current_gpu_uuid="GPU-def")


class SmokeSummaryAndIgnoreTests(unittest.TestCase):
    def test_smoke_summary_shape(self):
        self.assertEqual(de.smoke_summary(), {"mode": "smoke", "test_accessed": False, "completed": True})

    def test_results_smoke_is_gitignored(self):
        ignored = subprocess.run(["git", "check-ignore", "results/smoke/maxvit512/real_only/seed_17/smoke.json"],
                                 cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(ignored.returncode, 0)


class SyntheticMembershipTests(unittest.TestCase):
    def test_no_synthetic_rows_pass_trivially(self):
        self.assertTrue(de.verify_smoke_synthetic_membership(ROOT, _rows(), "real_only"))

    def test_real_g02_manifest_membership(self):
        # Uses the committed selection; reads the signed manifest (fast, no per-image re-hash).
        if not (ROOT / "configs/selected_generators.json").is_file():
            self.skipTest("selection not present")
        try:
            import downstream_protocol as dp
            import classifier_dataset_builder as builder
            payload = dp.load_selected_generators(ROOT)
            records = builder.load_selected_filtered_records(ROOT, payload, "finetuned", verify_file_content=False)
        except Exception:
            self.skipTest("signed manifest not available in this checkout")
        good = [{"source": "synthetic", "sha256": records[0]["sha256"], "label": 1}]
        bad = [{"source": "synthetic", "sha256": "0" * 64, "label": 1}]
        self.assertTrue(de.verify_smoke_synthetic_membership(ROOT, good, "real_plus_best_finetuned_positive"))
        self.assertFalse(de.verify_smoke_synthetic_membership(ROOT, bad, "real_plus_best_finetuned_positive"))


if __name__ == "__main__":
    unittest.main()
