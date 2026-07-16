"""Smoke-mode machinery for the classifier notebooks (no GPU or real model is loaded)."""
from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
UTILITY = ROOT / "notebooks/utility"
sys.path.insert(0, str(UTILITY))
import downstream_experiment as de  # noqa: E402
import classifier_checkpoint_io as ckio  # noqa: E402

OLD_NOTEBOOK_DIR = "4_downstream" + "_classifiers"  # split so this file does not itself match the search


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
        self.assertFalse((ROOT / f"notebooks/{OLD_NOTEBOOK_DIR}").exists())

    def test_no_source_file_references_old_dir(self):
        # Pure-Python scan of source files (no git); excludes data/experiments/results and this test.
        needle = f"notebooks/{OLD_NOTEBOOK_DIR}"
        skip_dirs = {".git", "data", "experiments", "results", "__pycache__", ".ipynb_checkpoints"}
        offenders = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path == Path(__file__):
                continue
            if skip_dirs & set(path.relative_to(ROOT).parts):
                continue
            if path.suffix not in {".py", ".md", ".ipynb", ".json", ".txt", ".cfg", ".toml"}:
                continue
            try:
                if needle in path.read_text(encoding="utf-8", errors="ignore"):
                    offenders.append(path.relative_to(ROOT).as_posix())
            except OSError:
                pass
        self.assertEqual(offenders, [])

    def test_standard_notebooks_ignore_invalid_smoke_update_environment(self):
        for name in ("07_MaxViT512_Downstream.ipynb", "08_MammoFM_Downstream.ipynb"):
            source = json.loads((ROOT / "notebooks/04_classifiers" / name).read_text())["cells"][2]["source"]
            smoke_line = next(line for line in source.splitlines() if line.startswith("SMOKE_UPDATES ="))
            with mock.patch.dict(os.environ, {"MAMMODIFFUSION_SMOKE_UPDATES": "invalid"}):
                namespace = {"os": os, "RUN_MODE": "standard"}
                exec(smoke_line, namespace)
            self.assertEqual(namespace["SMOKE_UPDATES"], 2)


class RunModeTests(unittest.TestCase):
    def test_standard_and_smoke_paths(self):
        std = de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu=1)
        self.assertIn("results/publication_v2/downstream/maxvit512/real_only/seed_17", std["results_dir"])
        smoke = de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu=1, run_mode="smoke")
        self.assertIn("results/smoke/maxvit512/real_only/seed_17", smoke["results_dir"])
        self.assertNotIn("publication_v2", smoke["results_dir"])

    def test_unknown_run_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, run_mode="production")

    def test_gpu_selector_index_or_uuid_is_preserved(self):
        self.assertEqual(de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu=1)["gpu"], 1)
        self.assertEqual(de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu="GPU-abc")["gpu"], "GPU-abc")

    def test_smoke_updates_one_or_two_only(self):
        self.assertEqual(de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu=1,
                         run_mode="smoke", smoke_updates=1)["policy"]["max_optimizer_updates"], 1)
        self.assertEqual(de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu=1,
                         run_mode="smoke", smoke_updates=2)["policy"]["max_optimizer_updates"], 2)
        with self.assertRaises(ValueError):
            de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, run_mode="smoke", smoke_updates=3)

    def test_standard_ignores_smoke_updates_and_is_unchanged(self):
        std = de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu=1, smoke_updates=1)
        self.assertGreater(int(std["policy"]["max_optimizer_updates"]), 2)
        self.assertNotIn("smoke_updates", std)


class BudgetTests(unittest.TestCase):
    def test_smoke_budget_guard_rejects_over_two(self):
        cfg = de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu=1)
        cfg["run_mode"] = "smoke"
        with self.assertRaisesRegex(RuntimeError, "two optimizer updates"):
            de.training_budget(cfg)


class SubsetTests(unittest.TestCase):
    def test_deterministic_and_both_classes(self):
        rows = _rows()
        a = de.deterministic_smoke_subset(rows)
        b = de.deterministic_smoke_subset(list(reversed(rows)))
        self.assertEqual([r["image_id"] for r in a], [r["image_id"] for r in b])
        self.assertEqual(sum(1 for r in a if r["label"] == 0), 8)
        self.assertEqual(sum(1 for r in a if r["label"] == 1), 8)

    def test_synthetic_included_when_requested(self):
        rows = _rows(n_syn=12)
        self.assertEqual(sum(1 for r in de.deterministic_smoke_subset(rows, include_synthetic=False)
                             if r["source"] == "synthetic"), 0)
        self.assertEqual(sum(1 for r in de.deterministic_smoke_subset(rows, include_synthetic=True)
                             if r["source"] == "synthetic"), 8)

    def test_only_exact_positive_synthetic_rows_are_accepted(self):
        rows = _rows(n_syn=8)
        rows += [
            {"patient_id": None, "image_id": "wrong-source", "label": 1, "source": "synthetic_augmented"},
            {"patient_id": None, "image_id": "wrong-label", "label": 0, "source": "synthetic"},
        ]
        picked = de.deterministic_smoke_subset(rows, include_synthetic=True)
        synthetic = [row for row in picked if row["source"] == "synthetic"]
        self.assertEqual(len(synthetic), 8)
        self.assertTrue(all(row["label"] == 1 for row in synthetic))

    def test_insufficient_class_is_rejected(self):
        with self.assertRaises(RuntimeError):
            de.deterministic_smoke_subset(_rows(n_neg=3))

    def test_zero_train_validation_patient_overlap(self):
        train = de.deterministic_smoke_subset(_rows())
        val = de.deterministic_smoke_subset([{**r, "patient_id": f"v{r['patient_id']}"} for r in _rows()])
        self.assertEqual(de.audit_dataset(train, val)["train_validation_patient_overlap"], [])


class ForbiddenPathTests(unittest.TestCase):
    def test_rejects_test_traversal_case_and_final_evaluation(self):
        for bad in ("data/processed/train/../test/1/x.png",       # '..' traversal into test
                    "data/processed/Test/1/x.png",                # capitalised Test component
                    "results/final_evaluation/x.png",
                    "data/locked_test/x.png", "data/historical_test/x.png"):
            with self.assertRaises(RuntimeError):
                de.assert_no_forbidden_data_paths(ROOT, [{"path": bad, "label": 1}])

    def test_symlink_from_train_into_test_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/processed/test/1").mkdir(parents=True)
            (root / "data/processed/train/1").mkdir(parents=True)
            target = root / "data/processed/test/1/leak.png"
            target.write_bytes(b"x")
            link = root / "data/processed/train/1/link.png"
            link.symlink_to(target)
            with self.assertRaises(RuntimeError):
                de.assert_no_forbidden_data_paths(root, [{"path": "data/processed/train/1/link.png", "label": 1}])

    def test_allows_train_and_val_paths(self):
        de.assert_no_forbidden_data_paths(ROOT, [
            {"path": "data/processed/train/1/x.png", "label": 1},
            {"processed_path": "data/processed/val/0/y.png", "label": 0}])


class ResumeContinuityTests(unittest.TestCase):
    def _payload(self, step=1, uuid="GPU-abc"):
        return {"global_step": step, "model_state_dict": {"w": 1}, "optimizer_state_dict": {"m": 1},
                "scheduler_state_dict": {"s": 1}, "rng_states": {"python": 1}, "gpu_uuid": uuid}

    def test_accepts_both_conventions(self):
        de.verify_resume_continuity(self._payload(1), current_gpu_uuid="GPU-abc")
        de.verify_resume_continuity({"global_step": 1, "model_state": [1], "optimizer_state": [1],
                                     "gpu_uuid": "GPU-abc"}, current_gpu_uuid="GPU-abc")

    def test_zero_step_missing_state_or_gpu_mismatch_reject(self):
        with self.assertRaises(RuntimeError):
            de.verify_resume_continuity(self._payload(0), current_gpu_uuid="GPU-abc")
        with self.assertRaises(RuntimeError):
            de.verify_resume_continuity({"global_step": 1, "gpu_uuid": "GPU-abc"}, current_gpu_uuid="GPU-abc")
        with self.assertRaises(RuntimeError):
            de.verify_resume_continuity(self._payload(1, "GPU-abc"), current_gpu_uuid="GPU-def")


_WORKER = textwrap.dedent("""
    import sys
    sys.path.insert(0, {utility!r})
    from pathlib import Path
    from classifier_architecture_adapters import TinyAdapter
    run_dir = Path({run_dir!r})
    adapter = TinyAdapter("maxvit512", {{}}, ".")
    rows = [{{"label": 0}}, {{"label": 1}}]
    result = adapter.train(rows, rows, run_dir / "checkpoint_best.pt", seed=17, run_dir=run_dir,
        architecture="maxvit512", experiment_id="maxvit512__real_only__seed17",
        dataset_variant_id="real_only", training_policy="p", config_signature="c",
        dataset_signature="d", resume=True, gpu_uuid={gpu_uuid!r}, run_mode="smoke")
    print("RESUMED_FROM=" + str(result.get("resumed_from")))
""")


class TwoProcessResumeTests(unittest.TestCase):
    def test_real_checkpoint_resume_one_to_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()

            def run(gpu):
                script = _WORKER.format(utility=str(UTILITY), run_dir=str(run_dir), gpu_uuid=gpu)
                proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                return proc.stdout

            run("GPU-abc")  # process 1 -> global_step 1
            after_first = ckio.read_resume_checkpoint(ckio.resume_checkpoint_path(run_dir))
            self.assertEqual(after_first["global_step"], 1)

            stdout2 = run("GPU-abc")  # process 2 -> resume -> global_step 2
            after_second = ckio.read_resume_checkpoint(ckio.resume_checkpoint_path(run_dir))
            self.assertEqual(after_second["global_step"], 2)
            self.assertIsNotNone(after_second.get("model_state_dict"))
            self.assertIsNotNone(after_second.get("optimizer_state_dict"))
            self.assertEqual(after_second["gpu_uuid"], "GPU-abc")
            self.assertIn("RESUMED_FROM=checkpoint", stdout2)
            self.assertNotIn("RESUMED_FROM=None", stdout2)
            de.verify_resume_continuity(after_second, current_gpu_uuid="GPU-abc")

    def test_resume_with_different_gpu_uuid_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            script1 = _WORKER.format(utility=str(UTILITY), run_dir=str(run_dir), gpu_uuid="GPU-abc")
            subprocess.run([sys.executable, "-c", script1], capture_output=True, text=True, check=True)
            script2 = _WORKER.format(utility=str(UTILITY), run_dir=str(run_dir), gpu_uuid="GPU-other")
            proc = subprocess.run([sys.executable, "-c", script2], capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("does not match current", proc.stderr)


class SmokeOutputAndIgnoreTests(unittest.TestCase):
    def test_finalize_writes_only_completed_resumed_two_update_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            de.write_smoke_run_config(out, {"architecture": "maxvit512", "condition": "real_only", "seed": 17,
                                            "gpu": 1, "gpu_uuid": "GPU-abc", "gpu_name": "RTX", "smoke_updates": 2})
            import json
            run_config = json.loads((out / "run_config.json").read_text())
            self.assertEqual(run_config["gpu_uuid"], "GPU-abc")
            self.assertFalse(run_config["test_accessed"])
            de.finalize_smoke_run(out, optimizer_updates=2, resumed=True)
            smoke = json.loads((out / "smoke.json").read_text())
            self.assertEqual(smoke, {"mode": "smoke", "test_accessed": False, "completed": True,
                                     "optimizer_updates": 2, "resumed": True})

    def test_finalize_rejects_incomplete_conditions_without_smoke_json(self):
        for updates, resumed in ((1, True), (2, False), (1, False)):
            with self.subTest(optimizer_updates=updates, resumed=resumed), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                with self.assertRaisesRegex(RuntimeError, "Smoke finalization requires"):
                    de.finalize_smoke_run(out, optimizer_updates=updates, resumed=resumed)
                self.assertFalse((out / "smoke.json").exists())

    def test_smoke_pretraining_artifacts_survive_training_error(self):
        class FailingAdapter:
            def train(self, *args, **kwargs):
                raise RuntimeError("intentional training error")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configuration = de.experiment_configuration(ROOT, "maxvit512", "real_only", 17, gpu=1,
                                                         run_mode="smoke", smoke_updates=2)
            configuration.update({"gpu_uuid": "GPU-abc", "gpu_name": "RTX", "gpu_physical_index": 1})
            dataset = {"train_rows": _rows()[:16], "validation_rows": _rows()[:16],
                       "provenance": {"signature": "dataset"},
                       "audit": {"full": {"full": True}, "smoke": {"smoke": True}}}
            with mock.patch.object(de, "get_adapter", return_value=FailingAdapter()):
                with self.assertRaisesRegex(RuntimeError, "intentional training error"):
                    de.train(root, configuration, dataset, tiny=True)
            output = de.experiment_dir(root, "maxvit512", "real_only", 17, run_mode="smoke")
            self.assertTrue((output / "run_config.json").is_file())
            self.assertTrue((output / "dataset_audit.json").is_file())
            self.assertFalse((output / "train_log.csv").exists())
            self.assertFalse((output / "smoke.json").exists())

    def test_results_smoke_rule_in_gitignore_without_git(self):
        # Read .gitignore directly; must not depend on git or the .git directory.
        text = (ROOT / ".gitignore").read_text()
        self.assertTrue(any(line.strip() == "results/smoke/" for line in text.splitlines()))


class SyntheticMembershipTests(unittest.TestCase):
    def test_no_synthetic_rows_pass_trivially(self):
        self.assertTrue(de.verify_smoke_synthetic_membership(ROOT, _rows(), "real_only"))

    def test_real_g02_manifest_membership(self):
        if not (ROOT / "configs/selected_generators.json").is_file():
            self.skipTest("selection not present")
        try:
            import downstream_protocol as dp
            import classifier_dataset_builder as builder
            payload = dp.load_selected_generators(ROOT)
            records = builder.load_selected_filtered_records(ROOT, payload, "finetuned", verify_file_content=False)
        except Exception:
            self.skipTest("signed manifest not available")
        good = [{"source": "synthetic", "sha256": records[0]["sha256"], "label": 1}]
        bad = [{"source": "synthetic", "sha256": "0" * 64, "label": 1}]
        self.assertTrue(de.verify_smoke_synthetic_membership(ROOT, good, "real_plus_best_finetuned_positive"))
        self.assertFalse(de.verify_smoke_synthetic_membership(ROOT, bad, "real_plus_best_finetuned_positive"))


if __name__ == "__main__":
    unittest.main()
