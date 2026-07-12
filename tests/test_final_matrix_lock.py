from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
sys.path.insert(0, str(ROOT / "scripts"))

import finalize_locked_test_stage as lock  # noqa: E402
import classifier_checkpoint_io as ckio  # noqa: E402


def build_lockable_project(root: Path) -> None:
    (root / "configs").mkdir(parents=True)
    for name, payload in (
        ("dataset_variant_registry.json", {"variants": []}),
        ("classifier_experiment_matrix.json", {"schema_version": 1, "jobs": []}),
        ("classifier_training_protocols.json", {"policies": {"maxvit512": {"framework": "pytorch_timm"}}}),
        ("final_generator_registry.json", {"generators": []}),
    ):
        (root / "configs" / name).write_text(json.dumps(payload))

    (root / "results/generator_comparison").mkdir(parents=True)
    (root / "results/generator_comparison/selected_generator_union.json").write_text(
        json.dumps({"selected_generator_union": ["gA"]}))

    (root / "results/final_evaluation_v2").mkdir(parents=True)
    (root / "results/final_evaluation_v2/primary_finalists_manifest.json").write_text(json.dumps({
        "primary_finalists": {"maxvit512": {"best_dataset_variant_id": "R"}},
        "secondary_locked_panel": ["maxvit512__R__seed17"],
    }))

    (root / "data/processed/metadata").mkdir(parents=True)
    (root / "data/processed/metadata/test.csv").write_text("patient_id,label\n1,0\n2,1\n")

    run = ckio.run_dir(root, "maxvit512", "R", "maxvit512_standard", 17)
    run.mkdir(parents=True)
    (run / "model.pt").write_bytes(b"weights")
    matrix_path = root / "configs/classifier_experiment_matrix.json"
    matrix = json.loads(matrix_path.read_text())
    matrix["jobs"].append({
        "experiment_id": "maxvit512__R__seed17", "stage": 1, "architecture": "maxvit512",
        "dataset_variant_id": "R", "training_policy": "maxvit512_standard", "seed": 17,
        "status": "VALIDATED", "manifest_path": str((run / "run_manifest.json").relative_to(root)),
    })
    matrix_path.write_text(json.dumps(matrix))


class PreconditionTests(unittest.TestCase):
    def test_refuses_when_no_selected_union(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_lockable_project(root)
            (root / "results/generator_comparison/selected_generator_union.json").unlink()
            problems = lock.preconditions(root)
            self.assertTrue(any("selected_generator_union" in p for p in problems))

    def test_refuses_when_union_empty(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_lockable_project(root)
            (root / "results/generator_comparison/selected_generator_union.json").write_text(json.dumps({"selected_generator_union": []}))
            problems = lock.preconditions(root)
            self.assertTrue(any("empty" in p for p in problems))

    def test_refuses_when_no_primary_finalists_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_lockable_project(root)
            (root / "results/final_evaluation_v2/primary_finalists_manifest.json").unlink()
            problems = lock.preconditions(root)
            self.assertTrue(any("primary_finalists_manifest" in p for p in problems))

    def test_ready_when_all_preconditions_met(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_lockable_project(root)
            self.assertEqual(lock.preconditions(root), [])


class NoWriteWithoutConfirmationTests(unittest.TestCase):
    def test_main_without_confirm_flag_writes_nothing(self):
        import subprocess, sys as _sys
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_lockable_project(root)
            result = subprocess.run([_sys.executable, str(ROOT / "scripts/finalize_locked_test_stage.py"),
                                      "--project-root", str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertFalse((root / "results/final_evaluation_v2/EXPERIMENT_MATRIX_LOCKED").exists())


class FinalizeAndVerifyTests(unittest.TestCase):
    def test_finalize_collects_nested_stage2_primary_categories(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_lockable_project(root)
            lock_dir = root / "results/final_evaluation_v2"
            payload = json.loads((lock_dir / "primary_finalists_manifest.json").read_text())
            payload["primary_finalists"] = {"maxvit512": {
                "R_baseline": {"experiment_id": "maxvit512__R__ensemble",
                               "seed_experiment_ids": ["maxvit512__R__seed17"]},
                "best_RS_CONTROLLED": {"experiment_id": "maxvit512__R__ensemble",
                                       "seed_experiment_ids": ["maxvit512__R__seed17"]},
                "best_RAS_FULL": {"status": "missing_preregistered_validation"},
            }}
            (lock_dir / "primary_finalists_manifest.json").write_text(json.dumps(payload))
            marker = lock.finalize(root)
            primary = json.loads((lock_dir / "primary_panel_manifest.json").read_text())
            self.assertEqual(primary["experiment_ids"], ["maxvit512__R__ensemble"])
            self.assertEqual(marker["n_primary_finalists"], 1)
    def test_finalize_writes_all_required_artifacts(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_lockable_project(root)
            lock.finalize(root)
            lock_dir = root / "results/final_evaluation_v2"
            for name in ("EXPERIMENT_MATRIX_LOCKED", "experiment_matrix_manifest.json",
                         "primary_finalists_manifest.json", "secondary_panel_manifest.json", "test_dataset_manifest.json"):
                self.assertTrue((lock_dir / name).is_file(), name)

    def test_verify_passes_immediately_after_finalize(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_lockable_project(root)
            lock.finalize(root)
            valid, problems = lock.verify_lock_still_valid(root)
            self.assertTrue(valid, problems)

    def test_verify_fails_before_any_lock_exists(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_lockable_project(root)
            valid, problems = lock.verify_lock_still_valid(root)
            self.assertFalse(valid)

    def test_verify_fails_if_checkpoint_changes_after_lock(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_lockable_project(root)
            lock.finalize(root)
            run = ckio.run_dir(root, "maxvit512", "R", "maxvit512_standard", 17)
            (run / "model.pt").write_bytes(b"DIFFERENT WEIGHTS NOW")
            valid, problems = lock.verify_lock_still_valid(root)
            self.assertFalse(valid)
            self.assertTrue(any("checkpoint" in p for p in problems))

    def test_verify_fails_if_test_csv_changes_after_lock(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_lockable_project(root)
            lock.finalize(root)
            (root / "data/processed/metadata/test.csv").write_text("patient_id,label\n1,0\n2,1\n3,0\n")
            valid, problems = lock.verify_lock_still_valid(root)
            self.assertFalse(valid)
            self.assertTrue(any("test" in p for p in problems))

    def test_verify_fails_if_dataset_registry_changes_after_lock(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_lockable_project(root)
            lock.finalize(root)
            (root / "configs/dataset_variant_registry.json").write_text(json.dumps({"variants": [{"dataset_variant_id": "NEW"}]}))
            valid, problems = lock.verify_lock_still_valid(root)
            self.assertFalse(valid)

    def test_lock_signature_is_deterministic_for_identical_state(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_lockable_project(root)
            marker1 = lock.finalize(root)
            marker2 = lock.finalize(root)
            self.assertEqual(marker1["lock_signature"], marker2["lock_signature"])


if __name__ == "__main__":
    unittest.main()
