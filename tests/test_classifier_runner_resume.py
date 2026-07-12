from __future__ import annotations
import json, os, sys, tempfile, time, unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import classifier_checkpoint_io as ckio  # noqa: E402
import classifier_run_manifest as manifest  # noqa: E402
import classifier_experiment_runner as runner  # noqa: E402
import classifier_dataset_builder as cdb  # noqa: E402


def write_checkpoint(run: Path, content: bytes = b"weights") -> Path:
    run.mkdir(parents=True, exist_ok=True)
    ckpt = run / "model.pt"
    ckpt.write_bytes(content)
    ckio.write_checkpoint_metadata(run, architecture="maxvit512", dataset_variant_id="R", training_policy="p",
                                    seed=17, checkpoint=ckpt, dataset_manifest_sha256="abc", protocol_signature="def")
    return ckpt


class CheckpointIoTests(unittest.TestCase):
    def test_resume_checkpoint_rotates_and_falls_back_from_corrupt_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            expected = {"architecture": "raddino", "seed": 17, "config_signature": "c", "dataset_signature": "d"}
            ckio.save_resume_checkpoint(run, {**expected, "global_step": 10})
            ckio.save_resume_checkpoint(run, {**expected, "global_step": 20})
            ckio.resume_checkpoint_path(run).write_bytes(b"corrupt")
            payload, source = ckio.load_resume_checkpoint(run, expected)
            self.assertEqual(source, "checkpoint_previous")
            self.assertEqual(payload["global_step"], 10)

    def test_resume_checkpoint_rejects_config_and_dataset_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            ckio.save_resume_checkpoint(run, {"architecture": "maxvit512", "seed": 42,
                                               "config_signature": "old", "dataset_signature": "old"})
            payload, reason = ckio.load_resume_checkpoint(run, {"architecture": "maxvit512", "seed": 42,
                                                                 "config_signature": "new", "dataset_signature": "new"})
            self.assertIsNone(payload)
            self.assertIn("incompatible", reason)

    def test_experiment_id_roundtrip(self):
        eid = ckio.experiment_id("maxvit512", "RSB_CONTROLLED_G04", 17)
        self.assertEqual(eid, "maxvit512__RSB_CONTROLLED_G04__seed17")
        parsed = ckio.parse_experiment_id(eid)
        self.assertEqual(parsed, {"architecture": "maxvit512", "dataset_variant_id": "RSB_CONTROLLED_G04", "seed": 17})

    def test_run_dir_and_results_dir_do_not_collide_and_use_canonical_layout(self):
        root = Path("/tmp/fake-root")
        run = ckio.run_dir(root, "maxvit512", "R", "maxvit512_standard", 17)
        res = ckio.results_dir(root, "maxvit512", "R", "maxvit512_standard", 17)
        self.assertIn("experiments/classifiers_matrix", str(run))
        self.assertIn("results/classifiers_matrix", str(res))
        self.assertTrue(str(run).endswith("maxvit512/R/maxvit512_standard/seed_17"))

    def test_checkpoint_present_and_verified(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / "run"; write_checkpoint(run)
            verified, reason = ckio.checkpoint_is_verified(run, "pytorch_timm")
            self.assertTrue(verified, reason)

    def test_checkpoint_missing_is_not_verified(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / "run"; run.mkdir()
            verified, reason = ckio.checkpoint_is_verified(run, "pytorch_timm")
            self.assertFalse(verified)
            self.assertIn("no checkpoint_metadata.json", reason)

    def test_checkpoint_changed_after_metadata_is_incompatible(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / "run"; write_checkpoint(run)
            (run / "model.pt").write_bytes(b"different-weights-now")
            verified, reason = ckio.checkpoint_is_verified(run, "pytorch_timm")
            self.assertFalse(verified)
            self.assertIn("incompatible", reason)

    def test_legacy_alias_only_matches_declared_architecture(self):
        variant = {"legacy_experiment_ids": ["maxvit512_02a_real_only", "resnet50_01a_real_only"]}
        registry = {"experiments": [
            {"experiment_id": "maxvit512_02a_real_only", "architecture": "MaxViT-512"},
            {"experiment_id": "resnet50_01a_real_only", "architecture": "ResNet-50"},
        ]}
        self.assertEqual(ckio.legacy_alias_for(variant, registry, "MaxViT-512"), "maxvit512_02a_real_only")
        self.assertEqual(ckio.legacy_alias_for(variant, registry, "RAD-DINO"), None)

    def test_legacy_alias_never_invents_a_match(self):
        variant = {"legacy_experiment_ids": []}
        registry = {"experiments": [{"experiment_id": "maxvit512_02a_real_only", "architecture": "MaxViT-512"}]}
        self.assertIsNone(ckio.legacy_alias_for(variant, registry, "MaxViT-512"))


class RunManifestStateTests(unittest.TestCase):
    def test_reconstructed_state_pending_when_nothing_exists(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / "run"
            self.assertEqual(manifest.reconstruct_state(run, "pytorch_timm")["state"], "PENDING")

    def test_reconstructed_state_trained_when_checkpoint_verified_and_no_validation(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / "run"; write_checkpoint(run)
            self.assertEqual(manifest.reconstruct_state(run, "pytorch_timm")["state"], "TRAINED")

    def test_reconstructed_state_validated_when_metrics_present(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / "run"; write_checkpoint(run)
            (run / "validation_metrics.json").write_text("{}")
            self.assertEqual(manifest.reconstruct_state(run, "pytorch_timm")["state"], "VALIDATED")

    def test_reconstructed_state_complete_when_ensemble_marker_present(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / "run"; write_checkpoint(run)
            (run / "validation_metrics.json").write_text("{}")
            (run.parent / "ensemble_complete.json").write_text("{}")
            self.assertEqual(manifest.reconstruct_state(run, "pytorch_timm")["state"], "COMPLETE")

    def test_explicit_blocked_state_is_not_silently_overridden_by_rescan(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / "run"; write_checkpoint(run)  # would otherwise reconstruct as TRAINED
            manifest.write_state(run, "BLOCKED", reason="operator hold")
            self.assertEqual(manifest.reconstruct_state(run, "pytorch_timm")["state"], "BLOCKED")

    def test_claim_then_release_allows_a_second_claim(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / "run"
            self.assertTrue(manifest.acquire_claim(run, "worker-a", os.getpid()))
            self.assertFalse(manifest.acquire_claim(run, "worker-b", os.getpid()))  # same live pid: still held
            manifest.release_claim(run)
            self.assertTrue(manifest.acquire_claim(run, "worker-b", os.getpid()))

    def test_stale_lock_from_dead_pid_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / "run"; run.mkdir(parents=True)
            dead_pid = 999999999  # astronomically unlikely to be a live PID in this sandbox
            manifest.lock_path(run).write_text(json.dumps({"worker_id": "ghost", "pid": dead_pid, "claimed_at": time.time()}))
            self.assertTrue(manifest.acquire_claim(run, "worker-new", os.getpid()))

    def test_running_state_reflects_a_live_claim_with_no_checkpoint_yet(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / "run"
            manifest.acquire_claim(run, "worker-a", os.getpid())
            self.assertEqual(manifest.reconstruct_state(run, "pytorch_timm")["state"], "RUNNING")

    def test_invalid_state_name_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                manifest.write_state(Path(t) / "run", "NOT_A_REAL_STATE")


class RunnerPlanTests(unittest.TestCase):
    def _project(self, root: Path) -> None:
        (root / "configs").mkdir(parents=True)
        (root / "configs/dataset_variant_registry.json").write_text(json.dumps({"variants": [
            {"dataset_variant_id": "R", "status": "ready", "legacy_experiment_ids": []},
            {"dataset_variant_id": "BROKEN", "status": "invalid", "invalid_reason": "no synthetic evidence"},
        ]}))
        (root / "configs/classifier_training_protocols.json").write_text(json.dumps({"policies": {
            "maxvit512": {"architecture": "MaxViT-512", "framework": "pytorch_timm"},
        }}))
        (root / "configs/final_classifier_registry.json").write_text(json.dumps({"experiments": []}))

    def test_plan_errors_on_invalid_dataset_variant(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._project(root)
            result = runner.plan(root, "maxvit512", "BROKEN", 17)
            self.assertEqual(result["action"], "error")

    def test_plan_trains_when_no_checkpoint_and_no_legacy_alias(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._project(root)
            result = runner.plan(root, "maxvit512", "R", 17)
            self.assertEqual(result["action"], "train")
            self.assertEqual(result["state"], "PENDING")

    def test_plan_skips_training_when_checkpoint_verified(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._project(root)
            job = runner.resolve_job(root, "maxvit512", "R", 17)
            write_checkpoint(job["run_dir"])
            result = runner.plan(root, "maxvit512", "R", 17)
            self.assertEqual(result["action"], "skip_training")
            # checkpoint verified only answers the training question; validation is separate
            # and has not run yet (no validation_metrics.json), so it must still be requested.
            self.assertTrue(result["needs_validation"])

    def test_plan_needs_no_further_work_once_validated(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._project(root)
            job = runner.resolve_job(root, "maxvit512", "R", 17)
            write_checkpoint(job["run_dir"])
            (job["run_dir"] / "validation_metrics.json").write_text("{}")
            result = runner.plan(root, "maxvit512", "R", 17)
            self.assertEqual(result["action"], "skip_training")
            self.assertFalse(result["needs_validation"])

    def test_plan_flags_validation_needed_when_trained_but_not_validated(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._project(root)
            job = runner.resolve_job(root, "maxvit512", "R", 17)
            write_checkpoint(job["run_dir"])
            result = runner.plan(root, "maxvit512", "R", 17)
            self.assertTrue(result["needs_validation"])

    def test_plan_retrains_matrix_seed_even_when_legacy_alias_exists(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._project(root)
            registry = json.loads((root / "configs/dataset_variant_registry.json").read_text())
            registry["variants"][0]["legacy_experiment_ids"] = ["maxvit512_02a_real_only"]
            (root / "configs/dataset_variant_registry.json").write_text(json.dumps(registry))
            (root / "configs/final_classifier_registry.json").write_text(json.dumps({"experiments": [
                {"experiment_id": "maxvit512_02a_real_only", "architecture": "MaxViT-512"}]}))
            result = runner.plan(root, "maxvit512", "R", 17)
            self.assertEqual(result["action"], "train")
            self.assertIsNone(result["legacy_checkpoint_alias"])

    def test_three_seeds_produce_three_independent_run_dirs(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._project(root)
            dirs = {runner.resolve_job(root, "maxvit512", "R", seed)["run_dir"] for seed in (17, 42, 73)}
            self.assertEqual(len(dirs), 3)

    def _tiny_bundle(self):
        rows = [{"processed_path": "synthetic-fixture", "label": label, "image_id": str(i)}
                for i, label in enumerate((0, 1, 0, 1))]
        return rows, rows, {"signature": "tiny-dataset-signature"}

    def test_train_mode_without_train_fn_uses_registered_adapter(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._project(root)
            result = runner.run_train(root, "maxvit512", "R", 17, tiny=True,
                                      dataset_bundle=self._tiny_bundle())
            self.assertEqual(result["status"], "trained")
            job = runner.resolve_job(root, "maxvit512", "R", 17)
            state = manifest.read_manifest(job["run_dir"])
            self.assertEqual(state["state"], "TRAINED")

    def test_train_mode_with_injected_train_fn_writes_trained_state(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._project(root)

            def fake_train_fn(run_dir, policy, variant, seed):
                write_checkpoint(run_dir)
                return run_dir / "model.pt"

            result = runner.run_train(root, "maxvit512", "R", 17, train_fn=fake_train_fn)
            self.assertEqual(result["status"], "trained")
            job = runner.resolve_job(root, "maxvit512", "R", 17)
            self.assertEqual(manifest.read_manifest(job["run_dir"])["state"], "TRAINED")

    def test_train_mode_releases_claim_even_on_failure(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._project(root)
            class BrokenAdapter:
                def train(self, *_args, **_kwargs):
                    raise RuntimeError("intentional adapter failure")
            with self.assertRaises(RuntimeError):
                runner.run_train(root, "maxvit512", "R", 17, adapter=BrokenAdapter(),
                                 dataset_bundle=self._tiny_bundle())
            job = runner.resolve_job(root, "maxvit512", "R", 17)
            self.assertFalse(manifest.lock_path(job["run_dir"]).is_file())

    def test_auto_train_validate_and_three_seed_ensemble_with_tiny_adapter(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._project(root)
            for seed in (17, 42, 73):
                result = runner.run_auto(root, "maxvit512", "R", seed, tiny=True,
                                         dataset_bundle=self._tiny_bundle())
                self.assertEqual(result["status"], "validated")
            ensemble = root / "results/classifiers_matrix/maxvit512/R/maxvit512_standard/ensemble_validation_manifest.json"
            payload = json.loads(ensemble.read_text())
            self.assertEqual(payload["seeds"], [17, 42, 73])
            self.assertEqual(payload["aggregation"], "mean_probability")
            self.assertFalse(payload["test_access"])

    def test_cli_refuses_locked_test_mode(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._project(root)
            test_args = ["prog", "--experiment-id", "maxvit512__R__seed17", "--mode", "locked-test", "--project-root", str(root)]
            with patch.object(sys, "argv", test_args):
                with self.assertRaises(SystemExit) as ctx:
                    runner.main()
            self.assertEqual(ctx.exception.code, 2)


class DatasetBuilderCandidateResolutionTests(unittest.TestCase):
    def test_stale_metrics_path_falls_back_to_canonical_directory(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            metrics = root / "results/diffusers/gX/metrics/final_test_metrics.json"
            metrics.parent.mkdir(parents=True)
            metrics.write_text(json.dumps({"per_class": {"positive": {"generated_dir": "/gone/machine/experiments/old_name/final/positive_filtered"}}}))
            canonical = root / "experiments/diffusers/gX/generated_images/final/positive"
            canonical.mkdir(parents=True)
            for i in range(5):
                (canonical / f"img_{i}.png").write_bytes(b"png")
            entry = {"id": "gX", "metrics": "results/diffusers/gX/metrics/final_test_metrics.json"}
            found, precision = cdb._synthetic_candidate_files(root, entry, "positive")
            self.assertEqual(len(found), 5)
            self.assertEqual(precision, "canonical_directory_fallback_unverified_against_metrics_count")

    def test_metrics_declared_directory_preferred_over_canonical_when_both_exist(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            declared = root / "data/synthetic/gY/positive"; declared.mkdir(parents=True)
            for i in range(3): (declared / f"a_{i}.png").write_bytes(b"png")
            canonical = root / "experiments/diffusers/gY/generated_images/final/positive"; canonical.mkdir(parents=True)
            for i in range(9): (canonical / f"b_{i}.png").write_bytes(b"png")
            metrics = root / "results/diffusers/gY/metrics/final_filtered_vs_test.json"
            metrics.parent.mkdir(parents=True)
            metrics.write_text(json.dumps({"config": {"synthetic_dir": str(declared)}, "metrics": {"n_synthetic_filtered": 3, "target_label": 1}}))
            entry = {"id": "gY", "metrics": "results/diffusers/gY/metrics/final_filtered_vs_test.json"}
            found, precision = cdb._synthetic_candidate_files(root, entry, "positive")
            self.assertEqual(len(found), 3)
            self.assertEqual(precision, "metrics_declared_directory")

    def test_no_candidates_found_when_neither_path_exists(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            entry = {"id": "gZ"}
            found, precision = cdb._synthetic_candidate_files(root, entry, "positive")
            self.assertEqual(found, [])
            self.assertEqual(precision, "no_candidates_found")

    def test_build_file_list_raises_clean_error_when_budget_exceeds_available_candidates(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            variant = {"dataset_variant_id": "V", "real_source": None, "augmentation_source": None,
                       "synthetic_generator_id": "gZ", "synthetic_count_by_class": {"positive": 10}, "seed": 42}
            gen_registry = {"generators": [{"id": "gZ", "classes": ["positive"]}]}
            with self.assertRaises(ValueError) as ctx:
                cdb.build_file_list(root, variant, gen_registry)
            self.assertIn("no_candidates_found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
