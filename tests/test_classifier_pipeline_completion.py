from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
sys.path.insert(0, str(ROOT / "scripts"))

import classifier_final_report as final_report
import classifier_gpu_gate as gpu_gate
import classifier_gpu_scheduler as gpu_scheduler
import classifier_pipeline_contracts as contracts
import classifier_preflight as preflight
import classifier_pipeline_status as pipeline_status
import classifier_run_manifest as run_manifest
import classifier_checkpoint_io as checkpoint_io
import classifier_experiment_runner as experiment_runner
import create_classifier_stage2_notebooks as stage2_notebooks
import finalize_validation_stage as validation_finalizer
import run_classifier_experiment_matrix as matrix_runner
import resume_classifier_experiment_matrix as resume_matrix
import finalize_locked_test_stage as lock_finalizer
import locked_matrix_inference
import run_locked_classifier_inference


class CanonicalStateMachineTests(unittest.TestCase):
    def test_required_valid_transitions(self):
        for previous, target in (
            ("PENDING", "RUNNING"), ("RUNNING", "INTERRUPTED_RESUMABLE"),
            ("INTERRUPTED_RESUMABLE", "RUNNING"), ("RUNNING", "FAILED_RETRYABLE"),
            ("FAILED_RETRYABLE", "RUNNING"), ("RUNNING", "FAILED_FINAL"),
            ("RUNNING", "TRAINED"), ("TRAINED", "VALIDATED"), ("TRAINED", "VALIDATING"),
            ("VALIDATING", "VALIDATED"), ("VALIDATED", "ENSEMBLED"),
            ("ENSEMBLED", "COMPLETE"),
        ):
            self.assertTrue(contracts.transition_allowed(previous, target), (previous, target))

    def test_invalid_scientific_shortcuts_and_terminal_restart_are_rejected(self):
        for previous, target in (("PENDING", "VALIDATED"), ("TRAINED", "COMPLETE"),
                                 ("FAILED_FINAL", "RUNNING"), ("VALIDATED", "COMPLETE")):
            self.assertFalse(contracts.transition_allowed(previous, target), (previous, target))
        self.assertTrue(contracts.transition_allowed("FAILED_FINAL", "PENDING", explicit_reset=True))

    def test_signed_state_manifest_detects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            run_manifest.write_state(run, "RUNNING")
            payload = json.loads((run / "run_manifest.json").read_text()); payload["state"] = "COMPLETE"
            (run / "run_manifest.json").write_text(json.dumps(payload))
            self.assertIsNone(run_manifest.read_manifest(run))

    def test_failed_final_requires_explicit_reasoned_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "configs").mkdir()
            job = {"experiment_id": "resnet50__R__seed17", "stage": 1, "architecture": "resnet50",
                   "dataset_variant_id": "R", "training_policy": "resnet50_standard", "seed": 17,
                   "status": "FAILED_FINAL"}
            contracts.atomic_json(root / "configs/classifier_experiment_matrix.json", {
                "schema_version": 2, "pipeline_namespace": contracts.PIPELINE_NAMESPACE, "jobs": [job]})
            run = checkpoint_io.run_dir(root, "resnet50", "R", "resnet50_standard", 17)
            run_manifest.write_state(run, "RUNNING"); run_manifest.write_state(run, "FAILED_FINAL")
            with self.assertRaises(ValueError): resume_matrix.reset_failed_final(root, job["experiment_id"], "")
            result = resume_matrix.reset_failed_final(root, job["experiment_id"], "incident fixed")
            self.assertEqual(result["to"], "PENDING")


class GpuGateAndSchedulerGuardTests(unittest.TestCase):
    def _record(self, architecture: str, timestamp: str) -> tuple[dict, dict]:
        common = {"architecture": architecture, "environment_signature": "env", "gpu_name": "fixture-gpu",
                  "gpu_uuid": "GPU-fixture", "total_vram_mb": 16000, "physical_batch_size": 2,
                  "gradient_accumulation_steps": 4, "measured_at": timestamp,
                  "code_revision": "fixture", "fixture_signature": "zeros-v1"}
        return ({**common, "effective_batch_size": 8, "peak_allocated_mb": 1200,
                 "peak_reserved_mb": 1400},
                {**common, "forward_pass": True, "backward_pass": True,
                 "checkpoint_save_load": True})

    def test_signed_profile_and_smoke_bundle_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); runtime = root / "results/runtime_profiles"; runtime.mkdir(parents=True)
            now = datetime.now(timezone.utc); timestamp = now.isoformat()
            records = [self._record(architecture, timestamp) for architecture in contracts.ARCHITECTURES]
            (runtime / "classifier_vram_profiles.json").write_text(json.dumps(
                gpu_gate.make_bundle("gpu_profile_bundle", [record[0] for record in records])))
            (runtime / "classifier_gpu_smoke_results.json").write_text(json.dumps(
                gpu_gate.make_bundle("gpu_smoke_bundle", [record[1] for record in records])))
            result = gpu_gate.validate_gate(root, now=now)
            self.assertTrue(result["ready_for_real_launch"], result["errors"])
            payload = json.loads((runtime / "classifier_vram_profiles.json").read_text())
            payload["records"][0]["peak_allocated_mb"] = 1
            (runtime / "classifier_vram_profiles.json").write_text(json.dumps(payload))
            self.assertFalse(gpu_gate.validate_gate(root, now=now)["profiles_valid"])

    def test_double_scheduler_lock_and_dead_owner_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertTrue(matrix_runner.acquire_scheduler_lock(root, 1))
            self.assertFalse(matrix_runner.acquire_scheduler_lock(root, 1))
            matrix_runner.release_scheduler_lock(root, 1)
            path = matrix_runner.scheduler_lock_path(root, 1); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"pid": 99999999, "stage": 1}))
            self.assertTrue(matrix_runner.acquire_scheduler_lock(root, 1))

    def test_real_scheduler_mode_explicitly_rejects_missing_architecture_profile(self):
        gpu = [{"index": 0, "name": "NVIDIA GeForce RTX 5060 Ti", "uuid": "GPU-x",
                "total_vram_mb": 16000, "free_vram_mb": 15000}]
        scheduler = gpu_scheduler.Scheduler(gpu, vram_profiles={}, strict_profiles=True)
        decision = scheduler.try_admit({"experiment_id": "x", "architecture": "maxvit512",
            "resource_profile": "heavy", "gpu_eligibility": ["rtx_5060_ti_16gb"]})
        self.assertFalse(decision["admitted"])
        self.assertIn("missing", decision["reason"].lower())


class StaticPreflightFixtureTests(unittest.TestCase):
    def test_expected_baseline_is_ready_to_profile_not_ready_to_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "configs").mkdir(); (root / "results/notebook_inventory").mkdir(parents=True)
            inventory = []
            for index in range(112):
                status = "READY" if index < 100 else "BLOCKED"
                relative = f"notebooks/n{index}.ipynb"; path = root / relative; path.parent.mkdir(exist_ok=True)
                path.write_text("{}")
                inventory.append({"stage": 1, "path": relative, "dataset_status": status,
                                  "note_blocker": None if status == "READY" else "fixture blocker"})
            (root / "results/notebook_inventory/notebook_inventory.json").write_text(json.dumps(inventory))
            variants = [{"dataset_variant_id": f"V{i}", "status": "ready", "regime": "stage1_screening"}
                        for i in range(25)]
            v2 = {"schema_version": 2, "pipeline_namespace": contracts.PIPELINE_NAMESPACE}
            (root / "configs/dataset_variant_registry.json").write_text(json.dumps({**v2, "variants": variants}))
            policies = {architecture: {"framework": "fixture"} for architecture in contracts.ARCHITECTURES}
            (root / "configs/classifier_training_protocols.json").write_text(json.dumps({**v2, "policies": policies}))
            jobs = []
            for architecture in contracts.ARCHITECTURES:
                for variant in variants:
                    for seed in contracts.REQUIRED_SEEDS:
                        jobs.append({"experiment_id": f"{architecture}__{variant['dataset_variant_id']}__seed{seed}",
                            "stage": 1, "architecture": architecture, "dataset_variant_id": variant["dataset_variant_id"],
                            "training_policy": f"{architecture}_standard", "seed": seed, "status": "PENDING"})
            (root / "configs/classifier_experiment_matrix.json").write_text(json.dumps({"schema_version": 2,
                "pipeline_namespace": contracts.PIPELINE_NAMESPACE, "jobs": jobs}))
            for relative in preflight.REQUIRED_SCRIPTS:
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("# fixture\n")
            result = preflight.build_preflight(root, deep_dataset_check=False)
            self.assertEqual(result["readiness"], "READY_TO_PROFILE_GPU", result["errors"])
            self.assertEqual(result["counts"]["stage1_jobs"], 300)
            status = pipeline_status.build_status(root)
            self.assertFalse(status["read_locked_test"])
            self.assertEqual(status["stages"]["1"]["jobs_by_state"], {"PENDING": 300})
            self.assertIn("scientific_lock", status)


class SelectionAndStage2ContractTests(unittest.TestCase):
    def test_incomplete_stage1_union_cannot_be_persisted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "configs").mkdir()
            job = {"experiment_id": "resnet50__RSB_CONTROLLED_G01__seed17", "stage": 1,
                   "architecture": "resnet50", "dataset_variant_id": "RSB_CONTROLLED_G01",
                   "training_policy": "resnet50_standard", "seed": 17, "status": "PENDING"}
            (root / "configs/classifier_experiment_matrix.json").write_text(json.dumps({
                "schema_version": 2, "pipeline_namespace": contracts.PIPELINE_NAMESPACE, "jobs": [job]}))
            payload = validation_finalizer.compute_selected_generator_union(root)
            self.assertFalse(payload["scientific_completion"]["complete"])
            with self.assertRaises(RuntimeError):
                validation_finalizer.write_selected_union(root, payload)

    def test_stage2_rejects_legacy_or_tampered_union(self):
        with self.assertRaises(ValueError):
            stage2_notebooks.verify_union({"schema_version": 1, "selected_generator_union": ["G01"]})
        leaderboard = contracts.signed_payload({"schema_version": 2,
            "pipeline_namespace": contracts.PIPELINE_NAMESPACE,
            "artifact_type": "classifier_stage1_validation_leaderboard", "rows": []})
        union = contracts.signed_payload({"schema_version": 2,
            "pipeline_namespace": contracts.PIPELINE_NAMESPACE,
            "artifact_type": "classifier_selected_generator_union", "selected_generator_union": ["G01"],
            "selection_used_test_data": False, "scientific_completion": {"complete": True},
            "leaderboard": leaderboard, "leaderboard_signature": leaderboard["signature"]})
        self.assertEqual(stage2_notebooks.verify_union(union), ["G01"])
        union["leaderboard"]["rows"] = ["changed"]
        with self.assertRaises(ValueError):
            stage2_notebooks.verify_union(union)

    def test_stage2_fixture_notebook_generation_is_byte_idempotent(self):
        leaderboard = contracts.signed_payload({"schema_version": 2,
            "pipeline_namespace": contracts.PIPELINE_NAMESPACE,
            "artifact_type": "classifier_stage1_validation_leaderboard", "rows": []})
        union = contracts.signed_payload({"schema_version": 2,
            "pipeline_namespace": contracts.PIPELINE_NAMESPACE,
            "artifact_type": "classifier_selected_generator_union", "selected_generator_union": ["G01"],
            "selection_used_test_data": False, "scientific_completion": {"complete": True},
            "leaderboard": leaderboard, "leaderboard_signature": leaderboard["signature"]})
        variant = {"dataset_variant_id": "RAS_CONTROLLED_G01", "budget_regime": "controlled",
                   "synthetic_generator_id": "G01", "status": "ready"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); union_path = root / "union.json"; union_path.write_text(json.dumps(union))
            with patch.object(stage2_notebooks, "build_stage2_variants", return_value=[variant]), \
                 patch.object(stage2_notebooks.stage1, "dataset_audit", return_value=("READY", None)):
                first = stage2_notebooks.generate(root, union_path)
                snapshots = {row["path"]: (root / row["path"]).read_bytes() for row in first}
                second = stage2_notebooks.generate(root, union_path)
                self.assertEqual(first, second)
                self.assertEqual(snapshots, {row["path"]: (root / row["path"]).read_bytes() for row in second})


class EnsembleRecoveryContractTests(unittest.TestCase):
    def _fixture(self, root: Path, *, omit_seed: int | None = None, signature_by_seed=None,
                 reverse_seed: int | None = None) -> None:
        (root / "configs").mkdir()
        (root / "configs/classifier_training_protocols.json").write_text(json.dumps({"policies": {
            "maxvit512": {"framework": "pytorch_timm"}}}))
        rows = [{"patient_id": "p0", "image_id": "i0", "label": 0, "probability": 0.1,
                 "processed_path": "a.png"},
                {"patient_id": "p1", "image_id": "i1", "label": 1, "probability": 0.9,
                 "processed_path": "b.png"}]
        for seed in contracts.REQUIRED_SEEDS:
            if seed == omit_seed: continue
            run = checkpoint_io.run_dir(root, "maxvit512", "R", "maxvit512_standard", seed)
            result = checkpoint_io.results_dir(root, "maxvit512", "R", "maxvit512_standard", seed)
            run.mkdir(parents=True); result.mkdir(parents=True)
            checkpoint = run / "model.pt"; checkpoint.write_bytes(f"weights-{seed}".encode())
            checkpoint_io.write_checkpoint_metadata(run, architecture="maxvit512", dataset_variant_id="R",
                training_policy="maxvit512_standard", seed=seed, checkpoint=checkpoint,
                dataset_manifest_sha256="dataset", protocol_signature="protocol")
            validation_rows = list(reversed(rows)) if seed == reverse_seed else rows
            payload = contracts.signed_payload({"schema_version": 2,
                "pipeline_namespace": contracts.PIPELINE_NAMESPACE,
                "artifact_type": "classifier_validation_predictions", "architecture": "maxvit512",
                "experiment_id": f"maxvit512__R__seed{seed}", "dataset_variant_id": "R",
                "training_policy": "maxvit512_standard", "seed": seed, "split": "validation",
                "dataset_signature": "dataset", "validation_signature": (signature_by_seed or {}).get(seed, "validation"),
                "checkpoint_signature": checkpoint_io.checkpoint_signature(checkpoint),
                "labels": [row["label"] for row in validation_rows],
                "probabilities": [row["probability"] for row in validation_rows], "rows": validation_rows})
            contracts.atomic_json(result / f"validation_predictions_seed_{seed}.json", payload)
            metrics = contracts.signed_payload({"schema_version": 2,
                "pipeline_namespace": contracts.PIPELINE_NAMESPACE,
                "artifact_type": "classifier_validation_metrics", "pr_auc": 1.0, "roc_auc": 1.0})
            contracts.atomic_json(result / f"validation_metrics_seed_{seed}.json", metrics)
            for state in ("RUNNING", "TRAINED", "VALIDATING", "VALIDATED"):
                run_manifest.write_state(run, state)

    def test_reordered_seed_rows_align_and_valid_ensemble_is_noop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self._fixture(root, reverse_seed=42)
            first = experiment_runner.build_ensemble_if_ready(root, "maxvit512", "R")
            self.assertEqual(first["status"], "complete")
            manifest_path = Path(first["manifest"]); before = manifest_path.read_bytes()
            second = experiment_runner.build_ensemble_if_ready(root, "maxvit512", "R")
            self.assertEqual(second["status"], "already_complete")
            self.assertEqual(before, manifest_path.read_bytes())

    def test_missing_seed_waits_and_signature_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self._fixture(root, omit_seed=73)
            self.assertEqual(experiment_runner.build_ensemble_if_ready(root, "maxvit512", "R")["status"],
                             "waiting_for_seeds")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self._fixture(root, signature_by_seed={42: "different"})
            with self.assertRaisesRegex(RuntimeError, "signatures differ"):
                experiment_runner.build_ensemble_if_ready(root, "maxvit512", "R")

    def test_corruption_after_complete_requires_incident(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self._fixture(root)
            result = experiment_runner.build_ensemble_if_ready(root, "maxvit512", "R")
            path = Path(result["manifest"]); payload = json.loads(path.read_text()); payload["metrics"]["pr_auc"] = 0
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "incident repair"):
                experiment_runner.build_ensemble_if_ready(root, "maxvit512", "R")


class RunnerArtifactCompatibilityTests(unittest.TestCase):
    def test_v2_plan_rejects_checkpoint_metadata_from_another_seed_or_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "configs").mkdir()
            v2 = {"schema_version": 2, "pipeline_namespace": contracts.PIPELINE_NAMESPACE}
            (root / "configs/dataset_variant_registry.json").write_text(json.dumps({**v2, "variants": [
                {"dataset_variant_id": "R", "status": "ready", "legacy_experiment_ids": []}]}))
            (root / "configs/classifier_training_protocols.json").write_text(json.dumps({**v2, "policies": {
                "maxvit512": {"framework": "pytorch_timm"}}}))
            job = {"experiment_id": "maxvit512__R__seed17", "stage": 1, "architecture": "maxvit512",
                "dataset_variant_id": "R", "training_policy": "maxvit512_standard", "seed": 17, "status": "PENDING"}
            (root / "configs/classifier_experiment_matrix.json").write_text(json.dumps({**v2, "jobs": [job]}))
            run = checkpoint_io.run_dir(root, "maxvit512", "R", "maxvit512_standard", 17); run.mkdir(parents=True)
            checkpoint = run / "model.pt"; checkpoint.write_bytes(b"wrong-job")
            checkpoint_io.write_checkpoint_metadata(run, architecture="maxvit512", dataset_variant_id="R",
                training_policy="different_policy", seed=42, checkpoint=checkpoint,
                dataset_manifest_sha256="dataset", protocol_signature="protocol")
            result = experiment_runner.plan(root, "maxvit512", "R", 17)
            self.assertEqual(result["state"], "PENDING")
            self.assertEqual(result["action"], "train")


class PatientLevelFinalReportTests(unittest.TestCase):
    def test_multi_image_patient_aggregation_and_completion_marker(self):
        rows = [
            {"patient_id": "p0", "image_id": "a", "label": 0, "prob_ensemble": 0.1, "threshold": 0.5},
            {"patient_id": "p0", "image_id": "b", "label": 0, "prob_ensemble": 0.3, "threshold": 0.5},
            {"patient_id": "p1", "image_id": "c", "label": 1, "prob_ensemble": 0.8, "threshold": 0.5},
            {"patient_id": "p2", "image_id": "d", "label": 0, "prob_ensemble": 0.2, "threshold": 0.5},
            {"patient_id": "p3", "image_id": "e", "label": 1, "prob_ensemble": 0.9, "threshold": 0.5},
        ]
        patient = final_report.aggregate_patient_rows(rows)
        self.assertEqual(patient[0]["probability"], 0.2)
        self.assertEqual(patient[0]["n_images"], 2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); output = root / "results/final_evaluation_v2"; output.mkdir(parents=True)
            prediction = output / "predictions/primary/resnet50__R__ensemble.csv"
            prediction.parent.mkdir(parents=True)
            final_report.atomic_csv(prediction, rows)
            lock_signature = "fixture-lock"
            lock_payload = contracts.signed_payload({"schema_version": 2,
                "pipeline_namespace": contracts.PIPELINE_NAMESPACE,
                "artifact_type": "classifier_scientific_lock", "lock_signature": lock_signature})
            contracts.atomic_json(output / "EXPERIMENT_MATRIX_LOCKED", lock_payload)
            manifest = contracts.signed_payload({"schema_version": 2,
                "pipeline_namespace": contracts.PIPELINE_NAMESPACE,
                "artifact_type": "classifier_locked_predictions", "lock_signature": lock_signature,
                "outputs": [{"panel": "primary", "experiment_id": "resnet50__R__ensemble",
                             "path": str(prediction.relative_to(root)), "sha256": contracts.sha256_file(prediction)}]})
            contracts.atomic_json(output / "locked_test_predictions_manifest.json", manifest)
            (output / "LOCKED_TEST_COMPLETE").write_text("fixture\n")
            report = final_report.build_report(root, n_bootstrap=20, write_figures=False)
            self.assertTrue(report["final_aggregation_complete"])
            self.assertEqual(report["locked_test_evaluation"]["metrics"][0]["n_patients"], 4)
            self.assertTrue((output / "FINAL_AGGREGATION_COMPLETE").is_file())
            self.assertEqual(report, final_report.build_report(root, n_bootstrap=20, write_figures=False))


class ScientificLockAndOneShotTests(unittest.TestCase):
    def test_failed_attempt_requires_named_retry_and_never_allows_completed_rerun(self):
        from tests.test_final_matrix_lock import build_lockable_project
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); build_lockable_project(root)
            matrix_path = root / "configs/classifier_experiment_matrix.json"
            matrix = json.loads(matrix_path.read_text())
            for job in matrix["jobs"]:
                run = checkpoint_io.run_dir(root, job["architecture"], job["dataset_variant_id"],
                                            job["training_policy"], job["seed"])
                job["checkpoint_path"] = str((run / "model.pt").relative_to(root))
            matrix_path.write_text(json.dumps(matrix))
            lock_finalizer.finalize(root)
            def fail(_job, _checkpoint, _rows):
                raise RuntimeError("synthetic technical incident")
            with self.assertRaisesRegex(RuntimeError, "technical incident"):
                locked_matrix_inference.run_locked(root, predictor_fn=fail)
            with self.assertRaises(PermissionError):
                locked_matrix_inference.run_locked(root, predictor_fn=lambda *_: [0.5, 0.5])
            incident = "INC-FIXTURE-001"
            run_locked_classifier_inference.authorize_retry(root, incident)
            result = locked_matrix_inference.run_locked(root, predictor_fn=lambda *_: [0.1, 0.9],
                                                        incident_token=incident)
            self.assertTrue(result["one_shot"])
            with self.assertRaises(PermissionError):
                locked_matrix_inference.run_locked(root, predictor_fn=lambda *_: [0.1, 0.9])

    def test_v2_lock_is_signed_revision_bound_and_immutable(self):
        from tests.test_final_matrix_lock import build_lockable_project
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); build_lockable_project(root)
            lock_dir = root / "results/final_evaluation_v2"
            matrix_path = root / "configs/classifier_experiment_matrix.json"
            matrix = json.loads(matrix_path.read_text()); matrix.update({
                "schema_version": 2, "pipeline_namespace": contracts.PIPELINE_NAMESPACE})
            for job in matrix["jobs"]: job["status"] = "COMPLETE"
            matrix_path.write_text(json.dumps(matrix))
            union = contracts.signed_payload({"schema_version": 2,
                "pipeline_namespace": contracts.PIPELINE_NAMESPACE,
                "artifact_type": "classifier_selected_generator_union", "selected_generator_union": ["gA"],
                "scientific_completion": {"complete": True}})
            contracts.atomic_json(root / "results/generator_comparison/selected_generator_union.json", union)
            panel = json.loads((lock_dir / "primary_finalists_manifest.json").read_text())
            logical = "maxvit512__R__ensemble"; seeds = [f"maxvit512__R__seed{seed}" for seed in (17, 42, 73)]
            panel.update({"schema_version": 2, "pipeline_namespace": contracts.PIPELINE_NAMESPACE,
                "artifact_type": "classifier_final_panel_selection", "primary_locked_panel": [logical],
                "secondary_locked_panel": [], "seed_experiment_ids_by_logical": {logical: seeds},
                "stage2_completion": {"complete": True}, "test_data_used": False})
            panel = contracts.signed_payload(panel); contracts.atomic_json(lock_dir / "primary_finalists_manifest.json", panel)
            marker = lock_finalizer.finalize(root); contracts.verify_signed_payload(marker)
            self.assertNotEqual(marker["code_revision"], "")
            panel["primary_locked_panel"] = []
            contracts.atomic_json(lock_dir / "primary_finalists_manifest.json", panel)
            with self.assertRaisesRegex(RuntimeError, "immutable"):
                lock_finalizer.finalize(root)


if __name__ == "__main__":
    unittest.main()
