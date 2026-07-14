from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import classifier_checkpoint_io as ckio  # noqa: E402
import classifier_experiment_runner as runner  # noqa: E402
import classifier_run_manifest as manifest  # noqa: E402
from classifier_pipeline_contracts import signed_payload  # noqa: E402
from downstream_lifecycle import aggregate_patient, create_test_lock, inventory  # noqa: E402


class EnsembleAndLockTests(unittest.TestCase):
    def fixture_root(self, temporary: str) -> Path:
        root = Path(temporary)
        (root / "configs").mkdir()
        for name in ("downstream_classifier_protocol.json", "downstream_classifier_jobs.json",
                     "generator_benchmark_protocol.json", "generator_registry.json"):
            shutil.copy(ROOT / "configs" / name, root / "configs" / name)
        return root

    def seed_artifact(self, root: Path, seed: int, keys=(('p0', 'i0'), ('p1', 'i1'))):
        architecture, condition, policy = "maxvit512", "real_only", "maxvit512_fixed_protocol"
        run = ckio.run_dir(root, architecture, condition, policy, seed)
        result = ckio.results_dir(root, architecture, condition, policy, seed)
        run.mkdir(parents=True); result.mkdir(parents=True)
        checkpoint = ckio.checkpoint_path(run, "pytorch_timm"); checkpoint.write_bytes(f"seed-{seed}".encode())
        ckio.write_checkpoint_metadata(run, architecture=architecture, dataset_variant_id=condition,
                                       training_policy=policy, seed=seed, checkpoint=checkpoint,
                                       dataset_manifest_sha256="dataset", protocol_signature="protocol")
        rows = [{"patient_id": patient, "image_id": image, "label": index,
                 "probability": 0.2 + 0.6 * index + seed * 1e-6, "processed_path": f"/{image}.png"}
                for index, (patient, image) in enumerate(keys)]
        predictions = signed_payload({"schema_version": 2, "pipeline_namespace": "mammodiffusion.downstream_validation.v1",
            "artifact_type": "classifier_validation_predictions", "architecture": architecture,
            "experiment_id": ckio.experiment_id(architecture, condition, seed), "dataset_variant_id": condition,
            "training_policy": policy, "seed": seed, "split": "validation", "dataset_signature": "dataset",
            "validation_signature": "validation", "checkpoint_signature": ckio.checkpoint_signature(checkpoint),
            "labels": [row["label"] for row in rows], "probabilities": [row["probability"] for row in rows], "rows": rows})
        (result / f"validation_predictions_seed_{seed}.json").write_text(json.dumps(predictions))
        metric = signed_payload({"schema_version": 2, "pipeline_namespace": "mammodiffusion.downstream_validation.v1",
                                 "artifact_type": "classifier_validation_metrics", "pr_auc": 1.0, "roc_auc": 1.0})
        (result / f"validation_metrics_seed_{seed}.json").write_text(json.dumps(metric))
        for state in ("RUNNING", "TRAINED", "VALIDATING", "VALIDATED"):
            manifest.write_state(run, state)

    def test_ensemble_waits_for_exact_three_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_root(temporary); self.seed_artifact(root, 17)
            result = runner.build_ensemble_if_ready(root, "maxvit512", "real_only")
            self.assertEqual(result, {"status": "waiting_for_seeds", "missing_seed": 42})

    def test_ensemble_checks_prediction_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_root(temporary)
            self.seed_artifact(root, 17); self.seed_artifact(root, 42)
            self.seed_artifact(root, 73, keys=(('p0', 'i0'), ('p2', 'i2')))
            with self.assertRaisesRegex(RuntimeError, "missing or inconsistent"):
                runner.build_ensemble_if_ready(root, "maxvit512", "real_only")

    def test_ensemble_completes_and_averages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_root(temporary)
            for seed in (17, 42, 73): self.seed_artifact(root, seed)
            result = runner.build_ensemble_if_ready(root, "maxvit512", "real_only")
            self.assertEqual(result["status"], "complete")
            payload = json.loads(Path(result["manifest"]).read_text())
            self.assertEqual(payload["seeds"], [17, 42, 73])
            self.assertFalse(payload["test_access"])

    def test_patient_aggregation_is_not_image_independent(self):
        rows = [{"patient_id": "p", "label": 1, "probability": 0.2},
                {"patient_id": "p", "label": 1, "probability": 0.8}]
        self.assertEqual(aggregate_patient(rows), [{"patient_id": "p", "label": 1, "probability": 0.5, "n_images": 2}])

    def test_patient_aggregation_rejects_label_conflict(self):
        with self.assertRaises(ValueError):
            aggregate_patient([{"patient_id": "p", "label": 0, "probability": 0.2},
                               {"patient_id": "p", "label": 1, "probability": 0.8}])

    def test_status_does_not_require_or_read_test_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_root(temporary)
            report = inventory(root)
            self.assertEqual(report["job_counts"], {"PENDING": 24})
            self.assertEqual(report["locked_test_status"], "unopened")

    def test_lock_requires_all_jobs_before_test_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture_root(temporary)
            with self.assertRaises(FileNotFoundError):
                create_test_lock(root)  # approval is the first fail-closed prerequisite


if __name__ == "__main__":
    unittest.main()
