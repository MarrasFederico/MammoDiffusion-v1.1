from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
sys.path.insert(0, str(ROOT / "scripts"))

import classifier_architecture_adapters as adapters  # noqa: E402
import classifier_checkpoint_io as ckio  # noqa: E402
import classifier_dataset_builder as datasets  # noqa: E402
import classifier_experiment_runner as runner  # noqa: E402
import classifier_gpu_scheduler as scheduler  # noqa: E402
import finalize_locked_test_stage as lock  # noqa: E402


class ScientificDatasetSignatureTests(unittest.TestCase):
    def _file_list(self, root: Path) -> dict:
        (root / "data/train").mkdir(parents=True)
        (root / "data/train/a.png").write_bytes(b"pixels-a")
        return {"negative": [{"path": "data/train/a.png", "source": "augmented",
                               "patient_id": "p1", "image_id": "i1",
                               "augmentation_type": "rotate",
                               "source_split": "train",
                               "source_original_path": "data/processed/train/0/p1_i1.png"}],
                "positive": []}

    def test_train_png_content_change_invalidates_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = self._file_list(root)
            before = datasets.dataset_manifest_signature(root, files)
            (root / "data/train/a.png").write_bytes(b"different pixels")
            self.assertNotEqual(before, datasets.dataset_manifest_signature(root, files))

    def test_augmentation_patient_provenance_change_invalidates_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = self._file_list(root)
            before = datasets.dataset_manifest_signature(root, files)
            files["negative"][0]["patient_id"] = "p2"
            self.assertNotEqual(before, datasets.dataset_manifest_signature(root, files))

    def test_validation_label_or_path_change_invalidates_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/val").mkdir(parents=True)
            (root / "data/val/a.png").write_bytes(b"a")
            (root / "data/val/b.png").write_bytes(b"b")
            row = {"processed_path": str(root / "data/val/a.png"), "label": 0,
                   "source": "real_validation", "patient_id": "p1", "image_id": "i1"}
            before = datasets.validation_manifest_signature(root, [row])
            changed_label = {**row, "label": 1}
            changed_path = {**row, "processed_path": str(root / "data/val/b.png")}
            self.assertNotEqual(before, datasets.validation_manifest_signature(root, [changed_label]))
            self.assertNotEqual(before, datasets.validation_manifest_signature(root, [changed_path]))


class PyTorchRestartPolicyTests(unittest.TestCase):
    def test_restart_replays_incomplete_epoch_sample_order_from_start(self):
        import torch
        sample_ids = list(range(12))

        def order():
            generator = torch.Generator().manual_seed(73)
            loader = torch.utils.data.DataLoader(sample_ids, batch_size=3, shuffle=True, generator=generator)
            return [int(value) for batch in loader for value in batch]

        original = order()
        ids_seen_before_crash = original[:6]
        start_epoch, start_batch = adapters._pytorch_resume_position({"epoch": 4, "batch_index": 1})
        restarted = order()
        self.assertEqual((start_epoch, start_batch), (4, 0))
        self.assertEqual(ids_seen_before_crash, restarted[:6])
        self.assertEqual(original, restarted)


class ExclusiveSchedulerRegressionTests(unittest.TestCase):
    def test_busy_3060_does_not_block_exclusive_job_on_idle_5060(self):
        gpus = [
            {"index": 0, "name": "NVIDIA GeForce RTX 3060", "uuid": "3060", "total_vram_mb": 12288, "free_vram_mb": 60000},
            {"index": 1, "name": "NVIDIA GeForce RTX 5060 Ti", "uuid": "5060", "total_vram_mb": 16384, "free_vram_mb": 60000},
        ]
        profiles = {"resnet50": {"peak_allocated_mb": 1000}}
        sched = scheduler.Scheduler(gpus, profiles)
        base = {"architecture": "resnet50", "gpu_eligibility": ["rtx_3060_12gb", "rtx_5060_ti_16gb"]}
        self.assertTrue(sched.try_admit({**base, "experiment_id": "busy", "resource_profile": "light"})["admitted"])
        result = sched.try_admit({**base, "experiment_id": "exclusive", "resource_profile": "exclusive"})
        self.assertTrue(result["admitted"], result)
        self.assertEqual(result["gpu_key"], "rtx_5060_ti_16gb")
        blocked = sched.try_admit({**base, "experiment_id": "follower", "resource_profile": "light",
                                   "gpu_eligibility": ["rtx_5060_ti_16gb"]})
        self.assertFalse(blocked["admitted"], "an admitted exclusive job must retain its GPU")


class EnsembleRecoveryTests(unittest.TestCase):
    def test_validated_seed_without_ensemble_triggers_recovery(self):
        final_plan = {"action": "skip_training", "state": "VALIDATED", "needs_validation": False}
        with patch.object(runner, "plan", return_value=final_plan), \
             patch.object(runner, "resolve_job", return_value={"policy": {}}), \
             patch.object(runner, "build_ensemble_if_ready", return_value={"status": "complete"}) as build:
            result = runner.run_auto(Path("/unused"), "maxvit512", "R", 17, adapter=object())
        build.assert_called_once_with(Path("/unused"), "maxvit512", "R")
        self.assertEqual(result["status"], "ensemble_recovery")


class LockFinalizerRegressionTests(unittest.TestCase):
    def _project(self, root: Path) -> None:
        (root / "configs").mkdir(parents=True)
        (root / "data/processed/metadata").mkdir(parents=True)
        (root / "data/processed/test/0").mkdir(parents=True)
        image = root / "data/processed/test/0/p1_i1.png"; image.write_bytes(b"test-pixels")
        (root / "data/processed/metadata/test.csv").write_text(
            "patient_id,image_id,label,processed_path\np1,i1,0,data/processed/test/0/p1_i1.png\n")
        architecture, variant, policy = "maxvit512", "R", "maxvit512_standard"
        jobs = []
        for seed in (17, 42, 73):
            run = root / f"experiments/classifiers_matrix/{architecture}/{variant}/{policy}/seed_{seed}"
            checkpoint = run / "model.pt"; checkpoint.parent.mkdir(parents=True, exist_ok=True); checkpoint.write_bytes(f"w{seed}".encode())
            ckio.write_checkpoint_metadata(run, architecture=architecture, dataset_variant_id=variant,
                                           training_policy=policy, seed=seed, checkpoint=checkpoint,
                                           dataset_manifest_sha256="data", protocol_signature="protocol")
            jobs.append({"experiment_id": f"{architecture}__{variant}__seed{seed}", "architecture": architecture,
                         "dataset_variant_id": variant, "training_policy": policy, "seed": seed, "status": "COMPLETE",
                         "manifest_path": str((run / "run_manifest.json").relative_to(root)),
                         "checkpoint_path": str(checkpoint.relative_to(root))})
        (root / "configs/classifier_experiment_matrix.json").write_text(json.dumps({"jobs": jobs}))
        (root / "configs/classifier_training_protocols.json").write_text(json.dumps({"policies": {
            architecture: {"framework": "pytorch_timm"}}}))
        for name in ("dataset_variant_registry.json", "final_generator_registry.json"):
            (root / "configs" / name).write_text("{}")
        ensemble = root / f"results/classifiers_matrix/{architecture}/{variant}/{policy}/ensemble"
        (ensemble / "predictions").mkdir(parents=True)
        (ensemble / "metrics").mkdir(parents=True)
        (ensemble / "predictions/ensemble_validation_predictions.csv").write_text("x\n")
        (ensemble / "predictions/ensemble_validation_predictions.json").write_text("{}\n")
        (ensemble / "metrics/locked_validation_threshold.json").write_text('{"threshold": 0.5}\n')
        lock_dir = root / lock.LOCK_DIR; lock_dir.mkdir(parents=True)
        logical = f"{architecture}__{variant}__ensemble"
        (lock_dir / "primary_finalists_manifest.json").write_text(json.dumps({
            "primary_finalists": {}, "secondary_locked_panel": [logical], "ablation_panel": [],
            "seed_experiment_ids_by_logical": {logical: [job["experiment_id"] for job in jobs]},
        }))

    def test_logical_secondary_panel_expands_only_for_checkpoint_signing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self._project(root)
            marker = lock.finalize(root)
            lock_dir = root / lock.LOCK_DIR
            panel = json.loads((lock_dir / "secondary_panel_manifest.json").read_text())
            checkpoints = json.loads((lock_dir / "primary_finalists_checkpoints.json").read_text())
            self.assertEqual(panel["experiment_ids"], ["maxvit512__R__ensemble"])
            self.assertEqual(sorted(checkpoints), [f"maxvit512__R__seed{s}" for s in (17, 42, 73)])
            self.assertIn("lock_signature", marker)
            import locked_matrix_inference
            calls = []
            def predictor(job, _checkpoint, test_rows):
                calls.append(job["experiment_id"])
                return [0.25 for _ in test_rows]
            inference = locked_matrix_inference.run_locked(root, predictor_fn=predictor)
            self.assertEqual(len(inference["outputs"]), 1)
            self.assertEqual(len(calls), 3)

    def test_missing_test_png_blocks_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self._project(root)
            (root / "data/processed/test/0/p1_i1.png").unlink()
            with self.assertRaisesRegex(RuntimeError, "exist and be readable"):
                lock.finalize(root)


@unittest.skipUnless(os.environ.get("RUN_REAL_TF_SAVE_LOAD") == "1", "enable in tf-gpu verification tier")
class RealTensorFlowFinalCheckpointTests(unittest.TestCase):
    def test_hdf5_final_checkpoint_loads_in_new_process_with_identical_predictions(self):
        import numpy as np
        import tensorflow as tf
        from resnet50_utils import build_resnet50_model
        tf.config.set_visible_devices([], "GPU")
        policy = {"input_size": [32, 32]}
        adapter = adapters.ArchitectureAdapter("resnet50", policy, ROOT)
        model, _ = build_resnet50_model((32, 32), pretrained=False)
        inputs = np.arange(2 * 32 * 32 * 3, dtype="float32").reshape(2, 32, 32, 3) / 255.0
        expected = model.predict(inputs, verbose=0).reshape(-1)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.keras"
            adapter.save_checkpoint(model, checkpoint)
            code = ("import json,numpy as np,sys; sys.path.insert(0,sys.argv[1]); "
                    "from classifier_architecture_adapters import ArchitectureAdapter; "
                    "m=ArchitectureAdapter('resnet50',{'input_size':[32,32]},sys.argv[2]).load_checkpoint(sys.argv[3]); "
                    "x=np.arange(2*32*32*3,dtype='float32').reshape(2,32,32,3)/255.; "
                    "print('PRED='+json.dumps(m.predict(x,verbose=0).reshape(-1).tolist()))")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            output = subprocess.check_output([sys.executable, "-c", code, str(ROOT / "notebooks/utility"),
                                              str(ROOT), str(checkpoint)], text=True, env=env)
            actual = json.loads(next(line[5:] for line in output.splitlines() if line.startswith("PRED=")))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
