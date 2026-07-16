"""Model-free regression tests for classifier checkpoint/resume boundaries."""
from __future__ import annotations

import sys
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import classifier_architecture_adapters as adapters  # noqa: E402
import classifier_checkpoint_io as checkpoint_io  # noqa: E402
import downstream_experiment as experiment  # noqa: E402


class ClassifierResumeBoundaryTests(unittest.TestCase):
    def test_periodic_checkpoint_resume_does_not_repeat_recorded_block(self):
        """A resumable checkpoint represents a completed validation block."""
        block_rows = [
            {"accounting_field": "real_negative_seen"},
            {"accounting_field": "real_positive_seen"},
            {"accounting_field": "finetuned_synthetic_seen"},
        ]
        periodic_checkpoint = {
            "epoch": 1,
            "batch_index": len(block_rows) - 1,
            "global_step": len(block_rows),
            "source_accounting": {
                "accounting_mode": "actual",
                "real_negative_seen": 1,
                "real_positive_seen": 1,
                "traditional_augmented_seen": 0,
                "finetuned_synthetic_seen": 1,
                "fromscratch_synthetic_seen": 0,
            },
        }

        with self.assertRaisesRegex(RuntimeError, "validated block boundary"):
            adapters._pytorch_resume_position(periodic_checkpoint)

        checkpoint = {**periodic_checkpoint, "epoch": 2, "batch_index": -1}
        resume_epoch, resume_batch = adapters._pytorch_resume_position(checkpoint)
        replayed = block_rows[resume_batch:] if resume_epoch == 1 else []

        self.assertEqual((resume_epoch, resume_batch), (2, 0))
        self.assertEqual(replayed, [])
        self.assertEqual(checkpoint["global_step"], 3)
        self.assertEqual(
            sum(checkpoint["source_accounting"][field] for field in adapters.ACCOUNTING_FIELDS),
            3,
        )

    def test_real_policy_signature_controls_resume_compatibility(self):
        configuration = experiment.experiment_configuration(
            ROOT, "maxvit512", "real_only", 17
        )
        policy = configuration["policy"]

        def signature(value):
            canonical = json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        self.assertEqual(configuration["policy_signature"], signature(policy))
        expected = {
            "architecture": "maxvit512",
            "experiment_id": configuration["experiment_id"],
            "dataset_variant_id": "real_only",
            "training_policy": configuration["training_policy_name"],
            "config_signature": configuration["policy_signature"],
            "dataset_signature": "dataset-signature",
            "seed": 17,
        }
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            checkpoint_io.save_resume_checkpoint(run, {**expected, "epoch": 2, "global_step": 250})
            payload, _ = checkpoint_io.load_resume_checkpoint(run, expected)
            self.assertIsNotNone(payload)

            mutations = {
                "learning rate": lambda p: p["training_phases"][0].__setitem__("learning_rate", 9e-4),
                "batch size": lambda p: p.__setitem__("physical_batch_size", 4),
                "gradient accumulation": lambda p: p.__setitem__("gradient_accumulation_steps", 4),
                "max optimizer updates": lambda p: p.__setitem__("max_optimizer_updates", 3200),
            }
            for name, mutate in mutations.items():
                changed = copy.deepcopy(policy)
                mutate(changed)
                incompatible = {**expected, "config_signature": signature(changed)}
                with self.subTest(name=name):
                    payload, reason = checkpoint_io.load_resume_checkpoint(run, incompatible)
                    self.assertIsNone(payload)
                    self.assertIn("config_signature", reason)

    def test_validation_cells_reload_checkpoint_after_kernel_restart(self):
        for name in ("07_MaxViT512_Downstream.ipynb", "08_MammoFM_Downstream.ipynb"):
            notebook = json.loads((ROOT / "notebooks/04_classifiers" / name).read_text())
            validation_cells = [
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
                if cell.get("cell_type") == "code" and "run_validation(" in "".join(cell.get("source", []))
            ]
            self.assertEqual(len(validation_cells), 1)
            source = validation_cells[0]
            self.assertIn("existing_outputs = load_existing_outputs(ROOT, configuration)", source)
            self.assertIn('checkpoint = existing_outputs["checkpoint"]', source)
            self.assertIn('raise RuntimeError("No trained checkpoint is available.")', source)
            self.assertNotIn("training_result", source)


if __name__ == "__main__":
    unittest.main()
