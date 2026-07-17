"""Model-free regression tests for classifier checkpoint/resume boundaries."""
from __future__ import annotations

import sys
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import classifier_architecture_adapters as adapters  # noqa: E402
import classifier_checkpoint_io as checkpoint_io  # noqa: E402
import classifier_experiment as experiment  # noqa: E402


class ClassifierResumeBoundaryTests(unittest.TestCase):
    def test_classifier_notebooks_require_no_terminal_environment_variables(self):
        for name in ("01_MaxViT512.ipynb", "02_MammoFM.ipynb"):
            text = (ROOT / "notebooks/04_classifiers" / name).read_text()
            self.assertNotIn("MAMMODIFFUSION_GPU", text)
            self.assertNotIn("MAMMOFM_LOCAL_CHECKPOINT_PATH", text)
            self.assertNotIn("GPU_SELECTOR", text)
            self.assertNotIn("GPU-82ec33a5-8b4f-40d0-ead7-8d1d9679055d", text)

    def test_classifier_training_is_explicit_and_compilable_in_notebooks(self):
        builders = {
            "01_MaxViT512.ipynb": "build_maxvit_model(",
            "02_MammoFM.ipynb": "build_mammofm_model(",
        }
        required = (
            "TRAINING_VERBOSE = True",
            "PROGRESS_EVERY_UPDATES = 25",
            "VALIDATION_PROGRESS_EVERY_BATCHES = 10",
            "torch.optim.AdamW(",
            "BinaryFocalLoss()",
            "for epoch in range(",
            "for batch_index, batch in enumerate(train_loader):",
            "loss.backward()",
            "optimizer.step()",
            "with torch.no_grad():",
            "scheduler.step(",
            "save_training_checkpoint(",
            "checkpoint_best.pt",
            "readable_duration(",
            "ETA=",
            "report_validation",
        )
        for name, builder in builders.items():
            notebook = json.loads(
                (ROOT / "notebooks/04_classifiers" / name).read_text()
            )
            code = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
                if cell.get("cell_type") == "code"
            )
            self.assertIn(builder, code)
            for token in required:
                self.assertIn(token, code)
            self.assertNotIn("training_result = train(", code)
            self.assertNotIn("fit_mammofm(", code)
            for index, cell in enumerate(notebook["cells"]):
                if cell.get("cell_type") == "code":
                    compile(
                        "".join(cell.get("source", [])),
                        f"{name}:cell:{index}",
                        "exec",
                    )

    def test_classifier_notebooks_cycle_all_conditions_and_seeds(self):
        for name in ("01_MaxViT512.ipynb", "02_MammoFM.ipynb"):
            notebook = json.loads(
                (ROOT / "notebooks/04_classifiers" / name).read_text()
            )
            code = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
                if cell.get("cell_type") == "code"
            )
            self.assertIn("CONDITIONS = (", code)
            for condition in (
                "real_only",
                "real_augmented",
                "real_plus_best_finetuned_positive",
                "real_plus_best_fromscratch_positive",
            ):
                self.assertIn(f"'{condition}'", code)
            self.assertIn("SEEDS = (17, 42, 73)", code)
            self.assertIn(
                "def train_one_job(condition, seed, configuration, dataset):", code
            )
            self.assertIn("for condition in CONDITIONS:", code)
            self.assertIn("for seed in SEEDS:", code)
            self.assertIn("configurations[condition][seed]", code)
            self.assertIn("datasets[condition][seed]", code)
            self.assertIn("validation_results[condition][seed]", code)
            self.assertIn("checkpoint_io.load_resume_checkpoint(", code)
            self.assertNotIn("SEED = 17", code)
            self.assertNotIn("CONDITION = 'real_only'", code)

    def test_terminal_resume_is_finalized_without_another_epoch(self):
        limits = {
            "max_optimizer_updates": 6400,
            "max_epochs": 26,
            "early_stopping_patience": 10,
        }
        terminal = {
            "epoch": 24,
            "global_step": 5750,
            "early_stopping_counter": 10,
        }
        interrupted = {**terminal, "early_stopping_counter": 9}

        self.assertEqual(
            checkpoint_io.terminal_reason(terminal, limits), "early_stopping"
        )
        self.assertIsNone(checkpoint_io.terminal_reason(interrupted, limits))

        for name in ("01_MaxViT512.ipynb", "02_MammoFM.ipynb"):
            notebook = json.loads(
                (ROOT / "notebooks/04_classifiers" / name).read_text()
            )
            code = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
                if cell.get("cell_type") == "code"
            )
            terminal_check = code.index(
                "completion_reason = checkpoint_io.terminal_reason("
            )
            training_loop = code.index("for epoch in range(start_epoch, epochs + 1):")
            self.assertLess(terminal_check, training_loop)
            self.assertIn("checkpoint_io.inspect_completed_run(", code)
            self.assertIn("SKIP completed run", code)
            self.assertIn("checkpoint_io.completion_path(", code)
            self.assertIn("torch.default_generator.manual_seed(seed)", code)
            self.assertIn("torch.cuda.manual_seed(seed)", code)
            self.assertNotIn("torch.manual_seed(seed)", code)
            self.assertNotIn("torch.cuda.manual_seed_all(seed)", code)

    def test_finalized_pre_marker_run_is_recognized(self):
        expected = {
            "architecture": "maxvit512",
            "experiment_id": "maxvit512__real_only__seed17",
            "dataset_variant_id": "real_only",
            "training_policy": "maxvit512_fixed_protocol",
            "config_signature": "policy-signature",
            "dataset_signature": "dataset-signature",
            "seed": 17,
        }
        limits = {
            "max_optimizer_updates": 6400,
            "max_epochs": 26,
            "early_stopping_patience": 10,
        }
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for name in checkpoint_io.FINAL_ARTIFACTS:
                (run / name).write_text("placeholder", encoding="utf-8")
            (run / "configuration.json").write_text(
                json.dumps(
                    {
                        "architecture": "maxvit512",
                        "condition": "real_only",
                        "seed": 17,
                        "policy_signature": "policy-signature",
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_io.save_resume_checkpoint(
                run,
                {
                    **expected,
                    "epoch": 24,
                    "global_step": 5750,
                    "early_stopping_counter": 10,
                    "best_epoch": 13,
                    "gpu_uuid": "GPU-test",
                },
            )

            completion, source = checkpoint_io.inspect_completed_run(
                run, expected, limits
            )

            self.assertEqual(source, "recovered_from_checkpoint_latest")
            self.assertEqual(completion["status"], "complete")
            self.assertEqual(completion["completion_reason"], "early_stopping")
            self.assertEqual(completion["optimizer_updates_completed"], 5750)

    def test_no_hidden_high_level_training_entry_point_remains(self):
        self.assertFalse(hasattr(experiment, "train"))
        adapter = adapters.ArchitectureAdapter("maxvit512", {}, ROOT)
        self.assertFalse(hasattr(adapter, "train"))

    def test_mammofm_adapter_uses_official_local_cache_without_environment(self):
        build_model = mock.Mock(return_value=("model",))
        fake_utils = SimpleNamespace(
            DEFAULT_HF_REPO="batmanLab/Mammo-FM",
            DEFAULT_CHECKPOINT_NAME="Mammo-FM_BatmanlabTrained_CLIP.tar",
            build_mammofm_model=build_model,
        )
        with mock.patch.dict(sys.modules, {"mammofm_utils": fake_utils}):
            model = adapters.ArchitectureAdapter("mammofm", {}, ROOT).build_model()

        self.assertEqual(model, "model")
        build_model.assert_called_once_with(
            hf_repo="batmanLab/Mammo-FM",
            checkpoint_name="Mammo-FM_BatmanlabTrained_CLIP.tar",
            use_local_checkpoint=False,
            local_files_only=True,
        )

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
        for name in ("01_MaxViT512.ipynb", "02_MammoFM.ipynb"):
            notebook = json.loads((ROOT / "notebooks/04_classifiers" / name).read_text())
            validation_cells = [
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
                if cell.get("cell_type") == "code" and "run_validation(" in "".join(cell.get("source", []))
            ]
            self.assertEqual(len(validation_cells), 1)
            source = validation_cells[0]
            self.assertIn(
                "seed: load_existing_outputs(ROOT, configurations[condition][seed])",
                source,
            )
            self.assertIn(
                "seed: existing_outputs_by_job[condition][seed]['checkpoint']",
                source,
            )
            self.assertIn("missing_checkpoint_jobs", source)
            self.assertIn("No trained checkpoint is available for jobs", source)
            self.assertNotIn("training_result", source)


if __name__ == "__main__":
    unittest.main()
