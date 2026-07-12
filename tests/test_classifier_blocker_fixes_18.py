"""Regression tests for prompt_sol_ultimi_blocker_18.md, one class per numbered blocker.

Fast/unit-level where the bug is reachable without real pretrained weights or GPU; real
adapter-level resume/best-state coverage lives in test_classifier_resume_integration.py.
Nothing here starts Stage 1 or opens the locked test.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import classifier_checkpoint_io as ckio  # noqa: E402


class FitMammofmSignatureTests(unittest.TestCase):
    """Blocker 2: fit_mammofm must accept on_before_optimizer_step (and on_epoch_begin)."""

    def test_fit_mammofm_accepts_and_invokes_all_callbacks(self):
        import torch
        import mammofm_utils as amp_utils

        torch.manual_seed(0)
        model = torch.nn.Linear(4, 1)
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,)).float()

        class TinyLoader:
            def __iter__(self):
                yield x[:4], y[:4]
                yield x[4:], y[4:]

            def __len__(self):
                return 2

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        calls = {"before_opt": [], "opt": [], "epoch_begin": [], "epoch_end": []}

        class WrappedModel(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner(x)

        wrapped = WrappedModel(model)

        history = amp_utils.fit_mammofm(
            wrapped, TinyLoader(), TinyLoader(), optimizer, criterion, epochs=1, device=torch.device("cpu"),
            use_amp=False,
            on_before_optimizer_step=lambda step, batch: calls["before_opt"].append((step, batch)),
            on_optimizer_step=lambda step, batch: calls["opt"].append((step, batch)),
            on_epoch_begin=lambda epoch: calls["epoch_begin"].append(epoch),
            on_epoch_end=lambda epoch, step, scaler, hist, metrics, improved: calls["epoch_end"].append(epoch),
        )
        # This call must not raise TypeError (the exact reported failure): all 4 callbacks
        # were real, invocable keyword arguments, and each fired at least once.
        self.assertTrue(calls["before_opt"])
        self.assertTrue(calls["opt"])
        self.assertEqual(calls["epoch_begin"], [1])
        self.assertEqual(calls["epoch_end"], [1])
        self.assertIn("loss", history.history)


class CorruptedResumeTests(unittest.TestCase):
    """Blocker 5: a resume file present but entirely corrupted/incompatible must never lead
    to a silent from-scratch restart.
    """

    def _corrupt_dir(self, root: Path) -> Path:
        run = root / "run"
        run.mkdir(parents=True)
        # Two ways to be "present but unusable": literally corrupt bytes, and well-formed but
        # scientifically incompatible (wrong dataset/config signature).
        (run / "checkpoint_latest.pkl").write_bytes(b"not a pickle at all")
        with (run / "checkpoint_previous.pkl").open("wb") as fh:
            pickle.dump({"schema_version": 2, "architecture": "other-architecture", "seed": 999}, fh)
        return run

    def test_ckio_reports_all_invalid_distinctly_from_no_file(self):
        with tempfile.TemporaryDirectory() as t:
            run = self._corrupt_dir(Path(t))
            expected = {"architecture": "maxvit512", "experiment_id": "x", "dataset_variant_id": "d",
                        "training_policy": "p", "config_signature": "c", "dataset_signature": "s", "seed": 17}
            resume, source = ckio.load_resume_checkpoint(run, expected)
            self.assertIsNone(resume)
            self.assertNotEqual(source, "no resume checkpoint")

    def test_no_file_at_all_reports_the_distinct_no_checkpoint_reason(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / "run"; run.mkdir(parents=True)
            expected = {"architecture": "maxvit512", "seed": 17}
            resume, source = ckio.load_resume_checkpoint(run, expected)
            self.assertIsNone(resume)
            self.assertEqual(source, "no resume checkpoint")

    def test_adapter_train_refuses_to_silently_discard_invalid_resume(self):
        import classifier_architecture_adapters as caa
        with tempfile.TemporaryDirectory() as t:
            run = self._corrupt_dir(Path(t))
            policy = {"input_size": [64, 64]}  # never reached: the guard raises first
            adapter = caa.ArchitectureAdapter("mammofm", policy, ROOT)
            context = {"run_dir": run, "architecture": "mammofm", "experiment_id": "x",
                       "dataset_variant_id": "d", "training_policy": "p",
                       "config_signature": "c", "dataset_signature": "s"}
            os.environ.pop("ALLOW_DISCARD_INVALID_RESUME", None)
            with self.assertRaises(RuntimeError) as ctx:
                adapter.train([], [], run / "final.pt", seed=17, **context)
            self.assertIn("invalid/incompatible", str(ctx.exception))

    def test_adapter_train_proceeds_when_explicitly_allowed_to_discard(self):
        import classifier_architecture_adapters as caa
        with tempfile.TemporaryDirectory() as t:
            run = self._corrupt_dir(Path(t))
            policy = {"input_size": [64, 64]}
            adapter = caa.ArchitectureAdapter("mammofm", policy, ROOT)
            context = {"run_dir": run, "architecture": "mammofm", "experiment_id": "x",
                       "dataset_variant_id": "d", "training_policy": "p",
                       "config_signature": "c", "dataset_signature": "s"}
            os.environ["ALLOW_DISCARD_INVALID_RESUME"] = "True"
            try:
                # Now it should get *past* the corrupted-resume guard and fail for the next
                # unrelated reason instead (no MAMMOFM_LOCAL_CHECKPOINT_PATH set here) - proving
                # the guard, not something else, was what previously stopped it.
                with self.assertRaises(RuntimeError) as ctx:
                    adapter.train([], [], run / "final.pt", seed=17, **context)
                self.assertNotIn("invalid/incompatible", str(ctx.exception))
            finally:
                os.environ.pop("ALLOW_DISCARD_INVALID_RESUME", None)


class OomStateRestartTests(unittest.TestCase):
    """Blocker 6: OOM retry/exclusive state must survive a scheduler process restart."""

    def test_fresh_scheduler_process_resumes_oom_count_from_disk(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_classifier_experiment_matrix as rcem
        from classifier_gpu_scheduler import OomState

        with tempfile.TemporaryDirectory() as t:
            run_dir = Path(t) / "run"; run_dir.mkdir(parents=True)
            protocol = {"physical_batch_size": 16, "gradient_accumulation_steps": 1, "effective_batch_size": 16}
            job = {}

            # "Session 1": a real OomState experiences one OOM and persists it to disk, exactly
            # as the scheduler's own on_oom handling does.
            first = rcem.load_or_create_oom_state(run_dir, job, protocol)
            first.record_oom()
            (run_dir / "oom_override.json").write_text(json.dumps({
                "physical_batch_size": first.physical_batch_size,
                "gradient_accumulation_steps": first.gradient_accumulation_steps,
                "effective_batch_size": first.effective_batch_size,
                "oom_count": first.oom_count, "forced_exclusive": first.forced_exclusive,
                "history": first.history,
            }))

            # "Session 2": brand new process, brand new empty oom_states dict (simulated here
            # by simply not reusing `first`) - must pick up oom_count=1 from disk, not restart
            # at 0, so a *second* OOM in this new session correctly forces exclusive rather
            # than being treated as the first.
            second = rcem.load_or_create_oom_state(run_dir, job, protocol)
            self.assertEqual(second.oom_count, 1)
            self.assertEqual(second.physical_batch_size, first.physical_batch_size)
            second.record_oom()
            self.assertTrue(second.forced_exclusive)
            self.assertEqual(second.oom_count, 2)
            # A *third* OOM (still within this "session 2", but exercising the same persisted
            # continuity) must finally exhaust retries -- proving the count genuinely carried
            # over rather than having been silently reset to 0 or 1 by the restart.
            second.record_oom()
            self.assertFalse(second.should_retry())

    def test_no_override_file_yields_fresh_state_from_policy_defaults(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_classifier_experiment_matrix as rcem

        with tempfile.TemporaryDirectory() as t:
            run_dir = Path(t) / "run"; run_dir.mkdir(parents=True)
            protocol = {"physical_batch_size": 8, "gradient_accumulation_steps": 2, "effective_batch_size": 16}
            state = rcem.load_or_create_oom_state(run_dir, {}, protocol)
            self.assertEqual(state.oom_count, 0)
            self.assertEqual(state.physical_batch_size, 8)

    def test_corrupt_override_file_falls_back_to_fresh_state(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_classifier_experiment_matrix as rcem

        with tempfile.TemporaryDirectory() as t:
            run_dir = Path(t) / "run"; run_dir.mkdir(parents=True)
            (run_dir / "oom_override.json").write_text("{not valid json")
            protocol = {"physical_batch_size": 8, "gradient_accumulation_steps": 2, "effective_batch_size": 16}
            state = rcem.load_or_create_oom_state(run_dir, {}, protocol)
            self.assertEqual(state.oom_count, 0)


class ThreadIsolationTests(unittest.TestCase):
    """Blocker 6 (CPU/RAM): launched workers must not oversubscribe host threads."""

    def test_launch_job_sets_thread_limit_env_vars(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_classifier_experiment_matrix as rcem
        from unittest.mock import patch, MagicMock

        captured = {}

        def fake_popen(cmd, cwd=None, env=None):
            captured["env"] = env
            return MagicMock(poll=lambda: 0, returncode=0)

        with patch("subprocess.Popen", side_effect=fake_popen):
            rcem.launch_job(ROOT, {"experiment_id": "x"}, gpu_index=0)
        for key, expected in rcem.THREAD_ENV_VARS.items():
            self.assertEqual(captured["env"].get(key), expected)


class ReportingTests(unittest.TestCase):
    """Blocker 7: aggregated curves/ROC-PR/error-examples must use real multi-seed data."""

    def _write_seed_history(self, base, seed, n=5, offset=0.0):
        directory = base / f"seed_{seed}"; directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"training_history_seed_{seed}.csv"
        import csv as _csv
        with path.open("w", newline="") as fh:
            writer = _csv.DictWriter(fh, fieldnames=["loss", "val_loss", "val_auc", "lr"])
            writer.writeheader()
            for i in range(n):
                writer.writerow({"loss": 1.0 - i * 0.1 + offset, "val_loss": 0.9 - i * 0.1 + offset,
                                  "val_auc": 0.5 + i * 0.05, "lr": 0.001})

    def _write_seed_predictions(self, base, seed, image_path):
        directory = base / f"seed_{seed}"; directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"validation_predictions_seed_{seed}.csv"
        rows = [
            {"patient_id": "p1", "image_id": "i1", "label": 1, "probability": 0.9, "processed_path": str(image_path)},
            {"patient_id": "p2", "image_id": "i2", "label": 0, "probability": 0.1, "processed_path": str(image_path)},
            {"patient_id": "p3", "image_id": "i3", "label": 1, "probability": 0.3, "processed_path": str(image_path)},
            {"patient_id": "p4", "image_id": "i4", "label": 0, "probability": 0.8, "processed_path": str(image_path)},
        ]
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "notebooks/utility"))
        import classifier_reporting as cr
        cr.write_csv(path, rows, fields=["patient_id", "image_id", "label", "probability", "processed_path"])
        return rows

    def test_training_curves_all_seeds_is_a_real_multi_series_plot_not_text(self):
        import classifier_reporting as cr
        with tempfile.TemporaryDirectory() as t:
            base = Path(t)
            self._write_seed_history(base, 17, offset=0.0)
            self._write_seed_history(base, 42, offset=0.1)
            out = cr.render_training_curves_all_seeds(base)
            self.assertIsNotNone(out)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)

    def test_training_curves_all_seeds_none_when_no_data(self):
        import classifier_reporting as cr
        with tempfile.TemporaryDirectory() as t:
            self.assertIsNone(cr.render_training_curves_all_seeds(Path(t)))

    def test_validation_figures_include_every_seed_present_plus_ensemble(self):
        import classifier_reporting as cr
        from PIL import Image
        with tempfile.TemporaryDirectory() as t:
            base = Path(t)
            image_path = base / "sample.png"
            Image.new("L", (32, 32), color=128).save(image_path)
            self._write_seed_predictions(base, 17, image_path)
            self._write_seed_predictions(base, 42, image_path)
            ensemble_rows = [
                {"patient_id": "p1", "image_id": "i1", "label": 1, "probability": 0.85, "processed_path": str(image_path)},
                {"patient_id": "p2", "image_id": "i2", "label": 0, "probability": 0.15, "processed_path": str(image_path)},
                {"patient_id": "p3", "image_id": "i3", "label": 1, "probability": 0.4, "processed_path": str(image_path)},
                {"patient_id": "p4", "image_id": "i4", "label": 0, "probability": 0.75, "processed_path": str(image_path)},
            ]
            (base / "ensemble/predictions").mkdir(parents=True)
            (base / "ensemble/metrics").mkdir(parents=True)
            cr.write_csv(base / "ensemble/predictions/ensemble_validation_predictions.csv", ensemble_rows,
                         fields=["patient_id", "image_id", "label", "probability", "processed_path"])
            cr.atomic_json(base / "ensemble/metrics/ensemble_validation_metrics.json", {"threshold": 0.5})

            made = cr.render_validation_figures(base, seeds=(17, 42, 73))
            names = {p.name for p in made}
            self.assertIn("roc_curve_all_seeds_and_ensemble.png", names)
            self.assertIn("pr_curve_all_seeds_and_ensemble.png", names)
            self.assertIn("validation_error_examples.png", names)
            error_csv = base / "ensemble/validation_error_cases.csv"
            self.assertTrue(error_csv.is_file())
            self.assertIn("processed_path", error_csv.read_text())

    def test_error_examples_figure_uses_real_image_not_text_only(self):
        # Regression guard: the figure file must exist and be a real (non-trivial) PNG once
        # at least one case has a resolvable processed_path -- not merely a monospace text dump.
        import classifier_reporting as cr
        from PIL import Image
        with tempfile.TemporaryDirectory() as t:
            base = Path(t)
            image_path = base / "sample.png"
            Image.new("L", (32, 32), color=200).save(image_path)
            ensemble_rows = [
                {"patient_id": "p1", "image_id": "i1", "label": 1, "probability": 0.9, "processed_path": str(image_path)},
                {"patient_id": "p2", "image_id": "i2", "label": 0, "probability": 0.05, "processed_path": str(image_path)},
            ]
            (base / "ensemble/predictions").mkdir(parents=True)
            (base / "ensemble/metrics").mkdir(parents=True)
            cr.write_csv(base / "ensemble/predictions/ensemble_validation_predictions.csv", ensemble_rows,
                         fields=["patient_id", "image_id", "label", "probability", "processed_path"])
            cr.atomic_json(base / "ensemble/metrics/ensemble_validation_metrics.json", {"threshold": 0.5})
            cr.render_validation_figures(base, seeds=())
            out = base / "ensemble/figures/validation_error_examples.png"
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 2000)  # a real rendered image grid, not a tiny text-only stub


class InterpretabilityTests(unittest.TestCase):
    """Blocker 8: real/synthetic sample limits are independent; fallback selection persists once."""

    def test_persist_fallback_selection_writes_manifest_once(self):
        import classifier_interpretability as ci
        with tempfile.TemporaryDirectory() as t:
            shared_path = Path(t) / "configs/interpretability_validation_samples.json"
            selected = [{"patient_id": "p2", "image_id": "i2"}, {"patient_id": "p1", "image_id": "i1"}]
            ci._persist_fallback_selection(shared_path, selected)
            self.assertTrue(shared_path.is_file())
            payload = json.loads(shared_path.read_text())
            self.assertEqual(len(payload["samples"]), 2)
            self.assertIn("policy", payload)

    def test_generate_configuration_attributions_signature_accepts_independent_limits(self):
        import inspect
        import classifier_interpretability as ci
        sig = inspect.signature(ci.generate_configuration_attributions)
        self.assertIn("real_limit", sig.parameters)
        self.assertIn("synthetic_limit", sig.parameters)
        self.assertNotIn("limit", sig.parameters)  # old single-limit name must be gone, not just aliased

    def test_display_new_attributions_tolerates_missing_ipython_and_missing_files(self):
        import classifier_interpretability as ci
        with tempfile.TemporaryDirectory() as t:
            # Must not raise even for nonexistent paths (headless/CLI/test environment).
            ci.display_new_attributions(["nonexistent/path.png"], Path(t))

    def test_generator_script_passes_both_limits_and_correct_parameter_names(self):
        script = (ROOT / "scripts/create_classifier_matrix_notebooks.py").read_text()
        self.assertIn("real_limit=GRADCAM_NUM_REAL_SAMPLES", script)
        self.assertIn("synthetic_limit=GRADCAM_NUM_SYNTHETIC_SAMPLES", script)
        self.assertNotIn("limit=GRADCAM_NUM_REAL_SAMPLES)", script)  # the old, single-limit call


class AugmentationProvenanceTests(unittest.TestCase):
    """Blocker 9: augmented images must carry real provenance and reject leakage/unknowns."""

    def _project(self, root: Path, *, augmented_rows, train_rows=None):
        (root / "data/processed/metadata").mkdir(parents=True)
        train_rows = train_rows if train_rows is not None else [
            {"patient_id": "P1", "image_id": "I1", "label": "1"},
            {"patient_id": "P2", "image_id": "I2", "label": "0"},
        ]
        with (root / "data/processed/metadata/train.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["patient_id", "image_id", "label"]); w.writeheader(); w.writerows(train_rows)
        (root / "data/real_augmented").mkdir(parents=True, exist_ok=True)
        with (root / "data/real_augmented/metadata.csv").open("w", newline="") as fh:
            fields = ["file_name", "label", "patient_id", "image_id", "source", "original_processed_path"]
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(augmented_rows)

    def test_real_provenance_is_loaded_for_valid_augmented_row(self):
        import classifier_dataset_builder as cdb
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self._project(root, augmented_rows=[
                {"file_name": "data/real_augmented/a0.png", "label": "1", "patient_id": "P1", "image_id": "I1",
                 "source": "positive_augmentation", "original_processed_path": "/x/data/processed/train/1/P1_I1.png"},
            ])
            result = cdb._augmented_files_by_class(root)
            self.assertEqual(len(result["positive"]), 1)
            self.assertEqual(result["positive"][0]["patient_id"], "P1")
            self.assertEqual(result["positive"][0]["source_split"], "train")

    def test_real_source_rows_in_metadata_are_not_double_counted_as_augmented(self):
        import classifier_dataset_builder as cdb
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self._project(root, augmented_rows=[
                {"file_name": "data/processed/train/1/P1_I1.png", "label": "1", "patient_id": "P1", "image_id": "I1",
                 "source": "real", "original_processed_path": "/x/data/processed/train/1/P1_I1.png"},
                {"file_name": "data/real_augmented/a0.png", "label": "1", "patient_id": "P1", "image_id": "I1",
                 "source": "positive_augmentation", "original_processed_path": "/x/data/processed/train/1/P1_I1.png"},
            ])
            result = cdb._augmented_files_by_class(root)
            self.assertEqual(len(result["positive"]), 1)  # only the augmented row, not the "real" one

    def test_rejects_augmented_row_with_no_source_patient(self):
        import classifier_dataset_builder as cdb
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self._project(root, augmented_rows=[
                {"file_name": "data/real_augmented/orphan.png", "label": "1", "patient_id": "",
                 "image_id": "", "source": "positive_augmentation", "original_processed_path": ""},
            ])
            with self.assertRaises(cdb.AugmentationProvenanceError):
                cdb._augmented_files_by_class(root)

    def test_rejects_augmented_row_sourced_from_validation_or_test(self):
        import classifier_dataset_builder as cdb
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self._project(root, augmented_rows=[
                {"file_name": "data/real_augmented/leak.png", "label": "1", "patient_id": "P1", "image_id": "I1",
                 "source": "positive_augmentation", "original_processed_path": "/x/data/processed/val/1/P1_I1.png"},
            ])
            with self.assertRaises(cdb.AugmentationProvenanceError):
                cdb._augmented_files_by_class(root)

    def test_rejects_augmented_row_whose_patient_is_not_in_train_csv(self):
        import classifier_dataset_builder as cdb
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self._project(root, augmented_rows=[
                {"file_name": "data/real_augmented/unknown.png", "label": "1", "patient_id": "P999", "image_id": "I999",
                 "source": "positive_augmentation", "original_processed_path": "/x/data/processed/train/1/P999_I999.png"},
            ])
            with self.assertRaises(cdb.AugmentationProvenanceError):
                cdb._augmented_files_by_class(root)

    def test_missing_metadata_csv_yields_empty_not_an_unprovenanced_fallback(self):
        import classifier_dataset_builder as cdb
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "data/processed/metadata").mkdir(parents=True)
            (root / "data/processed/metadata/train.csv").write_text("patient_id,image_id,label\n")
            result = cdb._augmented_files_by_class(root)
            self.assertEqual(result, {"negative": [], "positive": []})

    def test_build_file_list_propagates_patient_id_for_leakage_checks(self):
        import classifier_dataset_builder as cdb
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self._project(root, augmented_rows=[
                {"file_name": "data/real_augmented/a0.png", "label": "1", "patient_id": "P1", "image_id": "I1",
                 "source": "positive_augmentation", "original_processed_path": "/x/data/processed/train/1/P1_I1.png"},
            ])
            (root / "data/real_augmented/a0.png").write_bytes(b"png")
            variant = {"real_source": None, "augmentation_source": "data/real_augmented",
                       "synthetic_generator_id": None, "synthetic_count_by_class": {}, "dataset_variant_id": "TEST"}
            files = cdb.build_file_list(root, variant)
            self.assertEqual(files["positive"][0]["patient_id"], "P1")
            rows = cdb.rows_from_file_list(root, files)
            self.assertEqual(rows[0]["patient_id"], "P1")


class LockedPanelDedupTests(unittest.TestCase):
    """Blocker 10: the secondary panel must store one logical ensemble id per configuration,
    and locked_matrix_inference.run_locked() must dedupe by logical stem regardless.
    """

    def _build_matrix(self, root, architecture="maxvit512", variant="R"):
        (root / "configs").mkdir(parents=True, exist_ok=True)
        jobs = [{"experiment_id": f"{architecture}__{variant}__seed{seed}", "stage": 2,
                 "architecture": architecture, "dataset_variant_id": variant,
                 "training_policy": f"{architecture}_standard", "seed": seed,
                 "manifest_path": f"experiments/classifiers_matrix/{architecture}/{variant}/p/seed_{seed}/run_manifest.json",
                 "checkpoint_path": f"experiments/classifiers_matrix/{architecture}/{variant}/p/seed_{seed}/model.pt",
                 "status": "COMPLETE"} for seed in (17, 42, 73)]
        (root / "configs/classifier_experiment_matrix.json").write_text(json.dumps({"schema_version": 1, "jobs": jobs}))
        (root / "configs/classifier_training_protocols.json").write_text(json.dumps({"policies": {
            architecture: {"framework": "pytorch_timm"}}}))
        for job in jobs:
            ckpt = root / job["checkpoint_path"]; ckpt.parent.mkdir(parents=True, exist_ok=True); ckpt.write_bytes(b"weights")
        base = root / "results/classifiers_matrix" / architecture / variant / f"{architecture}_standard" / "ensemble"
        (base / "metrics").mkdir(parents=True, exist_ok=True)
        (base / "metrics/locked_validation_threshold.json").write_text(json.dumps({"threshold": 0.5}))
        return jobs

    def test_finalize_stage2_panels_never_flattens_seeds_into_panel(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import finalize_validation_stage as fvs
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self._build_matrix(root)
            base = root / "results/classifiers_matrix/maxvit512/R/maxvit512_standard/ensemble/manifests"
            base.mkdir(parents=True, exist_ok=True)
            (base / "ensemble_validation_manifest.json").write_text(json.dumps({
                "architecture": "maxvit512", "dataset_variant_id": "R", "seeds": [17, 42, 73],
                "test_access": False, "metrics": {"pr_auc": 0.9, "roc_auc": 0.9}, "signature": "sig",
            }))
            payload = fvs.finalize_stage2_panels(root)
            panel = payload["secondary_locked_panel"]
            self.assertEqual(panel, ["maxvit512__R__ensemble"])  # one logical id, not 3 seed ids
            self.assertNotIn("maxvit512__R__seed17", panel)

    def test_run_locked_infers_each_configuration_exactly_once(self):
        sys.path.insert(0, str(ROOT / "notebooks/utility"))
        import locked_matrix_inference as lmi
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self._build_matrix(root)
            (root / "data/processed/metadata").mkdir(parents=True, exist_ok=True)
            (root / "data/processed/metadata/test.csv").write_text(
                "patient_id,image_id,label,processed_path\np1,i1,0,data/processed/test/0/p1_i1.png\n")
            (root / "data/processed/test/0").mkdir(parents=True, exist_ok=True)
            (root / "data/processed/test/0/p1_i1.png").write_bytes(b"png")

            lock_dir = root / "results/final_evaluation_v2"; lock_dir.mkdir(parents=True)
            (lock_dir / "EXPERIMENT_MATRIX_LOCKED").write_text("{}")
            # Deliberately list the SAME configuration three times, once per raw seed id, the
            # way the pre-fix secondary_locked_panel used to - the runner must still only infer
            # and write it once.
            (lock_dir / "primary_panel_manifest.json").write_text(json.dumps({"experiment_ids": []}))
            (lock_dir / "secondary_panel_manifest.json").write_text(json.dumps({"experiment_ids": [
                "maxvit512__R__seed17", "maxvit512__R__seed42", "maxvit512__R__seed73"]}))
            (lock_dir / "ablation_panel_manifest.json").write_text(json.dumps({"experiment_ids": []}))

            calls = []

            def fake_predictor(job, checkpoint, test_rows):
                calls.append(job["experiment_id"])
                return [0.9 for _ in test_rows]

            import finalize_locked_test_stage as lock_mod
            original_verify = lock_mod.verify_lock_still_valid
            lock_mod.verify_lock_still_valid = lambda root: (True, [])
            try:
                manifest = lmi.run_locked(root, predictor_fn=fake_predictor)
            finally:
                lock_mod.verify_lock_still_valid = original_verify

            self.assertEqual(len(manifest["outputs"]), 1)  # not 3
            self.assertEqual(len(calls), 3)  # one predict call per seed of the *one* ensemble
            predictions_dir = lock_dir / "predictions/secondary"
            csvs = list(predictions_dir.glob("*.csv"))
            self.assertEqual(len(csvs), 1)
            self.assertEqual(csvs[0].stem, "maxvit512__R__ensemble")

    def test_run_locked_resolves_relative_test_paths_under_root_not_cwd(self):
        sys.path.insert(0, str(ROOT / "notebooks/utility"))
        import locked_matrix_inference as lmi
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self._build_matrix(root)
            (root / "data/processed/metadata").mkdir(parents=True, exist_ok=True)
            (root / "data/processed/metadata/test.csv").write_text(
                "patient_id,image_id,label,processed_path\np1,i1,0,data/processed/test/0/p1_i1.png\n")
            (root / "data/processed/test/0").mkdir(parents=True, exist_ok=True)
            (root / "data/processed/test/0/p1_i1.png").write_bytes(b"png")
            lock_dir = root / "results/final_evaluation_v2"; lock_dir.mkdir(parents=True)
            (lock_dir / "EXPERIMENT_MATRIX_LOCKED").write_text("{}")
            (lock_dir / "primary_panel_manifest.json").write_text(json.dumps({"experiment_ids": ["maxvit512__R__ensemble"]}))
            (lock_dir / "secondary_panel_manifest.json").write_text(json.dumps({"experiment_ids": []}))
            (lock_dir / "ablation_panel_manifest.json").write_text(json.dumps({"experiment_ids": []}))

            seen_paths = []

            def fake_predictor(job, checkpoint, test_rows):
                seen_paths.extend(r["processed_path"] for r in test_rows)
                return [0.5 for _ in test_rows]

            import finalize_locked_test_stage as lock_mod
            original_verify = lock_mod.verify_lock_still_valid
            lock_mod.verify_lock_still_valid = lambda root: (True, [])
            other_cwd = tempfile.mkdtemp()
            original_cwd = os.getcwd()
            try:
                os.chdir(other_cwd)  # a cwd with no relationship to `root` at all
                lmi.run_locked(root, predictor_fn=fake_predictor)
            finally:
                os.chdir(original_cwd)
                lock_mod.verify_lock_still_valid = original_verify
            self.assertTrue(all(Path(p).is_absolute() for p in seen_paths))
            self.assertTrue(all(Path(p).is_file() for p in seen_paths))


class TestImageSignatureTests(unittest.TestCase):
    """Blocker 10 (test dataset signature): a modified test PNG must invalidate the lock even
    when test.csv itself is unchanged.
    """

    def _project(self, root: Path) -> None:
        (root / "data/processed/metadata").mkdir(parents=True)
        (root / "data/processed/test/0").mkdir(parents=True)
        (root / "data/processed/test/0/p1_i1.png").write_bytes(b"original pixels")
        (root / "data/processed/metadata/test.csv").write_text(
            "patient_id,image_id,label,processed_path\np1,i1,0,data/processed/test/0/p1_i1.png\n")

    def test_manifest_includes_image_signatures(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import finalize_locked_test_stage as lock
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._project(root)
            manifest = lock.build_test_dataset_manifest(root)
            self.assertIn("image_signatures_sha256", manifest)
            self.assertEqual(manifest["n_images_missing"], 0)

    def test_editing_test_png_changes_signature_even_though_csv_is_unchanged(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import finalize_locked_test_stage as lock
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._project(root)
            before = lock.build_test_dataset_manifest(root)
            (root / "data/processed/test/0/p1_i1.png").write_bytes(b"tampered pixels")
            after = lock.build_test_dataset_manifest(root)
            self.assertEqual(before["test_csv_sha256"], after["test_csv_sha256"])  # csv itself untouched
            self.assertNotEqual(before["image_signatures_sha256"], after["image_signatures_sha256"])

    def test_verify_lock_reports_image_tampering_explicitly(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import finalize_locked_test_stage as lock
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self._project(root)
            lock_dir = root / "results/final_evaluation_v2"; lock_dir.mkdir(parents=True)
            (lock_dir / "test_dataset_manifest.json").write_text(json.dumps(lock.build_test_dataset_manifest(root)))
            (lock_dir / "experiment_matrix_manifest.json").write_text(json.dumps({"schema_version": 1}))
            (lock_dir / "primary_finalists_checkpoints.json").write_text(json.dumps({}))
            (lock_dir / "EXPERIMENT_MATRIX_LOCKED").write_text("{}")
            (root / "configs").mkdir(exist_ok=True)
            (root / "configs/classifier_experiment_matrix.json").write_text(json.dumps({"jobs": []}))
            (root / "configs/classifier_training_protocols.json").write_text(json.dumps({"policies": {}}))
            (root / "data/processed/test/0/p1_i1.png").write_bytes(b"tampered")
            for name in ("primary_finalists_manifest.json", "primary_panel_manifest.json",
                         "secondary_panel_manifest.json", "ablation_panel_manifest.json"):
                (lock_dir / name).write_text("{}")
            valid, problems = lock.verify_lock_still_valid(root)
            self.assertFalse(valid)
            self.assertTrue(any("image content changed" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
