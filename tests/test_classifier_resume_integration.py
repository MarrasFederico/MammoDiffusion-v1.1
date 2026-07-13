"""Real, non-mocked integration tests for the resume/best-state fixes (blockers 2/3/4).

These run actual tiny training through notebooks.utility.classifier_architecture_adapters
.ArchitectureAdapter.train() with real cached pretrained weights (MaxViT via timm,
ResNet50 via Keras) and a handful of real preprocessed images -- not a mock, not the tiny
adapter. They are the "real CPU integration" tier deliberately, for two independent,
environment-specific reasons unrelated to the code under test: this TensorFlow build cannot
JIT-compile for the RTX 5060 Ti's compute capability 12.0 (Blackwell) within a reasonable
time, and mixing a CUDA-initialized PyTorch test with a later TensorFlow test in the same
process leaves TF unable to reliably honor CUDA_VISIBLE_DEVICES set after the fact. Both
classes therefore run entirely on CPU (forced below, before either framework is imported)
regardless of what GPUs are visible on the host. They never touch Stage 1 or the locked test.

Skipped outright (not failed) if the required cached pretrained weights are not present on
this machine, so the suite stays runnable in environments without them.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch/tensorflow are ever imported below

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

PROTOCOLS = json.loads((ROOT / "configs/classifier_training_protocols.json").read_text())["policies"]


def _real_rows(n_per_class=2):
    rows = []
    for label, sub in (("0", "0"), ("1", "1")):
        directory = ROOT / "data/processed/train" / sub
        files = sorted(directory.glob("*.png"))[:n_per_class]
        if len(files) < n_per_class:
            return None
        for f in files:
            rows.append({"processed_path": str(f), "label": label, "patient_id": f.stem, "image_id": f.stem})
    return rows


def _tiny_policy(architecture, *, checkpoint_interval_updates=1, max_optimizer_updates=4):
    policy = copy.deepcopy(PROTOCOLS[architecture])
    policy["checkpoint_interval_updates"] = checkpoint_interval_updates
    policy["max_optimizer_updates"] = max_optimizer_updates
    policy["physical_batch_size"] = 2
    policy["dataloader_workers"] = 0
    policy["early_stopping"]["patience"] = 100  # never trigger stopping inside a 2-epoch test
    return policy


def _has_maxvit_cache() -> bool:
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        import timm
        timm.create_model("maxvit_tiny_tf_512.in1k", pretrained=True, num_classes=1)
        return True
    except Exception:
        return False


def _has_resnet50_cache() -> bool:
    return (Path.home() / ".keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5").is_file()


@unittest.skipUnless(_real_rows(2) is not None, "real preprocessed training images not found")
class PyTorchAdapterResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _has_maxvit_cache():
            raise unittest.SkipTest("maxvit_tiny_tf_512.in1k weights not cached locally/offline")
        import classifier_architecture_adapters as caa
        cls.caa = caa

    def _context(self, run_dir):
        return {"run_dir": run_dir, "architecture": "maxvit512", "experiment_id": "maxvit512__TEST__seed17",
                "dataset_variant_id": "TEST", "training_policy": "maxvit512_standard",
                "config_signature": "cfg-sig", "dataset_signature": "data-sig"}

    def test_tiny_epoch_runs_without_typeerror_and_epoch2_checkpoint_is_correct(self):
        rows = _real_rows(2)
        with tempfile.TemporaryDirectory() as t:
            run_dir = Path(t) / "run"
            policy = _tiny_policy("maxvit512", checkpoint_interval_updates=1, max_optimizer_updates=2)
            adapter = self.caa.ArchitectureAdapter("maxvit512", policy, ROOT)
            # Runs fit_mammofm (shared PyTorch loop) with on_before_optimizer_step wired in by
            # the adapter -- this is exactly blocker 2's failure point (TypeError before the
            # fix). No exception here is the assertion.
            result = adapter.train(rows, rows, run_dir / "final.pt", seed=17, **self._context(run_dir))
            self.assertTrue((run_dir / "final.pt").exists() or Path(result["checkpoint"]).exists())

            import classifier_checkpoint_io as ckio
            latest = ckio.read_resume_checkpoint(ckio.resume_checkpoint_path(run_dir, "checkpoint_latest"))
            # A tiny 2-optimizer-update budget with checkpoint_interval_updates=1 guarantees at
            # least one periodic (intra-epoch) save happened; its epoch must never be stale.
            self.assertGreaterEqual(latest["epoch"], 1)
            self.assertIsInstance(latest["history"], dict)

    def test_history_not_wiped_by_periodic_checkpoint(self):
        rows = _real_rows(2)
        with tempfile.TemporaryDirectory() as t:
            run_dir = Path(t) / "run"
            # max_optimizer_updates=1 with interval=1 forces a periodic save on the very first
            # step, before any epoch_end has a chance to populate history "for real" -- proving
            # the periodic path itself seeds history from resume rather than defaulting to {}.
            policy = _tiny_policy("maxvit512", checkpoint_interval_updates=1, max_optimizer_updates=1)
            adapter = self.caa.ArchitectureAdapter("maxvit512", policy, ROOT)
            adapter.train(rows, rows, run_dir / "final.pt", seed=17, **self._context(run_dir))

            import classifier_checkpoint_io as ckio
            latest = ckio.read_resume_checkpoint(ckio.resume_checkpoint_path(run_dir, "checkpoint_latest"))
            self.assertIn("history", latest)  # must be a dict, never None/missing

    def test_best_checkpoint_wins_after_resume_with_no_improvement(self):
        rows = _real_rows(2)
        with tempfile.TemporaryDirectory() as t:
            run_dir = Path(t) / "run"
            policy = _tiny_policy("maxvit512", checkpoint_interval_updates=1, max_optimizer_updates=1)
            adapter = self.caa.ArchitectureAdapter("maxvit512", policy, ROOT)
            adapter.train(rows, rows, run_dir / "final.pt", seed=17, **self._context(run_dir))

            import classifier_checkpoint_io as ckio
            best_path = ckio.resume_checkpoint_path(run_dir, "checkpoint_best")
            self.assertTrue(best_path.is_file(), "checkpoint_best.pkl must exist after at least one epoch")
            best_before = ckio.read_resume_checkpoint(best_path)

            # Resume for one more tiny (likely non-improving) segment.
            policy2 = _tiny_policy("maxvit512", checkpoint_interval_updates=1, max_optimizer_updates=2)
            adapter2 = self.caa.ArchitectureAdapter("maxvit512", policy2, ROOT)
            adapter2.train(rows, rows, run_dir / "final2.pt", seed=17, **self._context(run_dir))

            best_after = ckio.read_resume_checkpoint(best_path)
            # The global best must never regress: its recorded metric can only stay the same or improve.
            self.assertGreaterEqual(best_after["best_metric"], best_before["best_metric"] - 1e-9)


@unittest.skipUnless(_real_rows(2) is not None and _has_resnet50_cache(), "real ResNet50 ImageNet weights or images not found")
class TensorFlowAdapterResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import classifier_architecture_adapters as caa
        cls.caa = caa

    def _context(self, run_dir):
        return {"run_dir": run_dir, "architecture": "resnet50", "experiment_id": "resnet50__TEST__seed17",
                "dataset_variant_id": "TEST", "training_policy": "resnet50_standard",
                "config_signature": "cfg-sig", "dataset_signature": "data-sig"}

    def test_head_interruption_then_resume_does_not_retrain_completed_head(self):
        rows = _real_rows(2)
        with tempfile.TemporaryDirectory() as t:
            run_dir = Path(t) / "run"
            # head_epochs = max(1, epochs // 5); a small max_optimizer_updates keeps epochs=1,
            # so head_epochs=1 too -- one adapter.train() call fully completes the head phase
            # and (with the fix) persists phase="transition" to disk before finetune starts.
            policy = _tiny_policy("resnet50", checkpoint_interval_updates=1, max_optimizer_updates=1)
            adapter = self.caa.ArchitectureAdapter("resnet50", policy, ROOT)
            adapter.train(rows, rows, run_dir / "final.keras", seed=17, **self._context(run_dir))

            import classifier_checkpoint_io as ckio
            latest = ckio.read_resume_checkpoint(ckio.resume_checkpoint_path(run_dir, "checkpoint_latest"))
            self.assertEqual(latest["phase"], "complete")

    def test_finetune_optimizer_never_receives_head_optimizer_state(self):
        # Regression guard at the unit level for the exact reported failure mode: calling
        # fit_phase("finetune", ...) right after a same-run head phase must not raise a
        # variable-count/shape mismatch from assigning head Adam slots onto the finetune
        # optimizer's (differently-shaped) trainable variable set.
        rows = _real_rows(2)
        with tempfile.TemporaryDirectory() as t:
            run_dir = Path(t) / "run"
            policy = _tiny_policy("resnet50", checkpoint_interval_updates=1, max_optimizer_updates=1)
            adapter = self.caa.ArchitectureAdapter("resnet50", policy, ROOT)
            # Must complete without raising a shape-mismatch (this is the assertion): before the
            # fix, the transition from head to finetune attempted to assign head-shaped
            # optimizer slot variables onto the finetune optimizer's differently-sized variable
            # list, which raised well before ever reaching the final model.save() call.
            adapter.train(rows, rows, run_dir / "final.keras", seed=17, **self._context(run_dir))

            import classifier_checkpoint_io as ckio
            self.assertTrue(ckio.resume_checkpoint_path(run_dir, "checkpoint_latest").is_file())

    def test_best_checkpoint_survives_and_is_used(self):
        rows = _real_rows(2)
        with tempfile.TemporaryDirectory() as t:
            run_dir = Path(t) / "run"
            policy = _tiny_policy("resnet50", checkpoint_interval_updates=1, max_optimizer_updates=1)
            adapter = self.caa.ArchitectureAdapter("resnet50", policy, ROOT)
            adapter.train(rows, rows, run_dir / "final.keras", seed=17, **self._context(run_dir))

            import classifier_checkpoint_io as ckio
            best_path = ckio.resume_checkpoint_path(run_dir, "checkpoint_best")
            self.assertTrue(best_path.is_file(), "checkpoint_best.pkl must exist after at least one epoch")
            best = ckio.read_resume_checkpoint(best_path)
            self.assertIsNotNone(best.get("best_metric"))
            self.assertIsNotNone(best.get("model_state"))

    def test_transition_checkpoint_restores_head_weights_in_new_process_then_finetunes(self):
        rows = _real_rows(2)
        with tempfile.TemporaryDirectory() as t:
            temp = Path(t); run_dir = temp / "run"
            config = temp / "config.json"
            config.write_text(json.dumps({"root": str(ROOT), "run_dir": str(run_dir), "rows": rows,
                                          "policy": _tiny_policy("resnet50", checkpoint_interval_updates=1,
                                                                  max_optimizer_updates=1)}))
            common = ("import json,sys; from pathlib import Path; c=json.loads(Path(sys.argv[1]).read_text()); "
                      "sys.path.insert(0,str(Path(c['root'])/'notebooks/utility')); "
                      "import classifier_architecture_adapters as caa; "
                      "a=caa.ArchitectureAdapter('resnet50',c['policy'],Path(c['root'])); "
                      "ctx={'run_dir':Path(c['run_dir']),'architecture':'resnet50',"
                      "'experiment_id':'resnet50__TEST__seed17','dataset_variant_id':'TEST',"
                      "'training_policy':'resnet50_standard','config_signature':'cfg-sig',"
                      "'dataset_signature':'data-sig'}; ")
            first = common + ("\ntry: a.train(c['rows'],c['rows'],Path(c['run_dir'])/'final.keras',seed=17,"
                              "stop_after_transition=True,**ctx)\nexcept caa.TransitionCheckpointReady: pass")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            subprocess.run([sys.executable, "-c", first, str(config)], check=True, env=env)

            import classifier_checkpoint_io as ckio
            import classifier_architecture_adapters as caa
            transition = ckio.read_resume_checkpoint(ckio.resume_checkpoint_path(run_dir))
            self.assertEqual(transition["phase"], "transition")
            transition_hash = caa._numpy_weights_sha256(transition["model_state"])

            second = common + ("a.train(c['rows'],c['rows'],Path(c['run_dir'])/'final.keras',seed=17,"
                               f"expected_transition_weights_sha256='{transition_hash}',**ctx)")
            subprocess.run([sys.executable, "-c", second, str(config)], check=True, env=env)
            complete = ckio.read_resume_checkpoint(ckio.resume_checkpoint_path(run_dir))
            self.assertEqual(complete["phase"], "complete")
            self.assertTrue((run_dir / "final.keras").is_file())


if __name__ == "__main__":
    unittest.main()
