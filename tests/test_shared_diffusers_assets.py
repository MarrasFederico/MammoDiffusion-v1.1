from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import shared_diffusers_assets as assets
from audit_shared_diffusers_assets import audit


class SharedDiffusersAssetsTests(unittest.TestCase):
    def fake_repo(self, root: Path) -> tuple[Path, str]:
        repo = root / "diffusers"; repo.mkdir(); subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
        script_dir = repo / "examples/text_to_image"; script_dir.mkdir(parents=True)
        (script_dir / "train_text_to_image.py").write_text("pass\n"); (script_dir / "train_text_to_image_lora.py").write_text("pass\n")
        (repo / "src/diffusers").mkdir(parents=True); (repo / "src/diffusers/__init__.py").write_text("")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True); subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        return repo, head

    def fake_model(self, root: Path) -> Path:
        model = root / "stable-diffusion-2-1-base"
        for name in assets.REQUIRED_SD21_COMPONENTS:
            path = model / name
            if "." in name: path.parent.mkdir(parents=True, exist_ok=True); path.write_text("{}")
            else: path.mkdir(parents=True); (path / "config.json").write_text("{}")
        (model / "tokenizer/tokenizer_config.json").write_text("{}")
        (model / "unet/diffusion_pytorch_model.safetensors").write_bytes(b"weight")
        return model

    def test_revision_pinned_and_dirty_tree_not_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, head = self.fake_repo(Path(tmp)); (repo / "dirty.txt").write_text("keep")
            resolved = assets.ensure_shared_diffusers_repo(repo, revision=head)
            self.assertEqual(resolved, repo.resolve()); self.assertTrue((repo / "dirty.txt").is_file())
            with self.assertRaisesRegex(RuntimeError, "Refusing checkout of dirty"):
                assets.ensure_shared_diffusers_repo(repo, revision="0" * 40)

    def test_training_scripts_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, _ = self.fake_repo(Path(tmp))
            self.assertEqual(assets.shared_diffusers_train_script(False, repo).name, "train_text_to_image.py")
            self.assertEqual(assets.shared_diffusers_train_script(True, repo).name, "train_text_to_image_lora.py")

    def test_model_verification_and_content_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.fake_model(Path(tmp)); before = assets.shared_sd21_signature(model)
            (model / "unet/diffusion_pytorch_model.safetensors").write_bytes(b"changed")
            after = assets.shared_sd21_signature(model); self.assertNotEqual(before["sha256"], after["sha256"])

    def test_asset_lock_serializes_callers(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "asset.lock"; order = []
            def worker(name):
                with assets._atomic_lock(lock): order.append((name, "enter")); time.sleep(.02); order.append((name, "exit"))
            threads = [threading.Thread(target=worker, args=(name,)) for name in ("a", "b")]
            [thread.start() for thread in threads]; [thread.join() for thread in threads]
            self.assertIn(order, [[("a","enter"),("a","exit"),("b","enter"),("b","exit")], [("b","enter"),("b","exit"),("a","enter"),("a","exit")]])

    def test_notebooks_share_assets_and_no_experiment_repo(self):
        paths = sorted((ROOT / "notebooks/2_diffusers").glob("0[1-4]_*.ipynb"))
        for path in paths:
            text = path.read_text(); self.assertIn("SHARED_DIFFUSERS_REPO_DIR", text); self.assertIn("SHARED_SD21_BASE_DIR", text)
            self.assertNotIn('EXPERIMENT_DIR / \\"diffusers_repo\\"', text)

    def test_default_diffusers_checkout_is_shared_under_notebooks(self):
        expected = ROOT / "notebooks" / "utility" / "diffusers_repo"
        self.assertEqual(assets.resolve_shared_diffusers_repo(), expected.resolve())

    def test_editable_install_is_immediately_importable_in_running_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "diffusers"
            package = repo / "src" / "diffusers"
            package.mkdir(parents=True)
            init_file = package / "__init__.py"
            init_file.write_text("FIRST_RUN_MARKER = True\n")

            original_path = list(sys.path)
            previous_modules = {
                name: module
                for name, module in sys.modules.items()
                if name == "diffusers" or name.startswith("diffusers.")
            }
            for name in previous_modules:
                sys.modules.pop(name, None)
            stale = ModuleType("diffusers")
            stale.__file__ = str(Path(tmp) / "stale" / "diffusers" / "__init__.py")
            sys.modules["diffusers"] = stale
            sys.modules["diffusers.stale"] = ModuleType("diffusers.stale")
            try:
                with mock.patch.object(assets.subprocess, "run") as pip_run:
                    imported = assets.ensure_diffusers_editable_install(repo)
                pip_run.assert_called_once_with(
                    [sys.executable, "-m", "pip", "install", "-e", str(repo.resolve())],
                    check=True,
                )
                self.assertEqual(imported, init_file.resolve())
                self.assertTrue(importlib.import_module("diffusers").FIRST_RUN_MARKER)
                self.assertEqual(sys.path[0], str((repo / "src").resolve()))
                self.assertNotIn("diffusers.stale", sys.modules)
            finally:
                for name in list(sys.modules):
                    if name == "diffusers" or name.startswith("diffusers."):
                        sys.modules.pop(name, None)
                sys.modules.update(previous_modules)
                sys.path[:] = original_path

    def test_duplicate_audit_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); marker = root / "marker"; marker.write_text("keep")
            audit(root); self.assertEqual(marker.read_text(), "keep")

    def test_project_root_with_readme_and_notebooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "notebooks").mkdir(); (root / "README.md").write_text("x")
            self.assertEqual(assets.project_root(root), root.resolve())

    def test_project_root_without_readme_but_with_mammodiffusion_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notebooks" / "utility").mkdir(parents=True)
            (root / "notebooks" / "2_diffusers").mkdir(parents=True)
            self.assertEqual(assets.project_root(root), root.resolve())
            configs_root = Path(tmp) / "configs_variant"
            (configs_root / "configs").mkdir(parents=True)
            (configs_root / "configs" / "final_classifier_registry.json").write_text("{}")
            (configs_root / "notebooks").mkdir()
            self.assertEqual(assets.project_root(configs_root), configs_root.resolve())

    def test_project_root_env_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "notebooks").mkdir(); (root / "README.md").write_text("x")
            old = os.environ.get("MAMMODIFFUSION_PROJECT_ROOT")
            os.environ["MAMMODIFFUSION_PROJECT_ROOT"] = str(root)
            try:
                self.assertEqual(assets.project_root(), root.resolve())
            finally:
                if old is None: del os.environ["MAMMODIFFUSION_PROJECT_ROOT"]
                else: os.environ["MAMMODIFFUSION_PROJECT_ROOT"] = old

    def test_import_from_clean_copy_never_crashes(self):
        # A bare copy of just this one file, with no notebooks/, README.md, or configs/ anywhere
        # above it -- importing it must not crash even though project_root() would eventually
        # fail if something actually asked for PROJECT_ROOT.
        with tempfile.TemporaryDirectory() as tmp:
            isolated = Path(tmp) / "isolated"; isolated.mkdir()
            shutil.copy(ROOT / "notebooks/utility/shared_diffusers_assets.py", isolated / "shared_diffusers_assets.py")
            result = subprocess.run(
                [sys.executable, "-c", "import shared_diffusers_assets; print('IMPORT_OK')"],
                cwd=str(isolated), capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("IMPORT_OK", result.stdout)


if __name__ == "__main__": unittest.main()
