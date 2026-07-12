"""Canonical, content-aware access to shared Diffusers and SD 2.1 assets."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

DIFFUSERS_REVISION = "3759fab56d3170a04d747e918a13e55fda6681e2"
REQUIRED_SD21_COMPONENTS = (
    "model_index.json", "vae", "unet", "scheduler", "text_encoder", "tokenizer",
)


def _looks_like_mammodiffusion_root(candidate: Path) -> bool:
    """Any one of several independent signatures is enough to recognise the project root.

    A README.md-less copy (a packaged code-audit subset, a stripped CI checkout, ...) must still
    resolve correctly -- root discovery must not be so fragile that it blocks importing this
    module from anything but the exact, complete, real repository.
    """
    if (candidate / "notebooks").is_dir() and (candidate / "README.md").is_file():
        return True
    if (candidate / "notebooks" / "utility").is_dir() and (candidate / "notebooks" / "2_diffusers").is_dir():
        return True
    if (candidate / "configs" / "final_classifier_registry.json").is_file() and (candidate / "notebooks").is_dir():
        return True
    return False


def project_root(start: str | Path | None = None) -> Path:
    if start is None:
        env_override = os.environ.get("MAMMODIFFUSION_PROJECT_ROOT")
        if env_override:
            candidate = Path(env_override).expanduser().resolve()
            if _looks_like_mammodiffusion_root(candidate):
                return candidate
            raise FileNotFoundError(f"MAMMODIFFUSION_PROJECT_ROOT does not look like a MammoDiffusion project root: {candidate}")
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if _looks_like_mammodiffusion_root(candidate):
            return candidate
    raise FileNotFoundError("MammoDiffusion project root not found")


_project_root_cache: Path | None = None


def _cached_project_root() -> Path:
    """Resolve project_root() lazily, once. Importing this module must never crash just because
    the root can't be discovered from the current working directory -- that error (if any) is
    deferred until something actually needs the root."""
    global _project_root_cache
    if _project_root_cache is None:
        _project_root_cache = project_root()
    return _project_root_cache


def _default_shared_diffusers_repo_dir() -> Path:
    return _cached_project_root() / "experiments" / "diffusers" / "03_sd21_vae_finetuned" / "diffusers_repo"


def _default_shared_sd21_base_dir() -> Path:
    return _cached_project_root() / "notebooks" / "pretrained_model" / "stable-diffusion-2-1-base"


def __getattr__(name: str):
    # PEP 562: PROJECT_ROOT/SHARED_DIFFUSERS_REPO_DIR/SHARED_SD21_BASE_DIR are resolved lazily,
    # on first external access (module.attr / from module import attr), instead of at import time.
    if name == "PROJECT_ROOT":
        return _cached_project_root()
    if name == "SHARED_DIFFUSERS_REPO_DIR":
        return _default_shared_diffusers_repo_dir()
    if name == "SHARED_SD21_BASE_DIR":
        return _default_shared_sd21_base_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextmanager
def _atomic_lock(path: Path, timeout: float = 120.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(), "created_at": time.time()}).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                pid = int(json.loads(path.read_text())["pid"])
            except Exception:
                pid = -1
            if pid > 0 and not _pid_alive(pid):
                path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for asset lock: {path}")
            time.sleep(0.1)
    try:
        yield
    finally:
        try:
            if json.loads(path.read_text()).get("pid") == os.getpid():
                path.unlink()
        except (OSError, ValueError, json.JSONDecodeError):
            pass


def resolve_shared_diffusers_repo(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).resolve()
    env_value = os.environ.get("MAMMODIFFUSION_DIFFUSERS_REPO")
    return (Path(env_value) if env_value else _default_shared_diffusers_repo_dir()).resolve()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, capture_output=True)
    return result.stdout.strip()


def verify_diffusers_revision(repo: str | Path, revision: str = DIFFUSERS_REVISION) -> dict:
    repo = Path(repo).resolve()
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"Not a Diffusers Git working tree: {repo}")
    head = _git(repo, "rev-parse", "HEAD")
    if head != revision:
        raise RuntimeError(f"Diffusers revision mismatch: expected {revision}, found {head} in {repo}")
    return {"path": str(repo), "revision": head, "dirty": bool(_git(repo, "status", "--porcelain"))}


def ensure_shared_diffusers_repo(
    path: str | Path | None = None, revision: str = DIFFUSERS_REVISION,
    url: str = "https://github.com/huggingface/diffusers.git",
) -> Path:
    repo = resolve_shared_diffusers_repo(path)
    with _atomic_lock(repo.parent / ".diffusers_assets.lock"):
        if not repo.exists():
            subprocess.run(["git", "clone", url, str(repo)], check=True)
        info = verify_diffusers_revision(repo, revision) if _git(repo, "rev-parse", "HEAD") == revision else None
        if info is None:
            if _git(repo, "status", "--porcelain"):
                raise RuntimeError(f"Refusing checkout of dirty Diffusers working tree: {repo}")
            subprocess.run(["git", "-C", str(repo), "fetch", "origin", revision], check=True)
            subprocess.run(["git", "-C", str(repo), "checkout", "--detach", revision], check=True)
            verify_diffusers_revision(repo, revision)
    return repo


def ensure_diffusers_editable_install(repo: str | Path) -> Path:
    repo = Path(repo).resolve()
    src = (repo / "src").resolve()
    try:
        module = importlib.import_module("diffusers")
        imported = Path(module.__file__).resolve()
        if src in imported.parents:
            return imported
    except ImportError:
        pass
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(repo)], check=True)
    importlib.invalidate_caches()
    imported = Path(importlib.import_module("diffusers").__file__).resolve()
    if src not in imported.parents:
        raise RuntimeError(f"Imported diffusers is not from shared repository: {imported}")
    return imported


def shared_diffusers_train_script(lora: bool = False, repo: str | Path | None = None) -> Path:
    name = "train_text_to_image_lora.py" if lora else "train_text_to_image.py"
    script = resolve_shared_diffusers_repo(repo) / "examples" / "text_to_image" / name
    if not script.is_file():
        raise FileNotFoundError(script)
    return script.resolve()


def resolve_shared_sd21_base(path: str | Path | None = None) -> Path:
    # Keep the canonical project-facing path in manifests even when it is a
    # compatibility symlink to the single physical copy.
    if path is not None:
        return Path(path).expanduser().absolute()
    env_value = os.environ.get("MAMMODIFFUSION_SD21_BASE")
    return (Path(env_value) if env_value else _default_shared_sd21_base_dir()).expanduser().absolute()


def verify_shared_sd21_base(path: str | Path | None = None) -> Path:
    root = resolve_shared_sd21_base(path)
    missing = [component for component in REQUIRED_SD21_COMPONENTS if not (root / component).exists()]
    config_candidates = (root / "tokenizer" / "tokenizer_config.json", root / "tokenizer" / "config.json")
    if not any(item.is_file() for item in config_candidates):
        missing.append("tokenizer config")
    if missing:
        raise FileNotFoundError(f"Incomplete shared SD2.1 base at {root}; missing: {missing}")
    return root


def _files(root: Path) -> Iterable[Path]:
    return sorted((p for p in root.rglob("*") if p.is_file() and not p.is_symlink()), key=lambda p: p.relative_to(root).as_posix())


def shared_sd21_signature(path: str | Path | None = None) -> dict:
    root = verify_shared_sd21_base(path)
    digest = hashlib.sha256()
    count = size = 0
    for file in _files(root):
        relative = file.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big")); digest.update(relative)
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        count += 1; size += file.stat().st_size
    return {"algorithm": "sha256", "sha256": digest.hexdigest(), "file_count": count, "size_bytes": size}


def find_legacy_diffusers_copies(root: str | Path | None = None) -> list[Path]:
    root, shared = Path(root) if root else _cached_project_root(), resolve_shared_diffusers_repo()
    return [path for path in sorted(root.rglob("diffusers_repo")) if path.is_dir() and path.resolve() != shared]


def find_legacy_sd21_base_copies(root: str | Path | None = None) -> list[Path]:
    root, shared = Path(root) if root else _cached_project_root(), resolve_shared_sd21_base()
    shared_target = shared.resolve()
    return [path for path in sorted(root.rglob("stable-diffusion-2-1-base"))
            if path.is_dir() and path.absolute() != shared and path.resolve() != shared_target]
