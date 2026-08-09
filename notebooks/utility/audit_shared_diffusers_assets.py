#!/usr/bin/env python3
"""Audit and conservatively remove verified duplicate Diffusers/SD2.1 assets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import shared_diffusers_assets
from shared_diffusers_assets import (
    DIFFUSERS_REVISION, find_duplicate_diffusers_copies,
    find_duplicate_sd21_base_copies, resolve_shared_diffusers_repo,
    resolve_shared_sd21_base, shared_sd21_signature, verify_diffusers_revision,
    verify_shared_sd21_base,
)


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def has_active_symlink_dependency(root: Path, target: Path) -> bool:
    for item in root.rglob("*"):
        if item.is_symlink():
            try:
                if item.resolve() == target.resolve() or target.resolve() in item.resolve().parents:
                    return True
            except OSError:
                return True
    return False


def audit(root: Path | None = None) -> dict:
    root = root if root is not None else shared_diffusers_assets.PROJECT_ROOT
    shared_repo, shared_model = resolve_shared_diffusers_repo(), resolve_shared_sd21_base()
    try: repo_info = verify_diffusers_revision(shared_repo)
    except Exception as exc: repo_info = {"valid": False, "error": str(exc)}
    try:
        verify_shared_sd21_base(shared_model); model_signature = shared_sd21_signature(shared_model)
    except Exception as exc: model_signature = {"valid": False, "error": str(exc)}
    repos, models = [], []
    for path in find_duplicate_diffusers_copies(root):
        try: info = verify_diffusers_revision(path)
        except Exception as exc: info = {"valid": False, "error": str(exc)}
        repos.append({"path": str(path), "size_bytes": directory_size(path), **info})
    for path in find_duplicate_sd21_base_copies(root):
        try:
            verify_shared_sd21_base(path); signature = shared_sd21_signature(path)
            valid = signature == model_signature
        except Exception as exc:
            signature, valid = {"error": str(exc)}, False
        models.append({"path": str(path), "size_bytes": directory_size(path), "signature": signature, "matches_shared": valid})
    return {"shared_diffusers": {"path": str(shared_repo), **repo_info},
            "shared_sd21": {"path": str(shared_model), "signature": model_signature},
            "duplicate_diffusers": repos, "duplicate_sd21": models,
            "recoverable_bytes": sum(x["size_bytes"] for x in repos if x.get("revision") == DIFFUSERS_REVISION and not x.get("dirty")) + sum(x["size_bytes"] for x in models if x["matches_shared"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--remove-verified-duplicate-diffusers-repos", action="store_true")
    parser.add_argument("--remove-verified-duplicate-sd21-copies", action="store_true")
    args = parser.parse_args(); report = audit()
    print(report)
    for item in report["duplicate_diffusers"]:
        path = Path(item["path"])
        if args.remove_verified_duplicate_diffusers_repos and item.get("revision") == DIFFUSERS_REVISION and not item.get("dirty") and not has_active_symlink_dependency(shared_diffusers_assets.PROJECT_ROOT, path):
            shutil.rmtree(path)
    for item in report["duplicate_sd21"]:
        path = Path(item["path"])
        if args.remove_verified_duplicate_sd21_copies and item.get("matches_shared") and not has_active_symlink_dependency(shared_diffusers_assets.PROJECT_ROOT, path):
            shutil.rmtree(path)


if __name__ == "__main__":
    main()
