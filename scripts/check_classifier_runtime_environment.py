#!/usr/bin/env python3
"""Offline runtime audit for classifier training; never downloads packages or weights."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH_DEPS = {
    "resnet50": ("tensorflow",),
    "maxvit512": ("torch", "timm"),
    "mammofm": ("torch", "omegaconf"),
    "raddino": ("torch", "transformers"),
}
INSTALL = {"tensorflow": "pip install tensorflow", "torch": "install PyTorch for the local CUDA version",
           "timm": "pip install timm", "omegaconf": "pip install omegaconf", "transformers": "pip install transformers"}


def _gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception as exc:
        return [f"unavailable: {type(exc).__name__}: {exc}"]


def audit(root: Path) -> dict:
    modules = {name: importlib.util.find_spec(name) is not None for deps in ARCH_DEPS.values() for name in deps}
    assets = {
        "mammofm": bool(list(root.glob("**/*Mammo*FM*"))) or bool(list(root.glob("**/*mammofm*checkpoint*"))),
        "raddino": bool(list(root.glob("**/*RAD*DINO*"))) or bool(list(root.glob("**/*rad*dino*"))),
    }
    disk = shutil.disk_usage(root)
    architectures = {}
    for arch, deps in ARCH_DEPS.items():
        missing = [dep for dep in deps if not modules[dep]]
        if arch in assets and not assets[arch]: missing.append(f"{arch}_asset")
        architectures[arch] = {"status": "PASS" if not missing else "FAIL", "missing": missing,
            "commands": [INSTALL.get(dep, f"provide local {dep}; automatic download is disabled") for dep in missing]}
    return {"schema_version": 1, "offline": True, "gpu": _gpu_info(), "modules": modules,
            "assets": assets, "dataset_registry": (root / "configs/dataset_variant_registry.json").is_file(),
            "notebook_dependencies": {name: importlib.util.find_spec(name) is not None for name in ("nbformat", "nbclient", "matplotlib", "pandas")},
            "disk_free_gib": round(disk.free / 2**30, 2), "writable": (root / "results").exists(),
            "architectures": architectures}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", default=str(ROOT)); parser.add_argument("--json")
    args = parser.parse_args(); report = audit(Path(args.project_root))
    for arch, row in report["architectures"].items(): print(f"{row['status']:4} {arch}: {', '.join(row['missing']) or 'ready'}")
    if args.json: Path(args.json).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__": main()
