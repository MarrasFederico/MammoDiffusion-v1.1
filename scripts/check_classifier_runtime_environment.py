#!/usr/bin/env python3
"""Offline runtime audit for classifier training; never downloads packages or weights."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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


def _hf_model_dirs(repo: str) -> list[Path]:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    return list((hub / ("models--" + repo.replace("/", "--")) / "snapshots").glob("*"))


def _transformer_asset(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file() and bool(list(path.glob("*.safetensors")) or list(path.glob("pytorch_model*.bin")))


def _mammofm_asset(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024 and path.suffix in {".tar", ".pt", ".pth"}


def audit(root: Path) -> dict:
    modules = {name: importlib.util.find_spec(name) is not None for deps in ARCH_DEPS.values() for name in deps}
    mammofm_env = Path(os.environ.get("MAMMOFM_LOCAL_CHECKPOINT_PATH", "__missing__"))
    raddino_env = Path(os.environ.get("RADDINO_MODEL_PATH", "__missing__"))
    mammofm_candidates = [mammofm_env] + [p for d in _hf_model_dirs("batmanLab/Mammo-FM") for p in d.glob("*.tar")]
    raddino_candidates = [raddino_env] + _hf_model_dirs("microsoft/rad-dino")
    maxvit_candidates = _hf_model_dirs("timm/maxvit_tiny_tf_512.in1k")
    keras_resnet = Path.home() / ".keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    assets = {"resnet50": str(keras_resnet) if keras_resnet.is_file() else None,
              "maxvit512": next((str(p) for p in maxvit_candidates if _transformer_asset(p)), None),
              "mammofm": next((str(p) for p in mammofm_candidates if _mammofm_asset(p)), None),
              "raddino": next((str(p) for p in raddino_candidates if _transformer_asset(p)), None)}
    disk = shutil.disk_usage(root)
    gpu = _gpu_info(); gpu_available = bool(gpu) and not str(gpu[0]).startswith("unavailable:")
    architectures = {}
    for arch, deps in ARCH_DEPS.items():
        missing = [dep for dep in deps if not modules[dep]]
        if not gpu_available: missing.append("cuda_gpu")
        if not assets.get(arch): missing.append(f"{arch}_asset")
        architectures[arch] = {"status": "PASS" if not missing else "FAIL", "missing": missing,
            "commands": [INSTALL.get(dep, f"provide local {dep}; automatic download is disabled") for dep in missing]}
    return {"schema_version": 1, "offline": True, "gpu": gpu, "modules": modules,
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
