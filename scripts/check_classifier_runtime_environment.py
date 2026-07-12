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
    "mammofm": ("torch", "omegaconf", "efficientnet_pytorch"),
    "raddino": ("torch", "transformers"),
}
INSTALL = {"tensorflow": "conda install tensorflow", "torch": "install PyTorch for the local CUDA version",
           "timm": "pip install timm", "omegaconf": "pip install omegaconf", "efficientnet_pytorch": "pip install efficientnet-pytorch",
           "transformers": "pip install transformers", "cuda_gpu": "restore NVIDIA driver / CUDA device visibility"}


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


def _runtime_gpu_report() -> dict:
    report = {"tensorflow": {"available": False, "devices": [], "error": None},
              "torch": {"available": False, "device_count": 0, "devices": [], "error": None}}
    try:
        import tensorflow as tf
        report["tensorflow"]["devices"] = [device.name for device in tf.config.list_physical_devices("GPU")]
        report["tensorflow"]["available"] = bool(report["tensorflow"]["devices"])
    except Exception as exc:
        report["tensorflow"]["error"] = repr(exc)
    try:
        import torch
        report["torch"]["available"] = bool(torch.cuda.is_available())
        report["torch"]["device_count"] = int(torch.cuda.device_count())
        if report["torch"]["available"]:
            report["torch"]["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except Exception as exc:
        report["torch"]["error"] = repr(exc)
    return report


def _asset_header_ok(arch: str, asset: str | None) -> bool:
    if not asset: return False
    path = Path(asset)
    try:
        if arch == "mammofm":
            with path.open("rb") as stream: return bool(stream.read(16))
        if arch == "raddino":
            return _transformer_asset(path) and json.loads((path / "config.json").read_text()).get("model_type") is not None
        if arch == "maxvit512":
            return _transformer_asset(path)
        return path.is_file() and path.stat().st_size > 1024
    except Exception:
        return False


def _check_writable(root: Path) -> bool:
    probe = root / "results" / ".classifier_environment_probe"
    try:
        probe.parent.mkdir(parents=True, exist_ok=True); probe.write_text("ok"); probe.unlink(); return True
    except OSError:
        return False


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
    gpu = _gpu_info(); runtime_gpu = _runtime_gpu_report()
    architectures = {}
    for arch, deps in ARCH_DEPS.items():
        missing = [dep for dep in deps if not modules[dep]]
        framework_gpu = runtime_gpu["tensorflow"]["available"] if arch == "resnet50" else runtime_gpu["torch"]["available"]
        if not framework_gpu: missing.append("cuda_gpu")
        if not _asset_header_ok(arch, assets.get(arch)): missing.append(f"{arch}_asset")
        architectures[arch] = {"status": "PASS" if not missing else "FAIL", "missing": missing,
            "commands": [INSTALL.get(dep, f"provide local {dep}; automatic download is disabled") for dep in missing]}
    return {"schema_version": 2, "offline": True, "python": os.sys.executable, "gpu": gpu, "runtime_gpu": runtime_gpu, "modules": modules,
            "assets": assets, "dataset_registry": (root / "configs/dataset_variant_registry.json").is_file(),
            "notebook_dependencies": {name: importlib.util.find_spec(name) is not None for name in ("nbformat", "nbclient", "matplotlib", "pandas")},
            "disk_free_gib": round(disk.free / 2**30, 2), "writable": _check_writable(root),
            "architectures": architectures}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", default=str(ROOT)); parser.add_argument("--json")
    args = parser.parse_args(); report = audit(Path(args.project_root))
    for arch, row in report["architectures"].items(): print(f"{row['status']:4} {arch}: {', '.join(row['missing']) or 'ready'}")
    if args.json:
        out = Path(args.json); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__": main()
