#!/usr/bin/env python3
"""Real, short forward/backward VRAM probe per architecture (spec section 10.4).

    python scripts/profile_classifier_vram.py
    python scripts/profile_classifier_vram.py --architecture maxvit512

Writes results/runtime_profiles/classifier_vram_profiles.json, consumed by
classifier_gpu_scheduler.Scheduler as the authoritative estimated_peak_mb per architecture
instead of the scheduler's conservative resource-profile fallback guesses.

This performs real GPU work (loads a real backbone, runs a few real batches) — it is listed
among the "next commands" for the user to run on the target machine, not executed as part of
code-readiness verification.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))


def _probe_pytorch(architecture: str, policy: dict, n_batches: int) -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"error": "CUDA not available in this process"}
    device = torch.device("cuda")

    if architecture == "maxvit512":
        from maxvit_utils import build_maxvit_model
        model = build_maxvit_model(num_classes=1, pretrained=True)
    elif architecture == "mammofm":
        from mammofm_utils import build_mammofm_model
        model = build_mammofm_model()[0]
    elif architecture == "raddino":
        from medfoundation_utils import build_medfoundation_model
        model = build_medfoundation_model()[0]
    else:
        return {"error": f"no pytorch builder registered for {architecture}"}

    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=policy["training_phases"][0]["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()

    batch_size = policy["physical_batch_size"]
    h, w = policy["input_size"]
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(n_batches):
        images = torch.randn(batch_size, 3, h, w, device=device)
        labels = torch.randint(0, 2, (batch_size, 1), device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=policy.get("amp", False)):
            logits = model(images)
            loss = criterion(logits.view(-1, 1), labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        "elapsed_seconds_for_n_batches": elapsed,
        "n_batches": n_batches,
        "physical_batch_size": batch_size,
        "effective_batch_size": policy["effective_batch_size"],
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_uuid": getattr(torch.cuda.get_device_properties(device), "uuid", None),
    }


def _probe_resnet50(policy: dict, n_batches: int) -> dict:
    import numpy as np
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return {"error": "no GPU visible to TensorFlow in this process"}
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg",
                                            input_shape=(*policy["input_size"], 3))
    head = tf.keras.Sequential([model, tf.keras.layers.Dense(256), tf.keras.layers.BatchNormalization(),
                                 tf.keras.layers.LeakyReLU(), tf.keras.layers.Dropout(0.5),
                                 tf.keras.layers.Dense(1, activation="sigmoid")])
    head.compile(optimizer=tf.keras.optimizers.Adam(policy["training_phases"][0]["learning_rate"]), loss="binary_crossentropy")

    batch_size = policy["physical_batch_size"]
    h, w = policy["input_size"]
    x = np.random.rand(batch_size, h, w, 3).astype("float32")
    y = np.random.randint(0, 2, size=(batch_size, 1)).astype("float32")

    tf.config.experimental.reset_memory_stats(gpus[0].name.replace("/physical_device:", ""))
    t0 = time.perf_counter()
    for _ in range(n_batches):
        head.train_on_batch(x, y)
    elapsed = time.perf_counter() - t0
    mem = tf.config.experimental.get_memory_info(gpus[0].name.replace("/physical_device:", ""))

    return {
        "peak_allocated_mb": mem["peak"] / (1024 ** 2),
        "peak_reserved_mb": mem["peak"] / (1024 ** 2),
        "elapsed_seconds_for_n_batches": elapsed,
        "n_batches": n_batches,
        "physical_batch_size": batch_size,
    }


def profile_architecture(architecture: str, policy: dict, n_batches: int) -> dict:
    try:
        if policy["framework"] == "tensorflow_keras":
            result = _probe_resnet50(policy, n_batches)
        else:
            result = _probe_pytorch(architecture, policy, n_batches)
    except Exception as exc:  # noqa: BLE001
        result = {"error": f"{type(exc).__name__}: {exc}"}
    result["architecture"] = architecture
    result["framework"] = policy["framework"]
    result["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", default=None, help="Profile only this architecture (default: all)")
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Resolve policies/assets without loading a model or writing profiles")
    parser.add_argument("--smoke-tiny", action="store_true", help="Exercise the adapter contract without GPU/model loading")
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args()

    root = Path(args.project_root)
    protocols = json.loads((root / "configs/classifier_training_protocols.json").read_text())["policies"]
    architectures = [args.architecture] if args.architecture else list(protocols)

    if args.dry_run:
        print(json.dumps({architecture: {
            "framework": protocols[architecture]["framework"],
            "physical_batch_size": protocols[architecture]["physical_batch_size"],
            "gradient_accumulation_steps": protocols[architecture]["gradient_accumulation_steps"],
            "effective_batch_size": protocols[architecture]["effective_batch_size"],
            "checkpoint_base": protocols[architecture]["checkpoint_base"],
        } for architecture in architectures}, indent=1))
        return
    if args.smoke_tiny:
        from classifier_architecture_adapters import get_adapter
        print(json.dumps({architecture: get_adapter(architecture, protocols[architecture], root, tiny=True)
                          .estimate_memory_profile() for architecture in architectures}, indent=1))
        return

    out_path = root / "results/runtime_profiles/classifier_vram_profiles.json"
    existing = {}
    if out_path.is_file():
        try:
            existing = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            pass

    for architecture in architectures:
        print(f"profiling {architecture} ...")
        result = profile_architecture(architecture, protocols[architecture], args.n_batches)
        existing[architecture] = result
        print(json.dumps(result, indent=1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=1) + "\n")
    print(f"written: {out_path.relative_to(root)}")


if __name__ == "__main__":
    main()
