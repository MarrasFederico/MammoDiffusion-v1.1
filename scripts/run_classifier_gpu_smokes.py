#!/usr/bin/env python3
"""Run future real-GPU synthetic-fixture smokes and write a signed certification bundle.

This command loads locally available registered models and performs forward, backward, and
final-checkpoint save/load checks. It must only be run explicitly on the target GPU host after
profiling; it never reads train, validation, or locked-test data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

from classifier_gpu_gate import SMOKE_PATH, environment_signature, make_bundle  # noqa: E402
from classifier_pipeline_contracts import ARCHITECTURES, code_revision, value_signature  # noqa: E402
from classifier_gpu_scheduler import query_gpus_live  # noqa: E402


def _selected_gpu(gpu_uuid: str | None) -> dict:
    gpus = query_gpus_live()
    if gpu_uuid:
        match = next((gpu for gpu in gpus if gpu["uuid"] == gpu_uuid), None)
        if match is None: raise ValueError(f"GPU UUID not found: {gpu_uuid}")
        return match
    if len(gpus) != 1:
        raise ValueError("--gpu-uuid is required when nvidia-smi exposes multiple GPUs")
    return gpus[0]


def smoke_architecture(root: Path, architecture: str, policy: dict, gpu: dict) -> dict:
    from classifier_architecture_adapters import ArchitectureAdapter
    adapter = ArchitectureAdapter(architecture, policy, root)
    forward = backward = checkpoint_roundtrip = False
    if policy["framework"] == "tensorflow_keras":
        import numpy as np
        import tensorflow as tf
        visible = tf.config.list_physical_devices("GPU")
        if not visible: raise RuntimeError("TensorFlow cannot see a GPU")
        model = adapter.build_model(pretrained=True)
        batch, (height, width) = int(policy["physical_batch_size"]), policy["input_size"]
        inputs = tf.convert_to_tensor(np.zeros((batch, height, width, 3), dtype="float32"))
        labels = tf.zeros((batch, 1), dtype=tf.float32)
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True); forward = True
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, predictions))
        gradients = tape.gradient(loss, model.trainable_variables)
        backward = any(gradient is not None for gradient in gradients)
    else:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("PyTorch cannot see a GPU")
        device = torch.device("cuda")
        model = adapter.build_model(pretrained=True).to(device).train()
        batch, (height, width) = int(policy["physical_batch_size"]), policy["input_size"]
        inputs = torch.zeros(batch, 3, height, width, device=device)
        labels = torch.zeros(batch, 1, device=device)
        predictions = model(inputs); forward = True
        loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions.reshape(-1, 1), labels)
        loss.backward(); backward = any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    with tempfile.TemporaryDirectory() as temporary:
        extension = "model.keras" if architecture == "resnet50" else "model.pt"
        checkpoint = Path(temporary) / extension
        adapter.save_checkpoint(model, checkpoint)
        adapter.load_checkpoint(checkpoint, strict=True)
        checkpoint_roundtrip = True
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    fixture_signature = value_signature({"kind": "zero_tensor_binary_fixture", "input_size": policy["input_size"],
                                         "batch": policy["physical_batch_size"]})
    return {"architecture": architecture, "environment_signature": environment_signature(policy),
            "gpu_name": gpu["name"], "gpu_uuid": gpu["uuid"], "total_vram_mb": gpu["total_vram_mb"],
            "physical_batch_size": policy["physical_batch_size"],
            "gradient_accumulation_steps": policy["gradient_accumulation_steps"],
            "forward_pass": forward, "backward_pass": backward, "checkpoint_save_load": checkpoint_roundtrip,
            "measured_at": now, "code_revision": code_revision(root), "fixture_signature": fixture_signature}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.project_root); gpu = _selected_gpu(args.gpu_uuid)
    protocols = json.loads((root / "configs/classifier_training_protocols.json").read_text())["policies"]
    record = smoke_architecture(root, args.architecture, protocols[args.architecture], gpu)
    path = root / SMOKE_PATH
    existing = json.loads(path.read_text()) if path.is_file() else {"records": []}
    if path.is_file():
        from classifier_pipeline_contracts import verify_signed_payload
        verify_signed_payload(existing)
        if existing.get("artifact_type") != "gpu_smoke_bundle":
            raise ValueError("existing GPU smoke file has an incompatible artifact type")
    records = [row for row in existing.get("records", []) if row.get("architecture") != args.architecture] + [record]
    from classifier_pipeline_contracts import atomic_json
    atomic_json(path, make_bundle("gpu_smoke_bundle", sorted(records, key=lambda row: row["architecture"])))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
