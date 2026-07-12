"""Canonical experiment/results directory layout and checkpoint provenance for the classifier
experiment matrix (spec section 8). Never breaks the pre-existing legacy layout under
experiments/classifiers/<arch>/...; legacy runs are only ever linked via
dataset_variant `legacy_experiment_ids`, never overwritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPERIMENTS_ROOT = "experiments/classifiers_matrix"
RESULTS_ROOT = "results/classifiers_matrix"


def experiment_id(architecture: str, dataset_variant_id: str, seed: int) -> str:
    return f"{architecture}__{dataset_variant_id}__seed{seed}"


def parse_experiment_id(experiment_id_str: str) -> dict:
    architecture, dataset_variant_id, seed_part = experiment_id_str.split("__")
    if not seed_part.startswith("seed"):
        raise ValueError(f"malformed experiment id: {experiment_id_str}")
    return {"architecture": architecture, "dataset_variant_id": dataset_variant_id, "seed": int(seed_part[len("seed"):])}


def run_dir(root: Path, architecture: str, dataset_variant_id: str, training_policy: str, seed: int) -> Path:
    return root / EXPERIMENTS_ROOT / architecture / dataset_variant_id / training_policy / f"seed_{seed}"


def results_dir(root: Path, architecture: str, dataset_variant_id: str, training_policy: str, seed: int) -> Path:
    return root / RESULTS_ROOT / architecture / dataset_variant_id / training_policy / f"seed_{seed}"


def checkpoint_path(run: Path, framework: str) -> Path:
    return run / ("model.keras" if framework == "tensorflow_keras" else "model.pt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_signature(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_checkpoint_metadata(run: Path, *, architecture: str, dataset_variant_id: str, training_policy: str,
                               seed: int, checkpoint: Path, dataset_manifest_sha256: str, protocol_signature: str) -> Path:
    run.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1, "architecture": architecture, "dataset_variant_id": dataset_variant_id,
        "training_policy": training_policy, "seed": seed,
        "checkpoint_signature": checkpoint_signature(checkpoint),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "protocol_signature": protocol_signature,
    }
    out = run / "checkpoint_metadata.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    return out


def read_checkpoint_metadata(run: Path) -> dict | None:
    path = run / "checkpoint_metadata.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def checkpoint_is_verified(run: Path, framework: str) -> tuple[bool, str]:
    meta = read_checkpoint_metadata(run)
    if meta is None:
        return False, "no checkpoint_metadata.json"
    recorded = meta.get("checkpoint_signature")
    if not recorded:
        return False, "checkpoint_metadata.json has no signature"
    ckpt = checkpoint_path(run, framework)
    current = checkpoint_signature(ckpt)
    if current is None:
        return False, f"checkpoint file missing: {ckpt}"
    if current["sha256"] != recorded["sha256"] or current["size_bytes"] != recorded["size_bytes"]:
        return False, "checkpoint file changed since metadata was written (incompatible)"
    return True, "verified"


def legacy_alias_for(dataset_variant: dict, classifier_registry: dict, architecture_display_name: str) -> str | None:
    """Return the legacy experiment_id (if any) in final_classifier_registry.json whose
    checkpoint could satisfy this (architecture, dataset_variant) pair, so the runner can skip
    retraining a combination that already has a verified legacy checkpoint. Never invents a
    match: only looks at dataset_variant["legacy_experiment_ids"], populated exclusively by
    dataset_variant_registry.py from real registry cross-references.
    """
    candidates = dataset_variant.get("legacy_experiment_ids", [])
    by_id = {e["experiment_id"]: e for e in classifier_registry.get("experiments", [])}
    for exp_id in candidates:
        entry = by_id.get(exp_id)
        if entry and entry.get("architecture") == architecture_display_name:
            return exp_id
    return None
