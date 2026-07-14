"""Checkpoint/resume provenance for one compact downstream experiment."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

from classifier_pipeline_contracts import PIPELINE_NAMESPACE, atomic_json, value_signature

EXPERIMENTS_ROOT = "experiments/downstream_classifiers"
RESULTS_ROOT = "results/downstream_classifiers"
RESUME_NAMES = ("checkpoint_latest", "checkpoint_previous", "checkpoint_best")


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


def resume_checkpoint_path(run: Path, name: str = "checkpoint_latest") -> Path:
    if name not in RESUME_NAMES:
        raise ValueError(f"unknown resume checkpoint: {name}")
    return run / f"{name}.pkl"


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_pickle(path: Path, payload: dict) -> Path:
    """Durable temporary-file -> fsync -> atomic rename checkpoint write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    _fsync_directory(path.parent)
    return path


def read_resume_checkpoint(path: Path) -> dict:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("unsupported resume checkpoint schema")
    return payload


def save_resume_checkpoint(run: Path, payload: dict, *, best: bool = False) -> Path:
    """Rotate a valid latest; quarantine a corrupt one before publishing the replacement."""
    latest = resume_checkpoint_path(run)
    previous = resume_checkpoint_path(run, "checkpoint_previous")
    if latest.is_file():
        try:
            read_resume_checkpoint(latest)
        except Exception:
            corrupt = latest.with_name(latest.name + ".corrupt")
            if corrupt.exists(): corrupt.unlink()
            os.replace(latest, corrupt)
        else:
            os.replace(latest, previous)
    normalized = {"schema_version": 2, **payload}
    atomic_pickle(latest, normalized)
    if best:
        atomic_pickle(resume_checkpoint_path(run, "checkpoint_best"), normalized)
    return latest


def load_resume_checkpoint(run: Path, expected: dict) -> tuple[dict | None, str]:
    """Load latest, then previous; reject scientific-provenance mismatches."""
    errors = []
    for name in ("checkpoint_latest", "checkpoint_previous"):
        path = resume_checkpoint_path(run, name)
        if not path.is_file():
            continue
        try:
            payload = read_resume_checkpoint(path)
            mismatches = {key: (payload.get(key), value) for key, value in expected.items()
                          if payload.get(key) != value}
            if mismatches:
                errors.append(f"{name}: incompatible {mismatches}")
                continue
            return payload, name
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return None, "; ".join(errors) if errors else "no resume checkpoint"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_signature(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_checkpoint_metadata(run: Path, *, architecture: str, dataset_variant_id: str, training_policy: str,
                               seed: int, checkpoint: Path, dataset_manifest_sha256: str, protocol_signature: str) -> Path:
    run.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE,
        "artifact_type": "classifier_final_checkpoint", "architecture": architecture, "dataset_variant_id": dataset_variant_id,
        "training_policy": training_policy, "seed": seed,
        "checkpoint_signature": checkpoint_signature(checkpoint),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "protocol_signature": protocol_signature,
    }
    payload["metadata_signature"] = value_signature(payload)
    out = run / "checkpoint_metadata.json"
    atomic_json(out, payload)
    return out


def read_checkpoint_metadata(run: Path) -> dict | None:
    path = run / "checkpoint_metadata.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def checkpoint_is_verified(run: Path, framework: str, expected: dict | None = None) -> tuple[bool, str]:
    meta = read_checkpoint_metadata(run)
    if meta is None:
        return False, "no checkpoint_metadata.json"
    if meta.get("schema_version") != 2 or meta.get("pipeline_namespace") != PIPELINE_NAMESPACE:
        return False, "checkpoint metadata schema/namespace is incompatible"
    recorded_metadata_signature = meta.get("metadata_signature")
    unsigned_meta = {key: value for key, value in meta.items() if key != "metadata_signature"}
    if not recorded_metadata_signature or recorded_metadata_signature != value_signature(unsigned_meta):
        return False, "checkpoint metadata is unsigned or changed"
    if expected:
        mismatches = {key: (meta.get(key), value) for key, value in expected.items() if meta.get(key) != value}
        if mismatches:
            return False, f"checkpoint metadata belongs to another scientific job: {mismatches}"
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
