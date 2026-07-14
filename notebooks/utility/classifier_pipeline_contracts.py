"""Shared, dependency-free contracts for the classifier-matrix v2 lifecycle.

This module is intentionally safe to import in static preflight and status commands: it never
imports a ML framework, opens the locked test split, or creates runtime directories.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

PIPELINE_NAMESPACE = "mammodiffusion.classifier_matrix.v2"
CONTRACT_SCHEMA_VERSION = 2
REQUIRED_SEEDS = (17, 42, 73)
ARCHITECTURES = ("resnet50", "maxvit512", "mammofm", "raddino")

STATES = (
    "PENDING", "ADMITTED", "CLAIMED", "RUNNING", "INTERRUPTED_RESUMABLE",
    "FAILED_RETRYABLE", "FAILED_FINAL", "TRAINED", "VALIDATING", "VALIDATED",
    "ENSEMBLED", "ENSEMBLE_READY", "COMPLETE", "BLOCKED", "INVALIDATED",
)

_ALIASES = {"CLAIMED": "ADMITTED", "ENSEMBLE_READY": "ENSEMBLED"}
_TRANSITIONS = {
    "PENDING": {"ADMITTED", "RUNNING", "BLOCKED", "FAILED_FINAL"},
    "ADMITTED": {"RUNNING", "FAILED_RETRYABLE", "FAILED_FINAL"},
    "RUNNING": {"INTERRUPTED_RESUMABLE", "FAILED_RETRYABLE", "FAILED_FINAL", "TRAINED"},
    "INTERRUPTED_RESUMABLE": {"ADMITTED", "RUNNING", "FAILED_FINAL"},
    # Retry targets are phase-aware: RUNNING is the training retry, VALIDATING and ENSEMBLED
    # are accepted only by their artifact-guarded runner functions.
    "FAILED_RETRYABLE": {"ADMITTED", "RUNNING", "VALIDATING", "ENSEMBLED", "FAILED_FINAL"},
    "TRAINED": {"VALIDATING", "VALIDATED", "FAILED_RETRYABLE", "FAILED_FINAL"},
    "VALIDATING": {"VALIDATED", "FAILED_RETRYABLE", "FAILED_FINAL"},
    "VALIDATED": {"ENSEMBLED", "FAILED_RETRYABLE", "FAILED_FINAL"},
    "ENSEMBLED": {"COMPLETE", "FAILED_RETRYABLE", "FAILED_FINAL"},
    "COMPLETE": set(), "BLOCKED": set(), "FAILED_FINAL": set(), "INVALIDATED": set(),
}


def canonical_state(state: str) -> str:
    if state not in STATES:
        raise ValueError(f"unknown classifier state: {state}")
    return _ALIASES.get(state, state)


def transition_allowed(previous: str, target: str, *, explicit_reset: bool = False) -> bool:
    previous_c, target_c = canonical_state(previous), canonical_state(target)
    if previous_c == target_c:
        return True
    if explicit_reset and previous_c not in {"ADMITTED", "RUNNING", "VALIDATING"} and target_c == "PENDING":
        return True
    return target_c in _TRANSITIONS[previous_c]


def require_transition(previous: str, target: str, *, explicit_reset: bool = False) -> None:
    if not transition_allowed(previous, target, explicit_reset=explicit_reset):
        raise ValueError(f"invalid classifier state transition: {previous} -> {target}")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def value_signature(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def signed_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("signature", None)
    return {**unsigned, "signature": value_signature(unsigned)}


def verify_signed_payload(payload: Mapping[str, Any], *, namespace: str = PIPELINE_NAMESPACE,
                          schema_version: int = CONTRACT_SCHEMA_VERSION) -> None:
    if payload.get("pipeline_namespace") != namespace:
        raise ValueError("artifact is not in the classifier-matrix v2 namespace")
    if int(payload.get("schema_version", -1)) != schema_version:
        raise ValueError(f"unsupported classifier artifact schema: {payload.get('schema_version')}")
    signature = payload.get("signature")
    if not signature:
        raise ValueError("artifact is unsigned")
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    if signature != value_signature(unsigned):
        raise ValueError("artifact content signature mismatch")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, root: Path) -> dict[str, Any]:
    resolved, project = Path(path).resolve(), Path(root).resolve()
    try:
        relative = resolved.relative_to(project).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact path is outside project root: {resolved}") from exc
    if not resolved.is_file() or not os.access(resolved, os.R_OK):
        raise FileNotFoundError(f"artifact is missing or unreadable: {resolved}")
    return {"relative_path": relative, "size_bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def code_revision(root: Path) -> str:
    override = os.environ.get("MAMMODIFFUSION_CODE_REVISION")
    if override:
        return override
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unavailable"


def atomic_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def validate_matrix(payload: Mapping[str, Any], *, expected_stage1_jobs: int | None = None) -> list[dict]:
    if int(payload.get("schema_version", -1)) not in (1, 2):
        raise ValueError("unsupported classifier matrix schema")
    if payload.get("pipeline_namespace") not in (None, PIPELINE_NAMESPACE):
        raise ValueError("legacy/foreign matrix cannot be used by classifier-matrix v2")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("classifier matrix jobs must be a list")
    ids, scientific_keys = set(), set()
    for job in jobs:
        required = {"experiment_id", "stage", "architecture", "dataset_variant_id", "training_policy", "seed", "status"}
        missing = required - set(job)
        if missing:
            raise ValueError(f"matrix job lacks fields: {sorted(missing)}")
        if job["architecture"] not in ARCHITECTURES or int(job["seed"]) not in REQUIRED_SEEDS:
            raise ValueError(f"invalid architecture/seed in {job['experiment_id']}")
        if job["experiment_id"] in ids:
            raise ValueError(f"duplicate experiment_id: {job['experiment_id']}")
        key = (int(job["stage"]), job["architecture"], job["dataset_variant_id"], int(job["seed"]))
        if key in scientific_keys:
            raise ValueError(f"duplicate scientific job key: {key}")
        canonical_state(job["status"])
        ids.add(job["experiment_id"]); scientific_keys.add(key)
    if expected_stage1_jobs is not None and sum(int(job["stage"]) == 1 for job in jobs) != expected_stage1_jobs:
        raise ValueError(f"Stage 1 matrix must contain exactly {expected_stage1_jobs} jobs")
    return jobs


def require_v2_artifact(payload: Mapping[str, Any], *, artifact_type: str) -> None:
    verify_signed_payload(payload)
    if payload.get("artifact_type") != artifact_type:
        raise ValueError(f"expected {artifact_type}, got {payload.get('artifact_type')}")


__all__ = [
    "ARCHITECTURES", "CONTRACT_SCHEMA_VERSION", "PIPELINE_NAMESPACE", "REQUIRED_SEEDS", "STATES",
    "atomic_json", "canonical_json", "canonical_state", "code_revision", "file_identity",
    "require_transition", "require_v2_artifact", "sha256_file", "signed_payload", "transition_allowed",
    "validate_matrix", "value_signature", "verify_signed_payload",
]
