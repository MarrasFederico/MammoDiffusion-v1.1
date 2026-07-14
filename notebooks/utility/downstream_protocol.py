"""Dependency-light contracts for the 24-job downstream validation and locked test."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

NAMESPACE = "mammodiffusion.downstream_validation.v1"
ARCHITECTURES = ("maxvit512", "mammofm")
CONDITIONS = (
    "real_only",
    "real_augmented",
    "real_plus_best_finetuned_positive",
    "real_plus_best_fromscratch_positive",
)
SEEDS = (17, 42, 73)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def value_signature(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def signed_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("signature", None)
    return {**unsigned, "signature": value_signature(unsigned)}


def verify_signed_payload(value: Mapping[str, Any], artifact_type: str | None = None) -> None:
    signature = value.get("signature")
    unsigned = {key: item for key, item in value.items() if key != "signature"}
    if not signature or signature != value_signature(unsigned):
        raise ValueError("artifact signature is missing or invalid")
    if artifact_type and value.get("artifact_type") != artifact_type:
        raise ValueError(f"expected {artifact_type}, got {value.get('artifact_type')}")


def atomic_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def code_revision(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unavailable"


def load_protocol(root: Path) -> dict[str, Any]:
    payload = json.loads((Path(root) / "configs/downstream_classifier_protocol.json").read_text())
    if payload.get("pipeline_namespace") != NAMESPACE:
        raise ValueError("unsupported downstream protocol")
    return payload


def load_jobs(root: Path) -> dict[str, Any]:
    payload = json.loads((Path(root) / "configs/downstream_classifier_jobs.json").read_text())
    validate_jobs(payload)
    return payload


def validate_jobs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("pipeline_namespace") != NAMESPACE:
        raise ValueError("unsupported downstream job namespace")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 24:
        raise ValueError("primary downstream matrix must contain exactly 24 jobs")
    ids, keys = set(), set()
    for job in jobs:
        architecture, condition, seed = job.get("architecture"), job.get("condition"), int(job.get("seed", -1))
        expected_id = experiment_id(architecture, condition, seed)
        if architecture not in ARCHITECTURES or condition not in CONDITIONS or seed not in SEEDS:
            raise ValueError(f"invalid downstream job: {job}")
        if job.get("experiment_id") != expected_id:
            raise ValueError(f"non-canonical experiment ID: {job.get('experiment_id')}")
        key = architecture, condition, seed
        if expected_id in ids or key in keys:
            raise ValueError(f"duplicate downstream job: {expected_id}")
        ids.add(expected_id); keys.add(key)
    expected = {(architecture, condition, seed) for architecture in ARCHITECTURES
                for condition in CONDITIONS for seed in SEEDS}
    if keys != expected:
        raise ValueError("downstream jobs do not cover the exact 2 x 4 x 3 design")
    return jobs


def experiment_id(architecture: str, condition: str, seed: int) -> str:
    return f"{architecture}__{condition}__seed{int(seed)}"


def parse_experiment_id(value: str) -> dict[str, Any]:
    architecture, condition, seed = value.split("__")
    if not seed.startswith("seed"):
        raise ValueError(value)
    result = {"architecture": architecture, "condition": condition, "seed": int(seed[4:])}
    if experiment_id(**result) != value:
        raise ValueError(value)
    return result


def load_approval(root: Path, *, required: bool = True) -> dict[str, Any] | None:
    path = Path(root) / "configs/approved_generators.json"
    if not path.is_file():
        if required:
            raise FileNotFoundError("configs/approved_generators.json is required for synthetic conditions")
        return None
    payload = json.loads(path.read_text())
    verify_signed_payload(payload, "approved_generator_selection")
    if payload.get("pipeline_namespace") != NAMESPACE or payload.get("test_access") is not False:
        raise ValueError("approved generator selection has an invalid namespace or test-access flag")
    manifest_path = Path(root) / "results/generator_benchmark/generator_benchmark_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("approved selection requires its generator benchmark manifest")
    manifest = json.loads(manifest_path.read_text())
    from generator_benchmark import verify_signature  # local import avoids a hard metric dependency
    verify_signature(manifest)
    if payload.get("benchmark_manifest_signature") != manifest.get("signature"):
        raise ValueError("approved generator selection is stale for the current benchmark manifest")
    return payload


def resolve_condition(root: Path, condition: str) -> dict[str, Any]:
    protocol = load_protocol(root)
    if condition not in CONDITIONS:
        raise ValueError(f"unknown downstream condition: {condition}")
    definition = dict(protocol["conditions"][condition])
    family = definition.pop("synthetic_family", None)
    generator_id = None
    approval_signature = None
    if family:
        approval = load_approval(root, required=True)
        key = "best_finetuned_generator" if family == "finetuned" else "best_from_scratch_generator"
        generator_id = approval[key]
        approval_signature = approval["signature"]
    return {
        "dataset_variant_id": condition,
        "condition": condition,
        "status": "ready",
        "real_source": bool(definition.get("real_source")),
        "augmentation_source": bool(definition.get("augmentation_source")),
        "augmentation_classes": definition.get("augmentation_classes", []),
        "synthetic_generator_id": generator_id,
        "synthetic_count_by_class": {"positive": int(definition.get("synthetic_positive_count", 0))} if generator_id else {},
        "seed": 20260714,
        "approved_generator_signature": approval_signature,
    }


def resolve_job(root: Path, architecture: str, condition: str, seed: int) -> dict[str, Any]:
    jobs = load_jobs(root)["jobs"]
    requested = experiment_id(architecture, condition, seed)
    if not any(job["experiment_id"] == requested for job in jobs):
        raise ValueError(f"job is not part of the primary protocol: {requested}")
    policy = dict(load_protocol(root)["architectures"][architecture])
    policy["seeds"] = list(SEEDS)
    variant = resolve_condition(root, condition)
    return {"experiment_id": requested, "variant": variant, "policy": policy,
            "training_policy_name": f"{architecture}_fixed_protocol"}


def approval_payload(root: Path, proposal: Mapping[str, Any]) -> dict[str, Any]:
    from generator_benchmark import load_registry, verify_signature
    verify_signature(proposal)
    if proposal.get("artifact_type") != "generator_selection_proposal" or proposal.get("test_access") is not False:
        raise ValueError("proposal is not a validation-only generator selection proposal")
    registry = load_registry(root)
    by_id = {entry["id"]: entry for entry in registry["generators"]}
    fine = proposal.get("best_finetuned_generator")
    scratch = proposal.get("best_from_scratch_generator")
    if by_id.get(fine, {}).get("scientific_family") != "finetuned":
        raise ValueError("proposed fine-tuned winner has the wrong family")
    if by_id.get(scratch, {}).get("scientific_family") != "from_scratch":
        raise ValueError("proposed from-scratch winner has the wrong family")
    return signed_payload({
        "schema_version": 1, "pipeline_namespace": NAMESPACE,
        "artifact_type": "approved_generator_selection",
        "best_finetuned_generator": fine,
        "best_from_scratch_generator": scratch,
        "benchmark_manifest_signature": proposal["benchmark_manifest_signature"],
        "proposal_signature": proposal["signature"],
        "selection_rationale": proposal.get("selection_rationale"),
        "approval_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_revision": code_revision(root),
        "test_access": False,
    })


__all__ = [
    "ARCHITECTURES", "CONDITIONS", "NAMESPACE", "SEEDS", "approval_payload", "atomic_json", "code_revision",
    "experiment_id", "load_approval", "load_jobs", "load_protocol", "parse_experiment_id", "resolve_condition",
    "resolve_job", "signed_payload", "validate_jobs", "value_signature", "verify_signed_payload",
]
