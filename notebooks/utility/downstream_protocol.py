"""Small, dependency-light contracts for the notebook-driven 24-job design."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

NAMESPACE = "mammodiffusion.downstream_validation.v2"
ARCHITECTURES = ("maxvit512", "mammofm")
CONDITIONS = ("real_only", "real_augmented", "real_plus_best_finetuned_positive",
              "real_plus_best_fromscratch_positive")
SEEDS = (17, 42, 73)


def atomic_json(path: Path, value: Any) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def load_protocol(root: Path) -> dict[str, Any]:
    payload = json.loads((Path(root) / "configs/downstream_classifier_protocol.json").read_text())
    if payload.get("pipeline_namespace") != NAMESPACE: raise ValueError("unsupported downstream protocol")
    for architecture in ARCHITECTURES:
        policy = payload["architectures"][architecture]
        if policy["scheduler_params"]["monitor"] != "val_pr_auc": raise ValueError("scheduler must monitor val_pr_auc")
        if policy["early_stopping"]["monitor"] != "val_pr_auc": raise ValueError("early stopping must monitor val_pr_auc")
        if not policy["checkpoint_criterion"].startswith("val_pr_auc_max"): raise ValueError("checkpoint must use val_pr_auc")
    return payload


def experiment_id(architecture: str, condition: str, seed: int) -> str:
    return f"{architecture}__{condition}__seed{int(seed)}"


def parse_experiment_id(value: str) -> dict[str, Any]:
    architecture, condition, seed = value.split("__")
    if not seed.startswith("seed"): raise ValueError(value)
    result = {"architecture": architecture, "condition": condition, "seed": int(seed[4:])}
    if experiment_id(**result) != value: raise ValueError(value)
    return result


def validate_jobs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("pipeline_namespace") != NAMESPACE: raise ValueError("unsupported downstream job namespace")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 24: raise ValueError("primary downstream matrix must contain 24 jobs")
    keys = []
    for job in jobs:
        key = job.get("architecture"), job.get("condition"), int(job.get("seed", -1))
        if key[0] not in ARCHITECTURES or key[1] not in CONDITIONS or key[2] not in SEEDS: raise ValueError(f"invalid job: {job}")
        if job.get("experiment_id") != experiment_id(*key): raise ValueError("non-canonical experiment ID")
        keys.append(key)
    expected = {(a, c, s) for a in ARCHITECTURES for c in CONDITIONS for s in SEEDS}
    if len(set(keys)) != 24 or set(keys) != expected: raise ValueError("jobs must cover exact 2 x 4 x 3 design")
    return jobs


def load_jobs(root: Path) -> dict[str, Any]:
    payload = json.loads((Path(root) / "configs/downstream_classifier_jobs.json").read_text())
    validate_jobs(payload)
    return payload


def load_selected_generators(root: Path, *, required: bool = True) -> dict[str, Any] | None:
    path = Path(root) / "configs/selected_generators.json"
    if not path.is_file():
        if required: raise FileNotFoundError("run notebook 06 and save configs/selected_generators.json")
        return None
    payload = json.loads(path.read_text())
    try:
        from .generator_benchmark import load_registry
    except ImportError:
        from generator_benchmark import load_registry
    by_id = {entry["id"]: entry for entry in load_registry(root)["generators"]}
    for family, generator_id in (("finetuned", payload.get("finetuned")), ("from_scratch", payload.get("from_scratch"))):
        entry = by_id.get(generator_id)
        if not entry or entry.get("scientific_family") != family: raise ValueError(f"invalid selected {family} generator")
        if not entry.get("eligible_for_downstream_selection", False): raise ValueError(f"selected {generator_id} is not eligible")
    return payload


def resolve_condition(root: Path, condition: str) -> dict[str, Any]:
    protocol = load_protocol(root)
    if condition not in CONDITIONS: raise ValueError(f"unknown downstream condition: {condition}")
    definition = dict(protocol["conditions"][condition]); family = definition.pop("synthetic_family", None)
    generator_id = load_selected_generators(root)[family] if family else None
    return {"dataset_variant_id": condition, "condition": condition, "status": "ready",
            "real_source": bool(definition.get("real_source")),
            "augmentation_source": bool(definition.get("augmentation_source")),
            "augmentation_classes": definition.get("augmentation_classes", []),
            "synthetic_generator_id": generator_id,
            "synthetic_count_by_class": {"positive": int(definition.get("synthetic_positive_count", 0))} if generator_id else {},
            "seed": 20260714, "selected_generators_path": "configs/selected_generators.json" if generator_id else None}


def resolve_job(root: Path, architecture: str, condition: str, seed: int) -> dict[str, Any]:
    requested = experiment_id(architecture, condition, seed)
    if not any(job["experiment_id"] == requested for job in load_jobs(root)["jobs"]):
        raise ValueError(f"job is not part of the primary protocol: {requested}")
    policy = dict(load_protocol(root)["architectures"][architecture]); policy["seeds"] = list(SEEDS)
    return {"experiment_id": requested, "variant": resolve_condition(root, condition), "policy": policy,
            "training_policy_name": f"{architecture}_fixed_protocol"}


def logical_experiments() -> list[dict[str, Any]]:
    return [{"experiment_id": experiment_id(a, c, s), "architecture": a, "condition": c, "seed": s}
            for a in ARCHITECTURES for c in CONDITIONS for s in SEEDS]


__all__ = ["ARCHITECTURES", "CONDITIONS", "NAMESPACE", "SEEDS", "atomic_json", "experiment_id",
           "load_jobs", "load_protocol", "load_selected_generators", "logical_experiments", "parse_experiment_id",
           "resolve_condition", "resolve_job", "validate_jobs"]
