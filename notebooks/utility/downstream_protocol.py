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


SUPPORTED_SELECTION_SCHEMA_VERSIONS = (1, 2)
EXPECTED_FILTERED_IMAGE_COUNT = 1361


def _load_registry(root: Path) -> dict[str, Any]:
    try:
        from .generator_benchmark import load_registry
    except ImportError:
        from generator_benchmark import load_registry
    return load_registry(root)


def _file_sha256(path: Path) -> str:
    try:
        from .generator_benchmark import file_sha256
    except ImportError:
        from generator_benchmark import file_sha256
    return file_sha256(path)


def load_selected_generators(root: Path, *, required: bool = True) -> dict[str, Any] | None:
    """Load and lightly validate the selection file (schema, test access, registry role and family).

    Deeper content binding (amendment / benchmark / provenance hashes) is applied by
    ``validate_selection_content`` when a synthetic condition actually requires the selection.
    """
    path = Path(root) / "configs/selected_generators.json"
    if not path.is_file():
        if required: raise FileNotFoundError("run notebook 06 and save configs/selected_generators.json")
        return None
    payload = json.loads(path.read_text())
    schema_version = int(payload.get("schema_version", 1))
    if schema_version not in SUPPORTED_SELECTION_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported selected_generators schema_version: {schema_version}")
    if payload.get("test_access", False):
        raise ValueError("selected_generators.json must declare test_access = false")
    by_id = {entry["id"]: entry for entry in _load_registry(root)["generators"]}
    for family, generator_id in (("finetuned", payload.get("finetuned")), ("from_scratch", payload.get("from_scratch"))):
        entry = by_id.get(generator_id)
        if not entry or entry.get("scientific_family") != family: raise ValueError(f"invalid selected {family} generator")
        if not entry.get("eligible_for_downstream_selection", False): raise ValueError(f"selected {generator_id} is not eligible")
    return payload


def _require_content_hash(root: Path, relative: str | None, expected: str | None, label: str) -> None:
    if not relative or not expected:
        raise ValueError(f"selection is missing the {label} content binding")
    path = Path(root) / str(relative)
    if not path.is_file():
        raise ValueError(f"selection {label} content is missing: {relative}")
    if _file_sha256(path) != expected:
        raise ValueError(f"selection {label} content has changed since selection: {relative}")


def validate_selection_content(root: Path, payload: Mapping[str, Any]) -> None:
    """Content-aware validation used before building any synthetic dataset.

    Verifies that the amendment, benchmark summary, amended gate results and each selected generator's
    model / generation identity and FILTERED manifest still match the content recorded at selection
    time.  Validity depends only on scientific content, not on git state, locks, or approvals.
    """
    import csv
    if int(payload.get("schema_version", 1)) < 2:
        raise ValueError("a synthetic condition requires a content-aware (schema_version >= 2) selection")
    if payload.get("test_access", False):
        raise ValueError("selection declares test_access = true")
    if not payload.get("active_amendment"):
        raise ValueError("selection has no active amendment")
    _require_content_hash(root, payload.get("active_amendment"), payload.get("active_amendment_sha256"), "amendment")
    _require_content_hash(root, payload.get("benchmark_summary_path"), payload.get("benchmark_summary_sha256"),
                          "benchmark summary")
    _require_content_hash(root, payload.get("amended_gate_results_path"), payload.get("amended_gate_results_sha256"),
                          "amended gate results")

    gate_rows = {}
    with (Path(root) / str(payload["amended_gate_results_path"])).open(newline="") as stream:
        for row in csv.DictReader(stream):
            gate_rows[str(row["full_generator_id"])] = row
    by_id = {entry["id"]: entry for entry in _load_registry(root)["generators"]}
    identity = payload.get("selection_identity", {})
    for family in ("finetuned", "from_scratch"):
        generator_id = payload.get(family)
        entry = by_id.get(generator_id)
        if not entry or entry.get("scientific_family") != family:
            raise ValueError(f"invalid selected {family} generator")
        if not entry.get("eligible_for_downstream_selection", False):
            raise ValueError(f"selected {generator_id} is not selection-eligible")
        gate = gate_rows.get(generator_id)
        if not gate or str(gate.get("amended_safety_gate_eligible")).lower() not in ("true", "1"):
            raise ValueError(f"{generator_id} is not amended-safety-gate eligible")
        if str(gate.get("descriptive_family_rank")) != "1":
            raise ValueError(f"{generator_id} is not descriptive_family_rank 1")

        recorded = identity.get(family, {})
        if recorded.get("descriptive_family_rank") != 1:
            raise ValueError(f"{generator_id} recorded rank is not 1")
        if int(recorded.get("filtered_image_count", -1)) != EXPECTED_FILTERED_IMAGE_COUNT:
            raise ValueError(f"{generator_id} recorded FILTERED count is not {EXPECTED_FILTERED_IMAGE_COUNT}")

        provenance = json.loads((Path(root) / entry["provenance_manifest"]).read_text())
        if recorded.get("model_identity_sha256") != provenance.get("model_identity_sha256"):
            raise ValueError(f"{generator_id} model identity changed")
        if recorded.get("generation_identity_sha256") != provenance.get("generation_identity_sha256"):
            raise ValueError(f"{generator_id} generation identity changed")
        if recorded.get("filtered_manifest_path") != provenance.get("filtered_sample_manifest"):
            raise ValueError(f"{generator_id} FILTERED manifest path changed")
        expected_manifest_sha = provenance.get("manifest_sha256", {}).get("filtered_samples")
        if recorded.get("filtered_manifest_sha256") != expected_manifest_sha:
            raise ValueError(f"{generator_id} FILTERED manifest hash changed")
        manifest_path = Path(root) / str(recorded["filtered_manifest_path"])
        if "test" in manifest_path.parts:
            raise ValueError(f"{generator_id} FILTERED manifest references a test path")
        if not manifest_path.is_file() or _file_sha256(manifest_path) != expected_manifest_sha:
            raise ValueError(f"{generator_id} FILTERED manifest content changed")
        with manifest_path.open(newline="") as stream:
            actual_count = sum(1 for _ in csv.DictReader(stream))
        if actual_count != EXPECTED_FILTERED_IMAGE_COUNT:
            raise ValueError(f"{generator_id} FILTERED manifest has {actual_count} records, expected "
                             f"{EXPECTED_FILTERED_IMAGE_COUNT}")


def selection_summary(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Compact, model-free view of the selection for the notebook configuration sections."""
    if not payload:
        return None
    identity = payload.get("selection_identity", {})
    return {
        "active_amendment": payload.get("active_amendment"),
        "post_benchmark_amendment": payload.get("post_benchmark_amendment"),
        "benchmark_run_id": payload.get("benchmark_run_id"),
        "benchmark_HEAD": payload.get("benchmark_HEAD"),
        "test_access": payload.get("test_access"),
        "selected": {family: {
            "generator_id": record.get("generator_id"),
            "descriptive_family_rank": record.get("descriptive_family_rank"),
            "primary_metric": record.get("primary_metric"),
            "primary_metric_value": record.get("primary_metric_value"),
            "model_identity_sha256": record.get("model_identity_sha256"),
            "generation_identity_sha256": record.get("generation_identity_sha256"),
            "filtered_manifest_path": record.get("filtered_manifest_path"),
            "filtered_manifest_sha256": record.get("filtered_manifest_sha256"),
            "filtered_image_count": record.get("filtered_image_count"),
        } for family, record in identity.items()},
    }


def resolve_condition(root: Path, condition: str) -> dict[str, Any]:
    protocol = load_protocol(root)
    if condition not in CONDITIONS: raise ValueError(f"unknown downstream condition: {condition}")
    definition = dict(protocol["conditions"][condition]); family = definition.pop("synthetic_family", None)
    generator_id = None
    if family:
        payload = load_selected_generators(root)
        validate_selection_content(root, payload)  # synthetic conditions require the full content binding
        generator_id = payload[family]
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
           "resolve_condition", "resolve_job", "selection_summary", "validate_jobs", "validate_selection_content"]
