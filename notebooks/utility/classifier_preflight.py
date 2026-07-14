"""Read-only Stage-1 preflight; safe before GPU initialization and scientific execution."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from classifier_gpu_gate import validate_gate
from classifier_pipeline_contracts import ARCHITECTURES, PIPELINE_NAMESPACE, REQUIRED_SEEDS, validate_matrix, value_signature

EXPECTED_STAGE1_NOTEBOOKS = 112
EXPECTED_READY_NOTEBOOKS = 100
EXPECTED_BLOCKED_NOTEBOOKS = 12
EXPECTED_EXECUTABLE_VARIANTS = 25
EXPECTED_STAGE1_JOBS = 300

REQUIRED_SCRIPTS = (
    "scripts/preflight_classifier_pipeline.py",
    "scripts/profile_classifier_vram.py",
    "scripts/run_classifier_gpu_smokes.py",
    "scripts/run_classifier_static_tests.py",
    "scripts/run_classifier_experiment_matrix.py",
    "scripts/resume_classifier_experiment_matrix.py",
    "scripts/status_classifier_pipeline.py",
    "scripts/finalize_validation_stage.py",
    "scripts/create_classifier_stage2_notebooks.py",
    "scripts/build_classifier_experiment_matrix.py",
    "scripts/finalize_locked_test_stage.py",
    "scripts/run_locked_classifier_inference.py",
    "scripts/finalize_classifier_report.py",
)


def _load_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        errors.append(f"unreadable JSON {path}: {type(exc).__name__}: {exc}")
        return None


def _dataset_audit(root: Path, variants: list[dict], errors: list[str]) -> dict[str, dict]:
    import classifier_dataset_builder as datasets
    signatures = {}
    for variant in variants:
        try:
            train, validation, manifest = datasets.build_training_and_validation_rows(root, dict(variant))
            if not train or not validation:
                raise ValueError("train and validation must both be non-empty")
            if not manifest.get("training_signature") or not manifest.get("validation_signature"):
                raise ValueError("content-aware train/validation signatures are missing")
            recorded = manifest.get("manifest_signature")
            if not recorded or recorded != value_signature({key: value for key, value in manifest.items()
                                                             if key != "manifest_signature"}):
                raise ValueError("dataset manifest signature is missing or invalid")
            if any("test" in str(row.get("source", "")).lower() for row in train + validation):
                raise ValueError("test leakage detected in train/validation rows")
            signatures[variant["dataset_variant_id"]] = {
                "training_signature": manifest["training_signature"],
                "validation_signature": manifest["validation_signature"],
            }
        except Exception as exc:
            errors.append(f"dataset {variant.get('dataset_variant_id')}: {type(exc).__name__}: {exc}")
    return signatures


def build_preflight(root: Path, *, deep_dataset_check: bool = True) -> dict[str, Any]:
    root = Path(root)
    errors, warnings = [], []
    inventory_path = root / "results/notebook_inventory/notebook_inventory.json"
    matrix_path = root / "configs/classifier_experiment_matrix.json"
    registry_path = root / "configs/dataset_variant_registry.json"
    protocols_path = root / "configs/classifier_training_protocols.json"
    inventory = _load_json(inventory_path, errors) or []
    matrix = _load_json(matrix_path, errors) or {}
    registry = _load_json(registry_path, errors) or {}
    protocols = _load_json(protocols_path, errors) or {}
    for name, payload in (("matrix", matrix), ("dataset registry", registry), ("training protocols", protocols)):
        if payload.get("pipeline_namespace") != PIPELINE_NAMESPACE or int(payload.get("schema_version", -1)) != 2:
            errors.append(f"{name} is not a classifier-matrix v2 schema-2 artifact")

    stage1_inventory = [row for row in inventory if int(row.get("stage", -1)) == 1]
    notebook_status = Counter(row.get("dataset_status") for row in stage1_inventory)
    if len(stage1_inventory) != EXPECTED_STAGE1_NOTEBOOKS:
        errors.append(f"expected {EXPECTED_STAGE1_NOTEBOOKS} Stage 1 notebooks, got {len(stage1_inventory)}")
    if notebook_status != Counter({"READY": EXPECTED_READY_NOTEBOOKS, "BLOCKED": EXPECTED_BLOCKED_NOTEBOOKS}):
        errors.append(f"expected 100 READY/12 BLOCKED notebooks, got {dict(notebook_status)}")
    for row in stage1_inventory:
        path = root / str(row.get("path", ""))
        if not path.is_file(): errors.append(f"missing notebook: {row.get('path')}")
        if row.get("dataset_status") == "BLOCKED" and not str(row.get("note_blocker") or "").strip():
            errors.append(f"blocked notebook lacks a reason: {row.get('path')}")

    variants = registry.get("variants", []) if isinstance(registry, dict) else []
    executable = [variant for variant in variants if variant.get("status") in ("ready", "legacy") and
                  variant.get("regime") in {"base", "stage1_screening", "legacy_compatible"}]
    blocked = [variant for variant in variants if variant.get("status") == "blocked"]
    if len(executable) != EXPECTED_EXECUTABLE_VARIANTS:
        errors.append(f"expected {EXPECTED_EXECUTABLE_VARIANTS} executable Stage 1 variants, got {len(executable)}")
    if any(not str(variant.get("blocker") or variant.get("invalid_reason") or "").strip() for variant in blocked):
        errors.append("every blocked dataset variant must have an explicit scientific blocker")

    try:
        jobs = validate_matrix(matrix, expected_stage1_jobs=EXPECTED_STAGE1_JOBS)
    except Exception as exc:
        errors.append(f"invalid classifier matrix: {exc}"); jobs = []
    stage1_jobs = [job for job in jobs if int(job["stage"]) == 1]
    expected_keys = {(architecture, variant["dataset_variant_id"], seed) for architecture in ARCHITECTURES
                     for variant in executable for seed in REQUIRED_SEEDS}
    actual_keys = {(job["architecture"], job["dataset_variant_id"], int(job["seed"])) for job in stage1_jobs}
    if actual_keys != expected_keys:
        errors.append("Stage 1 matrix does not exactly cover architecture × executable variant × seed")
    if set((protocols.get("policies") or {})) != set(ARCHITECTURES):
        errors.append("training protocols must define exactly the four registered architectures")
    else:
        import classifier_checkpoint_io as ckio
        for job in stage1_jobs:
            run = ckio.run_dir(root, job["architecture"], job["dataset_variant_id"],
                               job["training_policy"], int(job["seed"]))
            checkpoint = ckio.checkpoint_path(run, protocols["policies"][job["architecture"]]["framework"])
            metadata = run / "checkpoint_metadata.json"
            if checkpoint.exists() or metadata.exists():
                verified, reason = ckio.checkpoint_is_verified(run,
                    protocols["policies"][job["architecture"]]["framework"], {
                        "architecture": job["architecture"], "dataset_variant_id": job["dataset_variant_id"],
                        "training_policy": job["training_policy"], "seed": int(job["seed"])})
                if not verified:
                    errors.append(f"incompatible checkpoint artifact for {job['experiment_id']}: {reason}")
    for relative in REQUIRED_SCRIPTS:
        if not (root / relative).is_file(): errors.append(f"missing required command: {relative}")

    dataset_signatures = _dataset_audit(root, executable, errors) if deep_dataset_check and executable else {}
    if not deep_dataset_check:
        warnings.append("deep dataset/path/signature check was not requested")

    gate = validate_gate(root)
    if errors:
        readiness = "BLOCKED"
    elif not gate["profiles_valid"]:
        readiness = "READY_TO_PROFILE_GPU"
    elif not gate["smokes_valid"]:
        readiness = "READY_TO_DRY_RUN"
    else:
        readiness = "READY_TO_RUN_STAGE1"
    return {
        "schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE,
        "artifact_type": "classifier_static_preflight_report", "read_only": True, "readiness": readiness,
        "counts": {"stage1_notebooks": len(stage1_inventory), "ready_notebooks": notebook_status.get("READY", 0),
                   "blocked_notebooks": notebook_status.get("BLOCKED", 0), "executable_variants": len(executable),
                   "stage1_jobs": len(stage1_jobs)},
        "dataset_signatures": dataset_signatures, "gpu_gate": gate,
        "errors": sorted(errors), "warnings": sorted(warnings),
        "blocked_variants": [{"dataset_variant_id": row.get("dataset_variant_id"),
                              "reason": row.get("blocker") or row.get("invalid_reason")} for row in blocked],
    }


def format_preflight(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [f"classifier pipeline preflight: {report['readiness']}",
             f"notebooks: {counts['stage1_notebooks']} ({counts['ready_notebooks']} READY / {counts['blocked_notebooks']} BLOCKED)",
             f"Stage 1: {counts['executable_variants']} variants / {counts['stage1_jobs']} jobs",
             f"GPU profiles: {'PASS' if report['gpu_gate']['profiles_valid'] else 'PENDING'}",
             f"GPU smokes: {'PASS' if report['gpu_gate']['smokes_valid'] else 'PENDING'}"]
    lines.extend(f"BLOCKER: {error}" for error in report["errors"])
    lines.extend(f"WARNING: {warning}" for warning in report["warnings"])
    return "\n".join(lines)


__all__ = ["build_preflight", "format_preflight"]
