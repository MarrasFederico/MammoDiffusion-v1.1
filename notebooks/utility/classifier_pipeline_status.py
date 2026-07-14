"""Read-only global status for classifier-matrix v2; never opens the locked test dataset."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from classifier_pipeline_contracts import PIPELINE_NAMESPACE, verify_signed_payload
import classifier_checkpoint_io as ckio
import classifier_run_manifest as crm


def _signed_status(path: Path, artifact_type: str | None = None) -> dict:
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    try:
        payload = json.loads(path.read_text()); verify_signed_payload(payload)
        if artifact_type and payload.get("artifact_type") != artifact_type:
            raise ValueError(f"expected {artifact_type}, got {payload.get('artifact_type')}")
        return {"status": "VALID", "path": str(path), "signature": payload["signature"], "payload": payload}
    except Exception as exc:
        return {"status": "INVALID", "path": str(path), "reason": str(exc)}


def build_status(root: Path) -> dict:
    root = Path(root); blockers = []
    inventory_path = root / "results/notebook_inventory/notebook_inventory.json"
    inventory = json.loads(inventory_path.read_text()) if inventory_path.is_file() else []
    notebook_counts = dict(Counter(row.get("dataset_status") for row in inventory if int(row.get("stage", 1)) == 1))

    matrix_path = root / "configs/classifier_experiment_matrix.json"
    matrix = json.loads(matrix_path.read_text()) if matrix_path.is_file() else {"jobs": []}
    jobs = matrix.get("jobs", []); by_stage = {}
    protocols_path = root / "configs/classifier_training_protocols.json"
    protocols = json.loads(protocols_path.read_text()).get("policies", {}) if protocols_path.is_file() else {}
    reconstructed = []
    for job in jobs:
        policy = protocols.get(job.get("architecture"), {})
        framework = policy.get("framework")
        if framework:
            state = crm.reconstruct_state(ckio.run_dir(root, job["architecture"], job["dataset_variant_id"],
                                                       job["training_policy"], int(job["seed"])), framework)["state"]
        else:
            state = job.get("status", "BLOCKED")
        reconstructed.append({**job, "status": state})
    jobs = reconstructed
    ensemble_paths = list((root / "results/classifiers_matrix").glob("*/*/*/ensemble/manifests/ensemble_validation_manifest.json"))
    valid_ensembles = set(); invalid_ensembles = []
    for path in ensemble_paths:
        status = _signed_status(path, "classifier_validation_ensemble")
        if status["status"] == "VALID":
            payload = status["payload"]
            valid_ensembles.add((payload["architecture"], payload["dataset_variant_id"], payload["training_policy"]))
        else:
            invalid_ensembles.append(str(path.relative_to(root)))
    for stage in (1, 2):
        selected = [job for job in jobs if int(job.get("stage", -1)) == stage]
        logical = {(job["architecture"], job["dataset_variant_id"], job["training_policy"]) for job in selected}
        by_stage[str(stage)] = {"jobs_total": len(selected), "jobs_by_state": dict(Counter(job["status"] for job in selected)),
            "ensembles_expected": len(logical), "ensembles_complete": len(logical & valid_ensembles),
            "ensembles_missing": len(logical - valid_ensembles),
            "scientific_completion": bool(logical) and not (logical - valid_ensembles)}

    union = _signed_status(root / "results/generator_comparison/selected_generator_union.json",
                           "classifier_selected_generator_union")
    panel = _signed_status(root / "results/final_evaluation_v2/primary_finalists_manifest.json",
                           "classifier_final_panel_selection")
    lock = _signed_status(root / "results/final_evaluation_v2/EXPERIMENT_MATRIX_LOCKED",
                          "classifier_scientific_lock")
    inference = _signed_status(root / "results/final_evaluation_v2/LOCKED_TEST_COMPLETE",
                               "classifier_locked_inference_completion")
    aggregation = _signed_status(root / "results/final_evaluation_v2/FINAL_AGGREGATION_COMPLETE",
                                 "classifier_final_aggregation_completion")
    try:
        from classifier_gpu_gate import validate_gate
        gpu_gate = validate_gate(root)
        gpu_gate["status"] = "PASS" if gpu_gate.get("ready_for_real_launch") else "BLOCKED"
        gpu_gate["problems"] = list(gpu_gate.get("errors", []))
    except Exception as exc:
        gpu_gate = {"status": "BLOCKED", "problems": [str(exc)]}
    if gpu_gate.get("status") != "PASS": blockers.extend(gpu_gate.get("problems", ["GPU certification incomplete"]))
    if notebook_counts != {"READY": 100, "BLOCKED": 12}:
        blockers.append(f"unexpected Stage 1 notebook inventory: {notebook_counts}")
    if invalid_ensembles: blockers.append(f"invalid v2 ensemble manifests: {len(invalid_ensembles)}")
    if union["status"] != "VALID": blockers.append(f"Stage 1 selected generator union is {union['status'].lower()}")
    if panel["status"] != "VALID": blockers.append(f"Stage 2 panel selection is {panel['status'].lower()}")
    return {"schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE, "read_locked_test": False,
        "notebooks_stage1": notebook_counts, "stages": by_stage,
        "selected_generator_union": {k: v for k, v in union.items() if k != "payload"},
        "panel_selection": {k: v for k, v in panel.items() if k != "payload"},
        "scientific_lock": {k: v for k, v in lock.items() if k != "payload"},
        "locked_inference": {k: v for k, v in inference.items() if k != "payload"},
        "final_aggregation": {k: v for k, v in aggregation.items() if k != "payload"},
        "gpu_certification": gpu_gate, "invalid_ensembles": invalid_ensembles,
        "current_blockers": sorted(set(blockers))}


def format_status(report: dict) -> str:
    lines = ["Classifier pipeline v2 status (locked test not read)",
             f"Stage 1 notebooks: {report['notebooks_stage1']}"]
    for stage, values in report["stages"].items():
        lines.append(f"Stage {stage}: jobs={values['jobs_total']} states={values['jobs_by_state']} "
                     f"ensembles={values['ensembles_complete']}/{values['ensembles_expected']}")
    lines += [f"Selected union: {report['selected_generator_union']['status']}",
              f"Panel selection: {report['panel_selection']['status']}",
              f"Scientific lock: {report['scientific_lock']['status']}",
              f"Locked inference: {report['locked_inference']['status']}",
              f"Final aggregation: {report['final_aggregation']['status']}",
              f"GPU certification: {report['gpu_certification'].get('status', 'BLOCKED')}"]
    if report["current_blockers"]:
        lines.append("Blockers:"); lines.extend(f" - {item}" for item in report["current_blockers"])
    return "\n".join(lines)
