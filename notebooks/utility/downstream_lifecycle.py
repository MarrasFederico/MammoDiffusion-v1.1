"""Validation finalization, test lock, one-shot inference, and publication reporting."""
from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import classifier_checkpoint_io as ckio
import classifier_metrics as metrics
import classifier_run_manifest as run_manifest
from classifier_architecture_adapters import get_adapter
from classifier_pipeline_contracts import sha256_file
from classifier_statistics import holm_correction, paired_stratified_bootstrap
from downstream_protocol import (
    ARCHITECTURES, CONDITIONS, NAMESPACE, SEEDS, atomic_json, code_revision, experiment_id,
    load_approval, load_jobs, load_protocol, signed_payload, verify_signed_payload,
)


def ensemble_dir(root: Path, architecture: str, condition: str) -> Path:
    return Path(root) / "results/downstream_classifiers" / architecture / condition / f"{architecture}_fixed_protocol" / "ensemble"


def job_state(root: Path, architecture: str, condition: str, seed: int) -> str:
    protocol = load_protocol(root)
    run = ckio.run_dir(root, architecture, condition, f"{architecture}_fixed_protocol", seed)
    return run_manifest.reconstruct_state(run, protocol["architectures"][architecture]["framework"])["state"]


def inventory(root: Path) -> dict[str, Any]:
    jobs = []
    for job in load_jobs(root)["jobs"]:
        state = job_state(root, job["architecture"], job["condition"], job["seed"])
        jobs.append({**job, "status": state})
    ensembles = []
    for architecture in ARCHITECTURES:
        for condition in CONDITIONS:
            path = ensemble_dir(root, architecture, condition) / "manifests/ensemble_validation_manifest.json"
            status = "MISSING"
            if path.is_file():
                try:
                    payload = json.loads(path.read_text()); verify_signed_payload(payload)
                    status = "COMPLETE" if payload.get("seeds") == list(SEEDS) and payload.get("test_access") is False else "INVALID"
                except Exception:
                    status = "INVALID"
            ensembles.append({"architecture": architecture, "condition": condition,
                              "status": status, "manifest": str(path.relative_to(root))})
    return {"jobs": jobs, "ensembles": ensembles,
            "job_counts": dict(Counter(job["status"] for job in jobs)),
            "ensemble_counts": dict(Counter(item["status"] for item in ensembles)),
            "approved_generators": (Path(root) / "configs/approved_generators.json").is_file(),
            "locked_test_status": "locked" if (Path(root) / "results/locked_test/downstream_test_lock.json").is_file() else "unopened"}


def _load_ensemble(root: Path, architecture: str, condition: str) -> dict[str, Any]:
    path = ensemble_dir(root, architecture, condition) / "manifests/ensemble_validation_manifest.json"
    payload = json.loads(path.read_text())
    verify_signed_payload(payload)
    if payload.get("seeds") != list(SEEDS) or payload.get("test_access") is not False:
        raise ValueError(f"invalid ensemble: {architecture}/{condition}")
    return payload


def aggregate_patient(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_patient: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        patient = str(row.get("patient_id") or "")
        if not patient:
            raise ValueError("patient-level aggregation requires patient_id")
        by_patient.setdefault(patient, []).append(row)
    output = []
    for patient, group in sorted(by_patient.items()):
        labels = {int(row["label"]) for row in group}
        if len(labels) != 1:
            raise ValueError(f"patient {patient} has inconsistent labels")
        probabilities = [float(row["probability"]) for row in group]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError(f"patient {patient} has invalid probabilities")
        output.append({"patient_id": patient, "label": labels.pop(),
                       "probability": float(sum(probabilities) / len(probabilities)), "n_images": len(group)})
    return output


def _patient_ensemble(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    predictions = payload["validation_predictions"]
    rows = [{"patient_id": key[0], "image_id": key[1], "label": label, "probability": probability}
            for key, label, probability in zip(predictions["keys"], predictions["labels"], predictions["probabilities"])]
    return aggregate_patient(rows)


def finalize_validation(root: Path) -> dict[str, Any]:
    root = Path(root)
    approval = load_approval(root, required=True)
    status = inventory(root)
    if status["job_counts"].get("COMPLETE", 0) != 24:
        raise RuntimeError("validation finalization requires 24/24 COMPLETE jobs")
    if status["ensemble_counts"].get("COMPLETE", 0) != 8:
        raise RuntimeError("validation finalization requires 8/8 complete seed ensembles")
    protocol = load_protocol(root)
    ensembles = {(architecture, condition): _load_ensemble(root, architecture, condition)
                 for architecture in ARCHITECTURES for condition in CONDITIONS}
    comparisons, p_values = [], {}
    for architecture in ARCHITECTURES:
        for left, right in protocol["evaluation"]["primary_comparisons_per_architecture"]:
            rows_left, rows_right = _patient_ensemble(ensembles[(architecture, left)]), _patient_ensemble(ensembles[(architecture, right)])
            left_by_id, right_by_id = {row["patient_id"]: row for row in rows_left}, {row["patient_id"]: row for row in rows_right}
            if set(left_by_id) != set(right_by_id):
                raise RuntimeError("patient keys differ between validation conditions")
            ids = sorted(left_by_id)
            labels = [left_by_id[key]["label"] for key in ids]
            if labels != [right_by_id[key]["label"] for key in ids]:
                raise RuntimeError("patient labels differ between validation conditions")
            result = paired_stratified_bootstrap(
                labels, [left_by_id[key]["probability"] for key in ids], [right_by_id[key]["probability"] for key in ids],
                metrics.pr_auc, n_bootstrap=int(protocol["evaluation"]["confidence_intervals"]["iterations"]),
                seed=int(protocol["evaluation"]["confidence_intervals"]["seed"]),
            )
            comparison_id = f"{architecture}:{left}_vs_{right}"
            p_values[comparison_id] = result["p_value_two_sided"]
            comparisons.append({"comparison_id": comparison_id, "architecture": architecture,
                                "condition_a": left, "condition_b": right, "metric": "pr_auc", **result})
    correction = holm_correction(p_values)
    payload = signed_payload({
        "schema_version": 2, "pipeline_namespace": NAMESPACE,
        "artifact_type": "downstream_validation_finalization",
        "created_at": datetime.now(timezone.utc).isoformat(), "code_revision": code_revision(root),
        "approved_generator_signature": approval["signature"], "jobs_complete": 24, "ensembles_complete": 8,
        "primary_metric": "pr_auc", "patient_level": True, "comparisons": comparisons,
        "holm_correction": correction, "test_access": False,
        "ensemble_signatures": {f"{a}:{c}": ensembles[(a, c)]["signature"] for a in ARCHITECTURES for c in CONDITIONS},
    })
    output = root / "results/downstream_validation/downstream_validation_finalized.json"
    atomic_json(output, payload)
    return {"status": "finalized", "output": str(output), "signature": payload["signature"]}


def _test_manifest_path(root: Path) -> Path:
    return root / "data/processed/metadata/test.csv"


def create_test_lock(root: Path) -> dict[str, Any]:
    root = Path(root)
    approval = load_approval(root, required=True)
    status = inventory(root)
    if status["job_counts"].get("COMPLETE", 0) != 24 or status["ensemble_counts"].get("COMPLETE", 0) != 8:
        raise RuntimeError("test lock requires 24/24 jobs and 8/8 ensembles")
    validation_path = root / "results/downstream_validation/downstream_validation_finalized.json"
    validation = json.loads(validation_path.read_text()); verify_signed_payload(validation, "downstream_validation_finalization")
    if validation.get("approved_generator_signature") != approval["signature"] or validation.get("test_access") is not False:
        raise ValueError("validation finalization is stale or accessed test data")
    test_manifest = _test_manifest_path(root)
    if not test_manifest.is_file():
        raise FileNotFoundError(test_manifest)
    test_image_artifacts = []
    with test_manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            relative = row.get("processed_path") or f"data/processed/test/{row['label']}/{row['image_id']}.png"
            path = (root / relative).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            test_image_artifacts.append({"patient_id": row.get("patient_id"), "image_id": row.get("image_id"),
                                         "relative_path": str(path.relative_to(root)), "sha256": sha256_file(path),
                                         "size_bytes": path.stat().st_size})
    checkpoints = []
    protocol = load_protocol(root)
    for architecture in ARCHITECTURES:
        framework = protocol["architectures"][architecture]["framework"]
        for condition in CONDITIONS:
            for seed in SEEDS:
                run = ckio.run_dir(root, architecture, condition, f"{architecture}_fixed_protocol", seed)
                valid, reason = ckio.checkpoint_is_verified(run, framework, {
                    "architecture": architecture, "dataset_variant_id": condition,
                    "training_policy": f"{architecture}_fixed_protocol", "seed": seed})
                if not valid:
                    raise RuntimeError(f"unverified checkpoint {architecture}/{condition}/{seed}: {reason}")
                path = ckio.checkpoint_path(run, framework)
                checkpoints.append({"experiment_id": experiment_id(architecture, condition, seed),
                                    "relative_path": str(path.relative_to(root)), "sha256": sha256_file(path),
                                    "size_bytes": path.stat().st_size})
    payload = signed_payload({
        "schema_version": 2, "pipeline_namespace": NAMESPACE, "artifact_type": "downstream_locked_test_lock",
        "created_at": datetime.now(timezone.utc).isoformat(), "code_revision": code_revision(root),
        "approved_generators": approval, "conditions": list(CONDITIONS), "architectures": list(ARCHITECTURES),
        "seeds": list(SEEDS), "ensemble_method": "mean_probability",
        "validation_finalization_signature": validation["signature"],
        "validation_thresholds": {f"{a}:{c}": _load_ensemble(root, a, c)["metrics"]["threshold"]
                                  for a in ARCHITECTURES for c in CONDITIONS},
        "statistical_comparisons": load_protocol(root)["evaluation"]["primary_comparisons_per_architecture"],
        "test_dataset_manifest": {"relative_path": str(test_manifest.relative_to(root)),
                                  "sha256": sha256_file(test_manifest), "size_bytes": test_manifest.stat().st_size},
        "test_image_artifacts": test_image_artifacts,
        "checkpoint_artifacts": checkpoints, "jobs_complete": 24, "ensembles_complete": 8,
        "one_shot": True, "post_test_model_selection": False, "post_test_threshold_tuning": False,
    })
    output = root / "results/locked_test/downstream_test_lock.json"
    if output.exists():
        raise FileExistsError("locked test already has a lock; it cannot be overwritten")
    atomic_json(output, payload)
    return {"status": "locked", "output": str(output), "signature": payload["signature"]}


def _load_test_rows(root: Path, expected_hash: str) -> list[dict[str, Any]]:
    path = _test_manifest_path(root)
    if sha256_file(path) != expected_hash:
        raise ValueError("test manifest changed after lock")
    with path.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    rows = []
    for row in source:
        image = row.get("processed_path") or f"data/processed/test/{row['label']}/{row['image_id']}.png"
        image_path = (root / image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        rows.append({"processed_path": str(image_path), "patient_id": row["patient_id"],
                     "image_id": row["image_id"], "label": int(row["label"]), "source": "real_locked_test"})
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["patient_id"])
        writer.writeheader(); writer.writerows(rows)


def run_locked_test(root: Path) -> dict[str, Any]:
    root = Path(root)
    lock_path = root / "results/locked_test/downstream_test_lock.json"
    lock = json.loads(lock_path.read_text()); verify_signed_payload(lock, "downstream_locked_test_lock")
    if code_revision(root) != lock["code_revision"]:
        raise ValueError("code revision changed after the downstream test lock")
    approval = load_approval(root, required=True)
    if approval["signature"] != lock["approved_generators"]["signature"]:
        raise ValueError("approved generator selection changed after the downstream test lock")
    completion = root / "results/locked_test/locked_inference_complete.json"
    claim = root / "results/locked_test/locked_inference_started.json"
    if completion.exists() or claim.exists():
        raise RuntimeError("one-shot locked inference has already started; rerun is forbidden")
    for artifact in lock["checkpoint_artifacts"]:
        path = root / artifact["relative_path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"locked checkpoint changed: {artifact['experiment_id']}")
    for artifact in lock["test_image_artifacts"]:
        path = root / artifact["relative_path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"locked test image changed: {artifact['patient_id']}/{artifact['image_id']}")
    test_rows = _load_test_rows(root, lock["test_dataset_manifest"]["sha256"])
    atomic_json(claim, signed_payload({"schema_version": 2, "pipeline_namespace": NAMESPACE,
                                      "artifact_type": "locked_inference_started",
                                      "lock_signature": lock["signature"], "started_at": datetime.now(timezone.utc).isoformat()}))
    protocol = load_protocol(root)
    outputs = []
    for architecture in ARCHITECTURES:
        policy = protocol["architectures"][architecture]
        adapter = get_adapter(architecture, policy, root)
        for condition in CONDITIONS:
            per_seed = []
            for seed in SEEDS:
                run = ckio.run_dir(root, architecture, condition, f"{architecture}_fixed_protocol", seed)
                checkpoint = ckio.checkpoint_path(run, policy["framework"])
                prediction = adapter.predict_locked_test(checkpoint, test_rows, seed=seed, lock_verified=True)
                per_seed.append(prediction["probabilities"])
                if prediction["labels"] != [row["label"] for row in test_rows]:
                    raise RuntimeError("locked prediction labels are misaligned")
            probabilities = metrics.ensemble_probabilities(per_seed)
            image_rows = [{"patient_id": row["patient_id"], "image_id": row["image_id"], "label": row["label"],
                           "probability": probability} for row, probability in zip(test_rows, probabilities)]
            patient_rows = aggregate_patient(image_rows)
            threshold = float(lock["validation_thresholds"][f"{architecture}:{condition}"])
            report = metrics.full_report([row["label"] for row in patient_rows],
                                         [row["probability"] for row in patient_rows], threshold=threshold)
            base = root / "results/locked_test" / architecture / condition
            _write_csv(base / "ensemble_test_predictions.csv", image_rows)
            _write_csv(base / "patient_level_test_predictions.csv", patient_rows)
            artifact = signed_payload({"schema_version": 2, "pipeline_namespace": NAMESPACE,
                                       "artifact_type": "locked_test_ensemble_result", "architecture": architecture,
                                       "condition": condition, "seeds": list(SEEDS), "threshold_source": "validation_lock",
                                       "patient_level_metrics": report, "lock_signature": lock["signature"]})
            atomic_json(base / "locked_test_metrics.json", artifact)
            outputs.append({"architecture": architecture, "condition": condition, "signature": artifact["signature"]})
    finished = signed_payload({"schema_version": 2, "pipeline_namespace": NAMESPACE,
                               "artifact_type": "locked_inference_completion", "completed_at": datetime.now(timezone.utc).isoformat(),
                               "lock_signature": lock["signature"], "outputs": outputs,
                               "post_test_model_selection": False, "post_test_threshold_tuning": False})
    atomic_json(completion, finished)
    return {"status": "complete", "output": str(completion), "results": len(outputs)}


def finalize_publication_report(root: Path) -> Path:
    root = Path(root)
    validation_path = root / "results/downstream_validation/downstream_validation_finalized.json"
    locked_path = root / "results/locked_test/locked_inference_complete.json"
    validation = json.loads(validation_path.read_text()); verify_signed_payload(validation, "downstream_validation_finalization")
    locked = json.loads(locked_path.read_text()); verify_signed_payload(locked, "locked_inference_completion")
    lines = ["# MammoDiffusion publication report", "", "## Research questions", "",
             "- RQ1: generator fidelity, diversity, coverage, and memorization.",
             "- RQ2: downstream utility versus real-only and traditional augmentation.",
             "- RQ3: robustness across MaxViT-512 and Mammo-FM.", "",
             "## Analysis partitions", "", "- Generator validation benchmark", "- Downstream validation",
             "- Locked test", "- Exploratory analyses (clearly non-primary)", "", "## Locked-test safeguards", "",
             "Thresholds and model selection were frozen on validation. Test results were produced once at patient level; no post-test tuning was permitted.", "",
             "## Limitations", "", "Single-dataset study; limited positive sample; filtering may alter diversity; RAD-DINO is not mammography-specific; no external validation; possible domain shift; Mammo-FM licensing restricts weight redistribution; some analyses are exploratory.", ""]
    output = root / "results/final_report/publication_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


__all__ = ["aggregate_patient", "create_test_lock", "ensemble_dir", "finalize_publication_report",
           "finalize_validation", "inventory", "job_state", "run_locked_test"]
