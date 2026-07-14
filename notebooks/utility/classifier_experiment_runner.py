"""Executable train/validation pipeline for one classifier matrix seed.

The normal runner never exposes locked-test data.  The same functions are called by generated
notebooks and by the GPU scheduler, so both paths create identical artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import csv
import copy
import inspect
import math
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import classifier_checkpoint_io as ckio  # noqa: E402
import classifier_run_manifest as manifest  # noqa: E402
from classifier_architecture_adapters import get_adapter  # noqa: E402
from classifier_pipeline_contracts import (  # noqa: E402
    PIPELINE_NAMESPACE, REQUIRED_SEEDS, code_revision, sha256_file, signed_payload, value_signature,
    verify_signed_payload,
)
from dataset_variant_registry import load_classifier_registry  # noqa: E402

MODES = ("plan", "auto", "train", "validate", "locked-test", "metrics-only")
SEEDS = (17, 42, 73)


class TrainingInterrupted(KeyboardInterrupt):
    pass


def _atomic_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(path)
    return path


def _signed_artifact(artifact_type: str, **fields) -> dict:
    return signed_payload({"schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE,
                           "artifact_type": artifact_type, **fields})


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["patient_id", "image_id", "label", "probability"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    return path


def _signature(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _retryable_failure(exc: BaseException) -> bool:
    message = str(exc).lower()
    transient_tokens = ("out of memory", "resourceexhausted", "temporarily unavailable",
                        "timeout", "connection reset", "worker crash")
    return isinstance(exc, (OSError, TimeoutError, ConnectionError, MemoryError)) or \
        any(token in message for token in transient_tokens)


def load_dataset_variant_registry(root: Path) -> dict:
    payload = json.loads((root / "configs/dataset_variant_registry.json").read_text())
    matrix_path = root / "configs/classifier_experiment_matrix.json"
    if matrix_path.is_file() and json.loads(matrix_path.read_text()).get("pipeline_namespace") == PIPELINE_NAMESPACE \
       and payload.get("pipeline_namespace") != PIPELINE_NAMESPACE:
        raise ValueError("legacy dataset registry cannot be used by classifier-matrix v2")
    return payload


def load_training_protocols(root: Path) -> dict:
    payload = json.loads((root / "configs/classifier_training_protocols.json").read_text())
    matrix_path = root / "configs/classifier_experiment_matrix.json"
    if matrix_path.is_file() and json.loads(matrix_path.read_text()).get("pipeline_namespace") == PIPELINE_NAMESPACE \
       and payload.get("pipeline_namespace") != PIPELINE_NAMESPACE:
        raise ValueError("legacy training protocols cannot be used by classifier-matrix v2")
    return payload


def resolve_job(root: Path, architecture: str, dataset_variant_id: str, seed: int) -> dict:
    variants = {v["dataset_variant_id"]: v for v in load_dataset_variant_registry(root)["variants"]}
    variant = variants.get(dataset_variant_id)
    if variant is None:
        raise ValueError(f"unknown dataset_variant_id: {dataset_variant_id}")
    protocols = load_training_protocols(root)["policies"]
    if architecture not in protocols:
        raise ValueError(f"unknown architecture policy: {architecture}")
    if seed not in tuple(protocols[architecture].get("seeds", SEEDS)):
        raise ValueError(f"seed {seed} is not registered for {architecture}")
    policy = copy.deepcopy(protocols[architecture])
    training_policy_name = f"{architecture}_standard"
    run = ckio.run_dir(root, architecture, dataset_variant_id, training_policy_name, seed)
    override = run / "oom_override.json"
    if override.is_file():
        payload = json.loads(override.read_text())
        policy["physical_batch_size"] = int(payload["physical_batch_size"])
        policy["gradient_accumulation_steps"] = int(payload["gradient_accumulation_steps"])
        if policy["physical_batch_size"] * policy["gradient_accumulation_steps"] != policy["effective_batch_size"]:
            raise RuntimeError("OOM override changes the registered effective batch")
    results = ckio.results_dir(root, architecture, dataset_variant_id, training_policy_name, seed)
    return {"variant": variant, "policy": policy, "training_policy_name": training_policy_name,
            "run_dir": run, "results_dir": results}


def plan(root: Path, architecture: str, dataset_variant_id: str, seed: int) -> dict:
    job = resolve_job(root, architecture, dataset_variant_id, seed)
    variant, policy, run = job["variant"], job["policy"], job["run_dir"]
    if variant.get("status") not in ("ready", "legacy"):
        return {"action": "error", "reason": f"dataset_variant {dataset_variant_id} is {variant.get('status')}: "
                f"{variant.get('blocker') or variant.get('invalid_reason')}"}
    state = manifest.reconstruct_state(run, policy["framework"])
    # Definitive v2 policy: all matrix seeds are trained homogeneously from scratch.  Legacy
    # artifacts remain historical baselines and are never copied into seed 17/42/73 runs.
    legacy_id = None
    if state["state"] in ("FAILED_FINAL", "BLOCKED", "INVALIDATED"):
        action = "blocked_terminal"
    elif state["state"] in ("TRAINED", "VALIDATING", "VALIDATED", "ENSEMBLE_READY", "COMPLETE"):
        action = "skip_training"
    elif state["state"] == "RUNNING":
        action = "wait_running_elsewhere"
    elif state["state"] == "INTERRUPTED_RESUMABLE":
        action = "resume_training"
    else:
        action = "train"
    needs_validation = state["state"] not in ("VALIDATED", "ENSEMBLE_READY", "COMPLETE") and action not in (
        "wait_running_elsewhere", "blocked_terminal")
    return {"experiment_id": ckio.experiment_id(architecture, dataset_variant_id, seed),
            "state": state["state"], "reason": state.get("reason"), "action": action,
            "legacy_checkpoint_alias": legacy_id, "needs_validation": needs_validation,
            "run_dir": str(run), "test_access": False}


def _dataset_bundle(root, job, supplied=None):
    if supplied is not None:
        bundle = supplied
    else:
        import classifier_dataset_builder as datasets
        train, validation, payload = datasets.build_training_and_validation_rows(root, job["variant"])
        _atomic_json(job["run_dir"] / "dataset_manifest.json", payload)
        bundle = (train, validation, payload)
    payload = bundle[2]
    if payload.get("pipeline_namespace") == PIPELINE_NAMESPACE:
        recorded = payload.get("manifest_signature")
        unsigned = {key: value for key, value in payload.items() if key != "manifest_signature"}
        if not recorded or recorded != value_signature(unsigned):
            raise RuntimeError("dataset manifest is unsigned or changed")
    return bundle


def _write_checkpoint_metadata(job, architecture, dataset_variant_id, seed, checkpoint, dataset_payload):
    ckio.write_checkpoint_metadata(
        job["run_dir"], architecture=architecture, dataset_variant_id=dataset_variant_id,
        training_policy=job["training_policy_name"], seed=seed, checkpoint=Path(checkpoint),
        dataset_manifest_sha256=dataset_payload["signature"], protocol_signature=_signature(job["policy"]),
    )


def _legacy_entry(root, alias):
    registry = load_classifier_registry(root)
    return next((entry for entry in registry.get("experiments", []) if entry["experiment_id"] == alias), None)


def import_legacy_checkpoint(root, job, architecture, dataset_variant_id, seed, adapter, alias, dataset_payload):
    entry = _legacy_entry(root, alias)
    if not entry or not entry.get("checkpoint_path"):
        raise RuntimeError(f"legacy alias {alias} has no checkpoint path")
    source = root / entry["checkpoint_path"]
    if not source.is_file():
        raise RuntimeError(f"legacy checkpoint is missing: {source}")
    destination = ckio.checkpoint_path(job["run_dir"], job["policy"]["framework"])
    model = adapter.load_checkpoint(source, strict=True)
    adapter.save_checkpoint(model, destination)
    _write_checkpoint_metadata(job, architecture, dataset_variant_id, seed, destination, dataset_payload)
    _atomic_json(job["run_dir"] / "legacy_import_manifest.json", {
        "schema_version": 1, "legacy_experiment_id": alias,
        "source_signature": ckio.checkpoint_signature(source),
        "normalized_checkpoint_signature": ckio.checkpoint_signature(destination),
    })
    return destination


def run_train(root: Path, architecture: str, dataset_variant_id: str, seed: int, train_fn=None,
              adapter=None, dataset_bundle=None, tiny=False, allow_retrain=False, resume=True) -> dict:
    job = resolve_job(root, architecture, dataset_variant_id, seed)
    run = job["run_dir"]
    checkpoint_expected = {"architecture": architecture, "dataset_variant_id": dataset_variant_id,
                           "training_policy": job["training_policy_name"], "seed": int(seed)}
    verified, _reason = ckio.checkpoint_is_verified(run, job["policy"]["framework"], checkpoint_expected)
    if verified and not allow_retrain:
        return {"status": "skipped_verified", "reason": "ALLOW_RETRAIN is false", "checkpoint": str(ckio.checkpoint_path(run, job["policy"]["framework"]))}
    if not resume and (ckio.resume_checkpoint_path(run).is_file() or ckio.resume_checkpoint_path(run, "checkpoint_previous").is_file()):
        return {"status": "resume_disabled", "reason": "RESUME is false; refusing to discard resumable state"}
    if not manifest.acquire_claim(run, worker_id=f"pid{os.getpid()}", pid=os.getpid()):
        return {"status": "not_claimed", "reason": "another worker holds a live claim on this run"}
    previous_handlers = {}
    def interrupt_handler(signum, _frame):
        raise TrainingInterrupted(f"signal {signum}")
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.getsignal(sig); signal.signal(sig, interrupt_handler)
    started_at = time.time()
    try:
        previous_manifest = manifest.read_manifest(run)
        if allow_retrain and previous_manifest and previous_manifest.get("state") in ("TRAINED", "VALIDATED", "ENSEMBLE_READY", "COMPLETE"):
            manifest.reset_terminal_state(run, reason="ALLOW_RETRAIN explicitly requested")
        manifest.write_state(run, "RUNNING", experiment_id=ckio.experiment_id(architecture, dataset_variant_id, seed))
        if train_fn is not None and dataset_bundle is None:
            train_rows, validation_rows = [], []
            dataset_payload = {"signature": "dependency-injected-test"}
        else:
            train_rows, validation_rows, dataset_payload = _dataset_bundle(root, job, dataset_bundle)
        experiment = ckio.experiment_id(architecture, dataset_variant_id, seed)
        config_signature = _signature(job["policy"])
        from classifier_gpu_gate import environment_signature
        _atomic_json(run / "run_metadata.json", _signed_artifact("classifier_run_metadata",
            experiment_id=experiment, architecture=architecture, dataset_variant_id=dataset_variant_id,
            training_policy=job["training_policy_name"], seed=int(seed), code_revision=code_revision(root)))
        _atomic_json(run / "resolved_configuration.json", _signed_artifact("classifier_resolved_configuration",
            experiment_id=experiment, config=job["policy"], config_signature=config_signature,
            dataset_signature=dataset_payload["signature"],
            validation_signature=dataset_payload.get("validation_signature", dataset_payload["signature"])))
        _atomic_json(run / "environment_manifest.json", _signed_artifact("classifier_environment",
            experiment_id=experiment, environment_signature=environment_signature(job["policy"]),
            code_revision=code_revision(root)))
        adapter = adapter or get_adapter(architecture, job["policy"], root, tiny=tiny)
        checkpoint = ckio.checkpoint_path(run, job["policy"]["framework"])
        if train_fn is not None:  # dependency injection remains test-only
            produced = Path(train_fn(run, job["policy"], job["variant"], seed))
            checkpoint = produced
        else:
            result = adapter.train(
                train_rows, validation_rows, checkpoint, seed=seed, run_dir=run,
                architecture=architecture, experiment_id=experiment,
                dataset_variant_id=dataset_variant_id, training_policy=job["training_policy_name"],
                config_signature=config_signature, dataset_signature=dataset_payload["signature"],
            )
            checkpoint = Path(result["checkpoint"])
            _atomic_json(job["results_dir"] / "training_history.json",
                         _signed_artifact("classifier_training_history", experiment_id=experiment, payload=result))
            history = result.get("history", {})
            if isinstance(history, dict):
                keys = list(history)
                length = max((len(v) for v in history.values() if isinstance(v, list)), default=0)
                _write_csv(job["results_dir"] / f"training_history_seed_{seed}.csv", [
                    {"epoch": index + 1, **{key: history[key][index] if isinstance(history[key], list) and index < len(history[key]) else None for key in keys}}
                    for index in range(length)])
        if not checkpoint.is_file():
            raise RuntimeError(f"adapter did not create checkpoint: {checkpoint}")
        if train_fn is None:
            adapter.load_checkpoint(checkpoint, strict=True)
        # Injected legacy tests may have already written compatible metadata.
        _write_checkpoint_metadata(job, architecture, dataset_variant_id, seed, checkpoint, dataset_payload)
        _atomic_json(job["results_dir"] / "resource_usage.json", _signed_artifact("classifier_resource_usage",
            experiment_id=experiment, elapsed_seconds=time.time() - started_at,
            gpu_metrics_available=False, gpu_metrics=None,
            note="GPU peak metrics are supplied by the signed scheduler/profile artifacts"))
        manifest.write_state(run, "TRAINED", checkpoint=str(checkpoint))
        return {"status": "trained", "checkpoint": str(checkpoint)}
    except TrainingInterrupted as exc:
        _atomic_json(run / "error_report.json", _signed_artifact("classifier_error_report",
            experiment_id=ckio.experiment_id(architecture, dataset_variant_id, seed), error_type=type(exc).__name__,
            message=str(exc), retryable=True))
        manifest.write_state(run, "INTERRUPTED_RESUMABLE", error=str(exc))
        return {"status": "interrupted_resumable", "reason": str(exc)}
    except Exception as exc:
        retryable = _retryable_failure(exc)
        _atomic_json(run / "error_report.json", _signed_artifact("classifier_error_report",
            experiment_id=ckio.experiment_id(architecture, dataset_variant_id, seed), error_type=type(exc).__name__,
            message=str(exc), retryable=retryable))
        manifest.write_state(run, "FAILED_RETRYABLE" if retryable else "FAILED_FINAL", error=str(exc))
        raise
    finally:
        for sig, handler in previous_handlers.items(): signal.signal(sig, handler)
        manifest.release_claim(run)


def run_metrics_only(root: Path, architecture: str, dataset_variant_id: str, seed: int) -> dict:
    import classifier_metrics as metrics
    job = resolve_job(root, architecture, dataset_variant_id, seed)
    predictions_path = job["results_dir"] / f"validation_predictions_seed_{seed}.json"
    if not predictions_path.is_file():
        raise RuntimeError(f"cannot recompute metrics: missing {predictions_path}")
    predictions = json.loads(predictions_path.read_text())
    report = metrics.full_report(predictions["labels"], predictions["probabilities"])
    artifact = _signed_artifact("classifier_validation_metrics",
        experiment_id=ckio.experiment_id(architecture, dataset_variant_id, seed), split="validation", **report)
    _atomic_json(job["results_dir"] / f"validation_metrics_seed_{seed}.json", artifact)
    return artifact


def run_validate(root: Path, architecture: str, dataset_variant_id: str, seed: int, adapter=None,
                 dataset_bundle=None, tiny=False) -> dict:
    job = resolve_job(root, architecture, dataset_variant_id, seed)
    if not manifest.acquire_claim(job["run_dir"], worker_id=f"validation-pid{os.getpid()}", pid=os.getpid()):
        return {"status": "not_claimed", "reason": "another worker holds a live claim"}
    try:
        manifest.write_state(job["run_dir"], "VALIDATING")
        _, validation_rows, dataset_payload = _dataset_bundle(root, job, dataset_bundle)
        verified, reason = ckio.checkpoint_is_verified(job["run_dir"], job["policy"]["framework"], {
            "architecture": architecture, "dataset_variant_id": dataset_variant_id,
            "training_policy": job["training_policy_name"], "seed": int(seed),
            "dataset_manifest_sha256": dataset_payload["signature"],
            "protocol_signature": _signature(job["policy"])})
        if not verified:
            raise RuntimeError(f"validation requires a compatible verified checkpoint: {reason}")
        adapter = adapter or get_adapter(architecture, job["policy"], root, tiny=tiny)
        checkpoint = ckio.checkpoint_path(job["run_dir"], job["policy"]["framework"])
        predictions = adapter.predict_validation(checkpoint, validation_rows, seed=seed)
        if len(predictions["labels"]) != len(validation_rows) or len(predictions["probabilities"]) != len(validation_rows):
            raise RuntimeError("validation prediction count does not match validation manifest")
        rows = [{"patient_id": row.get("patient_id"), "image_id": row.get("image_id"),
                 "label": int(label), "probability": float(probability),
                 "processed_path": row.get("processed_path")}
                for row, label, probability in zip(validation_rows, predictions["labels"], predictions["probabilities"])]
        if any(not row["patient_id"] or not row["image_id"] for row in rows):
            raise RuntimeError("validation predictions require patient_id and image_id")
        keys = [(row["patient_id"], row["image_id"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError("validation predictions contain duplicate patient_id/image_id keys")
        if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in predictions["probabilities"]):
            raise RuntimeError("validation predictions must be finite probabilities")
        predictions = _signed_artifact("classifier_validation_predictions",
            architecture=architecture, experiment_id=ckio.experiment_id(architecture, dataset_variant_id, seed),
            dataset_variant_id=dataset_variant_id, training_policy=job["training_policy_name"], seed=int(seed),
            split="validation", dataset_signature=dataset_payload["signature"],
            validation_signature=dataset_payload.get("validation_signature", dataset_payload["signature"]),
            checkpoint_signature=ckio.checkpoint_signature(checkpoint), labels=predictions["labels"],
            probabilities=predictions["probabilities"], sample_ids=predictions.get("sample_ids"), rows=rows)
        _atomic_json(job["results_dir"] / f"validation_predictions_seed_{seed}.json", predictions)
        _write_csv(job["results_dir"] / f"validation_predictions_seed_{seed}.csv", rows)
        report = run_metrics_only(root, architecture, dataset_variant_id, seed)
        _atomic_json(job["run_dir"] / "validation_complete.json", _signed_artifact(
            "classifier_validation_completion",
            results_dir=str(job["results_dir"]), metrics_signature=report["signature"],
            predictions_signature=predictions["signature"],
            validation_signature=predictions["validation_signature"]))
        manifest.write_state(job["run_dir"], "VALIDATED", metrics=report)
        ensemble = build_ensemble_if_ready(root, architecture, dataset_variant_id)
        return {"status": "validated", "metrics": report, "ensemble": ensemble}
    except Exception as exc:
        retryable = _retryable_failure(exc)
        _atomic_json(job["run_dir"] / "error_report.json", _signed_artifact("classifier_error_report",
            experiment_id=ckio.experiment_id(architecture, dataset_variant_id, seed), error_type=type(exc).__name__,
            message=str(exc), retryable=retryable))
        manifest.write_state(job["run_dir"], "FAILED_RETRYABLE" if retryable else "FAILED_FINAL", error=str(exc))
        raise
    finally:
        manifest.release_claim(job["run_dir"])


def build_ensemble_if_ready(root: Path, architecture: str, dataset_variant_id: str) -> dict:
    import classifier_metrics as metrics
    policy_name = f"{architecture}_standard"
    payloads, source_signatures = [], []
    common_validation_signature = common_dataset_signature = None
    canonical_by_seed = []
    for seed in REQUIRED_SEEDS:
        run = ckio.run_dir(root, architecture, dataset_variant_id, policy_name, seed)
        result = ckio.results_dir(root, architecture, dataset_variant_id, policy_name, seed)
        path = result / f"validation_predictions_seed_{seed}.json"
        if not path.is_file():
            return {"status": "waiting_for_seeds", "missing_seed": seed}
        payload = json.loads(path.read_text()); verify_signed_payload(payload)
        expected = {"artifact_type": "classifier_validation_predictions", "architecture": architecture,
                    "dataset_variant_id": dataset_variant_id, "training_policy": policy_name, "seed": int(seed),
                    "split": "validation"}
        mismatches = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
        if mismatches: raise RuntimeError(f"seed {seed} validation artifact mismatch: {mismatches}")
        verified, reason = ckio.checkpoint_is_verified(run, json.loads(
            (root / "configs/classifier_training_protocols.json").read_text())["policies"][architecture]["framework"], {
                "architecture": architecture, "dataset_variant_id": dataset_variant_id,
                "training_policy": policy_name, "seed": int(seed)})
        if not verified: raise RuntimeError(f"seed {seed} checkpoint is not verified: {reason}")
        rows = payload.get("rows") or []
        mapping = {}
        for row in rows:
            key = (str(row.get("patient_id")), str(row.get("image_id")))
            if key in mapping: raise RuntimeError(f"seed {seed} has duplicate validation key {key}")
            probability = float(row["probability"])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise RuntimeError(f"seed {seed} contains an invalid probability")
            mapping[key] = {**row, "label": int(row["label"]), "probability": probability}
        if not mapping: raise RuntimeError(f"seed {seed} validation artifact is empty")
        if common_validation_signature is None:
            common_validation_signature = payload["validation_signature"]
            common_dataset_signature = payload["dataset_signature"]
        elif (payload["validation_signature"], payload["dataset_signature"]) != (common_validation_signature, common_dataset_signature):
            raise RuntimeError("seed validation/dataset signatures differ")
        payloads.append(payload); canonical_by_seed.append(mapping)
        source_signatures.append({"seed": seed, "prediction_sha256": sha256_file(path),
                                  "prediction_signature": payload["signature"],
                                  "checkpoint_signature": payload["checkpoint_signature"]})
    keys = sorted(canonical_by_seed[0])
    if any(set(mapping) != set(keys) for mapping in canonical_by_seed[1:]):
        raise RuntimeError("seed validation keys are missing or inconsistent")
    labels = [canonical_by_seed[0][key]["label"] for key in keys]
    if any([mapping[key]["label"] for key in keys] != labels for mapping in canonical_by_seed[1:]):
        raise RuntimeError("seed validation labels are inconsistent")
    paths = [canonical_by_seed[0][key].get("processed_path") for key in keys]
    per_seed_probabilities = [[mapping[key]["probability"] for key in keys] for mapping in canonical_by_seed]
    probabilities = metrics.ensemble_probabilities(per_seed_probabilities)
    report = metrics.full_report(labels, probabilities)
    seed_reports = []
    for seed in REQUIRED_SEEDS:
        metrics_path = ckio.results_dir(root, architecture, dataset_variant_id, policy_name, seed) / f"validation_metrics_seed_{seed}.json"
        seed_report = json.loads(metrics_path.read_text()); verify_signed_payload(seed_report)
        seed_reports.append(seed_report)
    manifest_payload = signed_payload({
        "schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE, "artifact_type": "classifier_validation_ensemble",
        "architecture": architecture, "dataset_variant_id": dataset_variant_id,
        "training_policy": policy_name, "seeds": list(REQUIRED_SEEDS), "aggregation": "mean_probability",
        "threshold_source": "ensemble_validation_youden", "metrics": report,
        "seed_stability": {name: metrics.seed_stability([row[name] for row in seed_reports])
                           for name in ("pr_auc", "roc_auc")},
        "validation_predictions": {"labels": labels, "keys": keys, "probabilities": probabilities},
        "dataset_signature": common_dataset_signature, "validation_signature": common_validation_signature,
        "source_artifacts": source_signatures, "test_access": False,
    })
    ensemble_dir = root / "results/classifiers_matrix" / architecture / dataset_variant_id / policy_name / "ensemble"
    out = ensemble_dir / "manifests/ensemble_validation_manifest.json"
    if out.is_file():
        try:
            existing = json.loads(out.read_text()); verify_signed_payload(existing)
        except Exception:
            existing = None
        if existing and existing.get("source_artifacts") == source_signatures and existing.get("signature") == manifest_payload["signature"]:
            return {"status": "already_complete", "manifest": str(out), "metrics": existing["metrics"]}
        completed_states = [manifest.read_manifest(ckio.run_dir(root, architecture, dataset_variant_id, policy_name, seed)) or {}
                            for seed in REQUIRED_SEEDS]
        if any(row.get("state") == "COMPLETE" for row in completed_states):
            raise RuntimeError("existing ensemble is corrupt/incompatible after COMPLETE; explicit incident repair is required")
    _atomic_json(out, manifest_payload)
    ensemble_rows = [{"patient_id": patient_id, "image_id": image_id, "label": label, "probability": probability,
                      "processed_path": path}
                     for (patient_id, image_id), label, probability, path in zip(keys, labels, probabilities, paths)]
    _write_csv(ensemble_dir / "predictions/ensemble_validation_predictions.csv", ensemble_rows)
    _atomic_json(ensemble_dir / "predictions/ensemble_validation_predictions.json",
                 _signed_artifact("classifier_ensemble_validation_predictions",
                    architecture=architecture, dataset_variant_id=dataset_variant_id,
                    training_policy=policy_name, validation_signature=common_validation_signature,
                    source_ensemble_signature=manifest_payload["signature"], rows=ensemble_rows))
    _atomic_json(ensemble_dir / "metrics/ensemble_validation_metrics.json",
                 _signed_artifact("classifier_ensemble_validation_metrics",
                    architecture=architecture, dataset_variant_id=dataset_variant_id,
                    training_policy=policy_name, source_ensemble_signature=manifest_payload["signature"],
                    **report))
    _atomic_json(ensemble_dir / "metrics/locked_validation_threshold.json",
                 _signed_artifact("classifier_locked_validation_threshold",
                    architecture=architecture, dataset_variant_id=dataset_variant_id,
                    training_policy=policy_name, threshold=report["threshold"],
                    source="ensemble_validation", source_ensemble_signature=manifest_payload["signature"],
                    test_access=False))
    for seed in REQUIRED_SEEDS:
        run = ckio.run_dir(root, architecture, dataset_variant_id, policy_name, seed)
        manifest.write_state(run, "ENSEMBLE_READY", ensemble_manifest=str(out))
    _atomic_json(ckio.run_dir(root, architecture, dataset_variant_id, policy_name, REQUIRED_SEEDS[0]).parent /
                 "ensemble_complete.json", _signed_artifact("classifier_ensemble_completion",
                    manifest=str(out.relative_to(root)), ensemble_signature=manifest_payload["signature"],
                    seed_experiment_ids=[ckio.experiment_id(architecture, dataset_variant_id, seed) for seed in REQUIRED_SEEDS]))
    for seed in REQUIRED_SEEDS:
        run = ckio.run_dir(root, architecture, dataset_variant_id, policy_name, seed)
        manifest.write_state(run, "COMPLETE", ensemble_manifest=str(out))
    return {"status": "complete", "manifest": str(out), "metrics": report}


def run_auto(root: Path, architecture: str, dataset_variant_id: str, seed: int, adapter=None,
             dataset_bundle=None, tiny=False, allow_retrain=False, resume=True) -> dict:
    current = plan(root, architecture, dataset_variant_id, seed)
    if current["action"] == "error":
        return {"status": "blocked", **current}
    if current["action"] == "wait_running_elsewhere":
        return {"status": "running_elsewhere", **current}
    if current["action"] == "blocked_terminal":
        return {"status": "blocked_terminal_requires_explicit_reset", **current}
    job = resolve_job(root, architecture, dataset_variant_id, seed)
    adapter = adapter or get_adapter(architecture, job["policy"], root, tiny=tiny)
    if current["action"] == "reuse_legacy_checkpoint":
        _, _, dataset_payload = _dataset_bundle(root, job, dataset_bundle)
        import_legacy_checkpoint(root, job, architecture, dataset_variant_id, seed, adapter,
                                 current["legacy_checkpoint_alias"], dataset_payload)
    elif current["action"] in ("train", "resume_training"):
        trained = run_train(root, architecture, dataset_variant_id, seed, adapter=adapter,
                            dataset_bundle=dataset_bundle, tiny=tiny, allow_retrain=allow_retrain, resume=resume)
        if trained.get("status") in ("interrupted_resumable", "resume_disabled"):
            return trained
    if plan(root, architecture, dataset_variant_id, seed)["needs_validation"]:
        return run_validate(root, architecture, dataset_variant_id, seed, adapter=adapter,
                            dataset_bundle=dataset_bundle, tiny=tiny)
    final_plan = plan(root, architecture, dataset_variant_id, seed)
    if final_plan["state"] == "VALIDATED":
        ensemble = build_ensemble_if_ready(root, architecture, dataset_variant_id)
        return {"status": "ensemble_recovery", "ensemble": ensemble, "plan": final_plan}
    return {"status": "complete_or_validated", "plan": final_plan}


def execute_configuration(root: Path, architecture: str, dataset_variant_id: str, mode="auto",
                          run_seeds=SEEDS, tiny=False, allow_retrain=None, allow_overwrite_verified=None, resume=None) -> list[dict]:
    """Notebook entrypoint: one thin call handles all requested independent seeds."""
    # Generated notebooks from earlier revisions call this API without forwarding their
    # top-level flags. Read those caller globals for backwards-compatible, real enforcement.
    caller = inspect.currentframe().f_back.f_globals
    allow_retrain = bool(caller.get("ALLOW_RETRAIN", False)) if allow_retrain is None else bool(allow_retrain)
    allow_overwrite_verified = bool(caller.get("ALLOW_OVERWRITE_VERIFIED", False)) if allow_overwrite_verified is None else bool(allow_overwrite_verified)
    resume = bool(caller.get("RESUME", True)) if resume is None else bool(resume)
    results = []
    for seed in run_seeds:
        if mode == "plan": result = plan(root, architecture, dataset_variant_id, seed)
        elif mode == "auto": result = run_auto(root, architecture, dataset_variant_id, seed, tiny=tiny, allow_retrain=allow_retrain, resume=resume)
        elif mode == "train": result = run_train(root, architecture, dataset_variant_id, seed, tiny=tiny,
                                                   allow_retrain=allow_retrain or allow_overwrite_verified, resume=resume)
        elif mode == "validate": result = run_validate(root, architecture, dataset_variant_id, seed, tiny=tiny)
        elif mode == "metrics-only": result = run_metrics_only(root, architecture, dataset_variant_id, seed)
        else: raise PermissionError("locked test cannot be executed from a configuration notebook")
        results.append({"seed": seed, **result})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--mode", choices=MODES, default="plan")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--tiny", action="store_true", help="explicit dependency-free synthetic smoke run")
    args = parser.parse_args()
    parsed = ckio.parse_experiment_id(args.experiment_id)
    root = Path(args.project_root)
    if args.mode == "locked-test":
        print("REFUSED: locked-test cannot be started from classifier_experiment_runner.")
        raise SystemExit(2)
    functions = {"plan": plan, "auto": run_auto, "train": run_train,
                 "validate": run_validate, "metrics-only": run_metrics_only}
    kwargs = {} if args.mode in ("plan", "metrics-only") else {"tiny": args.tiny}
    result = functions[args.mode](root, parsed["architecture"], parsed["dataset_variant_id"], parsed["seed"], **kwargs)
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
