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
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import classifier_checkpoint_io as ckio  # noqa: E402
import classifier_run_manifest as manifest  # noqa: E402
from classifier_architecture_adapters import get_adapter  # noqa: E402
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


def load_dataset_variant_registry(root: Path) -> dict:
    return json.loads((root / "configs/dataset_variant_registry.json").read_text())


def load_training_protocols(root: Path) -> dict:
    return json.loads((root / "configs/classifier_training_protocols.json").read_text())


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
    if state["state"] in ("TRAINED", "VALIDATING", "VALIDATED", "ENSEMBLE_READY", "COMPLETE"):
        action = "skip_training"
    elif state["state"] == "RUNNING":
        action = "wait_running_elsewhere"
    elif state["state"] == "INTERRUPTED_RESUMABLE":
        action = "resume_training"
    else:
        action = "train"
    needs_validation = state["state"] not in ("VALIDATED", "ENSEMBLE_READY", "COMPLETE") and action != "wait_running_elsewhere"
    return {"experiment_id": ckio.experiment_id(architecture, dataset_variant_id, seed),
            "state": state["state"], "reason": state.get("reason"), "action": action,
            "legacy_checkpoint_alias": legacy_id, "needs_validation": needs_validation,
            "run_dir": str(run), "test_access": False}


def _dataset_bundle(root, job, supplied=None):
    if supplied is not None:
        return supplied
    import classifier_dataset_builder as datasets
    train, validation, payload = datasets.build_training_and_validation_rows(root, job["variant"])
    _atomic_json(job["run_dir"] / "dataset_manifest.json", payload)
    return train, validation, payload


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
    verified, _reason = ckio.checkpoint_is_verified(run, job["policy"]["framework"])
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
    try:
        manifest.write_state(run, "RUNNING", experiment_id=ckio.experiment_id(architecture, dataset_variant_id, seed))
        if train_fn is not None and dataset_bundle is None:
            train_rows, validation_rows = [], []
            dataset_payload = {"signature": "dependency-injected-test"}
        else:
            train_rows, validation_rows, dataset_payload = _dataset_bundle(root, job, dataset_bundle)
        adapter = adapter or get_adapter(architecture, job["policy"], root, tiny=tiny)
        checkpoint = ckio.checkpoint_path(run, job["policy"]["framework"])
        if train_fn is not None:  # dependency injection remains test-only
            produced = Path(train_fn(run, job["policy"], job["variant"], seed))
            checkpoint = produced
        else:
            result = adapter.train(
                train_rows, validation_rows, checkpoint, seed=seed, run_dir=run,
                architecture=architecture, experiment_id=ckio.experiment_id(architecture, dataset_variant_id, seed),
                dataset_variant_id=dataset_variant_id, training_policy=job["training_policy_name"],
                config_signature=_signature(job["policy"]), dataset_signature=dataset_payload["signature"],
            )
            checkpoint = Path(result["checkpoint"])
            _atomic_json(job["results_dir"] / "training_history.json", result)
            history = result.get("history", {})
            if isinstance(history, dict):
                keys = list(history)
                length = max((len(v) for v in history.values() if isinstance(v, list)), default=0)
                _write_csv(job["results_dir"] / f"training_history_seed_{seed}.csv", [
                    {"epoch": index + 1, **{key: history[key][index] if isinstance(history[key], list) and index < len(history[key]) else None for key in keys}}
                    for index in range(length)])
        if not checkpoint.is_file():
            raise RuntimeError(f"adapter did not create checkpoint: {checkpoint}")
        # Injected legacy tests may have already written compatible metadata.
        _write_checkpoint_metadata(job, architecture, dataset_variant_id, seed, checkpoint, dataset_payload)
        manifest.write_state(run, "TRAINED", checkpoint=str(checkpoint))
        return {"status": "trained", "checkpoint": str(checkpoint)}
    except TrainingInterrupted as exc:
        manifest.write_state(run, "INTERRUPTED_RESUMABLE", error=str(exc))
        return {"status": "interrupted_resumable", "reason": str(exc)}
    except Exception as exc:
        manifest.write_state(run, "FAILED_RETRYABLE", error=str(exc))
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
    _atomic_json(job["results_dir"] / f"validation_metrics_seed_{seed}.json", report)
    return report


def run_validate(root: Path, architecture: str, dataset_variant_id: str, seed: int, adapter=None,
                 dataset_bundle=None, tiny=False) -> dict:
    job = resolve_job(root, architecture, dataset_variant_id, seed)
    verified, reason = ckio.checkpoint_is_verified(job["run_dir"], job["policy"]["framework"])
    if not verified:
        raise RuntimeError(f"validation requires a verified checkpoint: {reason}")
    if not manifest.acquire_claim(job["run_dir"], worker_id=f"validation-pid{os.getpid()}", pid=os.getpid()):
        return {"status": "not_claimed", "reason": "another worker holds a live claim"}
    try:
        manifest.write_state(job["run_dir"], "VALIDATING")
        _, validation_rows, _ = _dataset_bundle(root, job, dataset_bundle)
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
        predictions.update({"schema_version": 2, "architecture": architecture,
                            "dataset_variant_id": dataset_variant_id, "seed": seed, "split": "validation"})
        predictions["rows"] = rows
        _atomic_json(job["results_dir"] / f"validation_predictions_seed_{seed}.json", predictions)
        _write_csv(job["results_dir"] / f"validation_predictions_seed_{seed}.csv", rows)
        report = run_metrics_only(root, architecture, dataset_variant_id, seed)
        _atomic_json(job["run_dir"] / "validation_complete.json", {
            "results_dir": str(job["results_dir"]), "metrics_signature": _signature(report)})
        manifest.write_state(job["run_dir"], "VALIDATED", metrics=report)
        ensemble = build_ensemble_if_ready(root, architecture, dataset_variant_id)
        return {"status": "validated", "metrics": report, "ensemble": ensemble}
    except Exception as exc:
        manifest.write_state(job["run_dir"], "FAILED_RETRYABLE", error=str(exc))
        raise
    finally:
        manifest.release_claim(job["run_dir"])


def build_ensemble_if_ready(root: Path, architecture: str, dataset_variant_id: str) -> dict:
    import classifier_metrics as metrics
    policy_name = f"{architecture}_standard"
    payloads = []
    for seed in SEEDS:
        run = ckio.run_dir(root, architecture, dataset_variant_id, policy_name, seed)
        result = ckio.results_dir(root, architecture, dataset_variant_id, policy_name, seed)
        path = result / f"validation_predictions_seed_{seed}.json"
        if not path.is_file():
            return {"status": "waiting_for_seeds", "missing_seed": seed}
        payloads.append(json.loads(path.read_text()))
    keys = [(row["patient_id"], row["image_id"]) for row in payloads[0]["rows"]]
    paths = [row.get("processed_path") for row in payloads[0]["rows"]]
    labels = payloads[0]["labels"]
    if any(p["labels"] != labels or [(row["patient_id"], row["image_id"]) for row in p["rows"]] != keys for p in payloads[1:]):
        raise RuntimeError("seed validation predictions are not aligned")
    probabilities = metrics.ensemble_probabilities([p["probabilities"] for p in payloads])
    report = metrics.full_report(labels, probabilities)
    seed_reports = [json.loads((ckio.results_dir(root, architecture, dataset_variant_id, policy_name, seed) /
                                f"validation_metrics_seed_{seed}.json").read_text()) for seed in SEEDS]
    manifest_payload = {
        "schema_version": 1, "architecture": architecture, "dataset_variant_id": dataset_variant_id,
        "training_policy": policy_name, "seeds": list(SEEDS), "aggregation": "mean_probability",
        "threshold_source": "ensemble_validation_youden", "metrics": report,
        "seed_stability": {name: metrics.seed_stability([row[name] for row in seed_reports])
                           for name in ("pr_auc", "roc_auc")},
        "validation_predictions": {"labels": labels, "keys": keys, "probabilities": probabilities},
        "test_access": False,
    }
    manifest_payload["signature"] = _signature(manifest_payload)
    ensemble_dir = root / "results/classifiers_matrix" / architecture / dataset_variant_id / policy_name / "ensemble"
    out = ensemble_dir / "manifests/ensemble_validation_manifest.json"
    _atomic_json(out, manifest_payload)
    ensemble_rows = [{"patient_id": patient_id, "image_id": image_id, "label": label, "probability": probability,
                      "processed_path": path}
                     for (patient_id, image_id), label, probability, path in zip(keys, labels, probabilities, paths)]
    _write_csv(ensemble_dir / "predictions/ensemble_validation_predictions.csv", ensemble_rows)
    _atomic_json(ensemble_dir / "predictions/ensemble_validation_predictions.json", {"rows": ensemble_rows})
    _atomic_json(ensemble_dir / "metrics/ensemble_validation_metrics.json", report)
    _atomic_json(ensemble_dir / "metrics/locked_validation_threshold.json", {
        "threshold": report["threshold"], "source": "ensemble_validation", "test_access": False})
    for seed in SEEDS:
        run = ckio.run_dir(root, architecture, dataset_variant_id, policy_name, seed)
        _atomic_json(run.parent / "ensemble_complete.json", {"manifest": str(out.relative_to(root)),
                                                              "signature": manifest_payload["signature"]})
        manifest.write_state(run, "COMPLETE", ensemble_manifest=str(out))
    return {"status": "complete", "manifest": str(out), "metrics": report}


def run_auto(root: Path, architecture: str, dataset_variant_id: str, seed: int, adapter=None,
             dataset_bundle=None, tiny=False, allow_retrain=False, resume=True) -> dict:
    current = plan(root, architecture, dataset_variant_id, seed)
    if current["action"] == "error":
        return {"status": "blocked", **current}
    if current["action"] == "wait_running_elsewhere":
        return {"status": "running_elsewhere", **current}
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
    return {"status": "complete_or_validated", "plan": plan(root, architecture, dataset_variant_id, seed)}


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
