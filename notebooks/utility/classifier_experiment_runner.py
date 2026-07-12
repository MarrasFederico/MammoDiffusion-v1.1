"""Executable train/validation pipeline for one classifier matrix seed.

The normal runner never exposes locked-test data.  The same functions are called by generated
notebooks and by the GPU scheduler, so both paths create identical artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import classifier_checkpoint_io as ckio  # noqa: E402
import classifier_run_manifest as manifest  # noqa: E402
from classifier_architecture_adapters import get_adapter  # noqa: E402
from dataset_variant_registry import load_classifier_registry  # noqa: E402

MODES = ("plan", "auto", "train", "validate", "locked-test", "metrics-only")
SEEDS = (17, 42, 73)


def _atomic_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(path)
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
    policy = protocols[architecture]
    training_policy_name = f"{architecture}_standard"
    run = ckio.run_dir(root, architecture, dataset_variant_id, training_policy_name, seed)
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
    legacy_id = None
    try:
        legacy_id = ckio.legacy_alias_for(variant, load_classifier_registry(root), policy["architecture"])
    except (OSError, json.JSONDecodeError):
        pass
    if state["state"] in ("TRAINED", "VALIDATING", "VALIDATED", "ENSEMBLE_READY", "COMPLETE"):
        action = "skip_training"
    elif state["state"] == "RUNNING":
        action = "wait_running_elsewhere"
    elif legacy_id is not None:
        action = "reuse_legacy_checkpoint"
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
              adapter=None, dataset_bundle=None, tiny=False) -> dict:
    job = resolve_job(root, architecture, dataset_variant_id, seed)
    run = job["run_dir"]
    if not manifest.acquire_claim(run, worker_id=f"pid{os.getpid()}", pid=os.getpid()):
        return {"status": "not_claimed", "reason": "another worker holds a live claim on this run"}
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
            result = adapter.train(train_rows, validation_rows, checkpoint, seed=seed)
            checkpoint = Path(result["checkpoint"])
            _atomic_json(run / "training_history.json", result)
        if not checkpoint.is_file():
            raise RuntimeError(f"adapter did not create checkpoint: {checkpoint}")
        # Injected legacy tests may have already written compatible metadata.
        _write_checkpoint_metadata(job, architecture, dataset_variant_id, seed, checkpoint, dataset_payload)
        manifest.write_state(run, "TRAINED", checkpoint=str(checkpoint))
        return {"status": "trained", "checkpoint": str(checkpoint)}
    except Exception as exc:
        manifest.write_state(run, "FAILED_RETRYABLE", error=str(exc))
        raise
    finally:
        manifest.release_claim(run)


def run_metrics_only(root: Path, architecture: str, dataset_variant_id: str, seed: int) -> dict:
    import classifier_metrics as metrics
    job = resolve_job(root, architecture, dataset_variant_id, seed)
    predictions_path = job["run_dir"] / "validation_predictions.json"
    if not predictions_path.is_file():
        raise RuntimeError(f"cannot recompute metrics: missing {predictions_path}")
    predictions = json.loads(predictions_path.read_text())
    report = metrics.full_report(predictions["labels"], predictions["probabilities"])
    _atomic_json(job["run_dir"] / "validation_metrics.json", report)
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
        predictions.update({"schema_version": 1, "architecture": architecture,
                            "dataset_variant_id": dataset_variant_id, "seed": seed, "split": "validation"})
        _atomic_json(job["run_dir"] / "validation_predictions.json", predictions)
        report = run_metrics_only(root, architecture, dataset_variant_id, seed)
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
        path = run / "validation_predictions.json"
        if not path.is_file():
            return {"status": "waiting_for_seeds", "missing_seed": seed}
        payloads.append(json.loads(path.read_text()))
    sample_ids = payloads[0].get("sample_ids")
    labels = payloads[0]["labels"]
    if any(p["labels"] != labels or p.get("sample_ids") != sample_ids for p in payloads[1:]):
        raise RuntimeError("seed validation predictions are not aligned")
    probabilities = metrics.ensemble_probabilities([p["probabilities"] for p in payloads])
    report = metrics.full_report(labels, probabilities)
    seed_reports = [json.loads((ckio.run_dir(root, architecture, dataset_variant_id, policy_name, seed) /
                                "validation_metrics.json").read_text()) for seed in SEEDS]
    manifest_payload = {
        "schema_version": 1, "architecture": architecture, "dataset_variant_id": dataset_variant_id,
        "training_policy": policy_name, "seeds": list(SEEDS), "aggregation": "mean_probability",
        "threshold_source": "ensemble_validation_youden", "metrics": report,
        "seed_stability": {name: metrics.seed_stability([row[name] for row in seed_reports])
                           for name in ("pr_auc", "roc_auc")},
        "validation_predictions": {"labels": labels, "sample_ids": sample_ids, "probabilities": probabilities},
        "test_access": False,
    }
    manifest_payload["signature"] = _signature(manifest_payload)
    out = (root / "results/classifiers_matrix" / architecture / dataset_variant_id / policy_name /
           "ensemble_validation_manifest.json")
    _atomic_json(out, manifest_payload)
    for seed in SEEDS:
        run = ckio.run_dir(root, architecture, dataset_variant_id, policy_name, seed)
        _atomic_json(run.parent / "ensemble_complete.json", {"manifest": str(out.relative_to(root)),
                                                              "signature": manifest_payload["signature"]})
        manifest.write_state(run, "COMPLETE", ensemble_manifest=str(out))
    return {"status": "complete", "manifest": str(out), "metrics": report}


def run_auto(root: Path, architecture: str, dataset_variant_id: str, seed: int, adapter=None,
             dataset_bundle=None, tiny=False) -> dict:
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
    elif current["action"] == "train":
        run_train(root, architecture, dataset_variant_id, seed, adapter=adapter,
                  dataset_bundle=dataset_bundle, tiny=tiny)
    if plan(root, architecture, dataset_variant_id, seed)["needs_validation"]:
        return run_validate(root, architecture, dataset_variant_id, seed, adapter=adapter,
                            dataset_bundle=dataset_bundle, tiny=tiny)
    return {"status": "complete_or_validated", "plan": plan(root, architecture, dataset_variant_id, seed)}


def execute_configuration(root: Path, architecture: str, dataset_variant_id: str, mode="auto",
                          run_seeds=SEEDS, tiny=False) -> list[dict]:
    """Notebook entrypoint: one thin call handles all requested independent seeds."""
    results = []
    for seed in run_seeds:
        if mode == "plan": result = plan(root, architecture, dataset_variant_id, seed)
        elif mode == "auto": result = run_auto(root, architecture, dataset_variant_id, seed, tiny=tiny)
        elif mode == "train": result = run_train(root, architecture, dataset_variant_id, seed, tiny=tiny)
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
