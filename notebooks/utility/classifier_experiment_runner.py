"""Config-driven runner for one classifier experiment matrix job (spec section 8).

    python -m notebooks.utility.classifier_experiment_runner \\
      --experiment-id maxvit512__RSB_CONTROLLED_G04__seed17 --mode auto

Modes: plan | auto | train | validate | locked-test | metrics-only. `locked-test` always
refuses to run from here — the real-run test lives exclusively behind
scripts/finalize_locked_test_stage.py + a v2 locked-test notebook, gated on --confirm-locked-test.

The heavy per-framework training call (`train_fn`) is intentionally a pluggable seam: this
module owns orchestration (skip-if-verified, manifest transitions, dataset resolution, metrics),
not four independent deep-learning training loops that already exist elsewhere in
notebooks/utility (maxvit_utils.fit, mammofm_utils.fit_mammofm, ...). Wiring a given
architecture's real fit() into `train_fn` is a separate, GPU-verified integration step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import classifier_checkpoint_io as ckio  # noqa: E402
import classifier_run_manifest as manifest  # noqa: E402
from dataset_variant_registry import load_classifier_registry  # noqa: E402

MODES = ("plan", "auto", "train", "validate", "locked-test", "metrics-only")


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
    policy = protocols[architecture]
    training_policy_name = f"{architecture}_standard"
    run = ckio.run_dir(root, architecture, dataset_variant_id, training_policy_name, seed)
    results = ckio.results_dir(root, architecture, dataset_variant_id, training_policy_name, seed)
    return {"variant": variant, "policy": policy, "training_policy_name": training_policy_name, "run_dir": run, "results_dir": results}


def plan(root: Path, architecture: str, dataset_variant_id: str, seed: int) -> dict:
    job = resolve_job(root, architecture, dataset_variant_id, seed)
    variant, policy, run = job["variant"], job["policy"], job["run_dir"]

    if variant.get("status") == "invalid":
        return {"action": "error", "reason": f"dataset_variant {dataset_variant_id} is invalid: {variant.get('invalid_reason')}"}

    state = manifest.reconstruct_state(run, policy["framework"])
    legacy_id = None
    try:
        classifier_registry = load_classifier_registry(root)
        legacy_id = ckio.legacy_alias_for(variant, classifier_registry, policy["architecture"])
    except (OSError, json.JSONDecodeError):
        pass

    if state["state"] in ("TRAINED", "VALIDATED", "COMPLETE"):
        action = "skip_training"
    elif state["state"] == "RUNNING":
        action = "wait_running_elsewhere"
    elif legacy_id is not None:
        action = "reuse_legacy_checkpoint"
    else:
        action = "train"

    needs_validation = state["state"] not in ("VALIDATED", "COMPLETE") and action != "wait_running_elsewhere"
    return {
        "experiment_id": ckio.experiment_id(architecture, dataset_variant_id, seed),
        "state": state["state"], "reason": state.get("reason"),
        "action": action, "legacy_checkpoint_alias": legacy_id,
        "needs_validation": needs_validation, "run_dir": str(run),
    }


def run_metrics_only(root: Path, architecture: str, dataset_variant_id: str, seed: int) -> dict:
    import classifier_metrics as cm  # noqa: PLC0415

    job = resolve_job(root, architecture, dataset_variant_id, seed)
    predictions_path = job["run_dir"] / "validation_predictions.json"
    if not predictions_path.is_file():
        raise RuntimeError(f"cannot recompute metrics: missing {predictions_path}")
    predictions = json.loads(predictions_path.read_text())
    report = cm.full_report(predictions["labels"], predictions["probabilities"])
    out = job["run_dir"] / "validation_metrics.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    return report


def run_train(root: Path, architecture: str, dataset_variant_id: str, seed: int, train_fn=None) -> dict:
    job = resolve_job(root, architecture, dataset_variant_id, seed)
    run, policy, variant = job["run_dir"], job["policy"], job["variant"]
    pid = os.getpid()
    if not manifest.acquire_claim(run, worker_id=f"pid{pid}", pid=pid):
        return {"status": "not_claimed", "reason": "another worker holds a live claim on this run"}
    try:
        manifest.write_state(run, "RUNNING", experiment_id=ckio.experiment_id(architecture, dataset_variant_id, seed))
        if train_fn is None:
            raise NotImplementedError(
                f"no train_fn registered for architecture={architecture!r}; orchestration "
                "(claiming, manifest transitions, dataset resolution) is implemented and tested, "
                "but wiring the real per-framework fit() call is a separate GPU-verified step. "
                "Pass train_fn=<callable(run_dir, policy, variant, seed)> to execute real training."
            )
        checkpoint = train_fn(run, policy, variant, seed)
        manifest.write_state(run, "TRAINED", checkpoint=str(checkpoint))
        return {"status": "trained", "checkpoint": str(checkpoint)}
    except Exception as exc:  # noqa: BLE001
        manifest.write_state(run, "FAILED_RETRYABLE", error=str(exc))
        raise
    finally:
        manifest.release_claim(run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--mode", choices=MODES, default="plan")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()

    root = Path(args.project_root)
    parsed = ckio.parse_experiment_id(args.experiment_id)
    architecture, dataset_variant_id, seed = parsed["architecture"], parsed["dataset_variant_id"], parsed["seed"]

    if args.mode == "locked-test":
        print("REFUSED: locked-test cannot be started from classifier_experiment_runner.py. "
              "Use scripts/finalize_locked_test_stage.py with --confirm-locked-test after freezing the matrix.")
        raise SystemExit(2)

    if args.mode in ("plan", "auto"):
        result = plan(root, architecture, dataset_variant_id, seed)
        print(json.dumps(result, indent=1))
        if args.mode == "auto" and result["action"] == "train":
            print("auto mode: training required but no train_fn wired from the CLI; "
                  "this is expected until per-architecture training is integrated (see module docstring).")
    elif args.mode == "metrics-only":
        print(json.dumps(run_metrics_only(root, architecture, dataset_variant_id, seed), indent=1))
    elif args.mode == "validate":
        print(json.dumps(plan(root, architecture, dataset_variant_id, seed), indent=1))
    elif args.mode == "train":
        print(json.dumps(run_train(root, architecture, dataset_variant_id, seed), indent=1))


if __name__ == "__main__":
    main()
