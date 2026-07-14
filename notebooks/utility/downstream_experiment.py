"""Direct notebook API for one MaxViT-512 or Mammo-FM experiment.

The API deliberately exposes dataset construction, audit, model loading, training and
validation as separate notebook steps.  Nothing runs at import time.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import classifier_metrics as metrics
    from .classifier_architecture_adapters import get_adapter
    from .downstream_protocol import ARCHITECTURES, CONDITIONS, SEEDS, atomic_json, resolve_job
except ImportError:
    import classifier_metrics as metrics
    from classifier_architecture_adapters import get_adapter
    from downstream_protocol import ARCHITECTURES, CONDITIONS, SEEDS, atomic_json, resolve_job


def experiment_configuration(root: Path, architecture: str, condition: str, seed: int, *, gpu: int = 0,
                             resume: bool = True) -> dict[str, Any]:
    if architecture not in ARCHITECTURES or condition not in CONDITIONS or int(seed) not in SEEDS:
        raise ValueError("configuration is outside the 2 x 4 x 3 protocol")
    resolved = resolve_job(Path(root), architecture, condition, int(seed))
    return {**resolved, "architecture": architecture, "condition": condition, "seed": int(seed),
            "gpu": int(gpu), "resume": bool(resume), "results_dir": str(experiment_dir(root, architecture, condition, seed))}


def experiment_dir(root: Path, architecture: str, condition: str, seed: int) -> Path:
    return Path(root) / "results/publication_v2/downstream" / architecture / condition / f"seed_{int(seed)}"


def configure_environment(configuration: Mapping[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(configuration["gpu"]))
    return {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "architecture": configuration["architecture"], "seed": configuration["seed"]}


def construct_dataset(root: Path, configuration: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from . import classifier_dataset_builder as builder
    except ImportError:
        import classifier_dataset_builder as builder
    train_rows, validation_rows, provenance = builder.build_training_and_validation_rows(
        Path(root), configuration["variant"])
    audit = audit_dataset(train_rows, validation_rows)
    if audit["train_validation_patient_overlap"]:
        raise RuntimeError("patient leakage detected before training")
    return {"train_rows": train_rows, "validation_rows": validation_rows,
            "provenance": provenance, "audit": audit}


def audit_dataset(train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    train_patients = {str(row.get("patient_id")) for row in train_rows if row.get("patient_id")}
    validation_patients = {str(row.get("patient_id")) for row in validation_rows if row.get("patient_id")}
    sources = Counter(str(row.get("source", "unknown")) for row in train_rows)
    labels = Counter(int(row["label"]) for row in train_rows)
    def source_count(*tokens: str) -> int:
        return sum(count for source, count in sources.items() if any(token in source.lower() for token in tokens))
    return {
        "number_of_real_negatives": sum(1 for row in train_rows if int(row["label"]) == 0 and "synthetic" not in str(row.get("source", ""))),
        "number_of_real_positives": sum(1 for row in train_rows if int(row["label"]) == 1 and "synthetic" not in str(row.get("source", "")) and "augment" not in str(row.get("source", ""))),
        "number_of_traditional_augmentations": source_count("augment"),
        "number_of_finetuned_synthetic_positives": source_count("finetuned"),
        "number_of_fromscratch_synthetic_positives": source_count("from_scratch", "fromscratch"),
        "number_of_patients": len(train_patients), "number_of_images": len(train_rows),
        "train_validation_patient_overlap": sorted(train_patients & validation_patients),
        "source_distribution": dict(sorted(sources.items())), "class_balance": dict(sorted(labels.items())),
        "validation_images": len(validation_rows), "validation_patients": len(validation_patients),
    }


def training_budget(configuration: Mapping[str, Any]) -> dict[str, Any]:
    policy = configuration["policy"]
    return {"max_optimizer_updates": int(policy["max_optimizer_updates"]),
            "checkpoint_metric": policy["checkpoint_criterion"],
            "scheduler_monitor": policy["scheduler_params"]["monitor"],
            "early_stopping_monitor": policy["early_stopping"]["monitor"],
            "effective_batch_size": int(policy["effective_batch_size"]),
            "validation_manifest": "data/processed/metadata/val.csv"}


def source_exposure(audit: Mapping[str, Any], optimizer_steps: int, effective_batch_size: int) -> dict[str, Any]:
    total_seen = int(optimizer_steps) * int(effective_batch_size)
    distribution = audit.get("source_distribution", {})
    total = sum(int(value) for value in distribution.values())
    rows = []
    for source, count in sorted(distribution.items()):
        seen = total_seen * int(count) / total if total else 0.0
        rows.append({"source": source, "available_samples": int(count), "samples_seen": seen,
                     "effective_epochs": seen / int(count) if count else 0.0})
    return {"optimizer_steps": int(optimizer_steps), "effective_batch_size": int(effective_batch_size),
            "samples_seen_by_source": rows, "sampler_policy": "documented proportional sampling; no hidden oversampling"}


def select_best_epoch(history: Sequence[Mapping[str, Any]], *, tolerance: float = 1e-12) -> Mapping[str, Any]:
    """Maximise PR-AUC; ties use lower validation loss, then the earlier epoch."""
    if not history: raise ValueError("history is empty")
    best = history[0]
    for candidate in history[1:]:
        pr_candidate, pr_best = float(candidate["val_pr_auc"]), float(best["val_pr_auc"])
        if pr_candidate > pr_best and not math.isclose(pr_candidate, pr_best, abs_tol=tolerance, rel_tol=tolerance):
            best = candidate
        elif math.isclose(pr_candidate, pr_best, abs_tol=tolerance, rel_tol=tolerance):
            loss_candidate, loss_best = float(candidate["val_loss"]), float(best["val_loss"])
            if loss_candidate < loss_best and not math.isclose(loss_candidate, loss_best, abs_tol=tolerance, rel_tol=tolerance):
                best = candidate
            elif math.isclose(loss_candidate, loss_best, abs_tol=tolerance, rel_tol=tolerance) and int(candidate["epoch"]) < int(best["epoch"]):
                best = candidate
    return best


def load_model(configuration: Mapping[str, Any], *, tiny: bool = False):
    adapter = get_adapter(configuration["architecture"], configuration["policy"], Path(configuration.get("root", ".")), tiny=tiny)
    model = adapter.build_model(seed=configuration["seed"])
    return adapter, model


def train(root: Path, configuration: Mapping[str, Any], dataset: Mapping[str, Any], *, tiny: bool = False) -> dict[str, Any]:
    """Train only when this function is explicitly called from notebook section 10."""
    output = experiment_dir(root, configuration["architecture"], configuration["condition"], configuration["seed"])
    output.mkdir(parents=True, exist_ok=True)
    adapter = get_adapter(configuration["architecture"], configuration["policy"], Path(root), tiny=tiny)
    checkpoint = output / "checkpoint_best"
    suffix = ".pt" if configuration["policy"]["framework"].startswith("pytorch") else ".keras"
    checkpoint = checkpoint.with_suffix(suffix)
    result = adapter.train(dataset["train_rows"], dataset["validation_rows"], checkpoint,
                           seed=configuration["seed"], run_dir=output, architecture=configuration["architecture"],
                           experiment_id=configuration["experiment_id"], dataset_variant_id=configuration["condition"],
                           training_policy=configuration["training_policy_name"], config_signature="informational",
                           dataset_signature=dataset["provenance"].get("signature", "informational"))
    configuration_payload = {key: configuration[key] for key in ("architecture", "condition", "seed", "gpu", "resume")}
    configuration_payload["training_budget"] = training_budget(configuration)
    atomic_json(output / "configuration.json", configuration_payload)
    dataset_summary = {**dataset["audit"], "validation_manifest": "data/processed/metadata/val.csv",
                       "validation_signature": dataset["provenance"].get("validation_signature")}
    atomic_json(output / "dataset_summary.json", dataset_summary)
    history = result.get("history", {})
    rows = _history_rows(history)
    _write_csv(output / "training_history.csv", rows)
    exposure = source_exposure(dataset["audit"], int(result.get("optimizer_updates", configuration["policy"]["max_optimizer_updates"])),
                               int(configuration["policy"]["effective_batch_size"]))
    atomic_json(output / "source_exposure.json", exposure)
    return {**result, "checkpoint": str(checkpoint), "output_dir": str(output), "source_exposure": exposure}


def _history_rows(history: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    length = max((len(values) for values in history.values() if isinstance(values, Sequence)), default=0)
    return [{"epoch": index + 1, **{name: values[index] if index < len(values) else None
             for name, values in history.items() if isinstance(values, Sequence)}} for index in range(length)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True); fields = sorted({key for row in rows for key in row}) or ["epoch"]
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path); return path


def run_validation(root: Path, configuration: Mapping[str, Any], dataset: Mapping[str, Any], checkpoint: str | Path,
                   *, tiny: bool = False) -> dict[str, Any]:
    adapter = get_adapter(configuration["architecture"], configuration["policy"], Path(root), tiny=tiny)
    prediction = adapter.predict_validation(Path(checkpoint), dataset["validation_rows"], seed=configuration["seed"])
    rows = [{"patient_id": source.get("patient_id"), "image_id": source.get("image_id"), "label": int(label),
             "probability": float(probability)} for source, label, probability in zip(
                 dataset["validation_rows"], prediction["labels"], prediction["probabilities"])]
    probabilities = [row["probability"] for row in rows]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities): raise ValueError("invalid probabilities")
    report = metrics.full_report([row["label"] for row in rows], probabilities)
    output = experiment_dir(root, configuration["architecture"], configuration["condition"], configuration["seed"])
    _write_csv(output / "validation_predictions.csv", rows); atomic_json(output / "validation_metrics.json", report)
    return {"rows": rows, "metrics": report, "prediction_path": str(output / "validation_predictions.csv")}


def resume_status(root: Path, configuration: Mapping[str, Any]) -> dict[str, Any]:
    output = experiment_dir(root, configuration["architecture"], configuration["condition"], configuration["seed"])
    candidates = sorted(output.glob("checkpoint_*.pkl")) + sorted(output.glob("checkpoint_best.*"))
    return {"resume_requested": bool(configuration["resume"]), "available": bool(candidates),
            "candidates": [str(path) for path in candidates]}


def saved_artifacts(root: Path, configuration: Mapping[str, Any]) -> list[str]:
    output = experiment_dir(root, configuration["architecture"], configuration["condition"], configuration["seed"])
    return [str(path.relative_to(root)) for path in sorted(output.rglob("*")) if path.is_file()]


__all__ = ["audit_dataset", "configure_environment", "construct_dataset", "experiment_configuration",
           "experiment_dir", "load_model", "resume_status", "run_validation", "saved_artifacts",
           "select_best_epoch", "source_exposure", "train", "training_budget"]
