"""Direct notebook API for one MaxViT-512 or Mammo-FM experiment.

The API deliberately exposes dataset construction, audit, model loading, training and
validation as separate notebook steps.  Nothing runs at import time.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

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


def configure_visible_gpu(requested_gpu: int, *, probe=None) -> dict[str, Any]:
    """Mask one physical GPU before framework initialization and verify the visible local device."""
    requested = str(int(requested_gpu))
    initialized = []
    torch_module = sys.modules.get("torch")
    if torch_module is not None and getattr(getattr(torch_module, "cuda", None), "is_initialized", lambda: False)():
        initialized.append("PyTorch")
    tensorflow_module = sys.modules.get("tensorflow")
    if tensorflow_module is not None:
        try:
            if tensorflow_module.config.list_logical_devices("GPU"): initialized.append("TensorFlow")
        except Exception:
            pass
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if initialized and current != requested:
        raise RuntimeError(
            f"GPU framework already initialized with CUDA_VISIBLE_DEVICES={current!r}; requested GPU {requested}. "
            "Restart the kernel and run the GPU configuration cell first."
        )
    if not initialized:
        os.environ["CUDA_VISIBLE_DEVICES"] = requested

    if probe is None:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(f"Requested physical GPU {requested}, but exactly one CUDA device is not visible")
        properties = torch.cuda.get_device_properties(0)
        detected = {"local_index": 0, "name": torch.cuda.get_device_name(0),
                    "memory_bytes": int(properties.total_memory), "visible_count": torch.cuda.device_count()}
    else:
        detected = dict(probe())
    if int(detected.get("visible_count", 0)) != 1 or int(detected.get("local_index", -1)) != 0:
        raise RuntimeError(f"Requested physical GPU {requested}, but visible-device verification failed: {detected}")
    if "physical_index" in detected and int(detected["physical_index"]) != int(requested_gpu):
        raise RuntimeError(f"Requested physical GPU {requested}, but probe detected physical GPU {detected['physical_index']}")
    return {"requested_physical_index": int(requested_gpu), "local_index": 0,
            "name": detected.get("name", "unknown"), "memory_bytes": int(detected.get("memory_bytes", 0)),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "frameworks_already_initialized": initialized}


def configure_environment(configuration: Mapping[str, Any], *, probe=None) -> dict[str, Any]:
    result = configure_visible_gpu(int(configuration["gpu"]), probe=probe)
    return {**result, "architecture": configuration["architecture"], "seed": configuration["seed"]}


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
            "validation_interval": int(policy["validation_interval_updates"]),
            "lr_schedule": policy.get("scheduler"),
            "early_stopping_policy": dict(policy["early_stopping"]),
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


def source_accounting(audit: Mapping[str, Any], optimizer_updates: int, effective_batch_size: int) -> dict[str, Any]:
    """Fixed-budget accounting with the publication protocol's five explicit source fields."""
    counts = {
        "real_negative_seen": int(audit.get("number_of_real_negatives", 0)),
        "real_positive_seen": int(audit.get("number_of_real_positives", 0)),
        "traditional_augmented_seen": int(audit.get("number_of_traditional_augmentations", 0)),
        "finetuned_synthetic_seen": int(audit.get("number_of_finetuned_synthetic_positives", 0)),
        "fromscratch_synthetic_seen": int(audit.get("number_of_fromscratch_synthetic_positives", 0)),
    }
    available = sum(counts.values()); total_seen = int(optimizer_updates) * int(effective_batch_size)
    return {key: (total_seen * value / available if available else 0.0) for key, value in counts.items()} | {
        "optimizer_updates": int(optimizer_updates), "effective_batch_size": int(effective_batch_size)}


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


def load_adapter(configuration: Mapping[str, Any], *, tiny: bool = False):
    return get_adapter(configuration["architecture"], configuration["policy"], Path(configuration.get("root", ".")), tiny=tiny)


def train(root: Path, configuration: Mapping[str, Any], dataset: Mapping[str, Any], *, tiny: bool = False) -> dict[str, Any]:
    """Train only when this function is explicitly called from notebook section 10."""
    output = experiment_dir(root, configuration["architecture"], configuration["condition"], configuration["seed"])
    resume_configuration = {**configuration, "dataset_signature": dataset["provenance"].get("signature", "informational")}
    resume = resume_status(root, resume_configuration)
    output.mkdir(parents=True, exist_ok=True)
    adapter = get_adapter(configuration["architecture"], configuration["policy"], Path(root), tiny=tiny)
    checkpoint = output / "checkpoint_best"
    suffix = ".pt" if configuration["policy"]["framework"].startswith("pytorch") else ".keras"
    checkpoint = checkpoint.with_suffix(suffix)
    result = adapter.train(dataset["train_rows"], dataset["validation_rows"], checkpoint,
                           seed=configuration["seed"], run_dir=output, architecture=configuration["architecture"],
                           experiment_id=configuration["experiment_id"], dataset_variant_id=configuration["condition"],
                           training_policy=configuration["training_policy_name"], config_signature="informational",
                           dataset_signature=dataset["provenance"].get("signature", "informational"),
                           resume=bool(configuration["resume"]))
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
    accounting = source_accounting(dataset["audit"], int(result.get("optimizer_updates", configuration["policy"]["max_optimizer_updates"])),
                                   int(configuration["policy"]["effective_batch_size"]))
    atomic_json(output / "source_accounting.json", accounting)
    return {**result, "checkpoint": str(checkpoint), "output_dir": str(output), "source_exposure": exposure,
            "source_accounting": accounting, "resume_status": resume}


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
             "probability": float(probability), "source": source.get("source"),
             "processed_path": source.get("processed_path") or source.get("path")} for source, label, probability in zip(
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
    requested = bool(configuration["resume"])
    if not requested and candidates and not bool(configuration.get("confirm_existing_output", False)):
        raise RuntimeError(
            f"RESUME=False and checkpoints already exist in {output}. Choose a new RUN_NAME/output directory "
            "or set an explicit overwrite confirmation; nothing was deleted."
        )
    payload, source = None, "resume disabled"
    if requested and candidates:
        try:
            from . import classifier_checkpoint_io as checkpoint_io
        except ImportError:
            import classifier_checkpoint_io as checkpoint_io
        expected = {"architecture": configuration["architecture"], "experiment_id": configuration["experiment_id"],
                    "dataset_variant_id": configuration["condition"], "training_policy": configuration["training_policy_name"],
                    "config_signature": str(configuration.get("config_signature", "informational")),
                    "dataset_signature": str(configuration.get("dataset_signature", "informational")), "seed": int(configuration["seed"])}
        payload, source = checkpoint_io.load_resume_checkpoint(output, expected)
        if payload is None and source != "no resume checkpoint":
            raise RuntimeError(f"No compatible resume checkpoint for architecture/condition/seed: {source}")
    return {"resume_requested": requested, "available": bool(candidates), "compatible": payload is not None,
            "resumed_from": source if payload is not None else None,
            "resume_epoch": payload.get("epoch") if payload else None,
            "resume_step": payload.get("global_step") if payload else None,
            "candidates": [str(path) for path in candidates]}


def load_existing_outputs(root: Path, configuration: Mapping[str, Any]) -> dict[str, Any]:
    output = experiment_dir(root, configuration["architecture"], configuration["condition"], configuration["seed"])
    history = []
    if (output / "training_history.csv").is_file():
        with (output / "training_history.csv").open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
    predictions = []
    if (output / "validation_predictions.csv").is_file():
        with (output / "validation_predictions.csv").open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        predictions = [{**row, "label": int(row["label"]), "probability": float(row["probability"])} for row in predictions]
    validation_metrics = json.loads((output / "validation_metrics.json").read_text()) if (output / "validation_metrics.json").is_file() else None
    accounting = json.loads((output / "source_accounting.json").read_text()) if (output / "source_accounting.json").is_file() else None
    checkpoints = sorted(path for path in output.glob("checkpoint_best.*") if path.suffix in {".pt", ".pth", ".keras", ".h5"})
    return {"history": history, "predictions": predictions, "validation_metrics": validation_metrics,
            "source_accounting": accounting, "checkpoint": str(checkpoints[0]) if checkpoints else None}


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file(): return []
    with Path(path).open(newline="", encoding="utf-8") as stream: rows = list(csv.DictReader(stream))
    return [{**row, "label": int(row["label"]), "probability": float(row["probability"])} for row in rows]


def plot_training_history(history: Sequence[Mapping[str, Any]]):
    import matplotlib.pyplot as plt
    if not history: raise ValueError("training history is empty")
    metrics_to_plot = (("loss", "Training loss"), ("val_loss", "Validation loss"),
                       ("val_pr_auc", "Validation PR-AUC"), ("val_auc", "Validation ROC-AUC"),
                       ("learning_rate", "Learning rate"), ("optimizer_steps", "Optimizer steps"))
    figure, axes = plt.subplots(2, 3, figsize=(14, 8)); epochs = [int(float(row.get("epoch", i + 1))) for i, row in enumerate(history)]
    for axis, (field, title) in zip(axes.flat, metrics_to_plot):
        values = [float(row[field]) if row.get(field) not in (None, "") else math.nan for row in history]
        axis.plot(epochs, values, marker="o"); axis.set_title(title); axis.set_xlabel("Epoch")
    figure.tight_layout(); return figure


def plot_source_accounting(accounting: Mapping[str, Any]):
    import matplotlib.pyplot as plt
    fields = ("real_negative_seen", "real_positive_seen", "traditional_augmented_seen",
              "finetuned_synthetic_seen", "fromscratch_synthetic_seen")
    figure, axis = plt.subplots(figsize=(9, 4)); axis.bar(fields, [float(accounting.get(name, 0)) for name in fields])
    axis.tick_params(axis="x", rotation=30); axis.set_ylabel("Samples seen"); figure.tight_layout(); return figure


def plot_validation_curves(rows: Sequence[Mapping[str, Any]], threshold: float = 0.5):
    import matplotlib.pyplot as plt
    labels = np.asarray([int(row["label"]) for row in rows]); probabilities = np.asarray([float(row["probability"]) for row in rows])
    if not len(rows) or len(set(labels.tolist())) < 2: raise ValueError("validation curves require both classes")
    order = np.argsort(-probabilities); sorted_labels = labels[order]
    tp, fp = np.cumsum(sorted_labels == 1), np.cumsum(sorted_labels == 0)
    recall = tp / max(1, int((labels == 1).sum())); precision = tp / np.maximum(tp + fp, 1)
    tpr, fpr = recall, fp / max(1, int((labels == 0).sum()))
    counts = metrics.confusion_counts(labels, probabilities, threshold)
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].plot(recall, precision); axes[0].set_title("PR curve")
    axes[1].plot(fpr, tpr); axes[1].plot([0, 1], [0, 1], "--"); axes[1].set_title("ROC curve")
    axes[2].imshow(matrix, cmap="Blues"); axes[2].set_title("Confusion matrix")
    axes[3].hist(probabilities[labels == 0], alpha=.6, label="negative"); axes[3].hist(probabilities[labels == 1], alpha=.6, label="positive"); axes[3].legend(); axes[3].set_title("Probability distributions")
    figure.tight_layout(); return figure


def plot_calibration(rows: Sequence[Mapping[str, Any]], bins: int = 10):
    import matplotlib.pyplot as plt
    labels = np.asarray([int(row["label"]) for row in rows]); probabilities = np.asarray([float(row["probability"]) for row in rows])
    edges = np.linspace(0, 1, int(bins) + 1); observed, predicted = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & (probabilities <= right if right == 1 else probabilities < right)
        if mask.any(): observed.append(float(labels[mask].mean())); predicted.append(float(probabilities[mask].mean()))
    figure, axis = plt.subplots(figsize=(5, 5)); axis.plot([0, 1], [0, 1], "--"); axis.plot(predicted, observed, marker="o")
    axis.set_xlabel("Mean predicted probability"); axis.set_ylabel("Observed frequency"); axis.set_title("Calibration curve")
    return figure


def build_error_case_table(rows: Sequence[Mapping[str, Any]], threshold: float = 0.5, *, limit: int | None = None):
    import pandas as pd
    output = []
    for source in rows:
        row = dict(source); label, probability = int(row["label"]), float(row["probability"]); predicted = probability >= threshold
        error_type = "false_negative" if label and not predicted else "false_positive" if not label and predicted else "correct"
        output.append({"patient_id": row.get("patient_id"), "image_id": row.get("image_id"), "label": label,
                       "probability": probability, "error_type": error_type,
                       "source/path": row.get("source") or row.get("processed_path") or row.get("path")})
    priority = {"false_positive": 0, "false_negative": 1, "correct": 2}
    output.sort(key=lambda row: (priority[row["error_type"]], -abs(row["probability"] - threshold),
                                 str(row["patient_id"]), str(row["image_id"])))
    selected = output if limit is None else output[:int(limit)]
    return pd.DataFrame(selected, columns=["patient_id", "image_id", "label", "probability", "error_type", "source/path"])


def saved_artifacts(root: Path, configuration: Mapping[str, Any]) -> list[str]:
    output = experiment_dir(root, configuration["architecture"], configuration["condition"], configuration["seed"])
    return [str(path.relative_to(root)) for path in sorted(output.rglob("*")) if path.is_file()]


__all__ = ["audit_dataset", "build_error_case_table", "configure_environment", "configure_visible_gpu", "construct_dataset", "experiment_configuration",
           "experiment_dir", "load_adapter", "load_existing_outputs", "load_model", "load_prediction_rows", "plot_calibration", "plot_source_accounting", "plot_training_history",
           "plot_validation_curves", "resume_status", "run_validation", "saved_artifacts",
           "select_best_epoch", "source_accounting", "source_exposure", "train", "training_budget"]
