"""Direct notebook API for one manual MaxViT-512 or Mammo-FM experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import classifier_metrics as metrics
    from .classifier_architecture_adapters import get_adapter
    from .classifier_protocol import ARCHITECTURES, CONDITIONS, SEEDS, atomic_json, resolve_job
except ImportError:
    import classifier_metrics as metrics
    from classifier_architecture_adapters import get_adapter
    from classifier_protocol import ARCHITECTURES, CONDITIONS, SEEDS, atomic_json, resolve_job


STANDARD_RESULTS_ROOT = "results/3_classifiers/seed_runs"
# Model checkpoints and other heavy intermediate outputs live under experiments/, following the
# project layout (the results tree keeps only the small CSV/JSON tables and the plots).
STANDARD_EXPERIMENTS_ROOT = "experiments/classifiers"
# Any data path that resolves to one of these tokens is rejected before training or validation.
FORBIDDEN_DATA_TOKENS = ("test.csv", "final_evaluation", "locked_test", "historical_test")
FORBIDDEN_PATH_COMPONENTS = ("test", "historical_internal_test")

ARCHITECTURE_DISPLAY_NAMES = {
    "maxvit512": "MaxViT-512",
    "mammofm": "Mammo-FM",
}
CONDITION_DISPLAY_NAMES = {
    "real_only": "Real only",
    "real_augmented": "Real + traditional augmentation",
    "real_plus_best_finetuned_positive": "Real + selected fine-tuned synthetic positives",
    "real_plus_best_fromscratch_positive": "Real + selected from-scratch synthetic positives",
}


def experiment_configuration(root: Path, architecture: str, condition: str, seed: int, *,
                             gpu: int | str | None = "auto") -> dict[str, Any]:
    if architecture not in ARCHITECTURES or condition not in CONDITIONS or int(seed) not in SEEDS:
        raise ValueError("configuration is outside the 2 x 4 x 3 protocol")
    resolved = resolve_job(Path(root), architecture, condition, int(seed))
    policy = dict(resolved["policy"])
    configuration = {**resolved, "policy": policy, "architecture": architecture, "condition": condition,
                     "seed": int(seed), "gpu": gpu}
    canonical_policy = json.dumps(policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    configuration["policy_signature"] = hashlib.sha256(canonical_policy.encode("utf-8")).hexdigest()
    configuration["config_signature"] = configuration["policy_signature"]
    configuration["results_dir"] = str(experiment_dir(root, architecture, condition, seed))
    configuration["checkpoint_dir"] = str(checkpoint_dir(root, architecture, condition, seed))
    return configuration


def experiment_dir(root: Path, architecture: str, condition: str, seed: int) -> Path:
    """Results directory (small CSV/JSON tables and plots) for one condition/seed run."""
    return Path(root) / STANDARD_RESULTS_ROOT / architecture / condition / f"seed_{int(seed)}"


def checkpoint_dir(root: Path, architecture: str, condition: str, seed: int) -> Path:
    """Model directory (checkpoints and heavy intermediate outputs) under experiments/classifiers/."""
    return Path(root) / STANDARD_EXPERIMENTS_ROOT / architecture / condition / f"seed_{int(seed)}"


GPU_UUID_PREFIX = "GPU-"


def _normalize_uuid(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith(GPU_UUID_PREFIX.lower()):
        text = text[len(GPU_UUID_PREFIX):]
    return text


def _nvidia_smi_inventory() -> list[dict[str, Any]]:
    """One nvidia-smi query mapping physical index -> UUID/name/memory, before any framework import."""
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True)
    inventory = []
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4 and parts[0] != "":
            inventory.append({"index": int(parts[0]), "uuid": parts[1], "name": parts[2],
                              "memory_total": parts[3]})
    return inventory


def _torch_observe() -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        return {"visible_count": 0}
    if torch.cuda.device_count() != 1:
        return {"visible_count": int(torch.cuda.device_count())}
    properties = torch.cuda.get_device_properties(0)
    raw_uuid = getattr(properties, "uuid", None)
    return {"visible_count": int(torch.cuda.device_count()), "local_index": 0,
            "name": torch.cuda.get_device_name(0), "memory_bytes": int(properties.total_memory),
            "uuid": f"{GPU_UUID_PREFIX}{raw_uuid}" if raw_uuid is not None else ""}


def _initialized_frameworks() -> list[str]:
    initialized = []
    torch_module = sys.modules.get("torch")
    if torch_module is not None and getattr(getattr(torch_module, "cuda", None), "is_initialized", lambda: False)():
        initialized.append("PyTorch")
    return initialized


def _gpu_memory_total_mib(row: Mapping[str, Any]) -> float:
    try:
        return float(str(row.get("memory_total", "")).strip())
    except (TypeError, ValueError):
        return -1.0


def _automatic_gpu(inventory: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the largest-memory GPU, breaking ties by the lowest physical index."""
    with_memory = [row for row in inventory if _gpu_memory_total_mib(row) >= 0]
    if not with_memory:
        raise RuntimeError(
            "nvidia-smi did not report memory.total for any GPU; automatic selection is unavailable"
        )
    return max(
        with_memory,
        key=lambda row: (_gpu_memory_total_mib(row), -int(row["index"])),
    )


def _resolve_gpu_selector(selector: int | str | None,
                          inventory: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve auto, a physical index or a GPU-... UUID to one inventory row, or fail."""
    if selector is None or (isinstance(selector, str) and selector.strip().lower() == "auto"):
        return _automatic_gpu(inventory)
    if isinstance(selector, str) and selector.startswith(GPU_UUID_PREFIX):
        matches = [row for row in inventory if _normalize_uuid(row["uuid"]) == _normalize_uuid(selector)]
        if len(matches) != 1:
            raise RuntimeError(f"GPU UUID {selector} is not present as exactly one device in the "
                               f"nvidia-smi inventory ({len(matches)} matches)")
        return matches[0]
    try:
        index = int(selector)
    except (TypeError, ValueError):
        raise RuntimeError(
            f"unsupported GPU selector {selector!r}: use 'auto', a physical index or a 'GPU-...' UUID"
        )
    matches = [row for row in inventory if int(row["index"]) == index]
    if len(matches) != 1:
        raise RuntimeError(f"physical GPU index {index} is not uniquely resolvable in the nvidia-smi "
                           f"inventory ({len(matches)} matches); refusing to fall back to a CUDA index")
    return matches[0]


def configure_visible_gpu(selector: int | str | None = "auto", *, inventory=None,
                          observe=None) -> dict[str, Any]:
    """Mask exactly one physical GPU by UUID before any framework initialization, then verify identity.

    ``selector='auto'`` chooses the GPU with the most total VRAM (lowest physical index on ties).
    An explicit selector can be a physical nvidia-smi index or a ``GPU-...`` UUID. Selection is
    resolved to a UUID via a single nvidia-smi query *before* PyTorch/TensorFlow are imported, and
    only the runtime UUID is ever written to ``CUDA_VISIBLE_DEVICES`` — never the numeric index.
    A selector that cannot be resolved to exactly one device, a UUID absent from the inventory, an
    observed identity that differs from the request, or an already-initialized framework with a
    different mask are all hard failures; there is no silent CUDA-index fallback and no warning-only
    path. On success ``physical_identity_verified`` is always ``True``.
    """
    initialized = _initialized_frameworks()
    entries = list(inventory() if inventory is not None else _nvidia_smi_inventory())
    if not entries:
        raise RuntimeError(
            "nvidia-smi inventory is unavailable; cannot select and verify a physical GPU"
        )
    resolved = _resolve_gpu_selector(selector, entries)
    resolved_uuid = resolved["uuid"]

    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if initialized and _normalize_uuid(current) != _normalize_uuid(resolved_uuid):
        raise RuntimeError(
            f"GPU framework already initialized with CUDA_VISIBLE_DEVICES={current!r}; requested "
            f"{selector!r} -> {resolved_uuid}. Restart the kernel and run the GPU configuration cell first.")
    if not initialized:
        os.environ["CUDA_VISIBLE_DEVICES"] = resolved_uuid

    observed = dict(observe() if observe is not None else _torch_observe())
    if int(observed.get("visible_count", 0)) != 1 or int(observed.get("local_index", -1)) != 0:
        raise RuntimeError(f"expected exactly one visible CUDA device at local index 0 after masking "
                           f"{resolved_uuid}: {observed}")
    observed_uuid = observed.get("uuid") or ""
    if not observed_uuid:
        raise RuntimeError(f"could not read the visible device UUID to prove physical identity for "
                           f"{resolved_uuid}; refusing to proceed")
    if _normalize_uuid(observed_uuid) != _normalize_uuid(resolved_uuid):
        raise RuntimeError(f"requested GPU UUID {resolved_uuid}, but CUDA local 0 reports UUID "
                           f"{observed_uuid}. Restart the kernel.")
    automatic = selector is None or (isinstance(selector, str) and selector.strip().lower() == "auto")
    return {"requested_selector": "auto" if automatic else selector,
            "selection_policy": ("maximum_total_memory_then_lowest_physical_index"
                                 if automatic else "explicit_selector"),
            "automatic_selection": automatic,
            "resolved_physical_index": int(resolved["index"]),
            "resolved_uuid": resolved_uuid, "observed_name": observed.get("name", "unknown"),
            "observed_total_memory": observed.get("memory_bytes", observed.get("memory_total")),
            "local_index": 0, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_identity_verified": True, "frameworks_already_initialized": initialized}


def configure_environment(configuration: Mapping[str, Any], *, inventory=None, observe=None) -> dict[str, Any]:
    result = configure_visible_gpu(configuration["gpu"], inventory=inventory, observe=observe)
    return {**result, "architecture": configuration["architecture"], "seed": configuration["seed"]}


def construct_dataset(root: Path, configuration: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from . import classifier_dataset_builder as builder
    except ImportError:
        import classifier_dataset_builder as builder
    train_rows, validation_rows, provenance = builder.build_training_and_validation_rows(
        Path(root), configuration["variant"])
    assert_no_forbidden_data_paths(root, train_rows + validation_rows)
    full_audit = audit_dataset(train_rows, validation_rows)
    if full_audit["train_validation_patient_overlap"]:
        raise RuntimeError("patient leakage detected before training")
    return {"train_rows": train_rows, "validation_rows": validation_rows,
            "provenance": provenance, "audit": {"full": full_audit}}


def audit_dataset(train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    train_patients = {str(row.get("patient_id")) for row in train_rows if row.get("patient_id")}
    validation_patients = {str(row.get("patient_id")) for row in validation_rows if row.get("patient_id")}
    sources = Counter(str(row.get("source", "unknown")) for row in train_rows)
    labels = Counter(int(row["label"]) for row in train_rows)
    accounting = Counter(_source_field(row) for row in train_rows)
    return {
        "number_of_real_negatives": accounting["real_negative_seen"],
        "number_of_real_positives": accounting["real_positive_seen"],
        "number_of_traditional_augmentations": accounting["traditional_augmented_seen"],
        "number_of_finetuned_synthetic_positives": accounting["finetuned_synthetic_seen"],
        "number_of_fromscratch_synthetic_positives": accounting["fromscratch_synthetic_seen"],
        "number_of_patients": len(train_patients), "number_of_images": len(train_rows),
        "train_validation_patient_overlap": sorted(train_patients & validation_patients),
        "source_distribution": dict(sorted(sources.items())), "class_balance": dict(sorted(labels.items())),
        "validation_images": len(validation_rows), "validation_patients": len(validation_patients),
    }


def _row_path(row: Mapping[str, Any]) -> str:
    return str(row.get("relative_path") or row.get("processed_path") or row.get("path") or "")


FORBIDDEN_PATH_COMPONENTS_SET = set(FORBIDDEN_PATH_COMPONENTS)


def assert_no_forbidden_data_paths(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject any row whose resolved data path (following symlinks) references a test/final split.

    Each path is resolved against ``root`` with ``resolve(strict=False)`` — so ``..`` traversal and
    symlink targets are inspected — and every path component is lowercased before matching.
    """
    project_root = Path(root).resolve()
    for row in rows:
        raw = _row_path(row)
        candidate = Path(raw)
        resolved = candidate if candidate.is_absolute() else (project_root / candidate)
        resolved = resolved.resolve(strict=False)  # collapses '..' and follows existing symlinks
        components = [part.lower() for part in resolved.parts]
        lowered = resolved.as_posix().lower()
        if (FORBIDDEN_PATH_COMPONENTS_SET & set(components)) or any(token in lowered for token in FORBIDDEN_DATA_TOKENS):
            raise RuntimeError(f"dataset references a forbidden (test/final-evaluation) path: {raw} -> {resolved}")


SOURCE_ACCOUNTING_FIELDS = ("real_negative_seen", "real_positive_seen", "traditional_augmented_seen",
                            "finetuned_synthetic_seen", "fromscratch_synthetic_seen")


def _source_field(row: Mapping[str, Any]) -> str:
    source = str(row.get("source", "")).lower()
    if source == "augmented": return "traditional_augmented_seen"
    if source == "synthetic":
        family = str(row.get("synthetic_family", "")).lower()
        if family == "finetuned": return "finetuned_synthetic_seen"
        if family == "from_scratch": return "fromscratch_synthetic_seen"
        raise ValueError(f"synthetic row has invalid synthetic_family: {row.get('synthetic_family')!r}")
    return "real_positive_seen" if int(row.get("label", 0)) == 1 else "real_negative_seen"


def source_accounting(processed_rows: Sequence[Mapping[str, Any]], previous: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Count examples after their batch is processed; prior counts make resume additive."""
    counts = {field: int((previous or {}).get(field, 0)) for field in SOURCE_ACCOUNTING_FIELDS}
    for row in processed_rows:
        counts[_source_field(row)] += 1
    return {"schema_version": 1, "accounting_mode": "actual", **counts,
            "total_samples_seen": sum(counts.values())}


def load_adapter(configuration: Mapping[str, Any]):
    return get_adapter(configuration["architecture"], configuration["policy"], Path(configuration.get("root", ".")))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True); fields = sorted({key for row in rows for key in row}) or ["epoch"]
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path); return path


def run_validation(root: Path, configuration: Mapping[str, Any], dataset: Mapping[str, Any],
                   checkpoint: str | Path) -> dict[str, Any]:
    adapter = get_adapter(configuration["architecture"], configuration["policy"], Path(root))
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


def run_test(root: Path, configuration: Mapping[str, Any], checkpoint: str | Path, test_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    adapter = get_adapter(configuration["architecture"], configuration["policy"], Path(root))
    prediction = adapter.predict_validation(Path(checkpoint), list(test_rows), seed=configuration["seed"])
    rows = [{"patient_id": source.get("patient_id"), "image_id": source.get("image_id"), "label": int(label),
             "probability": float(probability), "source": source.get("source"),
             "processed_path": source.get("processed_path") or source.get("path")} for source, label, probability in zip(
                 test_rows, prediction["labels"], prediction["probabilities"])]
    probabilities = [row["probability"] for row in rows]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities): raise ValueError("invalid probabilities")
    report = metrics.full_report([row["label"] for row in rows], probabilities)
    output = experiment_dir(root, configuration["architecture"], configuration["condition"], configuration["seed"])
    _write_csv(output / "test_predictions.csv", rows); atomic_json(output / "test_metrics.json", report)
    return {"rows": rows, "metrics": report, "prediction_path": str(output / "test_predictions.csv")}


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
    checkpoint_output = checkpoint_dir(root, configuration["architecture"], configuration["condition"], configuration["seed"])
    checkpoints = sorted(path for path in checkpoint_output.glob("checkpoint_best.*") if path.suffix in {".pt", ".pth"})
    return {"history": history, "predictions": predictions, "validation_metrics": validation_metrics,
            "source_accounting": accounting, "checkpoint": str(checkpoints[0]) if checkpoints else None}


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file(): return []
    with Path(path).open(newline="", encoding="utf-8") as stream: rows = list(csv.DictReader(stream))
    return [{**row, "label": int(row["label"]), "probability": float(row["probability"])} for row in rows]


def title_classifier_figure(figure, architecture: str, condition: str, seed: int,
                            analysis: str):
    """Add an unambiguous experiment title to a classifier diagnostic figure."""
    architecture_label = ARCHITECTURE_DISPLAY_NAMES.get(str(architecture), str(architecture))
    condition_label = CONDITION_DISPLAY_NAMES.get(str(condition), str(condition))
    figure.suptitle(
        f"{architecture_label} | {condition_label} | Seed {int(seed)}\n{analysis}",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    return figure


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
    if accounting.get("accounting_mode") != "actual":
        raise ValueError("source accounting must use accounting_mode='actual'")
    fields = SOURCE_ACCOUNTING_FIELDS
    values = [float(accounting.get(name, 0)) for name in fields]
    ylabel = "Samples actually processed"
    figure, axis = plt.subplots(figsize=(9, 4)); axis.bar(fields, values)
    axis.tick_params(axis="x", rotation=30); axis.set_ylabel(ylabel); figure.tight_layout(); return figure


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


__all__ = ["ARCHITECTURE_DISPLAY_NAMES", "CONDITION_DISPLAY_NAMES",
           "assert_no_forbidden_data_paths", "audit_dataset", "build_error_case_table", "configure_environment",
           "configure_visible_gpu", "construct_dataset", "experiment_configuration",
           "experiment_dir", "checkpoint_dir", "load_adapter", "load_existing_outputs", "load_prediction_rows",
           "plot_calibration", "plot_source_accounting", "plot_training_history",
           "plot_validation_curves", "run_validation", "STANDARD_RESULTS_ROOT",
           "source_accounting", "title_classifier_figure"]
