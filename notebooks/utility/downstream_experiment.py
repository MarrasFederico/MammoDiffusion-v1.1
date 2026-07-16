"""Direct notebook API for one MaxViT-512 or Mammo-FM experiment.

The API deliberately exposes dataset construction, audit, model loading, training and
validation as separate notebook steps.  Nothing runs at import time.
"""
from __future__ import annotations

import csv
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
    from .downstream_protocol import ARCHITECTURES, CONDITIONS, SEEDS, atomic_json, resolve_job
except ImportError:
    import classifier_metrics as metrics
    from classifier_architecture_adapters import get_adapter
    from downstream_protocol import ARCHITECTURES, CONDITIONS, SEEDS, atomic_json, resolve_job


RUN_MODES = ("standard", "smoke")
STANDARD_RESULTS_ROOT = "results/publication_v2/downstream"
SMOKE_RESULTS_ROOT = "results/smoke"
SMOKE_MIN_PER_CLASS = 8
SMOKE_BUDGET = {"physical_batch_size": 2, "gradient_accumulation_steps": 1, "max_optimizer_updates": 2,
                "validation_batches": 1, "checkpoint_interval_updates": 1, "dataloader_workers": 0}
# Any data path that resolves to one of these tokens is rejected in smoke mode.
FORBIDDEN_DATA_TOKENS = ("test.csv", "final_evaluation", "locked_test", "historical_test")
FORBIDDEN_PATH_COMPONENTS = ("test", "historical_internal_test")


def _check_run_mode(run_mode: str) -> str:
    if run_mode not in RUN_MODES:
        raise ValueError(f"unknown RUN_MODE {run_mode!r}; expected one of {RUN_MODES}")
    return run_mode


def experiment_configuration(root: Path, architecture: str, condition: str, seed: int, *, gpu: int | str = 0,
                             resume: bool = True, run_mode: str = "standard", smoke_updates: int = 2) -> dict[str, Any]:
    if architecture not in ARCHITECTURES or condition not in CONDITIONS or int(seed) not in SEEDS:
        raise ValueError("configuration is outside the 2 x 4 x 3 protocol")
    _check_run_mode(run_mode)
    resolved = resolve_job(Path(root), architecture, condition, int(seed))
    policy = dict(resolved["policy"])
    configuration = {**resolved, "policy": policy, "architecture": architecture, "condition": condition,
                     "seed": int(seed), "gpu": gpu, "resume": bool(resume), "run_mode": run_mode}
    if run_mode == "smoke":
        if int(smoke_updates) not in (1, 2):
            raise ValueError("smoke_updates must be 1 or 2")
        configuration["smoke_updates"] = int(smoke_updates)
        configuration["policy"] = smoke_policy(policy, int(smoke_updates))  # runtime copy; official protocol untouched
    configuration["results_dir"] = str(experiment_dir(root, architecture, condition, seed, run_mode=run_mode))
    return configuration


def smoke_policy(policy: Mapping[str, Any], smoke_updates: int = 2) -> dict[str, Any]:
    """Runtime copy of a training policy limited to one or two optimizer updates for a smoke run."""
    if int(smoke_updates) not in (1, 2):
        raise ValueError("smoke_updates must be 1 or 2")
    smoke = dict(policy)
    smoke["max_optimizer_updates"] = int(smoke_updates)
    smoke["effective_batch_size"] = SMOKE_BUDGET["physical_batch_size"] * SMOKE_BUDGET["gradient_accumulation_steps"]
    smoke["physical_batch_size"] = SMOKE_BUDGET["physical_batch_size"]
    smoke["gradient_accumulation_steps"] = SMOKE_BUDGET["gradient_accumulation_steps"]
    smoke["validation_interval_updates"] = 1
    smoke["validation_batches"] = SMOKE_BUDGET["validation_batches"]
    smoke["checkpoint_interval_updates"] = SMOKE_BUDGET["checkpoint_interval_updates"]
    smoke["dataloader_workers"] = SMOKE_BUDGET["dataloader_workers"]
    return smoke


def experiment_dir(root: Path, architecture: str, condition: str, seed: int, *, run_mode: str = "standard") -> Path:
    _check_run_mode(run_mode)
    base = SMOKE_RESULTS_ROOT if run_mode == "smoke" else STANDARD_RESULTS_ROOT
    return Path(root) / base / architecture / condition / f"seed_{int(seed)}"


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
    tensorflow_module = sys.modules.get("tensorflow")
    if tensorflow_module is not None:
        try:
            if tensorflow_module.config.list_logical_devices("GPU"): initialized.append("TensorFlow")
        except Exception:
            pass
    return initialized


def _resolve_gpu_selector(selector: int | str, inventory: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve a physical index or a GPU-... UUID to exactly one inventory row, or fail."""
    if isinstance(selector, str) and selector.startswith(GPU_UUID_PREFIX):
        matches = [row for row in inventory if _normalize_uuid(row["uuid"]) == _normalize_uuid(selector)]
        if len(matches) != 1:
            raise RuntimeError(f"GPU UUID {selector} is not present as exactly one device in the "
                               f"nvidia-smi inventory ({len(matches)} matches)")
        return matches[0]
    try:
        index = int(selector)
    except (TypeError, ValueError):
        raise RuntimeError(f"unsupported GPU selector {selector!r}: pass a physical index or a 'GPU-...' UUID")
    matches = [row for row in inventory if int(row["index"]) == index]
    if len(matches) != 1:
        raise RuntimeError(f"physical GPU index {index} is not uniquely resolvable in the nvidia-smi "
                           f"inventory ({len(matches)} matches); refusing to fall back to a CUDA index")
    return matches[0]


def configure_visible_gpu(selector: int | str, *, inventory=None, observe=None) -> dict[str, Any]:
    """Mask exactly one physical GPU by UUID before any framework initialization, then verify identity.

    ``selector`` is a physical nvidia-smi index (int or numeric str) or a ``GPU-...`` UUID. A physical
    index is resolved to its UUID via a single nvidia-smi query *before* PyTorch/TensorFlow are
    imported, and only the UUID is ever written to ``CUDA_VISIBLE_DEVICES`` — never the numeric index.
    A selector that cannot be resolved to exactly one device, a UUID absent from the inventory, an
    observed identity that differs from the request, or an already-initialized framework with a
    different mask are all hard failures; there is no silent CUDA-index fallback and no warning-only
    path. On success ``physical_identity_verified`` is always ``True``.
    """
    initialized = _initialized_frameworks()
    entries = list(inventory() if inventory is not None else _nvidia_smi_inventory())
    if not entries:
        raise RuntimeError("nvidia-smi inventory is unavailable; refusing to select a GPU by CUDA index")
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
    return {"requested_selector": selector, "resolved_physical_index": int(resolved["index"]),
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
    full_audit = audit_dataset(train_rows, validation_rows)
    if full_audit["train_validation_patient_overlap"]:
        raise RuntimeError("patient leakage detected before training")
    run_mode = _check_run_mode(configuration.get("run_mode", "standard"))
    if run_mode != "smoke":
        return {"train_rows": train_rows, "validation_rows": validation_rows,
                "provenance": provenance, "audit": {"full": full_audit}}
    include_synthetic = bool(configuration["variant"].get("synthetic_generator_id"))
    train_rows = deterministic_smoke_subset(train_rows, include_synthetic=include_synthetic)
    validation_rows = deterministic_smoke_subset(validation_rows, include_synthetic=False)
    assert_no_forbidden_data_paths(root, train_rows + validation_rows)
    if not verify_smoke_synthetic_membership(root, train_rows, configuration["condition"]):
        raise RuntimeError("smoke synthetic rows are not members of the signed FILTERED manifest")
    smoke_audit = audit_dataset(train_rows, validation_rows)
    if smoke_audit["train_validation_patient_overlap"]:
        raise RuntimeError("patient leakage detected in smoke subset")
    return {"train_rows": train_rows, "validation_rows": validation_rows,
            "provenance": provenance, "audit": {"full": full_audit, "smoke": smoke_audit}}


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
    run_mode = _check_run_mode(configuration.get("run_mode", "standard"))
    max_updates = int(policy["max_optimizer_updates"])
    if run_mode == "smoke" and max_updates > SMOKE_BUDGET["max_optimizer_updates"]:
        raise RuntimeError("Smoke runs are limited to two optimizer updates.")
    return {"max_optimizer_updates": max_updates, "run_mode": run_mode,
            "checkpoint_metric": policy["checkpoint_criterion"],
            "scheduler_monitor": policy["scheduler_params"]["monitor"],
            "early_stopping_monitor": policy["early_stopping"]["monitor"],
            "effective_batch_size": int(policy["effective_batch_size"]),
            "validation_interval": int(policy["validation_interval_updates"]),
            "lr_schedule": policy.get("scheduler"),
            "early_stopping_policy": dict(policy["early_stopping"]),
            "validation_manifest": "data/processed/metadata/val.csv"}


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
            raise RuntimeError(f"smoke dataset references a forbidden (test/final-evaluation) path: {raw} -> {resolved}")


def deterministic_smoke_subset(rows: Sequence[Mapping[str, Any]], *, min_per_class: int = SMOKE_MIN_PER_CLASS,
                               include_synthetic: bool = False) -> list[dict[str, Any]]:
    """Deterministic smoke subset: >= min_per_class real negatives and positives (+ synthetic when asked).

    Ordered by (patient_id, image_id, path); no unregistered random sampling.
    """
    def key(row: Mapping[str, Any]):
        return (str(row.get("patient_id")), str(row.get("image_id")), _row_path(row))
    real = [row for row in rows if "synthetic" not in str(row.get("source", ""))]
    synthetic = [row for row in rows if "synthetic" in str(row.get("source", ""))]
    negatives = sorted((row for row in real if int(row["label"]) == 0), key=key)
    positives = sorted((row for row in real if int(row["label"]) == 1), key=key)
    if len(negatives) < min_per_class or len(positives) < min_per_class:
        raise RuntimeError(f"smoke subset needs >= {min_per_class} real negatives and positives, "
                           f"found {len(negatives)}/{len(positives)}")
    picked = [dict(row) for row in negatives[:min_per_class]] + [dict(row) for row in positives[:min_per_class]]
    if include_synthetic:
        synthetic_sorted = sorted(synthetic, key=key)
        if len(synthetic_sorted) < min_per_class:
            raise RuntimeError(f"smoke subset needs >= {min_per_class} synthetic positives, found {len(synthetic_sorted)}")
        picked += [dict(row) for row in synthetic_sorted[:min_per_class]]
    return picked


def verify_smoke_synthetic_membership(root: Path, rows: Sequence[Mapping[str, Any]], condition: str) -> bool:
    """Every synthetic smoke row must belong (by SHA-256) to the selected generator's signed manifest."""
    synthetic = [row for row in rows if "synthetic" in str(row.get("source", ""))]
    if not synthetic:
        return True
    try:
        from . import downstream_protocol as dp, classifier_dataset_builder as builder
    except ImportError:
        import downstream_protocol as dp, classifier_dataset_builder as builder
    family = "finetuned" if "finetuned" in condition else "from_scratch"
    payload = dp.load_selected_generators(Path(root))
    # The manifest file's own SHA-256 is verified here; per-image content is verified at resolve time.
    records = builder.load_selected_filtered_records(Path(root), payload, family, verify_file_content=False)
    signed = {record["sha256"] for record in records}
    return all(row.get("sha256") in signed for row in synthetic)


def source_exposure(audit: Mapping[str, Any], optimizer_steps: int, effective_batch_size: int) -> dict[str, Any]:
    total_seen = int(optimizer_steps) * int(effective_batch_size)
    distribution = audit.get("source_distribution", {})
    total = sum(int(value) for value in distribution.values())
    rows = []
    for source, count in sorted(distribution.items()):
        seen = total_seen * int(count) / total if total else 0.0
        rows.append({"source": source, "available_samples": int(count), "expected_samples_exposed": seen,
                     "effective_epochs": seen / int(count) if count else 0.0})
    return {"optimizer_steps": int(optimizer_steps), "effective_batch_size": int(effective_batch_size),
            "expected_source_exposure": rows, "accounting_mode": "proportional_estimate",
            "sampler_policy": "documented proportional sampling; no hidden oversampling"}


SOURCE_ACCOUNTING_FIELDS = ("real_negative_seen", "real_positive_seen", "traditional_augmented_seen",
                            "finetuned_synthetic_seen", "fromscratch_synthetic_seen")


def _source_field(row: Mapping[str, Any]) -> str:
    source = str(row.get("source", "")).lower()
    if "augment" in source: return "traditional_augmented_seen"
    if "finetuned" in source: return "finetuned_synthetic_seen"
    if "from_scratch" in source or "fromscratch" in source: return "fromscratch_synthetic_seen"
    return "real_positive_seen" if int(row.get("label", 0)) == 1 else "real_negative_seen"


def source_accounting(processed_rows: Sequence[Mapping[str, Any]], previous: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Count examples after their batch is processed; prior counts make resume additive."""
    counts = {field: int((previous or {}).get(field, 0)) for field in SOURCE_ACCOUNTING_FIELDS}
    for row in processed_rows:
        counts[_source_field(row)] += 1
    return {"schema_version": 1, "accounting_mode": "actual", **counts,
            "total_samples_seen": sum(counts.values())}


def proportional_source_accounting(audit: Mapping[str, Any], optimizer_updates: int,
                                   effective_batch_size: int) -> dict[str, Any]:
    """Explicit fallback: expected exposure, never fields named ``*_seen``."""
    exposure = source_exposure(audit, optimizer_updates, effective_batch_size)
    return {"schema_version": 1, "accounting_mode": "proportional_estimate",
            "optimizer_updates": int(optimizer_updates), "effective_batch_size": int(effective_batch_size),
            "expected_source_exposure": exposure["expected_source_exposure"]}


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
    run_mode = _check_run_mode(configuration.get("run_mode", "standard"))
    output = experiment_dir(root, configuration["architecture"], configuration["condition"], configuration["seed"], run_mode=run_mode)
    resume_configuration = {**configuration, "dataset_signature": dataset["provenance"].get("signature", "informational")}
    resume = resume_status(root, resume_configuration)
    output.mkdir(parents=True, exist_ok=True)
    active_audit = dataset["audit"].get("smoke") or dataset["audit"]["full"]
    gpu_uuid = configuration.get("gpu_uuid")
    if run_mode == "smoke" and not gpu_uuid:
        raise RuntimeError("smoke train() requires configuration['gpu_uuid']; run configure_environment first")
    adapter = get_adapter(configuration["architecture"], configuration["policy"], Path(root), tiny=tiny)
    checkpoint = output / "checkpoint_best"
    suffix = ".pt" if configuration["policy"]["framework"].startswith("pytorch") else ".keras"
    checkpoint = checkpoint.with_suffix(suffix)
    result = adapter.train(dataset["train_rows"], dataset["validation_rows"], checkpoint,
                           seed=configuration["seed"], run_dir=output, architecture=configuration["architecture"],
                           experiment_id=configuration["experiment_id"], dataset_variant_id=configuration["condition"],
                           training_policy=configuration["training_policy_name"], config_signature="informational",
                           dataset_signature=dataset["provenance"].get("signature", "informational"),
                           resume=bool(configuration["resume"]), gpu_uuid=gpu_uuid, run_mode=run_mode)
    optimizer_updates = int(result.get("optimizer_updates", configuration["policy"]["max_optimizer_updates"]))
    if run_mode == "smoke":
        write_smoke_run_config(output, configuration)
        atomic_json(output / "dataset_audit.json", dataset["audit"])
        _write_csv(output / "train_log.csv", _history_rows(result.get("history", {})))
        return {**result, "checkpoint": str(checkpoint), "output_dir": str(output), "resume_status": resume,
                "optimizer_updates": optimizer_updates}
    configuration_payload = {key: configuration[key] for key in ("architecture", "condition", "seed", "gpu", "resume")}
    configuration_payload["training_budget"] = training_budget(configuration)
    atomic_json(output / "configuration.json", configuration_payload)
    dataset_summary = {**active_audit, "validation_manifest": "data/processed/metadata/val.csv",
                       "validation_signature": dataset["provenance"].get("validation_signature")}
    atomic_json(output / "dataset_summary.json", dataset_summary)
    _write_csv(output / "training_history.csv", _history_rows(result.get("history", {})))
    exposure = source_exposure(active_audit, optimizer_updates, int(configuration["policy"]["effective_batch_size"]))
    atomic_json(output / "source_exposure.json", exposure)
    if result.get("processed_sample_rows") is not None:
        accounting = source_accounting(result["processed_sample_rows"], result.get("previous_source_accounting"))
    else:
        accounting = result.get("source_accounting") or proportional_source_accounting(
            active_audit, optimizer_updates, int(configuration["policy"]["effective_batch_size"]))
    atomic_json(output / "source_accounting.json", accounting)
    return {**result, "checkpoint": str(checkpoint), "output_dir": str(output), "source_exposure": exposure,
            "source_accounting": accounting, "resume_status": resume}


def write_smoke_run_config(output: Path, configuration: Mapping[str, Any]) -> Path:
    """Single smoke run_config.json with GPU identity and the smoke budget."""
    payload = {"mode": "smoke", "architecture": configuration["architecture"], "condition": configuration["condition"],
               "seed": int(configuration["seed"]), "gpu_selector_requested": configuration.get("gpu"),
               "gpu_uuid": configuration.get("gpu_uuid"), "gpu_name": configuration.get("gpu_name"),
               "gpu_physical_index": configuration.get("gpu_physical_index"),
               "smoke_updates": int(configuration.get("smoke_updates", 2)),
               "results_dir": str(output), "test_accessed": False}
    return atomic_json(Path(output) / "run_config.json", payload)


def finalize_smoke_run(output: Path, *, optimizer_updates: int, resumed: bool) -> Path:
    """Write smoke.json only after training and validation both complete."""
    payload = {"mode": "smoke", "test_accessed": False, "completed": True,
               "optimizer_updates": int(optimizer_updates), "resumed": bool(resumed)}
    return atomic_json(Path(output) / "smoke.json", payload)


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
    run_mode = _check_run_mode(configuration.get("run_mode", "standard"))
    output = experiment_dir(root, configuration["architecture"], configuration["condition"], configuration["seed"], run_mode=run_mode)
    if run_mode == "smoke":
        atomic_json(output / "validation.json", report)
        return {"rows": rows, "metrics": report, "prediction_path": str(output / "validation.json")}
    _write_csv(output / "validation_predictions.csv", rows); atomic_json(output / "validation_metrics.json", report)
    return {"rows": rows, "metrics": report, "prediction_path": str(output / "validation_predictions.csv")}


def resume_status(root: Path, configuration: Mapping[str, Any]) -> dict[str, Any]:
    output = experiment_dir(root, configuration["architecture"], configuration["condition"], configuration["seed"], run_mode=configuration.get("run_mode", "standard"))
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


def _present(payload: Mapping[str, Any], *keys: str) -> bool:
    return any(payload.get(key) not in (None, "") for key in keys)


def verify_resume_continuity(payload: Mapping[str, Any], *, current_gpu_uuid: str,
                             max_optimizer_updates: int = 2) -> dict[str, Any]:
    """Verify a real resume checkpoint continues training instead of restarting from zero.

    Accepts either checkpoint convention (``model_state_dict``/``optimizer_state_dict`` for
    PyTorch/Tiny, ``model_state``/``optimizer_state`` for Keras).  Requires ``global_step`` >= 1, model
    and optimizer state, and a ``gpu_uuid`` matching the current device.  For the PyTorch/Tiny format
    (``*_state_dict``) it also requires ``scheduler_state_dict`` and ``rng_states``.  A missing/zero
    step or a GPU mismatch is a hard failure.
    """
    if not _present(payload, "model_state_dict", "model_state"):
        raise RuntimeError("resume checkpoint is missing model state")
    if not _present(payload, "optimizer_state_dict", "optimizer_state"):
        raise RuntimeError("resume checkpoint is missing optimizer state")
    if payload.get("gpu_uuid") in (None, ""):
        raise RuntimeError("resume checkpoint is missing gpu_uuid")
    if payload.get("global_step") in (None, ""):
        raise RuntimeError("resume checkpoint is missing global_step")
    if "model_state_dict" in payload:  # PyTorch / Tiny format
        for key in ("scheduler_state_dict", "rng_states"):
            if key not in payload:
                raise RuntimeError(f"pytorch resume checkpoint is missing {key}")
    step = int(payload["global_step"])
    if step < 1:
        raise RuntimeError("resume checkpoint has global_step < 1; refusing to restart silently from zero")
    if _normalize_uuid(payload["gpu_uuid"]) != _normalize_uuid(current_gpu_uuid):
        raise RuntimeError(f"resume GPU identity {payload['gpu_uuid']} does not match current {current_gpu_uuid}")
    if step >= int(max_optimizer_updates):
        return {"resume_step": step, "next_step": step, "complete": True}
    return {"resume_step": step, "next_step": step + 1, "complete": False}


def load_existing_outputs(root: Path, configuration: Mapping[str, Any]) -> dict[str, Any]:
    output = experiment_dir(root, configuration["architecture"], configuration["condition"], configuration["seed"], run_mode=configuration.get("run_mode", "standard"))
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
    if accounting.get("accounting_mode") != "actual":
        rows = accounting.get("expected_source_exposure", [])
        fields = [str(row["source"]) for row in rows]
        values = [float(row["expected_samples_exposed"]) for row in rows]
        ylabel = "Expected samples exposed (proportional estimate)"
    else:
        fields = ("real_negative_seen", "real_positive_seen", "traditional_augmented_seen",
                  "finetuned_synthetic_seen", "fromscratch_synthetic_seen")
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


def saved_artifacts(root: Path, configuration: Mapping[str, Any]) -> list[str]:
    output = experiment_dir(root, configuration["architecture"], configuration["condition"], configuration["seed"], run_mode=configuration.get("run_mode", "standard"))
    return [str(path.relative_to(root)) for path in sorted(output.rglob("*")) if path.is_file()]


def smoke_summary(*, test_accessed: bool = False, completed: bool = True) -> dict[str, Any]:
    """Content of a smoke run's smoke.json (written only by a real smoke run, not by this module)."""
    return {"mode": "smoke", "test_accessed": bool(test_accessed), "completed": bool(completed)}


__all__ = ["assert_no_forbidden_data_paths", "audit_dataset", "build_error_case_table", "configure_environment",
           "configure_visible_gpu", "construct_dataset", "deterministic_smoke_subset", "experiment_configuration",
           "finalize_smoke_run", "write_smoke_run_config",
           "experiment_dir", "load_adapter", "load_existing_outputs", "load_model", "load_prediction_rows", "plot_calibration", "plot_source_accounting", "plot_training_history",
           "plot_validation_curves", "resume_status", "run_validation", "RUN_MODES", "saved_artifacts", "smoke_policy", "smoke_summary",
           "SMOKE_BUDGET", "SMOKE_RESULTS_ROOT", "STANDARD_RESULTS_ROOT", "verify_resume_continuity", "verify_smoke_synthetic_membership",
           "proportional_source_accounting", "select_best_epoch", "source_accounting", "source_exposure", "train", "training_budget"]
