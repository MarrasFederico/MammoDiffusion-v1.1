"""Framework-agnostic utilities for the locked classifier evaluation pipeline.

The module deliberately imports no deep-learning framework.  It centralises
content-aware provenance, validation-only selection, patient-level alignment
and paired statistics.  Paths stored in manifests are project-relative when
possible, making the artefacts portable across machines.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCHEMA_VERSION = 1
PREDICTION_COLUMNS = [
    "patient_id", "image_id", "path", "y_true", "y_score", "y_pred",
    "threshold", "threshold_method", "split", "experiment_id",
    "architecture", "dataset_variant", "synthetic_source", "training_mode",
    "checkpoint_path",
]
DATASET_VARIANTS = {
    "real_only", "real_plus_synthetic", "synthetic_only",
    "real_plus_augmented", "real_plus_synthetic_positive",
    "real_plus_augmented_plus_synthetic",
}
TRAINING_MODES = {"linear_probe", "partial_finetuning", "full_finetuning"}
ARCHITECTURE_FAMILY = {
    "ResNet-50": "resnet50",
    "MaxViT-512": "maxvit512",
    "Mammo-FM": "mammofm",
    "RAD-DINO": "raddino",
}


def canonical_test_prediction_paths(experiment: Mapping[str, Any]) -> dict[str, str]:
    """Return the only canonical prediction CSV/manifest paths for an experiment.

    Paths are deliberately project-relative so the registry, notebooks, lock and
    lightweight package remain portable.  Unknown architectures fail closed
    instead of silently falling back to another model family.
    """
    experiment_id = str(experiment.get("experiment_id") or "").strip()
    architecture = experiment.get("architecture")
    if not experiment_id:
        raise ValueError("Canonical prediction paths require experiment_id")
    try:
        family = ARCHITECTURE_FAMILY[str(architecture)]
    except KeyError as exc:
        raise ValueError(f"Unknown architecture for {experiment_id}: {architecture}") from exc
    base = Path("results/classifiers") / family / experiment_id / "final_test"
    return {
        "test_predictions_path": (base / "test_predictions.csv").as_posix(),
        "test_predictions_manifest_path": (base / "test_predictions.manifest.json").as_posix(),
    }


def strict_jsonable(value: Any) -> Any:
    """Return a recursively strict-JSON-compatible representation.

    Missing/non-finite scientific values become JSON ``null``.  This conversion
    happens after pandas has materialised records, so float columns cannot cast
    ``None`` back to NaN.
    """
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return strict_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return strict_jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): strict_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [strict_jsonable(v) for v in value]
    return value


_jsonable = strict_jsonable  # compatibility for existing notebooks


def strict_json_dumps(value: Any, **kwargs: Any) -> str:
    kwargs.setdefault("ensure_ascii", False)
    kwargs["allow_nan"] = False
    return json.dumps(strict_jsonable(value), **kwargs)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        strict_jsonable(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def content_signature(path: str | Path, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    """Return a SHA-256 signature based on bytes, not timestamps."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return {"algorithm": "sha256", "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def value_signature(value: Any) -> dict[str, str]:
    return {"algorithm": "sha256", "sha256": hashlib.sha256(canonical_json_bytes(value)).hexdigest()}


def unwrap_checkpoint_state_dict(raw_state: Any, known_prefixes: Sequence[str] = ("module.",)) -> Any:
    """Normalize a loaded checkpoint to a flat parameter-name -> tensor mapping.

    Deliberately framework-agnostic (operates on plain mappings, never imports torch). Handles
    raw state dicts, ``{"state_dict": ...}``/``{"model_state_dict": ...}`` wrappers, and a
    uniform DataParallel/DistributedDataParallel ``module.`` prefix. A prefix is only stripped
    when *every* key carries it -- a partial match is left untouched rather than guessed at,
    since silently mangling some keys and not others is worse than leaving a checkpoint
    incompatible (the caller's strict-load check will then report it honestly).
    """
    state_dict = raw_state
    if isinstance(state_dict, Mapping):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
    for prefix in known_prefixes:
        if state_dict and all(str(key).startswith(prefix) for key in state_dict):
            state_dict = {str(key)[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def checkpoint_key_mismatch(
    missing_keys: Iterable[str], unexpected_keys: Iterable[str], allowed_mismatches: Iterable[str] = (),
) -> dict[str, list[str]]:
    """Partition a ``load_state_dict(..., strict=False)`` key diff into allowed vs. unexplained.

    Used to implement a strict-by-default checkpoint load: call with ``strict=False`` only to
    *capture* the diff, then fail unless every differing key is in an explicit, documented
    allowlist -- never proceed silently with partial weights.
    """
    allowed = set(allowed_mismatches)
    return {
        "unexplained_missing": [k for k in missing_keys if k not in allowed],
        "unexplained_unexpected": [k for k in unexpected_keys if k not in allowed],
    }


def patient_ids_hash(patient_ids: Iterable[Any]) -> str:
    ids = sorted(str(x) for x in patient_ids)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def load_validation_threshold(path: str | Path, method: str | None = None) -> tuple[float, str]:
    """Load a frozen threshold from a validation JSON artefact."""
    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    keys = ("validation_threshold", "threshold_youden_from_val", "optimal_threshold_youden", "threshold")
    key = next((k for k in keys if payload.get(k) is not None), None)
    if key is None:
        raise ValueError(f"No validation threshold in {path}")
    threshold = float(payload[key])
    if not 0 <= threshold <= 1:
        raise ValueError("Validation threshold must be in [0, 1]")
    inferred = "youden_validation" if "youden" in key else "validation_frozen"
    return threshold, method or str(payload.get("threshold_method", inferred))


def validate_locked_test_configuration(config: Mapping[str, Any], project_root: str | Path = ".") -> dict[str, Any]:
    """Fail closed unless checkpoint, validation threshold and test CSV are usable."""
    root = Path(project_root)
    required = ("experiment_id", "checkpoint_path", "validation_metrics_path", "test_csv")
    missing = [k for k in required if not config.get(k)]
    if missing:
        raise ValueError(f"Missing locked configuration fields: {missing}")
    resolved = dict(config)
    for key in ("checkpoint_path", "validation_metrics_path", "test_csv"):
        path = Path(str(config[key]))
        resolved[key] = path if path.is_absolute() else root / path
        if not resolved[key].is_file():
            raise FileNotFoundError(resolved[key])
    threshold = config.get("validation_threshold")
    method = config.get("threshold_method") or config.get("validation_threshold_method")
    if threshold is None:
        threshold, method = load_validation_threshold(resolved["validation_metrics_path"], method)
    if not method or "test" in str(method).lower():
        raise ValueError("Threshold must be explicitly validation-derived")
    resolved["validation_threshold"] = float(threshold)
    resolved["threshold_method"] = str(method)
    resolved["checkpoint_signature"] = content_signature(resolved["checkpoint_path"])
    return resolved


def build_test_dataset_manifest(
    test_csv: str | Path,
    *,
    project_root: str | Path = ".",
    preprocessing: Mapping[str, Any] | None = None,
    include_image_signatures: bool = True,
) -> dict[str, Any]:
    root, csv_path = Path(project_root), Path(test_csv)
    csv_path = csv_path if csv_path.is_absolute() else root / csv_path
    frame = pd.read_csv(csv_path)
    patient_col = "patient_id"
    label_col = "patient_label" if "patient_label" in frame else "label"
    path_col = "processed_path" if "processed_path" in frame else "path"
    if patient_col not in frame or label_col not in frame or path_col not in frame:
        raise ValueError("Test CSV lacks patient_id, label and processed path")
    if frame[patient_col].duplicated().any():
        raise ValueError("Canonical test CSV must contain one row per patient")
    image_signatures: dict[str, Any] = {}
    if include_image_signatures:
        for raw in frame[path_col].astype(str):
            image_path = Path(raw)
            image_path = image_path if image_path.is_absolute() else root / image_path
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            image_signatures[raw] = content_signature(image_path)
    labels = frame[label_col].astype(int)
    return {
        "schema_version": SCHEMA_VERSION,
        "split": "test",
        "n_patients": int(len(frame)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "patient_ids_hash": patient_ids_hash(frame[patient_col]),
        "csv_signature": content_signature(csv_path),
        "image_signatures": image_signatures,
        "preprocessing": dict(preprocessing or {}),
    }


def standardize_prediction_dataframe(
    frame: pd.DataFrame,
    *,
    experiment: Mapping[str, Any],
    threshold: float,
    threshold_method: str,
    column_map: Mapping[str, str] | None = None,
    require_unique_patients: bool = True,
) -> pd.DataFrame:
    """Map a legacy prediction frame to the canonical patient-level schema."""
    df = frame.rename(columns=dict(column_map or {})).copy()
    aliases = {
        "label": "y_true", "label_true": "y_true", "probability": "y_score",
        "prediction": "y_pred", "processed_path": "path",
    }
    for old, new in aliases.items():
        if old in df and new not in df:
            df[new] = df[old]
    if "path" not in df or "y_true" not in df or "y_score" not in df:
        raise ValueError("Predictions require path, y_true and y_score")
    names = df["path"].astype(str).map(lambda x: Path(x).name)
    parts = names.str.replace(r"\.[^.]+$", "", regex=True).str.split("_")
    if "patient_id" not in df:
        df["patient_id"] = parts.str[0]
    if "image_id" not in df:
        df["image_id"] = parts.str[1]
    df["y_true"] = pd.to_numeric(df["y_true"], errors="raise").astype(int)
    df["y_score"] = pd.to_numeric(df["y_score"], errors="raise").astype(float)
    if not df["y_true"].isin([0, 1]).all() or not df["y_score"].between(0, 1).all():
        raise ValueError("Labels must be binary and scores probabilities")
    df["y_pred"] = (df["y_score"] >= float(threshold)).astype(int)
    df["threshold"] = float(threshold)
    df["threshold_method"] = threshold_method
    df["split"] = "test"
    fields = {
        "experiment_id": experiment.get("experiment_id"),
        "architecture": experiment.get("architecture"),
        "dataset_variant": experiment.get("training_dataset_variant", experiment.get("dataset_variant")),
        "synthetic_source": experiment.get("synthetic_source"),
        "training_mode": experiment.get("training_mode"),
        "checkpoint_path": experiment.get("checkpoint_path"),
    }
    for key, value in fields.items():
        df[key] = value
    if df["patient_id"].isna().any():
        raise ValueError("patient_id cannot be missing")
    if require_unique_patients and df["patient_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate patient_id values require explicit aggregation")
    return df[PREDICTION_COLUMNS].sort_values("patient_id", key=lambda x: x.astype(str)).reset_index(drop=True)


def aggregate_predictions_by_patient(frame: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    if method not in {"mean", "max"}:
        raise ValueError("Patient aggregation must be 'mean' or 'max'")
    required = {"patient_id", "y_true", "y_score", "threshold"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing aggregation columns: {sorted(required - set(frame.columns))}")
    if (frame.groupby("patient_id")["y_true"].nunique() > 1).any():
        raise ValueError("Conflicting labels for the same patient")
    score_fn = "mean" if method == "mean" else "max"
    first_cols = [c for c in frame.columns if c not in {"y_score", "y_pred", "image_id", "path"}]
    grouped = frame.groupby("patient_id", as_index=False).agg(
        **{c: (c, "first") for c in first_cols if c != "patient_id"},
        y_score=("y_score", score_fn), n_images_aggregated=("y_score", "size"),
    )
    grouped["y_pred"] = (grouped["y_score"] >= grouped["threshold"]).astype(int)
    grouped["image_id"] = "aggregated"
    grouped["path"] = "aggregated"
    return grouped.sort_values("patient_id", key=lambda x: x.astype(str)).reset_index(drop=True)


def compute_binary_metrics(y_true: Sequence[int], y_score: Sequence[float], threshold: float, ece_bins: int = 10) -> dict[str, Any]:
    y, score = np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float)
    if len(y) == 0 or len(y) != len(score) or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Invalid binary metric inputs")
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    npv = tn / (tn + fn) if tn + fn else math.nan
    bins = np.linspace(0, 1, ece_bins + 1)
    bucket = np.clip(np.digitize(score, bins[1:-1], right=True), 0, ece_bins - 1)
    ece = sum(
        np.mean(bucket == i) * abs(np.mean(y[bucket == i]) - np.mean(score[bucket == i]))
        for i in range(ece_bins) if np.any(bucket == i)
    )
    two_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "n_positive": int(y.sum()), "n_negative": int((y == 0).sum()),
        "roc_auc": float(roc_auc_score(y, score)) if two_classes else math.nan,
        "pr_auc": float(average_precision_score(y, score)) if two_classes else math.nan,
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else math.nan,
        "precision": float(precision_score(y, pred, zero_division=0)), "npv": float(npv),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)), "brier_score": float(brier_score_loss(y, score)),
        "ece": float(ece), "threshold": float(threshold),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def write_prediction_manifest(path: str | Path, manifest: Mapping[str, Any], prediction_path: str | Path) -> dict[str, Any]:
    payload = dict(manifest)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload["prediction_file_signature"] = content_signature(prediction_path)
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(strict_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def prediction_cache_status(manifest_path: str | Path, expected: Mapping[str, Any], prediction_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.is_file():
        return {"status": "CACHE_MISSING", "incompatible_keys": ["manifest"]}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        incompatible = [key for key, value in expected.items() if payload.get(key) != strict_jsonable(value)]
        pred = Path(prediction_path) if prediction_path else path.with_name("test_predictions.csv")
        if not pred.is_file():
            incompatible.append("prediction_file")
        elif payload.get("prediction_file_signature") != content_signature(pred):
            incompatible.append("prediction_file_signature")
        return {"status": "CACHE_VALID" if not incompatible else "CACHE_INCOMPATIBLE", "incompatible_keys": sorted(set(incompatible))}
    except (OSError, ValueError, json.JSONDecodeError):
        return {"status": "CACHE_INCOMPATIBLE", "incompatible_keys": ["manifest_unreadable"]}


def validate_prediction_cache(manifest_path: str | Path, expected: Mapping[str, Any], prediction_path: str | Path | None = None) -> bool:
    return prediction_cache_status(manifest_path, expected, prediction_path)["status"] == "CACHE_VALID"


def compare_patient_sets(frames: Mapping[str, pd.DataFrame], canonical_patient_ids: Iterable[Any] | None = None) -> dict[str, pd.DataFrame]:
    if not frames:
        raise ValueError("At least one prediction frame is required")
    invalid = [name for name, frame in frames.items() if not {"patient_id", "y_true"}.issubset(frame.columns)]
    if invalid:
        raise ValueError(f"Individual patient predictions required; aggregate metrics supplied for: {invalid}")
    sets = {name: set(df["patient_id"].astype(str)) for name, df in frames.items()}
    target = set(str(x) for x in canonical_patient_ids) if canonical_patient_ids is not None else next(iter(sets.values()))
    mismatches = {name: {"missing": sorted(target - ids), "extra": sorted(ids - target)} for name, ids in sets.items() if ids != target}
    if mismatches:
        raise ValueError(f"Patient sets differ: {mismatches}")
    order = sorted(target)
    aligned = {}
    for name, df in frames.items():
        indexed = df.assign(patient_id=df["patient_id"].astype(str)).set_index("patient_id")
        if indexed.index.duplicated().any():
            raise ValueError(f"Duplicate patients in {name}")
        aligned[name] = indexed.loc[order].reset_index()
    labels = [aligned[name]["y_true"].to_numpy() for name in aligned]
    if any(not np.array_equal(labels[0], other) for other in labels[1:]):
        raise ValueError("Positive class labels differ after patient alignment")
    return aligned


def build_experiment_registry(registry: str | Path | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(registry, (str, Path)):
        payload = json.loads(Path(registry).read_text(encoding="utf-8"))
        records = payload.get("experiments", payload) if isinstance(payload, dict) else payload
    else:
        records = registry
    records = [dict(x) for x in records]
    ids = [x.get("experiment_id") for x in records]
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("Registry experiment_id values must be present and unique")
    for item in records:
        variant, mode = item.get("training_dataset_variant"), item.get("training_mode")
        if variant not in DATASET_VARIANTS or mode not in TRAINING_MODES:
            raise ValueError(f"Invalid taxonomy for {item['experiment_id']}: {variant}, {mode}")
        item.setdefault("scientifically_eligible", bool(item.get("eligible_for_final_selection") or item.get("required_for_final_pipeline")))
        item.setdefault("selected_by_validation", bool(item.get("required_for_final_pipeline")))
        item.setdefault("blocked_reason", None)
        if item.get("selected_by_validation") or item.get("required_for_final_pipeline"):
            canonical = canonical_test_prediction_paths(item)
            for key, value in canonical.items():
                if not item.get(key):
                    item[key] = value
        if item.get("eligible_for_final_selection") is False and not item.get("exclusion_reason"):
            raise ValueError(f"Excluded experiment {item['experiment_id']} needs a reason")
    return records


def build_test_coverage_table(registry: Sequence[Mapping[str, Any]], project_root: str | Path = ".") -> pd.DataFrame:
    root = Path(project_root)
    rows = []
    for raw in registry:
        item = dict(raw)
        canonical = canonical_test_prediction_paths(item)
        for key, value in canonical.items():
            if not item.get(key):
                item[key] = value
        exists = lambda key: bool(item.get(key)) and (root / str(item[key])).is_file()
        checkpoint, validation, predictions, metrics = (exists("checkpoint_path"), exists("validation_metrics_path"), exists("test_predictions_path"), exists("test_metrics_path"))
        threshold = item.get("validation_threshold")
        provenance_level = "invalid"
        if predictions and exists("test_predictions_manifest_path"):
            try:
                prediction_path = root / str(item["test_predictions_path"])
                manifest_path = root / str(item["test_predictions_manifest_path"])
                prediction_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                provenance_level = str(prediction_manifest.get("provenance_level", "invalid"))
                required_manifest_fields = {
                    "experiment_id", "checkpoint_signature", "validation_metrics_signature",
                    "validation_threshold", "threshold_method", "patient_ids_hash",
                    "preprocessing", "model_config", "prediction_file_signature",
                    "pipeline_schema_version", "provenance_level", "test_used_for_selection",
                }
                incompatible = sorted(required_manifest_fields - set(prediction_manifest))
                checkpoint_path = root / str(item.get("checkpoint_path", ""))
                validation_path = root / str(item.get("validation_metrics_path", ""))
                expected_pairs = {
                    "experiment_id": item.get("experiment_id"),
                    "checkpoint_signature": content_signature(checkpoint_path) if checkpoint_path.is_file() else None,
                    "validation_metrics_signature": content_signature(validation_path) if validation_path.is_file() else None,
                    "validation_threshold": item.get("validation_threshold"),
                    "threshold_method": item.get("validation_threshold_method"),
                    "prediction_file_signature": content_signature(prediction_path),
                    "test_used_for_selection": False,
                }
                incompatible.extend(
                    key for key, expected in expected_pairs.items()
                    if prediction_manifest.get(key) != strict_jsonable(expected)
                )
                dataset_manifest_path = root / "results/final_evaluation/test_dataset_manifest.json"
                if dataset_manifest_path.is_file():
                    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
                    if prediction_manifest.get("test_dataset_manifest_signature") != content_signature(dataset_manifest_path):
                        incompatible.append("test_dataset_manifest_signature")
                    if prediction_manifest.get("patient_ids_hash") != dataset_manifest.get("patient_ids_hash"):
                        incompatible.append("patient_ids_hash")
                    test_csv = root / str(item.get("test_csv", "data/processed/metadata/test.csv"))
                    if not test_csv.is_file() or dataset_manifest.get("csv_signature") != (
                        content_signature(test_csv) if test_csv.is_file() else None
                    ):
                        incompatible.append("test_csv_signature")
                    for raw_path, recorded_signature in dataset_manifest.get("image_signatures", {}).items():
                        image_path = Path(raw_path)
                        image_path = image_path if image_path.is_absolute() else root / image_path
                        if not image_path.is_file() or content_signature(image_path) != recorded_signature:
                            incompatible.append("image_signatures")
                            break
                else:
                    incompatible.append("test_dataset_manifest_signature")
                if not isinstance(prediction_manifest.get("preprocessing"), Mapping) or not prediction_manifest.get("preprocessing"):
                    incompatible.append("preprocessing")
                if not isinstance(prediction_manifest.get("model_config"), Mapping) or not prediction_manifest.get("model_config"):
                    incompatible.append("model_config")
                if prediction_manifest.get("pipeline_schema_version") != SCHEMA_VERSION:
                    incompatible.append("pipeline_schema_version")
                if incompatible:
                    provenance_level = "invalid"
            except (OSError, ValueError, json.JSONDecodeError):
                provenance_level = "invalid"
        provenance = provenance_level in {"verified_native", "verified_recomputed"}
        missing = []
        if not checkpoint: missing.append("checkpoint")
        if not validation: missing.append("validation_metrics")
        if threshold is None: missing.append("validation_threshold")
        if item.get("required_for_final_pipeline") and not predictions and not item.get("test_notebook"): missing.append("test_predictions_or_notebook")
        if predictions and provenance: status = "TEST_PREDICTIONS_AVAILABLE"
        elif checkpoint and validation and threshold is not None and item.get("test_notebook"): status = "READY_FOR_TEST_INFERENCE"
        elif item.get("exclusion_reason") and any(
            marker in item.get("exclusion_reason", "").lower()
            for marker in ("escluso sulla validation", "seleziona un solo vincitore sulla validation", "ha auc validation superiore")
        ): status = "EXCLUDED_ON_VALIDATION"
        elif predictions and not provenance: status = "INVALID_PROVENANCE"
        elif missing: status = "INCOMPLETE"
        elif not item.get("required_for_final_pipeline"): status = "NOT_REQUIRED"
        else: status = "INCOMPLETE"
        # Three separate, non-ambiguous readiness signals instead of one conflated flag:
        # - inference_ready: a locked notebook exists and CAN be run (says nothing about whether
        #   it has actually been run yet).
        # - test_predictions_ready / final_aggregation_ready: verified predictions already exist.
        # operational_ready (kept for back-compat) is algebraically inference_ready OR
        # test_predictions_ready -- the exact same value the old single-flag formula produced.
        inference_ready = bool(checkpoint and validation and threshold is not None and item.get("test_notebook"))
        test_predictions_ready = bool(checkpoint and validation and threshold is not None and predictions and provenance)
        final_aggregation_ready = test_predictions_ready
        operational_ready = bool(inference_ready or test_predictions_ready)
        if final_aggregation_ready:
            blocker = None
        elif predictions and not provenance:
            blocker = "unverified_prediction_provenance"
        elif inference_ready:
            blocker = "test_inference_not_yet_run"
        else:
            blocker = ";".join(missing) or None
        effective_blocker = None if final_aggregation_ready else (
            blocker if inference_ready else (item.get("blocked_reason") or blocker)
        )
        item.update({
            "checkpoint_available": checkpoint, "validation_metrics_available": validation,
            "validation_threshold_available": threshold is not None,
            "test_predictions_available": predictions, "test_metrics_available": metrics,
            "test_inference_available": predictions, "provenance_valid": provenance,
            "provenance_level": provenance_level,
            "inference_ready": inference_ready, "test_predictions_ready": test_predictions_ready,
            "final_aggregation_ready": final_aggregation_ready, "operationally_ready": operational_ready,
            "blocked_reason": effective_blocker,
            "missing_components": ";".join(missing), "status": status,
            "recommended_action": item.get("recommended_action") or (
                "reuse_standardized_predictions" if provenance else
                "run_locked_test_notebook" if status == "READY_FOR_TEST_INFERENCE" else
                "document_exclusion" if status in {"NOT_REQUIRED", "EXCLUDED_ON_VALIDATION"} else
                "complete_missing_components"
            ),
        })
        if item["final_aggregation_ready"] and not item["test_predictions_ready"]:
            raise AssertionError("final_aggregation_ready requires test_predictions_ready")
        if item["status"] == "TEST_PREDICTIONS_AVAILABLE" and not item["test_predictions_ready"]:
            raise AssertionError("TEST_PREDICTIONS_AVAILABLE requires verified prediction files")
        if item["status"] == "READY_FOR_TEST_INFERENCE" and not item["inference_ready"]:
            raise AssertionError("READY_FOR_TEST_INFERENCE requires checkpoint, validation and notebook")
        rows.append(item)
    return pd.DataFrame(rows)


def select_validation_finalists(
    rows: pd.DataFrame | Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Select on validation fields only; columns prefixed test_ are ignored."""
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    cfg = {
        "include_baseline_per_architecture": True, "include_best_synthetic_per_architecture": True,
        "include_best_augmented_per_architecture": True, "include_best_overall_per_architecture": True,
        "max_finalists_total": 8,
    }
    cfg.update(policy or {})
    eligibility_column = "scientifically_eligible" if "scientifically_eligible" in df else "eligible_for_final_selection"
    eligible = df[df[eligibility_column].fillna(False).astype(bool)].copy()
    eligible = eligible[pd.to_numeric(eligible["validation_roc_auc"], errors="coerce").notna()]
    if eligible.empty:
        return []
    eligible["validation_roc_auc"] = pd.to_numeric(eligible["validation_roc_auc"])
    eligible["validation_pr_auc"] = pd.to_numeric(eligible.get("validation_pr_auc", np.nan), errors="coerce")
    reasons: dict[str, set[str]] = {}
    chosen: dict[str, dict[str, Any]] = {}
    def add(group: pd.DataFrame, reason: str) -> None:
        if group.empty: return
        row = group.sort_values(["validation_roc_auc", "validation_pr_auc"], ascending=False, na_position="last").iloc[0]
        eid = str(row["experiment_id"]); chosen[eid] = row.to_dict(); reasons.setdefault(eid, set()).add(reason)
    for arch, group in eligible.groupby("architecture", sort=True):
        if cfg["include_baseline_per_architecture"]: add(group[group["training_dataset_variant"] == "real_only"], "baseline_per_architecture")
        synth = group[group["training_dataset_variant"].isin(["real_plus_synthetic", "synthetic_only", "real_plus_synthetic_positive", "real_plus_augmented_plus_synthetic"])]
        if cfg["include_best_synthetic_per_architecture"]: add(synth, "best_synthetic_per_architecture")
        aug = group[group["training_dataset_variant"].isin(["real_plus_augmented", "real_plus_augmented_plus_synthetic"])]
        if cfg["include_best_augmented_per_architecture"]: add(aug, "best_augmented_per_architecture")
        if cfg["include_best_overall_per_architecture"]: add(group, "best_overall_per_architecture")
    primary_mask = eligible.get("required_for_primary_comparison", pd.Series(False, index=eligible.index)).fillna(False).astype(bool)
    for _, row in eligible[primary_mask].iterrows():
        eid = str(row["experiment_id"]); chosen[eid] = row.to_dict(); reasons.setdefault(eid, set()).add("primary_comparison")
    mandatory_ids = {
        str(row["experiment_id"]) for _, row in eligible.iterrows()
        if bool(row.get("required_for_primary_comparison", False))
        or bool(row.get("mandatory_baseline", False))
        or bool(row.get("scientifically_mandatory", False))
    }
    limit = int(cfg["max_finalists_total"])
    if len(mandatory_ids) > limit:
        raise RuntimeError("mandatory finalists exceed max_finalists_total")
    priority = {"primary_comparison": 0, "mandatory_baseline": 1, "scientifically_mandatory": 2,
                "best_overall_per_architecture": 3, "best_synthetic_per_architecture": 4,
                "best_augmented_per_architecture": 5, "baseline_per_architecture": 6}
    ranked = sorted(chosen.values(), key=lambda r: (
        min((priority.get(reason, 99) for reason in reasons[str(r["experiment_id"])]), default=99),
        -float(r["validation_roc_auc"]), str(r["experiment_id"])))
    retained = [row for row in ranked if str(row["experiment_id"]) in mandatory_ids]
    retained_ids = {str(row["experiment_id"]) for row in retained}
    for row in ranked:
        if len(retained) >= limit:
            break
        if str(row["experiment_id"]) not in retained_ids:
            retained.append(row)
            retained_ids.add(str(row["experiment_id"]))
    ranked = retained
    return [{**r, "selection_reason": sorted(reasons[str(r["experiment_id"])])} for r in ranked]


# Output key -> source key on a select_validation_finalists()/coverage-table row. Two keys are
# deliberately renamed (threshold_method, test_status) to match what lock_validation_finalists.py
# has always written and what 04y's real-run cell already reads (finalist["threshold_method"]) --
# a second, independent construction of this dict previously used different names and silently
# dropped several fields; this is now the single place that defines a "locked finalist entry".
FINALIST_LOCK_FIELD_MAP = {
    "experiment_id": "experiment_id", "selection_reason": "selection_reason", "checkpoint_path": "checkpoint_path",
    "validation_threshold": "validation_threshold", "threshold_method": "validation_threshold_method",
    "test_status": "status", "test_notebook": "test_notebook", "test_predictions_path": "test_predictions_path",
    "test_predictions_manifest_path": "test_predictions_manifest_path", "scientifically_eligible": "scientifically_eligible",
    "selected_by_validation": "selected_by_validation", "operationally_ready": "operationally_ready",
    "inference_ready": "inference_ready", "test_predictions_ready": "test_predictions_ready",
    "final_aggregation_ready": "final_aggregation_ready", "blocked_reason": "blocked_reason",
    "validation_metrics_source": "validation_metrics_source", "validation_threshold_source": "validation_threshold_source",
    "provenance_level": "provenance_level",
}


def build_locked_finalist_entries(finalists: Sequence[Mapping[str, Any]], project_root: str | Path = ".") -> list[dict[str, Any]]:
    """Build the per-finalist entries for ``finalists_manifest.json``.

    Preserves the full scientific/operational field set (see ``FINALIST_LOCK_FIELD_MAP``) and
    never invents a checkpoint signature: a missing checkpoint yields ``checkpoint_signature =
    None`` rather than raising, so a scientifically valid but operationally blocked finalist (e.g.
    a ResNet entry with no recoverable checkpoint) can still be locked.
    """
    root = Path(project_root)
    entries = []
    for item in finalists:
        checkpoint_path = item.get("checkpoint_path")
        checkpoint = root / str(checkpoint_path) if checkpoint_path else None
        entry = {out_key: item.get(src_key) for out_key, src_key in FINALIST_LOCK_FIELD_MAP.items()}
        entry.update(canonical_test_prediction_paths(item))
        entry["checkpoint_signature"] = content_signature(checkpoint) if checkpoint is not None and checkpoint.is_file() else None
        entries.append(entry)
    return entries


def lock_finalists_manifest(finalists: Sequence[Mapping[str, Any]], selection_policy: Mapping[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    """Freeze the validation-only finalist selection.

    ``scientific_selection_complete`` and ``final_aggregation_complete`` are deliberately
    separate: the scientific selection (which experiments are finalists) is settled the moment
    every finalist is a validated, eligible candidate, regardless of whether any inference has
    actually run yet. ``final_aggregation_complete`` only turns ``True`` once every finalist has
    real, verified test predictions (``final_aggregation_ready``) -- that is what gates whether
    the locked test/statistics pipeline is allowed to run for real. ``selection_complete`` is
    kept as a value-equal legacy alias of ``final_aggregation_complete`` for old readers; new code
    should read the two explicit fields instead.
    """
    blockers = [
        {"experiment_id": item.get("experiment_id"), "blocked_reason": item.get("blocked_reason") or "test_inference_not_yet_run"}
        for item in finalists if item.get("final_aggregation_ready") is not True
    ]
    scientific_selection_complete = all(
        bool(item.get("scientifically_eligible")) and bool(item.get("selected_by_validation")) for item in finalists
    )
    final_aggregation_complete = not blockers
    payload = {
        "schema_version": SCHEMA_VERSION, "selection_timestamp": datetime.now(timezone.utc).isoformat(),
        "selection_policy": dict(selection_policy), "primary_metric": "validation_roc_auc",
        "secondary_metric": "validation_pr_auc", "test_metrics_accessed": False,
        "locked": True,
        "scientific_selection_complete": scientific_selection_complete,
        "final_aggregation_complete": final_aggregation_complete,
        "selection_complete": final_aggregation_complete,
        "operational_blockers": blockers,
        "finalists": [_jsonable(dict(x)) for x in finalists],
    }
    payload["lock_signature"] = value_signature({k: v for k, v in payload.items() if k != "lock_signature"})
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(strict_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def validate_locked_finalists_manifest(
    manifest: str | Path | Mapping[str, Any], *, require_operational_complete: bool = True,
) -> dict[str, Any]:
    """Validate a locked finalists manifest.

    The scientific lock (``locked``, ``test_metrics_accessed``, structure, signature) is always
    checked. Whether the *operational* aggregation must already be complete
    (``final_aggregation_complete``) is controlled by ``require_operational_complete`` so callers
    can distinguish "the scientific selection is frozen" from "every finalist has real, verified
    test predictions" — these are not the same thing and must not be conflated.
    """
    payload = json.loads(Path(manifest).read_text(encoding="utf-8")) if isinstance(manifest, (str, Path)) else dict(manifest)
    if payload.get("locked") is not True or payload.get("test_metrics_accessed") is not False:
        raise ValueError("Finalists manifest is not a validation-only lock")
    if require_operational_complete and payload.get("final_aggregation_complete") is False:
        raise RuntimeError(f"Final aggregation is not complete yet: {payload.get('operational_blockers', [])}")
    expected = value_signature({k: v for k, v in payload.items() if k != "lock_signature"})
    if payload.get("lock_signature") != expected:
        raise ValueError("Finalists manifest lock signature mismatch")
    return payload


def paired_stratified_bootstrap(
    y_true: Sequence[int], scores_a: Sequence[float], scores_b: Sequence[float],
    metric: str = "roc_auc", iterations: int = 5000, seed: int = 42, ci_level: float = 0.95,
) -> dict[str, Any]:
    y, a, b = map(np.asarray, (y_true, scores_a, scores_b))
    if not (len(y) == len(a) == len(b)) or len(np.unique(y)) != 2:
        raise ValueError("Paired bootstrap requires aligned binary observations")
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng, diffs = np.random.default_rng(seed), []
    metric_fn = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}.get(metric)
    if metric_fn is None: raise ValueError("Unsupported bootstrap metric")
    for _ in range(iterations):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        try: diffs.append(float(metric_fn(y[idx], a[idx]) - metric_fn(y[idx], b[idx])))
        except ValueError: pass
    if not diffs:
        return {"status": "not_computable", "valid_iterations": 0, "requested_iterations": iterations}
    values = np.asarray(diffs); alpha = 1 - ci_level
    lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min((np.sum(values <= 0) + 1) / (len(values) + 1), (np.sum(values >= 0) + 1) / (len(values) + 1)))
    return {"status": "ok", "metric": metric, "mean_difference": float(values.mean()), "ci_lower": float(lower), "ci_upper": float(upper), "p_bootstrap": float(p), "valid_iterations": int(len(values)), "requested_iterations": int(iterations)}


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x); sorted_x = x[order]; n = len(x); ranks = np.empty(n, dtype=float); i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]: j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(n, dtype=float); out[order] = ranks
    return out


def _fast_delong(predictions: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m, n = label_1_count, predictions.shape[1] - label_1_count
    pos, neg = predictions[:, :m], predictions[:, m:]
    tx = np.vstack([_compute_midrank(row) for row in pos]); ty = np.vstack([_compute_midrank(row) for row in neg]); tz = np.vstack([_compute_midrank(row) for row in predictions])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01, bias=False)); sy = np.atleast_2d(np.cov(v10, bias=False))
    return aucs, sx / m + sy / n


def delong_roc_test(y_true: Sequence[int], scores_a: Sequence[float], scores_b: Sequence[float]) -> dict[str, Any]:
    y, a, b = np.asarray(y_true, int), np.asarray(scores_a, float), np.asarray(scores_b, float)
    base = {"status": "not_computable", "auc_a": math.nan, "auc_b": math.nan, "difference": math.nan, "variance": math.nan, "z_score": math.nan, "p_value": math.nan}
    if not (len(y) == len(a) == len(b)) or len(np.unique(y)) != 2 or min(np.sum(y == 0), np.sum(y == 1)) < 2: return base
    order = np.argsort(-y); m = int(y.sum())
    try:
        aucs, cov = _fast_delong(np.vstack([a, b])[:, order], m)
        variance = float(np.array([1, -1]) @ cov @ np.array([1, -1]))
        if not np.isfinite(variance) or variance <= 0: return {**base, "auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "difference": float(aucs[0] - aucs[1]), "variance": variance}
        z = float((aucs[0] - aucs[1]) / math.sqrt(variance)); p = float(2 * norm.sf(abs(z)))
        return {"status": "ok", "auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "difference": float(aucs[0] - aucs[1]), "variance": variance, "z_score": z, "p_value": p}
    except (ValueError, FloatingPointError, ZeroDivisionError): return base


def mcnemar_test(y_true: Sequence[int], pred_a: Sequence[int], pred_b: Sequence[int], exact_cutoff: int = 25) -> dict[str, Any]:
    y, a, bpred = map(np.asarray, (y_true, pred_a, pred_b))
    if not (len(y) == len(a) == len(bpred)): raise ValueError("McNemar inputs must be paired")
    correct_a, correct_b = a == y, bpred == y
    b = int(np.sum(correct_a & ~correct_b)); c = int(np.sum(~correct_a & correct_b)); discordant = b + c
    if discordant == 0: return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "exact_binomial"}
    if discordant < exact_cutoff:
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": float(binomtest(min(b, c), discordant, 0.5).pvalue), "method": "exact_binomial"}
    statistic = (abs(b - c) - 1) ** 2 / discordant
    return {"b": b, "c": c, "statistic": float(statistic), "p_value": float(chi2.sf(statistic, 1)), "method": "chi_square_continuity_corrected"}


def holm_adjustment(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float); adjusted = np.full(len(values), np.nan); valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]; running = 0.0; m = len(order)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * values[idx]); adjusted[idx] = min(1.0, running)
    return adjusted


__all__ = [
    "ARCHITECTURE_FAMILY", "FINALIST_LOCK_FIELD_MAP", "PREDICTION_COLUMNS",
    "aggregate_predictions_by_patient", "build_experiment_registry", "build_locked_finalist_entries",
    "build_test_coverage_table", "build_test_dataset_manifest", "canonical_test_prediction_paths", "checkpoint_key_mismatch", "compare_patient_sets",
    "compute_binary_metrics", "content_signature", "delong_roc_test", "holm_adjustment",
    "load_validation_threshold", "lock_finalists_manifest", "mcnemar_test", "paired_stratified_bootstrap",
    "patient_ids_hash", "prediction_cache_status", "select_validation_finalists", "standardize_prediction_dataframe",
    "strict_json_dumps", "strict_jsonable", "unwrap_checkpoint_state_dict",
    "validate_locked_finalists_manifest", "validate_locked_test_configuration",
    "validate_prediction_cache", "value_signature", "write_prediction_manifest",
]
