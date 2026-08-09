"""Notebook-first utilities for the validation-only generator benchmark.

Importing this module never loads a model, opens the test split, or starts computation.
The notebook calls each stage explicitly and may reuse cached embeddings.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
FAMILIES = ("finetuned", "from_scratch")
REPRESENTATIONS = ("raw", "filtered")
FEATURE_SPACES = ("inception_v3", "rad_dino")
BENCHMARK_ROOT = Path("results/2_diffusers/benchmark")
TRAIN_METADATA = Path("data/processed/metadata/train.csv")
VALIDATION_METADATA = Path("data/processed/metadata/val.csv")
AUGMENTATION_METADATA = Path("data/real_augmented/metadata.csv")
CANONICAL_OUTPUTS = (
    "candidate_audit.csv", "generator_summary.csv", "generator_ranking.csv",
    "resampling_plan.json", "paired_generator_differences.csv",
    "selection_summary.json", "figures/generator_summary.png",
)
class NonFiniteEmbeddingError(ValueError):
    """A feature extractor produced non-finite values for explicit samples."""

    def __init__(self, extractor: str, failures: Sequence[Mapping[str, Any]]):
        self.extractor = str(extractor)
        self.failures = [dict(row) for row in failures]
        details = "; ".join(
            f"{row['sample_id']} ({row['path']}): {row['cause']}" for row in self.failures
        )
        super().__init__(f"non-finite {self.extractor} embeddings: {details}")


def atomic_json(path: Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def load_protocol(root: Path) -> dict[str, Any]:
    protocol = json.loads((Path(root) / "configs/generator_benchmark_protocol.json").read_text())
    validate_protocol(protocol)
    return protocol


def load_registry(root: Path) -> dict[str, Any]:
    registry = json.loads((Path(root) / "configs/generator_registry.json").read_text())
    ids = [entry["id"] for entry in registry.get("generators", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("generator registry contains duplicate IDs")
    return registry


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if float(protocol.get("protocol_version", 0)) != 2.1:
        raise ValueError("unsupported generator benchmark protocol")
    if int(protocol.get("synthetic_pool_target", 0)) <= 1:
        raise ValueError("synthetic_pool_target must exceed one")
    if tuple(protocol.get("representations", [])) != REPRESENTATIONS:
        raise ValueError("raw and filtered representations must be registered separately")
    technical_policy = protocol.get("technical_validity_policy", {})
    raw_policy, filtered_policy = technical_policy.get("raw", {}), technical_policy.get("filtered", {})
    if raw_policy.get("near_black") != "warning_and_include" or \
            raw_policy.get("constant_range") != "warning_and_include":
        raise ValueError("RAW quality defects must be retained as warnings")
    if filtered_policy.get("near_black") != "fatal_for_official_ranking" or \
            filtered_policy.get("constant_range") != "fatal_for_official_ranking":
        raise ValueError("FILTERED quality defects must remain fatal for official ranking")
    if protocol.get("selection", {}).get("official_representation") != "filtered":
        raise ValueError("official family ranking must use FILTERED")
    if "test" in str(protocol.get("reference_sets", {}).get("distribution_metrics", "")).lower():
        raise ValueError("generator selection cannot use test data")
    resampling = protocol.get("resampling", {})
    if resampling.get("replace") is not False:
        raise ValueError("KID and PRDC repeated subsampling must use replace=False")
    fraction = float(resampling.get("subsampling_fraction", 0.0))
    if not 0.0 < fraction < 1.0:
        raise ValueError("subsampling_fraction must be strictly between zero and one")
    if int(resampling.get("stability_repetitions", 0)) < 2:
        raise ValueError("stability_repetitions must be at least two")
    if float(protocol.get("selection", {}).get("practical_equivalence_margin", -1)) < 0:
        raise ValueError("practical_equivalence_margin must be declared before comparison")
    if int(resampling.get("fid_repetitions", 999)) > 10:
        raise ValueError("FID is secondary and must use a small independent repetition count")
    ranking_metrics = [item["metric"] for item in protocol["selection"]["ranking"]]
    expected_ranking = ["raddino_kid", "raddino_coverage", "raddino_precision", "raddino_fid",
                        "inception_kid", "raddino_kid_std", "generator_id"]
    if ranking_metrics != expected_ranking:
        raise ValueError("ranking fields must match the canonical flat generator_summary columns")
    if protocol["selection"].get("primary_metric") != "raddino_kid":
        raise ValueError("KID must be the primary ranking metric")
    train_reference = str(protocol.get("reference_sets", {}).get("train_memorization", ""))
    if train_reference != "processed_train_and_traditional_augmentation_metadata" or "positive_only" in train_reference:
        raise ValueError("train memorization must use train metadata and traditional augmentations")
    fid_rows = [item for item in protocol["selection"]["ranking"] if "fid" in item["metric"]]
    if any(item.get("role") != "descriptive_tiebreak" for item in fid_rows):
        raise ValueError("FID may appear only as a descriptive tie-break")


def _reject_test_path(path: str | Path) -> None:
    parts = {part.lower().replace("-", "_") for part in Path(path).parts}
    if any(part in {"test", "locked_test", "historical_internal_test"} or part.startswith("test_") for part in parts):
        raise PermissionError(f"test access is forbidden during generator benchmarking: {path}")


def list_image_paths(path: Path) -> list[Path]:
    _reject_test_path(path)
    if not Path(path).is_dir():
        return []
    return sorted(item for item in Path(path).rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def metadata_positive_paths(root: Path, relative_csv: str) -> tuple[list[str], list[str]]:
    _reject_test_path(relative_csv)
    with (Path(root) / relative_csv).open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["label"]) == 1]
    paths, ids = [], []
    split = Path(relative_csv).stem
    for row in rows:
        value = row.get("processed_path") or f"data/processed/{split}/1/{row['image_id']}.png"
        _reject_test_path(value)
        paths.append(str((Path(root) / value).resolve()))
        ids.append(f"{row.get('patient_id')}::{row.get('image_id')}")
    return paths, ids


def training_corpus_from_metadata(root: Path, *,
                                  verify_files: bool = True) -> tuple[list[str], list[str], dict[str, int], dict[str, str]]:
    """Build the memorization pool in memory from existing train metadata.

    Traditional augmentations are included when their existing metadata file is
    present. Validation metadata is used only for the patient-overlap check; the
    classifier test split is never opened by the generator benchmark.
    """
    root = Path(root).resolve()
    train_csv = root / TRAIN_METADATA
    with train_csv.open(newline="", encoding="utf-8") as stream:
        train_rows = list(csv.DictReader(stream))
    required = {"patient_id", "image_id", "label", "split", "processed_path"}
    if not train_rows or required - set(train_rows[0]):
        raise ValueError(f"train metadata is empty or missing fields: {sorted(required - set(train_rows[0] if train_rows else {}))}")

    validation_patients: set[str] = set()
    validation_csv = root / VALIDATION_METADATA
    if not validation_csv.is_file():
        raise FileNotFoundError(f"validation metadata is missing: {validation_csv}")
    with validation_csv.open(newline="", encoding="utf-8") as stream:
        validation_patients = {str(row["patient_id"]).strip() for row in csv.DictReader(stream)}

    records: list[tuple[str, str, int, str]] = []
    real_by_key: dict[tuple[str, str], tuple[int, str]] = {}
    for row in train_rows:
        patient_id = str(row["patient_id"]).strip()
        image_id = str(row["image_id"]).strip()
        sample_id = f"real::{patient_id}::{image_id}"
        if str(row["split"]).strip().lower() != "train":
            raise ValueError(f"train metadata contains non-train split: {sample_id}")
        if not patient_id or patient_id in validation_patients:
            raise ValueError(f"train patient appears in validation or is missing: {patient_id!r}")
        label = int(row["label"])
        if label not in {0, 1}:
            raise ValueError(f"train metadata contains invalid label: {sample_id}")
        value = str(row["processed_path"]).strip()
        _reject_test_path(value)
        if Path(value).parts[:3] != ("data", "processed", "train"):
            raise ValueError(f"train metadata path is outside data/processed/train: {value}")
        records.append((sample_id, value, label, "real_train"))
        key = (patient_id, image_id)
        if key in real_by_key:
            raise ValueError(f"duplicate patient/image in train metadata: {key}")
        real_by_key[key] = (label, value)

    augmentation_csv = root / AUGMENTATION_METADATA
    if augmentation_csv.is_file():
        with augmentation_csv.open(newline="", encoding="utf-8") as stream:
            augmentation_rows = list(csv.DictReader(stream))
        augmentation_required = {"file_name", "label", "patient_id", "image_id", "source"}
        if augmentation_rows and augmentation_required - set(augmentation_rows[0]):
            raise ValueError(f"augmentation metadata is missing fields: {sorted(augmentation_required - set(augmentation_rows[0]))}")
        for row in augmentation_rows:
            if str(row["source"]).strip().lower() == "real":
                continue
            patient_id = str(row["patient_id"]).strip()
            image_id = str(row["image_id"]).strip()
            key = (patient_id, image_id)
            if key not in real_by_key:
                raise ValueError(f"augmentation has no matching train sample: {key}")
            label = int(row["label"])
            if label != real_by_key[key][0]:
                raise ValueError(f"augmentation label differs from train metadata: {key}")
            value = str(row["file_name"]).strip()
            _reject_test_path(value)
            if Path(value).parts[:2] != ("data", "real_augmented"):
                raise ValueError(f"augmentation path is outside data/real_augmented: {value}")
            sample_id = f"augmentation::{patient_id}::{image_id}::{Path(value).name}"
            records.append((sample_id, value, label, "traditional_augmentation"))

    paths, ids, labels, sources = [], [], {}, {}
    seen_paths: set[str] = set()
    for sample_id, value, label, source in records:
        path = root / value
        if verify_files and not path.is_file():
            raise FileNotFoundError(f"training corpus sample is missing: {value}")
        resolved = str(path.resolve())
        if resolved in seen_paths:
            raise ValueError(f"training metadata contains duplicate path: {value}")
        if sample_id in labels:
            raise ValueError(f"training metadata contains duplicate sample ID: {sample_id}")
        seen_paths.add(resolved)
        paths.append(resolved)
        ids.append(sample_id)
        labels[sample_id] = label
        sources[sample_id] = source
    return paths, ids, labels, sources


def evaluation_subset_size(synthetic_pool_count: int, real_reference_count: int, synthetic_pool_target: int = 1361,
                           subsampling_fraction: float = 1.0) -> int:
    """Validate the pool and return the balanced stability-subsample size."""
    synthetic_pool_count, real_reference_count = int(synthetic_pool_count), int(real_reference_count)
    if synthetic_pool_count < int(synthetic_pool_target):
        raise ValueError(f"synthetic pool below target: {synthetic_pool_count} < {synthetic_pool_target}")
    if real_reference_count < 2:
        raise ValueError("at least two real validation positives are required")
    fraction = float(subsampling_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("subsampling_fraction must be in (0, 1]")
    return int(math.floor(fraction * min(real_reference_count, synthetic_pool_count)))


def deterministic_sample(values: Sequence[str | Path], count: int, seed: int) -> list[str]:
    canonical = sorted(str(Path(value)) for value in values)
    if len(canonical) != len(set(canonical)):
        raise ValueError("candidate list contains duplicates")
    if len(canonical) < int(count):
        raise ValueError(f"need {count} unique candidates, found {len(canonical)}")
    return sorted(random.Random(int(seed)).sample(canonical, int(count)))


def balanced_subsample_indices(real_count: int, synthetic_count: int, subset_size: int, repetitions: int,
                               seed: int, *, nearest_neighbour_k: int | None = None) -> list[dict[str, Any]]:
    if subset_size > min(real_count, synthetic_count):
        raise ValueError("balanced subset exceeds an available pool")
    if nearest_neighbour_k is not None and subset_size <= int(nearest_neighbour_k):
        raise ValueError("PRDC subset_size must exceed nearest_neighbour_k")
    rows = []
    for repetition in range(int(repetitions)):
        rng = np.random.default_rng(int(seed) + repetition)
        rows.append({
            "repetition": repetition,
            "seed": int(seed) + repetition,
            "real_indices": rng.choice(real_count, subset_size, replace=False).tolist(),
            "synthetic_indices": rng.choice(synthetic_count, subset_size, replace=False).tolist(),
            "replace": False,
        })
    return rows


def _pairwise_squared(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    value = np.sum(left * left, axis=1)[:, None] + np.sum(right * right, axis=1)[None, :] - 2 * left @ right.T
    return np.maximum(value, 0.0)


def fid(real: np.ndarray, synthetic: np.ndarray) -> float:
    real, synthetic = np.asarray(real, dtype=np.float64), np.asarray(synthetic, dtype=np.float64)
    if min(len(real), len(synthetic)) < 2: raise ValueError("FID needs at least two samples per distribution")
    delta = real.mean(axis=0) - synthetic.mean(axis=0)
    centered_real, centered_synthetic = real - real.mean(axis=0), synthetic - synthetic.mean(axis=0)
    trace_real = float(np.sum(centered_real * centered_real) / (len(real) - 1))
    trace_synthetic = float(np.sum(centered_synthetic * centered_synthetic) / (len(synthetic) - 1))
    # Low-rank identity: Tr(sqrt(Cr Cs)) is the nuclear norm of the centered
    # cross-product divided by the covariance normalizers. This keeps FID usable
    # for 73 samples in 768/2048-dimensional feature spaces.
    cross = centered_real @ centered_synthetic.T / math.sqrt((len(real) - 1) * (len(synthetic) - 1))
    trace_root = float(np.linalg.svd(cross, compute_uv=False).sum())
    return float(delta @ delta + trace_real + trace_synthetic - 2 * trace_root)


def kid(real: np.ndarray, synthetic: np.ndarray) -> float:
    real, synthetic = np.asarray(real, dtype=np.float64), np.asarray(synthetic, dtype=np.float64)
    if min(len(real), len(synthetic)) < 2:
        raise ValueError("KID needs at least two samples per distribution")
    dimension = real.shape[1]
    rr, ss = (real @ real.T / dimension + 1.0) ** 3, (synthetic @ synthetic.T / dimension + 1.0) ** 3
    rs = (real @ synthetic.T / dimension + 1.0) ** 3
    return float((rr.sum() - np.trace(rr)) / (len(real) * (len(real) - 1))
                 + (ss.sum() - np.trace(ss)) / (len(synthetic) * (len(synthetic) - 1)) - 2 * rs.mean())


def prdc(real: np.ndarray, synthetic: np.ndarray, nearest_k: int = 5) -> dict[str, float]:
    real, synthetic = np.asarray(real, dtype=np.float64), np.asarray(synthetic, dtype=np.float64)
    if min(len(real), len(synthetic)) <= int(nearest_k):
        raise ValueError("PRDC subset_size must exceed nearest_neighbour_k")
    rr, ss, rs = (np.sqrt(_pairwise_squared(real, real)), np.sqrt(_pairwise_squared(synthetic, synthetic)),
                  np.sqrt(_pairwise_squared(real, synthetic)))
    real_radius = np.partition(rr, nearest_k, axis=1)[:, nearest_k]
    synthetic_radius = np.partition(ss, nearest_k, axis=1)[:, nearest_k]
    return {
        "precision": float(np.mean((rs < real_radius[:, None]).any(axis=0))),
        "recall": float(np.mean((rs < synthetic_radius[None, :]).any(axis=1))),
        "density": float(np.mean((rs < real_radius[:, None]).sum(axis=0)) / nearest_k),
        "coverage": float(np.mean(rs.min(axis=1) < real_radius)),
    }


def summarize(values: Sequence[float]) -> dict[str, Any]:
    valid = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    if not len(valid):
        return {"mean": None, "median": None, "standard_deviation": None,
                "percentile_2_5": None, "percentile_97_5": None, "valid_repetitions": 0}
    return {"mean": float(valid.mean()), "median": float(np.median(valid)),
            "standard_deviation": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
            "percentile_2_5": float(np.percentile(valid, 2.5)),
            "percentile_97_5": float(np.percentile(valid, 97.5)), "valid_repetitions": int(len(valid))}


def repeated_distribution_metrics(real: np.ndarray, synthetic: np.ndarray, protocol: Mapping[str, Any],
                                  *, seed: int | None = None,
                                  resampling_plan: Sequence[Mapping[str, Any]] | None = None
                                  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute a full estimate plus repeated-subsampling stability measurements."""
    cfg = protocol["resampling"]
    size = evaluation_subset_size(len(synthetic), len(real), int(protocol["synthetic_pool_target"]),
                                  float(cfg["subsampling_fraction"]))
    nearest_k = int(cfg["nearest_neighbour_k"])
    if size <= nearest_k:
        raise ValueError("stability subset_size must exceed nearest_neighbour_k")
    base_seed = int(seed if seed is not None else protocol["sampling"]["seed"])
    if resampling_plan is None:
        resampling_plan = balanced_subsample_indices(len(real), len(synthetic), size,
            int(cfg["stability_repetitions"]), base_seed, nearest_neighbour_k=nearest_k)
    plans = [dict(row) for row in resampling_plan]
    for plan in plans:
        if len(plan["real_indices"]) != size or len(plan["synthetic_indices"]) != size:
            raise ValueError("resampling plan does not match the protocol subset size")
        if max(plan["real_indices"], default=-1) >= len(real) or max(plan["synthetic_indices"], default=-1) >= len(synthetic):
            raise ValueError("resampling plan index exceeds an embedding pool")
    rows: list[dict[str, Any]] = []
    values: dict[str, list[float]] = {}
    for plan in plans:
        real_subset, synthetic_subset = real[plan["real_indices"]], synthetic[plan["synthetic_indices"]]
        measured = {"kid": kid(real_subset, synthetic_subset),
                    **prdc(real_subset, synthetic_subset, nearest_k=nearest_k)}
        rows.append({**plan, "metric_group": "kid_prdc_stability", "subset_size": size,
                     "interval_type": "repeated-subsampling stability interval", **measured})
        for metric, value in measured.items():
            values.setdefault(metric, []).append(float(value))
    # KID/FID use the complete declared pools; PRDC gets a separately balanced point estimate.
    full_synthetic_count = min(len(synthetic), int(protocol["synthetic_pool_target"]))
    synthetic_full = np.asarray(synthetic)[np.random.default_rng(base_seed).choice(
        len(synthetic), full_synthetic_count, replace=False)]
    real_full = np.asarray(real)
    balanced_count = min(len(real_full), len(synthetic_full))
    balanced_indices = np.random.default_rng(base_seed).choice(len(synthetic_full), balanced_count, replace=False)
    balanced_prdc = prdc(real_full, synthetic_full[balanced_indices], nearest_k=nearest_k)
    summaries = {metric: summarize(metric_values) for metric, metric_values in values.items()}
    return rows, {"full_pool_distribution_policy": "all real validation positives vs canonical synthetic pool",
                  "full_pool_real_count": len(real_full), "full_pool_synthetic_count": len(synthetic_full),
                  "full_pool_distribution_estimates": {"kid_full_pool": kid(real_full, synthetic_full),
                                                       "fid_full_pool": fid(real_full, synthetic_full)},
                  "fid_full_pool_caveat": "descriptive; the real validation-positive pool is small",
                  "balanced_prdc_point_real_count": len(real_full),
                  "balanced_prdc_point_synthetic_count": balanced_count,
                  "balanced_prdc_point_estimates": {
                      "precision_balanced_point": balanced_prdc["precision"],
                      "recall_balanced_point": balanced_prdc["recall"],
                      "density_balanced_point": balanced_prdc["density"],
                      "coverage_balanced_point": balanced_prdc["coverage"]},
                  "stability_subset_size": size, "stability_interval_type": "repeated-subsampling stability interval",
                  "stability_estimates": summaries}


def save_resampling_plan(path: Path, plan: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Path:
    return atomic_json(path, {"schema_version": 1, "interval_type": "repeated-subsampling stability interval",
        "subsampling_fraction": protocol["resampling"]["subsampling_fraction"], "repetitions": list(plan)})


def paired_kid_differences(left_rows: Sequence[Mapping[str, Any]], right_rows: Sequence[Mapping[str, Any]],
                           left_id: str = "left", right_id: str = "right") -> dict[str, Any]:
    left = {int(row["repetition"]): float(row["kid"]) for row in left_rows if row.get("kid") is not None}
    right = {int(row["repetition"]): float(row["kid"]) for row in right_rows if row.get("kid") is not None}
    if set(left) != set(right) or not left:
        raise ValueError("paired KID comparison requires the same non-empty repetition plan")
    differences = [left[index] - right[index] for index in sorted(left)]
    summary = summarize(differences)
    return {"left_generator_id": left_id, "right_generator_id": right_id,
            "difference_definition": "left_kid_minus_right_kid", "mean_paired_difference": summary["mean"],
            "median_paired_difference": summary["median"], "stability_interval_low": summary["percentile_2_5"],
            "stability_interval_high": summary["percentile_97_5"],
            "left_win_fraction": float(np.mean(np.asarray(differences) < 0)),
            "right_win_fraction": float(np.mean(np.asarray(differences) > 0)),
            "tie_fraction": float(np.mean(np.asarray(differences) == 0)), "paired_differences": differences,
            "interval_type": "repeated-subsampling stability interval"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_duplicate_generator_identities(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Annotate model duplicates and reject duplicate official ranking competitors.

    Same-model membership is taken from the registry's declarative ``duplicate_model_group`` (e.g.
    a sampling ablation that reuses another generator's checkpoint) rather than from a computed model
    identity hash. Two generators that share a declared model group cannot both enter the official
    family ranking.
    """
    annotated = []
    for row in rows:
        result = dict(row)
        members = sorted({str(member) for member in (result.get("duplicate_model_group") or [])})
        result["duplicate_model_group"] = "|".join(members) if len(members) > 1 else ""
        role = result.get("candidate_role")
        result["distinct_generator_for_ranking"] = bool(
            result.get("eligible_for_official_family_ranking") and role == "primary_candidate")
        annotated.append(result)
    by_group: dict[str, list[str]] = {}
    for row in annotated:
        key = row.get("duplicate_model_group")
        if key and row.get("distinct_generator_for_ranking"):
            by_group.setdefault(key, []).append(str(row.get("generator_id") or row.get("id")))
    for key, competitors in by_group.items():
        if len(competitors) > 1:
            raise ValueError(f"duplicate model group cannot enter official ranking twice: {sorted(competitors)}")
    return annotated


def _top_level_images(path: Path) -> list[Path]:
    _reject_test_path(path)
    return sorted(item for item in Path(path).iterdir()
                  if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES) if Path(path).is_dir() else []


def _identity_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def inception_v3_identity(torchvision_version: str, weights_enum: str, checkpoint_path: Path,
                          transform_configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Record the actual local torchvision weight artifact without initializing the model."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file(): raise FileNotFoundError(checkpoint_path)
    identity = {"library": "torchvision", "torchvision_version": str(torchvision_version),
                "weights_enum": str(weights_enum), "checkpoint_filename": checkpoint_path.name,
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "transform_configuration": dict(transform_configuration)}
    return {**identity, "identity_sha256": _identity_digest(identity)}


def rad_dino_identity(model_repository: str, resolved_cache_path: Path, *, commit_hash: str | None,
                      config_path: Path, weight_paths: Sequence[Path], processor_config_path: Path,
                      preprocessing_configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Record a fully local RAD-DINO snapshot; missing files defer the benchmark."""
    cache = Path(resolved_cache_path).resolve(); config = Path(config_path); processor = Path(processor_config_path)
    weights = sorted(Path(path) for path in weight_paths)
    if not cache.is_dir() or not config.is_file() or not processor.is_file() or not weights or not all(path.is_file() for path in weights):
        raise FileNotFoundError("complete local RAD-DINO config, processor and weights are required")
    shards = [{"filename": path.name, "sha256": file_sha256(path), "file_size": path.stat().st_size} for path in weights]
    composite = hashlib.sha256("".join(f"{row['filename']}:{row['sha256']}\n" for row in shards).encode()).hexdigest()
    identity = {"model_repository": str(model_repository), "resolved_local_cache_path": str(cache),
                "huggingface_commit_hash": commit_hash, "config_sha256": file_sha256(config),
                "weight_files": shards, "weight_composite_sha256": composite,
                "processor_config_sha256": file_sha256(processor),
                "preprocessing_configuration": dict(preprocessing_configuration)}
    return {**identity, "identity_sha256": _identity_digest(identity)}


def perceptual_hash(path: Path) -> str:
    from PIL import Image
    from scipy.fft import dctn
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float64)
    values = dctn(pixels, norm="ortho")[:8, :8].flatten()
    bits = values >= float(np.median(values[1:]))
    return f"{int(''.join('1' if bit else '0' for bit in bits), 2):016x}"


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def duplicate_diagnostics(paths: Sequence[str | Path], perceptual_distance: int = 2) -> dict[str, Any]:
    resolved = [Path(path) for path in paths]
    exact, hashes = [file_sha256(path) for path in resolved], [perceptual_hash(path) for path in resolved]
    exact_items = sum(count - 1 for count in Counter(exact).values() if count > 1)
    perceptual_items = sum(any(hamming_distance(value, earlier) <= perceptual_distance for earlier in hashes[:index])
                           for index, value in enumerate(hashes))
    total = len(resolved)
    return {"n_images": total, "synthetic_exact_duplicate_rate": exact_items / total if total else 0.0,
            "synthetic_duplicate_rate": perceptual_items / total if total else 0.0,
            "perceptual_hash_duplicate_rate": perceptual_items / total if total else 0.0}


def technical_audit(paths: Sequence[str | Path], expected_size: tuple[int, int] = (512, 512)) -> dict[str, Any]:
    from PIL import Image
    counters = Counter()
    feature_hashes, quality_hashes = [], []
    for value in paths:
        try:
            with Image.open(value) as image:
                image_size = image.size
                gray = np.asarray(image.convert("L"))
                counters["readable"] += 1
                wrong_shape = image_size != expected_size
                numeric_array = bool(
                    gray.size and np.issubdtype(gray.dtype, np.number) and np.isfinite(gray).all()
                )
                feature_extractable = numeric_array and not wrong_shape
                if wrong_shape: counters["wrong_shape"] += 1
                if not numeric_array: counters["invalid_numeric_array"] += 1
                if not feature_extractable:
                    continue
                gray = np.asarray(gray, dtype=np.uint8)
                counters["feature_extractable"] += 1
                feature_hashes.append(hashlib.sha256(gray.tobytes()).hexdigest())
                invalid_range = int(gray.max()) <= int(gray.min())
                near_black = np.count_nonzero(gray > 5) / gray.size < 0.01
                if invalid_range: counters["invalid_range"] += 1
                if near_black: counters["near_black"] += 1
                if not (invalid_range or near_black):
                    counters["quality_valid"] += 1
                    quality_hashes.append(hashlib.sha256(gray.tobytes()).hexdigest())
        except Exception:
            counters["corrupt"] += 1
    total = len(paths)
    feature_extractable = counters["feature_extractable"]
    quality_valid = counters["quality_valid"]
    unique_feature = len(set(feature_hashes))
    unique_quality = len(set(quality_hashes))
    return {
        "n_discovered": total, "n_readable": counters["readable"], "n_corrupt": counters["corrupt"],
        "n_wrong_shape": counters["wrong_shape"], "n_near_black": counters["near_black"],
        "n_invalid_range": counters["invalid_range"],
        "n_feature_extractable": feature_extractable,
        "n_feature_nonextractable": total - feature_extractable,
        "feature_extractable_rate": feature_extractable / total if total else 0.0,
        "n_unique_feature_extractable_content": unique_feature,
        "n_exact_duplicates_among_feature_extractable": max(0, feature_extractable - unique_feature),
        "n_quality_valid": quality_valid,
        "n_quality_invalid": total - quality_valid,
        "quality_validity_rate": quality_valid / total if total else 0.0,
        "n_unique_quality_valid_content": unique_quality,
        "n_exact_duplicates_among_quality_valid": max(0, quality_valid - unique_quality),
        # Compatibility aliases describe quality validity, not feature extractability.
        "n_technically_valid": quality_valid,
        "n_technically_invalid": total - quality_valid,
        "n_unique_valid_content": unique_quality,
        "n_exact_duplicates_among_valid": max(0, quality_valid - unique_quality),
        "technical_validity_rate": quality_valid / total if total else 0.0,
        # Backward-compatible rates used by the protocol gates.
        "n_unique_content": unique_quality,
        "n_exact_duplicates": max(0, quality_valid - unique_quality),
        "n_images": total, "corrupted_rate": counters["corrupt"] / total if total else 0.0,
        "unexpected_dimensions_rate": counters["wrong_shape"] / total if total else 0.0,
        "invalid_dynamic_range_rate": counters["invalid_range"] / total if total else 0.0,
        "near_black_rate": counters["near_black"] / total if total else 0.0,
    }


def technical_validity_row(generator_id: str, condition: str, paths: Sequence[str | Path], *,
                           minimum_unique: int = 1361, expected_size: tuple[int, int] = (512, 512),
                           maximum_exact_duplicate_rate: float = 0.01) -> dict[str, Any]:
    """Apply representation-aware execution and quality gates to one candidate condition."""
    audit = technical_audit(paths, expected_size)
    normalized = str(condition).strip().lower()
    if normalized not in REPRESENTATIONS:
        raise ValueError(f"unsupported representation: {condition}")
    fatal, warnings = [], []
    if audit["n_corrupt"]: fatal.append("corrupt_images")
    if audit["n_wrong_shape"]: fatal.append("wrong_shape")
    if audit["n_feature_extractable"] < int(minimum_unique): fatal.append("insufficient_feature_extractable_images")
    if audit["n_unique_feature_extractable_content"] < int(minimum_unique):
        fatal.append("insufficient_unique_feature_extractable_content")
    if normalized == "raw":
        if audit["n_near_black"]: warnings.append("near_black")
        if audit["n_invalid_range"]: warnings.append("invalid_range")
    else:
        if audit["n_near_black"]: fatal.append("near_black")
        if audit["n_invalid_range"]: fatal.append("invalid_range")
        if audit["n_quality_valid"] < int(minimum_unique): fatal.append("insufficient_quality_valid_images")
        if audit["n_unique_quality_valid_content"] < int(minimum_unique):
            fatal.append("insufficient_unique_quality_valid_content")
        duplicate_rate = (
            audit["n_exact_duplicates_among_quality_valid"] / audit["n_quality_valid"]
            if audit["n_quality_valid"] else 0.0
        )
        if duplicate_rate > float(maximum_exact_duplicate_rate):
            fatal.append("quality_valid_exact_duplicate_rate")
    fatal_text = "; ".join(dict.fromkeys(fatal))
    warning_text = "; ".join(dict.fromkeys(warnings))
    eligible = not fatal
    keys = (
        "n_discovered", "n_readable", "n_corrupt", "n_wrong_shape",
        "n_feature_extractable", "n_feature_nonextractable", "feature_extractable_rate",
        "n_unique_feature_extractable_content", "n_exact_duplicates_among_feature_extractable",
        "n_near_black", "n_invalid_range", "n_quality_valid", "n_quality_invalid",
        "quality_validity_rate", "n_unique_quality_valid_content",
        "n_exact_duplicates_among_quality_valid", "n_technically_valid", "n_technically_invalid",
        "n_unique_valid_content", "n_exact_duplicates_among_valid", "technical_validity_rate",
    )
    return {"generator_id": generator_id, "condition": normalized.upper(),
            **{key: audit[key] for key in keys},
            "eligible_for_distribution_metrics": eligible,
            "eligible_for_official_ranking": normalized == "filtered" and eligible,
            "quality_warning": bool(warnings),
            "warning_reasons": warning_text,
            "fatal_failure_reasons": fatal_text,
            # Compatibility alias: fatal reasons only, never quality warnings.
            "failure_reason": fatal_text}


def representation_preflight_rows(candidate_audits: Sequence[Mapping[str, Any]],
                                  technical_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Combine runtime asset availability and per-representation technical readiness."""
    technical = {(str(row["generator_id"]), str(row["condition"]).lower()): row
                 for row in technical_rows}
    rows = []
    for audit in candidate_audits:
        generator_id = str(audit["generator_id"])
        raw = technical.get((generator_id, "raw"))
        filtered = technical.get((generator_id, "filtered"))
        runtime_ready = bool(audit.get("eligible_for_benchmark_execution", False))
        descriptive = bool(audit.get("eligible_for_descriptive_benchmark", False))
        official = bool(audit.get("eligible_for_official_family_ranking", False))
        raw_ready = runtime_ready and descriptive and bool(raw and raw.get("eligible_for_distribution_metrics"))
        filtered_ready = runtime_ready and descriptive and bool(filtered and filtered.get("eligible_for_distribution_metrics"))
        filtered_official = filtered_ready and official and bool(filtered.get("eligible_for_official_ranking"))
        reasons = list(audit.get("blockers", []))
        if raw and raw.get("fatal_failure_reasons"): reasons.append(f"RAW:{raw['fatal_failure_reasons']}")
        if filtered and filtered.get("fatal_failure_reasons"):
            reasons.append(f"FILTERED:{filtered['fatal_failure_reasons']}")
        rows.append({"generator_id": generator_id, "scientific_family": audit.get("scientific_family"),
                     "candidate_role": audit.get("candidate_role"),
                     "raw_descriptive_ready": raw_ready,
                     "raw_quality_warning": bool(raw and raw.get("quality_warning")),
                     "filtered_descriptive_ready": filtered_ready,
                     "filtered_official_ranking_ready": filtered_official,
                     "block_reasons": "; ".join(dict.fromkeys(reasons))})
    return rows


def require_official_family_coverage(preflight_rows: Sequence[Mapping[str, Any]],
                                     families: Sequence[str] = FAMILIES) -> dict[str, int]:
    """Stop only when an entire official FILTERED family is unavailable."""
    counts = {family: sum(str(row.get("scientific_family")) == family and
                          bool(row.get("filtered_official_ranking_ready")) for row in preflight_rows)
              for family in families}
    missing = [family for family, count in counts.items() if count == 0]
    if missing:
        raise RuntimeError(f"No official FILTERED candidate remains for families: {', '.join(missing)}")
    return counts


def _read_manifest(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    raise ValueError("manifest must be JSON or CSV")


def audit_runtime_generator_assets(root: Path, entry: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the current host assets required immediately before benchmark execution.

    Checks only what carries a direct scientific meaning: each representation's positive pool exists
    with at least the target number of images, the scientific family is known, and the registered
    checkpoint and pools are present on disk. No historical audit record can block a generator whose
    local images are complete.
    """
    root = Path(root)
    target = int(protocol["synthetic_pool_target"])
    blockers = []
    representations = {}
    for representation in REPRESENTATIONS:
        relative = entry.get("samples", {}).get(f"{representation}_positive")
        paths = _top_level_images(root / relative) if relative else []
        representations[representation] = {"path": relative, "count": len(paths)}
        if len(paths) < target: blockers.append(f"insufficient_{representation}_positive_images:{len(paths)}<{target}")
    if entry.get("scientific_family") not in FAMILIES: blockers.append("invalid_scientific_family")
    role = entry.get("candidate_role", "primary_candidate")
    eligible = bool(entry.get("eligible_for_downstream_selection", False))
    descriptive = not blockers
    official = descriptive and eligible and role == "primary_candidate"
    expected = []
    for representation in REPRESENTATIONS:
        value = entry.get("samples", {}).get(f"{representation}_positive")
        if value: expected.append(root / str(value))
    if entry.get("checkpoint"): expected.append(root / str(entry["checkpoint"]))
    runtime_unavailable = any(not path.exists() for path in expected)
    runtime_mismatch = bool(blockers) and not runtime_unavailable
    runtime_verified = descriptive and not runtime_unavailable
    audit_mode = "runtime_assets_verified" if runtime_verified else (
        "runtime_assets_unavailable" if runtime_unavailable else "runtime_assets_mismatch")
    return {"generator_id": entry["id"], "scientific_family": entry.get("scientific_family"),
            "model_family": entry.get("model_family"), "model_variant": entry.get("model_variant"),
            "sampling_steps": entry.get("sampling_steps"), "candidate_role": role,
            "eligible_for_downstream_selection": eligible, "representations": representations,
            "blockers": sorted(set(blockers)), "eligible_for_descriptive_benchmark": descriptive,
            "eligible_for_official_family_ranking": official,
            "eligible_for_benchmark_execution": runtime_verified, "audit_mode": audit_mode,
            "parent_generator_id": entry.get("parent_generator_id"),
            "duplicate_model_group": list(entry.get("duplicate_model_group") or []),
            "runtime_assets_verified": runtime_verified, "runtime_assets_unavailable": runtime_unavailable,
            "runtime_assets_mismatch": runtime_mismatch}


def audit_candidate(root: Path, entry: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Backward-compatible name for the explicit runtime audit."""
    return audit_runtime_generator_assets(root, entry, protocol)


def discover_candidates(root: Path, protocol: Mapping[str, Any] | None = None,
                        registry: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    protocol, registry = protocol or load_protocol(root), registry or load_registry(root)
    rows = [audit_runtime_generator_assets(Path(root), entry, protocol) for entry in registry["generators"]
            if entry.get("benchmark", {}).get("enabled", False)]
    return detect_duplicate_generator_identities(rows)


def candidate_audit_document_rows(candidate_audits: Sequence[Mapping[str, Any]],
                                  audit_generated_at: str | None = None) -> list[dict[str, Any]]:
    """Serialize the metadata/runtime audit without running benchmark metrics."""
    generated_at = audit_generated_at or dt.datetime.now(dt.timezone.utc).isoformat()
    return [{
        "generator_id": row["generator_id"],
        "scientific_family": row["scientific_family"],
        "candidate_role": row["candidate_role"],
        "parent_generator_id": row.get("parent_generator_id"),
        "duplicate_model_group": row.get("duplicate_model_group", ""),
        "distinct_generator_for_ranking": row.get("distinct_generator_for_ranking", False),
        "audit_mode": row["audit_mode"],
        "audit_generated_at": generated_at,
        "project_root_independent_paths": True,
        "runtime_assets_verified": row["runtime_assets_verified"],
        "runtime_assets_unavailable": row["runtime_assets_unavailable"],
        "runtime_assets_mismatch": row["runtime_assets_mismatch"],
        "eligible_for_descriptive_benchmark": row["eligible_for_descriptive_benchmark"],
        "eligible_for_official_family_ranking": row["eligible_for_official_family_ranking"],
        "raw_count": row["representations"]["raw"]["count"],
        "filtered_count": row["representations"]["filtered"]["count"],
        "block_reasons": "; ".join(row["blockers"]),
    } for row in candidate_audits]


def write_embedding_cache(path: Path, features: np.ndarray, metadata: Mapping[str, Any]) -> tuple[Path, Path]:
    required = {"schema_version", "image_ids", "image_paths", "image_fingerprints", "extractor",
                "extractor_model_id", "extractor_weights_identifier", "extractor_identity", "extractor_identity_sha256", "preprocessing_signature",
                "feature_dimension", "code_version", "source_manifest_path", "source_manifest_sha256"}
    missing = required - set(metadata)
    if missing: raise ValueError(f"embedding cache metadata missing: {sorted(missing)}")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if len(features) != len(metadata["image_ids"]) or len(features) != len(metadata["image_paths"]):
        raise ValueError("embedding rows, image IDs and image paths must align")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(features))
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    atomic_json(metadata_path, dict(metadata))
    return path, metadata_path


def load_embedding_cache(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    return np.load(path, allow_pickle=False), json.loads(path.with_suffix(path.suffix + ".metadata.json").read_text())


def get_or_extract_embeddings(path: Path, image_paths: Sequence[str | Path], image_ids: Sequence[str], *,
                              extractor: str, preprocessing: str, code_version: str, source_manifest: str | None = None,
                              extractor_model_id: str | None = None,
                              extractor_weights_identifier: str | None = None,
                              extractor_identity: Mapping[str, Any] | None = None,
                              feature_dimension: int | None = None, metadata_csv: str | Path | None = None,
                              extract_fn: Callable[[Sequence[str | Path], str], np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    if len(image_paths) != len(image_ids):
        raise ValueError("image paths and IDs must align")
    resolved = [Path(value).resolve() for value in image_paths]
    common = Path(os.path.commonpath([str(value.parent) for value in resolved])) if resolved else Path(".")
    fingerprints = [{"image_id": str(image_id), "relative_path": os.path.relpath(value, common),
                     "file_size": value.stat().st_size, "sha256": file_sha256(value)}
                    for image_id, value in zip(image_ids, resolved)]
    # The per-image fingerprints above already invalidate the cache on any image change; an optional
    # Source metadata is optional and is used only to invalidate a stale embedding cache.
    manifest_path = Path(source_manifest) if source_manifest else None
    manifest_hash = None
    if manifest_path is not None:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"embedding source manifest is missing: {manifest_path}")
        manifest_hash = file_sha256(manifest_path)
    csv_path = Path(metadata_csv) if metadata_csv else None
    if csv_path is not None and not csv_path.is_file():
        raise FileNotFoundError(f"embedding metadata CSV is missing: {csv_path}")
    preprocessing_signature = hashlib.sha256(str(preprocessing).encode("utf-8")).hexdigest()
    declared_dimension = feature_dimension or {"inception_v3": 2048, "rad_dino": 768}.get(extractor)
    expected = {"schema_version": 2, "image_ids": list(image_ids),
                "image_paths": [str(value) for value in resolved], "image_fingerprints": fingerprints,
                "extractor": extractor, "extractor_model_id": extractor_model_id or extractor,
                "extractor_weights_identifier": extractor_weights_identifier or extractor,
                "extractor_identity": dict(extractor_identity or {}),
                "extractor_identity_sha256": _identity_digest(extractor_identity or {}),
                "preprocessing_signature": preprocessing_signature, "code_version": code_version,
                "source_manifest_path": str(manifest_path) if manifest_path else None,
                "source_manifest_sha256": manifest_hash,
                "metadata_csv_sha256": file_sha256(csv_path) if csv_path and csv_path.is_file() else None}
    if declared_dimension is not None:
        expected["feature_dimension"] = int(declared_dimension)
    if Path(path).is_file() and Path(path).with_suffix(Path(path).suffix + ".metadata.json").is_file():
        try:
            features, metadata = load_embedding_cache(path)
        except (OSError, ValueError, json.JSONDecodeError):
            features, metadata = None, {}
        valid = features is not None and all(metadata.get(key) == value for key, value in expected.items())
        valid = valid and np.asarray(features).ndim == 2 and len(features) == len(image_ids)
        valid = valid and np.isfinite(np.asarray(features)).all()
        valid = valid and int(metadata.get("feature_dimension", -1)) == int(features.shape[1])
        if declared_dimension is not None:
            valid = valid and int(metadata.get("feature_dimension", -1)) == int(declared_dimension)
        if valid:
            return features, {**metadata, "cache_event": "hit", "cache_invalidation_reason": ""}
    features = extract_fn(image_paths, extractor)
    array = np.asarray(features)
    if array.ndim != 2 or len(array) != len(image_ids):
        raise ValueError("extractor returned an invalid feature matrix")
    finite_rows = np.isfinite(array).all(axis=1)
    if not finite_rows.all():
        failures = [{"sample_id": str(image_ids[index]), "path": str(resolved[index]),
                     "extractor": extractor, "cause": "non_finite_feature_values"}
                    for index in np.flatnonzero(~finite_rows)]
        raise NonFiniteEmbeddingError(extractor, failures)
    if declared_dimension is not None and int(array.shape[1]) != int(declared_dimension):
        raise ValueError("extractor feature dimension differs from the declared dimension")
    metadata = {**expected, "feature_dimension": int(array.shape[1])}
    write_embedding_cache(path, array, metadata)
    reason = "cache_absent_or_metadata_mismatch"
    return array, {**metadata, "cache_event": "miss", "cache_invalidation_reason": reason}


def extract_features(paths: Sequence[str | Path], feature_space: str, *, device: str | None = None,
                     allow_model_download: bool = False,
                     local_model_path: str | Path | None = None) -> np.ndarray:
    """Extract frozen InceptionV3 or RAD-DINO embeddings; downloads are opt-in."""
    import torch
    from PIL import Image
    if feature_space == "inception_v3":
        from torchvision.models import Inception_V3_Weights, inception_v3
        weights = Inception_V3_Weights.DEFAULT
        checkpoint = Path(local_model_path) if local_model_path is not None else \
            Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
        if not allow_model_download and not checkpoint.is_file():
            raise FileNotFoundError("cached InceptionV3 weights required; downloads disabled")
        if checkpoint.is_file():
            model = inception_v3(weights=None, transform_input=False, init_weights=False)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        else:
            if not allow_model_download:
                raise FileNotFoundError("cached InceptionV3 weights required; downloads disabled")
            model = inception_v3(weights=weights, transform_input=False)
        if not allow_model_download and local_model_path is None:
            checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
            if not checkpoint.is_file():
                raise FileNotFoundError("cached InceptionV3 weights required; downloads disabled")
        model.fc = torch.nn.Identity()
        processor = lambda image: weights.transforms()(image.convert("RGB"))
    elif feature_space == "rad_dino":
        from transformers import AutoImageProcessor, AutoModel
        name = str(Path(local_model_path).resolve()) if local_model_path is not None else "microsoft/rad-dino"
        image_processor = AutoImageProcessor.from_pretrained(name, local_files_only=not allow_model_download)
        model = AutoModel.from_pretrained(name, local_files_only=not allow_model_download)
        processor = lambda image: image_processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]
    else: raise ValueError(feature_space)
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(target)
    output_rows = []
    with torch.inference_mode():
        for start in range(0, len(paths), 16):
            batch = torch.stack([processor(Image.open(path)) for path in paths[start:start + 16]]).to(target)
            output = model(batch)
            if feature_space == "rad_dino":
                output = output.pooler_output if getattr(output, "pooler_output", None) is not None else output.last_hidden_state[:, 0]
            output_rows.append(output.detach().cpu().float().numpy())
    return np.concatenate(output_rows, axis=0)


class FrozenLocalFeatureExtractor:
    """Load one frozen local encoder and reuse it for many extractions.

    Constructs the InceptionV3 (ImageNet-1K, 2048-d) or RAD-DINO (768-d) model a
    single time from a *local* path and keeps it resident, so repeated cache
    misses never reload the weights. Numerically mirrors :func:`extract_features`
    (same weights enum, preprocessing, batching) and never downloads anything.
    """

    def __init__(self, feature_space: str, local_model_path: str | Path, *,
                 device: str | None = None, batch_size: int = 16) -> None:
        import torch
        if feature_space not in FEATURE_SPACES:
            raise ValueError(feature_space)
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        self._torch = torch
        self.feature_space = feature_space
        self.local_model_path = Path(local_model_path)
        self.batch_size = int(batch_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if feature_space == "inception_v3":
            from torchvision.models import Inception_V3_Weights, inception_v3
            weights = Inception_V3_Weights.IMAGENET1K_V1
            if not self.local_model_path.is_file():
                raise FileNotFoundError(f"local InceptionV3 checkpoint required: {self.local_model_path}")
            model = inception_v3(weights=None, transform_input=False, init_weights=False)
            state = torch.load(self.local_model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.fc = torch.nn.Identity()
            transforms = weights.transforms()
            self._processor = lambda image: transforms(image.convert("RGB"))
            self.feature_dimension = 2048
        else:
            from transformers import AutoImageProcessor, AutoModel
            name = str(self.local_model_path.resolve())
            image_processor = AutoImageProcessor.from_pretrained(name, local_files_only=True)
            model = AutoModel.from_pretrained(name, local_files_only=True)
            self._processor = lambda image: image_processor(images=image.convert("RGB"),
                                                            return_tensors="pt")["pixel_values"][0]
            self.feature_dimension = 768
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.to(self.device)
        self._model = model

    def extract(self, paths: Sequence[str | Path]) -> np.ndarray:
        from PIL import Image
        torch = self._torch
        if self._model is None:
            raise RuntimeError("extractor has been closed")
        paths = list(paths)
        output_rows = []
        with torch.inference_mode():
            for start in range(0, len(paths), self.batch_size):
                chunk = paths[start:start + self.batch_size]
                batch = torch.stack([self._processor(Image.open(path)) for path in chunk]).to(self.device)
                output = self._model(batch)
                if self.feature_space == "rad_dino":
                    output = output.pooler_output if getattr(output, "pooler_output", None) is not None \
                        else output.last_hidden_state[:, 0]
                output_rows.append(output.detach().cpu().float().numpy())
        array = (np.concatenate(output_rows, axis=0) if output_rows
                 else np.empty((0, self.feature_dimension), dtype=np.float32)).astype(np.float32, copy=False)
        if array.ndim != 2 or int(array.shape[1]) != int(self.feature_dimension):
            raise ValueError("extractor feature dimension differs from the declared dimension")
        if not np.isfinite(array).all():
            raise ValueError(f"{self.feature_space} produced non-finite features")
        return array

    def close(self) -> None:
        """Release the model and free the CUDA cache; ``extract`` fails afterwards."""
        torch = self._torch
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "FrozenLocalFeatureExtractor":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def nearest_neighbours(query: np.ndarray, reference: np.ndarray, query_ids: Sequence[str],
                       reference_ids: Sequence[str], pool: str) -> list[dict[str, Any]]:
    if "test" in pool.lower(): raise PermissionError("test cannot be a nearest-neighbour pool")
    distances = np.sqrt(_pairwise_squared(query, reference)); nearest = distances.argmin(axis=1)
    return [{"synthetic_id": query_ids[index], "source_id": reference_ids[column], "pool": pool,
             "embedding_distance": float(distances[index, column])} for index, column in enumerate(nearest)]


def synthetic_nearest_neighbours(features: np.ndarray, image_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Find the nearest *other* synthetic feature; self-matches are excluded."""
    if len(features) < 2: raise ValueError("synthetic duplicate analysis requires at least two images")
    distances = np.sqrt(_pairwise_squared(features, features)); np.fill_diagonal(distances, np.inf)
    nearest = distances.argmin(axis=1)
    return [{"synthetic_id": image_ids[index], "source_id": image_ids[column], "pool": "synthetic_same_candidate",
             "embedding_distance": float(distances[index, column])} for index, column in enumerate(nearest)]


def image_similarity(left: Path, right: Path) -> dict[str, Any]:
    from PIL import Image
    with Image.open(left) as image: left_array = np.asarray(image.convert("L").resize((512, 512)), dtype=np.uint8)
    with Image.open(right) as image: right_array = np.asarray(image.convert("L").resize((512, 512)), dtype=np.uint8)
    return {"ssim": _ssim(left_array, right_array),
            "perceptual_hash_distance": hamming_distance(perceptual_hash(left), perceptual_hash(right)),
            "exact_hash_match": file_sha256(left) == file_sha256(right)}


def _ssim(left: np.ndarray, right: np.ndarray) -> float:
    """Gaussian-window grayscale SSIM without an optional scikit-image dependency."""
    from scipy.ndimage import gaussian_filter
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    mu_left, mu_right = gaussian_filter(left, 1.5), gaussian_filter(right, 1.5)
    var_left = np.maximum(0.0, gaussian_filter(left * left, 1.5) - mu_left * mu_left)
    var_right = np.maximum(0.0, gaussian_filter(right * right, 1.5) - mu_right * mu_right)
    covariance = gaussian_filter(left * right, 1.5) - mu_left * mu_right
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    numerator = (2 * mu_left * mu_right + c1) * (2 * covariance + c2)
    denominator = (mu_left * mu_left + mu_right * mu_right + c1) * (var_left + var_right + c2)
    return float(np.mean(numerator / np.maximum(denominator, np.finfo(float).eps)))


def deterministic_pair_indices(item_count: int, pair_count: int, seed: int) -> list[tuple[int, int]]:
    """Sample distinct unordered pairs deterministically without an all-pairs image comparison."""
    item_count, pair_count = int(item_count), int(pair_count)
    maximum = item_count * (item_count - 1) // 2
    if item_count < 2: raise ValueError("at least two images are required")
    if pair_count < 1: raise ValueError("pair_count must be positive")
    target = min(pair_count, maximum)
    rng, selected = random.Random(int(seed)), set()
    while len(selected) < target:
        left, right = sorted(rng.sample(range(item_count), 2))
        selected.add((left, right))
    return sorted(selected)


def multiscale_ssim(left: Path, right: Path, scales: Sequence[int] = (1, 2, 4)) -> float:
    """Grayscale MS-SSIM approximation using deterministic Gaussian-weighted SSIM scales."""
    from PIL import Image
    with Image.open(left) as image: left_image = image.convert("L").resize((512, 512))
    with Image.open(right) as image: right_image = image.convert("L").resize((512, 512))
    values = []
    for scale in scales:
        size = max(32, 512 // int(scale))
        a = np.asarray(left_image.resize((size, size)), dtype=np.uint8)
        b = np.asarray(right_image.resize((size, size)), dtype=np.uint8)
        values.append(_ssim(a, b))
    return float(np.prod(np.clip(values, 0.0, 1.0) ** (1.0 / len(values))))


def diversity_metrics(paths: Sequence[str | Path], features: np.ndarray, *, pair_count: int = 256,
                      seed: int = 20260714) -> dict[str, Any]:
    """Mandatory diversity metrics; LPIPS is deliberately not required or downloaded."""
    resolved = [Path(path) for path in paths]
    if len(resolved) != len(features): raise ValueError("image paths and embeddings must align")
    pairs = deterministic_pair_indices(len(resolved), pair_count, seed)
    similarities = [multiscale_ssim(resolved[left], resolved[right]) for left, right in pairs]
    nearest = synthetic_nearest_neighbours(np.asarray(features), [path.name for path in resolved])
    duplicates = duplicate_diagnostics(resolved)
    distances = [float(row["embedding_distance"]) for row in nearest]
    return {"pair_sampling_seed": int(seed), "requested_pairs": int(pair_count), "evaluated_pairs": len(pairs),
            "mean_ms_ssim": float(np.mean(similarities)), "ms_ssim_diversity": 1.0 - float(np.mean(similarities)),
            "synthetic_nearest_neighbour_distance_mean": float(np.mean(distances)),
            "synthetic_nearest_neighbour_distance_median": float(np.median(distances)), **duplicates,
            "lpips_status": "optional_not_evaluated"}


def build_train_memorization_rows(query: np.ndarray, reference: np.ndarray, query_ids: Sequence[str],
                                  reference_ids: Sequence[str], synthetic_paths: Mapping[str, Path],
                                  train_paths: Mapping[str, Path], flag_rule: Mapping[str, Any],
                                  train_labels: Mapping[str, Any] | None = None,
                                  train_sources: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Compare against the generator-specific declared training corpus, including negatives."""
    rows = nearest_neighbours(query, reference, query_ids, reference_ids, "declared_generator_training_corpus")
    enriched = enrich_similarity_rows(rows, synthetic_paths, train_paths, flag_rule, flag_name="memorization_flag")
    train_labels, train_sources = train_labels or {}, train_sources or {}
    exact_by_hash: dict[str, list[str]] = {}
    for train_id, path in train_paths.items():
        exact_by_hash.setdefault(file_sha256(Path(path)), []).append(str(train_id))
    output = []
    for row in enriched:
        exact_ids = sorted(exact_by_hash.get(file_sha256(Path(synthetic_paths[str(row["synthetic_id"])])), []))
        exact_id = exact_ids[0] if exact_ids else None
        output.append({"synthetic_id": row["synthetic_id"], "nearest_train_id": row["source_id"],
             "nearest_train_label": train_labels.get(str(row["source_id"])),
             "nearest_train_source": train_sources.get(str(row["source_id"])),
             "embedding_distance": row["embedding_distance"], "ssim": row["ssim"],
             "phash_distance": row["perceptual_hash_distance"], "exact_hash_match": bool(exact_ids),
             "exact_match_count": len(exact_ids), "exact_match_train_ids": exact_ids,
             "exact_match_train_id": exact_id, "exact_match_train_label": train_labels.get(str(exact_id)) if exact_id else None,
             "exact_match_train_source": train_sources.get(str(exact_id)) if exact_id else None,
             "memorization_flag": bool(row["memorization_flag"] or exact_ids)})
    return output


def build_validation_similarity_rows(query: np.ndarray, reference: np.ndarray, query_ids: Sequence[str],
                                     reference_ids: Sequence[str], synthetic_paths: Mapping[str, Path],
                                     validation_paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    rows = nearest_neighbours(query, reference, query_ids, reference_ids, "real_validation_positive")
    enriched = enrich_similarity_rows(rows, synthetic_paths, validation_paths,
                                      {"ssim_gte": 2.0, "perceptual_hash_distance_lte": -1},
                                      flag_name="_unused")
    return [{"synthetic_id": row["synthetic_id"], "nearest_validation_id": row["source_id"],
             "embedding_distance": row["embedding_distance"], "ssim": row["ssim"],
             "phash_distance": row["perceptual_hash_distance"]} for row in enriched]


def build_synthetic_duplication_rows(features: np.ndarray, image_ids: Sequence[str],
                                     synthetic_paths: Mapping[str, Path], flag_rule: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = synthetic_nearest_neighbours(features, image_ids)
    enriched = enrich_similarity_rows(rows, synthetic_paths, synthetic_paths, flag_rule, flag_name="duplicate_flag")
    return [{"synthetic_id": row["synthetic_id"], "nearest_synthetic_id": row["source_id"],
             "embedding_distance": row["embedding_distance"], "ssim": row["ssim"],
             "phash_distance": row["perceptual_hash_distance"], "duplicate_flag": row["duplicate_flag"]}
            for row in enriched]


def enrich_similarity_rows(rows: Sequence[Mapping[str, Any]], synthetic_paths: Mapping[str, Path],
                           reference_paths: Mapping[str, Path], flag_rule: Mapping[str, Any], *,
                           flag_name: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        similarity = image_similarity(synthetic_paths[str(row["synthetic_id"])], reference_paths[str(row["source_id"])])
        flagged = similarity["exact_hash_match"] or (similarity["ssim"] >= float(flag_rule["ssim_gte"])
                  and similarity["perceptual_hash_distance"] <= int(flag_rule["perceptual_hash_distance_lte"]))
        output.append({**row, **similarity, flag_name: bool(flagged)})
    return output


def deterministic_panel_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Choose closest, median and farthest examples with ID tie-breaking."""
    ordered = sorted(rows, key=lambda row: (float(row["embedding_distance"]), str(row["synthetic_id"])))
    if not ordered: return []
    return [ordered[index] for index in sorted(set((0, len(ordered) // 2, len(ordered) - 1)))]


def render_similarity_panel(rows: Sequence[Mapping[str, Any]], synthetic_paths: Mapping[str, Path],
                            reference_paths: Mapping[str, Path], output: Path, title: str) -> Path:
    import matplotlib.pyplot as plt
    from PIL import Image
    selected = deterministic_panel_rows(rows)
    figure, axes = plt.subplots(max(1, len(selected)), 2, figsize=(8, 4 * max(1, len(selected))), squeeze=False)
    for row_index, row in enumerate(selected):
        for column, (label, path) in enumerate((("synthetic", synthetic_paths[str(row["synthetic_id"])]),
                                                ("nearest reference", reference_paths[str(row["source_id"])]))):
            with Image.open(path) as image: axes[row_index, column].imshow(image.convert("L"), cmap="gray")
            axes[row_index, column].set_title(f"{label}\nd={float(row['embedding_distance']):.4f}")
            axes[row_index, column].axis("off")
    figure.suptitle(title); figure.tight_layout(); output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150); plt.close(figure); return output


def similarity_summaries(train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]],
                         synthetic_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep train memorization, validation similarity, and synthetic duplication independent."""
    def rate(rows: Sequence[Mapping[str, Any]], field: str) -> float:
        return sum(bool(row.get(field, False)) for row in rows) / len(rows) if rows else 0.0
    return {
        "train_memorization_rate": rate(train_rows, "memorization_flag"),
        "train_exact_duplicate_rate": rate(train_rows, "exact_hash_match"),
        "train_nearest_distance": [float(row["embedding_distance"]) for row in train_rows],
        "validation_similarity_rate": rate(validation_rows, "similarity_flag"),
        "validation_nearest_neighbour_distance": [float(row["embedding_distance"]) for row in validation_rows],
        "validation_structural_similarity": [float(row.get("ssim", math.nan)) for row in validation_rows],
        "synthetic_duplicate_rate": rate(synthetic_rows, "duplicate_flag"),
        "synthetic_nearest_distance": [float(row["embedding_distance"]) for row in synthetic_rows],
        "perceptual_hash_duplicate_rate": rate(synthetic_rows, "perceptual_hash_duplicate"),
    }


def eligibility_failures(row: Mapping[str, Any], gates: Mapping[str, Any]) -> list[str]:
    """Technical/scientific eligibility only: image count, exact-duplicate and memorization safety,
    corruption, metric completeness, test isolation and registry role. Perceptual-hash rate and
    RAD-DINO coverage remain descriptive ranking metrics rather than binary gates.
    """
    checks = [
        (int(row.get("valid_positive_images", 0)) >= int(gates["minimum_valid_positive_images"]), "valid_positive_images"),
        (_metric_value(row, "synthetic_exact_duplicate_rate") <= gates["maximum_exact_duplicate_rate"], "exact_duplicate_rate"),
        (_metric_value(row, "train_memorization_rate") <= gates["maximum_train_memorization_rate"], "train_memorization_rate"),
        (_metric_value(row, "corrupted_rate", "n_corrupt", default=0.0) <= gates["maximum_corrupted_file_rate"], "corrupt_files"),
        (_as_bool(row.get("metrics_complete")), "metrics_complete"),
        (not _as_bool(row.get("test_access")), "test_access"),
    ]
    if not _as_bool(row.get("eligible_for_selection", row.get("eligible_for_downstream_selection", False))):
        checks.append((False, "registry_role"))
    return [name for passed, name in checks if not passed]


def _metric_value(row: Mapping[str, Any], *names: str, default: float = math.inf) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try: return float(value)
            except (TypeError, ValueError): pass
    return float(default)


def rank_generator_family(rows: Sequence[Mapping[str, Any]], family: str,
                          gates: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Gate first, then apply the protocol-declared metric hierarchy without a weighted score."""
    candidates = []
    for source in rows:
        if str(source.get("family", source.get("scientific_family"))) != family: continue
        row = dict(source)
        failures = eligibility_failures(row, gates) if gates is not None else list(row.get("exclusion_reasons", []))
        registry_eligible = _as_bool(row.get("eligible_for_selection", row.get("eligible_for_downstream_selection", False)))
        if not registry_eligible and "registry_role" not in failures: failures.append("registry_role")
        row["exclusion_reasons"] = sorted(set(failures)); row["eligible"] = not row["exclusion_reasons"]
        candidates.append(row)
    required_metrics = {
        "raddino_kid": ("raddino_kid", "raddino_kid_mean"),
        "raddino_coverage": ("raddino_coverage", "coverage"),
        "raddino_precision": ("raddino_precision", "precision"),
        "raddino_fid": ("raddino_fid", "fid_descriptive"),
        "inception_kid": ("inception_kid", "inception_kid_mean"),
        "raddino_kid_std": ("raddino_kid_std", "kid_standard_deviation"),
    }
    eligible = [row for row in candidates if row["eligible"]]
    missing = [name for name, aliases in required_metrics.items()
               if any(all(row.get(alias) in (None, "") for alias in aliases) for row in eligible)]
    if missing:
        raise ValueError(f"ranking metrics must be available for every eligible candidate: {missing}")
    def key(row: Mapping[str, Any]):
        return (not row["eligible"],
                _metric_value(row, "raddino_kid", "raddino_kid_mean"),
                -_metric_value(row, "raddino_coverage", "coverage", default=-math.inf),
                -_metric_value(row, "raddino_precision", "precision", default=-math.inf),
                _metric_value(row, "raddino_fid", "fid_descriptive"),
                _metric_value(row, "inception_kid", "inception_kid_mean"),
                _metric_value(row, "raddino_kid_std", "kid_standard_deviation"),
                str(row.get("generator_id", "")))
    ordered = sorted(candidates, key=key)
    return [{**row, "family_rank": index + 1 if row["eligible"] else None}
            for index, row in enumerate(ordered)]


def practical_equivalence(paired: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the protocol-declared margin to paired stability differences."""
    low, high = float(paired["stability_interval_low"]), float(paired["stability_interval_high"])
    mean = float(paired["mean_paired_difference"])
    margin = float(protocol["selection"]["practical_equivalence_margin"])
    includes_zero = low <= 0.0 <= high
    return {"paired_stability_interval_includes_zero": includes_zero,
            "absolute_paired_mean_difference": abs(mean), "practical_equivalence_margin": margin,
            "practically_similar": includes_zero and abs(mean) <= margin}


INVALID_DURATION_STATUS = "unavailable_invalid_duration_semantics"


def efficiency_from_manifest(root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Import runtime efficiency without ever trusting an unverified duration.

    ``generation_seconds_per_image`` is accepted only when the manifest explicitly declares
    ``duration_semantics in {'wall_clock_full_generation', 'verified_seconds_per_image'}``,
    ``duration_unit == 'seconds'`` and ``measurement_complete == true``.  This holds even when the
    manifest carries a ``seconds_per_image`` field directly: an unverified ``seconds_per_image`` is
    *not* trusted, and an ``elapsed_seconds`` is divided by the image count only under
    ``wall_clock_full_generation``.  Otherwise the value is ``None`` and
    ``generation_efficiency_status`` is ``unavailable_invalid_duration_semantics``.  ``energy_kwh`` and
    ``peak_vram_mb`` are imported only when their own semantics are declared verified;
    ``checkpoint_size_bytes`` is kept whenever it is directly computable from the file.
    """
    base = {"generation_seconds_per_image": None, "peak_vram_mb": None, "energy_kwh": None,
            "checkpoint_size_bytes": None, "efficiency_source": None,
            "efficiency_status": "unavailable", "generation_efficiency_status": "unavailable"}
    value = entry.get("efficiency_manifest")
    path = Path(root) / str(value) if value else None
    checkpoint = Path(root) / str(entry.get("checkpoint", ""))
    checkpoint_size = checkpoint.stat().st_size if checkpoint.is_file() else None
    if not path or not path.is_file():
        return {**base, "checkpoint_size_bytes": checkpoint_size,
                "efficiency_source": str(value) if value else None,
                "efficiency_status": "unavailable" if checkpoint_size is None else "checkpoint_size_only",
                "generation_efficiency_status": "unavailable"}
    try:
        payload = _read_manifest(path)
    except Exception:
        return {**base, "checkpoint_size_bytes": checkpoint_size,
                "efficiency_status": "unavailable_unreadable_manifest",
                "generation_efficiency_status": "unavailable_unreadable_manifest"}
    if not isinstance(payload, Mapping):
        return {**base, "checkpoint_size_bytes": checkpoint_size,
                "efficiency_status": "unavailable_unreadable_manifest",
                "generation_efficiency_status": "unavailable_unreadable_manifest"}

    count = payload.get("images_generated") or payload.get("n_generated")
    if not count and payload.get("n_per_class") and payload.get("generated_classes"):
        count = int(payload["n_per_class"]) * len(payload["generated_classes"])

    unit_ok = str(payload.get("duration_unit")) == "seconds" and payload.get("measurement_complete") is True
    semantics = str(payload.get("duration_semantics"))
    seconds_per_image = None
    generation_status = "unavailable"
    direct = payload.get("seconds_per_image")
    elapsed = payload.get("elapsed_seconds")
    if direct is not None:
        # A directly recorded seconds_per_image requires verified_seconds_per_image semantics.
        if unit_ok and semantics == "verified_seconds_per_image":
            seconds_per_image, generation_status = float(direct), "available"
        else:
            generation_status = INVALID_DURATION_STATUS
    elif elapsed is not None and count:
        # elapsed_seconds may be divided by the image count only for a full wall-clock measurement.
        if unit_ok and semantics == "wall_clock_full_generation":
            seconds_per_image, generation_status = float(elapsed) / int(count), "available"
        else:
            generation_status = INVALID_DURATION_STATUS

    energy_verified = bool(payload.get("energy_semantics_verified"))
    vram_verified = payload.get("peak_vram_mb") is not None and bool(payload.get("vram_semantics_verified"))
    values = {"generation_seconds_per_image": seconds_per_image,
              "peak_vram_mb": payload.get("peak_vram_mb") if vram_verified else None,
              "energy_kwh": payload.get("energy_kwh") if energy_verified else None,
              "checkpoint_size_bytes": checkpoint_size, "efficiency_source": str(value)}
    available = any(values[name] is not None for name in
                    ("generation_seconds_per_image", "peak_vram_mb", "energy_kwh", "checkpoint_size_bytes"))
    efficiency_status = generation_status if generation_status == INVALID_DURATION_STATUS else (
        "available" if available else "unavailable")
    return {**values, "efficiency_status": efficiency_status, "generation_efficiency_status": generation_status}


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> Path:
    if not rows: raise ValueError(f"refusing to create an empty CSV: {path}")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def plot_generator_summary(rows: Sequence[Mapping[str, Any]]):
    import matplotlib.pyplot as plt
    if not rows: raise ValueError("generator summary is empty")
    labels = [f"{row['generator_id']}\n{row.get('condition', '')}" for row in rows]
    kid_values = [_metric_value(row, "raddino_kid", "raddino_kid_mean") for row in rows]
    coverage = [_metric_value(row, "raddino_coverage", "coverage", default=math.nan) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(max(10, len(rows) * 1.2), 4))
    axes[0].bar(labels, kid_values); axes[0].set_title("RAD-DINO KID (lower is better)")
    axes[1].bar(labels, coverage); axes[1].set_title("RAD-DINO coverage (higher is better)")
    for axis in axes: axis.tick_params(axis="x", rotation=90)
    figure.tight_layout(); return figure


def validate_selected_generators(finetuned: str, from_scratch: str, registry: Mapping[str, Any],
                                 benchmark_rows: Sequence[Mapping[str, Any]], synthetic_pool_target: int = 1361,
                                 gates: Mapping[str, Any] | None = None) -> dict[str, str]:
    by_id = {entry["id"]: entry for entry in registry["generators"]}
    # FILTERED is the official ranking representation; when the summary lists both RAW and FILTERED
    # rows per generator, validate the FILTERED row rather than whichever row happens to come last.
    filtered_rows = [row for row in benchmark_rows if row.get("condition") == "FILTERED"]
    results = {str(row["generator_id"]): row for row in (filtered_rows or benchmark_rows)}
    selected = {"finetuned": finetuned, "from_scratch": from_scratch}
    for family, generator_id in selected.items():
        entry = by_id.get(generator_id)
        if not entry: raise ValueError(f"unknown generator ID: {generator_id}")
        if entry.get("scientific_family") != family: raise ValueError(f"{generator_id} has wrong family")
        if not entry.get("eligible_for_downstream_selection", False): raise ValueError(f"{generator_id} is not selection-eligible")
        result = results.get(generator_id)
        if not result or not _as_bool(result.get("metrics_complete")): raise ValueError(f"benchmark incomplete for {generator_id}")
        if int(result.get("valid_positive_images", 0)) < int(synthetic_pool_target): raise ValueError(f"insufficient images for {generator_id}")
        if _as_bool(result.get("test_access")): raise ValueError("generator selection must not use test data")
        # Eligibility uses the live technical/scientific safety gates (the single source of truth),
        # not a precomputed summary column that may reflect a superseded gate policy.
        if gates is not None:
            failures = [name for name in eligibility_failures({**result, "eligible_for_downstream_selection": True}, gates)
                        if name != "registry_role"]
            if failures: raise ValueError(f"eligibility gates failed for {generator_id}: {failures}")
        elif not _as_bool(result.get("technical_gates_passed")):
            raise ValueError(f"technical gates failed for {generator_id}")
    return selected


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool): return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def save_selected_generators(root: Path, finetuned: str, from_scratch: str, benchmark_rows: Sequence[Mapping[str, Any]],
                             *, notes: str = "", manual_override: bool = False) -> Path:
    """Write the simple authoritative G02/G07 selection record."""
    protocol, registry = load_protocol(root), load_registry(root)
    selected = validate_selected_generators(finetuned, from_scratch, registry, benchmark_rows,
                                            protocol["synthetic_pool_target"], protocol["eligibility_gates"])
    target = int(protocol["synthetic_pool_target"])
    by_id = {entry["id"]: entry for entry in registry["generators"]}
    records = []
    for family, generator_id in selected.items():
        entry = by_id[generator_id]
        records.append({
            "generator_id": generator_id,
            "role": family,
            "model_family": entry["model_family"],
            "canonical_filtered_pool": entry["samples"]["filtered_positive"],
            "expected_count": target,
            "scientific_reason": notes or "Top eligible candidate under the declared RAD-DINO KID-first hierarchy.",
        })
    payload = {
        "finetuned": selected["finetuned"], "from_scratch": selected["from_scratch"],
        "schema_version": 3, "protocol_version": protocol["protocol_version"],
        "selected_at": "2026-07-21", "test_access": False, "generators": records,
    }
    path = atomic_json(Path(root) / "configs/selected_generators.json", payload)
    atomic_json(Path(root) / BENCHMARK_ROOT / "selection_summary.json", payload)
    return path


__all__ = ["BENCHMARK_ROOT", "TRAIN_METADATA", "VALIDATION_METADATA", "AUGMENTATION_METADATA",
           "CANONICAL_OUTPUTS",
           "FEATURE_SPACES", "FAMILIES", "REPRESENTATIONS", "atomic_json",
           "NonFiniteEmbeddingError",
           "audit_candidate", "audit_runtime_generator_assets",
           "balanced_subsample_indices", "deterministic_sample", "discover_candidates", "duplicate_diagnostics",
           "build_synthetic_duplication_rows", "build_train_memorization_rows", "build_validation_similarity_rows",
           "deterministic_pair_indices", "diversity_metrics", "efficiency_from_manifest", "eligibility_failures", "evaluation_subset_size", "extract_features", "FrozenLocalFeatureExtractor", "fid", "file_sha256", "get_or_extract_embeddings",
           "image_similarity", "inception_v3_identity", "kid", "list_image_paths", "load_embedding_cache", "load_protocol", "load_registry", "metadata_positive_paths",
           "multiscale_ssim", "nearest_neighbours", "paired_kid_differences", "plot_generator_summary", "practical_equivalence", "prdc", "rank_generator_family", "render_similarity_panel", "repeated_distribution_metrics", "save_resampling_plan", "save_selected_generators",
           "rad_dino_identity", "representation_preflight_rows", "require_official_family_coverage",
           "similarity_summaries", "summarize", "technical_audit", "validate_protocol",
           "synthetic_nearest_neighbours", "technical_validity_row", "training_corpus_from_metadata",
           "validate_selected_generators", "write_csv_rows", "write_embedding_cache"]
