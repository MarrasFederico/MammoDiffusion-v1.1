"""Notebook-first utilities for the validation-only generator benchmark.

Importing this module never loads a model, opens the test split, or starts computation.
The notebook calls each stage explicitly and may reuse cached embeddings.
"""
from __future__ import annotations

import csv
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
BENCHMARK_ROOT = Path("results/publication_v2/generator_benchmark")
CANONICAL_OUTPUTS = (
    "candidate_audit.csv", "technical_validity.csv", "embedding_cache/",
    "distribution_metrics_repetitions.csv", "distribution_metrics_summary.csv",
    "diversity_metrics.csv", "train_memorization.csv", "validation_similarity.csv",
    "synthetic_duplication.csv", "generator_summary.csv", "generator_ranking.csv",
    "resampling_plan.json", "paired_generator_differences.csv",
    "figures/", "diagnostic_panels/",
)


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
    if protocol.get("pipeline_namespace") != "mammodiffusion.generator_benchmark.v2":
        raise ValueError("unsupported generator benchmark protocol")
    if int(protocol.get("synthetic_pool_target", 0)) <= 1:
        raise ValueError("synthetic_pool_target must exceed one")
    if tuple(protocol.get("representations", [])) != REPRESENTATIONS:
        raise ValueError("raw and filtered representations must be registered separately")
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
        raise ValueError("practical_equivalence_margin must be preregistered")
    if int(resampling.get("fid_repetitions", 999)) > 10:
        raise ValueError("FID is secondary and must use a small independent repetition count")
    ranking_metrics = [item["metric"] for item in protocol["selection"]["ranking"]]
    if ranking_metrics[0] != "rad_dino.filtered.kid.full_reference":
        raise ValueError("KID must be the primary ranking metric")
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


def training_corpus_from_manifest(root: Path, relative_csv: str) -> tuple[list[str], list[str], dict[str, int], dict[str, str]]:
    """Load the complete generator-declared training corpus, preserving labels and sources."""
    _reject_test_path(relative_csv)
    manifest = Path(root) / relative_csv
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    paths, ids, labels, sources = [], [], {}, {}
    for index, row in enumerate(rows):
        image_id = str(row.get("image_id") or row.get("sample_id") or index)
        sample_id = f"{row.get('patient_id', '')}::{image_id}"
        value = row.get("processed_path") or row.get("path")
        if not value:
            label = int(row["label"])
            value = f"data/processed/train/{label}/{image_id}.png"
        _reject_test_path(value)
        paths.append(str((Path(root) / value).resolve())); ids.append(sample_id)
        labels[sample_id] = int(row["label"]); sources[sample_id] = str(row.get("source", "real_train"))
    if len(ids) != len(set(ids)):
        raise ValueError("training corpus manifest contains duplicate sample IDs")
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
    # Point estimates use every real reference. Synthetic balancing is deterministic and explicit.
    point_synthetic_count = min(len(synthetic), int(protocol["synthetic_pool_target"]))
    synthetic_point = np.asarray(synthetic)[np.random.default_rng(base_seed).choice(
        len(synthetic), point_synthetic_count, replace=False)]
    real_point = np.asarray(real)
    point = {"kid": kid(real_point, synthetic_point), "fid": fid(real_point, synthetic_point),
             **prdc(real_point, synthetic_point, nearest_k=nearest_k)}
    summaries = {metric: summarize(metric_values) for metric, metric_values in values.items()}
    return rows, {"full_reference_policy": "all real references; deterministic balanced synthetic subset",
                  "full_reference_real_count": len(real_point),
                  "full_reference_synthetic_count": len(synthetic_point), "full_reference_estimates": point,
                  "stability_subset_size": size, "stability_interval_type": "repeated-subsampling stability interval",
                  **summaries}


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
    content_hashes = []
    for value in paths:
        try:
            with Image.open(value) as image:
                gray = np.asarray(image.convert("L"), dtype=np.uint8)
                counters["readable"] += 1
                wrong_shape = image.size != expected_size
                invalid_range = not gray.size or int(gray.max()) <= int(gray.min())
                near_black = bool(gray.size and np.count_nonzero(gray > 5) / gray.size < 0.01)
                if wrong_shape: counters["wrong_shape"] += 1
                if invalid_range: counters["invalid_range"] += 1
                if near_black: counters["near_black"] += 1
                if not (wrong_shape or invalid_range or near_black):
                    counters["technically_valid"] += 1
                    content_hashes.append(hashlib.sha256(gray.tobytes()).hexdigest())
        except Exception:
            counters["corrupt"] += 1
    total = len(paths)
    unique = len(set(content_hashes))
    valid = counters["technically_valid"]
    return {
        "n_discovered": total, "n_readable": counters["readable"], "n_corrupt": counters["corrupt"],
        "n_wrong_shape": counters["wrong_shape"], "n_near_black": counters["near_black"],
        "n_invalid_range": counters["invalid_range"], "n_technically_valid": valid,
        "n_technically_invalid": counters["readable"] - valid,
        "n_unique_valid_content": unique,
        "n_exact_duplicates_among_valid": max(0, valid - unique),
        "technical_validity_rate": valid / total if total else 0.0,
        # Backward-compatible rates used by the protocol gates.
        "n_unique_content": unique, "n_exact_duplicates": max(0, valid - unique),
        "n_images": total, "corrupted_rate": counters["corrupt"] / total if total else 0.0,
        "unexpected_dimensions_rate": counters["wrong_shape"] / total if total else 0.0,
        "invalid_dynamic_range_rate": counters["invalid_range"] / total if total else 0.0,
        "near_black_rate": counters["near_black"] / total if total else 0.0,
    }


def technical_validity_row(generator_id: str, condition: str, paths: Sequence[str | Path], *,
                           minimum_unique: int = 1361, expected_size: tuple[int, int] = (512, 512)) -> dict[str, Any]:
    """Return the complete, display-ready technical-validity schema for one candidate condition."""
    audit = technical_audit(paths, expected_size)
    failures = []
    if audit["n_corrupt"]: failures.append("corrupt_images")
    if audit["n_wrong_shape"]: failures.append("wrong_shape")
    if audit["n_near_black"]: failures.append("near_black")
    if audit["n_invalid_range"]: failures.append("invalid_range")
    if audit["n_unique_valid_content"] < int(minimum_unique): failures.append("insufficient_unique_content")
    return {"generator_id": generator_id, "condition": condition.upper(),
            **{key: audit[key] for key in ("n_discovered", "n_readable", "n_corrupt", "n_wrong_shape",
                                           "n_near_black", "n_invalid_range", "n_technically_valid",
                                           "n_technically_invalid", "n_unique_valid_content",
                                           "n_exact_duplicates_among_valid", "technical_validity_rate")},
            "eligible_for_distribution_metrics": not failures,
            "failure_reason": "; ".join(failures)}


def _read_manifest(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    raise ValueError("manifest must be JSON or CSV")


def _manifest_sample_paths(payload: Any, representation: str) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    direct = payload.get("sample_paths") or payload.get("image_paths")
    if isinstance(direct, list):
        return [str(value) for value in direct]
    samples = payload.get("samples", {})
    if isinstance(samples, Mapping) and isinstance(samples.get(representation), list):
        return [str(value.get("path")) if isinstance(value, Mapping) else str(value)
                for value in samples[representation] if not isinstance(value, Mapping) or value.get("path")]
    image_sets = payload.get("image_sets", [])
    if isinstance(image_sets, list):
        return [str(value.get("path")) for value in image_sets if isinstance(value, Mapping) and value.get("path")]
    return []


def _manifest_sample_records(payload: Any, representation: str) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping): return []
    samples = payload.get("samples", {})
    values = samples.get(representation, []) if isinstance(samples, Mapping) else []
    return [value for value in values if isinstance(value, Mapping) and value.get("path") and value.get("sha256")]


def _path_names(values: Sequence[str | Path]) -> set[str]:
    return {Path(str(value)).name for value in values if str(value)}


def filter_acceptance_from_manifest(path: Path, filtered_paths: Sequence[str | Path],
                                    raw_paths: Sequence[str | Path] | None = None) -> dict[str, Any]:
    """Read acceptance only from a filter manifest, never from directory validity."""
    result = {"filter_acceptance_rate": None, "filter_provenance_complete": False,
              "n_raw_submitted_to_filter": None, "n_filter_accepted": None,
              "filter_manifest_valid": False, "filter_failure_reason": "missing_filter_manifest"}
    if not path.is_file():
        return result
    try:
        payload = _read_manifest(path)
        if isinstance(payload, list):
            raw_count = len(payload)
            accepted_rows = [row for row in payload if str(row.get("selected", row.get("accepted", ""))).lower() in {"1", "true", "yes"}]
            accepted_count = len(accepted_rows)
            declared = [row.get("filtered_path") or row.get("output_path") for row in accepted_rows]
            declared_raw = [row.get("raw_path") or row.get("source_path") or row.get("input_path") for row in payload]
        else:
            signature = payload.get("input_signature", payload)
            raw_count = int(signature.get("n_raw", payload.get("n_raw_submitted_to_filter", 0)))
            accepted_count = int(signature.get("n_selected", payload.get("n_filter_accepted", 0)))
            declared = _manifest_sample_paths(payload, "filtered")
            declared_raw = _manifest_sample_paths(payload, "raw")
        if raw_count <= 0 or accepted_count < 0 or accepted_count > raw_count:
            raise ValueError("invalid raw/accepted counts")
        if not declared or _path_names(declared) != _path_names(filtered_paths):
            raise ValueError("filtered sample set differs from filter manifest")
        if raw_paths is not None and (not declared_raw or _path_names(declared_raw) != _path_names(raw_paths)):
            raise ValueError("RAW sample set differs from filter manifest")
        return {"filter_acceptance_rate": accepted_count / raw_count,
                "filter_provenance_complete": True, "n_raw_submitted_to_filter": raw_count,
                "n_filter_accepted": accepted_count, "filter_manifest_valid": True,
                "filter_failure_reason": ""}
    except Exception as exc:
        return {**result, "filter_failure_reason": f"{type(exc).__name__}: {exc}"}


def _provenance_audit(root: Path, entry: Mapping[str, Any], representations: Mapping[str, Any]) -> dict[str, Any]:
    manifest_value = entry.get("provenance_manifest")
    path = Path(root) / str(manifest_value) if manifest_value else Path("")
    base = {"provenance_manifest_exists": bool(manifest_value and path.is_file()),
            "provenance_manifest_valid": False, "lineage_complete": False, "raw_manifest_valid": False,
            "filter_manifest_valid": False, "sample_set_matches_manifest": False,
            "training_corpus_manifest": None, "training_corpus_manifest_valid": False,
            "provenance_failure_reason": ""}
    reasons = []
    if not base["provenance_manifest_exists"]:
        reasons.append("provenance_manifest_missing")
        return {**base, "provenance_failure_reason": "; ".join(reasons)}
    try:
        payload = _read_manifest(path)
    except Exception as exc:
        reasons.append(f"provenance_manifest_unreadable:{type(exc).__name__}")
        return {**base, "provenance_failure_reason": "; ".join(reasons)}
    if not isinstance(payload, Mapping):
        reasons.append("provenance_manifest_not_object")
        return {**base, "provenance_failure_reason": "; ".join(reasons)}
    recorded_id = payload.get("generator_id") or payload.get("experiment_id") or payload.get("id")
    if recorded_id != entry.get("id"): reasons.append("wrong_generator_id")
    checkpoint = payload.get("checkpoint") or payload.get("checkpoint_path") or payload.get("selected_checkpoint") or payload.get("best_checkpoint")
    if not checkpoint: reasons.append("missing_checkpoint_identifier")
    elif entry.get("checkpoint") and Path(str(checkpoint)).name not in str(entry["checkpoint"]): reasons.append("wrong_checkpoint")
    training_manifest = payload.get("training_corpus_manifest") or entry.get("training_corpus_manifest")
    training_ids, training_labels = payload.get("training_image_ids"), payload.get("training_labels")
    training_path = Path(root) / str(training_manifest) if training_manifest else None
    training_valid = bool(training_ids and training_labels and len(training_ids) == len(training_labels))
    if training_path and training_path.is_file():
        try:
            training_payload = _read_manifest(training_path)
            if isinstance(training_payload, list):
                training_valid = bool(training_payload) and all(row.get("label") not in (None, "") and
                    (row.get("image_id") or row.get("sample_id") or row.get("processed_path") or row.get("path"))
                    for row in training_payload)
            elif isinstance(training_payload, Mapping):
                ids = training_payload.get("training_image_ids")
                labels = training_payload.get("training_labels")
                training_valid = bool(ids and labels and len(ids) == len(labels))
        except Exception:
            training_valid = False
    if not training_valid: reasons.append("missing_training_corpus")
    training_dataset = payload.get("training_dataset") or payload.get("training_corpus") or training_manifest
    if not training_dataset: reasons.append("missing_training_dataset_identifier")
    actual_paths = {name: list_image_paths(Path(root) / value["path"]) if value.get("path") else []
                    for name, value in representations.items()}
    actual_by_representation = {name: _path_names(values) for name, values in actual_paths.items()}
    records_by_representation = {name: _manifest_sample_records(payload, name) for name in REPRESENTATIONS}
    declared_by_representation = {name: _path_names([record["path"] for record in records])
                                  for name, records in records_by_representation.items()}
    sets_match = all(declared_by_representation[name] and
        declared_by_representation[name] == actual_by_representation[name] for name in REPRESENTATIONS)
    fingerprints_match = False
    if sets_match:
        actual_hashes = {name: {path.name: file_sha256(path) for path in values}
                         for name, values in actual_paths.items()}
        fingerprints_match = all(all(actual_hashes[name].get(Path(str(record["path"])).name) == str(record["sha256"])
            for record in records_by_representation[name]) for name in REPRESENTATIONS)
    samples_match = sets_match and fingerprints_match
    if not samples_match: reasons.append("sample_set_mismatch")
    raw_manifest = entry.get("raw_generation_manifest") or payload.get("raw_generation_manifest")
    raw_path = Path(root) / str(raw_manifest) if raw_manifest else path
    raw_valid = raw_path.is_file() and bool(declared_by_representation["raw"])
    if not raw_valid: reasons.append("raw_manifest_invalid")
    filtered_paths = list_image_paths(Path(root) / representations["filtered"]["path"]) if representations["filtered"].get("path") else []
    raw_paths = list_image_paths(Path(root) / representations["raw"]["path"]) if representations["raw"].get("path") else []
    filter_value = entry.get("filter_manifest") or payload.get("filter_manifest")
    filter_result = filter_acceptance_from_manifest(Path(root) / str(filter_value), filtered_paths, raw_paths) if filter_value else {
        "filter_manifest_valid": not entry.get("filtering_applied", False), "filter_provenance_complete": not entry.get("filtering_applied", False),
        "filter_acceptance_rate": 1.0 if not entry.get("filtering_applied", False) else None,
        "n_raw_submitted_to_filter": len(filtered_paths) if not entry.get("filtering_applied", False) else None,
        "n_filter_accepted": len(filtered_paths) if not entry.get("filtering_applied", False) else None,
        "filter_failure_reason": "" if not entry.get("filtering_applied", False) else "missing_filter_manifest"}
    if not filter_result["filter_manifest_valid"]: reasons.append(filter_result["filter_failure_reason"])
    lineage = raw_valid and filter_result["filter_manifest_valid"] and samples_match
    return {**base, **filter_result, "provenance_manifest_valid": not any(reason.startswith(("wrong_generator", "wrong_checkpoint", "provenance_", "missing_checkpoint", "missing_training_dataset")) for reason in reasons),
            "lineage_complete": lineage, "raw_manifest_valid": raw_valid,
            "sample_set_matches_manifest": samples_match, "training_corpus_manifest": training_manifest,
            "training_corpus_manifest_valid": training_valid,
            "provenance_failure_reason": "; ".join(dict.fromkeys(reasons))}


def audit_candidate(root: Path, entry: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    target = int(protocol["synthetic_pool_target"])
    blockers = []
    representations = {}
    for representation in REPRESENTATIONS:
        relative = entry.get("samples", {}).get(f"{representation}_positive")
        paths = list_image_paths(Path(root) / relative) if relative else []
        representations[representation] = {"path": relative, "count": len(paths)}
        if len(paths) < target: blockers.append(f"insufficient_{representation}_positive_images:{len(paths)}<{target}")
    if entry.get("scientific_family") not in FAMILIES: blockers.append("invalid_scientific_family")
    provenance = _provenance_audit(Path(root), entry, representations)
    gates = protocol["eligibility_gates"]
    if gates.get("require_provenance") and not provenance["provenance_manifest_valid"]: blockers.append("provenance_invalid")
    if gates.get("require_lineage") and not provenance["lineage_complete"]: blockers.append("lineage_incomplete")
    if entry.get("candidate_role", "primary_candidate") == "primary_candidate":
        if entry.get("filtering_applied") and not provenance.get("filter_provenance_complete"): blockers.append("filter_provenance_incomplete")
        if not provenance.get("training_corpus_manifest_valid"): blockers.append("training_corpus_missing")
    role = entry.get("candidate_role", "primary_candidate")
    eligible = bool(entry.get("eligible_for_downstream_selection", False))
    return {"generator_id": entry["id"], "scientific_family": entry.get("scientific_family"),
            "model_family": entry.get("model_family"), "model_variant": entry.get("model_variant"),
            "sampling_steps": entry.get("sampling_steps"), "candidate_role": role,
            "eligible_for_downstream_selection": eligible, "representations": representations, **provenance,
            "blockers": sorted(set(blockers)), "eligible_for_benchmark_execution": not blockers}


def discover_candidates(root: Path, protocol: Mapping[str, Any] | None = None,
                        registry: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    protocol, registry = protocol or load_protocol(root), registry or load_registry(root)
    return [audit_candidate(Path(root), entry, protocol) for entry in registry["generators"]
            if entry.get("benchmark", {}).get("enabled", False)]


def write_embedding_cache(path: Path, features: np.ndarray, metadata: Mapping[str, Any]) -> tuple[Path, Path]:
    required = {"schema_version", "image_ids", "image_paths", "image_fingerprints", "extractor",
                "extractor_model_id", "extractor_weights_identifier", "preprocessing_signature",
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
                              extractor: str, preprocessing: str, code_version: str, source_manifest: str,
                              extractor_model_id: str | None = None,
                              extractor_weights_identifier: str | None = None,
                              feature_dimension: int | None = None, metadata_csv: str | Path | None = None,
                              extract_fn: Callable[[Sequence[str | Path], str], np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    if len(image_paths) != len(image_ids):
        raise ValueError("image paths and IDs must align")
    resolved = [Path(value).resolve() for value in image_paths]
    common = Path(os.path.commonpath([str(value.parent) for value in resolved])) if resolved else Path(".")
    fingerprints = [{"image_id": str(image_id), "relative_path": os.path.relpath(value, common),
                     "file_size": value.stat().st_size, "sha256": file_sha256(value)}
                    for image_id, value in zip(image_ids, resolved)]
    manifest_path = Path(source_manifest)
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
                "preprocessing_signature": preprocessing_signature, "code_version": code_version,
                "source_manifest_path": str(manifest_path), "source_manifest_sha256": manifest_hash,
                "metadata_csv_sha256": file_sha256(csv_path) if csv_path and csv_path.is_file() else None}
    if declared_dimension is not None:
        expected["feature_dimension"] = int(declared_dimension)
    if Path(path).is_file() and Path(path).with_suffix(Path(path).suffix + ".metadata.json").is_file():
        try:
            features, metadata = load_embedding_cache(path)
        except (OSError, ValueError, json.JSONDecodeError):
            features, metadata = None, {}
        valid = features is not None and all(metadata.get(key) == value for key, value in expected.items())
        valid = valid and int(metadata.get("feature_dimension", -1)) == int(features.shape[1])
        if declared_dimension is not None:
            valid = valid and int(metadata.get("feature_dimension", -1)) == int(declared_dimension)
        if valid:
            return features, metadata
    features = extract_fn(image_paths, extractor)
    if np.asarray(features).ndim != 2 or len(features) != len(image_ids):
        raise ValueError("extractor returned an invalid feature matrix")
    if declared_dimension is not None and int(features.shape[1]) != int(declared_dimension):
        raise ValueError("extractor feature dimension differs from the declared dimension")
    metadata = {**expected, "feature_dimension": int(features.shape[1])}
    write_embedding_cache(path, features, metadata)
    return features, metadata


def extract_features(paths: Sequence[str | Path], feature_space: str, *, device: str | None = None,
                     allow_model_download: bool = False) -> np.ndarray:
    """Extract frozen InceptionV3 or RAD-DINO embeddings; downloads are opt-in."""
    import torch
    from PIL import Image
    if feature_space == "inception_v3":
        from torchvision.models import Inception_V3_Weights, inception_v3
        weights = Inception_V3_Weights.DEFAULT
        if not allow_model_download:
            checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
            if not checkpoint.is_file(): raise FileNotFoundError("cached InceptionV3 weights required; downloads disabled")
        model = inception_v3(weights=weights, transform_input=False); model.fc = torch.nn.Identity()
        processor = lambda image: weights.transforms()(image.convert("RGB"))
    elif feature_space == "rad_dino":
        from transformers import AutoImageProcessor, AutoModel
        name = "microsoft/rad-dino"
        image_processor = AutoImageProcessor.from_pretrained(name, local_files_only=not allow_model_download)
        model = AutoModel.from_pretrained(name, local_files_only=not allow_model_download)
        processor = lambda image: image_processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]
    else: raise ValueError(feature_space)
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); model.eval().to(target)
    output_rows = []
    with torch.inference_mode():
        for start in range(0, len(paths), 16):
            batch = torch.stack([processor(Image.open(path)) for path in paths[start:start + 16]]).to(target)
            output = model(batch)
            if feature_space == "rad_dino":
                output = output.pooler_output if getattr(output, "pooler_output", None) is not None else output.last_hidden_state[:, 0]
            output_rows.append(output.detach().cpu().float().numpy())
    return np.concatenate(output_rows, axis=0)


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
    exact_by_hash = {file_sha256(Path(path)): str(train_id) for train_id, path in train_paths.items()}
    output = []
    for row in enriched:
        exact_id = exact_by_hash.get(file_sha256(Path(synthetic_paths[str(row["synthetic_id"])])))
        output.append({"synthetic_id": row["synthetic_id"], "nearest_train_id": row["source_id"],
             "nearest_train_label": train_labels.get(str(row["source_id"])),
             "nearest_train_source": train_sources.get(str(row["source_id"])),
             "embedding_distance": row["embedding_distance"], "ssim": row["ssim"],
             "phash_distance": row["perceptual_hash_distance"], "exact_hash_match": exact_id is not None,
             "exact_match_train_id": exact_id,
             "memorization_flag": bool(row["memorization_flag"] or exact_id is not None)})
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
    checks = [
        (int(row.get("valid_positive_images", 0)) >= int(gates["minimum_valid_positive_images"]), "valid_positive_images"),
        (_metric_value(row, "synthetic_exact_duplicate_rate") <= gates["maximum_exact_duplicate_rate"], "exact_duplicate_rate"),
        (_metric_value(row, "perceptual_hash_duplicate_rate") <= gates["maximum_perceptual_duplicate_rate"], "perceptual_duplicate_rate"),
        (_metric_value(row, "train_memorization_rate") <= gates["maximum_train_memorization_rate"], "train_memorization_rate"),
        (_metric_value(row, "raddino_coverage", "coverage", default=-math.inf) >= gates["minimum_rad_dino_coverage"], "rad_dino_coverage"),
        (_metric_value(row, "filter_acceptance_rate", default=-math.inf) >= gates["minimum_filter_acceptance_rate"], "filter_acceptance_rate"),
        (_as_bool(row.get("metrics_complete")), "metrics_complete"),
        (not _as_bool(row.get("test_access")), "test_access"),
    ]
    if gates.get("require_lineage"):
        checks.append((_as_bool(row.get("lineage_complete")), "lineage"))
    if gates.get("require_provenance"):
        checks.append((_as_bool(row.get("provenance_manifest_valid")), "provenance"))
    checks.append((_as_bool(row.get("training_corpus_manifest_valid")), "training_corpus"))
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
    """Gate first, then apply the preregistered metric hierarchy without a weighted score."""
    candidates = []
    for source in rows:
        if str(source.get("family", source.get("scientific_family"))) != family: continue
        row = dict(source)
        failures = eligibility_failures(row, gates) if gates is not None else list(row.get("exclusion_reasons", []))
        registry_eligible = _as_bool(row.get("eligible_for_selection", row.get("eligible_for_downstream_selection", False)))
        if not registry_eligible and "registry_role" not in failures: failures.append("registry_role")
        row["exclusion_reasons"] = sorted(set(failures)); row["eligible"] = not row["exclusion_reasons"]
        candidates.append(row)
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
    """Apply the preregistered margin to paired stability differences."""
    low, high = float(paired["stability_interval_low"]), float(paired["stability_interval_high"])
    mean = float(paired["mean_paired_difference"])
    margin = float(protocol["selection"]["practical_equivalence_margin"])
    includes_zero = low <= 0.0 <= high
    return {"paired_stability_interval_includes_zero": includes_zero,
            "absolute_paired_mean_difference": abs(mean), "practical_equivalence_margin": margin,
            "practically_similar": includes_zero and abs(mean) <= margin}


def efficiency_from_manifest(root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Import only explicitly recorded runtime measurements; never estimate missing values."""
    unavailable = {"generation_seconds_per_image": None, "peak_vram_mb": None, "energy_kwh": None,
                   "checkpoint_size_bytes": None, "efficiency_source": None,
                   "efficiency_status": "unavailable", "generation_efficiency_status": "unavailable"}
    value = entry.get("efficiency_manifest")
    path = Path(root) / str(value) if value else None
    if not path or not path.is_file(): return unavailable
    try:
        payload = _read_manifest(path)
        if not isinstance(payload, Mapping): return unavailable
        count = payload.get("images_generated") or payload.get("n_generated")
        if not count and payload.get("n_per_class") and payload.get("generated_classes"):
            count = int(payload["n_per_class"]) * len(payload["generated_classes"])
        seconds_per_image = payload.get("seconds_per_image")
        if seconds_per_image is None and payload.get("elapsed_seconds") is not None and count:
            seconds_per_image = float(payload["elapsed_seconds"]) / int(count)
        checkpoint = Path(root) / str(entry.get("checkpoint", ""))
        values = {"generation_seconds_per_image": seconds_per_image,
                  "peak_vram_mb": payload.get("peak_vram_mb"), "energy_kwh": payload.get("energy_kwh"),
                  "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
                  "efficiency_source": str(value)}
        status = "available" if any(values[key] is not None for key in (
            "generation_seconds_per_image", "peak_vram_mb", "energy_kwh", "checkpoint_size_bytes")) else "unavailable"
        return {**values, "efficiency_status": status, "generation_efficiency_status": status}
    except Exception:
        return unavailable


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> Path:
    if not rows: raise ValueError(f"refusing to create an empty CSV: {path}")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
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
                                 benchmark_rows: Sequence[Mapping[str, Any]], synthetic_pool_target: int = 1361) -> dict[str, str]:
    by_id = {entry["id"]: entry for entry in registry["generators"]}
    results = {str(row["generator_id"]): row for row in benchmark_rows}
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
        if not _as_bool(result.get("technical_gates_passed")): raise ValueError(f"technical gates failed for {generator_id}")
    return selected


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool): return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def save_selected_generators(root: Path, finetuned: str, from_scratch: str, benchmark_rows: Sequence[Mapping[str, Any]],
                             *, notes: str = "", manual_override: bool = False) -> Path:
    protocol, registry = load_protocol(root), load_registry(root)
    selected = validate_selected_generators(finetuned, from_scratch, registry, benchmark_rows,
                                            protocol["synthetic_pool_target"])
    payload = {**selected, "selection_basis": {"primary_metric": "raddino_kid",
               "benchmark_results": protocol["outputs"]["metrics"], "manual_override": bool(manual_override), "notes": notes}}
    return atomic_json(Path(root) / "configs/selected_generators.json", payload)


__all__ = ["BENCHMARK_ROOT", "CANONICAL_OUTPUTS", "FEATURE_SPACES", "FAMILIES", "REPRESENTATIONS", "atomic_json", "audit_candidate",
           "balanced_subsample_indices", "deterministic_sample", "discover_candidates", "duplicate_diagnostics",
           "build_synthetic_duplication_rows", "build_train_memorization_rows", "build_validation_similarity_rows",
           "deterministic_pair_indices", "diversity_metrics", "efficiency_from_manifest", "eligibility_failures", "evaluation_subset_size", "extract_features", "fid", "filter_acceptance_from_manifest", "get_or_extract_embeddings",
           "image_similarity", "kid", "list_image_paths", "load_embedding_cache", "load_protocol", "load_registry", "metadata_positive_paths",
           "multiscale_ssim", "nearest_neighbours", "paired_kid_differences", "plot_generator_summary", "practical_equivalence", "prdc", "rank_generator_family", "render_similarity_panel", "repeated_distribution_metrics", "save_resampling_plan", "save_selected_generators",
           "similarity_summaries", "summarize", "technical_audit", "validate_protocol",
           "synthetic_nearest_neighbours", "technical_validity_row", "training_corpus_from_manifest", "validate_selected_generators", "write_csv_rows", "write_embedding_cache"]
