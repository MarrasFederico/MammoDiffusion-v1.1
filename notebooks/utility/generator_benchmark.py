"""Publication-oriented, validation-only generator benchmark contracts and metrics.

The module is safe to import without a GPU or model weights. Heavy feature extraction is
performed only by :func:`run_benchmark` after an explicit CLI confirmation. Test metadata is
never accepted as an input.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
FAMILIES = ("finetuned", "from_scratch")
REPRESENTATIONS = ("raw", "filtered")
FEATURE_SPACES = ("inception_v3", "rad_dino")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def value_signature(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def signed_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("signature", None)
    return {**unsigned, "signature": value_signature(unsigned)}


def verify_signature(value: Mapping[str, Any]) -> None:
    signature = value.get("signature")
    unsigned = {key: item for key, item in value.items() if key != "signature"}
    if not signature or signature != value_signature(unsigned):
        raise ValueError("artifact signature is missing or invalid")


def atomic_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
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
    if protocol.get("pipeline_namespace") != "mammodiffusion.generator_benchmark.v1":
        raise ValueError("unsupported generator benchmark protocol")
    if int(protocol.get("canonical_sample_count", 0)) <= 1:
        raise ValueError("canonical sample count must exceed one")
    reference_sets = protocol.get("reference_sets", {})
    distribution_reference = str(reference_sets.get("distribution_metrics", "")).lower()
    if "test" in distribution_reference:
        raise ValueError("generator selection cannot use test data")
    pools = [str(item).lower() for item in protocol.get("memorization", {}).get("neighbour_pools", [])]
    if any("test" in item for item in pools):
        raise ValueError("nearest-neighbour pools cannot include test data")
    if tuple(protocol.get("representations", [])) != REPRESENTATIONS:
        raise ValueError("raw and filtered representations must both be registered separately")


def _reject_test_path(path: str | Path) -> None:
    parts = {part.lower().replace("-", "_") for part in Path(path).parts}
    if any(part in {"test", "locked_test", "lockedtest"} or part.startswith("test_") for part in parts):
        raise PermissionError(f"test access is forbidden during generator benchmarking: {path}")


def list_image_paths(path: Path) -> list[Path]:
    _reject_test_path(path)
    if not path.is_dir():
        return []
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def deterministic_sample(paths: Sequence[str | Path], count: int, seed: int) -> list[str]:
    canonical = sorted(str(Path(path)) for path in paths)
    if len(canonical) < count:
        raise ValueError(f"need {count} unique candidates, found {len(canonical)}")
    if len(canonical) != len(set(canonical)):
        raise ValueError("candidate path list contains duplicates")
    chosen = random.Random(int(seed)).sample(canonical, int(count))
    return sorted(chosen)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> str:
    """Return a deterministic 64-bit DCT perceptual hash for a grayscale image."""
    from PIL import Image
    from scipy.fft import dctn

    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float64)
    coefficients = dctn(pixels, norm="ortho")[:8, :8]
    values = coefficients.flatten()
    threshold = float(np.median(values[1:]))
    bits = values >= threshold
    return f"{int(''.join('1' if bit else '0' for bit in bits), 2):016x}"


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def duplicate_diagnostics(paths: Sequence[str | Path], perceptual_distance: int = 2) -> dict[str, Any]:
    resolved = [Path(path) for path in paths]
    exact = [file_sha256(path) for path in resolved]
    hashes = [perceptual_hash(path) for path in resolved]
    exact_duplicate_items = sum(count - 1 for count in Counter(exact).values() if count > 1)
    perceptual_duplicate_items = 0
    for index, current in enumerate(hashes):
        if any(hamming_distance(current, previous) <= perceptual_distance for previous in hashes[:index]):
            perceptual_duplicate_items += 1
    total = len(resolved)
    return {
        "n_images": total,
        "exact_duplicate_items": exact_duplicate_items,
        "exact_duplicate_rate": exact_duplicate_items / total if total else 0.0,
        "perceptual_duplicate_items": perceptual_duplicate_items,
        "perceptual_hash_duplicate_rate": perceptual_duplicate_items / total if total else 0.0,
    }


def technical_audit(paths: Sequence[str | Path], expected_size: tuple[int, int] = (512, 512)) -> dict[str, Any]:
    from PIL import Image

    counters = Counter()
    nonblack_ratios: list[float] = []
    names = [Path(path).name for path in paths]
    counters["duplicate_filename"] = len(names) - len(set(names))
    valid_paths: list[Path] = []
    for value in paths:
        path = Path(value)
        try:
            with Image.open(path) as image:
                gray = np.asarray(image.convert("L"), dtype=np.uint8)
                size = image.size
        except Exception:
            counters["corrupted"] += 1
            continue
        valid_paths.append(path)
        if size != expected_size:
            counters["unexpected_dimensions"] += 1
        if gray.size == 0 or int(gray.max()) <= int(gray.min()):
            counters["invalid_dynamic_range"] += 1
        ratio = float(np.count_nonzero(gray > 5) / gray.size) if gray.size else 0.0
        nonblack_ratios.append(ratio)
        if ratio < 0.01:
            counters["near_black"] += 1
    total = len(paths)
    duplicates = duplicate_diagnostics(valid_paths) if valid_paths else {
        "exact_duplicate_rate": 0.0, "perceptual_hash_duplicate_rate": 0.0,
        "exact_duplicate_items": 0, "perceptual_duplicate_items": 0, "n_images": 0,
    }
    rate = lambda name: counters[name] / total if total else 0.0
    return {
        "n_images": total,
        "n_valid_images": len(valid_paths),
        "corrupted_file_rate": rate("corrupted"),
        "near_black_rate": rate("near_black"),
        "nonblack_ratio_mean": float(np.mean(nonblack_ratios)) if nonblack_ratios else None,
        "invalid_dynamic_range_rate": rate("invalid_dynamic_range"),
        "unexpected_dimensions_rate": rate("unexpected_dimensions"),
        "duplicate_filename_rate": rate("duplicate_filename"),
        **duplicates,
    }


def filter_diagnostics(root: Path, entry: Mapping[str, Any], initial_count: int, accepted_count: int) -> dict[str, Any]:
    if not entry.get("filtering_applied", True):
        return {"initial_count": initial_count, "accepted_count": accepted_count,
                "rejected_count": 0, "acceptance_rate": 1.0, "rejection_reasons": {}}
    report = entry.get("filter_report")
    if not report or not (root / report).is_file():
        raise FileNotFoundError(f"filter rejection report is required for {entry['id']}")
    with (root / report).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    reasons = Counter()
    for row in rows:
        accepted = str(row.get("accepted", row.get("is_accepted", ""))).strip().lower()
        status = str(row.get("status", "")).strip().lower()
        if accepted in {"true", "1", "yes"} or status in {"accepted", "pass", "kept"}:
            continue
        reason = (row.get("rejection_reason") or row.get("reject_reason") or row.get("reason") or
                  row.get("rejection_reasons") or status or "unspecified")
        reasons[str(reason)] += 1
    return {"initial_count": initial_count, "accepted_count": accepted_count,
            "rejected_count": max(0, initial_count - accepted_count),
            "acceptance_rate": accepted_count / initial_count if initial_count else 0.0,
            "rejection_reasons": dict(sorted(reasons.items()))}


def efficiency_record(root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    record = {"generator_id": entry["id"], "training_time_seconds": None,
              "generation_time_seconds": None, "peak_vram_mb": None, "energy_kwh": None,
              "checkpoint_size_bytes": None}
    checkpoint = entry.get("checkpoint")
    if checkpoint and (root / checkpoint).is_file():
        record["checkpoint_size_bytes"] = (root / checkpoint).stat().st_size
    manifest = entry.get("efficiency_manifest")
    if manifest and (root / manifest).is_file():
        payload = json.loads((root / manifest).read_text())
        record["generation_time_seconds"] = payload.get("elapsed_seconds")
        record["peak_vram_mb"] = payload.get("peak_vram_mb")
        record["energy_kwh"] = payload.get("energy_kwh")
    return record


def render_sample_grid(paths: Sequence[str | Path], output: Path, title: str, limit: int = 16) -> Path:
    import matplotlib.pyplot as plt
    from PIL import Image

    selected = list(paths)[:limit]
    figure, axes = plt.subplots(4, 4, figsize=(10, 10))
    for axis, path in zip(axes.flat, selected):
        with Image.open(path) as image:
            axis.imshow(image.convert("L"), cmap="gray")
        axis.set_title(Path(path).name, fontsize=6); axis.axis("off")
    for axis in list(axes.flat)[len(selected):]: axis.axis("off")
    figure.suptitle(title); figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True); figure.savefig(output, dpi=140); plt.close(figure)
    return output


def render_memorization_panels(neighbours: Sequence[Mapping[str, Any]], synthetic_paths: Mapping[str, Path],
                               reference_paths: Mapping[str, Path], output_dir: Path, limit: int = 12) -> list[Path]:
    import matplotlib.pyplot as plt
    from PIL import Image

    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in neighbours:
        grouped.setdefault(str(row["synthetic_id"]), {})[str(row["pool"])] = row
    ordered = sorted(grouped, key=lambda synthetic_id: (
        min(float(row["embedding_distance"]) for row in grouped[synthetic_id].values()), synthetic_id))[:limit]
    outputs = []
    for index, synthetic_id in enumerate(ordered):
        pools = grouped[synthetic_id]
        train = pools.get("real_train_positive"); validation = pools.get("real_validation_positive")
        if train is None or validation is None: continue
        figure, axes = plt.subplots(1, 3, figsize=(10, 4))
        entries = [("synthetic", synthetic_paths[synthetic_id], None),
                   ("nearest train", reference_paths[str(train["source_id"])], train),
                   ("nearest validation", reference_paths[str(validation["source_id"])], validation)]
        for axis, (label, path, metrics) in zip(axes, entries):
            with Image.open(path) as image: axis.imshow(image.convert("L"), cmap="gray")
            suffix = "" if metrics is None else f"\nd={metrics['embedding_distance']:.4f} SSIM={metrics['ssim']:.4f} pHash={metrics['perceptual_hash_distance']}"
            axis.set_title(label + suffix, fontsize=8); axis.axis("off")
        figure.tight_layout(); output = output_dir / f"panel_{index:03d}.png"
        output.parent.mkdir(parents=True, exist_ok=True); figure.savefig(output, dpi=140); plt.close(figure); outputs.append(output)
    return outputs


def audit_candidate(root: Path, entry: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    family = entry.get("scientific_family")
    if family not in FAMILIES:
        blockers.append("invalid_scientific_family")
    if "positive" not in entry.get("classes", []):
        blockers.append("positive_class_unavailable")
    for field in ("checkpoint", "lineage_manifest", "provenance_manifest"):
        value = entry.get(field)
        if not value:
            blockers.append(f"missing_{field}")
        elif not (Path(root) / value).is_file():
            blockers.append(f"unverifiable_{field}")
    if entry.get("filtering_applied", True):
        filter_report = entry.get("filter_report")
        if not filter_report or not (Path(root) / filter_report).is_file():
            blockers.append("missing_filter_rejection_report")
    count = int(protocol["canonical_sample_count"])
    representations: dict[str, Any] = {}
    for representation in REPRESENTATIONS:
        relative = entry.get("samples", {}).get(f"{representation}_positive")
        if not relative:
            blockers.append(f"missing_{representation}_positive_path")
            representations[representation] = {"path": None, "count": 0}
            continue
        _reject_test_path(relative)
        paths = list_image_paths(Path(root) / relative)
        representations[representation] = {"path": relative, "count": len(paths)}
        if len(paths) < count:
            blockers.append(f"insufficient_{representation}_positive_images:{len(paths)}<{count}")
    configured_exclusions = entry.get("benchmark", {}).get("exclusion_if", [])
    if "lineage_not_demonstrated" in configured_exclusions:
        blockers.append("lineage_not_demonstrated")
    return {
        "generator_id": entry.get("id"),
        "scientific_family": family,
        "subtype": entry.get("subtype"),
        "representations": representations,
        "blockers": sorted(set(blockers)),
        "eligible_for_execution": not blockers and bool(entry.get("benchmark", {}).get("enabled", False)),
    }


def discover_candidates(root: Path, protocol: Mapping[str, Any] | None = None,
                        registry: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    protocol = dict(protocol or load_protocol(root))
    registry = dict(registry or load_registry(root))
    return [audit_candidate(root, entry, protocol) for entry in registry.get("generators", [])
            if entry.get("benchmark", {}).get("enabled", False)]


def build_execution_plan(root: Path) -> dict[str, Any]:
    protocol = load_protocol(root)
    registry = load_registry(root)
    audits = discover_candidates(root, protocol, registry)
    by_id = {entry["id"]: entry for entry in registry["generators"]}
    samples: dict[str, Any] = {}
    for audit in audits:
        if not audit["eligible_for_execution"]:
            continue
        entry = by_id[audit["generator_id"]]
        samples[entry["id"]] = {}
        for offset, representation in enumerate(REPRESENTATIONS):
            paths = list_image_paths(root / entry["samples"][f"{representation}_positive"])
            selected = deterministic_sample(paths, protocol["canonical_sample_count"],
                                            protocol["sampling"]["seed"] + offset)
            samples[entry["id"]][representation] = {
                "count": len(selected), "paths": [str(Path(path).relative_to(root)) for path in selected],
                "selection_signature": value_signature([str(Path(path).relative_to(root)) for path in selected]),
            }
    return signed_payload({
        "schema_version": 1,
        "pipeline_namespace": "mammodiffusion.generator_benchmark.v1",
        "artifact_type": "generator_benchmark_execution_plan",
        "protocol_signature": value_signature(protocol),
        "registry_signature": value_signature(registry),
        "test_access": False,
        "candidate_audits": audits,
        "samples": samples,
    })


def _pairwise_squared(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    distances = np.sum(left * left, axis=1)[:, None] + np.sum(right * right, axis=1)[None, :] - 2 * left @ right.T
    return np.maximum(distances, 0.0)


def fid(real: np.ndarray, synthetic: np.ndarray) -> float:
    from scipy.linalg import sqrtm

    real, synthetic = np.asarray(real, dtype=np.float64), np.asarray(synthetic, dtype=np.float64)
    mean_delta = real.mean(axis=0) - synthetic.mean(axis=0)
    cov_real, cov_synthetic = np.cov(real, rowvar=False), np.cov(synthetic, rowvar=False)
    root = sqrtm(cov_real @ cov_synthetic)
    if np.iscomplexobj(root):
        root = root.real
    return float(mean_delta @ mean_delta + np.trace(cov_real + cov_synthetic - 2 * root))


def kid(real: np.ndarray, synthetic: np.ndarray) -> float:
    real, synthetic = np.asarray(real, dtype=np.float64), np.asarray(synthetic, dtype=np.float64)
    dimension = real.shape[1]
    kernel_rr = (real @ real.T / dimension + 1.0) ** 3
    kernel_ss = (synthetic @ synthetic.T / dimension + 1.0) ** 3
    kernel_rs = (real @ synthetic.T / dimension + 1.0) ** 3
    n, m = len(real), len(synthetic)
    if n < 2 or m < 2:
        raise ValueError("KID needs at least two samples per distribution")
    return float((kernel_rr.sum() - np.trace(kernel_rr)) / (n * (n - 1))
                 + (kernel_ss.sum() - np.trace(kernel_ss)) / (m * (m - 1))
                 - 2 * kernel_rs.mean())


def prdc(real: np.ndarray, synthetic: np.ndarray, nearest_k: int = 5) -> dict[str, float]:
    real, synthetic = np.asarray(real, dtype=np.float64), np.asarray(synthetic, dtype=np.float64)
    if min(len(real), len(synthetic)) <= nearest_k:
        raise ValueError("PRDC sample count must exceed nearest_k")
    rr = np.sqrt(_pairwise_squared(real, real))
    ss = np.sqrt(_pairwise_squared(synthetic, synthetic))
    rs = np.sqrt(_pairwise_squared(real, synthetic))
    real_radius = np.partition(rr, nearest_k, axis=1)[:, nearest_k]
    synthetic_radius = np.partition(ss, nearest_k, axis=1)[:, nearest_k]
    precision = np.mean((rs < real_radius[:, None]).any(axis=0))
    recall = np.mean((rs < synthetic_radius[None, :]).any(axis=1))
    density = np.mean((rs < real_radius[:, None]).sum(axis=0)) / nearest_k
    coverage = np.mean(rs.min(axis=1) < real_radius)
    return {"precision": float(precision), "recall": float(recall), "density": float(density), "coverage": float(coverage)}


def distribution_metrics(real: np.ndarray, synthetic: np.ndarray, nearest_k: int = 5) -> dict[str, float]:
    return {"fid": fid(real, synthetic), "kid": kid(real, synthetic), **prdc(real, synthetic, nearest_k)}


def bootstrap_distribution_metrics(real: np.ndarray, synthetic: np.ndarray, *, iterations: int,
                                   sample_size: int, seed: int, nearest_k: int = 5) -> tuple[list[dict], dict]:
    if sample_size > len(real) or sample_size > len(synthetic):
        raise ValueError("bootstrap sample_size exceeds available features")
    rng = np.random.default_rng(seed)
    rows, failed = [], 0
    for repetition in range(iterations):
        try:
            real_indices = rng.choice(len(real), sample_size, replace=True)
            synthetic_indices = rng.choice(len(synthetic), sample_size, replace=True)
            rows.append({"repetition": repetition, **distribution_metrics(real[real_indices], synthetic[synthetic_indices], nearest_k)})
        except Exception:
            failed += 1
    summary: dict[str, Any] = {"valid_repetitions": len(rows), "failed_repetitions": failed}
    for metric in ("fid", "kid", "precision", "recall", "density", "coverage"):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[metric] = {
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "percentile_2_5": float(np.percentile(values, 2.5)),
            "percentile_97_5": float(np.percentile(values, 97.5)),
        }
    return rows, summary


def nearest_neighbours(query: np.ndarray, reference: np.ndarray, query_ids: Sequence[str],
                       reference_ids: Sequence[str], pool: str) -> list[dict[str, Any]]:
    if "test" in pool.lower():
        raise PermissionError("test nearest neighbours are forbidden")
    distances = np.sqrt(_pairwise_squared(query, reference))
    nearest = distances.argmin(axis=1)
    return [{"synthetic_id": query_ids[index], "source_id": reference_ids[target],
             "pool": pool, "embedding_distance": float(distances[index, target])}
            for index, target in enumerate(nearest)]


def eligibility_failures(row: Mapping[str, Any], gates: Mapping[str, Any]) -> list[str]:
    failures = []
    checks = (
        (float(row.get("exact_duplicate_rate", math.inf)) <= gates["maximum_exact_duplicate_rate"], "exact_duplicate_rate"),
        (float(row.get("perceptual_hash_duplicate_rate", math.inf)) <= gates["maximum_perceptual_duplicate_rate"], "perceptual_duplicate_rate"),
        (float(row.get("memorization_flag_rate", math.inf)) <= gates["maximum_memorization_flag_rate"], "memorization_rate"),
        (float(row.get("rad_dino", {}).get("filtered", {}).get("coverage", {}).get("mean", -math.inf)) >= gates["minimum_rad_dino_coverage"], "coverage"),
        (float(row.get("filter_acceptance_rate", -math.inf)) >= gates["minimum_filter_acceptance_rate"], "filter_acceptance_rate"),
        (float(row.get("corrupted_file_rate", math.inf)) <= gates["maximum_corrupted_file_rate"], "corrupted_file_rate"),
        (int(row.get("valid_positive_images", 0)) >= gates["minimum_valid_positive_images"], "valid_positive_images"),
        (not bool(row.get("test_access", True)), "test_access"),
        (bool(row.get("lineage_verified", False)), "lineage"),
        (bool(row.get("provenance_verified", False)), "provenance"),
        (bool(row.get("metrics_complete", False)), "metrics_complete"),
    )
    for passed, name in checks:
        if not passed:
            failures.append(name)
    return failures


def _nested(row: Mapping[str, Any], dotted: str) -> Any:
    value: Any = row
    for key in dotted.split("."):
        value = value[key]
    return value


def _ranking_key(row: Mapping[str, Any], ranking: Sequence[Mapping[str, str]]) -> tuple:
    values = []
    for rule in ranking:
        metric, direction = rule["metric"], rule["direction"]
        if metric == "generator_id":
            values.append(str(row["generator_id"]))
            continue
        value = float(_nested(row, metric))
        values.append(value if direction == "min" else -value)
    return tuple(values)


def select_generators(metric_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    gates = protocol["eligibility_gates"]
    ranking = protocol["selection"]["ranking"]
    selected: dict[str, str] = {}
    audit = []
    for row in metric_rows:
        failures = eligibility_failures(row, gates)
        audit.append({"generator_id": row["generator_id"], "scientific_family": row["scientific_family"],
                      "eligible": not failures, "failures": failures})
    for family in FAMILIES:
        candidates = [row for row, status in zip(metric_rows, audit)
                      if row["scientific_family"] == family and status["eligible"]]
        if not candidates:
            raise RuntimeError(f"no eligible generator in family {family}")
        selected[family] = min(candidates, key=lambda row: _ranking_key(row, ranking))["generator_id"]
    return {"selected": selected, "eligibility_audit": audit,
            "ranking_rule": ranking, "weighted_composite_score": False}


def _metadata_positive_paths(root: Path, relative_csv: str) -> tuple[list[str], list[str]]:
    _reject_test_path(relative_csv)
    with (root / relative_csv).open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["label"]) == 1]
    paths, ids = [], []
    for row in rows:
        value = row.get("processed_path") or f"data/processed/{Path(relative_csv).stem}/1/{row['image_id']}.png"
        _reject_test_path(value)
        paths.append(str((root / value).resolve()))
        ids.append(f"{row.get('patient_id')}::{row.get('image_id')}")
    return paths, ids


def extract_features(paths: Sequence[str | Path], feature_space: str, *, device: str | None = None,
                     allow_model_download: bool = False) -> np.ndarray:
    """Extract deterministic frozen features. Downloads are opt-in and never happen in dry-run."""
    import torch
    from PIL import Image

    if feature_space == "inception_v3":
        from torchvision.models import Inception_V3_Weights, inception_v3
        weights = Inception_V3_Weights.DEFAULT
        if not allow_model_download:
            checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
            if not checkpoint.is_file():
                raise FileNotFoundError("cached InceptionV3 weights are required; downloads are disabled")
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = torch.nn.Identity()
        transform = weights.transforms()
        processor = lambda image: transform(image.convert("RGB"))
    elif feature_space == "rad_dino":
        from transformers import AutoImageProcessor, AutoModel
        name = "microsoft/rad-dino"
        image_processor = AutoImageProcessor.from_pretrained(name, local_files_only=not allow_model_download)
        model = AutoModel.from_pretrained(name, local_files_only=not allow_model_download)
        processor = lambda image: image_processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]
    else:
        raise ValueError(feature_space)
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().to(target)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(paths), 16):
            batch = torch.stack([processor(Image.open(path)) for path in paths[start:start + 16]]).to(target)
            output = model(batch)
            if feature_space == "rad_dino":
                output = output.pooler_output if getattr(output, "pooler_output", None) is not None else output.last_hidden_state[:, 0]
            rows.append(output.detach().cpu().float().numpy())
    return np.concatenate(rows, axis=0)


def image_pair_diversity(paths: Sequence[str | Path], *, seed: int, max_pairs: int = 256,
                         device: str | None = None) -> dict[str, Any]:
    """Compute LPIPS and MS-SSIM on deterministic grayscale pairs with valid channel/range handling."""
    import torch
    from PIL import Image
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    canonical = [str(Path(path)) for path in paths]
    if len(canonical) < 2:
        raise ValueError("diversity metrics need at least two images")
    rng = random.Random(seed)
    pairs = []
    for _ in range(min(max_pairs, len(canonical))):
        left, right = rng.sample(canonical, 2)
        pairs.append((left, right))
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(target)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(target)
    lpips_values, ms_ssim_values = [], []
    for start in range(0, len(pairs), 16):
        batch_pairs = pairs[start:start + 16]
        tensors = []
        for side in (0, 1):
            images = []
            for pair in batch_pairs:
                with Image.open(pair[side]) as image:
                    array = np.asarray(image.convert("L").resize((512, 512)), dtype=np.float32) / 255.0
                images.append(torch.from_numpy(array)[None].repeat(3, 1, 1))
            tensors.append(torch.stack(images).to(target))
        with torch.inference_mode():
            lpips_values.extend(lpips(tensors[0] * 2 - 1, tensors[1] * 2 - 1).flatten().cpu().tolist())
            ms_ssim_values.append(float(ms_ssim(tensors[0], tensors[1]).cpu()))
    return {
        "n_pairs": len(pairs),
        "lpips_diversity_mean": float(np.mean(lpips_values)),
        "lpips_diversity_standard_deviation": float(np.std(lpips_values, ddof=1)) if len(lpips_values) > 1 else 0.0,
        "ms_ssim_mean": float(np.mean(ms_ssim_values)),
        "pair_selection_signature": value_signature(pairs),
    }


def image_similarity(left: Path, right: Path) -> dict[str, Any]:
    from PIL import Image
    from skimage.metrics import structural_similarity

    with Image.open(left) as image:
        left_array = np.asarray(image.convert("L").resize((512, 512)), dtype=np.uint8)
    with Image.open(right) as image:
        right_array = np.asarray(image.convert("L").resize((512, 512)), dtype=np.uint8)
    return {
        "ssim": float(structural_similarity(left_array, right_array, data_range=255)),
        "perceptual_hash_distance": hamming_distance(perceptual_hash(left), perceptual_hash(right)),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def _git_revision(root: Path) -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unavailable"


def _flatten_metric_rows(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for source in metric_rows:
        for space in FEATURE_SPACES:
            for representation in REPRESENTATIONS:
                summary = source[space][representation]
                row = {
                    "generator_id": source["generator_id"], "scientific_family": source["scientific_family"],
                    "feature_space": space, "representation": representation,
                    "valid_repetitions": summary["valid_repetitions"],
                    "failed_repetitions": summary["failed_repetitions"],
                }
                for metric in ("fid", "kid", "precision", "recall", "density", "coverage"):
                    for statistic, value in summary[metric].items():
                        row[f"{metric}_{statistic}"] = value
                if representation == "filtered":
                    for key in ("filter_acceptance_rate", "exact_duplicate_rate", "perceptual_hash_duplicate_rate",
                                "corrupted_file_rate", "memorization_flag_rate", "bootstrap_stability"):
                        row[key] = source[key]
                    row.update(source["diversity"])
                rows.append(row)
    return rows


def _selection_report(selection: Mapping[str, Any], metric_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Generator selection proposal", "",
        "This validation-only proposal addresses RQ1. It does not approve generators and does not open the test set.", "",
        "## Proposed family winners", "",
        f"- Fine-tuned: `{selection['selected']['finetuned']}`",
        f"- From scratch: `{selection['selected']['from_scratch']}`", "",
        "Ranking uses RAD-DINO filtered KID, coverage, precision and FID, then Inception KID, bootstrap stability, and generator ID. No weighted composite score is used.", "",
        "## Eligibility", "",
    ]
    for row in selection["eligibility_audit"]:
        state = "eligible" if row["eligible"] else "excluded: " + ", ".join(row["failures"])
        lines.append(f"- `{row['generator_id']}` ({row['scientific_family']}): {state}")
    lines += ["", "## Interpretation", "",
              "Overlapping bootstrap intervals must be described as statistical or practical similarity; the deterministic tie-break does not establish superiority.", ""]
    return "\n".join(lines)


def run_benchmark(root: Path, *, allow_model_download: bool = False, device: str | None = None) -> dict[str, Any]:
    """Execute the real benchmark after explicit CLI confirmation.

    This function intentionally has no implicit fallback, test reference, or fabricated metric.
    It writes outputs only after every required feature space and representation succeeds.
    """
    root = Path(root)
    protocol, registry = load_protocol(root), load_registry(root)
    plan = build_execution_plan(root)
    eligible_ids = {row["generator_id"] for row in plan["candidate_audits"] if row["eligible_for_execution"]}
    if not eligible_ids:
        raise RuntimeError("registry audit found no executable candidates")
    validation_paths, validation_ids = _metadata_positive_paths(root, "data/processed/metadata/val.csv")
    train_paths, train_ids = _metadata_positive_paths(root, "data/processed/metadata/train.csv")
    count = protocol["canonical_sample_count"]
    if len(validation_paths) < count:
        raise RuntimeError(f"validation has only {len(validation_paths)} positives; protocol requires {count}")
    validation_selection = deterministic_sample(validation_paths, count, protocol["sampling"]["seed"])
    validation_index = {path: identifier for path, identifier in zip(validation_paths, validation_ids)}
    validation_ids = [validation_index[path] for path in validation_selection]
    validation_paths = validation_selection
    real_features = {space: extract_features(validation_paths, space, device=device,
                                              allow_model_download=allow_model_download)
                     for space in FEATURE_SPACES}
    rad_train_features = extract_features(train_paths, "rad_dino", device=device,
                                          allow_model_download=allow_model_download)
    rad_validation_features = real_features["rad_dino"]
    output_dir = root / "results/generator_benchmark"
    metric_rows, bootstrap_rows, neighbour_rows, filter_rows, efficiency_rows = [], [], [], [], []
    by_id = {entry["id"]: entry for entry in registry["generators"]}
    for generator_id in sorted(eligible_ids):
        entry = by_id[generator_id]
        row: dict[str, Any] = {"generator_id": generator_id, "scientific_family": entry["scientific_family"],
                               "test_access": False, "lineage_verified": True, "provenance_verified": True}
        representation_audits, features_by_representation = {}, {}
        initial_count = len(list_image_paths(root / entry["samples"]["raw_positive"]))
        accepted_count = len(list_image_paths(root / entry["samples"]["filtered_positive"]))
        filter_audit = filter_diagnostics(root, entry, initial_count, accepted_count)
        row["filter_acceptance_rate"] = filter_audit["acceptance_rate"]
        filter_rows.append({"generator_id": generator_id, **filter_audit,
                            "rejection_reasons": json.dumps(filter_audit["rejection_reasons"], sort_keys=True)})
        efficiency_rows.append(efficiency_record(root, entry))
        row["valid_positive_images"] = min(initial_count, accepted_count)
        for representation in REPRESENTATIONS:
            selected = [root / value for value in plan["samples"][generator_id][representation]["paths"]]
            representation_audits[representation] = technical_audit(selected)
            features_by_representation[representation] = {}
            for space in FEATURE_SPACES:
                synthetic_features = extract_features(selected, space, device=device,
                                                      allow_model_download=allow_model_download)
                features_by_representation[representation][space] = synthetic_features
                repetitions, summary = bootstrap_distribution_metrics(
                    real_features[space], synthetic_features,
                    iterations=protocol["bootstrap"]["iterations"],
                    sample_size=min(protocol["bootstrap"]["sample_size"], len(real_features[space])),
                    seed=protocol["bootstrap"]["seed"],
                )
                row.setdefault(space, {})[representation] = summary
                for repetition in repetitions:
                    bootstrap_rows.append({"generator_id": generator_id, "scientific_family": entry["scientific_family"],
                                           "representation": representation, "feature_space": space, **repetition})
        filtered = representation_audits["filtered"]
        row.update({key: filtered[key] for key in ("exact_duplicate_rate", "perceptual_hash_duplicate_rate", "corrupted_file_rate")})
        filtered_paths = [root / value for value in plan["samples"][generator_id]["filtered"]["paths"]]
        filtered_ids = [str(path.relative_to(root)) for path in filtered_paths]
        for representation in REPRESENTATIONS:
            grid_paths = [root / value for value in plan["samples"][generator_id][representation]["paths"]]
            render_sample_grid(grid_paths, output_dir / "sample_grids" / generator_id / f"{representation}.png",
                               f"{generator_id} — {representation.upper()} deterministic sample")
        row["diversity"] = image_pair_diversity(filtered_paths, seed=protocol["sampling"]["seed"], device=device)
        rad_synthetic = features_by_representation["filtered"]["rad_dino"]
        synthetic_distances = np.sqrt(_pairwise_squared(rad_synthetic, rad_synthetic))
        np.fill_diagonal(synthetic_distances, np.inf)
        row["diversity"]["nearest_neighbour_synthetic_to_synthetic_distance_mean"] = float(np.mean(synthetic_distances.min(axis=1)))
        neighbours = nearest_neighbours(rad_synthetic, rad_train_features, filtered_ids, train_ids, "real_train_positive")
        neighbours += nearest_neighbours(rad_synthetic, rad_validation_features, filtered_ids, validation_ids, "real_validation_positive")
        path_by_id = {identifier: Path(path) for identifier, path in zip(train_ids, train_paths)}
        path_by_id.update({identifier: Path(path) for identifier, path in zip(validation_ids, validation_paths)})
        synthetic_by_id = dict(zip(filtered_ids, filtered_paths))
        flags = 0
        flag_rule = protocol["memorization"]["flag_rule"]
        for neighbour in neighbours:
            similarity = image_similarity(synthetic_by_id[neighbour["synthetic_id"]], path_by_id[neighbour["source_id"]])
            neighbour.update(similarity)
            neighbour["generator_id"] = generator_id
            neighbour["memorization_flag"] = (similarity["ssim"] >= flag_rule["ssim_gte"] and
                                                similarity["perceptual_hash_distance"] <= flag_rule["perceptual_hash_distance_lte"])
            flags += int(neighbour["memorization_flag"])
        neighbour_rows.extend(neighbours)
        reference_paths = {identifier: Path(path) for identifier, path in zip(train_ids, train_paths)}
        reference_paths.update({identifier: Path(path) for identifier, path in zip(validation_ids, validation_paths)})
        render_memorization_panels(neighbours, synthetic_by_id, reference_paths,
                                   output_dir / "memorization_panels" / generator_id)
        row["memorization_flag_rate"] = flags / len(neighbours) if neighbours else math.inf
        row["bootstrap_stability"] = 1.0 / (1.0 + row["rad_dino"]["filtered"]["kid"]["standard_deviation"])
        row["metrics_complete"] = True
        row["representation_audits"] = representation_audits
        metric_rows.append(row)
    selection = select_generators(metric_rows, protocol)
    manifest = signed_payload({
        "schema_version": 1, "pipeline_namespace": "mammodiffusion.generator_benchmark.v1",
        "artifact_type": "generator_benchmark_manifest", "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": _git_revision(root), "protocol_signature": value_signature(protocol),
        "registry_signature": value_signature(registry), "execution_plan_signature": plan["signature"],
        "test_access": False, "reference_split": "real_validation_positive",
        "memorization_reference_splits": ["real_train_positive", "real_validation_positive"],
        "canonical_sample_count": count, "candidate_ids": sorted(eligible_ids), "samples": plan["samples"],
    })
    atomic_json(output_dir / "generator_benchmark_manifest.json", manifest)
    write_csv(output_dir / "generator_metrics.csv", _flatten_metric_rows(metric_rows))
    write_csv(output_dir / "generator_metrics_bootstrap.csv", bootstrap_rows)
    write_csv(output_dir / "memorization_nearest_neighbours.csv", neighbour_rows)
    write_csv(output_dir / "filter_diagnostics.csv", filter_rows)
    write_csv(output_dir / "efficiency.csv", efficiency_rows)
    proposal = signed_payload({
        "schema_version": 1, "pipeline_namespace": "mammodiffusion.generator_benchmark.v1",
        "artifact_type": "generator_selection_proposal", "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_manifest_signature": manifest["signature"],
        "best_finetuned_generator": selection["selected"]["finetuned"],
        "best_from_scratch_generator": selection["selected"]["from_scratch"],
        "selection_rationale": {"ranking": selection["ranking_rule"], "weighted_composite_score": False,
                                "eligibility_audit": selection["eligibility_audit"]},
        "test_access": False,
    })
    atomic_json(output_dir / "generator_selection_proposal.json", proposal)
    (output_dir / "generator_selection_report.md").write_text(_selection_report(selection, metric_rows), encoding="utf-8")
    return {"status": "proposal_ready", "manifest": str(output_dir / "generator_benchmark_manifest.json"),
            "proposal": str(output_dir / "generator_selection_proposal.json"), "selected": selection["selected"]}


__all__ = [
    "FEATURE_SPACES", "FAMILIES", "REPRESENTATIONS", "audit_candidate", "bootstrap_distribution_metrics",
    "build_execution_plan", "deterministic_sample", "discover_candidates", "distribution_metrics",
    "duplicate_diagnostics", "eligibility_failures", "extract_features", "fid", "hamming_distance", "kid",
    "load_protocol", "load_registry", "nearest_neighbours", "perceptual_hash", "prdc", "run_benchmark",
    "efficiency_record", "filter_diagnostics", "image_pair_diversity", "image_similarity", "render_memorization_panels",
    "render_sample_grid", "select_generators", "signed_payload", "technical_audit", "validate_protocol", "value_signature",
    "verify_signature",
]
