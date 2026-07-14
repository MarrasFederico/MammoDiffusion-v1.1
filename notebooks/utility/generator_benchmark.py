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
    if int(resampling.get("fid_repetitions", 999)) > 10:
        raise ValueError("FID is secondary and must use a small independent repetition count")
    ranking_metrics = [item["metric"] for item in protocol["selection"]["ranking"]]
    if ranking_metrics[0] != "rad_dino.filtered.kid.mean" or any("fid" in item for item in ranking_metrics):
        raise ValueError("KID must be primary and FID must remain descriptive")


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


def evaluation_subset_size(synthetic_pool_count: int, real_reference_count: int, synthetic_pool_target: int = 1361) -> int:
    """Validate the synthetic pool and return the balanced per-repetition size."""
    synthetic_pool_count, real_reference_count = int(synthetic_pool_count), int(real_reference_count)
    if synthetic_pool_count < int(synthetic_pool_target):
        raise ValueError(f"synthetic pool below target: {synthetic_pool_count} < {synthetic_pool_target}")
    if real_reference_count < 2:
        raise ValueError("at least two real validation positives are required")
    return min(real_reference_count, synthetic_pool_count)


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
    from scipy.linalg import sqrtm
    real, synthetic = np.asarray(real, dtype=np.float64), np.asarray(synthetic, dtype=np.float64)
    delta = real.mean(axis=0) - synthetic.mean(axis=0)
    cov_real, cov_synthetic = np.cov(real, rowvar=False), np.cov(synthetic, rowvar=False)
    root = sqrtm(cov_real @ cov_synthetic)
    if np.iscomplexobj(root):
        root = root.real
    return float(delta @ delta + np.trace(cov_real + cov_synthetic - 2 * root))


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
                                  *, seed: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute KID/PRDC/FID with independent, balanced, no-replacement repetitions."""
    cfg = protocol["resampling"]
    size = evaluation_subset_size(len(synthetic), len(real), int(protocol["synthetic_pool_target"]))
    base_seed = int(seed if seed is not None else protocol["sampling"]["seed"])
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    specifications = (("kid", int(cfg["kid_repetitions"]), None),
                      ("prdc", int(cfg["prdc_repetitions"]), int(cfg["nearest_neighbour_k"])),
                      ("fid", int(cfg["fid_repetitions"]), None))
    for offset, (name, repetitions, nearest_k) in enumerate(specifications):
        plans = balanced_subsample_indices(len(real), len(synthetic), size, repetitions, base_seed + offset * 100000,
                                           nearest_neighbour_k=nearest_k)
        values: dict[str, list[float]] = {}
        for plan in plans:
            real_subset, synthetic_subset = real[plan["real_indices"]], synthetic[plan["synthetic_indices"]]
            measured = {"kid": kid, "fid": fid}[name](real_subset, synthetic_subset) if name != "prdc" \
                else prdc(real_subset, synthetic_subset, nearest_k=nearest_k or 5)
            metrics = measured if isinstance(measured, dict) else {name: measured}
            rows.append({**plan, "metric_group": name, "subset_size": size, **metrics})
            for metric, value in metrics.items():
                values.setdefault(metric, []).append(float(value))
        summaries.update({metric: summarize(metric_values) for metric, metric_values in values.items()})
    return rows, {"evaluation_subset_size": size, **summaries}


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
    for value in paths:
        try:
            with Image.open(value) as image:
                gray = np.asarray(image.convert("L"), dtype=np.uint8)
                if image.size != expected_size: counters["unexpected_dimensions"] += 1
                if not gray.size or int(gray.max()) <= int(gray.min()): counters["invalid_dynamic_range"] += 1
                if gray.size and np.count_nonzero(gray > 5) / gray.size < 0.01: counters["near_black"] += 1
        except Exception:
            counters["corrupted"] += 1
    total = len(paths)
    return {"n_images": total, **{f"{key}_rate": counters[key] / total if total else 0.0
                                   for key in ("corrupted", "unexpected_dimensions", "invalid_dynamic_range", "near_black")}}


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
    if not entry.get("provenance_manifest"): blockers.append("missing_provenance_manifest")
    role = entry.get("candidate_role", "primary_candidate")
    eligible = bool(entry.get("eligible_for_downstream_selection", False))
    return {"generator_id": entry["id"], "scientific_family": entry.get("scientific_family"),
            "model_family": entry.get("model_family"), "model_variant": entry.get("model_variant"),
            "sampling_steps": entry.get("sampling_steps"), "candidate_role": role,
            "eligible_for_downstream_selection": eligible, "representations": representations,
            "blockers": sorted(set(blockers)), "eligible_for_benchmark_execution": not blockers}


def discover_candidates(root: Path, protocol: Mapping[str, Any] | None = None,
                        registry: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    protocol, registry = protocol or load_protocol(root), registry or load_registry(root)
    return [audit_candidate(Path(root), entry, protocol) for entry in registry["generators"]
            if entry.get("benchmark", {}).get("enabled", False)]


def write_embedding_cache(path: Path, features: np.ndarray, metadata: Mapping[str, Any]) -> tuple[Path, Path]:
    required = {"image_ids", "image_paths", "extractor", "preprocessing", "dimension", "code_version", "source_manifest"}
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
                              extract_fn: Callable[[Sequence[str | Path], str], np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    if Path(path).is_file() and Path(path).with_suffix(Path(path).suffix + ".metadata.json").is_file():
        features, metadata = load_embedding_cache(path)
        if metadata["image_ids"] == list(image_ids) and metadata["extractor"] == extractor:
            return features, metadata
    features = extract_fn(image_paths, extractor)
    metadata = {"image_ids": list(image_ids), "image_paths": [str(value) for value in image_paths],
                "extractor": extractor, "preprocessing": preprocessing, "dimension": int(features.shape[1]),
                "code_version": code_version, "source_manifest": source_manifest}
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
    from skimage.metrics import structural_similarity
    with Image.open(left) as image: left_array = np.asarray(image.convert("L").resize((512, 512)), dtype=np.uint8)
    with Image.open(right) as image: right_array = np.asarray(image.convert("L").resize((512, 512)), dtype=np.uint8)
    return {"ssim": float(structural_similarity(left_array, right_array, data_range=255)),
            "perceptual_hash_distance": hamming_distance(perceptual_hash(left), perceptual_hash(right)),
            "exact_hash_match": file_sha256(left) == file_sha256(right)}


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
        (float(row.get("synthetic_exact_duplicate_rate", math.inf)) <= gates["maximum_exact_duplicate_rate"], "exact_duplicate_rate"),
        (float(row.get("perceptual_hash_duplicate_rate", math.inf)) <= gates["maximum_perceptual_duplicate_rate"], "perceptual_duplicate_rate"),
        (float(row.get("train_memorization_rate", math.inf)) <= gates["maximum_train_memorization_rate"], "train_memorization_rate"),
        (bool(row.get("metrics_complete")), "metrics_complete"),
        (row.get("test_access") is False, "test_access"),
    ]
    if not row.get("eligible_for_downstream_selection", False): checks.append((False, "registry_role"))
    return [name for passed, name in checks if not passed]


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
                             *, notes: str = "") -> Path:
    protocol, registry = load_protocol(root), load_registry(root)
    selected = validate_selected_generators(finetuned, from_scratch, registry, benchmark_rows,
                                            protocol["synthetic_pool_target"])
    payload = {**selected, "selection_basis": {"primary_metric": protocol["selection"]["primary_metric"],
               "benchmark_result_path": protocol["outputs"]["metrics"], "notes": notes}}
    return atomic_json(Path(root) / "configs/selected_generators.json", payload)


__all__ = ["FEATURE_SPACES", "FAMILIES", "REPRESENTATIONS", "atomic_json", "audit_candidate",
           "balanced_subsample_indices", "deterministic_sample", "discover_candidates", "duplicate_diagnostics",
           "eligibility_failures", "evaluation_subset_size", "extract_features", "fid", "get_or_extract_embeddings",
           "image_similarity", "kid", "list_image_paths", "load_embedding_cache", "load_protocol", "load_registry", "metadata_positive_paths",
           "nearest_neighbours", "prdc", "render_similarity_panel", "repeated_distribution_metrics", "save_selected_generators",
           "similarity_summaries", "summarize", "technical_audit", "validate_protocol",
           "synthetic_nearest_neighbours", "validate_selected_generators", "write_embedding_cache"]
