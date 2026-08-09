"""Resolve one publication-protocol condition into train/validation file lists.

Synthetic positives come straight from the selected generator's canonical FILTERED pool
directory (``data/synthetic/<generator_id>/positive``, as recorded in the generator registry).
The pool is listed deterministically and verified for the expected image count, readability,
uniqueness, and the absence of any test-split path using direct, local checks.
The only hashing that remains here computes a dataset signature used purely
for resume/checkpoint compatibility, which is never propagated into configs or downstream gates.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

CLASS_LABEL = {"negative": 0, "positive": 1}
EXPECTED_FILTERED_IMAGE_COUNT = 1361


def deterministic_sample_signature(paths: list[str], count: int, seed: int) -> dict:
    import random
    canonical = sorted(paths)
    if len(canonical) < count:
        raise ValueError(f"need {count} synthetic files, found {len(canonical)}")
    picked = sorted(random.Random(seed).sample(canonical, count))
    signature = hashlib.sha256("\n".join(picked).encode("utf-8")).hexdigest()
    return {"picked": picked, "count": count, "seed": seed, "sha256": signature}

REAL_TRAIN_DIR = "data/processed/train"
AUGMENTED_DIR = "data/real_augmented"
VALIDATION_METADATA = "data/processed/metadata/val.csv"
TEST_METADATA = "data/processed/metadata/test.csv"
_FILE_HASH_CACHE: dict[tuple[str, int, int, int], str] = {}

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_TEST_PATH_COMPONENTS = {"test", "historical_internal_test"}


def _load_selection_payload(root: Path) -> dict | None:
    try:
        from . import classifier_protocol as dp
    except ImportError:
        import classifier_protocol as dp
    return dp.load_selected_generators(root, required=False)


def selected_family_for_generator(root: Path, generator_id: str) -> str | None:
    """Return the primary family whose approved selection is this generator, or None."""
    payload = _load_selection_payload(root)
    if not payload:
        return None
    for family in ("finetuned", "from_scratch"):
        if payload.get(family) == generator_id:
            return family
    return None


def selected_pool_dir(root: Path, generator_id: str, klass: str = "positive") -> Path:
    """Canonical FILTERED pool directory for a generator's class, taken from the registry.

    Falls back to the conventional ``data/synthetic/<generator_id>/<class>`` layout when the
    registry entry does not record an explicit ``samples['filtered_<class>']`` path.
    """
    root = Path(root)
    registry = json.loads((root / "configs/generator_registry.json").read_text())
    entry = next((g for g in registry["generators"] if g["id"] == generator_id), None)
    if entry is None:
        raise ValueError(f"generator {generator_id} is not in the registry")
    registered = (entry.get("samples") or {}).get(f"filtered_{klass}")
    return (root / registered) if registered else (root / "data/synthetic" / generator_id / klass)


def load_selected_pool_records(root: Path, generator_id: str, family: str, *,
                               expected_count: int = EXPECTED_FILTERED_IMAGE_COUNT) -> list[dict]:
    """List the selected generator's canonical FILTERED positive pool as ordered image records.

    Reads the pool directory directly (no manifest, no SHA chain), sorts deterministically by
    relative path, and verifies that it holds exactly ``expected_count`` unique, readable images
    with a supported extension, none of which references a test split. Because the canonical pools
    contain exactly the evaluated set, no separate file list is needed to pin the count.
    """
    root = Path(root)
    project_root = root.resolve()
    pool = selected_pool_dir(root, generator_id, "positive")
    if not pool.is_dir():
        raise ValueError(f"FILTERED positive pool is missing for {generator_id}: {pool}")
    relatives: list[str] = []
    for path in sorted(pool.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(project_root):
            raise ValueError(f"pool file resolves outside the project root: {path}")
        relative = resolved.relative_to(project_root).as_posix()
        if _TEST_PATH_COMPONENTS & set(Path(relative).parts):
            raise ValueError(f"FILTERED pool references a test path: {relative}")
        if not os.access(resolved, os.R_OK):
            raise ValueError(f"FILTERED pool file is unreadable: {relative}")
        relatives.append(relative)
    relatives.sort()
    n = len(relatives)
    if n != expected_count:
        raise ValueError(f"{generator_id} FILTERED positive pool has {n} images, expected {expected_count}")
    if len(set(relatives)) != n:
        raise ValueError(f"{generator_id} FILTERED positive pool has duplicate relative paths")
    pool_parents = {Path(rel).parent.as_posix() for rel in relatives}
    if len(pool_parents) != 1 or not next(iter(pool_parents)).endswith("/positive"):
        raise ValueError(f"{generator_id} FILTERED records are not a single canonical positive pool")
    return [{"path": rel, "source": "synthetic", "generator_id": generator_id,
             "synthetic_family": family, "sample_id": Path(rel).stem} for rel in relatives]


def _sha256_file_cached(path: Path) -> str:
    stat = path.stat()
    # ctime prevents a same-size edit with a deliberately restored mtime from reusing stale
    # bytes during long notebook-generation processes.
    key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _FILE_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[key] = value
    return value


def _real_files_by_class(root: Path) -> dict[str, list[dict]]:
    metadata = root / "data/processed/metadata/train.csv"
    with metadata.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    label_to_class = {v: k for k, v in CLASS_LABEL.items()}
    by_class: dict[str, list[str]] = {k: [] for k in CLASS_LABEL}
    for row in rows:
        try:
            klass = label_to_class.get(int(row["label"]))
        except (TypeError, ValueError):
            klass = None
        if klass is None:
            continue
        processed_path = row.get("processed_path")
        by_class[klass].append({
            "path": processed_path if processed_path else f"{REAL_TRAIN_DIR}/{row['label']}/{row['image_id']}.png",
            "patient_id": row.get("patient_id"), "image_id": row.get("image_id"),
        })
    return by_class


class AugmentationSourceError(RuntimeError):
    """Raised when an augmented file cannot be linked to a real train-split source image."""


def _real_train_patient_ids(root: Path) -> set[str]:
    metadata = root / "data/processed/metadata/train.csv"
    with metadata.open(newline="", encoding="utf-8") as fh:
        return {row["patient_id"] for row in csv.DictReader(fh)}


def _augmented_files_by_class(root: Path) -> dict[str, list[dict]]:
    """Read data/real_augmented/metadata.csv (written by
    notebooks/1_preprocessing/02_Data_Augmentation_Trad.ipynb), which already links every
    augmented file to its real source patient_id/image_id and original_processed_path.

    Never falls back to inferring the source from the filename alone: an augmented file with
    no traceable, train-split source patient is rejected outright rather than silently
    included without a verified train source.
    """
    label_to_class = {v: k for k, v in CLASS_LABEL.items()}
    by_class: dict[str, list[dict]] = {k: [] for k in CLASS_LABEL}
    manifest = root / AUGMENTED_DIR / "metadata.csv"
    if not manifest.is_file():
        return by_class
    train_patients = _real_train_patient_ids(root)
    with manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if row.get("source") == "real":
            continue  # metadata.csv also re-lists the real source rows; only augmented here
        try:
            klass = label_to_class.get(int(row.get("label", "")))
        except (TypeError, ValueError):
            klass = None
        if klass is None:
            continue
        patient_id = row.get("patient_id")
        original_path = row.get("original_processed_path", "")
        if not patient_id or patient_id not in train_patients:
            raise AugmentationSourceError(
                f"augmented file {row.get('file_name')} has no train-split source patient "
                f"(patient_id={patient_id!r}): refusing to include it")
        if "/processed/train/" not in original_path.replace("\\", "/"):
            raise AugmentationSourceError(
                f"augmented file {row.get('file_name')} traces to a non-train source "
                f"({original_path!r}): refusing to include it")
        by_class[klass].append({
            "path": row["file_name"], "patient_id": patient_id, "image_id": row.get("image_id"),
            "augmentation_type": row.get("source"), "source_split": "train",
            "source_original_path": original_path,
        })
    return by_class


def _synthetic_candidate_files(root: Path, generator_entry: dict, klass: str) -> tuple[list[str], str]:
    """Real, on-disk candidate filenames for a generator's class, plus a precision tag.

    Tries the metrics-declared directory first (rerooted for a foreign-mount absolute path,
    covering both the flat `config.synthetic_dir`/`input_signature.filtered_dir` schema and the
    per-class `per_class.<class>.generated_dir` schema), then falls back to the canonical
    post-migration `experiments/diffusers/<gid>/generated_images/final/<class>/` layout.

    That canonical directory can contain more files than the scientifically-evaluated count
    recorded in the metrics JSON (observed for the sd21 family: metrics report a validated
    1361/class subset, the current directory holds 2722/class) — a pre-existing ambiguity this
    function does not silently resolve. Callers must check the returned precision tag.
    """
    gid = generator_entry["id"]
    registered = (generator_entry.get("samples") or {}).get(f"filtered_{klass}")
    if registered:
        scan_dir = root / registered
        if scan_dir.is_dir():
            found = sorted(str(p.relative_to(root)) for p in scan_dir.rglob("*")
                           if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
            if found:
                return found, "approved_generator_registry_filtered_path"
    metrics_rel = generator_entry.get("metrics")
    if metrics_rel:
        path = root / metrics_rel
        if path.is_file():
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                payload = {}
            per_class_dir = ((payload.get("per_class") or {}).get(klass) or {}).get("generated_dir")
            synth_dir = per_class_dir or (payload.get("config") or {}).get("synthetic_dir") or (payload.get("input_signature") or {}).get("filtered_dir")
            if synth_dir:
                raw = Path(synth_dir)
                if raw.is_absolute():
                    marker = next((name for name in ("data", "experiments", "results") if name in raw.parts), None)
                    base = root.joinpath(*raw.parts[raw.parts.index(marker):]) if marker else raw
                else:
                    base = root / raw
                scan_dir = (base.parent / klass) if base.name in CLASS_LABEL else base
                if scan_dir.is_dir():
                    found = sorted(str(p.relative_to(root)) for p in scan_dir.glob("*.png"))
                    if found:
                        return found, "metrics_declared_directory"

    canonical = root / "experiments/diffusers" / gid / "generated_images/final" / klass
    if canonical.is_dir():
        found = sorted(str(p.relative_to(root)) for p in canonical.glob("*.png"))
        if found:
            return found, "canonical_directory_fallback_unverified_against_metrics_count"

    return [], "no_candidates_found"


def build_file_list(root: Path, variant: dict, generator_registry: dict | None = None) -> dict:
    """Return {"negative": [{"path":..., "source":...}, ...], "positive": [...]} for one variant."""
    real = _real_files_by_class(root) if variant.get("real_source") else {k: [] for k in CLASS_LABEL}
    augmented = _augmented_files_by_class(root) if variant.get("augmentation_source") else {k: [] for k in CLASS_LABEL}
    allowed_augmented = set(variant.get("augmentation_classes") or CLASS_LABEL)
    augmented = {klass: values if klass in allowed_augmented else [] for klass, values in augmented.items()}

    synthetic: dict[str, list[str]] = {k: [] for k in CLASS_LABEL}
    synthetic_records: dict[str, list[dict]] = {k: [] for k in CLASS_LABEL}
    gid = variant.get("synthetic_generator_id")
    if gid and variant.get("synthetic_count_by_class"):
        selected_family = selected_family_for_generator(root, gid)
        if selected_family is not None:
            # Publication path: consume the selected generator's canonical FILTERED positive pool.
            records = load_selected_pool_records(root, gid, selected_family)
            for klass, k_count in variant["synthetic_count_by_class"].items():
                if klass != "positive":
                    raise ValueError(f"selected synthetic condition adds positives only, got class {klass}")
                if len(records) != k_count:
                    raise ValueError(f"variant {variant['dataset_variant_id']}: FILTERED pool has "
                                     f"{len(records)} images, synthetic_count_by_class requires {k_count}")
                synthetic[klass] = [record["path"] for record in records]
                synthetic_records[klass] = [dict(record) for record in records]
        else:
            # Older non-publication utilities may still scan and sample an unselected generator.
            if generator_registry is None:
                generator_registry = json.loads((root / "configs/generator_registry.json").read_text())
            entry = next((g for g in generator_registry["generators"] if g["id"] == gid), None)
            if entry is None:
                raise ValueError(f"generator {gid} referenced by variant {variant['dataset_variant_id']} not found in registry")
            for klass, k_count in variant["synthetic_count_by_class"].items():
                candidates, precision = _synthetic_candidate_files(root, entry, klass)
                if len(candidates) < k_count:
                    raise ValueError(f"variant {variant['dataset_variant_id']}: only {len(candidates)} synthetic files on disk for "
                                      f"{gid}/{klass} ({precision}), need {k_count}")
                sig = deterministic_sample_signature(candidates, k_count, seed=variant.get("seed", 42))
                synthetic[klass] = sig["picked"]
                variant.setdefault("_resolved_synthetic_signature", {})[klass] = {
                    **{k: v for k, v in sig.items() if k != "picked"}, "file_source_precision": precision}

    files = {}
    for klass in CLASS_LABEL:
        entries = [{**p, "source": "real"} if isinstance(p, dict) else {"path": p, "source": "real"}
                   for p in real.get(klass, [])]
        entries += [{**p, "source": "augmented"} if isinstance(p, dict) else {"path": p, "source": "augmented"}
                    for p in augmented.get(klass, [])]
        if synthetic_records[klass]:
            entries += synthetic_records[klass]
        else:
            entries += [{"path": p, "source": "synthetic"} for p in synthetic.get(klass, [])]
        files[klass] = entries
    return files


def _content_aware_record(root: Path, *, klass: str, entry: dict) -> dict:
    """Resume-safety record for one input image."""
    path = resolve_project_path(root, entry["path"])
    if not path.is_file():
        raise FileNotFoundError(f"dataset file does not exist: {path}")
    try:
        relative_path = path.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"dataset file is outside project root: {path}") from exc
    return {
        "relative_path": relative_path,
        "file_size": path.stat().st_size,
        "sha256": _sha256_file_cached(path),
        "label": int(CLASS_LABEL[klass]),
        "source": entry.get("source"),
        "patient_id": entry.get("patient_id"),
        "image_id": entry.get("image_id") or path.stem,
        "augmentation_type": entry.get("augmentation_type"),
        "augmentation_source_split": entry.get("source_split"),
        "source_original_path": entry.get("source_original_path"),
    }


def dataset_manifest_signature(root: Path, file_list: dict) -> str:
    records = []
    for klass in sorted(file_list):
        for entry in sorted(file_list[klass], key=lambda value: value["path"]):
            records.append(_content_aware_record(Path(root), klass=klass, entry=entry))
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validation_manifest_signature(root: Path, rows: list[dict]) -> str:
    records = []
    for row in sorted(rows, key=lambda value: (str(value.get("patient_id")), str(value.get("image_id")), value["processed_path"])):
        klass = next(name for name, label in CLASS_LABEL.items() if int(label) == int(row["label"]))
        records.append(_content_aware_record(Path(root), klass=klass, entry={
            "path": row["processed_path"], "source": row.get("source", "real_validation"),
            "patient_id": row.get("patient_id"), "image_id": row.get("image_id"),
        }))
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_project_path(root: Path, value: str) -> Path:
    """Resolve paths stored by older mounts without accepting files outside the project."""
    path = Path(value)
    if not path.is_absolute():
        candidate = root / path
    else:
        try:
            relative = path.relative_to(root)
        except ValueError:
            markers = ("data", "experiments", "results")
            parts = path.parts
            marker = next((name for name in markers if name in parts), None)
            if marker is None:
                raise ValueError(f"path cannot be safely rerooted under project: {value}")
            candidate = root.joinpath(*parts[parts.index(marker):])
        else:
            candidate = root / relative
    return candidate.resolve()


def rows_from_file_list(root: Path, file_list: dict) -> list[dict]:
    """Flatten a training-only file list and fail before a loader sees missing files."""
    rows = []
    seen = set()
    for klass, label in CLASS_LABEL.items():
        for entry in file_list.get(klass, []):
            path = resolve_project_path(root, entry["path"])
            if not path.is_file():
                raise FileNotFoundError(f"dataset file does not exist: {path}")
            key = str(path)
            if key in seen:
                raise ValueError(f"duplicate dataset file: {path}")
            seen.add(key)
            row = {"processed_path": key, "label": int(label), "source": entry["source"],
                   "patient_id": entry.get("patient_id"),
                   "image_id": entry.get("image_id") or path.stem}
            for field in ("synthetic_family", "generator_id", "sample_id"):
                if field in entry:
                    row[field] = entry[field]
            rows.append(row)
    return rows


def validation_rows(root: Path) -> list[dict]:
    """Load the real patient-held-out validation split; synthetic data is forbidden here."""
    metadata = root / VALIDATION_METADATA
    with metadata.open(newline="", encoding="utf-8") as fh:
        source_rows = list(csv.DictReader(fh))
    rows = []
    for row in source_rows:
        raw = row.get("processed_path") or f"data/processed/val/{row['label']}/{row['image_id']}.png"
        path = resolve_project_path(root, raw)
        if not path.is_file():
            raise FileNotFoundError(f"validation file does not exist: {path}")
        rows.append({
            "processed_path": str(path), "label": int(row["label"]), "source": "real_validation",
            "patient_id": row.get("patient_id"), "image_id": row.get("image_id"),
        })
    assert_no_synthetic_evaluation(rows, split="validation")
    return rows


def test_rows(root: Path) -> list[dict]:
    """Load the real patient-held-out test split; synthetic data is forbidden here."""
    metadata = root / TEST_METADATA
    with metadata.open(newline="", encoding="utf-8") as fh:
        source_rows = list(csv.DictReader(fh))
    rows = []
    for row in source_rows:
        raw = row.get("processed_path") or f"data/processed/test/{row['label']}/{row['image_id']}.png"
        path = resolve_project_path(root, raw)
        if not path.is_file():
            raise FileNotFoundError(f"test file does not exist: {path}")
        rows.append({
            "processed_path": str(path), "label": int(row["label"]), "source": "real_test",
            "patient_id": row.get("patient_id"), "image_id": row.get("image_id"),
        })
    assert_no_synthetic_evaluation(rows, split="test")
    return rows


def assert_no_synthetic_evaluation(rows: list[dict], split: str) -> None:
    if split not in ("validation", "test", "locked-test"):
        return
    offenders = [row for row in rows if "synthetic" in str(row.get("source", "")).lower()]
    if offenders:
        raise RuntimeError(f"synthetic samples are forbidden in {split}: {len(offenders)} found")


def build_training_and_validation_rows(root: Path, variant: dict) -> tuple[list[dict], list[dict], dict]:
    """Canonical dataset entrypoint shared by notebooks and the CLI runner."""
    if variant.get("status") not in ("ready", "legacy"):
        raise RuntimeError(
            f"dataset {variant.get('dataset_variant_id')} is {variant.get('status')}: "
            f"{variant.get('blocker') or variant.get('invalid_reason') or 'not executable'}"
        )
    file_list = build_file_list(root, variant)
    train_rows = rows_from_file_list(root, file_list)
    val_rows = validation_rows(root)
    train_patients = {str(row.get("patient_id")) for row in train_rows if row.get("patient_id")}
    val_patients = {str(row.get("patient_id")) for row in val_rows if row.get("patient_id")}
    overlap = train_patients & val_patients
    if overlap:
        raise RuntimeError(f"patient leakage between train and validation: {sorted(overlap)[:10]}")
    # These content hashes exist only to detect a changed dataset when resuming a checkpoint;
    # they are never written into configs or read by any downstream scientific gate.
    training_signature = dataset_manifest_signature(root, file_list)
    validation_signature = validation_manifest_signature(root, val_rows)
    combined_signature = hashlib.sha256(json.dumps({
        "training_signature": training_signature, "validation_signature": validation_signature,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    dataset_metadata = {
        "schema_version": 3,
        "artifact_type": "classifier_dataset_manifest",
        "dataset_variant_id": variant["dataset_variant_id"],
        "counts": {klass: len(entries) for klass, entries in file_list.items()},
        "signature": combined_signature,
        "training_signature": training_signature,
        "validation_signature": validation_signature,
        "files": file_list,
        "train_patient_ids": sorted(train_patients),
        "validation_patient_ids": sorted(val_patients),
        "patient_overlap": [],
    }
    return train_rows, val_rows, dataset_metadata
