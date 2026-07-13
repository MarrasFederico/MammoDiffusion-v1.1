"""Resolves a dataset_variant registry entry into an actual, signed, deterministic file list
for classifier training (spec section 8). Real/augmented files are enumerated directly;
synthetic files are drawn with dataset_variant_registry.deterministic_sample_signature so the
same variant always yields the same picks regardless of filesystem enumeration order.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_variant_registry import CLASS_LABEL, deterministic_sample_signature  # noqa: E402

REAL_TRAIN_DIR = "data/processed/train"
AUGMENTED_DIR = "data/real_augmented"
VALIDATION_METADATA = "data/processed/metadata/val.csv"
_FILE_HASH_CACHE: dict[tuple[str, int, int, int], str] = {}


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
        klass = label_to_class.get(row["label"])
        if klass is None:
            continue
        processed_path = row.get("processed_path")
        by_class[klass].append({
            "path": processed_path if processed_path else f"{REAL_TRAIN_DIR}/{row['label']}/{row['image_id']}.png",
            "patient_id": row.get("patient_id"), "image_id": row.get("image_id"),
        })
    return by_class


class AugmentationProvenanceError(RuntimeError):
    """Raised when an augmented file cannot be traced to a real train-split source image."""


def _real_train_patient_ids(root: Path) -> set[str]:
    metadata = root / "data/processed/metadata/train.csv"
    with metadata.open(newline="", encoding="utf-8") as fh:
        return {row["patient_id"] for row in csv.DictReader(fh)}


def _augmented_files_by_class(root: Path) -> dict[str, list[dict]]:
    """Read data/real_augmented/metadata.csv (written by
    notebooks/1_preprocessing/02_Data_Augmentation_Trad.ipynb), which already links every
    augmented file to its real source patient_id/image_id and original_processed_path.

    Never falls back to inferring provenance from the filename alone: an augmented file with
    no traceable, train-split source patient is rejected outright rather than silently
    included as an unprovenanced image (spec: "il runtime deve rifiutare augmented prive di
    provenance o provenienti da validation/test").
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
        klass = label_to_class.get(row.get("label"))
        if klass is None:
            continue
        patient_id = row.get("patient_id")
        original_path = row.get("original_processed_path", "")
        if not patient_id or patient_id not in train_patients:
            raise AugmentationProvenanceError(
                f"augmented file {row.get('file_name')} has no train-split source patient "
                f"(patient_id={patient_id!r}): refusing to include it")
        if "/processed/train/" not in original_path.replace("\\", "/"):
            raise AugmentationProvenanceError(
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
    from dataset_variant_registry import _reroot_under_project  # noqa: PLC0415

    gid = generator_entry["id"]
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
                base = _reroot_under_project(synth_dir, root)
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

    synthetic: dict[str, list[str]] = {k: [] for k in CLASS_LABEL}
    gid = variant.get("synthetic_generator_id")
    if gid and variant.get("synthetic_count_by_class"):
        if generator_registry is None:
            from dataset_variant_registry import load_generator_registry  # noqa: PLC0415
            generator_registry = load_generator_registry(root)
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
        entries += [{"path": p, "source": "synthetic"} for p in synthetic.get(klass, [])]
        files[klass] = entries
    return files


def _content_aware_record(root: Path, *, klass: str, entry: dict) -> dict:
    """Canonical scientific identity for one image and its provenance."""
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
            rows.append({"processed_path": key, "label": int(label), "source": entry["source"],
                         "patient_id": entry.get("patient_id"),
                         "image_id": entry.get("image_id") or path.stem})
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
    training_signature = dataset_manifest_signature(root, file_list)
    validation_signature = validation_manifest_signature(root, val_rows)
    combined_signature = hashlib.sha256(json.dumps({
        "training_signature": training_signature, "validation_signature": validation_signature,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    manifest_payload = {
        "schema_version": 3,
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
    return train_rows, val_rows, manifest_payload


def write_dataset_manifest(root: Path, variant: dict, file_list: dict, out_path: Path) -> dict:
    payload = {
        "schema_version": 1,
        "dataset_variant_id": variant["dataset_variant_id"],
        "counts": {k: len(v) for k, v in file_list.items()},
        "signature": dataset_manifest_signature(root, file_list),
        "files": file_list,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    return payload
