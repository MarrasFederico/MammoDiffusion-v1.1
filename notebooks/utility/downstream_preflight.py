"""Metadata-only preflight audit of the four downstream conditions.

Resolves each condition and inspects the synthetic manifests and real metadata *without* loading any
model, training anything, running inference, or reading the test split.  It reports counts and
integrity checks so the notebook-first pipeline can confirm the dataset construction is coherent
before any classifier is trained.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import downstream_protocol as dp  # noqa: E402
import generator_benchmark as gb  # noqa: E402

CONDITIONS = dp.CONDITIONS
TRAIN_METADATA = "data/processed/metadata/train.csv"
VAL_METADATA = "data/processed/metadata/val.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _is_test_path(value: str) -> bool:
    parts = Path(str(value)).parts
    return "test" in parts or "historical_internal_test" in parts


def real_patient_audit(root: Path) -> dict[str, Any]:
    """Patient counts and the train/validation patient-disjointness check (metadata only)."""
    root = Path(root)
    train = _read_csv(root / TRAIN_METADATA)
    validation = _read_csv(root / VAL_METADATA)
    train_patients = {row["patient_id"] for row in train}
    val_patients = {row["patient_id"] for row in validation}
    overlap = sorted(train_patients & val_patients)
    return {
        "train_images": len(train),
        "validation_images": len(validation),
        "train_patients": len(train_patients),
        "validation_patients": len(val_patients),
        "train_validation_patient_overlap": len(overlap),
        "overlapping_patients": overlap[:20],
        "train_positive_images": sum(1 for row in train if str(row.get("label")) == "1"),
        "validation_positive_images": sum(1 for row in validation if str(row.get("label")) == "1"),
    }


def synthetic_manifest_audit(root: Path, generator_id: str) -> dict[str, Any]:
    """Count, class/source accounting and integrity of one selected generator's FILTERED manifest."""
    root = Path(root)
    registry = gb.load_registry(root)
    entry = next(item for item in registry["generators"] if item["id"] == generator_id)
    import json
    provenance = json.loads((root / entry["provenance_manifest"]).read_text())
    manifest_relative = str(provenance["filtered_sample_manifest"])
    rows = _read_csv(root / manifest_relative)
    sample_ids = [row["sample_id"] for row in rows]
    relative_paths = [row["relative_path"] for row in rows]
    positive = sum(1 for path in relative_paths if "/positive/" in path)
    negative = sum(1 for path in relative_paths if "/negative/" in path)
    sources = {row.get("source_raw_sample_id") for row in rows}
    test_paths = [path for path in relative_paths if _is_test_path(path)]
    missing = [path for path in relative_paths if not (root / path).is_file()]
    return {
        "generator_id": generator_id,
        "synthetic_manifest": manifest_relative,
        "synthetic_sample_count": len(rows),
        "class_counts": {"positive": positive, "negative": negative},
        "distinct_source_count": len(sources),
        "test_paths_found": len(test_paths),
        "duplicate_sample_ids": len(sample_ids) - len(set(sample_ids)),
        "missing_files": len(missing),
        "missing_examples": missing[:5],
    }


def _dataset_builder():
    try:
        from . import classifier_dataset_builder as builder
    except ImportError:
        import classifier_dataset_builder as builder
    return builder


def built_vs_manifest_audit(root: Path, condition: str) -> dict[str, Any]:
    """Build the real synthetic file list (no model) and compare it to the signed manifest by content."""
    root = Path(root)
    builder = _dataset_builder()
    variant = dp.resolve_condition(root, condition)
    generator_id = variant["synthetic_generator_id"]
    family = builder.selected_family_for_generator(root, generator_id)
    payload = dp.load_selected_generators(root)
    records = builder.load_selected_filtered_records(root, payload, family, verify_file_content=True)
    file_list = builder.build_file_list(root, variant)
    audit = builder.audit_built_synthetic_set(root, file_list, records)
    # Directory extras that the signed-manifest path correctly ignores (informational only).
    pool = Path(records[0]["relative_path"]).parent
    on_disk = {p.relative_to(root).as_posix() for p in (root / pool).glob("*")
               if p.is_file() and p.suffix.lower() in builder.SUPPORTED_IMAGE_EXTENSIONS}
    manifest_paths = {record["relative_path"] for record in records}
    modified = sum(1 for record in records
                   if (root / record["relative_path"]).is_file()
                   and (root / record["relative_path"]).stat().st_size != record["file_size"])
    return {
        "selected_generator": generator_id,
        "selected_manifest": str(payload["selection_identity"][family]["filtered_manifest_path"]),
        "manifest_sha256": payload["selection_identity"][family]["filtered_manifest_sha256"],
        "manifest_record_count": len(records),
        "built_synthetic_count": audit["actual_count"],
        "directory_file_count": len(on_disk),
        "directory_extras_ignored": len(on_disk - manifest_paths),
        "exact_path_set_match": audit["exact_path_set_match"],
        "exact_sha256_set_match": audit["exact_sha256_set_match"],
        "test_paths": audit["test_paths"],
        "missing_files": audit["missing_files"],
        "modified_files": modified,
        "duplicate_sample_ids": audit["duplicate_sample_ids"],
        "duplicate_relative_paths": audit["duplicate_paths"],
    }


def audit_condition(root: Path, condition: str) -> dict[str, Any]:
    resolved = dp.resolve_condition(root, condition)
    report = {
        "condition": condition,
        "resolved_status": resolved["status"],
        "real_source": resolved["real_source"],
        "augmentation_source": resolved["augmentation_source"],
        "augmentation_classes": resolved["augmentation_classes"],
        "synthetic_generator_id": resolved["synthetic_generator_id"],
        "synthetic_count_by_class": resolved["synthetic_count_by_class"],
    }
    if resolved["synthetic_generator_id"]:
        report["synthetic"] = synthetic_manifest_audit(root, resolved["synthetic_generator_id"])
        report["built"] = built_vs_manifest_audit(root, condition)
    return report


def audit_all_conditions(root: Path) -> dict[str, Any]:
    return {
        "real_patient_audit": real_patient_audit(root),
        "conditions": [audit_condition(root, condition) for condition in CONDITIONS],
    }


def main() -> None:
    root = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / "configs").is_dir())
    report = audit_all_conditions(root)
    real = report["real_patient_audit"]
    print("real_patient_audit:")
    print(f"  train images={real['train_images']} patients={real['train_patients']} "
          f"pos={real['train_positive_images']}")
    print(f"  validation images={real['validation_images']} patients={real['validation_patients']} "
          f"pos={real['validation_positive_images']}")
    print(f"  train/validation patient overlap={real['train_validation_patient_overlap']}")
    for entry in report["conditions"]:
        print(f"{entry['condition']}: generator={entry['synthetic_generator_id']} "
              f"count={entry['synthetic_count_by_class']}")
        if "synthetic" in entry:
            syn = entry["synthetic"]
            print(f"    manifest={syn['synthetic_manifest']}")
            print(f"    samples={syn['synthetic_sample_count']} classes={syn['class_counts']} "
                  f"sources={syn['distinct_source_count']}")
            print(f"    test_paths={syn['test_paths_found']} dup_ids={syn['duplicate_sample_ids']} "
                  f"missing_files={syn['missing_files']}")
        if "built" in entry:
            b = entry["built"]
            print(f"    built: manifest_records={b['manifest_record_count']} "
                  f"built_synthetic={b['built_synthetic_count']} "
                  f"directory_files={b['directory_file_count']} extras_ignored={b['directory_extras_ignored']}")
            print(f"    built: exact_path_set_match={b['exact_path_set_match']} "
                  f"exact_sha256_set_match={b['exact_sha256_set_match']} test_paths={b['test_paths']} "
                  f"missing={b['missing_files']} modified={b['modified_files']} "
                  f"dup_ids={b['duplicate_sample_ids']} dup_paths={b['duplicate_relative_paths']}")


if __name__ == "__main__":
    main()
