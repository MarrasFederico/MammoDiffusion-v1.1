"""Metadata-only preflight audit of the four classifier conditions.

Resolves each condition and inspects the selected generator's canonical FILTERED positive pool and
the real metadata *without* loading any model, training anything, running inference, or reading the
test split.  It reports counts and integrity checks so the notebook-first pipeline can confirm the
dataset construction is coherent before any classifier is trained.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import classifier_protocol as dp  # noqa: E402

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


def _dataset_builder():
    try:
        from . import classifier_dataset_builder as builder
    except ImportError:
        import classifier_dataset_builder as builder
    return builder


def synthetic_pool_audit(root: Path, generator_id: str) -> dict[str, Any]:
    """Count and integrity of one selected generator's canonical FILTERED positive pool.

    Lists the pool directory recorded in the registry (no manifest, no SHA), and reports the image
    count, any test-split path, duplicate relative paths, and unreadable files.
    """
    root = Path(root)
    project_root = root.resolve()
    builder = _dataset_builder()
    pool = builder.selected_pool_dir(root, generator_id, "positive")
    files = sorted(p for p in pool.iterdir()
                   if p.is_file() and p.suffix.lower() in builder.SUPPORTED_IMAGE_EXTENSIONS) if pool.is_dir() else []
    relatives, unreadable = [], 0
    for path in files:
        resolved = path.resolve()
        relative = resolved.relative_to(project_root).as_posix() if resolved.is_relative_to(project_root) else str(resolved)
        relatives.append(relative)
        if not os.access(resolved, os.R_OK):
            unreadable += 1
    test_paths = [rel for rel in relatives if _is_test_path(rel)]
    return {
        "generator_id": generator_id,
        "filtered_pool_dir": pool.relative_to(root).as_posix() if pool.is_relative_to(root) else str(pool),
        "synthetic_sample_count": len(relatives),
        "class_counts": {"positive": len(relatives), "negative": 0},
        "test_paths_found": len(test_paths),
        "duplicate_relative_paths": len(relatives) - len(set(relatives)),
        "unreadable_files": unreadable,
    }


def built_pool_audit(root: Path, condition: str) -> dict[str, Any]:
    """Build the synthetic file list (no model) and confirm it is exactly the canonical pool."""
    root = Path(root)
    builder = _dataset_builder()
    variant = dp.resolve_condition(root, condition)
    generator_id = variant["synthetic_generator_id"]
    family = builder.selected_family_for_generator(root, generator_id)
    records = builder.load_selected_pool_records(root, generator_id, family)
    file_list = builder.build_file_list(root, variant)
    built = [entry for entry in file_list.get("positive", []) if entry.get("source") == "synthetic"]
    built_paths = [entry["path"] for entry in built]
    pool_paths = {record["path"] for record in records}
    test_paths = sum(1 for path in built_paths if builder._TEST_PATH_COMPONENTS & set(Path(path).parts))
    return {
        "selected_generator": generator_id,
        "filtered_pool_records": len(records),
        "built_synthetic_count": len(built),
        "exact_path_set_match": set(built_paths) == pool_paths,
        "test_paths": test_paths,
        "duplicate_relative_paths": len(built_paths) - len(set(built_paths)),
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
        report["synthetic"] = synthetic_pool_audit(root, resolved["synthetic_generator_id"])
        report["built"] = built_pool_audit(root, condition)
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
            print(f"    pool={syn['filtered_pool_dir']}")
            print(f"    samples={syn['synthetic_sample_count']} classes={syn['class_counts']}")
            print(f"    test_paths={syn['test_paths_found']} dup_paths={syn['duplicate_relative_paths']} "
                  f"unreadable={syn['unreadable_files']}")
        if "built" in entry:
            b = entry["built"]
            print(f"    built: pool_records={b['filtered_pool_records']} "
                  f"built_synthetic={b['built_synthetic_count']}")
            print(f"    built: exact_path_set_match={b['exact_path_set_match']} "
                  f"test_paths={b['test_paths']} dup_paths={b['duplicate_relative_paths']}")


if __name__ == "__main__":
    main()
