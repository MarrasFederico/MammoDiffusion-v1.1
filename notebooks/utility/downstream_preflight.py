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


if __name__ == "__main__":
    main()
