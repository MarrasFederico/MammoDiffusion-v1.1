"""Validation helpers for safely reusing the canonical processed cohort.

The preprocessing notebook normally derives this cohort from the raw RSNA
metadata and images.  A published project, however, may be distributed with
the canonical processed artifacts but without the restricted or bulky source
archive.  This module verifies that the materialized cohort is internally
complete before the notebook is allowed to enter that read-only reuse mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_SPLITS = ("train", "val", "test")
EXPECTED_LABELS = (0, 1)
REQUIRED_COLUMNS = (
    "patient_id",
    "image_id",
    "laterality",
    "view",
    "label",
    "cancer",
    "patient_label",
    "split",
    "source",
    "processed_path",
    "visual_side_before",
    "visual_side_after",
    "left_ratio_before",
    "right_ratio_before",
    "flipped_by_visual_rule",
    "normalized_tissue_side",
    "normalized_laterality",
)


def _resolve_project_path(project_root: Path, value: object) -> Path:
    """Resolve a manifest path without making the project location implicit."""

    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def audit_processed_dataset(
    project_root: str | Path,
    processed_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Audit whether a processed cohort is safe to reuse.

    Readiness requires a non-empty canonical manifest, the expected binary
    train/validation/test partitions, one row per patient and output path,
    reconciliation of every split manifest with the canonical manifest, and
    an existing image for every recorded ``processed_path``.  Additional files
    are tolerated because diagnostic exports may legitimately coexist with the
    canonical cohort.

    The function is deliberately read-only.  It returns diagnostic reasons and
    summary counts so callers can report why reuse was accepted or rejected.
    """

    root = Path(project_root).expanduser().resolve()
    output_dir = (
        Path(processed_dir).expanduser().resolve()
        if processed_dir is not None
        else (root / "data" / "processed").resolve()
    )
    metadata_dir = output_dir / "metadata"
    manifest_path = metadata_dir / "all_processed.csv"
    split_paths = {split: metadata_dir / f"{split}.csv" for split in EXPECTED_SPLITS}
    reasons: list[str] = []

    required_manifests = {"all": manifest_path, **split_paths}
    missing_manifests = [
        str(path) for path in required_manifests.values() if not path.is_file()
    ]
    if missing_manifests:
        reasons.append("missing manifest files: " + ", ".join(missing_manifests))
        return {
            "ready": False,
            "reasons": reasons,
            "project_root": str(root),
            "processed_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "row_count": 0,
            "split_counts": {},
            "label_counts": {},
            "missing_image_count": 0,
        }

    try:
        manifest = pd.read_csv(manifest_path)
    except Exception as exc:
        reasons.append(f"canonical manifest is unreadable: {exc}")
        manifest = pd.DataFrame()

    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(manifest.columns))
    if missing_columns:
        reasons.append("canonical manifest lacks columns: " + ", ".join(missing_columns))
    if manifest.empty:
        reasons.append("canonical manifest contains no rows")

    if reasons:
        return {
            "ready": False,
            "reasons": reasons,
            "project_root": str(root),
            "processed_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "row_count": int(len(manifest)),
            "split_counts": {},
            "label_counts": {},
            "missing_image_count": 0,
        }

    checked = manifest.copy()
    for column in REQUIRED_COLUMNS:
        if checked[column].isna().any():
            reasons.append(f"canonical manifest contains null values in {column}")

    checked["patient_id"] = checked["patient_id"].astype(str)
    checked["image_id"] = checked["image_id"].astype(str)
    checked["split"] = checked["split"].astype(str)
    try:
        checked["label"] = pd.to_numeric(checked["label"], errors="raise").astype(int)
    except Exception as exc:
        reasons.append(f"label values are not integral: {exc}")

    observed_splits = set(checked["split"].unique())
    if observed_splits != set(EXPECTED_SPLITS):
        reasons.append(
            "split values do not match the canonical partition set: "
            f"observed={sorted(observed_splits)}"
        )
    if "label" in checked and pd.api.types.is_integer_dtype(checked["label"]):
        observed_labels = set(int(value) for value in checked["label"].unique())
        if observed_labels != set(EXPECTED_LABELS):
            reasons.append(
                "label values do not match the expected binary classes: "
                f"observed={sorted(observed_labels)}"
            )
        for split in EXPECTED_SPLITS:
            for label in EXPECTED_LABELS:
                if not ((checked["split"] == split) & (checked["label"] == label)).any():
                    reasons.append(f"partition {split!r} contains no class-{label} samples")

    if checked["patient_id"].duplicated().any():
        reasons.append("patient_id is not unique across the canonical cohort")
    if checked["processed_path"].astype(str).duplicated().any():
        reasons.append("processed_path is not unique across the canonical cohort")

    root_prefix = str(root) + "/"
    resolved_paths: list[Path] = []
    outside_project_count = 0
    for value in checked["processed_path"]:
        resolved = _resolve_project_path(root, value)
        resolved_paths.append(resolved)
        if resolved != root and not str(resolved).startswith(root_prefix):
            outside_project_count += 1
    if outside_project_count:
        reasons.append(
            f"{outside_project_count} processed paths resolve outside the project root"
        )

    missing_images = [path for path in resolved_paths if not path.is_file()]
    if missing_images:
        preview = ", ".join(str(path) for path in missing_images[:3])
        reasons.append(
            f"{len(missing_images)} processed images are missing"
            + (f" (first entries: {preview})" if preview else "")
        )

    canonical_by_split = {
        split: set(
            checked.loc[checked["split"] == split, "processed_path"].astype(str)
        )
        for split in EXPECTED_SPLITS
    }
    for split, split_path in split_paths.items():
        try:
            split_manifest = pd.read_csv(split_path)
        except Exception as exc:
            reasons.append(f"{split} manifest is unreadable: {exc}")
            continue
        split_missing_columns = sorted(
            {"split", "processed_path"}.difference(split_manifest.columns)
        )
        if split_missing_columns:
            reasons.append(
                f"{split} manifest lacks columns: {', '.join(split_missing_columns)}"
            )
            continue
        if set(split_manifest["split"].astype(str).unique()) != {split}:
            reasons.append(f"{split} manifest contains rows assigned to another split")
        recorded_paths = set(split_manifest["processed_path"].astype(str))
        if recorded_paths != canonical_by_split[split]:
            reasons.append(
                f"{split} manifest does not reconcile with all_processed.csv"
            )

    split_counts = {
        split: int((checked["split"] == split).sum()) for split in EXPECTED_SPLITS
    }
    label_counts = {
        str(label): int((checked["label"] == label).sum())
        for label in EXPECTED_LABELS
    }
    return {
        "ready": not reasons,
        "reasons": reasons,
        "project_root": str(root),
        "processed_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "row_count": int(len(checked)),
        "split_counts": split_counts,
        "label_counts": label_counts,
        "missing_image_count": int(len(missing_images)),
    }


__all__ = [
    "EXPECTED_LABELS",
    "EXPECTED_SPLITS",
    "REQUIRED_COLUMNS",
    "audit_processed_dataset",
]
