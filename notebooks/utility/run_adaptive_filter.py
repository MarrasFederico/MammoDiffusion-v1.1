from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import re
from datetime import datetime
from pathlib import Path


FILTER_SUMMARY_SCHEMA_VERSION = 2
FILTER_SCHEMA_VERSION = "adaptive_mask_v2"
from parallel_generation_utils import GENERATED_PNG_PATTERN
_CANDIDATE_NAME = re.compile(GENERATED_PNG_PATTERN, re.IGNORECASE)


def count_pngs(directory: Path) -> int:
    """Count only readable, non-temporary PNG files."""
    return len(candidate_png_paths(directory))


def candidate_png_paths(directory: Path) -> list[Path]:
    """Return the filter's effective dataset: readable, non-temporary images."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    from PIL import Image
    result = []
    for path in sorted(directory.glob("*.png")):
        if path.name.startswith(".tmp_"):
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            result.append(path)
        except Exception:
            continue
    return (
        [path for path in result if _CANDIDATE_NAME.fullmatch(path.name)]
        if any(_CANDIDATE_NAME.fullmatch(path.name) for path in result)
        else result
    )


def file_sha256(path: Path) -> str:
    """Hash a file with SHA-256 to detect exact duplicates."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duplicate_png_groups(directory: Path) -> dict[str, list[str]]:
    """Group PNGs by content hash and return only groups with multiple files."""
    by_hash: dict[str, list[str]] = {}
    for path in candidate_png_paths(directory):
        by_hash.setdefault(file_sha256(path), []).append(path.name)
    return {digest: names for digest, names in by_hash.items() if len(names) > 1}


def real_train_paths(
    train_metadata: Path,
    data_processed_dir: Path,
    label: int,
    seed: int,
) -> list[Path]:
    """Return a reproducibly shuffled real-training reference set for one label."""
    import pandas as pd

    from ldm_project_paths import normalize_processed_path

    metadata = pd.read_csv(train_metadata)
    required = {"label", "processed_path"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing columns in {train_metadata}: {sorted(missing)}")

    subset = metadata[metadata["label"].astype(int) == int(label)]
    if subset.empty:
        raise ValueError(f"No images with label={label} in {train_metadata}")
    subset = subset.sample(n=len(subset), random_state=seed)

    paths: list[Path] = []
    for _, row in subset.iterrows():
        if "split" in metadata.columns:
            candidate = normalize_processed_path(row, data_processed_dir)
        else:
            raw_path = Path(str(row["processed_path"]).replace("\\", "/"))
            candidate = data_processed_dir / train_metadata.stem.lower() / str(int(label)) / raw_path.name
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        paths.append(candidate.resolve())
    return paths


def selected_output_name(class_name: str, rank: int) -> str:
    """Name an accepted image from its class and ranking position."""
    # Preserve the historical convention established by notebook 03b.
    prefix = "selected_pos" if class_name == "positive" else f"selected_{class_name}"
    return f"{prefix}_{rank:04d}.png"


def filter_ranking_config(args: argparse.Namespace) -> dict[str, object]:
    """Collect ranking parameters used to validate a saved summary against the current run."""
    return {
        "schema_version": FILTER_SUMMARY_SCHEMA_VERSION,
        "filter_schema_version": FILTER_SCHEMA_VERSION,
        "selection_score": "robust_distance_from_train_reference",
        "sort_order": "descending_selection_score",
        "n_selected": int(args.n_selected),
        "eval_seed": int(args.eval_seed),
        "reference_seed": int(args.eval_seed) + int(args.label),
        "nonblack_threshold": int(args.nonblack_threshold),
        "output_naming": "selected_output_name_v1",
    }


def paths_match(left: object, right: Path) -> bool:
    """Compare resolved paths without relative-path or trailing-slash ambiguity."""
    if left is None:
        return False
    left_path = Path(str(left)).expanduser()
    right_path = Path(right).expanduser()
    try:
        return left_path.resolve() == right_path.resolve()
    except OSError:
        return left_path == right_path


def summary_is_complete_and_compatible(args: argparse.Namespace) -> bool:
    """Check whether a saved summary still matches this run's schema, parameters, paths, and counts."""
    summary_path = Path(args.summary_json)
    raw_dir = Path(args.raw_dir)
    filtered_dir = Path(args.filtered_dir)
    n_selected = int(args.n_selected)

    if not summary_path.is_file():
        return False
    try:
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False

    required_non_null = [
        "schema_version",
        "filter_schema_version",
        "status",
        "experiment_name",
        "class_name",
        "label",
        "raw_dir",
        "filtered_dir",
        "n_raw",
        "n_selected",
        "thresholds",
        "reference_medians",
        "reference_iqrs",
        "n_real_train_reference",
        "reject_reason_counts",
        "ranking_config",
    ]
    if any(summary.get(key) is None for key in required_non_null):
        return False
    if summary.get("status") != "completed":
        return False
    if summary.get("schema_version") != FILTER_SUMMARY_SCHEMA_VERSION:
        return False
    if summary.get("filter_schema_version") != FILTER_SCHEMA_VERSION:
        return False
    if summary.get("experiment_name") != args.experiment_name:
        return False
    if summary.get("class_name") != args.class_name:
        return False
    if int(summary.get("label")) != int(args.label):
        return False
    if int(summary.get("n_raw")) != count_pngs(raw_dir):
        return False
    if int(summary.get("n_selected")) != n_selected:
        return False
    if not paths_match(summary.get("raw_dir"), raw_dir):
        return False
    if not paths_match(summary.get("filtered_dir"), filtered_dir):
        return False
    if summary.get("ranking_config") != filter_ranking_config(args):
        return False
    if count_pngs(filtered_dir) != n_selected:
        return False
    if duplicate_png_groups(filtered_dir):
        return False
    return True


def run_filter(args: argparse.Namespace) -> None:
    """Run the complete adaptive filter for one class.

    Score each RAW image against reference statistics, select the best
    ``n_selected`` candidates, copy accepted files to ``filtered_dir``, and
    write the CSV report and JSON summary. Reuse a complete compatible result.
    """
    raw_dir = Path(args.raw_dir)
    filtered_dir = Path(args.filtered_dir)
    report_csv = Path(args.report_csv)
    summary_json = Path(args.summary_json)
    n_selected = int(args.n_selected)

    if count_pngs(filtered_dir) == n_selected and summary_is_complete_and_compatible(args):
        print(
            f"{args.class_name}: {n_selected} filtered images already exist "
            "with a compatible summary; skipping"
        )
        return
    if count_pngs(filtered_dir) == n_selected:
        print(
            f"{args.class_name}: filtered directory is complete but the summary is missing "
            "or incompatible; rerunning the filter."
        )

    raw_paths = candidate_png_paths(raw_dir)
    if not raw_paths:
        raise RuntimeError(f"No RAW images in {raw_dir}")

    import pandas as pd

    from adaptive_mammography_filter import (
        compute_reference_statistics,
        evaluate_candidate,
        image_quality_metrics,
        estimate_foreground_mask,
    )

    _ = image_quality_metrics, estimate_foreground_mask
    reference_paths = real_train_paths(
        train_metadata=Path(args.train_metadata),
        data_processed_dir=Path(args.data_processed_dir),
        label=int(args.label),
        seed=int(args.eval_seed) + int(args.label),
    )
    reference = compute_reference_statistics(
        reference_paths,
        nonblack_threshold=int(args.nonblack_threshold),
        verbose=True,
    )

    candidate_rows = []
    for index, path in enumerate(raw_paths, start=1):
        candidate = evaluate_candidate(
            path,
            reference,
            nonblack_threshold=int(args.nonblack_threshold),
        )
        candidate_rows.append({
            "class": args.class_name,
            "source_path": candidate["raw_path"],
            "source_name": candidate["raw_filename"],
            **{
                key: value
                for key, value in candidate.items()
                if key not in {"raw_path", "raw_filename", "score", "reason"}
            },
            "reject_reason": candidate["reason"],
            "selection_score": candidate["score"],
        })
        if index % 250 == 0 or index == len(raw_paths):
            print(f"  Analyzing {args.class_name}: {index}/{len(raw_paths)}")

    candidates = pd.DataFrame(candidate_rows)
    accepted = (
        candidates.loc[~candidates["rejected"]]
        .sort_values("selection_score", ascending=False)
        .copy()
    )
    reject_counts = {
        str(reason): int(count)
        for reason, count in candidates.loc[
            candidates["rejected"], "reject_reason"
        ].value_counts().items()
    }

    report_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    if len(accepted) < n_selected:
        candidates.to_csv(report_csv, index=False)
        raise RuntimeError(
            f"After filtering, {args.class_name} has {len(accepted)} images; "
            f"{n_selected} are required."
        )

    selected = accepted.head(n_selected).copy()
    selected["selection_rank"] = range(n_selected)
    rank_by_path = dict(zip(selected["source_path"], selected["selection_rank"]))
    candidates["selected"] = candidates["source_path"].isin(rank_by_path)
    candidates["selection_rank"] = candidates["source_path"].map(rank_by_path)

    filtered_dir.mkdir(parents=True, exist_ok=True)
    for path in filtered_dir.glob("*.png"):
        path.unlink()
    for rank, source_path in enumerate(selected["source_path"]):
        shutil.copy2(source_path, filtered_dir / selected_output_name(args.class_name, rank))

    if count_pngs(filtered_dir) != n_selected:
        raise RuntimeError(f"Invalid filtered-image count for {args.class_name}.")
    duplicates = duplicate_png_groups(filtered_dir)
    if duplicates:
        raise RuntimeError(f"The filtered {args.class_name} dataset contains duplicates.")

    candidates.to_csv(report_csv, index=False)
    summary = {
        "schema_version": FILTER_SUMMARY_SCHEMA_VERSION,
        "filter_schema_version": FILTER_SCHEMA_VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": args.experiment_name,
        "class_name": args.class_name,
        "experiment": args.experiment_name,
        "class": args.class_name,
        "label": int(args.label),
        "raw_dir": str(raw_dir),
        "filtered_dir": str(filtered_dir),
        "n_raw": len(candidates),
        "n_accepted": len(accepted),
        "n_rejected": int(candidates["rejected"].sum()),
        "n_selected": n_selected,
        "thresholds": reference["thresholds"],
        "reference_medians": reference["medians"],
        "reference_iqrs": reference["iqrs"],
        "n_real_train_reference": reference["n_train"],
        "reject_reason_counts": reject_counts,
        "ranking_config": filter_ranking_config(args),
        "status": "completed",
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"{args.class_name}: selected {n_selected} images")
    print("Report:", report_csv)
    print("Summary:", summary_json)


def parse_args() -> argparse.Namespace:
    """Define and parse the filter script's command-line arguments."""
    parser = argparse.ArgumentParser(description="Adaptive mammography filter for notebook 03b.")
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--label", required=True, type=int)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--filtered-dir", required=True, type=Path)
    parser.add_argument("--n-selected", required=True, type=int)
    parser.add_argument("--train-metadata", required=True, type=Path)
    parser.add_argument("--data-processed-dir", required=True, type=Path)
    parser.add_argument("--report-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--eval-seed", required=True, type=int)
    parser.add_argument("--nonblack-threshold", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    """Run the filter and print failures to stderr before propagating them."""
    try:
        run_filter(parse_args())
    except Exception as exc:
        print(f"Adaptive filter error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
