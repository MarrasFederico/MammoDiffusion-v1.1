"""Regenerate the canonical generator summary with strict, verified efficiency semantics.

This does NOT recompute any generative metric, embedding, or benchmark quantity.  It reads the
existing ``generator_summary.csv`` run snapshot and rewrites *only* the efficiency columns through
the corrected canonical ``generator_benchmark.efficiency_from_manifest``, which refuses to report a
duration as available without verified semantics.

Outputs (runtime artifacts):

* ``generator_summary_corrected.csv`` / ``generator_ranking_corrected.csv`` — canonical sources for
  reports and downstream;
* ``efficiency_correction.json`` — traceability (source/corrected SHA-256, old/new values).

Run from the repository root: ``python notebooks/utility/correct_efficiency_summary.py``.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / "configs").is_dir())
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import generator_benchmark as gb  # noqa: E402

BENCHMARK = ROOT / gb.BENCHMARK_ROOT
AUDIT = BENCHMARK

EFFICIENCY_FIELDS = ("generation_seconds_per_image", "peak_vram_mb", "energy_kwh",
                     "checkpoint_size_bytes", "efficiency_source", "efficiency_status",
                     "generation_efficiency_status")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _latest_execution_record() -> tuple[dict, str | None]:
    """Return the newest benchmark execution record without assuming a historical run ID."""
    records = sorted((BENCHMARK / "runs").glob("*/execution_config.json"))
    if not records:
        return {}, None
    path = records[-1]
    return json.loads(path.read_text()), str(path.relative_to(ROOT))


def main() -> None:
    registry = gb.load_registry(ROOT)
    execution_record, execution_record_path = _latest_execution_record()
    by_id = {entry["id"]: entry for entry in registry["generators"]}
    summary_path = BENCHMARK / "generator_summary.csv"
    rows = _read_csv(summary_path)
    fieldnames = list(rows[0].keys()) if rows else []

    old_values: dict[str, dict] = {}
    new_values: dict[str, dict] = {}
    affected: list[str] = []
    corrected_rows = []
    for row in rows:
        entry = by_id[row["generator_id"]]
        fixed = gb.efficiency_from_manifest(ROOT, entry)
        key = f"{row['generator_id']}:{row['condition']}"
        before = {field: row.get(field) for field in EFFICIENCY_FIELDS}
        after = {field: fixed.get(field) for field in EFFICIENCY_FIELDS}
        if before != {field: ("" if after[field] is None else after[field]) for field in EFFICIENCY_FIELDS} \
                and str(before.get("generation_seconds_per_image")) != str(after.get("generation_seconds_per_image")):
            affected.append(key)
            old_values[key] = before
            new_values[key] = after
        corrected_rows.append({**row, **{field: fixed.get(field) for field in EFFICIENCY_FIELDS}})

    corrected_summary = BENCHMARK / "generator_summary_corrected.csv"
    gb.write_csv_rows(corrected_summary, corrected_rows, fieldnames=fieldnames)
    # The scientific ranking (family_rank) does not depend on efficiency, but a file named
    # "corrected" must not keep the invalid microsecond durations: rewrite the efficiency columns
    # through the same strict canonical parser while leaving the ranking order untouched.
    ranking_source = BENCHMARK / "generator_ranking.csv"
    corrected_ranking = BENCHMARK / "generator_ranking_corrected.csv"
    if ranking_source.is_file():
        ranking_rows = _read_csv(ranking_source)
        ranking_fields = list(ranking_rows[0].keys()) if ranking_rows else []
        corrected_ranking_rows = []
        for row in ranking_rows:
            fixed = gb.efficiency_from_manifest(ROOT, by_id[row["generator_id"]])
            corrected_ranking_rows.append({**row, **{field: fixed.get(field)
                                                     for field in EFFICIENCY_FIELDS if field in row}})
        gb.write_csv_rows(corrected_ranking, corrected_ranking_rows, fieldnames=ranking_fields)

    AUDIT.mkdir(parents=True, exist_ok=True)
    correction = {
        "source_generator_summary_sha256": gb.file_sha256(summary_path),
        "corrected_generator_summary_sha256": gb.file_sha256(corrected_summary),
        "correction_reason": ("The recorded elapsed_seconds imply physically impossible microsecond-per-image "
                              "generation and the manifests declare no verified duration semantics; the strict "
                              "canonical parser marks such durations unavailable_invalid_duration_semantics and "
                              "drops unverified energy_kwh / peak_vram_mb while keeping checkpoint_size_bytes."),
        "affected_generators": sorted(set(k.split(":")[0] for k in affected)),
        "old_values": old_values,
        "new_values": new_values,
        "benchmark_execution_status": execution_record.get("status"),
        "test_access": False,
    }
    (AUDIT / "efficiency_correction.json").write_text(json.dumps(correction, indent=1))
    print("corrected summary ->", corrected_summary)
    print("affected generators:", correction["affected_generators"])
    for key in sorted(new_values):
        print(f"  {key}: seconds_per_image={new_values[key]['generation_seconds_per_image']} "
              f"status={new_values[key]['efficiency_status']} energy={new_values[key]['energy_kwh']} "
              f"vram={new_values[key]['peak_vram_mb']} ckpt={new_values[key]['checkpoint_size_bytes']}")


if __name__ == "__main__":
    main()
