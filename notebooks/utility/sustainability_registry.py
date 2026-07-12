"""Canonical sustainability/energy event schema and deduplicated aggregation (spec section 11).

Deliberately pandas-free (stdlib csv/json only) so it stays importable in the lightweight
`base` conda env alongside the rest of notebooks/utility's widely-reused modules.

Every event is one phase of one run. Two totals are always reported side by side and never
silently merged: `actual_project_energy` (everything really attempted, including failed and
resumed segments) and `canonical_pipeline_energy` (the reproducible pipeline cost, deduplicated,
no failures, no PLAN_ONLY/dry-run noise).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

PHASES = ("preprocessing", "augmentation", "generator_training", "generation", "filtering",
          "classifier_training", "validation", "locked_test", "metrics")
STATUSES = ("started", "completed", "failed", "resumed", "reused")
VALUE_PRECISION = ("measured", "estimated", "reconstructed", "legacy_unverified", "missing")

EVENT_FIELDS = ("run_id", "experiment_id", "dataset_variant_id", "architecture", "seed", "phase",
                 "status", "parent_run_id", "canonical", "reused_artifact", "start_time", "end_time",
                 "elapsed_seconds", "energy_kwh", "co2_kg", "peak_ram_mb", "peak_vram_mb", "gpu_uuid",
                 "gpu_name", "num_images", "optimizer_updates", "epochs", "source_log", "signature",
                 "value_precision")


def validate_event(event: dict) -> list[str]:
    errors = []
    if event.get("phase") not in PHASES:
        errors.append(f"invalid phase: {event.get('phase')}")
    if event.get("status") not in STATUSES:
        errors.append(f"invalid status: {event.get('status')}")
    if not event.get("run_id"):
        errors.append("missing run_id")
    precision = event.get("value_precision")
    if precision is not None and precision not in VALUE_PRECISION:
        errors.append(f"invalid value_precision: {precision}")
    for numeric_field in ("elapsed_seconds", "energy_kwh", "co2_kg"):
        value = event.get(numeric_field)
        if value is not None and (isinstance(value, float) and value != value):  # NaN check, no numpy needed
            errors.append(f"{numeric_field} is NaN: rejected, must be a real measured/estimated number or omitted")
    return errors


def append_event(events_path: Path, event: dict) -> None:
    errors = validate_event(event)
    if errors:
        raise ValueError(f"invalid sustainability event: {errors}")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_events(events_path: Path) -> list[dict]:
    if not events_path.is_file():
        return []
    events = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def deduplicate_canonical_events(events: list[dict]) -> list[dict]:
    """Events that make up the reproducible pipeline: canonical, non-dry-run, non-PLAN_ONLY,
    completed or a resumed segment that ends in completion, one entry per run_id (last write
    wins), and never a duplicate of a "reused" artifact's own originating run.
    """
    by_run_id: dict[str, dict] = {}
    for event in events:
        if not event.get("canonical", False):
            continue
        if event.get("status") == "reused":
            continue  # cost was already attributed to the run that produced the artifact
        run_id = event["run_id"]
        existing = by_run_id.get(run_id)
        if existing is None or event.get("end_time", "") >= existing.get("end_time", ""):
            by_run_id[run_id] = event
    return [e for e in by_run_id.values() if e.get("status") == "completed"]


def actual_vs_canonical(events: list[dict]) -> dict:
    actual_ids_seen = set()
    actual_kwh = actual_co2 = actual_seconds = 0.0
    for event in events:
        # actual_project_energy: every real attempt, deduplicated only by exact run_id repeat
        # (a literal re-logged duplicate of the same event), never by outcome.
        dedup_key = (event["run_id"], event.get("phase"), event.get("status"), event.get("start_time"))
        if dedup_key in actual_ids_seen:
            continue
        actual_ids_seen.add(dedup_key)
        actual_kwh += event.get("energy_kwh") or 0.0
        actual_co2 += event.get("co2_kg") or 0.0
        actual_seconds += event.get("elapsed_seconds") or 0.0

    canonical = deduplicate_canonical_events(events)
    canonical_kwh = sum(e.get("energy_kwh") or 0.0 for e in canonical)
    canonical_co2 = sum(e.get("co2_kg") or 0.0 for e in canonical)
    canonical_seconds = sum(e.get("elapsed_seconds") or 0.0 for e in canonical)

    return {
        "actual_project_energy_kwh": actual_kwh, "actual_project_co2_kg": actual_co2, "actual_project_seconds": actual_seconds,
        "canonical_pipeline_energy_kwh": canonical_kwh, "canonical_pipeline_co2_kg": canonical_co2, "canonical_pipeline_seconds": canonical_seconds,
        "retry_and_failure_overhead_kwh": max(0.0, actual_kwh - canonical_kwh),
        "n_events_actual": len(actual_ids_seen), "n_events_canonical": len(canonical),
    }


def sum_resumed_segments(events: list[dict], run_id: str) -> dict:
    """Sum non-overlapping resume segments belonging to the same canonical training run,
    ordered by start_time, so a training that was resumed 3 times is counted once, not 3x
    (spec 11.1: "somma segmenti non sovrapposti appartenenti allo stesso training canonico").
    """
    segments = sorted((e for e in events if e["run_id"] == run_id and e.get("canonical")), key=lambda e: e.get("start_time") or "")
    total_seconds = sum(e.get("elapsed_seconds") or 0.0 for e in segments)
    total_kwh = sum(e.get("energy_kwh") or 0.0 for e in segments)
    total_co2 = sum(e.get("co2_kg") or 0.0 for e in segments)
    return {"run_id": run_id, "n_segments": len(segments), "elapsed_seconds": total_seconds,
            "energy_kwh": total_kwh, "co2_kg": total_co2}


def group_by_phase(events: list[dict]) -> dict[str, dict]:
    canonical = deduplicate_canonical_events(events)
    by_phase: dict[str, dict] = defaultdict(lambda: {"energy_kwh": 0.0, "co2_kg": 0.0, "elapsed_seconds": 0.0, "n_events": 0})
    for event in canonical:
        bucket = by_phase[event["phase"]]
        bucket["energy_kwh"] += event.get("energy_kwh") or 0.0
        bucket["co2_kg"] += event.get("co2_kg") or 0.0
        bucket["elapsed_seconds"] += event.get("elapsed_seconds") or 0.0
        bucket["n_events"] += 1
    return dict(by_phase)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_summary_by_run(root: Path, events: list[dict]) -> Path:
    canonical = deduplicate_canonical_events(events)
    fieldnames = ["run_id", "experiment_id", "phase", "architecture", "energy_kwh", "co2_kg", "elapsed_seconds", "value_precision"]
    out = root / "results/sustainability/summary_by_run.csv"
    _write_csv(out, canonical, fieldnames)
    return out


def write_summary_by_experiment(root: Path, events: list[dict]) -> Path:
    canonical = deduplicate_canonical_events(events)
    by_experiment: dict[str, dict] = defaultdict(lambda: {"energy_kwh": 0.0, "co2_kg": 0.0, "elapsed_seconds": 0.0, "n_runs": 0})
    for event in canonical:
        key = event.get("experiment_id") or "unknown"
        bucket = by_experiment[key]
        bucket["energy_kwh"] += event.get("energy_kwh") or 0.0
        bucket["co2_kg"] += event.get("co2_kg") or 0.0
        bucket["elapsed_seconds"] += event.get("elapsed_seconds") or 0.0
        bucket["n_runs"] += 1
    rows = [{"experiment_id": k, **v} for k, v in by_experiment.items()]
    out = root / "results/sustainability/summary_by_experiment.csv"
    _write_csv(out, rows, ["experiment_id", "energy_kwh", "co2_kg", "elapsed_seconds", "n_runs"])
    return out


def normalized_metrics(event: dict) -> dict:
    """kWh/1000 images, seconds/1000 images, kWh/optimizer-update — spec 13.2 normalized set."""
    n_images = event.get("num_images") or 0
    updates = event.get("optimizer_updates") or 0
    energy = event.get("energy_kwh") or 0.0
    seconds = event.get("elapsed_seconds") or 0.0
    return {
        "kwh_per_1000_images": (energy / n_images * 1000) if n_images else None,
        "seconds_per_1000_images": (seconds / n_images * 1000) if n_images else None,
        "kwh_per_optimizer_update": (energy / updates) if updates else None,
    }
