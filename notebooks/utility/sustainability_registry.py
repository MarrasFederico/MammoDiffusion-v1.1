"""Canonical sustainability event schema and deduplicated event loading.

Deliberately pandas-free (stdlib JSON only) so it stays importable in the lightweight
`base` conda env alongside the rest of notebooks/utility's widely-reused modules.

Every event is one phase of one run. This module validates and deduplicates the frozen event
registry; the supported v1.1 analysis intentionally derives energy only from elapsed time and the
documented 0.170 kW assumption. Stored CodeCarbon energy/CO2 values are historical fields, not
supported measurements, so this module does not aggregate them into release results.
"""
from __future__ import annotations

import json
from pathlib import Path

PHASES = ("preprocessing", "augmentation", "generator_training", "generation", "filtering",
          "classifier_training", "validation", "locked_test", "metrics")
STATUSES = ("started", "completed", "failed", "resumed", "reused")
VALUE_PRECISION = ("measured", "estimated", "reconstructed", "legacy_unverified", "missing")

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
    """Events that make up the reproducible pipeline: canonical, non-dry-run, completed or a resumed segment that ends in completion, one entry per run_id (last write
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
