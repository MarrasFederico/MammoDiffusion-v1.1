#!/usr/bin/env python3
"""Normalize all on-disk legacy EcoTracker/CodeCarbon-style records idempotently."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import sustainability_registry as registry  # noqa: E402


def phase_of(path: Path, row: dict) -> str:
    text = " ".join((str(path).lower(), str(row.get("label", "")).lower(), str(row.get("phase", "")).lower()))
    if "preprocess" in text: return "preprocessing"
    if "augment" in text: return "augmentation"
    if "filter" in text: return "filtering"
    if "validation" in text or "evaluation" in text or "select_best" in text: return "validation"
    if "generation" in text or "generate" in text or "sampling" in text: return "generation"
    if "classifier" in text: return "classifier_training"
    if "train" in text or "finetun" in text or "vae_" in text or "ldm_" in text: return "generator_training"
    return "metrics"


def records(path: Path):
    try:
        if path.suffix == ".jsonl":
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip(): yield line_number, json.loads(line)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict): yield 1, payload
            elif isinstance(payload, list):
                for index, row in enumerate(payload, 1): yield index, row
    except (OSError, json.JSONDecodeError):
        return


def discover(root: Path) -> list[Path]:
    paths = []
    for base in (root / "experiments", root / "results"):
        if not base.exists(): continue
        for path in base.rglob("*.json*"):
            name = str(path).lower()
            if "results/sustainability/" in name:
                continue  # never recursively import this script's own canonical outputs
            if any(token in name for token in ("sustain", "ecotracker", "emission", "codecarbon")):
                paths.append(path)
    return sorted(set(paths))


def normalize(root: Path) -> list[dict]:
    events, seen = [], set()
    for path in discover(root):
        for line_number, row in records(path):
            if not isinstance(row, dict) or row.get("energy_kwh") is None:
                continue
            content = json.dumps(row, sort_keys=True, separators=(",", ":"))
            signature = hashlib.sha256(content.encode()).hexdigest()
            if signature in seen: continue
            seen.add(signature)
            relative = str(path.relative_to(root))
            parts = path.parts
            experiment_id = next((part for part in parts if part[:2].isdigit() and "_" in part), path.parent.name)
            status = row.get("status", "completed")
            if status not in registry.STATUSES: status = "completed" if row.get("returncode", 0) == 0 else "failed"
            precision = "estimated" if any(key.endswith("_estimated") and value for key, value in row.items()) else "measured"
            run_id = f"legacy-{signature[:20]}"
            event = {
                "run_id": run_id, "experiment_id": experiment_id, "dataset_variant_id": None,
                "architecture": None, "seed": row.get("seed"), "phase": phase_of(path, row), "status": status,
                "parent_run_id": None, "canonical": status == "completed", "reused_artifact": False,
                "start_time": None, "end_time": row.get("timestamp"),
                "elapsed_seconds": row.get("elapsed_seconds"), "energy_kwh": row.get("energy_kwh"),
                "co2_kg": row.get("co2_kg"), "peak_ram_mb": row.get("peak_ram_mb"),
                "peak_vram_mb": row.get("peak_vram_mb"), "gpu_uuid": row.get("gpu_uuid"),
                "gpu_name": row.get("gpu_name"),
                "num_images": row.get("num_images") or row.get("n_images") or row.get("n_per_class"),
                "optimizer_updates": row.get("optimizer_updates") or row.get("total_steps") or row.get("max_train_steps"),
                "epochs": row.get("epochs") or row.get("epochs_run"), "source_log": f"{relative}:{line_number}",
                "signature": signature, "value_precision": precision,
            }
            errors = registry.validate_event(event)
            if errors: raise ValueError(f"{relative}:{line_number}: {errors}")
            events.append(event)
    return sorted(events, key=lambda event: (event["source_log"], event["signature"]))


def main() -> None:
    events = normalize(ROOT)
    out = ROOT / "results/sustainability/canonical_events.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in events))
    registry.write_summary_by_run(ROOT, events)
    registry.write_summary_by_experiment(ROOT, events)
    totals = registry.actual_vs_canonical(events)
    (out.parent / "actual_vs_canonical.json").write_text(json.dumps(totals, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"events": len(events), **totals}, indent=1))


if __name__ == "__main__": main()
