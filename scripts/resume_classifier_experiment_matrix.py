#!/usr/bin/env python3
"""Rescan every job in configs/classifier_experiment_matrix.json and refresh its status from
on-disk artifacts (spec section 9: state must be reconstructible, never queue-only).

    python scripts/resume_classifier_experiment_matrix.py --stage 1

Also identifies stale claims left by a crashed worker (they are atomically reclaimed by the next
worker claim) and reports jobs that moved into a terminal or
retryable state since the matrix file was last written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import classifier_checkpoint_io as ckio  # noqa: E402
import classifier_run_manifest as crm  # noqa: E402
from classifier_pipeline_contracts import atomic_json, canonical_state, validate_matrix  # noqa: E402


def resume(root: Path, stage: int | None = None) -> dict:
    matrix_path = root / "configs/classifier_experiment_matrix.json"
    payload = json.loads(matrix_path.read_text())
    validate_matrix(payload)
    protocols = json.loads((root / "configs/classifier_training_protocols.json").read_text())["policies"]

    changed = []
    for job in payload["jobs"]:
        if stage is not None and job["stage"] != stage:
            continue
        framework = protocols[job["architecture"]]["framework"]
        run = ckio.run_dir(root, job["architecture"], job["dataset_variant_id"], job["training_policy"], job["seed"])
        new_state = crm.reconstruct_state(run, framework)["state"]
        canonical_state(new_state)
        if new_state != job["status"]:
            changed.append({"experiment_id": job["experiment_id"], "from": job["status"], "to": new_state})
            job["status"] = new_state

    atomic_json(matrix_path, payload)
    return {"changed": changed, "total_scanned": sum(1 for j in payload["jobs"] if stage is None or j["stage"] == stage)}


def reset_failed_final(root: Path, experiment_id: str, reason: str) -> dict:
    """Explicit operator recovery; never inferred automatically from a terminal scientific state."""
    if not reason.strip():
        raise ValueError("--reason is required for a FAILED_FINAL reset")
    matrix_path = root / "configs/classifier_experiment_matrix.json"
    payload = json.loads(matrix_path.read_text()); jobs = validate_matrix(payload)
    job = next((row for row in jobs if row["experiment_id"] == experiment_id), None)
    if job is None: raise ValueError(f"unknown experiment ID: {experiment_id}")
    run = ckio.run_dir(root, job["architecture"], job["dataset_variant_id"], job["training_policy"], job["seed"])
    current = crm.read_manifest(run)
    if not current or current.get("state") != "FAILED_FINAL":
        raise ValueError(f"explicit reset requires FAILED_FINAL, got {(current or {}).get('state')}")
    crm.reset_terminal_state(run, reason=reason)
    job["status"] = "PENDING"; atomic_json(matrix_path, payload)
    return {"experiment_id": experiment_id, "from": "FAILED_FINAL", "to": "PENDING", "reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), default=None)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--reset-failed-final", metavar="EXPERIMENT_ID")
    parser.add_argument("--reason")
    args = parser.parse_args()
    if args.reset_failed_final:
        result = reset_failed_final(Path(args.project_root), args.reset_failed_final, args.reason or "")
        print(json.dumps(result, indent=1)); return
    result = resume(Path(args.project_root), args.stage)
    print(f"scanned {result['total_scanned']} jobs, {len(result['changed'])} status changes")
    for c in result["changed"]:
        print(f"  {c['experiment_id']}: {c['from']} -> {c['to']}")


if __name__ == "__main__":
    main()
