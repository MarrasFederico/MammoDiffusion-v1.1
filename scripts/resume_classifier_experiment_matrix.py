#!/usr/bin/env python3
"""Rescan every job in configs/classifier_experiment_matrix.json and refresh its status from
on-disk artifacts (spec section 9: state must be reconstructible, never queue-only).

    python scripts/resume_classifier_experiment_matrix.py --stage 1

Also clears stale claims left by a crashed worker (a live PID check already happens inside
classifier_run_manifest.reconstruct_state) and reports jobs that moved into a terminal or
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


def resume(root: Path, stage: int | None = None) -> dict:
    matrix_path = root / "configs/classifier_experiment_matrix.json"
    payload = json.loads(matrix_path.read_text())
    protocols = json.loads((root / "configs/classifier_training_protocols.json").read_text())["policies"]

    changed = []
    for job in payload["jobs"]:
        if stage is not None and job["stage"] != stage:
            continue
        framework = protocols[job["architecture"]]["framework"]
        run = ckio.run_dir(root, job["architecture"], job["dataset_variant_id"], job["training_policy"], job["seed"])
        new_state = crm.reconstruct_state(run, framework)["state"]
        if new_state != job["status"]:
            changed.append({"experiment_id": job["experiment_id"], "from": job["status"], "to": new_state})
            job["status"] = new_state

    matrix_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    return {"changed": changed, "total_scanned": sum(1 for j in payload["jobs"] if stage is None or j["stage"] == stage)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), default=None)
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args()
    result = resume(Path(args.project_root), args.stage)
    print(f"scanned {result['total_scanned']} jobs, {len(result['changed'])} status changes")
    for c in result["changed"]:
        print(f"  {c['experiment_id']}: {c['from']} -> {c['to']}")


if __name__ == "__main__":
    main()
