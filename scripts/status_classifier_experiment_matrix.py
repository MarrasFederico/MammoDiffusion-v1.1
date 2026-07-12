#!/usr/bin/env python3
"""CLI status dashboard for the classifier experiment matrix (spec sections 9/10/18).

    python scripts/status_classifier_experiment_matrix.py
    python scripts/status_classifier_experiment_matrix.py --stage 1 --architecture maxvit512
    python scripts/status_classifier_experiment_matrix.py --watch

Read-only: never claims, trains, or mutates a job. --watch reruns the report every
`--interval` seconds (default 30) until interrupted.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_report(root: Path, stage: int | None = None, architecture: str | None = None) -> dict:
    matrix_path = root / "configs/classifier_experiment_matrix.json"
    if not matrix_path.is_file():
        return {"jobs": [], "by_status": {}, "by_architecture": {}, "by_gpu_eligibility": {}}

    jobs = json.loads(matrix_path.read_text())["jobs"]
    if stage is not None:
        jobs = [j for j in jobs if j["stage"] == stage]
    if architecture is not None:
        jobs = [j for j in jobs if j["architecture"] == architecture]

    by_status = Counter(j["status"] for j in jobs)
    by_architecture = Counter(j["architecture"] for j in jobs)
    by_profile = Counter(j["resource_profile"] for j in jobs)
    running = [j for j in jobs if j["status"] == "RUNNING"]
    failed = [j for j in jobs if j["status"] in ("FAILED_RETRYABLE", "FAILED_FINAL", "BLOCKED")]
    return {
        "total": len(jobs), "by_status": dict(by_status), "by_architecture": dict(by_architecture),
        "by_resource_profile": dict(by_profile), "running": [j["experiment_id"] for j in running],
        "failed_or_blocked": [{"experiment_id": j["experiment_id"], "status": j["status"]} for j in failed],
    }


def print_report(report: dict) -> None:
    print(f"total jobs: {report['total']}")
    print("by status:", json.dumps(report["by_status"]))
    print("by architecture:", json.dumps(report["by_architecture"]))
    print("by resource profile:", json.dumps(report["by_resource_profile"]))
    if report["running"]:
        print("running:", ", ".join(report["running"]))
    if report["failed_or_blocked"]:
        print("failed/blocked:")
        for f in report["failed_or_blocked"]:
            print(f"  {f['experiment_id']}: {f['status']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), default=None)
    parser.add_argument("--architecture", default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args()

    root = Path(args.project_root)
    while True:
        report = build_report(root, args.stage, args.architecture)
        print_report(report)
        if not args.watch:
            break
        print(f"--- refreshing in {args.interval}s (Ctrl-C to stop) ---")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
