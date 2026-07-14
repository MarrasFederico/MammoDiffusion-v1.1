#!/usr/bin/env python3
"""Explicit one-shot entrypoint for classifier-matrix v2 locked inference.

The command is intentionally unavailable before the immutable scientific lock exists. A failed
attempt may be retried only after recording a named incident authorization; already completed
prediction tables are reused and are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
sys.path.insert(0, str(ROOT / "scripts"))

from classifier_pipeline_contracts import PIPELINE_NAMESPACE, atomic_json, signed_payload  # noqa: E402
from locked_matrix_inference import run_locked  # noqa: E402
import finalize_locked_test_stage as lock  # noqa: E402


def authorize_retry(root: Path, incident_id: str) -> Path:
    lock_dir = root / lock.LOCK_DIR
    if not incident_id.strip():
        raise ValueError("incident ID must be non-empty")
    if not (lock_dir / "LOCKED_TEST_FAILED").is_file() or (lock_dir / "LOCKED_TEST_COMPLETE").is_file():
        raise RuntimeError("retry authorization requires a failed, incomplete locked attempt")
    lock_payload = json.loads((lock_dir / "EXPERIMENT_MATRIX_LOCKED").read_text())
    payload = signed_payload({
        "schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE,
        "artifact_type": "classifier_locked_retry_authorization", "incident_id": incident_id,
        "lock_signature": lock_payload.get("lock_signature"),
        "scope": "technical_retry_only_no_panel_or_threshold_changes",
    })
    path = lock_dir / "LOCKED_TEST_RETRY_AUTHORIZATION"
    if path.is_file() and json.loads(path.read_text()) != payload:
        raise RuntimeError("a different retry authorization already exists")
    atomic_json(path, payload)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--confirm-locked-inference", action="store_true")
    parser.add_argument("--incident-token")
    parser.add_argument("--authorize-retry", metavar="INCIDENT_ID")
    args = parser.parse_args()
    root = Path(args.project_root)
    if args.authorize_retry:
        print(authorize_retry(root, args.authorize_retry).relative_to(root))
        return
    if not args.confirm_locked_inference:
        raise SystemExit("refusing to open the locked test without --confirm-locked-inference")
    print(json.dumps(run_locked(root, incident_token=args.incident_token), indent=1))


if __name__ == "__main__":
    main()
