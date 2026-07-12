"""State machine + atomic locking for classifier experiment matrix jobs (spec section 9).

State is always reconstructible from on-disk artifacts (checkpoint, validation predictions,
metrics files) — the manifest file is a cache of that reconstruction, never the sole truth.
Lock files use the .lock/.claim extensions already excluded by .gitignore.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

STATES = ("PENDING", "CLAIMED", "RUNNING", "INTERRUPTED_RESUMABLE", "TRAINED", "VALIDATING", "VALIDATED",
          "ENSEMBLE_READY", "COMPLETE",
          "BLOCKED", "FAILED_RETRYABLE", "FAILED_FINAL", "INVALIDATED")


def manifest_path(run: Path) -> Path:
    return run / "run_manifest.json"


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(path)


def write_state(run: Path, state: str, **fields) -> dict:
    if state not in STATES:
        raise ValueError(f"invalid state: {state}")
    payload = {"schema_version": 1, "state": state, "updated_at": time.time(), **fields}
    _atomic_write(manifest_path(run), payload)
    return payload


def read_manifest(run: Path) -> dict | None:
    path = manifest_path(run)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def reconstruct_state(run: Path, framework: str) -> dict:
    """Derive the true state from artifacts, ignoring a stale/missing manifest file.

    Import is local to avoid a hard dependency between the two modules at collection time in
    minimal test environments.
    """
    from classifier_checkpoint_io import checkpoint_is_verified  # noqa: PLC0415

    manifest = read_manifest(run)
    if manifest is not None and manifest.get("state") in ("BLOCKED", "FAILED_FINAL", "INVALIDATED"):
        return manifest  # terminal/explicit states are never silently overridden by a rescan

    verified, reason = checkpoint_is_verified(run, framework)
    if not verified:
        if (run / "checkpoint_latest.pkl").is_file() or (run / "checkpoint_previous.pkl").is_file():
            return {"state": "INTERRUPTED_RESUMABLE", "reason": "resume checkpoint available"}
        lock = lock_path(run)
        if lock.is_file() and _pid_is_running(_read_lock_pid(lock)):
            return {"state": "RUNNING", "reason": reason}
        if lock.is_file():
            return {"state": "FAILED_RETRYABLE", "reason": f"stale lock, checkpoint not verified: {reason}"}
        return {"state": "PENDING", "reason": reason}

    validation_marker = run / "validation_complete.json"
    legacy_validation_metrics = run / "validation_metrics.json"
    if not validation_marker.is_file() and not legacy_validation_metrics.is_file():
        return {"state": "TRAINED", "reason": "checkpoint verified, validation metrics missing"}

    ensemble_ready_marker = run.parent / "ensemble_complete.json"
    if ensemble_ready_marker.is_file():
        return {"state": "COMPLETE", "reason": "checkpoint + validation + ensemble all present"}
    return {"state": "VALIDATED", "reason": "checkpoint + validation metrics present"}


# --- atomic claim / stale-PID recovery (mirrors the .lock convention used elsewhere in the repo) ---

def lock_path(run: Path) -> Path:
    return run / "job.lock"


def _pid_is_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _read_lock_pid(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text())
        return int(payload.get("pid"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def acquire_claim(run: Path, worker_id: str, pid: int) -> bool:
    """Best-effort atomic claim: True if this call created the lock (or recovered a stale one)."""
    run.mkdir(parents=True, exist_ok=True)
    lock = lock_path(run)
    if lock.is_file():
        existing_pid = _read_lock_pid(lock)
        if _pid_is_running(existing_pid):
            return False
        # stale lock left by a dead process: safe to reclaim
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # lost a race against another worker's fresh claim
        existing_pid = _read_lock_pid(lock)
        return not _pid_is_running(existing_pid) and _force_reclaim(lock, worker_id, pid)
    else:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"worker_id": worker_id, "pid": pid, "claimed_at": time.time()}))
        return True


def _force_reclaim(lock: Path, worker_id: str, pid: int) -> bool:
    lock.write_text(json.dumps({"worker_id": worker_id, "pid": pid, "claimed_at": time.time()}))
    return True


def release_claim(run: Path) -> None:
    lock = lock_path(run)
    if lock.is_file():
        lock.unlink()
