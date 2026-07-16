"""Checkpoint/resume provenance for one compact classifier experiment."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

RESUME_NAMES = ("checkpoint_latest", "checkpoint_previous", "checkpoint_best")


def resume_checkpoint_path(run: Path, name: str = "checkpoint_latest") -> Path:
    if name not in RESUME_NAMES:
        raise ValueError(f"unknown resume checkpoint: {name}")
    return run / f"{name}.pkl"


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_pickle(path: Path, payload: dict) -> Path:
    """Durable temporary-file -> fsync -> atomic rename checkpoint write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    _fsync_directory(path.parent)
    return path


def read_resume_checkpoint(path: Path) -> dict:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("unsupported resume checkpoint schema")
    return payload


def save_resume_checkpoint(run: Path, payload: dict, *, best: bool = False) -> Path:
    """Rotate a valid latest; quarantine a corrupt one before publishing the replacement."""
    latest = resume_checkpoint_path(run)
    previous = resume_checkpoint_path(run, "checkpoint_previous")
    if latest.is_file():
        try:
            read_resume_checkpoint(latest)
        except Exception:
            corrupt = latest.with_name(latest.name + ".corrupt")
            if corrupt.exists(): corrupt.unlink()
            os.replace(latest, corrupt)
        else:
            os.replace(latest, previous)
    normalized = {"schema_version": 2, **payload}
    atomic_pickle(latest, normalized)
    if best:
        atomic_pickle(resume_checkpoint_path(run, "checkpoint_best"), normalized)
    return latest


def load_resume_checkpoint(run: Path, expected: dict) -> tuple[dict | None, str]:
    """Load latest, then previous; reject scientific-provenance mismatches."""
    errors = []
    for name in ("checkpoint_latest", "checkpoint_previous"):
        path = resume_checkpoint_path(run, name)
        if not path.is_file():
            continue
        try:
            payload = read_resume_checkpoint(path)
            mismatches = {key: (payload.get(key), value) for key, value in expected.items()
                          if payload.get(key) != value}
            if mismatches:
                errors.append(f"{name}: incompatible {mismatches}")
                continue
            return payload, name
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return None, "; ".join(errors) if errors else "no resume checkpoint"
