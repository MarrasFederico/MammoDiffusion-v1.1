"""Checkpoint/resume provenance for one compact classifier experiment."""
from __future__ import annotations

import io
import json
import os
import pickle
from collections.abc import Mapping
from pathlib import Path

RESUME_NAMES = ("checkpoint_latest", "checkpoint_previous", "checkpoint_best")
COMPLETION_NAME = "run_complete.json"
FINAL_CHECKPOINT = "checkpoint_best.pt"
# Small tabular/JSON outputs that live in the results tree.
RESULT_ARTIFACTS = (
    "configuration.json",
    "dataset_summary.json",
    "model_summary.json",
    "source_accounting.json",
    "training_history.csv",
)
# The model weights live under experiments/; the run is complete only when both the result
# artifacts and the final checkpoint exist.
FINAL_ARTIFACTS = RESULT_ARTIFACTS + (FINAL_CHECKPOINT,)


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


def _to_cpu(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _to_cpu(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item, torch) for item in value)
    return value


def to_cpu(value):
    """Recursively detach tensors and store them on CPU for portable checkpoints."""
    try:
        import torch
    except ImportError:
        return value
    return _to_cpu(value, torch)


class _CPUUnpickler(pickle.Unpickler):
    """Read both new CPU checkpoints and pre-fix CUDA-backed pickle checkpoints."""

    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch

            def load_storage(payload: bytes):
                return torch.load(
                    io.BytesIO(payload), map_location="cpu", weights_only=False
                )

            return load_storage
        return super().find_class(module, name)


def read_resume_checkpoint(path: Path) -> dict:
    with path.open("rb") as stream:
        payload = _CPUUnpickler(stream).load()
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
    normalized = to_cpu({"schema_version": 2, **payload})
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


def terminal_reason(payload: Mapping | None, limits: Mapping) -> str | None:
    """Return why a validated checkpoint is terminal, otherwise ``None``."""
    if not payload:
        return None
    if int(payload.get("early_stopping_counter", 0)) >= int(
        limits["early_stopping_patience"]
    ):
        return "early_stopping"
    if int(payload.get("global_step", 0)) >= int(limits["max_optimizer_updates"]):
        return "max_optimizer_updates"
    if int(payload.get("epoch", 1)) > int(limits["max_epochs"]):
        return "max_epochs_secondary_limit"
    return None


def completion_path(run: Path) -> Path:
    return Path(run) / COMPLETION_NAME


def _completion_mismatches(payload: Mapping, expected: Mapping) -> dict:
    return {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }


def inspect_completed_run(
    results_run: Path, checkpoint_run: Path, expected: Mapping, limits: Mapping
) -> tuple[dict | None, str]:
    """Recognize a finalized run without allocating a model or touching CUDA.

    Result artifacts (JSON/CSV) live under ``results_run``; the model checkpoints live under
    ``checkpoint_run`` (``experiments/classifiers/...``).  A completion marker is authoritative only
    while all final artifacts still exist and its scientific identity matches.  Runs finalized before
    completion markers were introduced are recovered once from their terminal resume checkpoint; the
    notebook then backfills the marker.
    """
    results_run = Path(results_run)
    checkpoint_run = Path(checkpoint_run)
    marker_path = completion_path(results_run)
    missing_artifacts = [name for name in RESULT_ARTIFACTS if not (results_run / name).is_file()]
    if not (checkpoint_run / FINAL_CHECKPOINT).is_file():
        missing_artifacts.append(FINAL_CHECKPOINT)

    if marker_path.is_file():
        try:
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"invalid completion marker: {type(exc).__name__}: {exc}"
        if payload.get("schema_version") != 1 or payload.get("status") != "complete":
            return None, "invalid completion marker schema or status"
        mismatches = _completion_mismatches(payload, expected)
        if mismatches:
            return None, f"incompatible completion marker: {mismatches}"
        if missing_artifacts:
            return None, f"completion marker has missing artifacts: {missing_artifacts}"
        return payload, COMPLETION_NAME

    if missing_artifacts:
        return None, f"missing final artifacts: {missing_artifacts}"

    resume, resume_source = load_resume_checkpoint(checkpoint_run, dict(expected))
    if resume is None:
        return None, f"final artifacts have no compatible checkpoint: {resume_source}"
    reason = terminal_reason(resume, limits)
    if reason is None:
        return None, "final artifacts are paired with a non-terminal checkpoint"

    try:
        configuration = json.loads(
            (results_run / "configuration.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid configuration artifact: {type(exc).__name__}: {exc}"
    configuration_expected = {
        "architecture": expected["architecture"],
        "condition": expected["dataset_variant_id"],
        "seed": expected["seed"],
        "policy_signature": expected["config_signature"],
    }
    configuration_mismatches = _completion_mismatches(
        configuration, configuration_expected
    )
    if configuration_mismatches:
        return None, f"incompatible configuration artifact: {configuration_mismatches}"

    payload = {
        "schema_version": 1,
        "status": "complete",
        **dict(expected),
        "completion_reason": reason,
        "optimizer_updates_limit": int(limits["max_optimizer_updates"]),
        "optimizer_updates_completed": int(resume["global_step"]),
        "next_epoch": int(resume["epoch"]),
        "best_epoch": resume.get("best_epoch"),
        "early_stopping_counter": int(resume.get("early_stopping_counter", 0)),
        "checkpoint_gpu_uuid": resume.get("gpu_uuid"),
        "runtime_gpu_uuid": resume.get("gpu_uuid"),
        "gpu_changed": False,
        "final_checkpoint": "checkpoint_best.pt",
        "artifacts": list(FINAL_ARTIFACTS),
    }
    return payload, f"recovered_from_{resume_source}"
