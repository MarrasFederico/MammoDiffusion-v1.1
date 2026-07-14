"""Signed GPU profile/smoke certification gate for real classifier-matrix launches."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import importlib.metadata
import platform

from classifier_pipeline_contracts import (
    ARCHITECTURES, PIPELINE_NAMESPACE, code_revision, signed_payload, verify_signed_payload,
)

PROFILE_PATH = "results/runtime_profiles/classifier_vram_profiles.json"
SMOKE_PATH = "results/runtime_profiles/classifier_gpu_smoke_results.json"
MAX_CERTIFICATE_AGE_DAYS = 30

PROFILE_FIELDS = {
    "architecture", "environment_signature", "gpu_name", "gpu_uuid", "total_vram_mb",
    "physical_batch_size", "gradient_accumulation_steps", "effective_batch_size",
    "peak_allocated_mb", "peak_reserved_mb", "measured_at", "code_revision",
    "fixture_signature",
}
SMOKE_FIELDS = {
    "architecture", "environment_signature", "gpu_name", "gpu_uuid", "total_vram_mb",
    "physical_batch_size", "gradient_accumulation_steps", "forward_pass",
    "backward_pass", "checkpoint_save_load", "measured_at", "code_revision",
    "fixture_signature",
}


def make_bundle(artifact_type: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return signed_payload({"schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE,
                           "artifact_type": artifact_type, "records": records})


def environment_signature(policy: Mapping[str, Any]) -> str:
    from classifier_pipeline_contracts import value_signature
    versions = {}
    for distribution in ("tensorflow", "torch", "timm", "transformers", "numpy"):
        try: versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError: versions[distribution] = None
    return value_signature({"python": platform.python_version(), "platform": platform.platform(),
                            "packages": versions, "policy": dict(policy)})


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid GPU certificate timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_bundle(payload: Mapping[str, Any], *, artifact_type: str, fields: set[str],
                     root: Path, now: datetime, max_age_days: int) -> dict[str, dict]:
    verify_signed_payload(payload)
    if payload.get("artifact_type") != artifact_type:
        raise ValueError(f"expected {artifact_type}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("GPU certificate records must be a list")
    by_architecture = {}
    revision = code_revision(root)
    for record in records:
        missing = fields - set(record)
        if missing:
            raise ValueError(f"GPU certificate record lacks {sorted(missing)}")
        architecture = record["architecture"]
        if architecture not in ARCHITECTURES or architecture in by_architecture:
            raise ValueError(f"invalid/duplicate GPU certificate architecture: {architecture}")
        if not record["gpu_uuid"] or float(record["total_vram_mb"]) <= 0:
            raise ValueError(f"{architecture}: GPU identity/VRAM is invalid")
        if int(record["physical_batch_size"]) <= 0 or int(record["gradient_accumulation_steps"]) <= 0:
            raise ValueError(f"{architecture}: invalid tested batch policy")
        age = now - _timestamp(record["measured_at"])
        if age.total_seconds() < 0 or age.days > max_age_days:
            raise ValueError(f"{architecture}: GPU certificate is expired or from the future")
        if revision != "unavailable" and record["code_revision"] != revision:
            raise ValueError(f"{architecture}: GPU certificate code revision is incompatible")
        by_architecture[architecture] = dict(record)
    if set(by_architecture) != set(ARCHITECTURES):
        raise ValueError(f"GPU certificate must cover all architectures; got {sorted(by_architecture)}")
    return by_architecture


def validate_gate(root: Path, *, now: datetime | None = None,
                  max_age_days: int = MAX_CERTIFICATE_AGE_DAYS) -> dict[str, Any]:
    import json
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    profile_path, smoke_path = root / PROFILE_PATH, root / SMOKE_PATH
    result = {"profiles_valid": False, "smokes_valid": False, "ready_for_real_launch": False,
              "profile_path": PROFILE_PATH, "smoke_path": SMOKE_PATH, "errors": []}
    if not profile_path.is_file():
        result["errors"].append("missing signed GPU profile bundle")
        return result
    try:
        profiles = _validate_bundle(json.loads(profile_path.read_text()), artifact_type="gpu_profile_bundle",
                                    fields=PROFILE_FIELDS, root=root, now=now, max_age_days=max_age_days)
        result["profiles_valid"] = True
    except Exception as exc:
        result["errors"].append(f"invalid GPU profiles: {exc}")
        return result
    if not smoke_path.is_file():
        result["errors"].append("missing signed GPU smoke bundle")
        return result
    try:
        smokes = _validate_bundle(json.loads(smoke_path.read_text()), artifact_type="gpu_smoke_bundle",
                                  fields=SMOKE_FIELDS, root=root, now=now, max_age_days=max_age_days)
        for architecture in ARCHITECTURES:
            smoke, profile = smokes[architecture], profiles[architecture]
            if not all(smoke[field] is True for field in ("forward_pass", "backward_pass", "checkpoint_save_load")):
                raise ValueError(f"{architecture}: forward/backward/checkpoint smoke did not PASS")
            for field in ("environment_signature", "gpu_uuid", "physical_batch_size", "gradient_accumulation_steps",
                          "fixture_signature", "code_revision"):
                if smoke[field] != profile[field]:
                    raise ValueError(f"{architecture}: profile/smoke mismatch in {field}")
        result["smokes_valid"] = True
        result["ready_for_real_launch"] = True
    except Exception as exc:
        result["errors"].append(f"invalid GPU smokes: {exc}")
    return result


def require_real_launch_gate(root: Path) -> dict[str, Any]:
    result = validate_gate(root)
    if not result["ready_for_real_launch"]:
        raise RuntimeError("real Stage 1/2 launch blocked by GPU certification gate: " + "; ".join(result["errors"]))
    return result


__all__ = ["MAX_CERTIFICATE_AGE_DAYS", "PROFILE_PATH", "SMOKE_PATH", "environment_signature", "make_bundle",
           "require_real_launch_gate", "validate_gate"]
