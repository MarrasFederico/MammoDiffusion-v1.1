"""Content-aware phase planning for idempotent notebook Run All execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

VALID_MODES = {
    "training": {"auto", "run", "skip"},
    "generation": {"auto", "run", "skip"},
    "evaluation": {"auto", "run", "skip", "recompute"},
    "filter": {"auto", "run", "skip", "recompute"},
    "validation": {"auto", "run", "skip", "recompute"},
    "locked_test": {"manual", "run", "skip"},
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def indexed_pngs(path: str | Path, prefix: str) -> tuple[list[int], list[str]]:
    root = Path(path); indices, invalid = [], []
    for item in sorted(root.glob("*.png")) if root.is_dir() else []:
        stem = item.stem
        if not stem.startswith(prefix) or not stem[len(prefix):].isdigit(): invalid.append(item.name)
        else: indices.append(int(stem[len(prefix):]))
    return indices, invalid


def verify_file(spec: dict, root: Path) -> tuple[bool, str]:
    path = root / spec["path"]
    if not path.is_file(): return False, f"missing file: {spec['path']}"
    if spec.get("size_bytes") is not None and path.stat().st_size != spec["size_bytes"]:
        return False, f"size mismatch: {spec['path']}"
    if spec.get("sha256") and file_sha256(path) != spec["sha256"]:
        return False, f"signature mismatch: {spec['path']}"
    return True, f"verified file: {spec['path']}"


def verify_images(spec: dict, root: Path) -> tuple[bool, str]:
    path = root / spec["path"]
    if "allowed_prefixes" in spec:
        files = sorted(path.glob("*.png")) if path.is_dir() else []
        invalid = [p.name for p in files if not any(p.stem.startswith(prefix) and p.stem[len(prefix):].isdigit() for prefix in spec["allowed_prefixes"])]
        if invalid: return False, f"invalid image names in {spec['path']}: {invalid[:3]}"
        if len(files) != spec["count"]: return False, f"image count mismatch in {spec['path']}: {len(files)}/{spec['count']}"
        identities = {(p.stat().st_size, file_sha256(p)) for p in files}
        if len(identities) != len(files): return False, f"duplicate image content in {spec['path']}"
        return True, f"verified {len(files)} named, content-unique images: {spec['path']}"
    indices, invalid = indexed_pngs(path, spec["prefix"])
    expected = list(range(spec["start_index"], spec["start_index"] + spec["count"]))
    if invalid: return False, f"invalid image names in {spec['path']}: {invalid[:3]}"
    if indices != expected: return False, f"image index set mismatch in {spec['path']}: {len(indices)}/{len(expected)}"
    if len(indices) != len(set(indices)): return False, f"duplicate indices in {spec['path']}"
    return True, f"verified {len(indices)} indexed images: {spec['path']}"


def load_runtime_manifest(experiment_dir: str | Path) -> dict:
    path = Path(experiment_dir) / "legacy_runtime_manifest.json"
    if not path.is_file(): return {"valid": False, "reason": f"missing manifest: {path}", "phases": {}}
    try: payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc: return {"valid": False, "reason": f"invalid manifest: {exc}", "phases": {}}
    if payload.get("schema_version") != 1 or payload.get("provenance") != "legacy_normalized_verified":
        return {"valid": False, "reason": "manifest schema/provenance is not promotable", "phases": {}}
    root, results = Path(experiment_dir), {}
    for phase, spec in payload.get("phases", {}).items():
        checks = []
        for item in spec.get("files", []): checks.append(verify_file(item, root))
        for item in spec.get("image_sets", []): checks.append(verify_images(item, root))
        ok = bool(checks) and all(value for value, _ in checks)
        results[phase] = {"complete": ok, "reason": "; ".join(reason for _, reason in checks) or "no evidence"}
    return {"valid": True, "reason": "manifest parsed", "phases": results, "payload": payload}


def resolve_action(phase: str, mode: str, complete: bool, reason: str) -> dict:
    if mode not in VALID_MODES[phase]: raise ValueError(f"Invalid {phase} mode: {mode}")
    if mode == "manual": return {"phase": phase, "status": "manual", "action": "skip", "reason": "manual locked-test gate"}
    if mode == "skip":
        if not complete: raise RuntimeError(f"Cannot skip incomplete {phase}: {reason}")
        action = "skip"
    elif mode == "run": action = "run"
    elif mode == "recompute": action = "recompute"
    else: action = "skip" if complete else "run"
    return {"phase": phase, "status": "complete" if complete else "missing_or_invalid", "action": action, "reason": reason}


def plan_experiment(experiment_dir: str | Path, modes: dict[str, str]) -> list[dict]:
    audit = load_runtime_manifest(experiment_dir); plan = []
    for phase, mode in modes.items():
        evidence = audit["phases"].get(phase, {"complete": False, "reason": audit["reason"]})
        plan.append(resolve_action(phase, mode, evidence["complete"], evidence["reason"]))
    return plan


def print_plan(plan: list[dict]) -> None:
    print("| Fase | Stato | Azione prevista | Motivo |")
    print("|---|---|---|---|")
    for row in plan: print(f"| {row['phase']} | {row['status']} | {row['action']} | {row['reason']} |")


def phase_should_run(plan: list[dict], phase: str, plan_only: bool = False) -> bool:
    row = next(item for item in plan if item["phase"] == phase)
    label = phase.upper()
    if plan_only: print(f"PLAN ONLY — {label}: {row['action']} — {row['reason']}"); return False
    if row["action"] == "skip": print(f"SKIP {label} — {row['reason']}"); return False
    if row["action"] == "recompute": print(f"RECOMPUTE {label} — immagini/checkpoint riusati")
    else: print(f"RUN {label} — {row['reason']}")
    return True
