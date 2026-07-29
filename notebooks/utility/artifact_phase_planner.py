"""Content-aware phase planning for idempotent notebook Run All execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

VALID_MODES = {
    "training": {"auto", "run", "skip"},
    "generation": {"auto", "run", "skip"},
    "evaluation": {"auto", "run", "skip", "recompute", "frozen"},
    "filter": {"auto", "run", "skip", "recompute"},
    "validation": {"auto", "run", "skip", "recompute"},
    "locked_test": {"manual", "run", "skip"},
}

# Phases where "auto" discovering incomplete evidence implies a heavy, costly
# re-run (full training or full regeneration of a whole image set). These are
# the only phases gated by ALLOW_HEAVY_RETRAIN / ALLOW_FULL_REGENERATION.
HEAVY_PHASES = {"training": "ALLOW_HEAVY_RETRAIN", "generation": "ALLOW_FULL_REGENERATION"}


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


def count_valid_images(spec: dict, root: Path) -> int:
    """Number of already-valid, correctly-named images for an image_set spec.

    Used to tell a resumable partial gap (some valid images exist) apart from a
    genuine from-scratch regeneration (none do) — only the latter is heavy.
    """
    path = root / spec["path"]
    if not path.is_dir(): return 0
    if "allowed_prefixes" in spec:
        return sum(1 for p in path.glob("*.png")
                    if any(p.stem.startswith(prefix) and p.stem[len(prefix):].isdigit() for prefix in spec["allowed_prefixes"]))
    indices, _invalid = indexed_pngs(path, spec["prefix"])
    return len(indices)


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
    path = Path(experiment_dir) / "runtime_manifest.json"
    if not path.is_file(): return {"valid": False, "reason": f"missing manifest: {path}", "phases": {}}
    try: payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc: return {"valid": False, "reason": f"invalid manifest: {exc}", "phases": {}}
    if payload.get("schema_version") != 1 or not isinstance(payload.get("phases"), dict):
        return {"valid": False, "reason": "manifest schema is not valid", "phases": {}}
    root, results = Path(experiment_dir), {}
    for phase, spec in payload.get("phases", {}).items():
        checks = []
        for item in spec.get("files", []): checks.append(verify_file(item, root))
        for item in spec.get("image_sets", []): checks.append(verify_images(item, root))
        ok = bool(checks) and all(value for value, _ in checks)
        partial = any(count_valid_images(item, root) > 0 for item in spec.get("image_sets", []))
        results[phase] = {"complete": ok, "reason": "; ".join(reason for _, reason in checks) or "no evidence", "has_partial_progress": partial}
    return {"valid": True, "reason": "manifest parsed", "phases": results, "payload": payload}


def resolve_action(phase: str, mode: str, complete: bool, reason: str, allow_heavy: bool = True, has_partial_progress: bool = False) -> dict:
    if mode not in VALID_MODES[phase]: raise ValueError(f"Invalid {phase} mode: {mode}")
    if mode == "manual": return {"phase": phase, "status": "manual", "action": "skip", "reason": "manual locked-test gate"}
    if mode == "frozen":
        return {
            "phase": phase,
            "status": "frozen_selection",
            "action": "skip",
            "reason": "historical evaluation is frozen; the notebook must validate its recorded selection and immutable checkpoint",
        }
    if mode == "skip":
        if not complete: raise RuntimeError(f"Cannot skip incomplete {phase}: {reason}")
        action = "skip"
    elif mode == "run": action = "run"
    elif mode == "recompute": action = "recompute"
    elif complete: action = "skip"
    elif phase in HEAVY_PHASES and not allow_heavy and not has_partial_progress: action = "blocked"
    else: action = "run"
    status = "complete" if complete else "missing_or_invalid"
    if action == "blocked":
        flag = HEAVY_PHASES[phase]
        reason = f"heavy {phase} required but blocked: set {flag} = True to allow auto mode to proceed ({reason})"
    return {"phase": phase, "status": status, "action": action, "reason": reason}


def plan_experiment(experiment_dir: str | Path, modes: dict[str, str], allow_flags: dict[str, bool] | None = None) -> list[dict]:
    audit = load_runtime_manifest(experiment_dir); plan = []
    allow_flags = allow_flags or {}
    for phase, mode in modes.items():
        evidence = audit["phases"].get(phase, {"complete": False, "reason": audit["reason"], "has_partial_progress": False})
        allow_heavy = allow_flags.get(phase, True)
        partial = evidence.get("has_partial_progress", False)
        plan.append(resolve_action(phase, mode, evidence["complete"], evidence["reason"], allow_heavy, partial))
    return plan


def print_plan(plan: list[dict]) -> None:
    print("| Fase | Stato | Azione prevista | Motivo |")
    print("|---|---|---|---|")
    for row in plan: print(f"| {row['phase']} | {row['status']} | {row['action']} | {row['reason']} |")


def phase_should_run(plan: list[dict], phase: str, plan_only: bool = False) -> bool:
    row = next(item for item in plan if item["phase"] == phase)
    label = phase.upper()
    if plan_only: print(f"PLAN ONLY — {label}: {row['action']} — {row['reason']}"); return False
    if row["action"] == "blocked": raise RuntimeError(f"BLOCKED {label} — {row['reason']}")
    if row["action"] == "skip": print(f"SKIP {label} — {row['reason']}"); return False
    if row["action"] == "recompute": print(f"RECOMPUTE {label} — immagini/checkpoint riusati")
    else: print(f"RUN {label} — {row['reason']}")
    return True
