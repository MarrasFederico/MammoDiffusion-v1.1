#!/usr/bin/env python3
"""Build a read-only, content-aware inventory of the immediate legacy tree."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPES = ("experiments", "notebooks", "results")


def find_legacy_root() -> Path:
    expected = ROOT.parent / "Vecchia versione"
    candidates = [expected, *(p for p in ROOT.parent.iterdir() if p.is_dir() and p != ROOT)]
    for candidate in candidates:
        if all((candidate / scope).is_dir() for scope in SCOPES):
            return candidate.resolve()
    raise FileNotFoundError("No immediate sibling contains experiments/, notebooks/, and results/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".keras", ".h5", ".ckpt", ".safetensors", ".bin", ".pt", ".pth"}:
        return "checkpoint_or_model"
    if suffix == ".ipynb": return "notebook"
    if "manifest" in path.name.lower(): return "manifest"
    if suffix in {".csv", ".json", ".jsonl"}: return "metrics_metadata_or_log"
    if suffix in {".png", ".jpg", ".jpeg"}: return "image_or_plot"
    return "other"


def classify(relative: Path, current: Path, same: bool, kind: str) -> tuple[str, str]:
    if current.exists():
        if same: return "DUPLICATE_IDENTICAL", "Canonical path already has byte-identical content."
        if kind == "notebook": return "CURRENT_VERSION_PREFERRED", "Legacy notebooks are reference-only."
        return "AMBIGUOUS_DO_NOT_TOUCH", "Canonical path differs; manual provenance validation required."
    if kind == "notebook": return "LEGACY_UNVERIFIED_REFERENCE_ONLY", "Extract facts; do not overwrite current notebooks."
    if kind == "checkpoint_or_model": return "COPY_AFTER_VALIDATION", "Missing binary requires architecture and provenance validation."
    if relative.parts and relative.parts[0] == "results":
        return "COPY_AFTER_VALIDATION", "Missing result may be relevant but remains legacy_unverified."
    return "LEGACY_UNVERIFIED_REFERENCE_ONLY", "No active canonical reference has been established."


def main() -> None:
    legacy = find_legacy_root()
    rows = []
    for scope in SCOPES:
        for source in sorted((legacy / scope).rglob("*")):
            if not source.is_file() or source.is_symlink(): continue
            relative = source.relative_to(legacy)
            current = ROOT / relative
            source_hash = sha256(source)
            present = current.is_file()
            same = present and current.stat().st_size == source.stat().st_size and sha256(current) == source_hash
            kind = artifact_type(source)
            action, reason = classify(relative, current, same, kind)
            experiment = relative.parts[1] if len(relative.parts) > 1 else ""
            rows.append({
                "legacy_path": str(source), "canonical_path": str(current),
                "size_bytes": source.stat().st_size, "sha256": source_hash,
                "present_current": str(present).lower(), "same_content": str(same).lower(),
                "artifact_type": kind, "experiment": experiment,
                "provenance": "legacy_unverified", "action": action, "reason": reason,
                "references": "not established by filename alone",
            })
    fields = list(rows[0]) if rows else []
    with (ROOT / "legacy_recovery_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    counts = {name: sum(row["action"] == name for row in rows) for name in sorted({r["action"] for r in rows})}
    lines = ["# Legacy recovery inventory", "", f"Legacy root (read-only): `{legacy}`", "",
             f"Files examined: **{len(rows)}**.", "", "| Classification | Count |", "|---|---:|"]
    lines.extend(f"| `{name}` | {count} |" for name, count in counts.items())
    lines += ["", "The CSV is authoritative for per-file size, SHA-256, target, provenance, and decision.",
              "No artifact is promoted to verified solely because it exists in the legacy tree.", ""]
    (ROOT / "legacy_recovery_inventory.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(rows)} rows from {legacy}")


if __name__ == "__main__":
    main()
