"""Resolves a dataset_variant registry entry into an actual, signed, deterministic file list
for classifier training (spec section 8). Real/augmented files are enumerated directly;
synthetic files are drawn with dataset_variant_registry.deterministic_sample_signature so the
same variant always yields the same picks regardless of filesystem enumeration order.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_variant_registry import CLASS_LABEL, deterministic_sample_signature  # noqa: E402

REAL_TRAIN_DIR = "data/processed/train"
AUGMENTED_DIR = "data/real_augmented"


def _real_files_by_class(root: Path) -> dict[str, list[str]]:
    metadata = root / "data/processed/metadata/train.csv"
    with metadata.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    label_to_class = {v: k for k, v in CLASS_LABEL.items()}
    by_class: dict[str, list[str]] = {k: [] for k in CLASS_LABEL}
    for row in rows:
        klass = label_to_class.get(row["label"])
        if klass is None:
            continue
        processed_path = row.get("processed_path")
        by_class[klass].append(processed_path if processed_path else f"{REAL_TRAIN_DIR}/{row['label']}/{row['image_id']}.png")
    return by_class


def _augmented_files_by_class(root: Path) -> dict[str, list[str]]:
    directory = root / AUGMENTED_DIR
    by_class: dict[str, list[str]] = {k: [] for k in CLASS_LABEL}
    if not directory.is_dir():
        return by_class
    label_to_class = {v: k for k, v in CLASS_LABEL.items()}
    for path in sorted(directory.glob("*.png")):
        for label, klass in label_to_class.items():
            if f"_label{label}_" in path.name:
                by_class[klass].append(f"{AUGMENTED_DIR}/{path.name}")
                break
    return by_class


def _synthetic_candidate_files(root: Path, generator_entry: dict, klass: str) -> tuple[list[str], str]:
    """Real, on-disk candidate filenames for a generator's class, plus a precision tag.

    Tries the metrics-declared directory first (rerooted for a foreign-mount absolute path,
    covering both the flat `config.synthetic_dir`/`input_signature.filtered_dir` schema and the
    per-class `per_class.<class>.generated_dir` schema), then falls back to the canonical
    post-migration `experiments/diffusers/<gid>/generated_images/final/<class>/` layout.

    That canonical directory can contain more files than the scientifically-evaluated count
    recorded in the metrics JSON (observed for the sd21 family: metrics report a validated
    1361/class subset, the current directory holds 2722/class) — a pre-existing ambiguity this
    function does not silently resolve. Callers must check the returned precision tag.
    """
    from dataset_variant_registry import _reroot_under_project  # noqa: PLC0415

    gid = generator_entry["id"]
    metrics_rel = generator_entry.get("metrics")
    if metrics_rel:
        path = root / metrics_rel
        if path.is_file():
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                payload = {}
            per_class_dir = ((payload.get("per_class") or {}).get(klass) or {}).get("generated_dir")
            synth_dir = per_class_dir or (payload.get("config") or {}).get("synthetic_dir") or (payload.get("input_signature") or {}).get("filtered_dir")
            if synth_dir:
                base = _reroot_under_project(synth_dir, root)
                scan_dir = (base.parent / klass) if base.name in CLASS_LABEL else base
                if scan_dir.is_dir():
                    found = sorted(str(p.relative_to(root)) for p in scan_dir.glob("*.png"))
                    if found:
                        return found, "metrics_declared_directory"

    canonical = root / "experiments/diffusers" / gid / "generated_images/final" / klass
    if canonical.is_dir():
        found = sorted(str(p.relative_to(root)) for p in canonical.glob("*.png"))
        if found:
            return found, "canonical_directory_fallback_unverified_against_metrics_count"

    return [], "no_candidates_found"


def build_file_list(root: Path, variant: dict, generator_registry: dict | None = None) -> dict:
    """Return {"negative": [{"path":..., "source":...}, ...], "positive": [...]} for one variant."""
    real = _real_files_by_class(root) if variant.get("real_source") else {k: [] for k in CLASS_LABEL}
    augmented = _augmented_files_by_class(root) if variant.get("augmentation_source") else {k: [] for k in CLASS_LABEL}

    synthetic: dict[str, list[str]] = {k: [] for k in CLASS_LABEL}
    gid = variant.get("synthetic_generator_id")
    if gid and variant.get("synthetic_count_by_class"):
        if generator_registry is None:
            from dataset_variant_registry import load_generator_registry  # noqa: PLC0415
            generator_registry = load_generator_registry(root)
        entry = next((g for g in generator_registry["generators"] if g["id"] == gid), None)
        if entry is None:
            raise ValueError(f"generator {gid} referenced by variant {variant['dataset_variant_id']} not found in registry")
        for klass, k_count in variant["synthetic_count_by_class"].items():
            candidates, precision = _synthetic_candidate_files(root, entry, klass)
            if len(candidates) < k_count:
                raise ValueError(f"variant {variant['dataset_variant_id']}: only {len(candidates)} synthetic files on disk for "
                                  f"{gid}/{klass} ({precision}), need {k_count}")
            sig = deterministic_sample_signature(candidates, k_count, seed=variant.get("seed", 42))
            synthetic[klass] = sig["picked"]
            variant.setdefault("_resolved_synthetic_signature", {})[klass] = {
                **{k: v for k, v in sig.items() if k != "picked"}, "file_source_precision": precision}

    files = {}
    for klass in CLASS_LABEL:
        entries = [{"path": p, "source": "real"} for p in real.get(klass, [])]
        entries += [{"path": p, "source": "augmented"} for p in augmented.get(klass, [])]
        entries += [{"path": p, "source": "synthetic"} for p in synthetic.get(klass, [])]
        files[klass] = entries
    return files


def dataset_manifest_signature(file_list: dict) -> str:
    import hashlib
    flat = []
    for klass in sorted(file_list):
        for entry in sorted(file_list[klass], key=lambda e: e["path"]):
            flat.append(f"{klass}|{entry['source']}|{entry['path']}")
    return hashlib.sha256("\n".join(flat).encode("utf-8")).hexdigest()


def write_dataset_manifest(root: Path, variant: dict, file_list: dict, out_path: Path) -> dict:
    payload = {
        "schema_version": 1,
        "dataset_variant_id": variant["dataset_variant_id"],
        "counts": {k: len(v) for k, v in file_list.items()},
        "signature": dataset_manifest_signature(file_list),
        "files": file_list,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    return payload
