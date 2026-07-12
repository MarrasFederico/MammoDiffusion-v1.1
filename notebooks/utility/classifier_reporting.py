"""Persist notebook-visible classifier diagnostics under the canonical results tree.

This module is deliberately independent from locked-test code.  Plotting is lazy so plan mode
still works in minimal environments.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def atomic_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)
    return path


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    return path


def canonical_results_dirs(base: Path) -> dict[str, Path]:
    dirs = {name: base / name for name in ("config", "dataset", "figures", "interpretability", "logs", "notebook")}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def persist_dataset_summary(base: Path, train_rows: list[dict], validation_rows: list[dict], manifest: dict) -> dict:
    dirs = canonical_results_dirs(base)
    counts = {}
    for row in train_rows:
        key = (str(row.get("source", "unknown")), int(row["label"]))
        counts[key] = counts.get(key, 0) + 1
    rows = [{"source": source, "label": label, "count": count} for (source, label), count in sorted(counts.items())]
    payload = {"train_samples": len(train_rows), "validation_samples": len(validation_rows),
               "validation_real_only": all(r.get("source") == "real" for r in validation_rows),
               "source_by_class": rows, "manifest_signature": manifest.get("signature")}
    write_csv(dirs["dataset"] / "dataset_summary.csv", rows)
    atomic_json(dirs["dataset"] / "dataset_summary.json", payload)
    return payload


def save_placeholder_figure(path: Path, title: str, message: str) -> Path:
    """Save an honest placeholder when results do not exist yet; never fabricate metrics."""
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.axis("off")
    ax.set_title(title); ax.text(.5, .5, message, ha="center", va="center", wrap=True)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


def persist_plan_figures(base: Path, has_synthetic: bool, has_augmented: bool) -> list[str]:
    figures = canonical_results_dirs(base)["figures"]
    specs = {
        "dataset_class_distribution.png": "Generated after dataset resolution",
        "dataset_source_distribution.png": "Generated after dataset resolution",
        "dataset_source_by_class.png": "Generated after dataset resolution",
        "training_samples_grid.png": "Deterministic samples are generated in train/auto mode",
    }
    paths = [str(save_placeholder_figure(figures / name, name.removesuffix('.png').replace('_', ' ').title(), msg))
             for name, msg in specs.items()]
    return paths


ATTRIBUTION_METHODS = {
    "resnet50": "Grad-CAM on final convolutional feature map",
    "maxvit512": "gradient-weighted spatial attribution on final MaxViT stage",
    "mammofm": "gradient-weighted token/spatial attribution selected from the resolved backbone",
    "raddino": "gradient-weighted patch-token attribution (not a plain attention map)",
}


def attribution_manifest(base: Path, architecture: str, sample_kind: str, samples: list[dict]) -> Path:
    if sample_kind not in {"real", "synthetic", "augmented"}:
        raise ValueError(sample_kind)
    return atomic_json(base / "interpretability" / sample_kind / "manifest.json", {
        "schema_version": 1, "architecture": architecture, "method": ATTRIBUTION_METHODS[architecture],
        "gradient_based": True, "ensemble": "mean of normalized seed heatmaps", "samples": samples,
        "test_access": False,
    })
