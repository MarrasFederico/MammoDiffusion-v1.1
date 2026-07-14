"""Validation-only discovery, seed ensembles and patient-level comparison utilities."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import classifier_metrics as metrics
    from .classifier_statistics import holm_correction, paired_stratified_bootstrap
    from .downstream_protocol import ARCHITECTURES, CONDITIONS, SEEDS, atomic_json, logical_experiments, load_protocol
except ImportError:
    import classifier_metrics as metrics
    from classifier_statistics import holm_correction, paired_stratified_bootstrap
    from downstream_protocol import ARCHITECTURES, CONDITIONS, SEEDS, atomic_json, logical_experiments, load_protocol

PUBLICATION_RESULTS = Path("results/publication_v2/downstream")


def result_dir(root: Path, architecture: str, condition: str, seed: int) -> Path:
    return Path(root) / PUBLICATION_RESULTS / architecture / condition / f"seed_{int(seed)}"


def discover_experiments(root: Path) -> list[dict[str, Any]]:
    """Inspect only publication_v2; legacy paths are never traversed."""
    output = []
    for job in logical_experiments():
        directory = result_dir(root, job["architecture"], job["condition"], job["seed"])
        required = [directory / "configuration.json", directory / "dataset_summary.json",
                    directory / "validation_predictions.csv", directory / "validation_metrics.json"]
        output.append({**job, "directory": str(directory), "complete": all(path.is_file() for path in required),
                       "missing": [path.name for path in required if not path.is_file()]})
    return output


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    output, keys = [], set()
    for row in rows:
        key = str(row.get("patient_id")), str(row.get("image_id"))
        if key in keys: raise ValueError(f"duplicate prediction key: {key}")
        keys.add(key); probability = float(row["probability"])
        if not math.isfinite(probability) or not 0 <= probability <= 1: raise ValueError("probabilities must be finite in [0, 1]")
        output.append({"patient_id": key[0], "image_id": key[1], "label": int(row["label"]), "probability": probability})
    return sorted(output, key=lambda row: (row["patient_id"], row["image_id"]))


def align_seed_predictions(per_seed_rows: Mapping[int, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    if set(per_seed_rows) != set(SEEDS): raise ValueError("ensemble requires exactly seeds 17, 42 and 73")
    canonical = None
    for seed in SEEDS:
        rows = list(per_seed_rows[seed]); keys = [(row["patient_id"], row["image_id"], int(row["label"])) for row in rows]
        if len(keys) != len(set((p, i) for p, i, _ in keys)): raise ValueError("duplicate seed predictions")
        if canonical is None: canonical = keys
        elif keys != canonical: raise ValueError("seed predictions have missing or inconsistent keys/labels")
    return [{"patient_id": row["patient_id"], "image_id": row["image_id"], "label": int(row["label"]),
             "probability": float(mean(float(per_seed_rows[seed][index]["probability"]) for seed in SEEDS))}
            for index, row in enumerate(per_seed_rows[SEEDS[0]])]


def aggregate_patient(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows: groups.setdefault(str(row["patient_id"]), []).append(row)
    output = []
    for patient_id, group in sorted(groups.items()):
        labels = {int(row["label"]) for row in group}
        if len(labels) != 1: raise ValueError(f"patient {patient_id} has inconsistent labels")
        output.append({"patient_id": patient_id, "label": labels.pop(),
                       "probability": float(mean(float(row["probability"]) for row in group)), "n_images": len(group)})
    return output


def patient_bootstrap_intervals(rows: Sequence[Mapping[str, Any]], *, iterations: int, seed: int,
                                threshold: float) -> dict[str, dict[str, float]]:
    """Stratified resampling of patients, never independent resampling of their images."""
    by_class = {label: [row for row in rows if int(row["label"]) == label] for label in (0, 1)}
    if not all(by_class.values()): raise ValueError("patient bootstrap requires both classes")
    rng = np.random.default_rng(int(seed)); values: dict[str, list[float]] = {}
    for _ in range(int(iterations)):
        sample = []
        for label in (0, 1):
            indices = rng.choice(len(by_class[label]), len(by_class[label]), replace=True)
            sample.extend(by_class[label][int(index)] for index in indices)
        report = metrics.full_report([row["label"] for row in sample], [row["probability"] for row in sample], threshold)
        for name in ("pr_auc", "roc_auc", "brier_score", "ece", "sensitivity_recall", "specificity", "balanced_accuracy"):
            values.setdefault(name, []).append(float(report[name]))
    return {name: {"percentile_2_5": float(np.percentile(metric_values, 2.5)),
                   "percentile_97_5": float(np.percentile(metric_values, 97.5))}
            for name, metric_values in values.items()}


def build_validation_ensemble(root: Path, architecture: str, condition: str) -> dict[str, Any]:
    summaries = [json.loads((result_dir(root, architecture, condition, seed) / "dataset_summary.json").read_text()) for seed in SEEDS]
    validation_descriptors = {(row.get("validation_manifest"), row.get("validation_signature")) for row in summaries}
    if len(validation_descriptors) != 1 or next(iter(validation_descriptors))[0] is None:
        raise ValueError("all seeds must use the same validation manifest and signature")
    per_seed = {seed: _read_predictions(result_dir(root, architecture, condition, seed) / "validation_predictions.csv")
                for seed in SEEDS}
    image_rows = align_seed_predictions(per_seed); patient_rows = aggregate_patient(image_rows)
    report = metrics.full_report([row["label"] for row in patient_rows], [row["probability"] for row in patient_rows])
    protocol = load_protocol(root)
    confidence_intervals = patient_bootstrap_intervals(
        patient_rows, iterations=int(protocol["evaluation"]["confidence_intervals"]["iterations"]),
        seed=int(protocol["evaluation"]["confidence_intervals"]["seed"]), threshold=float(report["threshold"]))
    seed_reports = []
    for seed in SEEDS:
        patient_seed = aggregate_patient(per_seed[seed])
        seed_reports.append({"seed": seed, **metrics.full_report([row["label"] for row in patient_seed],
                            [row["probability"] for row in patient_seed])})
    variability = {name: {"mean": mean(float(row[name]) for row in seed_reports),
                          "standard_deviation": stdev(float(row[name]) for row in seed_reports)}
                   for name in ("pr_auc", "roc_auc", "brier_score", "ece")}
    output = Path(root) / PUBLICATION_RESULTS / "ensembles" / architecture / condition
    _write_csv(output / "validation_predictions.csv", image_rows)
    _write_csv(output / "patient_level_predictions.csv", patient_rows)
    payload = {"architecture": architecture, "condition": condition, "seeds": list(SEEDS),
               "method": "mean_probability", "metrics": report, "confidence_intervals": confidence_intervals,
               "seed_metrics": seed_reports,
               "seed_variability": variability, "patient_level": True, "split": "validation",
               "validation_manifest": next(iter(validation_descriptors))[0],
               "validation_signature": next(iter(validation_descriptors))[1]}
    atomic_json(output / "ensemble_metrics.json", payload)
    return payload


def build_all_validation_ensembles(root: Path) -> list[dict[str, Any]]:
    return [build_validation_ensemble(root, architecture, condition)
            for architecture in ARCHITECTURES for condition in CONDITIONS]


def compare_validation(root: Path, ensembles: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    protocol = load_protocol(root)
    by_key = {(row["architecture"], row["condition"]): row for row in (ensembles or build_all_validation_ensembles(root))}
    comparisons, p_values = [], {}
    for architecture in ARCHITECTURES:
        for left, right in protocol["evaluation"]["primary_comparisons_per_architecture"]:
            left_rows = _read_patient_rows(Path(root) / PUBLICATION_RESULTS / "ensembles" / architecture / left / "patient_level_predictions.csv")
            right_rows = _read_patient_rows(Path(root) / PUBLICATION_RESULTS / "ensembles" / architecture / right / "patient_level_predictions.csv")
            if [row["patient_id"] for row in left_rows] != [row["patient_id"] for row in right_rows]:
                raise ValueError("patient alignment differs between conditions")
            labels = [row["label"] for row in left_rows]
            if labels != [row["label"] for row in right_rows]: raise ValueError("patient labels differ between conditions")
            comparison = paired_stratified_bootstrap(labels, [row["probability"] for row in left_rows],
                [row["probability"] for row in right_rows], metrics.pr_auc,
                n_bootstrap=int(protocol["evaluation"]["confidence_intervals"]["iterations"]),
                seed=int(protocol["evaluation"]["confidence_intervals"]["seed"]))
            comparison_id = f"{architecture}:{left}_vs_{right}"; p_values[comparison_id] = comparison["p_value_two_sided"]
            comparisons.append({"comparison_id": comparison_id, "architecture": architecture,
                                "condition_a": left, "condition_b": right, "metric": "pr_auc", **comparison})
    payload = {"split": "validation", "primary_metric": "pr_auc", "patient_level": True,
               "ensembles": list(by_key.values()), "comparisons": comparisons, "holm_correction": holm_correction(p_values)}
    atomic_json(Path(root) / "results/publication_v2/downstream/validation_comparison.json", payload)
    return payload


def _read_patient_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [{"patient_id": row["patient_id"], "label": int(row["label"]),
                 "probability": float(row["probability"])} for row in csv.DictReader(stream)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True); fields = list(rows[0]) if rows else ["patient_id"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return path


__all__ = ["PUBLICATION_RESULTS", "aggregate_patient", "align_seed_predictions", "build_all_validation_ensembles",
           "build_validation_ensemble", "compare_validation", "discover_experiments", "result_dir"]
