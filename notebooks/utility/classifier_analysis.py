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
    from .classifier_protocol import ARCHITECTURES, CONDITIONS, SEEDS, atomic_json, logical_experiments, load_protocol
except ImportError:
    import classifier_metrics as metrics
    from classifier_statistics import holm_correction, paired_stratified_bootstrap
    from classifier_protocol import ARCHITECTURES, CONDITIONS, SEEDS, atomic_json, logical_experiments, load_protocol

PUBLICATION_RESULTS = Path("results/publication_v2/classifiers")
ENSEMBLE_RESULTS = Path("results/publication_v2/classifier_ensembles")

ARCHITECTURE_DISPLAY_NAMES = {
    "maxvit512": "MaxViT-512",
    "mammofm": "Mammo-FM",
}
CONDITION_DISPLAY_NAMES = {
    "real_only": "Real only",
    "real_augmented": "Real + traditional augmentation",
    "real_plus_best_finetuned_positive": "Real + selected fine-tuned synthetic positives",
    "real_plus_best_fromscratch_positive": "Real + selected from-scratch synthetic positives",
}


def result_dir(root: Path, architecture: str, condition: str, seed: int) -> Path:
    return Path(root) / PUBLICATION_RESULTS / architecture / condition / f"seed_{int(seed)}"


def discover_experiments(root: Path) -> list[dict[str, Any]]:
    """Inspect only publication_v2; legacy paths are never traversed."""
    output = []
    for job in logical_experiments():
        directory = result_dir(root, job["architecture"], job["condition"], job["seed"])
        configuration = directory / "configuration.json"; dataset = directory / "dataset_summary.json"
        predictions = directory / "validation_predictions.csv"; validation_metrics = directory / "validation_metrics.json"
        checkpoint = any(path.suffix in {".pt", ".pth", ".keras", ".h5"} for path in directory.glob("checkpoint_best.*"))
        training_complete = configuration.is_file() and dataset.is_file() and checkpoint
        output.append({**job, "directory": str(directory), "training complete": training_complete,
                       "validation predictions present": predictions.is_file(),
                       "validation metrics present": validation_metrics.is_file(), "checkpoint present": checkpoint,
                       "complete": training_complete and predictions.is_file() and validation_metrics.is_file(),
                       "missing": [name for name, present in (("configuration.json", configuration.is_file()),
                                   ("dataset_summary.json", dataset.is_file()), ("checkpoint", checkpoint),
                                   ("validation_predictions.csv", predictions.is_file()),
                                   ("validation_metrics.json", validation_metrics.is_file())) if not present]})
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
        if not rows: raise ValueError("seed predictions must not be empty")
        if len(keys) != len(set((p, i) for p, i, _ in keys)): raise ValueError("duplicate seed predictions")
        if any(not math.isfinite(float(row["probability"])) or not 0 <= float(row["probability"]) <= 1 for row in rows):
            raise ValueError("probabilities must be finite in [0, 1]")
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
    output = Path(root) / ENSEMBLE_RESULTS / architecture / condition
    _write_csv(output / "ensemble_predictions.csv", image_rows)
    _write_csv(output / "patient_level_predictions.csv", patient_rows)
    _write_csv(output / "seed_metrics.csv", seed_reports)
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
            left_rows = _read_patient_rows(Path(root) / ENSEMBLE_RESULTS / architecture / left / "patient_level_predictions.csv")
            right_rows = _read_patient_rows(Path(root) / ENSEMBLE_RESULTS / architecture / right / "patient_level_predictions.csv")
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
    atomic_json(Path(root) / "results/publication_v2/classifiers/validation_comparison.json", payload)
    return payload


def ensemble_metric_table(ensembles: Sequence[Mapping[str, Any]]):
    import pandas as pd
    rows = []
    baseline = {(row["architecture"]): float(row["metrics"]["pr_auc"]) for row in ensembles
                if row["condition"] == "real_only"}
    for row in ensembles:
        report = row["metrics"]; intervals = row["confidence_intervals"]
        rows.append({"architecture": row["architecture"], "condition": row["condition"],
                     "pr_auc": report["pr_auc"], "pr_auc_ci_low": intervals["pr_auc"]["percentile_2_5"],
                     "pr_auc_ci_high": intervals["pr_auc"]["percentile_97_5"], "roc_auc": report["roc_auc"],
                     "brier": report["brier_score"], "ece": report["ece"],
                     "delta_pr_auc_vs_real_only": float(report["pr_auc"]) - baseline[row["architecture"]],
                     "seed_pr_auc_mean": row["seed_variability"]["pr_auc"]["mean"],
                     "seed_pr_auc_std": row["seed_variability"]["pr_auc"]["standard_deviation"]})
    return pd.DataFrame(rows)


def plot_ensemble_overview(ensembles: Sequence[Mapping[str, Any]]):
    import matplotlib.pyplot as plt
    table = ensemble_metric_table(ensembles)
    if len(table) != 8: raise ValueError("exactly eight logical ensembles are required")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [f"{ARCHITECTURE_DISPLAY_NAMES.get(row.architecture, row.architecture)}\n"
              f"{CONDITION_DISPLAY_NAMES.get(row.condition, row.condition)}"
              for row in table.itertuples()]
    lower = table["pr_auc"] - table["pr_auc_ci_low"]; upper = table["pr_auc_ci_high"] - table["pr_auc"]
    axes[0].errorbar(range(8), table["pr_auc"], yerr=[lower, upper], fmt="o"); axes[0].set_xticks(range(8), labels, rotation=90); axes[0].set_title("Validation PR-AUC with intervals")
    heat = table.pivot(index="architecture", columns="condition", values="pr_auc")
    heat_conditions = [CONDITION_DISPLAY_NAMES.get(value, value) for value in heat.columns]
    heat_architectures = [ARCHITECTURE_DISPLAY_NAMES.get(value, value) for value in heat.index]
    image = axes[1].imshow(heat.values, aspect="auto", cmap="viridis"); axes[1].set_xticks(range(len(heat.columns)), heat_conditions, rotation=90); axes[1].set_yticks(range(len(heat.index)), heat_architectures); axes[1].set_title("Architecture × condition PR-AUC"); figure.colorbar(image, ax=axes[1])
    figure.suptitle("Classifier validation ensembles | Patient-level mean of seeds 17, 42 and 73",
                    fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93)); return figure


def plot_ensemble_curves(root: Path, ensembles: Sequence[Mapping[str, Any]]):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(16, 10))
    for row in ensembles:
        path = Path(root) / ENSEMBLE_RESULTS / row["architecture"] / row["condition"] / "patient_level_predictions.csv"
        values = _read_patient_rows(path); labels = np.asarray([item["label"] for item in values]); probabilities = np.asarray([item["probability"] for item in values])
        order = np.argsort(-probabilities); ordered = labels[order]; tp, fp = np.cumsum(ordered == 1), np.cumsum(ordered == 0)
        recall = tp / max(1, int((labels == 1).sum())); precision = tp / np.maximum(tp + fp, 1); fpr = fp / max(1, int((labels == 0).sum()))
        arch_index = 0 if row["architecture"] == ARCHITECTURES[0] else 1
        label = CONDITION_DISPLAY_NAMES.get(row["condition"], row["condition"])
        axes[arch_index, 0].plot(recall, precision, label=label); axes[arch_index, 1].plot(fpr, recall, label=label)
        edges = np.linspace(0, 1, 6); observed, predicted = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (probabilities >= low) & (probabilities <= high if high == 1 else probabilities < high)
            if mask.any(): predicted.append(float(probabilities[mask].mean())); observed.append(float(labels[mask].mean()))
        axes[arch_index, 2].plot(predicted, observed, marker="o", label=label)
    for index, architecture in enumerate(ARCHITECTURES):
        architecture_label = ARCHITECTURE_DISPLAY_NAMES.get(architecture, architecture)
        axes[index, 0].set_title(f"{architecture_label} validation PR curves"); axes[index, 1].set_title(f"{architecture_label} validation ROC curves"); axes[index, 2].set_title(f"{architecture_label} validation calibration")
        axes[index, 2].plot([0, 1], [0, 1], "--", color="black")
        for axis in axes[index]: axis.legend(fontsize=7)
    figure.suptitle("Validation ensemble curves | Patient-level mean of seeds 17, 42 and 73",
                    fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95)); return figure


def _read_patient_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [{"patient_id": row["patient_id"], "label": int(row["label"]),
                 "probability": float(row["probability"])} for row in csv.DictReader(stream)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True); fields = list(rows[0]) if rows else ["patient_id"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return path


__all__ = ["ENSEMBLE_RESULTS", "PUBLICATION_RESULTS", "aggregate_patient", "align_seed_predictions", "build_all_validation_ensembles",
           "build_validation_ensemble", "compare_validation", "discover_experiments", "ensemble_metric_table",
           "plot_ensemble_curves", "plot_ensemble_overview", "result_dir"]
