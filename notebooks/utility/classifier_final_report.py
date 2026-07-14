"""Deterministic v2 locked-test aggregation and final scientific report.

This module is never imported by preflight/status. It accepts only the signed prediction manifest
created after the scientific lock and aggregates multiple images to one mean probability per
patient before confidence intervals or paired comparisons are computed.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np

from classifier_metrics import full_report, metrics_at_threshold, pr_auc, roc_auc, seed_stability
from classifier_pipeline_contracts import (
    PIPELINE_NAMESPACE, atomic_json, code_revision, sha256_file, signed_payload,
    verify_signed_payload,
)
from classifier_statistics import delong_test, holm_correction, mcnemar_test, paired_stratified_bootstrap


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def aggregate_patient_rows(rows: list[dict], probability_field: str = "prob_ensemble") -> list[dict]:
    """Mean image probability per patient; inconsistent labels/thresholds are rejected."""
    grouped: dict[str, list[dict]] = {}
    seen = set()
    for row in rows:
        key = (str(row["patient_id"]), str(row["image_id"]))
        if key in seen:
            raise ValueError(f"duplicate locked patient/image key: {key}")
        seen.add(key); grouped.setdefault(key[0], []).append(row)
    result = []
    for patient_id, images in sorted(grouped.items()):
        labels = {int(row["label"]) for row in images}
        thresholds = {float(row["threshold"]) for row in images}
        if len(labels) != 1 or len(thresholds) != 1:
            raise ValueError(f"patient {patient_id} has inconsistent label or locked threshold")
        probabilities = [float(row[probability_field]) for row in images]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError(f"patient {patient_id} has invalid probabilities")
        probability, threshold = float(np.mean(probabilities)), thresholds.pop()
        result.append({"patient_id": patient_id, "label": labels.pop(), "probability": probability,
                       "predicted_label": int(probability >= threshold), "threshold": threshold,
                       "n_images": len(images)})
    if not result:
        raise ValueError("locked prediction table is empty")
    return result


def bootstrap_ci(labels, probabilities, metric_fn, *, n_bootstrap=2000, seed=42) -> dict:
    labels = np.asarray(labels, dtype=int); probabilities = np.asarray(probabilities, dtype=float)
    positive, negative = np.where(labels == 1)[0], np.where(labels == 0)[0]
    if not len(positive) or not len(negative):
        raise ValueError("confidence interval requires both patient classes")
    rng = np.random.RandomState(seed); values = []
    for _ in range(n_bootstrap):
        indices = np.concatenate((rng.choice(positive, len(positive), replace=True),
                                  rng.choice(negative, len(negative), replace=True)))
        values.append(float(metric_fn(labels[indices], probabilities[indices])))
    low, high = np.percentile(values, [2.5, 97.5])
    return {"estimate": float(metric_fn(labels, probabilities)), "ci_95_low": float(low),
            "ci_95_high": float(high), "n_bootstrap": n_bootstrap, "seed": seed,
            "unit": "patient", "stratified": True}


def calibration_bins(labels, probabilities, n_bins=10) -> list[dict]:
    labels = np.asarray(labels, dtype=int); probabilities = np.asarray(probabilities, dtype=float)
    output = []
    for index, (low, high) in enumerate(zip(np.linspace(0, 1, n_bins + 1)[:-1],
                                            np.linspace(0, 1, n_bins + 1)[1:])):
        mask = (probabilities >= low) & (probabilities < high if high < 1 else probabilities <= high)
        if np.any(mask):
            output.append({"bin": index, "low": float(low), "high": float(high), "n": int(mask.sum()),
                           "mean_probability": float(probabilities[mask].mean()),
                           "observed_prevalence": float(labels[mask].mean())})
    return output


def _aligned(a: list[dict], b: list[dict]) -> tuple[list[int], list[float], list[float], list[int], list[int]]:
    by_a, by_b = ({row["patient_id"]: row for row in values} for values in (a, b))
    if set(by_a) != set(by_b):
        raise ValueError("finalist patient sets are not identical")
    keys = sorted(by_a); labels = [int(by_a[key]["label"]) for key in keys]
    if labels != [int(by_b[key]["label"]) for key in keys]:
        raise ValueError("finalist patient labels differ")
    return (labels, [float(by_a[key]["probability"]) for key in keys],
            [float(by_b[key]["probability"]) for key in keys],
            [int(by_a[key]["predicted_label"]) for key in keys],
            [int(by_b[key]["predicted_label"]) for key in keys])


def build_report(root: Path, *, n_bootstrap: int = 2000, write_figures: bool = True) -> dict:
    root = Path(root); output = root / "results/final_evaluation_v2"
    prediction_manifest = json.loads((output / "locked_test_predictions_manifest.json").read_text())
    verify_signed_payload(prediction_manifest)
    if prediction_manifest.get("artifact_type") != "classifier_locked_predictions":
        raise ValueError("not a v2 locked prediction manifest")
    lock_payload = json.loads((output / "EXPERIMENT_MATRIX_LOCKED").read_text())
    if prediction_manifest.get("lock_signature") != lock_payload.get("lock_signature"):
        raise ValueError("locked predictions belong to another scientific lock")
    if not (output / "LOCKED_TEST_COMPLETE").is_file():
        raise RuntimeError("locked inference is not complete")
    completion_path = output / "FINAL_AGGREGATION_COMPLETE"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text()); verify_signed_payload(completion)
        existing = json.loads((output / "final_report.json").read_text()); verify_signed_payload(existing)
        existing_bootstrap = next((ci["n_bootstrap"] for row in existing["locked_test_evaluation"]["metrics"]
                                   for ci in row["confidence_intervals"].values()), n_bootstrap)
        if completion.get("final_report_signature") != existing.get("signature") or \
           existing.get("locked_predictions_manifest_signature") != prediction_manifest.get("signature") or \
           existing_bootstrap != n_bootstrap:
            raise RuntimeError("final aggregation is immutable and does not match the requested inputs")
        return existing

    patient_tables, metric_rows, provenance = {}, [], []
    for item in prediction_manifest["outputs"]:
        path = root / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"locked prediction table changed: {item['path']}")
        source_rows = read_csv(path)
        patients = aggregate_patient_rows(source_rows); patient_tables[item["experiment_id"]] = patients
        labels = [row["label"] for row in patients]; probabilities = [row["probability"] for row in patients]
        threshold = patients[0]["threshold"]
        metrics = full_report(labels, probabilities, threshold)
        cis = {"roc_auc": bootstrap_ci(labels, probabilities, roc_auc, n_bootstrap=n_bootstrap),
               "pr_auc": bootstrap_ci(labels, probabilities, pr_auc, n_bootstrap=n_bootstrap),
               "sensitivity": bootstrap_ci(labels, probabilities,
                    lambda y, p: metrics_at_threshold(y, p, threshold)["sensitivity_recall"], n_bootstrap=n_bootstrap),
               "specificity": bootstrap_ci(labels, probabilities,
                    lambda y, p: metrics_at_threshold(y, p, threshold)["specificity"], n_bootstrap=n_bootstrap)}
        seed_metrics = {}
        for seed in (17, 42, 73):
            field = f"prob_seed_{seed}"
            if source_rows and field in source_rows[0]:
                seed_patients = aggregate_patient_rows(source_rows, field)
                seed_metrics[str(seed)] = full_report([row["label"] for row in seed_patients],
                                                       [row["probability"] for row in seed_patients], threshold)
        stability = ({metric: seed_stability([seed_metrics[str(seed)][metric] for seed in (17, 42, 73)])
                      for metric in ("roc_auc", "pr_auc")} if len(seed_metrics) == 3 else {})
        metric_rows.append({"panel": item["panel"], "experiment_id": item["experiment_id"],
                            "n_patients": len(patients), **metrics, "confidence_intervals": cis,
                            "calibration_bins": calibration_bins(labels, probabilities),
                            "seed_metrics": seed_metrics, "seed_stability": stability})
        provenance.append({"experiment_id": item["experiment_id"], "prediction_sha256": item["sha256"],
                           "source_path": item["path"], "aggregation": "mean_probability_per_patient"})
        atomic_csv(output / "patient_predictions" / f"{item['experiment_id']}.csv", patients)

    panels = {row["experiment_id"]: row["panel"] for row in metric_rows}
    comparisons = []
    for left, right in itertools.combinations(sorted(patient_tables), 2):
        labels, a, b, pa, pb = _aligned(patient_tables[left], patient_tables[right])
        family = "primary" if panels[left] == panels[right] == "primary" else "secondary"
        comparisons.append({"comparison_id": f"{left}__vs__{right}", "family": family,
            "left": left, "right": right,
            "roc_auc_delong": delong_test(labels, a, b),
            "pr_auc_bootstrap": paired_stratified_bootstrap(labels, a, b, pr_auc, n_bootstrap=n_bootstrap),
            "mcnemar": mcnemar_test(labels, pa, pb)})
    for family in ("primary", "secondary"):
        rows = [row for row in comparisons if row["family"] == family]
        for analysis in ("roc_auc_delong", "pr_auc_bootstrap", "mcnemar"):
            field = "p_value" if analysis != "pr_auc_bootstrap" else "p_value_two_sided"
            correction = holm_correction({row["comparison_id"]: row[analysis][field] for row in rows}) if rows else holm_correction({})
            for row in rows:
                row.setdefault("holm_within_family", {})[analysis] = {
                    "adjusted_p_value": correction["adjusted_p_values"][row["comparison_id"]],
                    "reject_null": correction["reject_null"][row["comparison_id"]]}

    artifact = signed_payload({"schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE,
        "artifact_type": "classifier_final_locked_report", "code_revision": code_revision(root),
        "lock_signature": lock_payload.get("lock_signature"), "unit_of_analysis": "patient",
        "locked_predictions_manifest_signature": prediction_manifest["signature"],
        "validation_selection": "Frozen by the signed Stage-2 panel manifest before locked inference.",
        "locked_test_evaluation": {"metrics": metric_rows, "comparisons": comparisons},
        "analysis_scope": {"descriptive": ["metrics", "calibration", "confusion_matrix"],
                           "inferential": ["patient_bootstrap", "DeLong", "McNemar", "Holm"]},
        "provenance": provenance,
        "limitations": ["Internal locked test only; external validation remains required.",
                        "Multiplicity is controlled within preregistered comparison families."],
        "scientific_selection_complete": True, "final_aggregation_complete": True})
    atomic_json(output / "final_report.json", artifact)
    flat_rows = [{key: value for key, value in row.items() if key not in (
        "confidence_intervals", "calibration_bins", "seed_metrics", "seed_stability")}
                 for row in metric_rows]
    atomic_csv(output / "final_metrics.csv", flat_rows)
    atomic_json(output / "final_comparisons.json", signed_payload({
        "schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE,
        "artifact_type": "classifier_final_comparisons", "comparisons": comparisons,
        "lock_signature": lock_payload.get("lock_signature")}))
    markdown = ["# MammoDiffusion classifier-matrix v2 — locked report", "",
        "## Validation selection", "", artifact["validation_selection"], "",
        "## Locked test evaluation", "", "All values below use one aggregated probability per patient.", "",
        "| Panel | Experiment | Patients | ROC-AUC | PR-AUC | Brier |", "|---|---|---:|---:|---:|---:|"]
    for row in metric_rows:
        markdown.append(f"| {row['panel']} | `{row['experiment_id']}` | {row['n_patients']} | {row['roc_auc']:.6f} | {row['pr_auc']:.6f} | {row['brier_score']:.6f} |")
    markdown += ["", "## Descriptive analysis", "", "Metrics, confusion counts, calibration bins, and Brier scores are recorded in `final_report.json`.",
                 "", "## Inferential analysis", "", "Paired patient-level comparisons and within-family Holm corrections are recorded in `final_comparisons.json`.",
                 "", "## Limitations", ""] + [f"- {item}" for item in artifact["limitations"]]
    (output / "final_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    if write_figures:
        _write_figures(output, patient_tables)
    atomic_json(output / "FINAL_AGGREGATION_COMPLETE", signed_payload({
        "schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE,
        "artifact_type": "classifier_final_aggregation_completion", "final_aggregation_complete": True,
        "final_report_signature": artifact["signature"], "lock_signature": lock_payload.get("lock_signature")}))
    return artifact


def _write_figures(output: Path, tables: dict[str, list[dict]]) -> None:
    import matplotlib.pyplot as plt
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, rows in sorted(tables.items()):
        bins = calibration_bins([r["label"] for r in rows], [r["probability"] for r in rows])
        ax.plot([r["mean_probability"] for r in bins], [r["observed_prevalence"] for r in bins], marker="o", label=name)
    ax.plot([0, 1], [0, 1], "--", color="black"); ax.set(xlabel="Mean probability", ylabel="Observed prevalence")
    ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(figures / "patient_calibration.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, rows in sorted(tables.items()):
        ax.hist([row["probability"] for row in rows], bins=np.linspace(0, 1, 11), alpha=.35, label=name)
    ax.set(xlabel="Patient-level probability", ylabel="Patients"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(figures / "patient_probability_distributions.png", dpi=160); plt.close(fig)
