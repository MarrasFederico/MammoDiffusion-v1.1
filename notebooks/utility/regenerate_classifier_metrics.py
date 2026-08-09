"""Regenerate held-out-test reports from saved predictions and validation reports.

No model, image, checkpoint or GPU is opened. Decision thresholds and the
specificity-target operating points are read from validation reports and are
kept fixed for every test metric and every bootstrap replicate. Use
``rebuild_classifier_reports.py`` to reconstruct those reports from validation
CSVs first and rebuild the complete validation-to-test chain.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

try:
    from . import classifier_metrics as metrics
    from .classifier_analysis import (
        ENSEMBLE_RESULTS,
        TEST_ENSEMBLE_RESULTS,
        _read_predictions,
        _write_csv,
        aggregate_patient,
        align_seed_predictions,
        patient_bootstrap_intervals,
        result_dir,
    )
    from .classifier_protocol import (
        ARCHITECTURES,
        CONDITIONS,
        SEEDS,
        atomic_json,
        load_protocol,
    )
    from .classifier_statistics import holm_correction, paired_stratified_bootstrap
except ImportError:
    import classifier_metrics as metrics
    from classifier_analysis import (
        ENSEMBLE_RESULTS,
        TEST_ENSEMBLE_RESULTS,
        _read_predictions,
        _write_csv,
        aggregate_patient,
        align_seed_predictions,
        patient_bootstrap_intervals,
        result_dir,
    )
    from classifier_protocol import ARCHITECTURES, CONDITIONS, SEEDS, atomic_json, load_protocol
    from classifier_statistics import holm_correction, paired_stratified_bootstrap


THRESHOLD_INDEPENDENT = ("roc_auc", "pr_auc", "brier_score", "ece")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frozen_report(rows: Sequence[Mapping[str, Any]], validation_report: Mapping[str, Any]) -> dict[str, Any]:
    specificity = validation_report.get("sensitivity_at_specificity_0_90") or {}
    if validation_report.get("threshold") is None or specificity.get("threshold") is None:
        raise ValueError("validation report does not contain both frozen operating points")
    return metrics.full_report(
        [int(row["label"]) for row in rows],
        [float(row["probability"]) for row in rows],
        float(validation_report["threshold"]),
        split="test",
        specificity_threshold=float(specificity["threshold"]),
    )


def _preserve_threshold_independent(report: dict[str, Any], previous: Mapping[str, Any]) -> dict[str, Any]:
    """Keep established threshold-independent values byte-for-byte after verifying them."""
    for name in THRESHOLD_INDEPENDENT:
        if previous.get(name) is None:
            continue
        if not math.isclose(float(previous[name]), float(report[name]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"threshold-independent metric changed unexpectedly: {name}")
        report[name] = previous[name]
    return report


def regenerate(source_root: Path, output_root: Path | None = None, *,
               bootstrap_iterations: int | None = None) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root or source_root).resolve()
    protocol = load_protocol(source_root)
    iterations = int(bootstrap_iterations or protocol["evaluation"]["confidence_intervals"]["iterations"])
    bootstrap_seed = int(protocol["evaluation"]["confidence_intervals"]["seed"])
    seed_report_count = 0
    for architecture in ARCHITECTURES:
        for condition in CONDITIONS:
            for seed in SEEDS:
                source = result_dir(source_root, architecture, condition, seed)
                validation = _read_json(source / "validation_metrics.json")
                test_rows = _read_predictions(source / "test_predictions.csv")
                current_path = source / "test_metrics.json"
                current = _read_json(current_path) if current_path.is_file() else {}
                report = _preserve_threshold_independent(_frozen_report(test_rows, validation), current)
                destination = result_dir(output_root, architecture, condition, seed) / "test_metrics.json"
                atomic_json(destination, report)
                seed_report_count += 1

    ensembles: list[dict[str, Any]] = []
    patient_rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for architecture in ARCHITECTURES:
        for condition in CONDITIONS:
            per_seed = {
                seed: _read_predictions(
                    result_dir(source_root, architecture, condition, seed) / "test_predictions.csv"
                )
                for seed in SEEDS
            }
            image_rows = align_seed_predictions(per_seed)
            patient_rows = aggregate_patient(image_rows)
            patient_rows_by_key[(architecture, condition)] = patient_rows
            validation_path = source_root / ENSEMBLE_RESULTS / architecture / condition / "ensemble_metrics.json"
            validation_payload = _read_json(validation_path)
            current_path = source_root / TEST_ENSEMBLE_RESULTS / architecture / condition / "ensemble_metrics.json"
            current = _read_json(current_path) if current_path.is_file() else {"metrics": {}}
            report = _preserve_threshold_independent(
                _frozen_report(patient_rows, validation_payload["metrics"]), current["metrics"]
            )
            confidence_intervals = patient_bootstrap_intervals(
                patient_rows, iterations=iterations, seed=bootstrap_seed,
                threshold=float(report["threshold"]), split="test",
                specificity_threshold=float(report["sensitivity_at_specificity_0_90"]["threshold"]),
            )
            seed_metric_reports = []
            for seed in SEEDS:
                validation_seed = _read_json(
                    result_dir(source_root, architecture, condition, seed) / "validation_metrics.json"
                )
                seed_metric_reports.append({
                    "seed": seed,
                    **_frozen_report(aggregate_patient(per_seed[seed]), validation_seed),
                })
            variability = {
                name: {
                    "mean": mean(float(row[name]) for row in seed_metric_reports),
                    "standard_deviation": stdev(float(row[name]) for row in seed_metric_reports),
                }
                for name in THRESHOLD_INDEPENDENT
            }
            destination = output_root / TEST_ENSEMBLE_RESULTS / architecture / condition
            _write_csv(destination / "ensemble_predictions.csv", image_rows)
            _write_csv(destination / "patient_level_predictions.csv", patient_rows)
            _write_csv(destination / "seed_metrics.csv", seed_metric_reports)
            payload = {
                "architecture": architecture, "condition": condition,
                "seeds": list(SEEDS), "method": "mean_probability", "metrics": report,
                "confidence_intervals": confidence_intervals, "seed_metrics": seed_metric_reports,
                "seed_variability": variability, "patient_level": True, "split": "test",
                "threshold_source": validation_path.relative_to(source_root).as_posix(),
                "specificity_threshold_source": validation_path.relative_to(source_root).as_posix(),
            }
            atomic_json(destination / "ensemble_metrics.json", payload)
            ensembles.append(payload)

    comparisons, p_values = [], {}
    for architecture in ARCHITECTURES:
        for left, right in protocol["evaluation"]["primary_comparisons_per_architecture"]:
            left_rows = patient_rows_by_key[(architecture, left)]
            right_rows = patient_rows_by_key[(architecture, right)]
            if [row["patient_id"] for row in left_rows] != [row["patient_id"] for row in right_rows]:
                raise ValueError("patient alignment differs between test conditions")
            labels = [row["label"] for row in left_rows]
            comparison = paired_stratified_bootstrap(
                labels, [row["probability"] for row in left_rows],
                [row["probability"] for row in right_rows], metrics.pr_auc,
                n_bootstrap=iterations, seed=bootstrap_seed,
            )
            comparison_id = f"{architecture}:{left}_vs_{right}"
            p_values[comparison_id] = comparison["p_value_two_sided"]
            comparisons.append({
                "comparison_id": comparison_id, "architecture": architecture,
                "condition_a": left, "condition_b": right, "metric": "pr_auc", **comparison,
            })
    atomic_json(output_root / "results/4_final_evaluation/results.json", {
        "split": "test", "primary_metric": "pr_auc", "patient_level": True,
        "ensembles": ensembles, "comparisons": comparisons,
        "holm_correction": holm_correction(p_values),
    })
    return {
        "source": "saved test_predictions.csv with frozen validation reports",
        "bootstrap_iterations": iterations, "bootstrap_seed": bootstrap_seed,
        "seed_reports_written": seed_report_count,
        "ensemble_reports_written": len(ensembles),
        "thresholds_source": "canonical validation results",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = regenerate(
        args.source_root, args.output_root,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
