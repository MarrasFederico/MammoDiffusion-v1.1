"""Simple opt-in, frozen-threshold and overwrite guards for final evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from .classifier_protocol import ARCHITECTURES, CONDITIONS, SEEDS, logical_experiments
except ImportError:
    from classifier_protocol import ARCHITECTURES, CONDITIONS, SEEDS, logical_experiments


EXPECTED_PATIENT_COUNT = 438


class FinalEvaluationDatasetAdapter:
    """Minimal interface for the already declared held-out RSNA test split."""

    def load_manifest(self, root: Path):
        raise RuntimeError("Final evaluation dataset adapter must implement load_manifest()")

    def build_dataset(self, root: Path, manifest):
        raise RuntimeError("Final evaluation dataset adapter must implement build_dataset()")

    def describe(self) -> Mapping[str, Any]:
        raise RuntimeError("Final evaluation dataset adapter must implement describe()")


def expected_experiment_ids() -> list[str]:
    return [row["experiment_id"] for row in logical_experiments()]


def expected_prediction_files() -> list[str]:
    files = []
    for architecture in ARCHITECTURES:
        for condition in CONDITIONS:
            for seed in SEEDS:
                files.append(
                    f"results/3_classifiers/seed_runs/{architecture}/{condition}/seed_{seed}/test_predictions.csv"
                )
    return files


def _threshold_pair(path: Path, *, nested: bool = False) -> dict[str, float]:
    import json

    if not path.is_file():
        raise FileNotFoundError(f"validation thresholds are missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {}) if nested else payload
    specificity = metrics.get("sensitivity_at_specificity_0_90") or {}
    if metrics.get("threshold") is None or specificity.get("threshold") is None:
        raise ValueError(f"validation artifact lacks frozen operating points: {path}")
    values = {
        "decision_threshold": float(metrics["threshold"]),
        "specificity_0_90_threshold": float(specificity["threshold"]),
    }
    if any(not 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError(f"validation threshold outside [0, 1]: {path}")
    return values


def frozen_validation_thresholds(root: Path) -> dict[str, dict[str, float]]:
    """Read the 24 seed and 8 ensemble operating points from canonical validation results."""
    root = Path(root)
    frozen: dict[str, dict[str, float]] = {}
    for architecture in ARCHITECTURES:
        for condition in CONDITIONS:
            for seed in SEEDS:
                path = (root / "results/3_classifiers/seed_runs" / architecture / condition
                        / f"seed_{seed}/validation_metrics.json")
                frozen[f"seed:{architecture}:{condition}:{seed}"] = _threshold_pair(path)
            path = (root / "results/3_classifiers/validation_ensembles" / architecture
                    / condition / "ensemble_metrics.json")
            frozen[f"ensemble:{architecture}:{condition}"] = _threshold_pair(path, nested=True)
    return frozen


def require_final_evaluation_opt_in(root: Path, *, run_final_evaluation: bool,
                                    overwrite_test_predictions: bool = False) -> dict[str, Any]:
    """Validate the single opt-in and refuse accidental prediction replacement."""
    if run_final_evaluation is not True:
        raise PermissionError("RUN_FINAL_EVALUATION must be exactly True")
    thresholds = frozen_validation_thresholds(root)
    existing = [
        relative for relative in expected_prediction_files()
        if (Path(root) / relative).exists()
    ]
    if existing and overwrite_test_predictions is not True:
        raise FileExistsError(
            "test predictions already exist; set OVERWRITE_TEST_PREDICTIONS=True separately "
            "only after explicit approval"
        )
    return {
        "expected_patient_count": EXPECTED_PATIENT_COUNT,
        "selected_experiments": expected_experiment_ids(),
        "validation_thresholds_frozen": thresholds,
    }


def run_final_evaluation(root: Path, *, run_final_evaluation: bool,
                         overwrite_test_predictions: bool = False,
                         adapter: FinalEvaluationDatasetAdapter | None = None,
                         evaluator: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]] | None = None) -> Mapping[str, Any]:
    """Run only after explicit opt-in, frozen thresholds, adapter and evaluator are present."""
    plan = require_final_evaluation_opt_in(
        root, run_final_evaluation=run_final_evaluation,
        overwrite_test_predictions=overwrite_test_predictions,
    )
    if adapter is None:
        raise RuntimeError("No final evaluation dataset adapter is configured")
    if evaluator is None:
        raise RuntimeError("No final evaluation function is configured")
    manifest = adapter.load_manifest(Path(root))
    dataset = adapter.build_dataset(Path(root), manifest)
    return evaluator(dataset, plan)


__all__ = [
    "EXPECTED_PATIENT_COUNT", "FinalEvaluationDatasetAdapter",
    "expected_experiment_ids", "expected_prediction_files", "frozen_validation_thresholds",
    "require_final_evaluation_opt_in",
    "run_final_evaluation",
]
