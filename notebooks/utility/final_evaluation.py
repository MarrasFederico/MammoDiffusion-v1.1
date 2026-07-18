"""Transparent final-evaluation guard, simple adapter interface and Markdown report."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .classifier_protocol import ARCHITECTURES, CONDITIONS, SEEDS, atomic_json, load_selected_generators
except ImportError:
    from classifier_protocol import ARCHITECTURES, CONDITIONS, SEEDS, atomic_json, load_selected_generators


REQUIRED_CHECKLIST = (
    "Generator benchmark completed", "Generators selected", "24 downstream jobs completed",
    "8 validation ensembles completed", "Validation analysis finalized",
    "Checkpoints selected using validation only", "Decision thresholds selected using validation only",
    "Statistical comparisons declared", "Final evaluation dataset identified", "No further model selection will occur",
)


class FinalEvaluationDatasetAdapter:
    """Minimal interface to implement only after an approved final dataset is chosen."""

    def load_manifest(self, root: Path):
        raise RuntimeError("Final evaluation dataset adapter must implement load_manifest()")

    def build_dataset(self, root: Path, manifest):
        raise RuntimeError("Final evaluation dataset adapter must implement build_dataset()")

    def describe(self) -> Mapping[str, Any]:
        raise RuntimeError("Final evaluation dataset adapter must implement describe()")


def final_dataset_status(adapter: FinalEvaluationDatasetAdapter | None = None) -> dict[str, str]:
    return {"Final evaluation dataset": "held-out test set", "Split provenance": "fixed before any test access",
            "Configured final adapter": type(adapter).__name__ if adapter is not None else "Not yet configured"}


def require_final_evaluation_opt_in(run_final_evaluation: bool, checklist: Mapping[str, bool]) -> None:
    if run_final_evaluation is not True:
        raise PermissionError("RUN_FINAL_EVALUATION must be exactly True")
    missing = [item for item in REQUIRED_CHECKLIST if checklist.get(item) is not True]
    if missing: raise RuntimeError(f"final-evaluation checklist incomplete: {missing}")


def save_protocol_snapshot(root: Path, *, selected_generators: Mapping[str, Any], seed_checkpoints: Mapping[str, Any],
                           validation_thresholds: Mapping[str, float], planned_comparisons: Sequence[Mapping[str, Any]],
                           final_evaluation_dataset_identifier: str, notes: str = "") -> Path:
    if not final_evaluation_dataset_identifier.strip(): raise ValueError("final evaluation dataset identifier is required")
    payload = {"selected_generators": {key: selected_generators[key] for key in ("finetuned", "from_scratch")},
               "architectures": list(ARCHITECTURES), "conditions": list(CONDITIONS),
               "seed_checkpoints": dict(seed_checkpoints),
               "ensemble_definitions": {f"{a}:{c}": {"seeds": list(SEEDS), "method": "mean_probability"}
                                        for a in ARCHITECTURES for c in CONDITIONS},
               "validation_selected_thresholds": dict(validation_thresholds),
               "planned_statistical_comparisons": list(planned_comparisons),
               "final_evaluation_dataset_identifier": final_evaluation_dataset_identifier, "notes": notes}
    return atomic_json(Path(root) / "results/4_final_evaluation_protocol.json", payload)


def run_final_evaluation(root: Path, *, run_final_evaluation: bool, checklist: Mapping[str, bool],
                         adapter: FinalEvaluationDatasetAdapter | None = None,
                         evaluator: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] | None = None) -> Mapping[str, Any]:
    """Access final data only after explicit opt-in, a complete checklist and a real adapter."""
    require_final_evaluation_opt_in(run_final_evaluation, checklist)
    if adapter is None:
        raise RuntimeError("No final evaluation dataset adapter is configured. Configure the held-out test-set adapter before running the final evaluation.")
    snapshot_path = Path(root) / "results/4_final_evaluation_protocol.json"
    if not snapshot_path.is_file(): raise FileNotFoundError("save the final-evaluation protocol snapshot first")
    manifest = adapter.load_manifest(Path(root)); dataset = adapter.build_dataset(Path(root), manifest)
    if evaluator is None: raise RuntimeError("No final evaluation function is configured for the selected dataset adapter")
    return evaluator(dataset, json.loads(snapshot_path.read_text()))


__all__ = ["FinalEvaluationDatasetAdapter", "REQUIRED_CHECKLIST", "final_dataset_status",
           "require_final_evaluation_opt_in", "run_final_evaluation", "save_protocol_snapshot"]
