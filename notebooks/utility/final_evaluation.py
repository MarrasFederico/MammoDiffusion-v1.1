"""Transparent final-evaluation guard, protocol snapshot and report generation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .downstream_protocol import ARCHITECTURES, CONDITIONS, SEEDS, atomic_json, load_selected_generators
except ImportError:
    from downstream_protocol import ARCHITECTURES, CONDITIONS, SEEDS, atomic_json, load_selected_generators


REQUIRED_CHECKLIST = (
    "Generator benchmark completed", "Generators selected", "24 downstream jobs completed",
    "8 validation ensembles completed", "Validation analysis finalized",
    "Checkpoints selected using validation only", "Decision thresholds selected using validation only",
    "Statistical comparisons declared", "Final evaluation dataset identified", "No further model selection will occur",
)


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
    return atomic_json(Path(root) / "results/final_evaluation_protocol.json", payload)


def run_final_evaluation(root: Path, *, run_final_evaluation: bool, checklist: Mapping[str, bool],
                         evaluator: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]) -> Mapping[str, Any]:
    """Run an explicitly supplied evaluator only after the visible notebook guard passes."""
    require_final_evaluation_opt_in(run_final_evaluation, checklist)
    snapshot_path = Path(root) / "results/final_evaluation_protocol.json"
    if not snapshot_path.is_file(): raise FileNotFoundError("save results/final_evaluation_protocol.json first")
    return evaluator(Path(root), json.loads(snapshot_path.read_text()))


def generate_publication_report(root: Path) -> Path:
    """Regenerate a factual report; missing phases are labelled, never fabricated."""
    root = Path(root)
    selections = load_selected_generators(root, required=False)
    benchmark = _optional_json(root / "results/publication_v2/generator_benchmark/benchmark_manifest.json")
    validation = _optional_json(root / "results/publication_v2/downstream/validation_comparison.json")
    final_results = _optional_json(root / "results/publication_v2/final_evaluation/results.json")
    sections = [
        ("Dataset", _render(benchmark, "dataset", "Not yet evaluated")),
        ("Generator candidates", _render(benchmark, "candidate_audits", "Not yet evaluated")),
        ("Generative benchmark", _render(benchmark, "metrics", "Not yet evaluated")),
        ("Fine-tuned selection", selections.get("finetuned") if selections else "Not yet evaluated"),
        ("From-scratch selection", selections.get("from_scratch") if selections else "Not yet evaluated"),
        ("RAW/FILTERED metrics", _render(benchmark, "representations", "Not yet evaluated")),
        ("Synthetic duplication", _render(benchmark, "synthetic_duplication", "Not yet evaluated")),
        ("Train memorization", _render(benchmark, "train_memorization", "Not yet evaluated")),
        ("Validation similarity", _render(benchmark, "validation_similarity", "Not yet evaluated")),
        ("Downstream validation", _render(validation, "ensembles", "Not yet evaluated")),
        ("Seed variability", _render(validation, "seed_variability", "Not yet evaluated")),
        ("Final evaluation", json.dumps(final_results, indent=2) if final_results else "Not yet evaluated"),
        ("Interpretability", _render(final_results or validation, "interpretability", "Not yet evaluated")),
        ("Efficiency", _render(benchmark, "efficiency", "Not yet evaluated")),
        ("Limitations", "Historical reuse of the internal test prevents describing it as an untouched independent confirmation. External or newly untouched validation is preferred. RAD-DINO is not mammography-specific; positive validation samples are limited; generator filtering may alter diversity."),
        ("Conclusions", "Not yet evaluated" if not validation else "Conclusions are restricted to validation until an honestly identified final evaluation is performed."),
    ]
    lines = ["# MammoDiffusion publication report", "", "Generated from repository result files; absent phases are reported explicitly.", ""]
    for title, content in sections: lines.extend([f"## {title}", "", str(content), ""])
    output = root / "results/publication_report.md"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    (root / "results/tables").mkdir(parents=True, exist_ok=True)
    (root / "results/figures").mkdir(parents=True, exist_ok=True)
    return output


def _optional_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _render(payload: Mapping[str, Any] | None, key: str, fallback: str) -> str:
    return json.dumps(payload[key], indent=2) if payload and key in payload else fallback


__all__ = ["REQUIRED_CHECKLIST", "generate_publication_report", "require_final_evaluation_opt_in",
           "run_final_evaluation", "save_protocol_snapshot"]
