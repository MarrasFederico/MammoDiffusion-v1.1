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
    return atomic_json(Path(root) / "results/final_evaluation_protocol.json", payload)


def run_final_evaluation(root: Path, *, run_final_evaluation: bool, checklist: Mapping[str, bool],
                         adapter: FinalEvaluationDatasetAdapter | None = None,
                         evaluator: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] | None = None) -> Mapping[str, Any]:
    """Access final data only after explicit opt-in, a complete checklist and a real adapter."""
    require_final_evaluation_opt_in(run_final_evaluation, checklist)
    if adapter is None:
        raise RuntimeError("No final evaluation dataset adapter is configured. Configure the held-out test-set adapter before running the final evaluation.")
    snapshot_path = Path(root) / "results/final_evaluation_protocol.json"
    if not snapshot_path.is_file(): raise FileNotFoundError("save the final-evaluation protocol snapshot first")
    manifest = adapter.load_manifest(Path(root)); dataset = adapter.build_dataset(Path(root), manifest)
    if evaluator is None: raise RuntimeError("No final evaluation function is configured for the selected dataset adapter")
    return evaluator(dataset, json.loads(snapshot_path.read_text()))


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> str:
    if not rows: return "Not yet evaluated"
    columns = list(columns or dict.fromkeys(key for row in rows for key in row))
    def clean(value: Any) -> str:
        if value is None or value == "": return "Not yet evaluated"
        if isinstance(value, float): return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(clean(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def format_metric_table(rows: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None) -> str:
    if not rows: return "Not yet evaluated"
    if isinstance(rows, Mapping): rows = [{"metric": key, "value": value} for key, value in rows.items()]
    return _markdown_table(list(rows))


def format_generator_comparison(rows: Sequence[Mapping[str, Any]] | None) -> str:
    if not rows: return "Not yet evaluated"
    preferred = ("generator_id", "condition", "family", "role", "eligible_for_selection", "raddino_kid",
                 "raddino_precision", "raddino_recall", "raddino_density", "raddino_coverage", "raddino_fid",
                 "inception_kid", "inception_fid", "ms_ssim_diversity", "synthetic_duplicate_rate",
                 "train_memorization_rate", "validation_nearest_neighbour_distance")
    return _markdown_table(rows, [name for name in preferred if any(name in row for row in rows)])


def format_classifier_results(payload: Mapping[str, Any] | None) -> str:
    if not payload: return "Not yet evaluated"
    rows = []
    for ensemble in payload.get("ensembles", []):
        report = ensemble.get("metrics", {})
        rows.append({"architecture": ensemble.get("architecture"), "condition": ensemble.get("condition"),
                     "split": ensemble.get("split", "validation"), "pr_auc": report.get("pr_auc"),
                     "roc_auc": report.get("roc_auc"), "brier": report.get("brier_score"), "ece": report.get("ece")})
    return _markdown_table(rows)


def format_limitations() -> str:
    return "\n".join((
        "- Generator selection used a human-approved post-benchmark methodological amendment "
        "(Option B). Original gate outcomes are preserved. Classifier findings must be interpreted "
        "in light of this amendment.",
        "- RAD-DINO is radiology-specific rather than mammography-specific.",
        "- The positive validation reference set is limited and FID is descriptive.",
        "- Generator filtering may alter measured diversity.",
    ))


def format_generator_benchmark_outcome(root: Path) -> str:
    """Distinguish the original zero-eligible outcome from the amendment, entirely from the data.

    The policy, the original outcome and the count of safety-eligible candidates are read from the
    active amendment and the committed selection evidence; nothing is hardcoded, so the report stays
    correct if the candidate count or amendment version changes.
    """
    protocol = _optional_json(root / "configs/generator_benchmark_protocol.json") or {}
    amendment_relative = protocol.get("active_amendment", "configs/generator_benchmark_protocol_amendment_v1.json")
    amendment = _optional_json(root / str(amendment_relative))
    if not amendment:
        return "Not yet evaluated"
    original = amendment.get("original_outcome", {})
    measured = original.get("official_candidates_measured", "Not yet evaluated")
    eligible = original.get("eligible_under_original_gates", "Not yet evaluated")
    evidence = _optional_json(root / "configs/generator_selection_evidence_v1.json") or {}
    safety_results = evidence.get("amended_safety_gate_results", [])
    safety_eligible = (sum(1 for row in safety_results if row.get("amended_safety_gate_eligible"))
                       if safety_results else "Not yet evaluated")
    return "\n".join((
        "**Original protocol:**",
        f"- Official candidates measured: {measured}",
        f"- Eligible under original gates: {eligible}",
        "",
        "**Post-benchmark amendment:**",
        f"- Policy: Option {amendment.get('selected_policy', '?')} "
        f"({amendment.get('status', 'approved_post_benchmark')})",
        f"- Safety-eligible official candidates: {safety_eligible}",
        "- Coverage balanced-point and pHash-only rate are descriptive metrics, not binary gates.",
    ))


def format_generator_selection(selections: Mapping[str, Any] | None) -> str:
    """Report the amended selection from the content-aware schema (never selection_basis)."""
    if not selections:
        return "Not yet evaluated"
    identity = selections.get("selection_identity", {})
    rows = []
    for family in ("finetuned", "from_scratch"):
        record = identity.get(family, {})
        rows.append({
            "family": family,
            "generator_id": record.get("generator_id", selections.get(family)),
            "descriptive_family_rank": record.get("descriptive_family_rank"),
            "primary_metric": record.get("primary_metric", selections.get("primary_metric")),
            "primary_metric_value": record.get("primary_metric_value"),
            "benchmark_run_id": selections.get("benchmark_run_id"),
            "post_benchmark_amendment": selections.get("post_benchmark_amendment"),
        })
    notes = [
        _markdown_table(rows),
        "",
        f"- Active amendment: `{selections.get('active_amendment')}`",
        f"- Post-benchmark amendment: {selections.get('post_benchmark_amendment')}",
        f"- Test access: {selections.get('test_access')}",
    ]
    if selections.get("selection_notes"):
        notes.append(f"- Notes: {selections['selection_notes']}")
    return "\n".join(notes)


def collect_figure_references(root: Path) -> str:
    result_root = Path(root) / "results"
    figures = sorted(path.relative_to(Path(root)).as_posix() for path in result_root.rglob("*")
                     if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}) if result_root.exists() else []
    return "\n".join(f"- `{path}`" for path in figures) if figures else "Not yet evaluated"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file(): return []
    with path.open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def _optional_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def generate_publication_report(root: Path) -> Path:
    """Generate twelve readable publication sections without invented numeric placeholders."""
    root = Path(root); publication = root / "results"
    # Prefer the efficiency-corrected canonical summary; never the microsecond-per-image snapshot.
    corrected = publication / "generator_benchmark/generator_summary_corrected.csv"
    generator_rows = _read_csv(corrected if corrected.is_file()
                               else publication / "generator_benchmark/generator_summary.csv")
    selections = load_selected_generators(root, required=False)
    validation = _optional_json(publication / "classifiers/validation_comparison.json")
    final_results = _optional_json(publication / "final_evaluation/results.json")
    generator_protocol = _optional_json(root / "configs/generator_benchmark_protocol.json") or {}
    classifier_protocol = _optional_json(root / "configs/classifier_protocol.json") or {}
    questions = [{"question": "RQ1", "text": generator_protocol.get("study_question", "Not yet evaluated")},
                 *[{"question": key, "text": value} for key, value in classifier_protocol.get("research_questions", {}).items()]]
    sections = (
        ("1. Research questions", _markdown_table(questions)),
        ("2. Dataset and split", "Generator selection and classifier comparison use validation data only; final evaluation uses the held-out test set, fixed before any test access."),
        ("3. Generative models", format_generator_comparison(generator_rows)),
        ("4. Generator benchmark", format_generator_benchmark_outcome(root)),
        ("5. Generator selection", format_generator_selection(selections)),
        ("6. Classifiers", _markdown_table([{"architecture": name, "conditions": 4, "seeds": "17, 42, 73"} for name in ARCHITECTURES])),
        ("7. Validation results", format_classifier_results(validation)),
        ("8. Final evaluation status/results", format_classifier_results(final_results) if final_results else format_metric_table(final_dataset_status())),
        ("9. Interpretability", collect_figure_references(root)),
        ("10. Efficiency", format_metric_table([{key: row.get(key) for key in ("generator_id", "generation_seconds_per_image", "peak_vram_mb", "energy_kwh", "checkpoint_size_bytes", "efficiency_source", "efficiency_status")} for row in generator_rows] if generator_rows else None)),
        ("11. Limitations", format_limitations()),
        ("12. Conclusions", "Not yet evaluated" if not final_results else "Conclusions are limited to the explicitly configured final evaluation."),
    )
    lines = ["# MammoDiffusion publication report", "", "Missing phases are labelled `Not yet evaluated`; no numeric result is fabricated.", ""]
    for title, content in sections: lines.extend([f"## {title}", "", content or "Not yet evaluated", ""])
    output = publication / "publication_report.md"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8"); return output


__all__ = ["FinalEvaluationDatasetAdapter", "REQUIRED_CHECKLIST", "collect_figure_references", "final_dataset_status",
           "format_classifier_results", "format_generator_benchmark_outcome", "format_generator_comparison",
           "format_generator_selection", "format_limitations", "format_metric_table",
           "generate_publication_report", "require_final_evaluation_opt_in", "run_final_evaluation", "save_protocol_snapshot"]
