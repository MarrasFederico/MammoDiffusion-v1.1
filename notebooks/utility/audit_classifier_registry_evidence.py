#!/usr/bin/env python3
"""Attach verifiable evidence to classifier registry values without reading test metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from final_classifier_evaluation import content_signature, strict_json_dumps

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs/final_classifier_registry.json"
PREDICTION_SCORE_FIELDS = {
    "resnet50_01b_real_synth_partial": "prob_real_synth",
    "resnet50_01c_real_synth_full": "prob_real_synth_all_layers",
    "maxvit512_02b_real_synth_partial": "prob_real_synth_part",
    "maxvit512_02c_real_synth_full": "prob_real_synth_all_layers",
    "maxvit512_02g_fromscratch_synthetic_partial": "prob_fromscratch_sd_vae_partial",
    "maxvit512_02h_fromscratch_synthetic_full": "prob_fromscratch_sd_vae_full",
}


def source_field(payload: dict, candidates: tuple[str, ...]) -> str | None:
    return next((field for field in candidates if payload.get(field) is not None), None)


def enrich(experiment: dict) -> dict:
    item = dict(experiment)
    metrics_path = ROOT / str(item.get("validation_metrics_path") or "")
    metrics, signature = {}, None
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")); signature = content_signature(metrics_path)
    auc_field = source_field(metrics, ("roc_auc", "auc", "validation_roc_auc"))
    pr_field = source_field(metrics, ("pr_auc", "pr_auc_average_precision", "average_precision"))
    threshold_field = source_field(metrics, ("validation_threshold", "threshold_youden_from_val", "optimal_threshold_youden", "threshold"))
    relative_metrics = item.get("validation_metrics_path")
    def evidence(field: str | None, precision: str = "rounded_legacy") -> dict:
        return {"path": relative_metrics, "signature": signature, "field": field, "source_precision": precision,
                "verified": bool(field and signature)}
    item["validation_metrics_source"] = evidence(auc_field)
    item["validation_roc_auc_evidence"] = evidence(auc_field)
    item["validation_pr_auc_evidence"] = evidence(pr_field)
    item["validation_threshold_source"] = evidence(threshold_field)
    item["validation_threshold_evidence"] = evidence(threshold_field)
    predictions_path = ROOT / str(item.get("validation_predictions_path") or "")
    score_field = PREDICTION_SCORE_FIELDS.get(item["experiment_id"])
    if score_field and predictions_path.is_file():
        frame = pd.read_csv(predictions_path); y_true = frame["label_true"].astype(int); scores = frame[score_field].astype(float)
        fpr, tpr, thresholds = roc_curve(y_true, scores); best = int((tpr - fpr).argmax())
        reconstructed = {"validation_roc_auc": float(roc_auc_score(y_true, scores)),
                         "validation_pr_auc": float(average_precision_score(y_true, scores)),
                         "validation_threshold": float(thresholds[best])}
        compatible = all(item.get(field) is None or round(float(item[field]), 4) == round(value, 4)
                         for field, value in reconstructed.items() if field != "validation_pr_auc")
        reconstruction_evidence = {"path": item["validation_predictions_path"], "signature": content_signature(predictions_path),
            "field": score_field, "label_field": "label_true", "n_rows": len(frame),
            "method": "sklearn.metrics.roc_curve; argmax(tpr-fpr)", "source_precision": "full_recomputed",
            "compatible_with_rounded_registry": compatible}
        if compatible:
            item.update(reconstructed)
            item["validation_metrics_source"] = dict(reconstruction_evidence)
            item["validation_roc_auc_evidence"] = dict(reconstruction_evidence)
            item["validation_pr_auc_evidence"] = dict(reconstruction_evidence)
            item["validation_threshold_source"] = dict(reconstruction_evidence)
            item["validation_threshold_evidence"] = dict(reconstruction_evidence)
        else:
            item["validation_reconstruction_mismatch"] = {"reconstructed": reconstructed, "evidence": reconstruction_evidence}
    item["scientifically_eligible"] = bool(item.get("eligible_for_final_selection") or item.get("required_for_final_pipeline"))
    item["selected_by_validation"] = bool(item.get("required_for_final_pipeline"))
    checkpoint = ROOT / str(item.get("checkpoint_path") or "")
    item["checkpoint_evidence"] = {"path": item.get("checkpoint_path"), "signature": content_signature(checkpoint) if checkpoint.is_file() else None,
                                    "verified": checkpoint.is_file()}
    item["operationally_ready"] = bool(checkpoint.is_file() and metrics_path.is_file() and item.get("validation_threshold") is not None and (item.get("test_notebook") or item.get("test_predictions_path")))
    item["blocked_reason"] = None if item["operationally_ready"] else (item.get("exclusion_reason") or "missing operational artefacts")
    notebook = ROOT / str(item.get("training_notebook") or "")
    for field in ("training_mode", "synthetic_source"):
        item[f"{field}_evidence"] = {"path": item.get("training_notebook"), "signature": content_signature(notebook) if notebook.is_file() else None,
                                             "field": field, "verified": notebook.is_file()}
    return item


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--write", action="store_true"); args = parser.parse_args()
    payload = json.loads(REGISTRY.read_text(encoding="utf-8")); payload["experiments"] = [enrich(item) for item in payload["experiments"]]
    if args.write: REGISTRY.write_text(strict_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(strict_json_dumps({"experiments": len(payload["experiments"]), "unverified": [x["experiment_id"] for x in payload["experiments"] if not x["validation_metrics_source"].get("verified", x["validation_metrics_source"].get("compatible_with_rounded_registry", False))]}, indent=2))


if __name__ == "__main__": main()
