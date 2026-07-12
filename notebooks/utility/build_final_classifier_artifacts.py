"""Build reproducible audit artefacts and normalize reusable legacy predictions.

This command performs no model loading or inference.  It is safe to rerun after
new classifier artefacts become available.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from final_classifier_evaluation import (  # noqa: E402
    build_experiment_registry,
    build_test_coverage_table,
    build_test_dataset_manifest,
    canonical_test_prediction_paths,
    compute_binary_metrics,
    content_signature,
    patient_ids_hash,
    strict_json_dumps,
    strict_jsonable,
    standardize_prediction_dataframe,
    write_prediction_manifest,
)


def normalize_mammofm_predictions(experiment: dict) -> None:
    legacy = experiment.get("legacy_test_predictions_path")
    if not legacy:
        return
    canonical_paths = canonical_test_prediction_paths(experiment)
    output = ROOT / canonical_paths["test_predictions_path"]
    output_manifest = ROOT / canonical_paths["test_predictions_manifest_path"]
    if output.is_file() and output_manifest.is_file():
        # Canonical outputs may be verified reinference results.  Artifact regeneration must
        # never downgrade or overwrite them with legacy-normalized predictions.
        return
    if output.exists() or output_manifest.exists():
        raise RuntimeError(
            f"Partial canonical prediction cache for {experiment['experiment_id']}: "
            f"CSV={output.is_file()}, manifest={output_manifest.is_file()}"
        )
    legacy_path = ROOT / legacy
    checkpoint = ROOT / experiment["checkpoint_path"]
    metrics = ROOT / experiment["validation_metrics_path"]
    canonical_test = pd.read_csv(ROOT / "data/processed/metadata/test.csv")
    raw = pd.read_csv(legacy_path)
    normalized = standardize_prediction_dataframe(
        raw,
        experiment=experiment,
        threshold=experiment["validation_threshold"],
        threshold_method=experiment["validation_threshold_method"],
    )
    expected = canonical_test[["patient_id", "image_id", "label", "processed_path"]].copy()
    expected["patient_id"] = expected["patient_id"].astype(str)
    normalized["patient_id"] = normalized["patient_id"].astype(str)
    joined = normalized.merge(expected, on="patient_id", how="left", suffixes=("", "_canonical"), validate="one_to_one")
    if joined["label"].isna().any() or not joined["y_true"].eq(joined["label"]).all():
        raise ValueError(f"Legacy labels/test cohort mismatch for {experiment['experiment_id']}")
    if set(joined["patient_id"]) != set(expected["patient_id"]):
        raise ValueError(f"Legacy patient set mismatch for {experiment['experiment_id']}")
    normalized["path"] = joined["processed_path"].values
    normalized["image_id"] = joined["image_id_canonical"].astype(str).values
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False)
    test_dataset_manifest_path = ROOT / "results/final_evaluation/test_dataset_manifest.json"
    if not test_dataset_manifest_path.is_file():
        test_dataset_manifest = build_test_dataset_manifest(
            ROOT / "data/processed/metadata/test.csv", project_root=ROOT,
            preprocessing={"view": "MLO", "resolution": 512, "grayscale": True, "right_breast_mirrored": True},
            include_image_signatures=True,
        )
        test_dataset_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        test_dataset_manifest_path.write_text(strict_json_dumps(test_dataset_manifest, indent=2) + "\n", encoding="utf-8")
    original_manifest_candidates = [
        legacy_path.with_suffix(".manifest.json"),
        legacy_path.with_name("test_predictions_manifest.json"),
    ]
    original_manifest = next((path for path in original_manifest_candidates if path.is_file()), None)
    checkpoint_link_verified = False
    if original_manifest:
        try:
            original_payload = json.loads(original_manifest.read_text(encoding="utf-8"))
            checkpoint_link_verified = original_payload.get("checkpoint_signature") == content_signature(checkpoint)
        except (OSError, ValueError, json.JSONDecodeError):
            checkpoint_link_verified = False
    provenance_level = "verified_native" if checkpoint_link_verified else "legacy_normalized_unverified"
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment["experiment_id"],
        "architecture": experiment["architecture"],
        "dataset_variant": experiment["training_dataset_variant"],
        "synthetic_source": experiment["synthetic_source"],
        "training_mode": experiment["training_mode"],
        "training_notebook": experiment["training_notebook"],
        "checkpoint_path": experiment["checkpoint_path"],
        "checkpoint_signature": content_signature(checkpoint),
        "validation_metrics_path": experiment["validation_metrics_path"],
        "validation_metrics_signature": content_signature(metrics),
        "validation_threshold": experiment["validation_threshold"],
        "threshold_method": experiment["validation_threshold_method"],
        "test_csv": "data/processed/metadata/test.csv",
        "test_csv_signature": content_signature(ROOT / "data/processed/metadata/test.csv"),
        "test_dataset_manifest_signature": content_signature(test_dataset_manifest_path),
        "patient_ids_hash": patient_ids_hash(normalized["patient_id"]),
        "n_patients": len(normalized),
        "n_positive": int(normalized["y_true"].sum()),
        "n_negative": int((normalized["y_true"] == 0).sum()),
        "preprocessing": {"view": "MLO", "resolution": 512, "grayscale": True, "right_breast_mirrored": True},
        "model_config": {"legacy_predictions_normalized_without_reinference": True},
        "legacy_prediction_signature": content_signature(legacy_path),
        "legacy_source_path": legacy_path.relative_to(ROOT).as_posix(),
        "provenance_level": provenance_level,
        "pipeline_schema_version": 1,
        "checkpoint_link_verified": checkpoint_link_verified,
        "original_prediction_manifest_path": original_manifest.relative_to(ROOT).as_posix() if original_manifest else None,
        "test_used_for_selection": False,
    }
    write_prediction_manifest(output_manifest, manifest, output)
    computed = compute_binary_metrics(normalized["y_true"], normalized["y_score"], experiment["validation_threshold"])
    (output.parent / "test_metrics.json").write_text(strict_json_dumps(computed, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([computed | {"confusion_matrix": json.dumps(computed["confusion_matrix"])}]).to_csv(output.parent / "test_metrics.csv", index=False)
    pd.DataFrame(computed["confusion_matrix"], index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(output.parent / "confusion_matrix.csv")


def main() -> None:
    registry_path = ROOT / "configs/final_classifier_registry.json"
    registry = build_experiment_registry(registry_path)
    for experiment in registry:
        normalize_mammofm_predictions(experiment)
    coverage = build_test_coverage_table(registry, ROOT)
    output_dir = ROOT / "results/final_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_dir / "classifier_test_coverage.csv", index=False)
    (output_dir / "classifier_test_coverage.json").write_text(
        strict_json_dumps(coverage.to_dict("records"), indent=2) + "\n",
        encoding="utf-8",
    )
    leaderboard = coverage.rename(columns={"training_dataset_variant": "dataset_variant"}).copy()
    leaderboard.to_csv(output_dir / "validation_leaderboard.csv", index=False)
    (output_dir / "validation_leaderboard.json").write_text(
        strict_json_dumps(leaderboard.to_dict("records"), indent=2) + "\n",
        encoding="utf-8",
    )
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt
    auc_rows = leaderboard[leaderboard["validation_roc_auc"].notna()].sort_values("validation_roc_auc")
    axis = auc_rows.plot.barh(x="display_name", y="validation_roc_auc", legend=False, figsize=(10, 9))
    axis.set_xlabel("Validation ROC-AUC")
    axis.figure.tight_layout()
    axis.figure.savefig(figures / "validation_auc_comparison.png", dpi=300)
    plt.close(axis.figure)
    pr_rows = leaderboard[leaderboard["validation_pr_auc"].notna()].sort_values("validation_pr_auc")
    axis = pr_rows.plot.barh(x="display_name", y="validation_pr_auc", legend=False, figsize=(9, 6))
    axis.set_xlabel("Validation PR-AUC")
    axis.figure.tight_layout()
    axis.figure.savefig(figures / "validation_pr_auc_comparison.png", dpi=300)
    plt.close(axis.figure)
    test_manifest = build_test_dataset_manifest(
        ROOT / "data/processed/metadata/test.csv",
        project_root=ROOT,
        preprocessing={"view": "MLO", "resolution": 512, "grayscale": True, "right_breast_mirrored": True},
        include_image_signatures=True,
    )
    (output_dir / "test_dataset_manifest.json").write_text(
        strict_json_dumps(test_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Normalized Mammo-FM predictions and audited {len(registry)} canonical experiments.")


if __name__ == "__main__":
    main()
