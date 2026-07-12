"""Create the locked final-evaluation notebooks from small, reviewable cells."""

import hashlib
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]


def write_notebook(relative_path, title, cells):
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}
    notebook["cells"] = [nbf.v4.new_markdown_cell(f"# {title}\n\nPipeline finale locked. Il validation seleziona; il test valuta. Il dry-run non carica modelli e non scrive artefatti scientifici.")] + [
        nbf.v4.new_code_cell(cell) if kind == "code" else nbf.v4.new_markdown_cell(cell) for kind, cell in cells
    ]
    for index, cell in enumerate(notebook["cells"]):
        stable_key = f"{relative_path}:{index}:{cell.cell_type}:{cell.source}".encode("utf-8")
        cell["id"] = hashlib.sha256(stable_key).hexdigest()[:12]
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)


COMMON_CONFIG = '''from pathlib import Path
import json, os, sys
import pandas as pd

def find_project_root(start=Path.cwd()):
    for candidate in [start, *start.parents]:
        if (candidate / "configs/final_classifier_registry.json").is_file():
            return candidate
    raise FileNotFoundError("Project root not found")

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "notebooks/utility"))
from final_classifier_evaluation import *

DRY_RUN = True
RECOMPUTE_TEST_PREDICTIONS = False
ALLOW_UNVERIFIED_LEGACY_PREDICTIONS = False
TEST_BATCH_SIZE = 8
TEST_NUM_WORKERS = 4
DEVICE = "auto"
PATIENT_AGGREGATION = "mean"
TEST_CSV = PROJECT_ROOT / "data/processed/metadata/test.csv"
TEST_DATASET_MANIFEST = PROJECT_ROOT / "results/final_evaluation/test_dataset_manifest.json"
REGISTRY_PATH = PROJECT_ROOT / "configs/final_classifier_registry.json"
'''


def classifier_cells(family, experiment_ids, loader_code):
    return [
        ("code", COMMON_CONFIG + f'''\nFAMILY = "{family}"
SUPPORTED_EXPERIMENT_IDS = {experiment_ids!r}
LOCKED_MANIFEST_PATH = PROJECT_ROOT / "results/final_evaluation/finalists_manifest.json"
if LOCKED_MANIFEST_PATH.is_file():
    # Scientific-only validation: the finalist selection can be frozen and valid even while some
    # finalists (in this or other families) are still operationally blocked. This notebook only
    # needs its OWN family's operationally-ready members; it must not be blocked by others.
    scientific_lock = validate_locked_finalists_manifest(LOCKED_MANIFEST_PATH, require_operational_complete=False)
    print("scientific_selection_complete:", scientific_lock.get("scientific_selection_complete"),
          "| final_aggregation_complete:", scientific_lock.get("final_aggregation_complete"),
          "| operational_blockers:", scientific_lock.get("operational_blockers"))
    ready_ids = {{x["experiment_id"] for x in scientific_lock["finalists"] if x.get("operationally_ready")}}
    EXPERIMENT_IDS = [eid for eid in SUPPORTED_EXPERIMENT_IDS if eid in ready_ids]
else:
    EXPERIMENT_IDS = SUPPORTED_EXPERIMENT_IDS
'''),
        ("code", '''registry = {x["experiment_id"]: x for x in build_experiment_registry(REGISTRY_PATH)}
experiments = [registry[eid] for eid in EXPERIMENT_IDS]
for exp in experiments:
    exp["test_csv"] = "data/processed/metadata/test.csv"
    checked = validate_locked_test_configuration(exp, PROJECT_ROOT)
    canonical_paths = canonical_test_prediction_paths(exp)
    output_dir = (PROJECT_ROOT / canonical_paths["test_predictions_path"]).parent
    print({
        "experiment": exp["experiment_id"], "checkpoint": exp["checkpoint_path"],
        "checkpoint_signature": checked["checkpoint_signature"], "validation_threshold": checked["validation_threshold"],
        "threshold_method": checked["threshold_method"], "test_n": len(pd.read_csv(TEST_CSV)),
        "cache_available": (output_dir / "test_predictions.csv").is_file(), "output": str(output_dir),
        "device": DEVICE, "dry_run": DRY_RUN,
    })
if DRY_RUN:
    print("DRY_RUN: nessun modello caricato, nessuna GPU allocata, nessun file scritto.")'''),
        ("code", loader_code),
    ]


MAXVIT_LOADER = '''if not DRY_RUN:
    import numpy as np
    import torch
    from maxvit_utils import build_maxvit_model, make_dataloader, predict_probs, resolve_normalization
    device = torch.device("cuda" if DEVICE == "auto" and torch.cuda.is_available() else ("cpu" if DEVICE == "auto" else DEVICE))
    test_df = pd.read_csv(TEST_CSV)
    test_df["resolved_path"] = test_df["processed_path"].map(lambda x: str(PROJECT_ROOT / x))
    for exp in experiments:
        canonical_paths = canonical_test_prediction_paths(exp)
        pred_path = PROJECT_ROOT / canonical_paths["test_predictions_path"]
        manifest_path = PROJECT_ROOT / canonical_paths["test_predictions_manifest_path"]
        output_dir = pred_path.parent
        expected_cache = {"experiment_id": exp["experiment_id"], "checkpoint_signature": content_signature(PROJECT_ROOT / exp["checkpoint_path"]), "validation_metrics_signature": content_signature(PROJECT_ROOT / exp["validation_metrics_path"]), "validation_threshold": exp["validation_threshold"], "threshold_method": exp["validation_threshold_method"], "test_dataset_manifest_signature": content_signature(TEST_DATASET_MANIFEST), "patient_ids_hash": patient_ids_hash(test_df.patient_id), "preprocessing": {"resolution": 512, "grayscale_to_rgb": True}, "model_config": {"model": "maxvit_tiny_tf_512.in1k"}, "pipeline_schema_version": 1, "provenance_level": "verified_recomputed"}
        cache = prediction_cache_status(manifest_path, expected_cache, pred_path); print(cache["status"], cache["incompatible_keys"])
        if cache["status"] == "CACHE_VALID" and not RECOMPUTE_TEST_PREDICTIONS: continue
        if pred_path.exists() and not RECOMPUTE_TEST_PREDICTIONS: raise RuntimeError(f"CACHE_INCOMPATIBLE: {cache['incompatible_keys']}")
        model = build_maxvit_model(num_classes=1, pretrained=False)
        state = unwrap_checkpoint_state_dict(torch.load(PROJECT_ROOT / exp["checkpoint_path"], map_location=device))
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        mismatch = checkpoint_key_mismatch(missing_keys, unexpected_keys)
        if mismatch["unexplained_missing"] or mismatch["unexplained_unexpected"]: raise RuntimeError(f"Checkpoint MaxViT incompatibile: {mismatch}")
        model.to(device).eval()
        mean, std, img_size = resolve_normalization(model)
        loader = make_dataloader(test_df, "resolved_path", "label", mean, std, img_size, TEST_BATCH_SIZE, False, False, 42, TEST_NUM_WORKERS)
        y_true, y_score = predict_probs(model, loader, device)
        raw = test_df[["patient_id", "image_id", "processed_path"]].rename(columns={"processed_path": "path"})
        raw["y_true"], raw["y_score"] = y_true.astype(int), y_score
        standardized = standardize_prediction_dataframe(raw, experiment=exp, threshold=exp["validation_threshold"], threshold_method=exp["validation_threshold_method"])
        output_dir.mkdir(parents=True, exist_ok=True); standardized.to_csv(pred_path, index=False)
        metrics = compute_binary_metrics(standardized.y_true, standardized.y_score, exp["validation_threshold"])
        (output_dir / "test_metrics.json").write_text(strict_json_dumps(metrics, indent=2) + "\\n")
        pd.DataFrame([metrics | {"confusion_matrix": json.dumps(metrics["confusion_matrix"])}]).to_csv(output_dir / "test_metrics.csv", index=False)
        pd.DataFrame(metrics["confusion_matrix"]).to_csv(output_dir / "confusion_matrix.csv", index=False)
        manifest = {**expected_cache, "preprocessing_details": {"resolution": img_size, "grayscale_to_rgb": True, "mean": mean, "std": std}, "n_patients": len(standardized), "test_used_for_selection": False}
        write_prediction_manifest(manifest_path, manifest, pred_path)'''


RADDINO_LOADER = '''if not DRY_RUN:
    import torch
    from medfoundation_utils import build_medfoundation_model, make_medfoundation_dataloader, predict_probs, resolve_normalization_medfoundation
    device = torch.device("cuda" if DEVICE == "auto" and torch.cuda.is_available() else ("cpu" if DEVICE == "auto" else DEVICE))
    test_df = pd.read_csv(TEST_CSV); test_df["resolved_path"] = test_df["processed_path"].map(lambda x: str(PROJECT_ROOT / x))
    for exp in experiments:
        canonical_paths = canonical_test_prediction_paths(exp)
        pred_path = PROJECT_ROOT / canonical_paths["test_predictions_path"]
        manifest_path = PROJECT_ROOT / canonical_paths["test_predictions_manifest_path"]
        output_dir = pred_path.parent
        expected_cache = {"experiment_id": exp["experiment_id"], "checkpoint_signature": content_signature(PROJECT_ROOT / exp["checkpoint_path"]), "validation_metrics_signature": content_signature(PROJECT_ROOT / exp["validation_metrics_path"]), "validation_threshold": exp["validation_threshold"], "threshold_method": exp["validation_threshold_method"], "test_dataset_manifest_signature": content_signature(TEST_DATASET_MANIFEST), "patient_ids_hash": patient_ids_hash(test_df.patient_id), "preprocessing": {"resolution": 512, "grayscale_to_rgb": True}, "model_config": {"model": "microsoft/rad-dino"}, "pipeline_schema_version": 1, "provenance_level": "verified_recomputed"}
        cache = prediction_cache_status(manifest_path, expected_cache, pred_path); print(cache["status"], cache["incompatible_keys"])
        if cache["status"] == "CACHE_VALID" and not RECOMPUTE_TEST_PREDICTIONS: continue
        if pred_path.exists() and not RECOMPUTE_TEST_PREDICTIONS: raise RuntimeError(f"CACHE_INCOMPATIBLE: {cache['incompatible_keys']}")
        model, processor, _ = build_medfoundation_model("microsoft/rad-dino", num_classes=1)
        state = unwrap_checkpoint_state_dict(torch.load(PROJECT_ROOT / exp["checkpoint_path"], map_location=device))
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        mismatch = checkpoint_key_mismatch(missing_keys, unexpected_keys)
        if mismatch["unexplained_missing"] or mismatch["unexplained_unexpected"]: raise RuntimeError(f"Checkpoint RAD-DINO incompatibile: {mismatch}")
        model.to(device).eval(); mean, std, img_size = resolve_normalization_medfoundation(processor)
        loader = make_medfoundation_dataloader(test_df, "resolved_path", "label", mean, std, img_size, TEST_BATCH_SIZE, False, False, TEST_NUM_WORKERS, 42, False)
        y_true, y_score = predict_probs(model, loader, device)
        raw = test_df[["patient_id", "image_id", "processed_path"]].rename(columns={"processed_path": "path"}); raw["y_true"], raw["y_score"] = y_true.astype(int), y_score
        standardized = standardize_prediction_dataframe(raw, experiment=exp, threshold=exp["validation_threshold"], threshold_method=exp["validation_threshold_method"])
        output_dir.mkdir(parents=True, exist_ok=True); standardized.to_csv(pred_path, index=False)
        metrics = compute_binary_metrics(standardized.y_true, standardized.y_score, exp["validation_threshold"])
        (output_dir / "test_metrics.json").write_text(strict_json_dumps(metrics, indent=2) + "\\n"); pd.DataFrame([metrics | {"confusion_matrix": json.dumps(metrics["confusion_matrix"])}]).to_csv(output_dir / "test_metrics.csv", index=False); pd.DataFrame(metrics["confusion_matrix"]).to_csv(output_dir / "confusion_matrix.csv", index=False)
        manifest = {**expected_cache, "preprocessing_details": {"resolution": img_size, "grayscale_to_rgb": True, "mean": mean, "std": std}, "n_patients": len(standardized), "test_used_for_selection": False}
        write_prediction_manifest(manifest_path, manifest, pred_path)'''


MAMMOFM_LOADER = '''if not DRY_RUN:
    import torch
    from mammofm_utils import build_mammofm_model, predict_with_probs, DEFAULT_HF_REPO, DEFAULT_CHECKPOINT_NAME
    device = torch.device("cuda" if DEVICE == "auto" and torch.cuda.is_available() else ("cpu" if DEVICE == "auto" else DEVICE))
    test_df = pd.read_csv(TEST_CSV); test_df["resolved_path"] = test_df["processed_path"].map(lambda x: str(PROJECT_ROOT / x))
    # Backbone + classification head must match the checkpoint exactly; no tolerated mismatch today.
    ALLOWED_MAMMOFM_CHECKPOINT_MISMATCHES = frozenset()
    for exp in experiments:
        canonical_paths = canonical_test_prediction_paths(exp)
        pred_path = PROJECT_ROOT / canonical_paths["test_predictions_path"]
        manifest_path = PROJECT_ROOT / canonical_paths["test_predictions_manifest_path"]
        output_dir = pred_path.parent
        expected_cache = {"experiment_id": exp["experiment_id"], "checkpoint_signature": content_signature(PROJECT_ROOT / exp["checkpoint_path"]), "validation_metrics_signature": content_signature(PROJECT_ROOT / exp["validation_metrics_path"]), "validation_threshold": exp["validation_threshold"], "threshold_method": exp["validation_threshold_method"], "test_dataset_manifest_signature": content_signature(TEST_DATASET_MANIFEST), "patient_ids_hash": patient_ids_hash(test_df.patient_id), "preprocessing": {"resolution": 512, "grayscale_to_rgb": True}, "model_config": {"model": "mammofm_efficientnet_b5", "hf_repo": DEFAULT_HF_REPO, "checkpoint_name": DEFAULT_CHECKPOINT_NAME}, "pipeline_schema_version": 1, "provenance_level": "verified_recomputed"}
        cache = prediction_cache_status(manifest_path, expected_cache, pred_path); print(cache["status"], cache["incompatible_keys"])
        if cache["status"] == "CACHE_VALID" and not RECOMPUTE_TEST_PREDICTIONS: continue
        if pred_path.exists() and not RECOMPUTE_TEST_PREDICTIONS: raise RuntimeError(f"CACHE_INCOMPATIBLE: {cache['incompatible_keys']}")
        model, mean, std, img_size, hidden_size, backend, source_desc = build_mammofm_model()
        raw_state = torch.load(PROJECT_ROOT / exp["checkpoint_path"], map_location=device)
        state_dict = unwrap_checkpoint_state_dict(raw_state)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        mismatch = checkpoint_key_mismatch(missing_keys, unexpected_keys, ALLOWED_MAMMOFM_CHECKPOINT_MISMATCHES)
        if mismatch["unexplained_missing"] or mismatch["unexplained_unexpected"]:
            raise RuntimeError(f"Checkpoint Mammo-FM per {exp['experiment_id']} non e' strettamente compatibile: {mismatch}. Nessuna voce in allowlist; non procedo con pesi parziali.")
        model.to(device).eval()
        y_true, y_score = predict_with_probs(model, test_df, "resolved_path", "label", mean, std, img_size, TEST_BATCH_SIZE, device)
        raw = test_df[["patient_id", "image_id", "processed_path"]].rename(columns={"processed_path": "path"})
        raw["y_true"], raw["y_score"] = y_true.astype(int), y_score
        standardized = standardize_prediction_dataframe(raw, experiment=exp, threshold=exp["validation_threshold"], threshold_method=exp["validation_threshold_method"])
        output_dir.mkdir(parents=True, exist_ok=True); standardized.to_csv(pred_path, index=False)
        metrics = compute_binary_metrics(standardized.y_true, standardized.y_score, exp["validation_threshold"])
        (output_dir / "test_metrics.json").write_text(strict_json_dumps(metrics, indent=2) + "\\n"); pd.DataFrame([metrics | {"confusion_matrix": json.dumps(metrics["confusion_matrix"])}]).to_csv(output_dir / "test_metrics.csv", index=False); pd.DataFrame(metrics["confusion_matrix"]).to_csv(output_dir / "confusion_matrix.csv", index=False)
        manifest = {**expected_cache, "preprocessing_details": {"resolution": img_size, "grayscale_to_rgb": True, "mean": mean, "std": std, "hidden_size": hidden_size, "backend": backend, "source_desc": source_desc}, "n_patients": len(standardized), "test_used_for_selection": False}
        write_prediction_manifest(manifest_path, manifest, pred_path)'''


LEADERBOARD_CELLS = [
    ("code", COMMON_CONFIG + '''
DRY_RUN = True
PRIMARY_SELECTION_METRIC = "validation_roc_auc"
SECONDARY_SELECTION_METRIC = "validation_pr_auc"
FINALIST_POLICY = {"include_baseline_per_architecture": True, "include_best_synthetic_per_architecture": True, "include_best_augmented_per_architecture": True, "include_best_overall_per_architecture": True, "max_finalists_total": 10}
OUTPUT_DIR = PROJECT_ROOT / "results/final_evaluation"'''),
    ("code", '''registry = build_experiment_registry(REGISTRY_PATH)
coverage = build_test_coverage_table(registry, PROJECT_ROOT)
finalists = select_validation_finalists(coverage, FINALIST_POLICY)
print(coverage[["experiment_id", "architecture", "validation_roc_auc", "scientifically_eligible", "selected_by_validation", "status", "blocked_reason", "exclusion_reason"]].to_string(index=False))
print("Finalisti validation:", [(x["experiment_id"], x["selection_reason"]) for x in finalists])
assert not any(key.startswith("test_") and key.endswith(("auc", "accuracy", "f1")) for key in [PRIMARY_SELECTION_METRIC, SECONDARY_SELECTION_METRIC])
if DRY_RUN: print("DRY_RUN: leaderboard/manifest/lock non scritti e metriche test non lette.")'''),
    ("code", '''if not DRY_RUN:
    import matplotlib.pyplot as plt
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    leaderboard = coverage.rename(columns={"training_dataset_variant": "dataset_variant"})
    leaderboard.to_csv(OUTPUT_DIR / "validation_leaderboard.csv", index=False)
    (OUTPUT_DIR / "validation_leaderboard.json").write_text(strict_json_dumps(leaderboard.to_dict("records"), indent=2) + "\\n")
    # Scientific finalists are kept even without a checkpoint (e.g. ResNet): build_locked_finalist_entries
    # never raises on a missing checkpoint and preserves the full scientific/operational field set.
    locked_entries = build_locked_finalist_entries(finalists, PROJECT_ROOT)
    manifest = lock_finalists_manifest(locked_entries, FINALIST_POLICY, OUTPUT_DIR / "finalists_manifest.json")
    (OUTPUT_DIR / "FINALISTS_LOCKED").write_text(manifest["lock_signature"]["sha256"] + "\\n")
    plot = leaderboard[leaderboard.validation_roc_auc.notna()].sort_values("validation_roc_auc")
    ax = plot.plot.barh(x="display_name", y="validation_roc_auc", legend=False, figsize=(9, 8)); ax.figure.tight_layout(); ax.figure.savefig(OUTPUT_DIR / "figures/validation_auc_comparison.png", dpi=300); plt.close(ax.figure)
    pr = leaderboard[leaderboard.validation_pr_auc.notna()].sort_values("validation_pr_auc")
    ax = pr.plot.barh(x="display_name", y="validation_pr_auc", legend=False, figsize=(9, 6)); ax.figure.tight_layout(); ax.figure.savefig(OUTPUT_DIR / "figures/validation_pr_auc_comparison.png", dpi=300); plt.close(ax.figure)'''),
]


LOCKED_TEST_CELLS = [
    ("code", COMMON_CONFIG + '''
DRY_RUN = True
MANIFEST_PATH = PROJECT_ROOT / "results/final_evaluation/finalists_manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "results/final_evaluation"
CENTRAL_PREDICTIONS = OUTPUT_DIR / "test_predictions"'''),
    ("code", '''# Scientific-only read: this must succeed and list blockers/missing predictions even while
# the lock is operationally incomplete (e.g. Mammo-FM/ResNet blockers) -- it must not crash here.
manifest = validate_locked_finalists_manifest(MANIFEST_PATH, require_operational_complete=False)
registry = {x["experiment_id"]: x for x in build_experiment_registry(REGISTRY_PATH)}
missing = []
for finalist in manifest["finalists"]:
    exp = registry[finalist["experiment_id"]]
    paths = canonical_test_prediction_paths(exp)
    pred = PROJECT_ROOT / paths["test_predictions_path"]
    if not pred.is_file(): missing.append({"experiment_id": exp["experiment_id"], "notebook": exp.get("test_notebook"), "expected": str(pred)})
print("Finalisti locked:", [x["experiment_id"] for x in manifest["finalists"]])
print("scientific_selection_complete:", manifest.get("scientific_selection_complete"), "| final_aggregation_complete:", manifest.get("final_aggregation_complete"))
print("Blocker operativi:", manifest.get("operational_blockers")); print("Predizioni mancanti:", missing)
if DRY_RUN: print("DRY_RUN: nessuna predizione copiata, nessuna metrica test scritta.")'''),
    ("code", '''if not DRY_RUN:
    # Operational gate: only proceed once every finalist has real, verified test predictions.
    manifest = validate_locked_finalists_manifest(MANIFEST_PATH, require_operational_complete=True)
    if missing: raise RuntimeError(f"Copertura test incompleta: {missing}")
    canonical = pd.read_csv(TEST_CSV); canonical_ids = canonical.patient_id.astype(str)
    CENTRAL_PREDICTIONS.mkdir(parents=True, exist_ok=True); frames = {}; source_paths = {}; metric_rows = []
    for finalist in manifest["finalists"]:
        exp = registry[finalist["experiment_id"]]
        paths = canonical_test_prediction_paths(exp)
        source = PROJECT_ROOT / paths["test_predictions_path"]
        source_manifest = PROJECT_ROOT / paths["test_predictions_manifest_path"]
        if not source_manifest.is_file(): raise RuntimeError(f"Manifest sorgente mancante: {source_manifest}")
        source_payload = json.loads(source_manifest.read_text())
        required = {"experiment_id": exp["experiment_id"], "checkpoint_signature": finalist["checkpoint_signature"], "validation_metrics_signature": content_signature(PROJECT_ROOT / exp["validation_metrics_path"]), "validation_threshold": finalist["validation_threshold"], "threshold_method": finalist["threshold_method"], "test_dataset_manifest_signature": content_signature(TEST_DATASET_MANIFEST), "patient_ids_hash": patient_ids_hash(canonical_ids), "test_used_for_selection": False}
        incompatible = [key for key, value in required.items() if source_payload.get(key) != strict_jsonable(value)]
        if source_payload.get("prediction_file_signature") != content_signature(source): incompatible.append("prediction_file_signature")
        level = source_payload.get("provenance_level", "invalid")
        if level == "legacy_normalized_unverified" and not ALLOW_UNVERIFIED_LEGACY_PREDICTIONS: incompatible.append("provenance_level")
        if level not in {"verified_native", "verified_recomputed", "legacy_normalized_unverified"}: incompatible.append("provenance_level")
        if incompatible: raise RuntimeError(f"Manifest sorgente incompatibile per {exp['experiment_id']}: {sorted(set(incompatible))}")
        frame = pd.read_csv(source); frames[exp["experiment_id"]] = frame; source_paths[exp["experiment_id"]] = source
    aligned = compare_patient_sets(frames, canonical_ids)
    for eid, frame in aligned.items():
        destination = CENTRAL_PREDICTIONS / f"{eid}.csv"; frame.to_csv(destination, index=False)
        metric_rows.append({"experiment_id": eid, **compute_binary_metrics(frame.y_true, frame.y_score, float(frame.threshold.iloc[0]))})
        exp = registry[eid]; source = source_paths[eid]; source_manifest = PROJECT_ROOT / canonical_test_prediction_paths(exp)["test_predictions_manifest_path"]
        write_prediction_manifest(CENTRAL_PREDICTIONS / f"{eid}.manifest.json", {"experiment_id": eid, "registry_signature": content_signature(REGISTRY_PATH), "source_prediction_manifest_path": str(source_manifest.relative_to(PROJECT_ROOT)), "source_prediction_manifest_signature": content_signature(source_manifest), "source_prediction_file_signature": content_signature(source), "finalists_lock_signature": manifest["lock_signature"], "test_dataset_manifest_signature": content_signature(TEST_DATASET_MANIFEST), "validation_threshold": float(frame.threshold.iloc[0]), "threshold_method": frame.threshold_method.iloc[0], "pipeline_schema_version": 1, "provenance_level": json.loads(source_manifest.read_text()).get("provenance_level", "invalid")}, destination)
    pd.DataFrame(metric_rows).to_csv(OUTPUT_DIR / "final_test_metrics.csv", index=False); (OUTPUT_DIR / "final_test_metrics.json").write_text(strict_json_dumps(metric_rows, indent=2) + "\\n")
    dataset_manifest = build_test_dataset_manifest(TEST_CSV, project_root=PROJECT_ROOT, preprocessing={"view": "MLO", "resolution": 512, "grayscale": True, "right_breast_mirrored": True}, include_image_signatures=True)
    (OUTPUT_DIR / "test_dataset_manifest.json").write_text(strict_json_dumps(dataset_manifest, indent=2) + "\\n")
    run = {"schema_version": 1, "finalists_lock_signature": manifest["lock_signature"], "test_dataset_signature": value_signature(dataset_manifest), "selection_performed": False, "n_models": len(frames)}
    (OUTPUT_DIR / "final_test_run_manifest.json").write_text(strict_json_dumps(run, indent=2) + "\\n")'''),
]


STATS_CELLS = [
    ("code", COMMON_CONFIG + '''
DRY_RUN = True
BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95
FINAL_RANKING_PRIMARY = "test_roc_auc"
FINAL_RANKING_SECONDARY = "test_pr_auc"
CALIBRATION_METRICS = ["Brier score", "ECE", "reliability curve"]
OUTPUT_DIR = PROJECT_ROOT / "results/final_evaluation"
PREDICTION_DIR = OUTPUT_DIR / "test_predictions"'''),
    ("code", '''# Scientific-only read: must list blockers and missing predictions without crashing even
# while the lock is operationally incomplete.
manifest = validate_locked_finalists_manifest(OUTPUT_DIR / "finalists_manifest.json", require_operational_complete=False)
expected = [x["experiment_id"] for x in manifest["finalists"]]
available = [eid for eid in expected if (PREDICTION_DIR / f"{eid}.csv").is_file()]
missing = sorted(set(expected) - set(available))
print("Modelli paired:", available)
print("scientific_selection_complete:", manifest.get("scientific_selection_complete"), "| final_aggregation_complete:", manifest.get("final_aggregation_complete"))
print("Blocker operativi:", manifest.get("operational_blockers")); print("Mancanti:", missing)
if DRY_RUN: print("DRY_RUN: nessun test statistico o figura scritto.")'''),
    ("code", '''if not DRY_RUN:
    # Operational gate: only proceed once every finalist has real, verified test predictions.
    manifest = validate_locked_finalists_manifest(OUTPUT_DIR / "finalists_manifest.json", require_operational_complete=True)
    if missing: raise RuntimeError(f"Predizioni centralizzate mancanti: {missing}")
    import itertools, matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    canonical = pd.read_csv(TEST_CSV); canonical_ids = canonical.patient_id.astype(str)
    finalist_by_id = {x["experiment_id"]: x for x in manifest["finalists"]}
    frames = {}
    for eid in expected:
        source_manifest_path = PREDICTION_DIR / f"{eid}.manifest.json"
        if not source_manifest_path.is_file(): raise RuntimeError(f"Manifest centralizzato mancante: {source_manifest_path}")
        payload = json.loads(source_manifest_path.read_text())
        finalist, pred_path = finalist_by_id[eid], PREDICTION_DIR / f"{eid}.csv"
        required = {"experiment_id": eid, "finalists_lock_signature": manifest["lock_signature"], "test_dataset_manifest_signature": content_signature(TEST_DATASET_MANIFEST), "validation_threshold": finalist["validation_threshold"], "threshold_method": finalist["threshold_method"]}
        incompatible = [key for key, value in required.items() if payload.get(key) != strict_jsonable(value)]
        if payload.get("prediction_file_signature") != content_signature(pred_path): incompatible.append("prediction_file_signature")
        level = payload.get("provenance_level", "invalid")
        if level == "legacy_normalized_unverified" and not ALLOW_UNVERIFIED_LEGACY_PREDICTIONS: incompatible.append("provenance_level")
        if level not in {"verified_native", "verified_recomputed", "legacy_normalized_unverified"}: incompatible.append("provenance_level")
        if incompatible: raise RuntimeError(f"Manifest centralizzato incompatibile per {eid}: {sorted(set(incompatible))}")
        frames[eid] = pd.read_csv(pred_path)
    aligned = compare_patient_sets(frames, canonical_ids); y = next(iter(aligned.values())).y_true.to_numpy()
    rows = []
    registry = {x["experiment_id"]: x for x in build_experiment_registry(REGISTRY_PATH)}
    for a, b in itertools.combinations(expected, 2):
        ga, gb = registry[a].get("primary_comparison_group"), registry[b].get("primary_comparison_group")
        family = "primary" if ga and ga == gb else "secondary"
        sa, sb = aligned[a].y_score.to_numpy(), aligned[b].y_score.to_numpy()
        boot = paired_stratified_bootstrap(y, sa, sb, "roc_auc", BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED, CI_LEVEL); boot_pr = paired_stratified_bootstrap(y, sa, sb, "pr_auc", BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED, CI_LEVEL); dl = delong_roc_test(y, sa, sb)
        pa, pb = aligned[a].y_pred.to_numpy(), aligned[b].y_pred.to_numpy(); mc = mcnemar_test(y, pa, pb)
        # One row per (pair, metric): ROC-AUC keeps DeLong as its paired p-value, PR-AUC uses the
        # bootstrap p-value (DeLong has no PR-AUC equivalent), McNemar is its own metric/row
        # instead of two columns bolted onto every row -- never mix different statistical tests
        # into the same Holm correction family.
        rows.append({"model_a": a, "model_b": b, "metric": "roc_auc", "comparison_family": family, **boot, "delong_p_value": dl["p_value"], "raw_p_value": dl["p_value"]})
        rows.append({"model_a": a, "model_b": b, "metric": "pr_auc", "comparison_family": family, **boot_pr, "raw_p_value": boot_pr["p_bootstrap"]})
        rows.append({"model_a": a, "model_b": b, "metric": "mcnemar", "comparison_family": family, "mcnemar_b": mc["b"], "mcnemar_c": mc["c"], "mcnemar_statistic": mc["statistic"], "mcnemar_method": mc["method"], "raw_p_value": mc["p_value"]})
    comparisons = pd.DataFrame(rows)
    comparisons["holm_family"] = comparisons["comparison_family"] + "_" + comparisons["metric"]
    comparisons["holm_adjusted_p_value"] = float("nan")
    for _, group in comparisons.groupby("holm_family"):
        comparisons.loc[group.index, "holm_adjusted_p_value"] = holm_adjustment(group["raw_p_value"])
    comparisons["significant_raw"] = comparisons.raw_p_value < .05; comparisons["significant_holm"] = comparisons.holm_adjusted_p_value < .05
    comparisons.to_csv(OUTPUT_DIR / "paired_comparisons.csv", index=False); (OUTPUT_DIR / "paired_comparisons.json").write_text(strict_json_dumps(comparisons.to_dict("records"), indent=2) + "\\n")
    ranking = pd.DataFrame([{"experiment_id": eid, "test_roc_auc": roc_auc_score(y, aligned[eid].y_score), "test_pr_auc": average_precision_score(y, aligned[eid].y_score)} for eid in expected]).sort_values(["test_roc_auc", "test_pr_auc"], ascending=False); ranking.to_csv(OUTPUT_DIR / "final_ranking.csv", index=False)
    figures = OUTPUT_DIR / "figures"; figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6));
    for eid in expected:
        fpr, tpr, _ = roc_curve(y, aligned[eid].y_score); ax.plot(fpr, tpr, label=f"{eid} ({roc_auc_score(y, aligned[eid].y_score):.3f})")
    ax.plot([0,1],[0,1], "k--"); ax.legend(fontsize=7); ax.set(xlabel="1 - specificity", ylabel="sensitivity"); fig.tight_layout(); fig.savefig(figures / "final_roc_curves.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6));
    for eid in expected:
        precision, recall, _ = precision_recall_curve(y, aligned[eid].y_score); ax.plot(recall, precision, label=f"{eid} ({average_precision_score(y, aligned[eid].y_score):.3f})")
    ax.axhline(y.mean(), ls="--", color="k"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(figures / "final_pr_curves.png", dpi=300); plt.close(fig)
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(8, 6))
    for eid in expected:
        frac, mean = calibration_curve(y, aligned[eid].y_score, n_bins=10, strategy="uniform"); ax.plot(mean, frac, marker="o", label=eid)
    ax.plot([0,1],[0,1], "k--"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(figures / "calibration_curves.png", dpi=300); plt.close(fig)
    fig, axes = plt.subplots(len(expected), 1, figsize=(6, 4 * len(expected)), squeeze=False)
    for ax, eid in zip(axes[:,0], expected): ConfusionMatrixDisplay.from_predictions(y, aligned[eid].y_pred, ax=ax); ax.set_title(eid)
    fig.tight_layout(); fig.savefig(figures / "confusion_matrices.png", dpi=300); plt.close(fig)
    roc_boot = comparisons[comparisons.metric.eq("roc_auc")].copy(); fig, ax = plt.subplots(figsize=(9, max(4, len(roc_boot)*.35))); ypos = range(len(roc_boot)); ax.errorbar(roc_boot.mean_difference, ypos, xerr=[roc_boot.mean_difference-roc_boot.ci_lower, roc_boot.ci_upper-roc_boot.mean_difference], fmt="o"); ax.axvline(0, color="k", ls="--"); ax.set_yticks(list(ypos), [f"{a} - {b}" for a,b in zip(roc_boot.model_a, roc_boot.model_b)]); fig.tight_layout(); fig.savefig(figures / "bootstrap_auc_differences.png", dpi=300); plt.close(fig)
    best = ranking.iloc[0]; significant = comparisons[comparisons.significant_holm.fillna(False)]; nonsignificant = comparisons[~comparisons.significant_holm.fillna(False)]
    legacy = [eid for eid in expected if json.loads((PREDICTION_DIR / f"{eid}.manifest.json").read_text()).get("provenance_level") == "legacy_normalized_unverified"]
    resnet_absent = not any(registry[eid]["architecture"] == "ResNet-50" for eid in expected)
    conclusions = f"# Conclusioni finali\\n\\nMiglior valore puntuale: **{best.experiment_id}**, ROC-AUC {best.test_roc_auc:.4f} e PR-AUC {best.test_pr_auc:.4f}. Il ranking puntuale non implica superiorità statistica.\\n\\nConfronti paired significativi dopo Holm: {len(significant)}; non significativi: {len(nonsignificant)}. Intervalli bootstrap e p-value sono negli artefatti paired.\\n\\nProvenance legacy accettata: {legacy or 'nessuna'}. ResNet assente dal confronto finale: {resnet_absent}.\\n\\n## Limiti metodologici\\n\\nIl test era già stato osservato durante lo sviluppo. Nessuna vittoria è dichiarata senza supporto paired corretto per molteplicità; serve conferma esterna o grouped cross-validation.\\n"
    (OUTPUT_DIR / "final_conclusions.md").write_text(conclusions)'''),
]


write_notebook("notebooks/3_classifiers/02k_MaxViT512_LockedFinalTest.ipynb", "MaxViT-512 — Locked Final Test", classifier_cells("maxvit512", ["maxvit512_02a_real_only", "maxvit512_02c_real_synth_full", "maxvit512_02h_fromscratch_synthetic_full", "maxvit512_02j_real_aug_synth_finetuned"], MAXVIT_LOADER))
write_notebook("notebooks/3_classifiers/04c_RADDINO_LockedFinalTest.ipynb", "RAD-DINO — Locked Final Test", classifier_cells("raddino", ["raddino_04a_real_only", "raddino_04b_real_synth"], RADDINO_LOADER))
write_notebook("notebooks/3_classifiers/03e_MammoFM_LockedFinalTest.ipynb", "Mammo-FM — Locked Final Test", classifier_cells("mammofm", ["mammofm_03a_real_only", "mammofm_03b_real_synth_finetuned", "mammofm_03d_real_augmented"], MAMMOFM_LOADER))
write_notebook("notebooks/4_comparisons_and_test/04x_Leaderboard_Validation_All_Classifiers.ipynb", "Leaderboard Validation — All Classifiers", LEADERBOARD_CELLS)
write_notebook("notebooks/4_comparisons_and_test/04y_Final_Test_Locked.ipynb", "Final Test Locked", LOCKED_TEST_CELLS)
write_notebook("notebooks/4_comparisons_and_test/04z_Final_Statistical_Comparison.ipynb", "Final Statistical Comparison — Paired", STATS_CELLS)
