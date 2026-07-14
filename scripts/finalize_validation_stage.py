#!/usr/bin/env python3
"""Aggregate validation-only results and freeze the next stage's inputs (spec sections 3.5, 14).

    python scripts/finalize_validation_stage.py --stage 1
    python scripts/finalize_validation_stage.py --stage 2

Stage 1: ranks every completed job by validation PR-AUC (primary) / ROC-AUC (secondary) per
architecture, computes GLOBAL_TOP_K_GENERATORS=3 by mean rank across the four classifier
families, unions in every generator that outright wins at least one family, and signs the
result as SELECTED_GENERATOR_UNION — the only input scripts/build_classifier_experiment_matrix.py
--stage 2 will accept. Reads validation predictions/metrics only; never opens a test file.

Stage 2: defines primary finalists (R, RA, best RS/RAS/S-only per architecture) and the primary
comparison families, and a secondary locked panel of every other completed, preregistered
configuration — preparation for scripts/finalize_locked_test_stage.py, not the lock itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
from classifier_pipeline_contracts import (  # noqa: E402
    PIPELINE_NAMESPACE, REQUIRED_SEEDS, atomic_json, code_revision, signed_payload, verify_signed_payload,
)

GLOBAL_TOP_K_GENERATORS = 3
FAMILIES = ("resnet50", "maxvit512", "mammofm", "raddino")


def _generator_of(dataset_variant_id: str) -> str | None:
    """Primary Stage-1 screening is exclusively RSB_CONTROLLED ensemble validation."""
    prefix = "RSB_CONTROLLED_"
    return dataset_variant_id[len(prefix):] if dataset_variant_id.startswith(prefix) else None


def load_completed_validations(root: Path, stage: int) -> list[dict]:
    matrix = json.loads((root / "configs/classifier_experiment_matrix.json").read_text())
    strict_v2 = matrix.get("pipeline_namespace") == PIPELINE_NAMESPACE
    if stage in (1, 2):
        rows = []
        new_paths = (root / "results/classifiers_matrix").glob("*/*/*/ensemble/manifests/ensemble_validation_manifest.json")
        legacy_paths = (root / "results/classifiers_matrix").glob("*/*/*/ensemble_validation_manifest.json")
        for path in sorted([*new_paths, *legacy_paths]):
            payload = json.loads(path.read_text())
            if strict_v2 and payload.get("pipeline_namespace") != PIPELINE_NAMESPACE:
                continue
            if payload.get("pipeline_namespace") is not None:
                try:
                    verify_signed_payload(payload)
                    if payload.get("artifact_type") != "classifier_validation_ensemble":
                        continue
                except ValueError:
                    continue
            vid = payload["dataset_variant_id"]
            is_stage1 = bool(_generator_of(vid))
            is_stage2 = vid.startswith(("RAS_", "S_ONLY_"))
            if (stage == 1 and not is_stage1) or (stage == 2 and not is_stage2):
                continue
            if payload.get("seeds") != [17, 42, 73] or payload.get("test_access") is not False:
                continue
            rows.append({"experiment_id": f"{payload['architecture']}__{vid}__ensemble",
                         "seed_experiment_ids": [f"{payload['architecture']}__{vid}__seed{seed}" for seed in (17, 42, 73)],
                         "stage": stage, "architecture": payload["architecture"], "dataset_variant_id": vid,
                         "status": "COMPLETE", "aggregation": "ensemble", **payload["metrics"],
                         "ensemble_signature": payload.get("signature")})
        return rows
    rows = []
    for job in matrix["jobs"]:
        if job["stage"] != stage or job["status"] not in ("VALIDATED", "COMPLETE"):
            continue
        metrics_path = root / job["validation_predictions_path"]
        seed = job["seed"]
        vmetrics_path = metrics_path.parent / f"validation_metrics_seed_{seed}.json"
        if not vmetrics_path.is_file():
            continue
        metrics = json.loads(vmetrics_path.read_text())
        rows.append({**job, "pr_auc": metrics.get("pr_auc"), "roc_auc": metrics.get("roc_auc")})
    return rows


def rank_by_generator(rows: list[dict], architecture: str) -> list[dict]:
    arch_rows = [r for r in rows if r["architecture"] == architecture and _generator_of(r["dataset_variant_id"])]
    by_generator: dict[str, list[dict]] = {}
    for r in arch_rows:
        by_generator.setdefault(_generator_of(r["dataset_variant_id"]), []).append(r)
    aggregated = []
    for gid, entries in by_generator.items():
        pr_aucs = [e["pr_auc"] for e in entries if e["pr_auc"] is not None]
        if not pr_aucs:
            continue
        aggregated.append({"generator_id": gid, "mean_pr_auc": sum(pr_aucs) / len(pr_aucs),
                            "n_configurations": len(pr_aucs), "aggregation": entries[0].get("aggregation", "seed_fixture"),
                            "roc_aucs": [e["roc_auc"] for e in entries],
                            "configuration_signatures": [e.get("ensemble_signature") for e in entries]})
    # Registered scientific tie-break: PR-AUC, then mean ROC-AUC, then stable generator ID.
    for entry in aggregated:
        values = [value for value in entry["roc_aucs"] if value is not None]
        entry["mean_roc_auc"] = sum(values) / len(values) if values else float("-inf")
    aggregated.sort(key=lambda e: (-e["mean_pr_auc"], -e["mean_roc_auc"], e["generator_id"]))
    for rank, entry in enumerate(aggregated, start=1):
        entry["rank"] = rank
    return aggregated


def _completed_seed_ids_for_matrix(root: Path, matrix: dict, stage: int, selection_rows: list[dict]) -> set[str]:
    """Completion covers the whole stage, while Stage-1 ranking intentionally uses only RSB_CONTROLLED."""
    if matrix.get("pipeline_namespace") != PIPELINE_NAMESPACE:
        return {seed for row in selection_rows for seed in row.get("seed_experiment_ids", [])}
    expected = {(job["architecture"], job["dataset_variant_id"], job["training_policy"])
                for job in matrix.get("jobs", []) if int(job.get("stage", -1)) == stage}
    completed = set()
    for path in sorted((root / "results/classifiers_matrix").glob(
            "*/*/*/ensemble/manifests/ensemble_validation_manifest.json")):
        try:
            payload = json.loads(path.read_text()); verify_signed_payload(payload)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        key = (payload.get("architecture"), payload.get("dataset_variant_id"), payload.get("training_policy"))
        if key not in expected or payload.get("artifact_type") != "classifier_validation_ensemble" or \
           payload.get("seeds") != [17, 42, 73] or payload.get("test_access") is not False:
            continue
        completed.update(f"{key[0]}__{key[1]}__seed{seed}" for seed in REQUIRED_SEEDS)
    return completed


def compute_selected_generator_union(root: Path, stage: int = 1) -> dict:
    rows = load_completed_validations(root, stage)
    per_family_ranking = {arch: rank_by_generator(rows, arch) for arch in FAMILIES}

    all_generators = {e["generator_id"] for ranking in per_family_ranking.values() for e in ranking}
    mean_ranks = {}
    for gid in all_generators:
        ranks = [next(e["rank"] for e in ranking if e["generator_id"] == gid)
                 for ranking in per_family_ranking.values() if any(e["generator_id"] == gid for e in ranking)]
        if ranks:
            mean_ranks[gid] = sum(ranks) / len(ranks)

    top_k = sorted(mean_ranks, key=lambda g: (mean_ranks[g], g))[:GLOBAL_TOP_K_GENERATORS]
    family_winners = {ranking[0]["generator_id"] for ranking in per_family_ranking.values() if ranking}
    union = sorted(set(top_k) | family_winners)

    matrix = json.loads((root / "configs/classifier_experiment_matrix.json").read_text())
    expected_seed_ids = {job["experiment_id"] for job in matrix.get("jobs", []) if int(job.get("stage", -1)) == stage}
    completed_seed_ids = _completed_seed_ids_for_matrix(root, matrix, stage, rows)
    missing_seed_ids = sorted(expected_seed_ids - completed_seed_ids)
    stage_complete = bool(expected_seed_ids) and not missing_seed_ids
    leaderboard = signed_payload({
        "schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE,
        "artifact_type": "classifier_stage1_validation_leaderboard", "stage": stage,
        "primary_metric": "pr_auc", "secondary_metric": "roc_auc",
        "tie_break": ["pr_auc_desc", "roc_auc_desc", "generator_id_asc"],
        "per_family_ranking": per_family_ranking,
        "ensemble_signatures": sorted({row.get("ensemble_signature") for row in rows if row.get("ensemble_signature")}),
        "test_data_used": False,
    })
    payload = signed_payload({
        "schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE,
        "artifact_type": "classifier_selected_generator_union", "stage": stage,
        "code_revision": code_revision(root), "global_top_k": GLOBAL_TOP_K_GENERATORS,
        "per_family_ranking": per_family_ranking, "mean_rank_by_generator": mean_ranks,
        "top_k_generators": top_k, "family_winners": sorted(family_winners),
        "selected_generator_union": union,
        "selection_used_test_data": False,
        "n_completed_jobs_considered": len(rows), "primary_screening_regime": "RSB_CONTROLLED",
        "aggregation_level": "three_seed_ensemble", "excluded_regimes": ["RSB_FULL", "RSP_CONTROLLED", "RSP_FULL"],
        "scientific_completion": {"complete": stage_complete, "expected_seed_jobs": len(expected_seed_ids),
                                  "completed_seed_jobs": len(expected_seed_ids) - len(missing_seed_ids),
                                  "missing_seed_experiment_ids": missing_seed_ids},
        "leaderboard_signature": leaderboard["signature"],
        "ensemble_signatures": leaderboard["ensemble_signatures"],
        "selection_rationale": "Global top-3 by mean family rank, unioned with every family winner.",
        "leaderboard": leaderboard,
    })
    return payload


def write_selected_union(root: Path, payload: dict) -> Path:
    verify_signed_payload(payload)
    completion = payload.get("scientific_completion", {})
    # Old schema-1 test fixtures have no matrix jobs; production schema-2 unions must be
    # complete and non-empty before any Stage-2 consumer can observe them.
    if completion.get("expected_seed_jobs", 0) and not completion.get("complete"):
        raise RuntimeError("Stage 1 is incomplete; refusing to write SELECTED_GENERATOR_UNION")
    if not payload.get("selected_generator_union"):
        raise RuntimeError("Stage 1 selected generator union is empty")
    out_dir = root / "results/generator_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(out_dir / "stage1_validation_leaderboard.json", payload["leaderboard"])
    completion_payload = signed_payload({
        "schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE,
        "artifact_type": "classifier_stage1_completion", "stage": 1,
        "code_revision": payload["code_revision"], "leaderboard_signature": payload["leaderboard_signature"],
        "union_signature": payload["signature"], **completion,
    })
    atomic_json(out_dir / "stage1_completion_manifest.json", completion_payload)
    out_path = out_dir / "selected_generator_union.json"
    if out_path.is_file() and out_path.read_text() != json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False) + "\n":
        raise RuntimeError("a different SELECTED_GENERATOR_UNION already exists; explicit incident review is required")
    atomic_json(out_path, payload)
    return out_path


def finalize_stage2_panels(root: Path) -> dict:
    rows = load_completed_validations(root, stage=2)
    # Required baselines and Stage-1 RS/positive-only panels remain eligible comparators even
    # though Stage 2 itself only schedules RAS/S_ONLY variants.
    new_paths = (root / "results/classifiers_matrix").glob("*/*/*/ensemble/manifests/ensemble_validation_manifest.json")
    legacy_paths = (root / "results/classifiers_matrix").glob("*/*/*/ensemble_validation_manifest.json")
    for path in sorted([*new_paths, *legacy_paths]):
        payload = json.loads(path.read_text())
        vid = payload["dataset_variant_id"]
        if not (vid in ("R", "RA") or vid.startswith(("RSB_CONTROLLED_", "RSB_FULL_", "RSP_"))):
            continue
        if payload.get("seeds") != [17, 42, 73] or payload.get("test_access") is not False:
            continue
        rows.append({"experiment_id": f"{payload['architecture']}__{vid}__ensemble",
                     "seed_experiment_ids": [f"{payload['architecture']}__{vid}__seed{seed}" for seed in (17, 42, 73)],
                     "stage": 1, "architecture": payload["architecture"], "dataset_variant_id": vid,
                     "status": "COMPLETE", "aggregation": "ensemble", **payload["metrics"],
                     "ensemble_signature": payload.get("signature")})
    primary_finalists: dict[str, dict] = {}
    for architecture in FAMILIES:
        arch_rows = [r for r in rows if r["architecture"] == architecture]
        if not arch_rows:
            primary_finalists[architecture] = {"status": "no_completed_stage2_jobs_yet"}
            continue
        categories = {
            "R_baseline": lambda v: v == "R", "RA_baseline": lambda v: v == "RA",
            "best_RS_CONTROLLED": lambda v: v.startswith("RSB_CONTROLLED_"),
            "best_RS_FULL": lambda v: v.startswith("RSB_FULL_"),
            "best_RAS_CONTROLLED": lambda v: v.startswith("RAS_CONTROLLED_"),
            "best_RAS_FULL": lambda v: v.startswith("RAS_FULL_"),
            "best_S_ONLY_CONTROLLED": lambda v: v.startswith("S_ONLY_CONTROLLED_"),
            "best_S_ONLY_FULL": lambda v: v.startswith("S_ONLY_FULL_"),
            "best_positive_only": lambda v: v.startswith("RSP_") or "POSITIVE" in v,
        }
        selected = {}
        for name, predicate in categories.items():
            candidates = [r for r in arch_rows if predicate(r["dataset_variant_id"])]
            selected[name] = (sorted(candidates, key=lambda r: (-(r["pr_auc"] or -1),
                              -(r["roc_auc"] or -1), r["dataset_variant_id"]))[0] if candidates
                              else {"status": "missing_preregistered_validation"})
        primary_finalists[architecture] = selected
    # One *logical* ensemble id per (architecture, dataset_variant) - never the three flattened
    # seed_experiment_ids, which made locked_matrix_inference.run_locked() infer and write the
    # same three-seed ensemble three times under three different output names.
    secondary_panel = sorted({row["experiment_id"] for row in rows if row.get("experiment_id")})
    seed_ids_by_logical = {row["experiment_id"]: row.get("seed_experiment_ids", []) for row in rows if row.get("experiment_id")}
    primary_panel = sorted({entry["experiment_id"] for categories in primary_finalists.values()
                            if isinstance(categories, dict) for entry in categories.values()
                            if isinstance(entry, dict) and entry.get("experiment_id")})
    invalid_mappings = sorted(logical for logical, seeds in seed_ids_by_logical.items()
                              if sorted(int(seed.rsplit("seed", 1)[1]) for seed in seeds) != [17, 42, 73])
    matrix = json.loads((root / "configs/classifier_experiment_matrix.json").read_text())
    expected_stage2 = {job["experiment_id"] for job in matrix.get("jobs", []) if int(job.get("stage", -1)) == 2}
    completed_stage2 = {seed for row in rows if int(row.get("stage", -1)) == 2
                        for seed in row.get("seed_experiment_ids", [])}
    missing_stage2 = sorted(expected_stage2 - completed_stage2)
    strict_v2 = matrix.get("pipeline_namespace") == PIPELINE_NAMESPACE
    if strict_v2:
        secondary_panel = [logical for logical in secondary_panel if logical not in set(primary_panel)]
    if strict_v2 and expected_stage2 and missing_stage2:
        raise RuntimeError(f"Stage 2 is incomplete; missing {len(missing_stage2)} seed jobs")
    source_union_signature = matrix.get("stage2_source_union_signature")
    if strict_v2 and expected_stage2:
        union_path = root / "results/generator_comparison/selected_generator_union.json"
        if not source_union_signature or not union_path.is_file():
            raise RuntimeError("Stage 2 matrix is not bound to a selected generator union")
        source_union = json.loads(union_path.read_text()); verify_signed_payload(source_union)
        if source_union.get("signature") != source_union_signature:
            raise RuntimeError("Stage 2 matrix selected-union signature is stale")
    if invalid_mappings:
        raise RuntimeError(f"logical ensembles lack exact seed mapping: {invalid_mappings}")
    payload = signed_payload({"schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE,
               "artifact_type": "classifier_final_panel_selection", "code_revision": code_revision(root),
               "stage2_source_union_signature": source_union_signature,
               "primary_locked_panel": primary_panel, "primary_finalists": primary_finalists,
               "secondary_locked_panel": secondary_panel, "seed_experiment_ids_by_logical": seed_ids_by_logical,
               "ablation_panel": [],
               "n_completed_jobs_considered": len(rows), "stage2_completion": {
                   "complete": bool(expected_stage2) and not missing_stage2,
                   "expected_seed_jobs": len(expected_stage2), "missing_seed_experiment_ids": missing_stage2},
               "selection_rationale": "Validation PR-AUC, ROC-AUC tie-break, then dataset variant ID.",
               "test_data_used": False})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.project_root)

    if args.stage == 1:
        payload = compute_selected_generator_union(root, stage=1)
        out = write_selected_union(root, payload)
        print(f"SELECTED_GENERATOR_UNION ({len(payload['selected_generator_union'])} generators): {payload['selected_generator_union']}")
        print(f"written: {out.relative_to(root)}")
        if payload["n_completed_jobs_considered"] == 0:
            print("WARNING: 0 completed Stage 1 validation jobs found; union is empty until the matrix actually runs.")
    else:
        payload = finalize_stage2_panels(root)
        out_dir = root / "results/final_evaluation_v2"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "primary_finalists_manifest.json"
        atomic_json(out_path, payload)
        print(f"written: {out_path.relative_to(root)}")


if __name__ == "__main__":
    main()
