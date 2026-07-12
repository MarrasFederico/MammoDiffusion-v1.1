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
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

GLOBAL_TOP_K_GENERATORS = 3
FAMILIES = ("resnet50", "maxvit512", "mammofm", "raddino")


def _generator_of(dataset_variant_id: str) -> str | None:
    """Primary Stage-1 screening is exclusively RSB_CONTROLLED ensemble validation."""
    prefix = "RSB_CONTROLLED_"
    return dataset_variant_id[len(prefix):] if dataset_variant_id.startswith(prefix) else None


def load_completed_validations(root: Path, stage: int) -> list[dict]:
    matrix = json.loads((root / "configs/classifier_experiment_matrix.json").read_text())
    if stage in (1, 2):
        rows = []
        for path in sorted((root / "results/classifiers_matrix").glob("*/*/*/ensemble_validation_manifest.json")):
            payload = json.loads(path.read_text())
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
        vmetrics_path = metrics_path.parent / "validation_metrics.json"
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
    aggregated.sort(key=lambda e: e["mean_pr_auc"], reverse=True)
    for rank, entry in enumerate(aggregated, start=1):
        entry["rank"] = rank
    return aggregated


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

    top_k = sorted(mean_ranks, key=lambda g: mean_ranks[g])[:GLOBAL_TOP_K_GENERATORS]
    family_winners = {ranking[0]["generator_id"] for ranking in per_family_ranking.values() if ranking}
    union = sorted(set(top_k) | family_winners)

    payload = {
        "schema_version": 1, "stage": stage, "global_top_k": GLOBAL_TOP_K_GENERATORS,
        "per_family_ranking": per_family_ranking, "mean_rank_by_generator": mean_ranks,
        "top_k_generators": top_k, "family_winners": sorted(family_winners),
        "selected_generator_union": union,
        "selection_used_test_data": False,
        "n_completed_jobs_considered": len(rows), "primary_screening_regime": "RSB_CONTROLLED",
        "aggregation_level": "three_seed_ensemble", "excluded_regimes": ["RSB_FULL", "RSP_CONTROLLED", "RSP_FULL"],
    }
    payload["signature"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return payload


def write_selected_union(root: Path, payload: dict) -> Path:
    out_dir = root / "results/generator_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selected_generator_union.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    return out_path


def finalize_stage2_panels(root: Path) -> dict:
    rows = load_completed_validations(root, stage=2)
    # Required baselines and Stage-1 RS/positive-only panels remain eligible comparators even
    # though Stage 2 itself only schedules RAS/S_ONLY variants.
    for path in sorted((root / "results/classifiers_matrix").glob("*/*/*/ensemble_validation_manifest.json")):
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
            selected[name] = (max(candidates, key=lambda r: (r["pr_auc"] or -1)) if candidates
                              else {"status": "missing_preregistered_validation"})
        primary_finalists[architecture] = selected
    secondary_panel = sorted({seed_id for row in rows for seed_id in row.get("seed_experiment_ids", [])})
    payload = {"schema_version": 2, "primary_finalists": primary_finalists,
               "secondary_locked_panel": secondary_panel, "ablation_panel": [],
               "n_completed_jobs_considered": len(rows), "test_data_used": False}
    payload["signature"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
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
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
        print(f"written: {out_path.relative_to(root)}")


if __name__ == "__main__":
    main()
