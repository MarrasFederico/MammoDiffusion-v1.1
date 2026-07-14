#!/usr/bin/env python3
"""Generate Stage-2 notebooks only from a non-empty, content-signed Stage-1 union."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
from dataset_variant_registry import build_stage2_variants  # noqa: E402
from classifier_pipeline_contracts import verify_signed_payload  # noqa: E402

_spec = importlib.util.spec_from_file_location("stage1_notebooks", ROOT / "scripts/create_classifier_matrix_notebooks.py")
stage1 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(stage1)


def verify_union(payload: dict) -> list[str]:
    verify_signed_payload(payload)
    if payload.get("artifact_type") != "classifier_selected_generator_union":
        raise ValueError("not a classifier-matrix v2 SELECTED_GENERATOR_UNION")
    union = payload.get("selected_generator_union") or []
    if not union:
        raise ValueError("SELECTED_GENERATOR_UNION is empty; complete and finalize Stage 1 first")
    if payload.get("selection_used_test_data") is not False:
        raise ValueError("Stage 2 union must explicitly prove that no test data was used")
    if payload.get("scientific_completion", {}).get("complete") is not True:
        raise ValueError("Stage 1 completion is not certified by the selected union")
    leaderboard = payload.get("leaderboard") or {}
    verify_signed_payload(leaderboard)
    if leaderboard.get("signature") != payload.get("leaderboard_signature"):
        raise ValueError("selected union is not bound to its Stage 1 leaderboard")
    return union


def generate(root: Path, union_path: Path) -> list[dict]:
    union_payload = json.loads(union_path.read_text())
    union = verify_union(union_payload)
    variants = build_stage2_variants(root, union)
    if not variants:
        raise ValueError("signed union produced no executable Stage 2 variants")
    rows = []
    for architecture, (prefix, _) in stage1.ARCHITECTURES.items():
        for variant in variants:
            status, blocker = stage1.dataset_audit(root, variant)
            vid = variant["dataset_variant_id"]
            path = root / "notebooks/3_classifiers_matrix" / architecture / f"{prefix}_{vid}.ipynb"
            notebook_payload = stage1.notebook(architecture, variant, status, blocker, stage=2)
            notebook_payload["metadata"]["mammodiffusion"]["selected_union_signature"] = union_payload["signature"]
            stage1.write_notebook(path, notebook_payload)
            rows.append({
                "path": str(path.relative_to(root)), "experiment_id": f"{architecture}__{vid}",
                "architecture": architecture, "dataset_variant": vid, "stage": 2,
                "regime": variant.get("budget_regime"), "generator": variant.get("synthetic_generator_id"),
                "training_policy": f"{architecture}_standard", "seeds": "17,42,73",
                "dataset_status": status, "checkpoint_legacy_available": False, "training_required": True,
                "validation_available": False, "test_allowed": False, "compile_status": "PASS",
                "dry_run_status": "PLAN_PASS" if status == "READY" else "BLOCKED",
                "generator_ownership": "scripts/create_classifier_stage2_notebooks.py", "note_blocker": blocker,
            })
    inventory_path = root / "results/notebook_inventory/notebook_inventory.json"
    existing = json.loads(inventory_path.read_text()) if inventory_path.is_file() else []
    stage1.write_inventory(root, [row for row in existing if int(row["stage"]) != 2] + rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-union", default="results/generator_comparison/selected_generator_union.json")
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.project_root)
    union_path = Path(args.selected_union)
    if not union_path.is_absolute(): union_path = root / union_path
    rows = generate(root, union_path)
    print(f"Stage 2 notebooks: {len(rows)}")


if __name__ == "__main__":
    main()
