#!/usr/bin/env python3
"""Generate configs/classifier_experiment_matrix.json (spec section 9).

    python scripts/build_classifier_experiment_matrix.py --stage 1
    python scripts/build_classifier_experiment_matrix.py --stage 2 \\
        --selected-union results/generator_comparison/selected_generator_union.json

Stage 1 covers every architecture x every ready base/stage1_screening/legacy_compatible dataset
variant x 3 seeds. Stage 2 only builds jobs for the generators in a SELECTED_GENERATOR_UNION that
has already been computed and signed by finalize_validation_stage.py --stage 1 — it refuses to
run otherwise, so nobody can accidentally hand-pick Stage 2 generators from outside the
validation-only selection process (spec 3.5).

Existing jobs are never blindly reset to PENDING: status is reconstructed from on-disk artifacts
(spec: "Lo stato deve essere ricostruibile dagli artefatti"), so rebuilding the matrix after some
training has already happened does not lose progress.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import classifier_checkpoint_io as ckio  # noqa: E402
import classifier_run_manifest as crm  # noqa: E402
from classifier_pipeline_contracts import PIPELINE_NAMESPACE, atomic_json  # noqa: E402
from dataset_variant_registry import build_stage2_variants  # noqa: E402

SEEDS = (17, 42, 73)
STAGE1_REGIMES = {"base", "stage1_screening", "legacy_compatible"}
STAGE2_REGIMES = {"stage2_advanced"}

# Heaviest declared phase profile stands for the whole job (spec 10.3): a job is only as
# light as its most demanding phase.
_PROFILE_RANK = {"light": 0, "medium": 1, "heavy": 2, "exclusive": 3}


def _job_resource_profile(policy: dict) -> str:
    phases = policy.get("resource_profile_by_phase", {})
    if not phases:
        return "medium"
    return max(phases.values(), key=lambda p: _PROFILE_RANK.get(p, 1))


def _gpu_eligibility(profile: str) -> list[str]:
    if profile in ("heavy", "exclusive"):
        return ["rtx_5060_ti_16gb"]
    return ["rtx_5060_ti_16gb", "rtx_3060_12gb"]


def build_jobs(root: Path, stage: int, selected_union: list[str] | None = None) -> list[dict]:
    dataset_registry = json.loads((root / "configs/dataset_variant_registry.json").read_text())
    protocols = json.loads((root / "configs/classifier_training_protocols.json").read_text())["policies"]

    if stage == 1:
        variants = [v for v in dataset_registry["variants"] if v["regime"] in STAGE1_REGIMES and v["status"] in ("ready", "legacy")]
    elif stage == 2:
        if selected_union is None:
            raise ValueError("Stage 2 requires --selected-union pointing at a signed SELECTED_GENERATOR_UNION file; "
                              "run scripts/finalize_validation_stage.py --stage 1 first.")
        variants = build_stage2_variants(root, selected_union)
        variants = [v for v in variants if v["status"] == "ready"]
    else:
        raise ValueError(f"unsupported stage: {stage}")

    jobs = []
    for architecture, policy in protocols.items():
        profile = _job_resource_profile(policy)
        training_policy_name = f"{architecture}_standard"
        for variant in variants:
            for seed in SEEDS:
                experiment_id = ckio.experiment_id(architecture, variant["dataset_variant_id"], seed)
                run = ckio.run_dir(root, architecture, variant["dataset_variant_id"], training_policy_name, seed)
                results = ckio.results_dir(root, architecture, variant["dataset_variant_id"], training_policy_name, seed)
                state = crm.reconstruct_state(run, policy["framework"])
                jobs.append({
                    "experiment_id": experiment_id,
                    "stage": stage,
                    "architecture": architecture,
                    "dataset_variant_id": variant["dataset_variant_id"],
                    "training_policy": training_policy_name,
                    "seed": seed,
                    "status": state["state"],
                    "resource_profile": profile,
                    "estimated_vram_mb": None,
                    "gpu_eligibility": _gpu_eligibility(profile),
                    "checkpoint_path": str(ckio.checkpoint_path(run, policy["framework"]).relative_to(root)),
                    "validation_predictions_path": str((ckio.results_dir(
                        root, architecture, variant["dataset_variant_id"], training_policy_name, seed) /
                        f"validation_predictions_seed_{seed}.json").relative_to(root)),
                    "manifest_path": str(crm.manifest_path(run).relative_to(root)),
                })
    return jobs


def build_and_write(root: Path, stage: int, selected_union: list[str] | None = None,
                    selected_union_signature: str | None = None) -> dict:
    if stage == 2:
        if selected_union is None:
            raise ValueError("Stage 2 requires a signed selected generator union")
        registry_path = root / "configs/dataset_variant_registry.json"
        registry = json.loads(registry_path.read_text())
        stage2_variants = build_stage2_variants(root, selected_union)
        registry["variants"] = [v for v in registry["variants"] if v.get("regime") != "stage2_advanced"] + stage2_variants
        atomic_json(registry_path, registry)
    jobs = build_jobs(root, stage, selected_union)
    out_path = root / "configs/classifier_experiment_matrix.json"
    existing = {"schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE, "jobs": []}
    if out_path.is_file():
        existing = json.loads(out_path.read_text())
    other_stage_jobs = [j for j in existing.get("jobs", []) if j["stage"] != stage]
    payload = {"schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE, "jobs": other_stage_jobs + jobs}
    if stage == 2:
        payload["stage2_source_union_signature"] = selected_union_signature
    atomic_json(out_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--selected-union", default=None, help="Path to a signed SELECTED_GENERATOR_UNION json (required for --stage 2)")
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args()

    root = Path(args.project_root)
    selected_union = None
    selected_union_signature = None
    if args.selected_union:
        payload = json.loads((root / args.selected_union).read_text()) if not Path(args.selected_union).is_absolute() \
            else json.loads(Path(args.selected_union).read_text())
        from create_classifier_stage2_notebooks import verify_union
        selected_union = verify_union(payload)
        selected_union_signature = payload["signature"]

    payload = build_and_write(root, args.stage, selected_union, selected_union_signature)
    stage_jobs = [j for j in payload["jobs"] if j["stage"] == args.stage]
    by_status: dict[str, int] = {}
    for j in stage_jobs:
        by_status[j["status"]] = by_status.get(j["status"], 0) + 1
    print(f"Stage {args.stage}: {len(stage_jobs)} jobs written to configs/classifier_experiment_matrix.json")
    print(json.dumps(by_status, indent=1))


if __name__ == "__main__":
    main()
