#!/usr/bin/env python3
"""Freeze the full experiment matrix, checkpoints, thresholds and test manifest (spec section 15).

    python scripts/finalize_locked_test_stage.py --confirm-locked-test

This is the ONLY script that produces results/final_evaluation_v2/EXPERIMENT_MATRIX_LOCKED. It
does not read a single test-set prediction or metric — it only computes and signs content-aware
signatures over everything that must not change afterwards (dataset registry, experiment matrix,
the three seed checkpoints and validation predictions of every primary finalist and secondary
panel entry, the ensemble validation threshold, and the test dataset manifest). The actual test
read happens later, in a v2 locked-test notebook that first calls `verify_lock_still_valid` and
refuses to proceed on any mismatch.

Requires --confirm-locked-test: this command is the last reversible step before the test set is
read once. Running it does NOT itself open the test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

LOCK_DIR = "results/final_evaluation_v2"


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def preconditions(root: Path) -> list[str]:
    problems = []
    union_path = root / "results/generator_comparison/selected_generator_union.json"
    if not union_path.is_file():
        problems.append("missing results/generator_comparison/selected_generator_union.json (run finalize_validation_stage.py --stage 1 first)")
    else:
        union = json.loads(union_path.read_text())
        if not union.get("selected_generator_union"):
            problems.append("SELECTED_GENERATOR_UNION is empty: Stage 1 has not actually completed any validation yet")

    finalists_path = root / LOCK_DIR / "primary_finalists_manifest.json"
    if not finalists_path.is_file():
        problems.append(f"missing {LOCK_DIR}/primary_finalists_manifest.json (run finalize_validation_stage.py --stage 2 first)")

    test_csv = root / "data/processed/metadata/test.csv"
    if not test_csv.is_file():
        problems.append("missing data/processed/metadata/test.csv")

    return problems


def build_experiment_matrix_manifest(root: Path) -> dict:
    matrix_path = root / "configs/classifier_experiment_matrix.json"
    registry_path = root / "configs/dataset_variant_registry.json"
    return {
        "schema_version": 1,
        "classifier_experiment_matrix_sha256": _sha256_file(matrix_path),
        "dataset_variant_registry_sha256": _sha256_file(registry_path),
        "classifier_training_protocols_sha256": _sha256_file(root / "configs/classifier_training_protocols.json"),
        "final_generator_registry_sha256": _sha256_file(root / "configs/final_generator_registry.json"),
    }


def build_test_dataset_manifest(root: Path) -> dict:
    test_csv = root / "data/processed/metadata/test.csv"
    import csv
    with test_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    patient_ids = sorted({r["patient_id"] for r in rows})
    return {
        "schema_version": 1, "test_csv_sha256": _sha256_file(test_csv),
        "n_rows": len(rows), "n_unique_patients": len(patient_ids),
        "patient_id_set_sha256": hashlib.sha256("\n".join(patient_ids).encode("utf-8")).hexdigest(),
    }


def build_finalist_checkpoint_manifest(root: Path, matrix: dict, experiment_ids: list[str]) -> dict:
    from classifier_checkpoint_io import checkpoint_path as ckpt_path_for  # noqa: PLC0415
    by_id = {j["experiment_id"]: j for j in matrix["jobs"]}
    protocols = json.loads((root / "configs/classifier_training_protocols.json").read_text())["policies"]
    entries = {}
    for eid in experiment_ids:
        job = by_id.get(eid)
        if job is None:
            entries[eid] = {"status": "not_in_matrix"}
            continue
        framework = protocols[job["architecture"]]["framework"]
        run = root / job["manifest_path"]
        run = run.parent
        ckpt = ckpt_path_for(run, framework)
        entries[eid] = {"checkpoint_sha256": _sha256_file(ckpt), "status": job["status"]}
    return entries


def verify_lock_still_valid(root: Path) -> tuple[bool, list[str]]:
    lock_dir = root / LOCK_DIR
    marker = lock_dir / "EXPERIMENT_MATRIX_LOCKED"
    if not marker.is_file():
        return False, ["no lock present: run finalize_locked_test_stage.py --confirm-locked-test first"]

    manifest = json.loads((lock_dir / "experiment_matrix_manifest.json").read_text())
    problems = []
    current = build_experiment_matrix_manifest(root)
    for key, recorded in manifest.items():
        if key == "schema_version":
            continue
        if current.get(key) != recorded:
            problems.append(f"{key} changed since lock: recorded={recorded} current={current.get(key)}")

    test_manifest = json.loads((lock_dir / "test_dataset_manifest.json").read_text())
    current_test = build_test_dataset_manifest(root)
    if current_test != {k: v for k, v in test_manifest.items() if k != "schema_version"} | {"schema_version": test_manifest["schema_version"]}:
        if current_test["test_csv_sha256"] != test_manifest["test_csv_sha256"]:
            problems.append("test dataset changed since lock (test.csv sha256 mismatch)")
        if current_test["patient_id_set_sha256"] != test_manifest["patient_id_set_sha256"]:
            problems.append("test patient set changed since lock")

    checkpoints = json.loads((lock_dir / "primary_finalists_checkpoints.json").read_text())
    matrix = json.loads((root / "configs/classifier_experiment_matrix.json").read_text())
    current_checkpoints = build_finalist_checkpoint_manifest(root, matrix, list(checkpoints.keys()))
    for eid, recorded in checkpoints.items():
        if current_checkpoints.get(eid, {}).get("checkpoint_sha256") != recorded.get("checkpoint_sha256"):
            problems.append(f"checkpoint for {eid} changed or is missing since lock")

    return (len(problems) == 0), problems


def finalize(root: Path) -> dict:
    lock_dir = root / LOCK_DIR
    lock_dir.mkdir(parents=True, exist_ok=True)

    finalists = json.loads((lock_dir / "primary_finalists_manifest.json").read_text())
    matrix = json.loads((root / "configs/classifier_experiment_matrix.json").read_text())

    experiment_matrix_manifest = build_experiment_matrix_manifest(root)
    test_dataset_manifest = build_test_dataset_manifest(root)
    secondary_panel = {"schema_version": 2, "experiment_ids": finalists.get("secondary_locked_panel", [])}
    ablation_panel = {"schema_version": 1, "experiment_ids": finalists.get("ablation_panel", [])}
    primary_ids = []
    primary_seed_ids = []
    for architecture_panel in finalists.get("primary_finalists", {}).values():
        if not isinstance(architecture_panel, dict):
            continue
        # schema v2: one entry per preregistered comparison category. Keep compatibility with
        # the older single-best schema while never counting a missing placeholder as a finalist.
        if architecture_panel.get("experiment_id"):
            primary_ids.append(architecture_panel["experiment_id"])
            primary_seed_ids.extend(architecture_panel.get("seed_experiment_ids", []))
        for category in architecture_panel.values():
            if isinstance(category, dict) and category.get("experiment_id"):
                primary_ids.append(category["experiment_id"])
                primary_seed_ids.extend(category.get("seed_experiment_ids", []))
    primary_ids = sorted(set(primary_ids))
    all_locked_ids = sorted(set(primary_seed_ids) | set(secondary_panel["experiment_ids"]) |
                            set(ablation_panel["experiment_ids"]))
    checkpoints = build_finalist_checkpoint_manifest(root, matrix, all_locked_ids)

    (lock_dir / "experiment_matrix_manifest.json").write_text(json.dumps(experiment_matrix_manifest, ensure_ascii=False, indent=1) + "\n")
    (lock_dir / "test_dataset_manifest.json").write_text(json.dumps(test_dataset_manifest, ensure_ascii=False, indent=1) + "\n")
    (lock_dir / "secondary_panel_manifest.json").write_text(json.dumps(secondary_panel, ensure_ascii=False, indent=1) + "\n")
    (lock_dir / "primary_panel_manifest.json").write_text(json.dumps({"schema_version": 2, "experiment_ids": primary_ids}, ensure_ascii=False, indent=1) + "\n")
    (lock_dir / "ablation_panel_manifest.json").write_text(json.dumps(ablation_panel, ensure_ascii=False, indent=1) + "\n")
    (lock_dir / "primary_finalists_checkpoints.json").write_text(json.dumps(checkpoints, ensure_ascii=False, indent=1) + "\n")

    lock_signature = _sha256_json({**experiment_matrix_manifest, **test_dataset_manifest, "checkpoints": checkpoints})
    marker_payload = {"locked_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "lock_signature": lock_signature,
                       "n_primary_finalists": len(primary_ids), "n_secondary_panel": len(secondary_panel["experiment_ids"])}
    (lock_dir / "EXPERIMENT_MATRIX_LOCKED").write_text(json.dumps(marker_payload, ensure_ascii=False, indent=1) + "\n")
    return marker_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-locked-test", action="store_true",
                         help="Required. Without this flag the script only reports readiness and writes nothing.")
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.project_root)

    problems = preconditions(root)
    if problems:
        print("NOT READY to lock:")
        for p in problems:
            print(f" - {p}")
        raise SystemExit(1)

    if not args.confirm_locked_test:
        print("Preconditions satisfied. Re-run with --confirm-locked-test to write the permanent lock. "
              "This does NOT open the test set by itself.")
        return

    marker = finalize(root)
    print("LOCKED.")
    print(json.dumps(marker, indent=1))
    print(f"\nNext step is a separate, explicit real-run of the v2 locked-test notebook — not this script.")


if __name__ == "__main__":
    main()
