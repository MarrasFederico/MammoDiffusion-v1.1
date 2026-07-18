"""Generate the size-matched perceptual-hash audit and the Option B amendment artifacts.

Consumes only already-computed benchmark CSVs, cached embeddings, and the image files needed for the
perceptual-hash / SSIM diagnostics. It never loads a feature encoder, re-extracts embeddings,
regenerates images, selects a generator by itself, or reads the test split.

Outputs (runtime artifacts, git-ignored) under
``results/2_diffusers/benchmark/gate_audit/``:

* ``original_outcome_identity.json``  — run_id, HEAD and SHA-256 of the frozen originals;
* ``phash_size_matched_repetitions.csv`` / ``phash_size_matched_summary.csv``;
* ``amended_gate_results.csv`` / ``amended_generator_ranking.csv`` / ``amendment_decision.json``.

Run from the repository root: ``python notebooks/utility/run_gate_amendment.py``.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / "configs").is_dir())
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import generator_benchmark as gb  # noqa: E402
import gate_audit as ga  # noqa: E402

BENCHMARK = ROOT / gb.BENCHMARK_ROOT
AUDIT = BENCHMARK / "gate_audit"
PHASH_CACHE = AUDIT / "phash_cache"
RUN_ID = "generator_benchmark_20260714T221055Z_cd05886c"
BENCHMARK_HEAD = "cd05886c0e7044325063d9e2db4bf2de6d285dc4"

OFFICIAL = ["02_sd21_filtered_100steps", "03_sd21_vae_finetuned", "04_sd21_lora",
            "07_ldm_sdvae_extra1361", "08_ldm_v3_sdvae_fromscratch"]
FINETUNED = {"02_sd21_filtered_100steps", "03_sd21_vae_finetuned", "04_sd21_lora"}
STABILITY_REPS = 200
SEED = 20260714


def short_id(generator_id: str) -> str:
    return "G" + str(generator_id)[:2]


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


# ---------------------------------------------------------------------------
# 1. Original outcome identity (SHA-256 + run_id).
# ---------------------------------------------------------------------------

def original_outcome_identity() -> dict:
    def sha(path: Path) -> str | None:
        return gb.file_sha256(path) if path.is_file() else None

    identity = {
        "benchmark_run_id": RUN_ID,
        "benchmark_HEAD": BENCHMARK_HEAD,
        "generator_summary_sha256": sha(AUDIT / "original_generator_summary.csv"),
        "generator_ranking_sha256": sha(AUDIT / "original_generator_ranking.csv"),
        "distribution_repetitions_sha256": sha(BENCHMARK / "distribution_metrics_repetitions.csv"),
        "protocol_sha256": sha(AUDIT / "original_protocol.json"),
        "original_outcome": {"official_candidates_measured": 5, "eligible_under_original_gates": 0,
                             "maximum_perceptual_duplicate_rate": 0.02, "minimum_rad_dino_coverage": 0.5}}
    (AUDIT / "original_outcome_identity.json").write_text(json.dumps(identity, indent=1))
    return identity


# ---------------------------------------------------------------------------
# 2. Size-matched perceptual-hash audit.
# ---------------------------------------------------------------------------

def _candidate_filtered_paths(entry: dict) -> list[Path]:
    provenance = json.loads((ROOT / entry["provenance_manifest"]).read_text())
    paths, _, _ = gb.canonical_samples_from_manifest(ROOT, provenance["filtered_sample_manifest"])
    return [Path(path) for path in paths]


def _real_pools() -> tuple[list[Path], list[Path]]:
    corpus_meta = json.loads(next((BENCHMARK / "embedding_cache" / "shared_training_corpora")
                                  .glob("*/rad_dino.npy.metadata.json")).read_text())
    train_pos = [Path(p) for i, p in zip(corpus_meta["image_ids"], corpus_meta["image_paths"])
                 if "::real::" in i and "/train/1/" in p]
    validation_meta = json.loads((BENCHMARK / "embedding_cache/_references/validation/rad_dino.npy.metadata.json").read_text())
    validation_pos = [Path(p) for p in validation_meta["image_paths"]]
    return train_pos, validation_pos


def size_matched_phash_audit(registry: dict) -> tuple[list[dict], list[dict]]:
    by_id = {g["id"]: g for g in registry["generators"]}
    cache = ga.PerceptualHashCache(PHASH_CACHE)
    per_rep: list[dict] = []
    summary: list[dict] = []

    def emit(label: str, comparison: str, evidence: dict, subset_size: int, repetitions: int, seed: int):
        rows = ga.repeated_size_matched_phash(evidence, subset_size=subset_size,
                                              repetitions=repetitions, seed=seed)
        for row in rows:
            per_rep.append({"pool_or_generator": label, "comparison": comparison, **row})
        for metric in ga.SIZE_MATCHED_METRICS:
            summary.append({"pool_or_generator": label, "comparison": comparison, "sample_size": subset_size,
                            **ga.summarize_size_matched(rows, metric)})

    train_pos, validation_pos = _real_pools()
    for path in train_pos + validation_pos:
        gb._reject_test_path(path)  # never touch the test split

    print("  building real pool evidence")
    validation_evidence = ga.phash_pool_evidence(validation_pos, cache=cache)
    train_evidence = ga.phash_pool_evidence(train_pos, cache=cache)
    # Real baselines at matched sizes (full pools) and reduced-numerosity split-half.
    emit("real_validation_positive", "real_full_73", validation_evidence, 73, 1, SEED)
    emit("real_validation_positive", "real_split_half_36", validation_evidence, 36, STABILITY_REPS, SEED + 1)
    emit("real_train_positive", "real_full_340", train_evidence, 340, 1, SEED)
    emit("real_train_positive", "real_split_half_170", train_evidence, 170, STABILITY_REPS, SEED + 2)

    for generator_id in OFFICIAL:
        print(f"  size-matched pHash {short_id(generator_id)}")
        evidence = ga.phash_pool_evidence(_candidate_filtered_paths(by_id[generator_id]), cache=cache)
        emit(short_id(generator_id), "synthetic_size_matched_73", evidence, 73, STABILITY_REPS, SEED)
        emit(short_id(generator_id), "synthetic_size_matched_340", evidence, 340, STABILITY_REPS, SEED)
    cache.save()
    return per_rep, summary


# ---------------------------------------------------------------------------
# 7. Amended (Option B) gate results, ranking and decision record.
# ---------------------------------------------------------------------------

def amended_results(summary_rows: list[dict], amendment: dict) -> tuple[list[dict], list[dict], dict]:
    gates = amendment["new_blocking_gates"]
    original_gates = json.loads((AUDIT / "original_protocol.json").read_text())["eligibility_gates"]
    confirmed_by_gen = {row["generator_id"]: float(row["confirmed_duplicate_rate"])
                        for row in read_csv(AUDIT / "perceptual_hash_diagnostics.csv")}

    official = [row for row in summary_rows
                if row["generator_id"] in OFFICIAL and row["condition"] == "FILTERED"]
    enriched = []
    for row in official:
        confirmed = confirmed_by_gen.get(short_id(row["generator_id"]))
        merged = dict(row)
        merged["confirmed_duplicate_rate"] = confirmed
        original_failures = gb.eligibility_failures(row, original_gates)
        amended_failures = ga.amended_safety_gate_failures(merged, gates, confirmed_duplicate_rate=confirmed)
        merged["family"] = "finetuned" if row["generator_id"] in FINETUNED else "from_scratch"
        merged["original_gate_failures"] = original_failures
        merged["amended_safety_gate_failures"] = amended_failures
        enriched.append(merged)

    results = []
    ranking_rows = []
    for family in ("finetuned", "from_scratch"):
        ranked = ga.descriptive_generator_ranking(
            [{**row, "raddino_kid": gb._metric_value(row, "raddino_kid"),
              "raddino_coverage": gb._metric_value(row, "raddino_coverage", default=float("-inf")),
              "raddino_precision": gb._metric_value(row, "raddino_precision", default=float("-inf")),
              "raddino_fid": gb._metric_value(row, "raddino_fid"),
              "inception_kid": gb._metric_value(row, "inception_kid"),
              "raddino_kid_std": gb._metric_value(row, "raddino_kid_std")}
             for row in enriched], family)
        for row in ranked:
            base = next(r for r in enriched if r["generator_id"] == row["generator_id"])
            record = {
                "generator_id": short_id(row["generator_id"]), "full_generator_id": row["generator_id"],
                "family": family,
                "original_gate_eligible": not base["original_gate_failures"],
                "original_gate_failures": ";".join(base["original_gate_failures"]),
                "amended_safety_gate_eligible": not base["amended_safety_gate_failures"],
                "amended_safety_gate_failures": ";".join(base["amended_safety_gate_failures"]),
                "descriptive_family_rank": row["descriptive_family_rank"],
                "selection_metric": "raddino_kid",
                "selection_metric_value": gb._metric_value(base, "raddino_kid")}
            results.append(record)
            ranking_rows.append({k: record[k] for k in
                                 ("family", "generator_id", "descriptive_family_rank", "selection_metric",
                                  "selection_metric_value", "amended_safety_gate_eligible")})

    decision = {
        "amendment_version": amendment["amendment_version"], "selected_policy": "B",
        "status": amendment["status"], "post_benchmark_amendment": True,
        "benchmark_run_id": RUN_ID, "benchmark_HEAD": BENCHMARK_HEAD, "test_access": False,
        "all_official_candidates_pass_safety_gates": all(r["amended_safety_gate_eligible"] for r in results),
        "n_eligible_under_original_gates": sum(r["original_gate_eligible"] for r in results),
        "descriptive_ranking": {
            "finetuned": [r["generator_id"] for r in results if r["family"] == "finetuned"],
            "from_scratch": [r["generator_id"] for r in results if r["family"] == "from_scratch"]},
        "proposed_selection": {"finetuned": "02_sd21_filtered_100steps",
                               "from_scratch": "07_ldm_sdvae_extra1361"}}
    return results, ranking_rows, decision


# ---------------------------------------------------------------------------
# 3. Coverage reporting: point / stability / pass-fraction kept distinct.
# ---------------------------------------------------------------------------

def coverage_stability_summary(summary_rows: list[dict]) -> list[dict]:
    repetitions = read_csv(BENCHMARK / "distribution_metrics_repetitions.csv")
    point_by_gen = {row["generator_id"]: gb._metric_value(row, "raddino_coverage", default=float("nan"))
                    for row in summary_rows if row["condition"] == "FILTERED"}
    rows = []
    for generator_id in OFFICIAL:
        coverage = [float(r["coverage"]) for r in repetitions
                    if r["generator_id"] == generator_id and r["condition"] == "FILTERED"
                    and r["extractor"] == "rad_dino" and r.get("coverage") not in (None, "")]
        stats = gb.summarize(coverage)
        above = sum(1 for value in coverage if value >= 0.5) / len(coverage) if coverage else 0.0
        rows.append({"generator_id": short_id(generator_id), "full_generator_id": generator_id,
                     "coverage_balanced_point": point_by_gen.get(generator_id),
                     "coverage_stability_mean": stats["mean"], "coverage_stability_median": stats["median"],
                     "coverage_stability_interval_low": stats["percentile_2_5"],
                     "coverage_stability_interval_high": stats["percentile_97_5"],
                     "fraction_of_repetitions_above_0_5": above,
                     "coverage_used_as_binary_gate": False})
    return rows


def main() -> None:
    registry = gb.load_registry(ROOT)
    amendment = json.loads((ROOT / "configs/generator_benchmark_protocol_amendment_v1.json").read_text())
    summary_rows = read_csv(BENCHMARK / "generator_summary.csv")
    AUDIT.mkdir(parents=True, exist_ok=True)

    print("[1/3] original outcome identity")
    original_outcome_identity()

    print("[2/3] size-matched perceptual-hash audit")
    per_rep, summary = size_matched_phash_audit(registry)
    gb.write_csv_rows(AUDIT / "phash_size_matched_repetitions.csv", per_rep)
    gb.write_csv_rows(AUDIT / "phash_size_matched_summary.csv", summary)

    print("[3/4] coverage stability reporting (point / mean / interval / pass fraction)")
    gb.write_csv_rows(AUDIT / "coverage_stability_summary.csv", coverage_stability_summary(summary_rows))

    print("[4/4] amended gate results + ranking + decision")
    results, ranking_rows, decision = amended_results(summary_rows, amendment)
    gb.write_csv_rows(AUDIT / "amended_gate_results.csv", results)
    gb.write_csv_rows(AUDIT / "amended_generator_ranking.csv", ranking_rows)
    (AUDIT / "amendment_decision.json").write_text(json.dumps(decision, indent=1))
    print("  all pass safety gates:", decision["all_official_candidates_pass_safety_gates"],
          "| eligible under original gates:", decision["n_eligible_under_original_gates"])
    print("  descriptive ranking:", decision["descriptive_ranking"])
    print("done ->", AUDIT)


if __name__ == "__main__":
    main()
