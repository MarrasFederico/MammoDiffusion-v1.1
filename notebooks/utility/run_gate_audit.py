"""Generate the post-benchmark gate calibration audit artifacts.

Consumes only already-computed benchmark CSVs, cached embeddings, and the image files needed for
perceptual-hash / SSIM diagnostics.  It never loads a feature encoder, never re-extracts embeddings,
never selects a generator, and never reads the test split.

Outputs (runtime artifacts, not committed) go under
``results/generator_benchmark/gate_audit/``.

Run from the repository root::

    python notebooks/utility/run_gate_audit.py

The active gate policy and the protocol configuration are left unchanged; the review only *proposes*
alternatives for human decision.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / "configs").is_dir())
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import generator_benchmark as gb  # noqa: E402
import gate_audit as ga  # noqa: E402

BENCHMARK = ROOT / gb.BENCHMARK_ROOT
AUDIT = BENCHMARK / "gate_audit"
PANELS = AUDIT / "phash_panels"

OFFICIAL = {"02_sd21_filtered_100steps", "03_sd21_vae_finetuned", "04_sd21_lora",
            "07_ldm_sdvae_extra1361", "08_ldm_v3_sdvae_fromscratch"}
FINETUNED = {"02_sd21_filtered_100steps", "03_sd21_vae_finetuned", "04_sd21_lora"}
FAMILY_PAIRS = {"finetuned": [("02_sd21_filtered_100steps", "03_sd21_vae_finetuned"),
                              ("02_sd21_filtered_100steps", "04_sd21_lora"),
                              ("03_sd21_vae_finetuned", "04_sd21_lora")],
                "from_scratch": [("07_ldm_sdvae_extra1361", "08_ldm_v3_sdvae_fromscratch")]}


def short_id(generator_id: str) -> str:
    return "G" + str(generator_id)[:2]


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_embeddings(path: Path) -> tuple[np.ndarray, dict]:
    features = np.load(path)
    metadata = json.loads(Path(str(path) + ".metadata.json").read_text())
    return features, metadata


# ---------------------------------------------------------------------------
# 3. Freeze the original gate outcome (never overwrite).
# ---------------------------------------------------------------------------

def freeze_original(summary_rows: list[dict], gates: dict, protocol: dict) -> dict:
    AUDIT.mkdir(parents=True, exist_ok=True)
    frozen = {}
    for name, source in (("original_generator_summary.csv", BENCHMARK / "generator_summary.csv"),
                         ("original_generator_ranking.csv", BENCHMARK / "generator_ranking.csv")):
        target = AUDIT / name
        if not target.exists() and source.exists():
            shutil.copy2(source, target)
        frozen[name] = target.exists()
    protocol_target = AUDIT / "original_protocol.json"
    if not protocol_target.exists():
        protocol_target.write_text(json.dumps(protocol, indent=1))
    gate_target = AUDIT / "original_gate_results.json"
    if not gate_target.exists():
        official = [row for row in summary_rows
                    if row["generator_id"] in OFFICIAL and row["condition"] == "FILTERED"]
        results = {"minimum_rad_dino_coverage": gates["minimum_rad_dino_coverage"],
                   "maximum_perceptual_duplicate_rate": gates["maximum_perceptual_duplicate_rate"],
                   "n_official_candidates": len(official), "candidates": {}}
        for row in official:
            failures = gb.eligibility_failures(row, gates)
            results["candidates"][short_id(row["generator_id"])] = {
                "generator_id": row["generator_id"],
                "perceptual_hash_duplicate_rate": float(row["perceptual_hash_duplicate_rate"]),
                "raddino_coverage": float(row["raddino_coverage"]),
                "failed_gates": failures,
                "eligible": not failures}
        gate_target.write_text(json.dumps(results, indent=1))
    return frozen


# ---------------------------------------------------------------------------
# 4-5. Perceptual-hash diagnostics, inspectable pairs, panels.
# ---------------------------------------------------------------------------

def candidate_filtered_paths(entry: dict) -> tuple[list[Path], list[str]]:
    provenance = json.loads((ROOT / entry["provenance_manifest"]).read_text())
    paths, ids, _ = gb.canonical_samples_from_manifest(ROOT, provenance["filtered_sample_manifest"])
    return [Path(path) for path in paths], list(ids)


def basename_embedding_map(cache_path: Path) -> dict[str, np.ndarray]:
    if not cache_path.exists():
        return {}
    features, metadata = load_embeddings(cache_path)
    mapping = {}
    for image_id, vector in zip(metadata["image_ids"], features):
        mapping[Path(image_id.split("::")[-1]).name] = np.asarray(vector, dtype=np.float64)
    return mapping


def phash_audit(registry: dict) -> tuple[list[dict], list[dict]]:
    by_id = {g["id"]: g for g in registry["generators"]}
    diagnostics_rows: list[dict] = []
    pair_rows: list[dict] = []
    PANELS.mkdir(parents=True, exist_ok=True)
    for generator_id in sorted(OFFICIAL):
        paths, ids = candidate_filtered_paths(by_id[generator_id])
        hash_ints = ga.perceptual_hash_ints(paths)
        exact_hashes = [gb.file_sha256(path) for path in paths]
        diagnostics = ga.perceptual_hash_cluster_diagnostics(paths, hash_ints=hash_ints)
        confirmed = ga.confirmed_duplicate_analysis(paths, hash_ints=hash_ints, exact_hashes=exact_hashes)
        legacy = ga.legacy_order_dependent_phash_rate(hash_ints)
        component_of = _component_lookup(len(paths), ga.phash_neighbour_pairs(hash_ints))
        embedding_of = basename_embedding_map(
            BENCHMARK / "embedding_cache" / generator_id / "filtered" / "rad_dino.npy")

        for row in confirmed["pairs"]:
            left, right = row["left_index"], row["right_index"]
            left_emb = embedding_of.get(paths[left].name)
            right_emb = embedding_of.get(paths[right].name)
            rad = (float(np.linalg.norm(left_emb - right_emb))
                   if left_emb is not None and right_emb is not None else None)
            pair_rows.append({
                "generator_id": short_id(generator_id), "condition": "FILTERED",
                "left_sample_id": ids[left], "right_sample_id": ids[right],
                "left_path": gb._project_relative(ROOT, paths[left]),
                "right_path": gb._project_relative(ROOT, paths[right]),
                "phash_distance": row["phash_distance"], "ssim": row["ssim"],
                "rad_dino_distance": rad, "exact_hash_match": row["exact_hash_match"],
                "confirmed_duplicate": row["confirmed_duplicate"],
                "component_id": component_of[left]})

        diagnostics_rows.append({
            "generator_id": short_id(generator_id), "full_generator_id": generator_id,
            "condition": "FILTERED", **{k: diagnostics[k] for k in (
                "n_images", "n_phash_pairs", "n_images_with_any_phash_neighbour",
                "phash_any_neighbour_rate", "n_phash_connected_components",
                "n_nontrivial_phash_components", "largest_phash_component_size",
                "phash_component_excess_count", "phash_component_excess_rate")},
            "exact_duplicate_rate": confirmed["exact_duplicate_rate"],
            "confirmed_duplicate_rate": confirmed["confirmed_duplicate_rate"],
            "legacy_order_dependent_phash_rate": legacy})
        _render_panels(generator_id, paths, confirmed["pairs"], component_of)
        print(f"  phash {short_id(generator_id)}: any_neighbour={diagnostics['phash_any_neighbour_rate']:.4f} "
              f"excess={diagnostics['phash_component_excess_rate']:.4f} "
              f"confirmed={confirmed['confirmed_duplicate_rate']:.4f} legacy={legacy:.4f}")
    return diagnostics_rows, pair_rows


def _component_lookup(n_nodes: int, pairs) -> list[int]:
    components = ga.connected_components(n_nodes, pairs)
    lookup = [0] * n_nodes
    for component_id, group in enumerate(components):
        for node in group:
            lookup[node] = component_id
    return lookup


def _render_panels(generator_id: str, paths, pairs, component_of) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    near = [row for row in pairs if row["phash_distance"] <= 2 or row["exact_hash_match"]]
    closest = sorted(near, key=lambda row: (row["phash_distance"], -row["ssim"],
                                            row["left_index"], row["right_index"]))[:10]
    if not closest:
        return
    figure, axes = plt.subplots(len(closest), 2, figsize=(6, 3 * len(closest)), squeeze=False)
    for index, row in enumerate(closest):
        for column, side in enumerate(("left_index", "right_index")):
            with Image.open(paths[row[side]]) as image:
                axes[index, column].imshow(image.convert("L"), cmap="gray")
            axes[index, column].axis("off")
        axes[index, 0].set_title(f"phash={row['phash_distance']} ssim={row['ssim']:.3f} "
                                 f"{'EXACT' if row['exact_hash_match'] else ''}"
                                 f"{' CONFIRMED' if row['confirmed_duplicate'] else ''}", fontsize=8)
    figure.suptitle(f"{short_id(generator_id)} FILTERED — 10 closest pHash pairs")
    figure.tight_layout()
    figure.savefig(PANELS / f"{generator_id}_closest_phash_pairs.png", dpi=110)
    plt.close(figure)


# ---------------------------------------------------------------------------
# 6. Real perceptual-hash baselines (non-test only).
# ---------------------------------------------------------------------------

def real_phash_baselines() -> list[dict]:
    corpus_cache = next((BENCHMARK / "embedding_cache" / "shared_training_corpora").glob("*/rad_dino.npy.metadata.json"))
    metadata = json.loads(corpus_cache.read_text())
    train_pos = [Path(p) for i, p in zip(metadata["image_ids"], metadata["image_paths"])
                 if "::real::" in i and "/train/1/" in p]
    augment = [Path(p) for i, p in zip(metadata["image_ids"], metadata["image_paths"])
               if "::positive_augmentation::" in i]
    validation_meta = json.loads((BENCHMARK / "embedding_cache/_references/validation/rad_dino.npy.metadata.json").read_text())
    validation_pos = [Path(p) for p in validation_meta["image_paths"]]

    rows = []
    for pool, paths, is_augmentation in (("real_train_positive", train_pos, False),
                                         ("real_validation_positive", validation_pos, False),
                                         ("real_positive_augmentation", augment, True)):
        for path in paths:
            gb._reject_test_path(path)  # hard guarantee: never touch the test split
        diagnostics = ga.perceptual_hash_cluster_diagnostics(paths)
        confirmed = ga.confirmed_duplicate_analysis(paths)
        rows.append({"pool": pool, "is_augmentation_baseline": is_augmentation,
                     "n_images": diagnostics["n_images"],
                     "phash_any_neighbour_rate": diagnostics["phash_any_neighbour_rate"],
                     "phash_component_excess_rate": diagnostics["phash_component_excess_rate"],
                     "confirmed_duplicate_rate": confirmed["confirmed_duplicate_rate"],
                     "largest_component_size": diagnostics["largest_phash_component_size"]})
        print(f"  real phash {pool}: n={diagnostics['n_images']} "
              f"any={diagnostics['phash_any_neighbour_rate']:.4f} "
              f"excess={diagnostics['phash_component_excess_rate']:.4f} "
              f"confirmed={confirmed['confirmed_duplicate_rate']:.4f}")
    return rows


# ---------------------------------------------------------------------------
# 7. Real-vs-real RAD-DINO PRDC baselines (cached embeddings only).
# ---------------------------------------------------------------------------

def real_real_prdc_baselines(protocol: dict) -> tuple[list[dict], list[dict]]:
    validation, _ = load_embeddings(BENCHMARK / "embedding_cache/_references/validation/rad_dino.npy")
    corpus_dir = next((BENCHMARK / "embedding_cache" / "shared_training_corpora").glob("*/rad_dino.npy")).parent
    corpus, corpus_meta = load_embeddings(corpus_dir / "rad_dino.npy")
    train_pos_idx = [i for i, (image_id, path) in enumerate(zip(corpus_meta["image_ids"], corpus_meta["image_paths"]))
                     if "::real::" in image_id and "/train/1/" in path]
    train_positive = corpus[train_pos_idx]

    seed = int(protocol["sampling"]["seed"])
    reps = int(protocol["resampling"]["stability_repetitions"])
    k = int(protocol["resampling"]["nearest_neighbour_k"])

    per_rep: list[dict] = []
    summary: list[dict] = []
    # Baseline A: prdc(real=validation, synthetic=train-positive sample) mirrors the benchmark layout.
    rows_a = ga.repeated_prdc_baseline(validation, train_positive, subset_reference=len(validation),
                                       subset_candidate=len(validation), repetitions=reps, seed=seed, nearest_k=k)
    # Baseline B: repeated deterministic split-half of the validation positives.
    rows_b = ga.split_half_prdc_baseline(validation, repetitions=reps, seed=seed + 100000, nearest_k=k)
    for name, rows, note in (("A_real_train_vs_validation", rows_a,
                              "prdc(real=validation_positive, synthetic=real_train_positive sample of 73)"),
                             ("B_validation_split_half", rows_b,
                              "prdc on a deterministic 36/37 split of the 73 validation positives")):
        for row in rows:
            per_rep.append({"baseline": name, **row})
        entry = {"baseline": name, "definition": note, "repetitions": len(rows),
                 "reference_pool": "real_validation_positive",
                 "candidate_pool": "real_train_positive" if name.startswith("A") else "real_validation_positive_half"}
        for metric in ("coverage", "precision", "recall", "density"):
            entry.update(ga.summarize_metric(rows, metric))
        summary.append(entry)
        print(f"  prdc {name}: coverage mean={entry['coverage_mean']:.4f} "
              f"[{entry['coverage_percentile_2_5']:.4f}, {entry['coverage_percentile_97_5']:.4f}]")
    return per_rep, summary


# ---------------------------------------------------------------------------
# 9. Descriptive, gate-independent ranking.
# ---------------------------------------------------------------------------

def descriptive_ranking(summary_rows: list[dict], gates: dict) -> list[dict]:
    official = []
    for row in summary_rows:
        if row["generator_id"] not in OFFICIAL or row["condition"] != "FILTERED":
            continue
        enriched = dict(row)
        for field in ("raddino_kid", "raddino_coverage", "raddino_precision", "raddino_fid",
                      "inception_kid", "raddino_kid_std"):
            enriched[field] = float(row[field]) if row.get(field) not in (None, "") else math.inf
        failures = gb.eligibility_failures(row, gates)
        enriched["selection_eligible_under_original_gates"] = not failures
        enriched["original_gate_failures"] = ";".join(failures)
        official.append(enriched)

    ordered: list[dict] = []
    for family in ("finetuned", "from_scratch"):
        ranked = ga.descriptive_generator_ranking(official, family)
        for row in ranked:
            ordered.append({"family": family, "generator_id": short_id(row["generator_id"]),
                            "full_generator_id": row["generator_id"],
                            "descriptive_family_rank": row["descriptive_family_rank"],
                            "raddino_kid": row["raddino_kid"], "raddino_coverage": row["raddino_coverage"],
                            "raddino_precision": row["raddino_precision"], "raddino_fid": row["raddino_fid"],
                            "inception_kid": row["inception_kid"], "raddino_kid_std": row["raddino_kid_std"],
                            "selection_eligible_under_original_gates": row["selection_eligible_under_original_gates"],
                            "original_gate_failures": row["original_gate_failures"]})
    return ordered


# ---------------------------------------------------------------------------
# 10. Paired differences (descriptive; produced even with no eligible candidate).
# ---------------------------------------------------------------------------

def paired_differences(repetition_rows: list[dict], protocol: dict) -> list[dict]:
    margin = float(protocol["selection"]["practical_equivalence_margin"])
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in repetition_rows:
        if row["condition"] == "FILTERED" and row["extractor"] == "rad_dino":
            grouped[row["generator_id"]].append(row)
    output = []
    for family, pairs in FAMILY_PAIRS.items():
        for left, right in pairs:
            if left not in grouped or right not in grouped:
                continue
            paired = gb.paired_kid_differences(grouped[left], grouped[right], short_id(left), short_id(right))
            equivalence = gb.practical_equivalence(paired, protocol)
            output.append({
                "family": family, "left_generator": short_id(left), "right_generator": short_id(right),
                "mean_paired_kid_difference": paired["mean_paired_difference"],
                "median_paired_kid_difference": paired["median_paired_difference"],
                "stability_interval_low": paired["stability_interval_low"],
                "stability_interval_high": paired["stability_interval_high"],
                "left_win_fraction": paired["left_win_fraction"],
                "right_win_fraction": paired["right_win_fraction"],
                "practical_equivalence_margin": margin,
                "practically_similar": equivalence["practically_similar"]})
    return output


# ---------------------------------------------------------------------------
# 11. Corrected efficiency metrics.
# ---------------------------------------------------------------------------

def corrected_summary(summary_rows: list[dict], registry: dict) -> list[dict]:
    by_id = {g["id"]: g for g in registry["generators"]}
    efficiency_fields = ("generation_seconds_per_image", "peak_vram_mb", "energy_kwh",
                         "checkpoint_size_bytes", "efficiency_source", "efficiency_status",
                         "generation_efficiency_status")
    corrected = []
    for row in summary_rows:
        entry = dict(row)
        fixed = ga.efficiency_from_manifest_strict(ROOT, by_id[row["generator_id"]])
        for field in efficiency_fields:
            entry[field] = fixed.get(field)
        corrected.append(entry)
    return corrected


# ---------------------------------------------------------------------------
# 8. Policy proposal (does NOT change the active gates).
# ---------------------------------------------------------------------------

def write_policy(summary_rows: list[dict], gates: dict, phash_rows: list[dict],
                 prdc_summary: list[dict], real_phash: list[dict], descriptive: list[dict]) -> None:
    coverage_a = next(r for r in prdc_summary if r["baseline"].startswith("A"))
    coverage_b = next(r for r in prdc_summary if r["baseline"].startswith("B"))
    proposal = {
        "status": "proposal_only_not_applied",
        "active_gates_unchanged": True,
        "protocol_config_unchanged": "configs/generator_benchmark_protocol.json",
        "options": {
            "A_original_protocol": {
                "phash_only_gate": gates["maximum_perceptual_duplicate_rate"],
                "coverage_point_gate": gates["minimum_rad_dino_coverage"],
                "outcome": "no candidate selectable; downstream not runnable"},
            "B_safety_gates_coverage_ranking": {
                "blocking_gates": ["exact_duplicate_rate", "confirmed_duplicate_rate", "train_memorization",
                                   "corruption", "filtered_validity", "provenance", "lineage"],
                "descriptive_metrics": ["phash_any_neighbour_rate", "rad_dino_coverage",
                                        "precision", "recall", "density"],
                "new_coverage_threshold": None},
            "C_calibrated_thresholds": {
                "requires": "explicit derivation from real-real baselines",
                "real_train_vs_validation_coverage": {
                    "mean": coverage_a["coverage_mean"],
                    "percentile_2_5": coverage_a["coverage_percentile_2_5"],
                    "percentile_97_5": coverage_a["coverage_percentile_97_5"]},
                "validation_split_half_coverage": {
                    "mean": coverage_b["coverage_mean"],
                    "percentile_2_5": coverage_b["coverage_percentile_2_5"],
                    "percentile_97_5": coverage_b["coverage_percentile_97_5"]},
                "proposed_threshold": None,
                "note": "no threshold asserted; any amendment is post-benchmark and needs human approval"}},
        "decision_required": True}
    (AUDIT / "gate_policy_proposal.json").write_text(json.dumps(proposal, indent=1))

    def table(rows, columns):
        header = "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n"
        body = "".join("| " + " | ".join(_fmt(row.get(c)) for c in columns) + " |\n" for row in rows)
        return header + body

    lines = [
        "# Gate policy review (post-benchmark, proposal only)\n",
        "This review evaluates the eligibility gates **after** the benchmark has run. It does not "
        "change the active gate policy and does not modify `configs/generator_benchmark_protocol.json`. "
        "The three options below are presented for human decision.\n",
        "## Original outcome\n",
        "Five official candidates were measured (G02, G03, G04, G07, G08 — FILTERED). "
        "**Zero** are eligible under the original gates: every candidate fails both "
        f"`maximum_perceptual_duplicate_rate` ({gates['maximum_perceptual_duplicate_rate']}) and "
        f"`minimum_rad_dino_coverage` ({gates['minimum_rad_dino_coverage']}).\n",
        "## Perceptual-hash diagnostics (order-independent)\n",
        "The original `perceptual_hash_duplicate_rate` gate uses an **order-dependent** count "
        "(an image is flagged if any *earlier* image is within pHash distance 2). The audit replaces it "
        "with order-independent cluster diagnostics and a preregistered `confirmed_duplicate_rate` "
        "(`exact | (pHash<=2 AND SSIM>=0.98)`).\n",
        table(phash_rows, ["generator_id", "phash_any_neighbour_rate", "phash_component_excess_rate",
                           "confirmed_duplicate_rate", "exact_duplicate_rate",
                           "legacy_order_dependent_phash_rate", "largest_phash_component_size"]),
        "\n### Real (non-test) perceptual-hash baselines\n",
        "Natural mammography images are visually similar; the augmentation baseline is labelled "
        "separately and excluded from the primary comparison.\n",
        table(real_phash, ["pool", "n_images", "phash_any_neighbour_rate", "phash_component_excess_rate",
                           "confirmed_duplicate_rate", "largest_component_size"]),
        "\n## RAD-DINO coverage audit\n",
        "The active gate uses a single balanced-point coverage from one 73-image synthetic subset. "
        "The benchmark already carries 200 repeated-subsampling measurements. Real-vs-real baselines "
        "bound what coverage looks like when the 'synthetic' pool is itself real:\n",
        table(prdc_summary, ["baseline", "coverage_mean", "coverage_median",
                             "coverage_percentile_2_5", "coverage_percentile_97_5"]),
        "\n## Descriptive ranking (gate-independent)\n",
        table(descriptive, ["family", "generator_id", "descriptive_family_rank", "raddino_kid",
                            "raddino_coverage", "selection_eligible_under_original_gates"]),
        "\n## Options\n",
        "### Option A — original protocol\n",
        f"- pHash-only rate `<= {gates['maximum_perceptual_duplicate_rate']}` as a hard gate.\n"
        f"- Coverage balanced point `>= {gates['minimum_rad_dino_coverage']}` as a hard gate.\n"
        "- **Outcome:** no candidate is selectable; the downstream experiment cannot run.\n",
        "### Option B — safety gates + coverage as ranking\n",
        "- Blocking gates: exact duplicate rate, confirmed duplicate rate, train memorization, corruption, "
        "FILTERED validity, provenance, lineage.\n"
        "- Descriptive / ranking metrics: pHash-only, RAD-DINO coverage, precision, recall, density.\n"
        "- No new coverage threshold is asserted.\n",
        "### Option C — calibrated thresholds\n",
        "- Any coverage threshold must be derived explicitly from the real-real baselines above "
        "(state the formula, baseline, percentile, resulting pass list, and the post-benchmark nature "
        "of the amendment). This review does **not** assert one.\n",
        "\n## Constraints\n",
        "- `configs/generator_benchmark_protocol.json` is unchanged.\n"
        "- The active gates are unchanged.\n"
        "- No generator was selected; no downstream classifier was trained; the test split was not read.\n",
    ]
    (AUDIT / "gate_policy_review.md").write_text("\n".join(lines))


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


# ---------------------------------------------------------------------------

def main() -> None:
    protocol = gb.load_protocol(ROOT)
    registry = gb.load_registry(ROOT)
    gates = protocol["eligibility_gates"]
    summary_rows = read_csv(BENCHMARK / "generator_summary.csv")
    repetition_rows = read_csv(BENCHMARK / "distribution_metrics_repetitions.csv")

    AUDIT.mkdir(parents=True, exist_ok=True)
    print("[1/8] freeze original gate outcome")
    freeze_original(summary_rows, gates, protocol)

    print("[2/8] perceptual-hash diagnostics + inspectable pairs + panels")
    phash_rows, pair_rows = phash_audit(registry)
    gb.write_csv_rows(AUDIT / "perceptual_hash_diagnostics.csv", phash_rows)
    gb.write_csv_rows(AUDIT / "perceptual_hash_pairs.csv", pair_rows)

    print("[3/8] real perceptual-hash baselines")
    real_phash = real_phash_baselines()
    gb.write_csv_rows(AUDIT / "real_phash_baselines.csv", real_phash)

    print("[4/8] real-vs-real RAD-DINO PRDC baselines")
    prdc_rep, prdc_summary = real_real_prdc_baselines(protocol)
    gb.write_csv_rows(AUDIT / "real_real_prdc_baselines.csv", prdc_rep)
    gb.write_csv_rows(AUDIT / "real_real_prdc_summary.csv", prdc_summary)

    print("[5/8] descriptive gate-independent ranking")
    descriptive = descriptive_ranking(summary_rows, gates)
    gb.write_csv_rows(AUDIT / "descriptive_generator_ranking.csv", descriptive)

    print("[6/8] paired generator differences")
    paired = paired_differences(repetition_rows, protocol)
    gb.write_csv_rows(AUDIT / "paired_generator_differences.csv", paired)

    print("[7/8] corrected efficiency metrics")
    corrected = corrected_summary(summary_rows, registry)
    gb.write_csv_rows(AUDIT / "generator_summary.csv", corrected)
    if (BENCHMARK / "generator_ranking.csv").exists():
        shutil.copy2(BENCHMARK / "generator_ranking.csv", AUDIT / "generator_ranking.csv")

    print("[8/8] policy review + proposal")
    write_policy(summary_rows, gates, phash_rows, prdc_summary, real_phash, descriptive)
    print("done ->", AUDIT)


if __name__ == "__main__":
    main()
