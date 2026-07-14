"""Canonical dataset-variant registry: builds and validates configs/dataset_variant_registry.json.

Generator identity/eligibility is imported from final_generator_registry.json (the single
source of truth for generators); this module only reasons about how real, augmented and
synthetic data are combined into a named, budgeted, train-only dataset variant. Nothing here
reads validation or test metrics: variant existence and counts come exclusively from train-side
evidence (real split metadata, augmentation directory, and each generator's own recorded
production counts).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from classifier_pipeline_contracts import PIPELINE_NAMESPACE, atomic_json

TWO_CLASS = ("negative", "positive")
CLASS_LABEL = {"negative": "0", "positive": "1"}
SAMPLING_SEED = 42

REAL_METADATA = "data/processed/metadata/train.csv"
AUGMENTED_DIR = "data/real_augmented"
AUGMENTED_LABEL_RE_GROUP = "label"


# --- generator introspection (final_generator_registry.json stays the single source of truth) ---

def load_generator_registry(root: Path) -> dict:
    return json.loads((root / "configs/final_generator_registry.json").read_text())


def load_classifier_registry(root: Path) -> dict:
    return json.loads((root / "configs/final_classifier_registry.json").read_text())


def generator_classes(entry: dict) -> tuple[str, ...]:
    return tuple(entry.get("classes", []))


def is_two_class(entry: dict) -> bool:
    return set(TWO_CLASS).issubset(set(generator_classes(entry)))


def is_positive_only(entry: dict) -> bool:
    return set(generator_classes(entry)) == {"positive"}


def is_usable(entry: dict) -> bool:
    return str(entry.get("status", "")).startswith("completed")


def _reroot_under_project(path_str: str, root: Path) -> Path:
    """Re-root a possibly-stale absolute path (a different machine/mount than this one)
    under the current project root, matching on the first recognizable project-relative
    anchor instead of trusting the stored absolute prefix.
    """
    raw = Path(path_str)
    for anchor in ("data", "experiments", "results", "notebooks"):
        if anchor in raw.parts:
            idx = raw.parts.index(anchor)
            return root.joinpath(*raw.parts[idx:])
    return raw if raw.is_absolute() else root / raw


def resolve_generator_class_count(root: Path, entry: dict, klass: str) -> dict:
    """Resolve how many accepted synthetic images generator `entry` produced for `klass`.

    Tries, in order of decreasing directness: the registry's own `produced_per_class` (only
    generators where a single count is authoritative for both classes, e.g. G01/G04), a
    per-class metrics field, a flat metrics field matched by class-specific target_label or a
    class-named metrics subdirectory, and finally a directory scan of the synthetic directory
    the generator's own metrics declare. Returns None (never a guess) when nothing verifies.
    """
    gid = entry["id"]
    if entry.get("produced_per_class") is not None and is_two_class(entry):
        n = int(entry["produced_per_class"])
        return {"count": n, "source_precision": "registry_direct", "detail": f"final_generator_registry.json produced_per_class ({gid})"}

    metrics_rel = entry.get("metrics")
    if not metrics_rel:
        return {"count": None, "source_precision": "unresolved", "detail": f"generator {gid} has no metrics/produced_per_class evidence"}

    p = Path(metrics_rel)
    candidate_paths = [root / metrics_rel, root / p.parent / klass / p.name]
    target_label = 1 if klass == "positive" else 0

    for path in candidate_paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        per_class = (payload.get("per_class") or {}).get(klass, {})
        if "n_generated" in per_class:
            return {"count": int(per_class["n_generated"]), "source_precision": "metrics_per_class", "detail": str(path.relative_to(root))}
        metrics = payload.get("metrics") or {}
        if "n_synthetic_filtered" in metrics and (metrics.get("target_label") == target_label or path.parent.name == klass):
            return {"count": int(metrics["n_synthetic_filtered"]), "source_precision": "metrics_flat", "detail": str(path.relative_to(root))}
        synth_dir = (payload.get("config") or {}).get("synthetic_dir") or (payload.get("input_signature") or {}).get("filtered_dir")
        if synth_dir:
            base = _reroot_under_project(synth_dir, root)
            scan_dir = (base.parent / klass) if base.name in TWO_CLASS else base
            if scan_dir.is_dir():
                n = sum(1 for _ in scan_dir.glob("*.png"))
                if n > 0:
                    return {"count": n, "source_precision": "directory_scan", "detail": str(scan_dir)}

    return {"count": None, "source_precision": "unresolved", "detail": f"no registry/metrics evidence found for {gid}/{klass}"}


def two_class_candidates(gen_registry: dict) -> list[dict]:
    return [g for g in gen_registry["generators"] if is_usable(g) and is_two_class(g)]


def positive_only_candidates(gen_registry: dict) -> list[dict]:
    return [g for g in gen_registry["generators"] if is_usable(g) and is_positive_only(g)]


# --- real / augmented train-side counts (train split only: val/test never enter a variant) ---

def real_count_by_class(root: Path) -> dict[str, int]:
    with (root / REAL_METADATA).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    counts = Counter(row["label"] for row in rows)
    return {klass: counts.get(label, 0) for klass, label in CLASS_LABEL.items()}


def augmented_count_by_class(root: Path) -> dict[str, int]:
    directory = root / AUGMENTED_DIR
    if not directory.is_dir():
        return {klass: 0 for klass in TWO_CLASS}
    counts = {klass: 0 for klass in TWO_CLASS}
    label_to_class = {v: k for k, v in CLASS_LABEL.items()}
    for path in directory.glob("*.png"):
        for label, klass in label_to_class.items():
            if f"_label{label}_" in path.name:
                counts[klass] += 1
                break
    return counts


def real_signature(root: Path) -> dict:
    path = root / REAL_METADATA
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {"algorithm": "sha256", "path": REAL_METADATA, "sha256": digest}


# --- deterministic, order-independent, stratified-by-class budgeted sampling ---

def deterministic_sample_signature(candidate_names: list[str], k: int, seed: int = SAMPLING_SEED) -> dict:
    """Pick exactly k names out of candidate_names deterministically and order-independently.

    Sorting first removes any dependency on filesystem enumeration order; a seeded RNG then
    performs the selection so re-running the builder reproduces byte-identical picks.
    """
    ordered = sorted(candidate_names)
    if k > len(ordered):
        raise ValueError(f"requested budget {k} exceeds {len(ordered)} available candidates")
    rng = random.Random(seed)
    picked = sorted(rng.sample(ordered, k))
    digest = hashlib.sha256("\n".join(picked).encode("utf-8")).hexdigest()
    return {"count": k, "seed": seed, "algorithm": "sha256", "sha256": digest,
            "policy": "deterministic_seeded_sample_of_lexicographically_sorted_candidates", "picked": picked}


def budget_signature(gid: str, klass: str, count: int, seed: int = SAMPLING_SEED) -> dict:
    """Signature for a controlled-budget draw when only counts (not filenames) are available.

    Real per-image filenames for every generator are not uniformly available to this module at
    registry-build time (some generators only expose an aggregate count); the signature is over
    the resolved (generator, class, count, seed) tuple so it stays reproducible and comparable
    across variants, and is clearly weaker than a filename-level signature (see field
    `signature_level` on the variant entry).
    """
    payload = f"{gid}|{klass}|{count}|{seed}"
    return {"algorithm": "sha256", "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(), "seed": seed, "policy": "deterministic_stratified_by_class"}


# --- variant construction (Stage 1: base + per-generator screening variants) ---

def _base_variant(variant_id: str, display_name: str, regime: str, root: Path, real: dict, augmented: dict) -> dict:
    total = {k: real.get(k, 0) + augmented.get(k, 0) for k in TWO_CLASS}
    return {
        "dataset_variant_id": variant_id,
        "display_name": display_name,
        "regime": regime,
        "budget_regime": "not_applicable",
        "real_source": REAL_METADATA,
        "augmentation_source": AUGMENTED_DIR if augmented and any(augmented.values()) else None,
        "synthetic_generator_id": None,
        "synthetic_sampling_variant": None,
        "classes": list(TWO_CLASS),
        "train_only": True,
        "real_count_by_class": real,
        "augmented_count_by_class": augmented,
        "synthetic_count_by_class": {},
        "total_count_by_class": total,
        "patient_policy": "train_split_only_no_val_test_leakage",
        "sampling_policy": "not_applicable",
        "seed": SAMPLING_SEED,
        "manifest_path": None,
        "signature": real_signature(root),
        "signature_level": "real_metadata_file",
        "status": "ready",
        "invalid_reason": None,
        "legacy_experiment_ids": [],
    }


def _synthetic_variant(variant_id: str, display_name: str, regime: str, budget_regime: str, root: Path,
                        real: dict, augmented: dict, synthetic: dict, gid: str, synth_classes: tuple[str, ...],
                        source_precisions: dict, seed: int = SAMPLING_SEED, include_real: bool = True,
                        include_augmented: bool = False) -> dict:
    # ``synth_classes`` constrains synthetic labels only. RSP/RAS positive-only variants still
    # contain the complete real (and, when requested, augmented) binary training split.
    used_real = {k: (real.get(k, 0) if include_real else 0) for k in TWO_CLASS}
    used_aug = {k: (augmented.get(k, 0) if include_augmented else 0) for k in TWO_CLASS}
    total = {k: used_real.get(k, 0) + used_aug.get(k, 0) + synthetic.get(k, 0) for k in TWO_CLASS}
    unresolved = [k for k in synth_classes if source_precisions.get(k) == "unresolved"]
    status = "invalid" if unresolved else "ready"
    reason = f"unresolved synthetic count for classes {unresolved} of {gid}" if unresolved else None
    return {
        "dataset_variant_id": variant_id,
        "display_name": display_name,
        "regime": regime,
        "budget_regime": budget_regime,
        "real_source": REAL_METADATA if include_real else None,
        "augmentation_source": AUGMENTED_DIR if include_augmented else None,
        "synthetic_generator_id": gid,
        "synthetic_sampling_variant": "default",
        "classes": list(synth_classes),
        "train_only": True,
        "real_count_by_class": used_real,
        "augmented_count_by_class": used_aug,
        "synthetic_count_by_class": synthetic,
        "total_count_by_class": total,
        "patient_policy": "train_split_only_no_val_test_leakage",
        "sampling_policy": "deterministic_stratified_by_class" if budget_regime == "controlled" else "full_available_documented_counts",
        "seed": seed,
        "manifest_path": None,
        "signature": {k: budget_signature(gid, k, synthetic.get(k, 0), seed) for k in synth_classes},
        "signature_level": "count_and_seed_tuple",
        "status": status,
        "invalid_reason": reason,
        "legacy_experiment_ids": [],
        "source_precision_by_class": source_precisions,
    }


def build_stage1_registry(root: Path) -> dict:
    gen_registry = load_generator_registry(root)
    real = real_count_by_class(root)
    augmented = augmented_count_by_class(root)

    two_class = two_class_candidates(gen_registry)
    positive_only = positive_only_candidates(gen_registry)

    # Resolve every candidate generator's per-class synthetic count once.
    resolved: dict[str, dict[str, dict]] = {}
    for entry in two_class:
        resolved[entry["id"]] = {k: resolve_generator_class_count(root, entry, k) for k in TWO_CLASS}
    for entry in positive_only:
        resolved[entry["id"]] = {"positive": resolve_generator_class_count(root, entry, "positive")}

    # COMMON_SYNTHETIC_PER_CLASS: max count usable identically by every two-class candidate
    # whose *both* classes resolved. Generators that fail to resolve are excluded from the
    # common budget (and flagged individually), never silently treated as zero.
    two_class_resolved_counts = []
    unresolved_two_class = []
    for entry in two_class:
        counts = resolved[entry["id"]]
        if all(counts[k]["count"] is not None for k in TWO_CLASS):
            two_class_resolved_counts.append(min(counts[k]["count"] for k in TWO_CLASS))
        else:
            unresolved_two_class.append(entry["id"])
    common_synthetic_per_class = min(two_class_resolved_counts) if two_class_resolved_counts else None

    # COMMON_SYNTHETIC_POSITIVE: positive-class budget shared by every generator that has a
    # positive class at all (two-class candidates + G05).
    positive_counts = [resolved[e["id"]]["positive"]["count"] for e in two_class if resolved[e["id"]]["positive"]["count"] is not None]
    positive_counts += [resolved[e["id"]]["positive"]["count"] for e in positive_only if resolved[e["id"]]["positive"]["count"] is not None]
    common_synthetic_positive = min(positive_counts) if positive_counts else None

    variants: list[dict] = []
    variants.append(_base_variant("R", "Real only", "base", root, real, {k: 0 for k in TWO_CLASS}))
    variants.append(_base_variant("RA", "Real + traditional augmentation", "base", root, real, augmented))

    for entry in two_class:
        gid = entry["id"]
        counts = resolved[gid]
        precisions = {k: counts[k]["source_precision"] for k in TWO_CLASS}

        if common_synthetic_per_class is not None and gid not in unresolved_two_class:
            budget = {k: common_synthetic_per_class for k in TWO_CLASS}
            variants.append(_synthetic_variant(
                f"RSB_CONTROLLED_{gid}", f"Real + synthetic balanced (controlled budget, {gid})",
                "stage1_screening", "controlled", root, real, augmented, budget, gid, TWO_CLASS, precisions))

        if all(counts[k]["count"] is not None for k in TWO_CLASS):
            full = {k: counts[k]["count"] for k in TWO_CLASS}
            variants.append(_synthetic_variant(
                f"RSB_FULL_{gid}", f"Real + synthetic balanced (full available, {gid})",
                "stage1_screening", "full_available", root, real, augmented, full, gid, TWO_CLASS, precisions))
        else:
            variants.append(_synthetic_variant(
                f"RSB_FULL_{gid}", f"Real + synthetic balanced (full available, {gid})",
                "stage1_screening", "full_available", root, real, augmented, {}, gid, TWO_CLASS, precisions))

        if common_synthetic_positive is not None and counts["positive"]["count"] is not None:
            variants.append(_synthetic_variant(
                f"RSP_CONTROLLED_{gid}", f"Real + synthetic positive-only (controlled budget, {gid})",
                "stage1_screening", "controlled", root, real, augmented,
                {"positive": common_synthetic_positive}, gid, ("positive",), {"positive": precisions["positive"]}))

    for entry in positive_only:
        gid = entry["id"]
        pcount = resolved[gid]["positive"]
        precisions = {"positive": pcount["source_precision"]}
        if common_synthetic_positive is not None and pcount["count"] is not None:
            variants.append(_synthetic_variant(
                f"RSP_CONTROLLED_{gid}", f"Real + synthetic positive-only (controlled budget, {gid})",
                "stage1_screening", "controlled", root, real, augmented,
                {"positive": common_synthetic_positive}, gid, ("positive",), precisions))
        if pcount["count"] is not None:
            variants.append(_synthetic_variant(
                f"RSP_FULL_{gid}", f"Real + synthetic positive-only (full available, {gid})",
                "stage1_screening", "full_available", root, real, augmented,
                {"positive": pcount["count"]}, gid, ("positive",), precisions))

    variants.extend(_legacy_compatible_variants(root, real, augmented))

    budgets = {
        "common_synthetic_per_class": common_synthetic_per_class,
        "common_synthetic_positive": common_synthetic_positive,
        "unresolved_two_class_generators": unresolved_two_class,
    }
    return {
        "schema_version": 2, "pipeline_namespace": PIPELINE_NAMESPACE,
        "generated_from": "configs/final_generator_registry.json + data/processed/metadata/train.csv + data/real_augmented",
        "sampling_seed": SAMPLING_SEED,
        "budgets": budgets,
        "variants": variants,
    }


def _legacy_compatible_variants(root: Path, real: dict, augmented: dict) -> list[dict]:
    """Preserve historically-run classifier dataset compositions under their own IDs.

    These reuse the *training-mode* partial/full-finetuning split already recorded in
    configs/final_classifier_registry.json (a different axis from the new controlled/full
    synthetic-budget regime) rather than re-deriving counts, and are tagged status=legacy so
    downstream selection code never confuses them with the Stage-1 budget-controlled variants.
    """
    try:
        classifier_registry = load_classifier_registry(root)
        gen_registry = load_generator_registry(root)
    except (OSError, json.JSONDecodeError):
        return []
    by_id = {g["id"]: g for g in gen_registry["generators"]}

    synth_to_generator = {"stable_diffusion_finetuned": "02_sd21_filtered_100steps", "ldm_sdvae": "07_ldm_sdvae_extra1361"}
    wanted = {
        ("real_plus_synthetic", "partial_finetuning", "stable_diffusion_finetuned"): ("RS_PARTIAL_G02", "Real + synthetic (legacy partial fine-tuning, G02)"),
        ("real_plus_synthetic", "full_finetuning", "stable_diffusion_finetuned"): ("RS_FULL_G02", "Real + synthetic (legacy full fine-tuning, G02)"),
        ("synthetic_only", "partial_finetuning", "stable_diffusion_finetuned"): ("SYNTHETIC_ONLY_PARTIAL_G02", "Synthetic only (legacy partial fine-tuning, G02)"),
        ("synthetic_only", "full_finetuning", "stable_diffusion_finetuned"): ("SYNTHETIC_ONLY_FULL_G02", "Synthetic only (legacy full fine-tuning, G02)"),
    }
    matched_ids: dict[str, list[str]] = {}
    for exp in classifier_registry.get("experiments", []):
        key = (exp.get("training_dataset_variant"), exp.get("training_mode"), exp.get("synthetic_source"))
        if key in wanted:
            matched_ids.setdefault(wanted[key][0], []).append(exp["experiment_id"])

    variants = []
    for (dv, tm, synth), (variant_id, display_name) in wanted.items():
        legacy_ids = matched_ids.get(variant_id, [])
        if not legacy_ids:
            continue  # spec: only emit "se realmente esistente" (only if it really exists)
        gid = synth_to_generator[synth]
        entry = by_id.get(gid)
        if entry is None:
            continue
        include_real = dv != "synthetic_only"
        counts = {k: resolve_generator_class_count(root, entry, k) for k in TWO_CLASS}
        precisions = {k: counts[k]["source_precision"] for k in TWO_CLASS}
        synthetic = {k: counts[k]["count"] for k in TWO_CLASS if counts[k]["count"] is not None}
        built = _synthetic_variant(
            variant_id, display_name, "legacy_compatible", "not_applicable", root, real, augmented,
            synthetic, gid, TWO_CLASS, precisions,
            include_real=include_real, include_augmented=False)
        built["status"] = "legacy" if len(synthetic) == len(TWO_CLASS) else "invalid"
        built["invalid_reason"] = None if len(synthetic) == len(TWO_CLASS) else f"could not resolve synthetic counts for {gid}"
        built["legacy_experiment_ids"] = legacy_ids
        built["sampling_policy"] = "legacy_training_mode_split_not_budget_regime"
        variants.append(built)
    return variants


# --- Stage 2 (deferred: only callable once SELECTED_GENERATOR_UNION is locked) ---

def build_stage2_variants(root: Path, selected_generator_union: list[str]) -> list[dict]:
    """Build RAS_*/S_ONLY_* variants for exactly the generators in the signed union.

    Must only be called after finalize_validation_stage.py has written and signed
    SELECTED_GENERATOR_UNION — never call this to pre-populate Stage 2 speculatively.
    """
    gen_registry = load_generator_registry(root)
    real = real_count_by_class(root)
    augmented = augmented_count_by_class(root)
    by_id = {g["id"]: g for g in gen_registry["generators"]}

    stage1 = build_stage1_registry(root)
    controlled_count = stage1["budgets"].get("common_synthetic_per_class")
    variants = []
    for gid in selected_generator_union:
        entry = by_id.get(gid)
        if entry is None or gid == "05_ldm_basic_fromscratch":
            continue  # G05 positive-only never gets a synthetic-only balanced variant
        counts = {k: resolve_generator_class_count(root, entry, k) for k in TWO_CLASS}
        precisions = {k: counts[k]["source_precision"] for k in TWO_CLASS}
        if not all(counts[k]["count"] is not None for k in TWO_CLASS):
            continue
        full = {k: counts[k]["count"] for k in TWO_CLASS}

        if controlled_count is not None:
            controlled = {k: controlled_count for k in TWO_CLASS}
            variants.append(_synthetic_variant(
                f"RAS_CONTROLLED_{gid}", f"Real + augmented + synthetic balanced (controlled, {gid})",
                "stage2_advanced", "controlled", root, real, augmented, controlled, gid, TWO_CLASS,
                precisions, include_real=True, include_augmented=True))
            variants.append(_synthetic_variant(
                f"S_ONLY_CONTROLLED_{gid}", f"Synthetic only balanced (controlled, {gid})",
                "stage2_advanced", "controlled", root, real, augmented, controlled, gid, TWO_CLASS,
                precisions, include_real=False, include_augmented=False))

        variants.append(_synthetic_variant(f"RAS_FULL_{gid}", f"Real + augmented + synthetic balanced (full, {gid})",
                                            "stage2_advanced", "full_available", root, real, augmented, full, gid, TWO_CLASS, precisions,
                                            include_real=True, include_augmented=True))
        variants.append(_synthetic_variant(f"S_ONLY_FULL_{gid}", f"Synthetic only balanced (full, {gid})",
                                            "stage2_advanced", "full_available", root, real, augmented, full, gid, TWO_CLASS, precisions,
                                            include_real=False, include_augmented=False))
        if entry.get("classes") and "positive" in entry["classes"]:
            pos_count = counts["positive"]["count"]
            variants.append(_synthetic_variant(f"RAS_POSITIVE_{gid}", f"Real + augmented + synthetic positive-only (full, {gid})",
                                                "stage2_advanced", "full_available", root, real, augmented,
                                                {"positive": pos_count}, gid, ("positive",), {"positive": precisions["positive"]},
                                                include_real=True, include_augmented=True))
    return variants


# --- validation ---

def validate_registry(registry: dict) -> list[str]:
    errors = []
    seen_ids = set()
    controlled_counts: dict[str, set[int]] = {}

    for variant in registry.get("variants", []):
        vid = variant["dataset_variant_id"]
        if vid in seen_ids:
            errors.append(f"duplicate dataset_variant_id: {vid}")
        seen_ids.add(vid)

        if not variant.get("train_only", False):
            errors.append(f"{vid}: train_only must be true")

        if "05_ldm_basic_fromscratch" == variant.get("synthetic_generator_id") and "negative" in variant.get("classes", []):
            errors.append(f"{vid}: G05 is positive-only and must never carry a negative class")

        if variant.get("budget_regime") == "controlled" and variant.get("status") == "ready":
            counts = set(variant.get("synthetic_count_by_class", {}).values())
            if len(counts) > 1:
                errors.append(f"{vid}: controlled-budget variant has non-uniform per-class counts {counts}")
            # Group by (regime, class shape): RSB_CONTROLLED_* (both classes) and
            # RSP_CONTROLLED_* (positive-only) are intentionally different budgets
            # (spec 3.3: "per positive-only usa un numero comune positivo separato").
            group_key = (variant["regime"], tuple(sorted(variant.get("classes", []))))
            controlled_counts.setdefault(group_key, set()).update(counts)

        if variant.get("status") == "ready" and variant.get("budget_regime") != "not_applicable" and not variant.get("signature"):
            errors.append(f"{vid}: ready synthetic variant is missing a signature")

    for (regime_key, class_shape), counts in controlled_counts.items():
        if len(counts) > 1:
            errors.append(f"regime {regime_key} classes={list(class_shape)}: controlled-budget variants disagree on class count across generators: {counts}")

    return errors


def build_and_write(root: Path) -> dict:
    registry = build_stage1_registry(root)
    # Counts from historical metric files are not enough to call a dataset executable.  Resolve
    # the actual candidates now and persist explicit blockers; this prevents a later registry
    # rebuild from silently turning G05/G04-full back into schedulable jobs.
    from classifier_dataset_builder import build_file_list  # noqa: PLC0415
    generator_registry = load_generator_registry(root)
    for variant in registry["variants"]:
        if variant.get("status") != "ready":
            continue
        try:
            build_file_list(root, variant, generator_registry)
        except (OSError, ValueError) as exc:
            variant["status"] = "blocked"
            variant["invalid_reason"] = f"runtime dataset resolution failed: {exc}"
    errors = validate_registry(registry)
    registry["validation_errors"] = errors
    out_path = root / "configs/dataset_variant_registry.json"
    atomic_json(out_path, registry)
    return registry


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    registry = build_and_write(root)
    errors = registry["validation_errors"]
    print(f"variants written: {len(registry['variants'])}")
    print(f"COMMON_SYNTHETIC_PER_CLASS: {registry['budgets']['common_synthetic_per_class']}")
    print(f"COMMON_SYNTHETIC_POSITIVE: {registry['budgets']['common_synthetic_positive']}")
    if errors:
        print(f"VALIDATION ERRORS ({len(errors)}):")
        for err in errors:
            print(f" - {err}")
        raise SystemExit(1)
    print("registry valid.")


if __name__ == "__main__":
    main()
