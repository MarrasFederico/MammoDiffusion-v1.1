"""Computational core for the G01-G08 generator comparison (spec section 12).

Pandas-free (stdlib csv/json + numpy) so it is importable and unit-testable in the lightweight
`base` env. notebooks/4_comparisons_and_test/00z_Confronto_Diffusori.ipynb calls into this
module rather than recomputing aggregation logic inline, keeping the notebook thin.
"""
from __future__ import annotations

import json
from pathlib import Path

TWO_CLASS_METRICS = ("FID", "IS_mean", "precision", "recall", "density", "coverage")


def load_generator_registry(root: Path) -> dict:
    return json.loads((root / "configs/final_generator_registry.json").read_text())


def _load_metrics_file(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def generator_metrics_by_class(root: Path, generator_entry: dict) -> dict:
    """Normalize the (at least 3 different) on-disk metrics schemas into one shape:
    {class: {FID, IS_mean, IS_std, precision, recall, density, coverage, n_generated} | None}.
    A missing/unreadable file is reported as None per class, never invented as zero (spec
    0.4/12: "Il registry non dichiara mai un file o una metrica mancante" implies the reverse
    too — never claim a value that was not actually read).
    """
    metrics_rel = generator_entry.get("metrics")
    result = {"negative": None, "positive": None}
    if not metrics_rel:
        return result

    root_path = root / metrics_rel
    payload = _load_metrics_file(root_path)
    if payload and "per_class" in payload:
        for klass in ("negative", "positive"):
            entry = payload["per_class"].get(klass)
            if entry:
                result[klass] = {k: entry.get(k) for k in ("FID", "IS_mean", "IS_std", "precision", "recall", "density", "coverage", "n_generated")}
        return result

    # schema B: flat metrics dict, either single-class (target_label) or split into class subdirs
    if payload and "metrics" in payload:
        m = payload["metrics"]
        klass = "positive" if m.get("target_label") == 1 else "negative" if m.get("target_label") == 0 else None
        if klass:
            result[klass] = {"FID": m.get("FID"), "IS_mean": m.get("IS_mean"), "IS_std": m.get("IS_std"),
                              "precision": m.get("precision"), "recall": m.get("recall"), "density": m.get("density"),
                              "coverage": m.get("coverage"), "n_generated": m.get("n_synthetic_filtered")}
    for klass in ("negative", "positive"):
        if result[klass] is None:
            sibling = root_path.parent / klass / root_path.name
            sibling_payload = _load_metrics_file(sibling)
            if sibling_payload and "metrics" in sibling_payload:
                m = sibling_payload["metrics"]
                result[klass] = {"FID": m.get("FID"), "IS_mean": m.get("IS_mean"), "IS_std": m.get("IS_std"),
                                  "precision": m.get("precision"), "recall": m.get("recall"), "density": m.get("density"),
                                  "coverage": m.get("coverage"), "n_generated": m.get("n_synthetic_filtered")}
    return result


def positive_class_comparison(root: Path, registry: dict | None = None) -> list[dict]:
    """Every generator that has positive-class data (spec 12.1): all 8 generators."""
    registry = registry or load_generator_registry(root)
    rows = []
    for gen in registry["generators"]:
        if "positive" not in gen.get("classes", []):
            continue
        metrics = generator_metrics_by_class(root, gen)
        rows.append({"generator_id": gen["id"], "family": gen.get("family"), "role": gen.get("role"), **(metrics["positive"] or {})})
    return rows


def two_class_comparison(root: Path, registry: dict | None = None) -> list[dict]:
    """Only generators with both classes (spec 12.2): never average a positive-only generator
    in with two-class ones.
    """
    registry = registry or load_generator_registry(root)
    rows = []
    for gen in registry["generators"]:
        if not {"negative", "positive"}.issubset(set(gen.get("classes", []))):
            continue
        metrics = generator_metrics_by_class(root, gen)
        if metrics["negative"] is None or metrics["positive"] is None:
            rows.append({"generator_id": gen["id"], "family": gen.get("family"), "status": "incomplete_metrics"})
            continue
        averaged = {}
        for key in TWO_CLASS_METRICS:
            va, vb = metrics["negative"].get(key), metrics["positive"].get(key)
            averaged[key] = (va + vb) / 2 if va is not None and vb is not None else None
        rows.append({"generator_id": gen["id"], "family": gen.get("family"), "status": "ok", **averaged})
    return rows


ABLATION_PAIRS = (
    ("01_sd21_baseline_50steps", "02_sd21_filtered_100steps", "sampling_steps_50_vs_100"),
    ("02_sd21_filtered_100steps", "03_sd21_vae_finetuned", "vae_base_vs_finetuned"),
    ("03_sd21_vae_finetuned", "04_sd21_lora", "full_finetune_vs_lora"),
    ("05_ldm_basic_fromscratch", "06_ldm_extra1361_fromscratch", "extra1361_effect"),
    ("07_ldm_sdvae_extra1361", "08_ldm_v3_sdvae_fromscratch", "ldm_v2_vs_v3"),
)


def ablation_comparison(root: Path, registry: dict | None = None) -> list[dict]:
    registry = registry or load_generator_registry(root)
    by_id = {g["id"]: g for g in registry["generators"]}
    rows = []
    for gid_a, gid_b, label in ABLATION_PAIRS:
        if gid_a not in by_id or gid_b not in by_id:
            continue
        ma = generator_metrics_by_class(root, by_id[gid_a])
        mb = generator_metrics_by_class(root, by_id[gid_b])
        row = {"ablation": label, "generator_a": gid_a, "generator_b": gid_b}
        for klass in ("positive", "negative"):
            if ma[klass] and mb[klass]:
                for key in ("FID", "IS_mean", "precision", "recall"):
                    va, vb = ma[klass].get(key), mb[klass].get(key)
                    if va is not None and vb is not None:
                        row[f"{klass}_delta_{key}_b_minus_a"] = vb - va
        rows.append(row)
    return rows


def sampling_ablation(root: Path, registry: dict | None = None, generator_id: str = "08_ldm_v3_sdvae_fromscratch") -> list[dict]:
    """G08's 25/50/75/100-step sweep (spec 3.1/12.4): reads sibling
    results/diffusers/<gid>_sampling_st<N>_<hash>/metrics/*/final_filtered_vs_test.json dirs.
    """
    registry = registry or load_generator_registry(root)
    entry = next((g for g in registry["generators"] if g["id"] == generator_id), None)
    if entry is None or "sampling_ablations" not in entry:
        return []
    results_dir = root / "results/diffusers"
    rows = []
    for steps in entry["sampling_ablations"]:
        matches = sorted(results_dir.glob(f"{generator_id}_sampling_st{steps}_*"))
        if not matches:
            rows.append({"sampling_steps": steps, "status": "not_found"})
            continue
        run_dir = matches[0]
        row = {"sampling_steps": steps, "status": "ok", "run_dir": str(run_dir.relative_to(root))}
        for klass in ("positive", "negative"):
            payload = _load_metrics_file(run_dir / "metrics" / klass / "final_filtered_vs_test.json") or \
                _load_metrics_file(run_dir / "metrics" / "final_filtered_vs_test.json")
            if payload and "metrics" in payload:
                row[f"{klass}_FID"] = payload["metrics"].get("FID")
        rows.append(row)
    return rows


def final_comparison_table(root: Path, registry: dict | None = None) -> list[dict]:
    """Spec 12.3: only generators whose role is a "final_comparison" variant."""
    registry = registry or load_generator_registry(root)
    return [row for row in two_class_comparison(root, registry)
            if next(g for g in registry["generators"] if g["id"] == row["generator_id"]).get("role", "").startswith("final_comparison")]


def write_outputs(root: Path, registry: dict | None = None) -> dict:
    import csv
    registry = registry or load_generator_registry(root)
    out_dir = root / "results/generator_comparison"
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    tables = {
        "positive_class_comparison": positive_class_comparison(root, registry),
        "two_class_comparison": two_class_comparison(root, registry),
        "ablation_comparison": ablation_comparison(root, registry),
        "sampling_ablation": sampling_ablation(root, registry),
        "final_comparison": final_comparison_table(root, registry),
    }
    for name, rows in tables.items():
        json_path = out_dir / "tables" / f"{name}.json"
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")
        if rows:
            fieldnames = sorted({k for r in rows for k in r})
            csv_path = out_dir / "tables" / f"{name}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

    write_summary_md(root, tables)
    return tables


def write_summary_md(root: Path, tables: dict) -> Path:
    n_positive = len(tables["positive_class_comparison"])
    complete_two_class = [r for r in tables["two_class_comparison"] if r["status"] == "ok"]
    incomplete_two_class = [r["generator_id"] for r in tables["two_class_comparison"] if r["status"] != "ok"]
    lines = [
        "# Confronto generatori — sintesi",
        "",
        f"Generatori con dati classe positiva: {n_positive}/8.",
        f"Generatori con confronto due-classi completo: {len(complete_two_class)}.",
        f"Generatori con metriche incomplete (esclusi dalla media, mai stimati): {incomplete_two_class or 'nessuno'}.",
        "",
        "Nessun vincitore downstream e' dichiarato qui: la selezione dei generatori per Stage 2 "
        "avviene esclusivamente su validation dei classificatori "
        "(scripts/finalize_validation_stage.py --stage 1), mai da FID/IS/PRDC da soli e mai dal test set.",
        "",
        "Tabelle: results/generator_comparison/tables/*.{csv,json}.",
    ]
    out_path = root / "results/generator_comparison/generator_comparison_summary.md"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
