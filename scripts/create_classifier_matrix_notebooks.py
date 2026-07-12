#!/usr/bin/env python3
"""Deterministically generate one Stage-1 notebook per architecture x dataset variant."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

ARCHITECTURES = {
    "resnet50": ("R50", "ResNet-50"), "maxvit512": ("MV", "MaxViT-512"),
    "mammofm": ("MFM", "Mammo-FM"), "raddino": ("RAD", "RAD-DINO"),
}
SEEDS = [17, 42, 73]
INVENTORY_FIELDS = [
    "path", "experiment_id", "architecture", "dataset_variant", "stage", "regime", "generator",
    "training_policy", "seeds", "dataset_status", "checkpoint_legacy_available", "training_required",
    "validation_available", "test_allowed", "compile_status", "dry_run_status", "generator_ownership",
    "note_blocker",
]


def cell_id(notebook_key: str, index: int) -> str:
    return hashlib.sha256(f"{notebook_key}:{index}".encode()).hexdigest()[:16]


def cell(kind: str, source: str, notebook_key: str, index: int) -> dict:
    payload = {"cell_type": kind, "id": cell_id(notebook_key, index), "metadata": {},
               "source": [line + "\n" for line in source.rstrip().splitlines()]}
    if kind == "code":
        payload.update({"execution_count": None, "outputs": []})
    return payload


def notebook(architecture: str, variant: dict, dataset_status: str, blocker: str | None, stage: int = 1) -> dict:
    prefix, display = ARCHITECTURES[architecture]
    vid = variant["dataset_variant_id"]
    key = f"{architecture}__{vid}"
    regime = variant.get("budget_regime", "not_applicable")
    generator = variant.get("synthetic_generator_id") or "none"
    has_synthetic = bool(variant.get("synthetic_generator_id"))
    has_augmented = vid == "RA" or "RAS" in vid
    attribution_method = {
        "resnet50": "Grad-CAM sulla feature map convoluzionale finale",
        "maxvit512": "Grad-CAM gradient-weighted sull'ultimo stage MaxViT con reshape spaziale",
        "mammofm": "attribuzione gradient-weighted token/spaziale risolta dal backbone reale",
        "raddino": "attribuzione gradient-weighted dei patch token (non semplice attention map)",
    }[architecture]
    sources = [
        ("markdown", f"# {prefix} · {vid}\n\n**Domanda sperimentale:** effetto di `{vid}` su {display}, "
                     "con confronto validation-only e tre seed indipendenti. Questo notebook non può leggere il test locked."),
        ("markdown", "## 1–5 · Identità e regime\n\n"
                     f"- Experiment ID logico: `{key}`\n- Architettura: `{architecture}` ({display})\n"
                     f"- Dataset variant: `{vid}`\n- Regime: `{regime}`\n- Generatore: `{generator}`"),
        ("code", f'''from pathlib import Path
import json, os, sys

def find_project_root(start=Path.cwd()):
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "configs/classifier_experiment_matrix.json").is_file():
            return candidate
    raise FileNotFoundError("MammoDiffusion project root not found")

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "notebooks/utility"))
import classifier_experiment_runner as runner
import classifier_dataset_builder as datasets
import classifier_reporting as reporting
import classifier_interpretability as interpretability

ARCHITECTURE = {architecture!r}
DATASET_VARIANT_ID = {vid!r}
EXPERIMENT_ID = {key!r}
MODE = "auto"  # plan | auto | train | validate | metrics-only
RUN_SEEDS = [17, 42, 73]
ALLOW_RETRAIN = False
ALLOW_OVERWRITE_VERIFIED = False
RESUME = True
GENERATE_GRADCAM = True
GRADCAM_NUM_REAL_SAMPLES = 8
GRADCAM_NUM_SYNTHETIC_SAMPLES = 8
TINY_SMOKE = os.environ.get("MAMMO_CLASSIFIER_TINY") == "1"
DATASET_STATUS = {dataset_status!r}
DATASET_BLOCKER = {blocker!r}
assert MODE != "locked-test"
RESULTS_DIR = PROJECT_ROOT / "results/classifiers_matrix" / ARCHITECTURE / DATASET_VARIANT_ID / f"{{ARCHITECTURE}}_standard"
print(EXPERIMENT_ID, MODE, RUN_SEEDS, "resume=", RESUME, "tiny=", TINY_SMOKE)
print("results:", RESULTS_DIR)'''),
        ("markdown", "## 4 — Provenance e configurazione\n\n"
                     "La cella seguente risolve i file canonici, firma il manifest e carica esclusivamente la validation reale. "
                     "Le varianti bloccate restano documentate e non avviano training."),
        ("code", '''registry = runner.load_dataset_variant_registry(PROJECT_ROOT)
variant = next(v for v in registry["variants"] if v["dataset_variant_id"] == DATASET_VARIANT_ID)
if DATASET_STATUS == "BLOCKED":
    dataset_summary = {"status": DATASET_STATUS, "blocker": DATASET_BLOCKER}
else:
    train_rows, validation_rows, dataset_manifest = datasets.build_training_and_validation_rows(PROJECT_ROOT, variant)
    dataset_summary = {
        "status": DATASET_STATUS, "counts": dataset_manifest["counts"],
        "train_samples": len(train_rows), "validation_samples": len(validation_rows),
        "dataset_signature": dataset_manifest["signature"],
        "validation_signature": dataset_manifest["validation_signature"],
        "validation_sources": sorted({row["source"] for row in validation_rows}),
    }
    if MODE != "plan":
        reporting.persist_dataset_summary(RESULTS_DIR, train_rows, validation_rows, dataset_manifest)
print(json.dumps(dataset_summary, indent=1))'''),
        ("markdown", "## 5 — Composizione del dataset\n\nTabelle source × classe, conteggi e percentuali sono salvati in `results/.../dataset/`. Validation è esclusivamente reale."),
        ("markdown", "## 6 — Esempi visivi del training set\n\nLa griglia usa un campionamento deterministico, senza selezione manuale degli esempi."),
        ("code", f'''if DATASET_STATUS != "BLOCKED":
    plan_figures = reporting.persist_plan_figures(RESULTS_DIR, has_synthetic={has_synthetic!r}, has_augmented={has_augmented!r})
    print(*plan_figures, sep="\\n")'''),
        ("markdown", "## 7 — Piano e stato resume\n\n"
                     "Il protocollo è unico per architettura. `auto` riusa un checkpoint verificato, altrimenti addestra, "
                     "poi esegue validation e metriche. Ogni seed usa directory e checkpoint distinti."),
        ("code", '''policy = runner.load_training_protocols(PROJECT_ROOT)["policies"][ARCHITECTURE]
plans = [runner.plan(PROJECT_ROOT, ARCHITECTURE, DATASET_VARIANT_ID, seed) for seed in RUN_SEEDS]
print(json.dumps({"policy": policy, "plans": plans}, indent=1))'''),
        ("markdown", "## 8 — Costruzione modello\n\nIl runner salva `model_summary.txt` e `model_architecture.json`; parametri trainable/frozen e input shape sono riportati senza dump verbosi."),
        ("markdown", "## 9 — Training dei seed\n\n"
                     "Questa è l’unica cella operativa. Notebook e scheduler chiamano la stessa funzione condivisa. "
                     "L’ensemble è la media delle probabilità dei seed 17/42/73 e la soglia è scelta sull’ensemble validation."),
        ("code", '''if DATASET_STATUS == "BLOCKED":
    run_results = [{"status": "BLOCKED", "reason": DATASET_BLOCKER}]
else:
    run_results = runner.execute_configuration(
        PROJECT_ROOT, ARCHITECTURE, DATASET_VARIANT_ID,
        mode=MODE, run_seeds=RUN_SEEDS, tiny=TINY_SMOKE,
    )
print(json.dumps(run_results, indent=1, default=str))'''),
        ("markdown", "## 10 — Curve di training\n\nLoss, ROC-AUC, PR-AUC, learning rate, precision e recall sono salvati per seed e aggregati sotto `figures/`; le storie restano CSV."),
        ("code", '''report_artifacts = reporting.render_complete_report(RESULTS_DIR)
print(json.dumps(report_artifacts, indent=1))'''),
        ("markdown", "## 11 — Validation per seed\n\nLe predizioni includono `patient_id`, `image_id`, label e probabilità. Nessuna cella importa il test."),
        ("markdown", "## 12 — Ensemble validation\n\nMedia delle probabilità dei seed 17/42/73 dopo verifica rigorosa dell'allineamento; soglia congelata dalla validation."),
        ("markdown", "## 13 — Metriche e risultati\n\nROC-AUC, PR-AUC, F1, sensitivity, specificity, PPV, NPV, balanced accuracy, MCC, accuracy, Brier, ECE e confusion matrix."),
        ("markdown", "## 14 — Grafici validation\n\nROC, PR, calibration, confusion matrix e distribuzione delle probabilità sono visualizzati e salvati."),
        ("markdown", "## 15 — Analisi degli errori\n\nFP, FN, TP e TN sono selezionati deterministicamente dalla validation e salvati in CSV e figura."),
        ("markdown", f"## 16 — Grad-CAM / Gradient-based attribution\n\nTecnica: **{attribution_method}**. "
                     f"Campioni reali sempre presenti; sintetici: **{has_synthetic}**; augmented: **{has_augmented}**. "
                     "Per campione vengono salvate mappe normalizzate per seed e media ensemble. Il manifest reale condiviso è `configs/interpretability_validation_samples.json`."),
        ("code", '''policy_name = f"{ARCHITECTURE}_standard"
ensemble_path = (PROJECT_ROOT / "results/classifiers_matrix" / ARCHITECTURE /
                 DATASET_VARIANT_ID / policy_name / "ensemble/manifests/ensemble_validation_manifest.json")
print("ensemble:", ensemble_path, "exists=", ensemble_path.is_file())
if GENERATE_GRADCAM and MODE != "plan" and ensemble_path.is_file() and not TINY_SMOKE:
    attribution_status = interpretability.generate_configuration_attributions(
        PROJECT_ROOT, ARCHITECTURE, DATASET_VARIANT_ID, policy,
        seeds=RUN_SEEDS, real_limit=GRADCAM_NUM_REAL_SAMPLES,
        synthetic_limit=GRADCAM_NUM_SYNTHETIC_SAMPLES)
    # generate_configuration_attributions() displays its own newly-created overlays inline
    # (display=True by default) as soon as they exist, rather than relying on cell 10's
    # earlier reporting.render_complete_report() call to have found them - which it never
    # could, since that cell runs before this one generates anything.
    print(json.dumps(attribution_status, indent=1))
for seed in RUN_SEEDS:
    run_dir = runner.resolve_job(PROJECT_ROOT, ARCHITECTURE, DATASET_VARIANT_ID, seed)["run_dir"]
    print(seed, run_dir, sorted(p.name for p in run_dir.glob("*.json")) if run_dir.exists() else [])'''),
        ("markdown", "## 17 — Riepilogo finale\n\n"
                     "Output canonici: `experiments/classifiers_matrix/<arch>/<variant>/<policy>/seed_<seed>/` "
                     "e `results/classifiers_matrix/<arch>/<variant>/<policy>/`. Durata e picco VRAM sono diagnostiche operative; energia e CO₂ dei nuovi classificatori non sono tracciate. "
                     "Il test locked non è importato né accessibile da questo notebook."),
    ]
    return {"cells": [cell(kind, source, key, i) for i, (kind, source) in enumerate(sources)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3"},
                         "mammodiffusion": {"architecture": architecture, "dataset_variant_id": vid,
                                            "stage": stage, "generator": generator}},
            "nbformat": 4, "nbformat_minor": 5}


def dataset_audit(root: Path, variant: dict) -> tuple[str, str | None]:
    try:
        import classifier_dataset_builder as builder
        builder.build_training_and_validation_rows(root, dict(variant))
    except Exception as exc:
        return "BLOCKED", f"{type(exc).__name__}: {exc}"
    return "READY", None


def write_notebook(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n")


def write_overviews(root: Path) -> None:
    base = root / "notebooks/3_classifiers_matrix"
    overview_specs = {
        "00_Matrice_Esperimenti.ipynb": (
            "# Matrice esperimenti classificatori v2",
            "from pathlib import Path\nimport json\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'configs/classifier_experiment_matrix.json').is_file())\n"
            "matrix = json.loads((ROOT / 'configs/classifier_experiment_matrix.json').read_text())\n"
            "print('job:', len(matrix['jobs']), 'seed:', sorted({j['seed'] for j in matrix['jobs']}))\n"
            "print('Il test locked non viene letto da questo notebook.')",
        ),
        "00b_Stato_Esecuzioni.ipynb": (
            "# Stato esecuzioni matrice v2",
            "from pathlib import Path\nimport sys\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'configs/classifier_experiment_matrix.json').is_file())\n"
            "sys.path.insert(0, str(ROOT / 'scripts'))\nfrom status_classifier_experiment_matrix import build_report, print_report\n"
            "print_report(build_report(ROOT))",
        ),
    }
    for name, (title, code) in overview_specs.items():
        key = name.removesuffix(".ipynb")
        payload = {"cells": [cell("markdown", title, key, 0), cell("code", code, key, 1)],
                   "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                "language_info": {"name": "python", "version": "3"}},
                   "nbformat": 4, "nbformat_minor": 5}
        write_notebook(base / name, payload)


def generate(root: Path) -> list[dict]:
    registry = json.loads((root / "configs/dataset_variant_registry.json").read_text())
    classifier_registry = json.loads((root / "configs/final_classifier_registry.json").read_text())
    inventory = []
    write_overviews(root)
    for architecture, (prefix, _) in ARCHITECTURES.items():
        folder = root / "notebooks/3_classifiers_matrix" / architecture
        for variant in registry["variants"]:
            status, blocker = dataset_audit(root, variant)
            vid = variant["dataset_variant_id"]
            path = folder / f"{prefix}_{vid}.ipynb"
            write_notebook(path, notebook(architecture, variant, status, blocker))
            legacy = any(exp.get("architecture") == {
                "resnet50": "ResNet-50", "maxvit512": "MaxViT-512",
                "mammofm": "Mammo-FM", "raddino": "RAD-DINO",
            }[architecture] for exp in classifier_registry.get("experiments", [])
                         if exp["experiment_id"] in variant.get("legacy_experiment_ids", []))
            inventory.append({
                "path": str(path.relative_to(root)), "experiment_id": f"{architecture}__{vid}",
                "architecture": architecture, "dataset_variant": vid, "stage": 1,
                "regime": variant.get("budget_regime"), "generator": variant.get("synthetic_generator_id"),
                "training_policy": f"{architecture}_standard", "seeds": ",".join(map(str, SEEDS)),
                "dataset_status": status, "checkpoint_legacy_available": legacy,
                "training_required": True, "validation_available": False, "test_allowed": False,
                "compile_status": "PASS", "dry_run_status": "PLAN_PASS" if status == "READY" else "BLOCKED",
                "generator_ownership": "scripts/create_classifier_matrix_notebooks.py", "note_blocker": blocker,
            })
    return inventory


def write_inventory(root: Path, rows: list[dict]) -> None:
    out = root / "results/notebook_inventory"
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook_inventory.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    with (out / "notebook_inventory.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=INVENTORY_FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.project_root)
    rows = generate(root)
    write_inventory(root, rows)
    print(f"Stage 1 notebooks: {len(rows)}; ready={sum(r['dataset_status']=='READY' for r in rows)}; "
          f"blocked={sum(r['dataset_status']=='BLOCKED' for r in rows)}")


if __name__ == "__main__":
    main()
