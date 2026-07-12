#!/usr/bin/env python3
"""Generate the new v2 analysis/orchestration notebooks (spec sections 10.8, 13, 16, 18).

    python scripts/generate_v2_analysis_notebooks.py

Reuses write_notebook() from notebooks/utility/create_final_classifier_notebooks.py (the
project's existing deterministic-notebook-generator convention) so cell construction and stable
IDs stay consistent with 04x/04y/04z. Every notebook here is thin: it imports and calls the
tested notebooks/utility modules built this session rather than reimplementing logic inline.

Nothing here executes the notebooks or reads a locked test set — this only writes .ipynb files.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

from create_final_classifier_notebooks import write_notebook  # noqa: E402

BOOTSTRAP = '''from pathlib import Path
import json, sys

def find_project_root(start=Path.cwd()):
    for candidate in [start, *start.parents]:
        if (candidate / "configs/final_generator_registry.json").is_file():
            return candidate
    raise FileNotFoundError("Project root not found")

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "notebooks/utility"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
'''

# ---------------------------------------------------------------------------------------------
# 00y_Analisi_Consumi_e_Sostenibilita.ipynb (spec section 13)
# ---------------------------------------------------------------------------------------------

SUSTAINABILITY_CELLS = [
    ("markdown", "## 1. Registro canonico\n\nQuesto notebook legge esclusivamente "
     "`results/sustainability/canonical_events.jsonl` (import legacy idempotente) tramite `sustainability_registry.py`. Non somma mai "
     "JSON/JSONL grezzi a mano, non mescola resume duplicati, e riporta sempre "
     "`actual_project_energy` accanto a `canonical_pipeline_energy` (mai uno al posto dell'altro)."),
    ("code", BOOTSTRAP + '''
import sustainability_registry as sr
import matplotlib.pyplot as plt
import numpy as np

EVENTS_PATH = PROJECT_ROOT / "results/sustainability/canonical_events.jsonl"
events = sr.load_events(EVENTS_PATH)
print(f"eventi caricati: {len(events)} da {EVENTS_PATH}")
if not events:
    print("Nessun evento canonico trovato. Eseguire scripts/import_legacy_sustainability_logs.py "
          "oppure collegare eco_tracker a sustainability_registry.append_event.")
'''),
    ("markdown", "## 2. Consumi assoluti (energia/CO2, scala log) — spec 13.1.1"),
    ("code", '''
by_phase = sr.group_by_phase(events)
if by_phase:
    phases = list(by_phase.keys())
    energies = [by_phase[p]["energy_kwh"] for p in phases]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(phases, energies)
    ax.set_yscale("log")
    ax.set_ylabel("kWh (scala log)")
    ax.set_title("Energia canonica per fase")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    OUT = PROJECT_ROOT / "results/sustainability/figures"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "energy_by_phase_log.png", dpi=200)
    for p, e in zip(phases, energies):
        print(f"{p}: {e:.6f} kWh")
'''),
    ("markdown", "## 3. Trade-off qualita'-energia (spec 13.2 trade-off) — per famiglia classificatore"),
    ("code", '''
# Richiede results/classifiers_matrix/*/*/validation_metrics.json popolati dalla matrice reale;
# finche' nessun job e' completo questa cella riporta un DataFrame vuoto in modo esplicito,
# mai valori inventati.
import csv
matrix_path = PROJECT_ROOT / "configs/classifier_experiment_matrix.json"
tradeoff_rows = []
if matrix_path.is_file():
    matrix = json.loads(matrix_path.read_text())
    for job in matrix["jobs"]:
        vmetrics = PROJECT_ROOT / job["validation_predictions_path"]
        vmetrics = vmetrics.parent / "validation_metrics.json"
        if vmetrics.is_file():
            metrics = json.loads(vmetrics.read_text())
            run_events = [e for e in events if e.get("experiment_id") == job["experiment_id"]]
            energy = sum(e.get("energy_kwh") or 0.0 for e in run_events if e.get("canonical"))
            tradeoff_rows.append({"experiment_id": job["experiment_id"], "pr_auc": metrics.get("pr_auc"), "energy_kwh": energy})
print(f"punti trade-off disponibili: {len(tradeoff_rows)}")
if tradeoff_rows:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter([r["energy_kwh"] for r in tradeoff_rows], [r["pr_auc"] for r in tradeoff_rows])
    ax.set_xlabel("kWh (canonico)"); ax.set_ylabel("validation PR-AUC")
    ax.set_title("PR-AUC vs kWh (Pareto qualita'-energia)")
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "results/sustainability/figures/pr_auc_vs_kwh.png", dpi=200)
'''),
    ("markdown", "## 4. Decomposizione per fase (stacked bar) — spec 13.2"),
    ("code", '''
if by_phase:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["progetto"], [sum(v["energy_kwh"] for v in by_phase.values())], label="totale")
    bottom = 0.0
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    for phase, vals in by_phase.items():
        ax2.bar(["pipeline"], [vals["energy_kwh"]], bottom=bottom, label=phase)
        bottom += vals["energy_kwh"]
    ax2.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    ax2.set_ylabel("kWh")
    fig2.tight_layout()
    fig2.savefig(PROJECT_ROOT / "results/sustainability/figures/phase_decomposition_stacked.png", dpi=200)
    plt.close(fig)
'''),
    ("markdown", "## 5. Actual vs canonical (spec 13.2 actual vs canonical)"),
    ("code", '''
totals = sr.actual_vs_canonical(events)
print(json.dumps(totals, indent=1))
'''),
    ("markdown", "## 6. Tabelle canoniche (spec 13.3)"),
    ("code", '''
sr.write_summary_by_run(PROJECT_ROOT, events)
sr.write_summary_by_experiment(PROJECT_ROOT, events)
summary_path = PROJECT_ROOT / "results/sustainability/sustainability_summary.md"
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(
    "# Sintesi sostenibilita'\\n\\n"
    f"Eventi canonici: {totals['n_events_canonical']} / eventi attuali: {totals['n_events_actual']}.\\n\\n"
    f"Energia canonica: {totals['canonical_pipeline_energy_kwh']:.6f} kWh. "
    f"Energia attuale (inclusi retry/fallimenti): {totals['actual_project_energy_kwh']:.6f} kWh.\\n\\n"
    "CodeCarbon fornisce stime, non misure dirette alla presa elettrica.\\n"
)
print(f"scritto: {summary_path.relative_to(PROJECT_ROOT)}")
'''),
]

# ---------------------------------------------------------------------------------------------
# 05_Experiment_Matrix_Status.ipynb (spec section 18: status/diagnostics only, no training)
# ---------------------------------------------------------------------------------------------

STATUS_CELLS = [
    ("markdown", "## Stato ed esperimento — sola lettura\n\nQuesto notebook non avvia mai training "
     "o scheduler; chiama solo `status_classifier_experiment_matrix.build_report` (script read-only)."),
    ("code", BOOTSTRAP + '''
import status_classifier_experiment_matrix as status_mod

STAGE = None  # imposta 1 o 2 per filtrare
report = status_mod.build_report(PROJECT_ROOT, stage=STAGE)
status_mod.print_report(report)
'''),
    ("markdown", "### Dettaglio per architettura"),
    ("code", '''
for architecture in ("resnet50", "maxvit512", "mammofm", "raddino"):
    arch_report = status_mod.build_report(PROJECT_ROOT, stage=STAGE, architecture=architecture)
    print(architecture, arch_report["by_status"])
'''),
]

# ---------------------------------------------------------------------------------------------
# 00_Classifier_Matrix_Orchestrator.ipynb (spec section 18)
# ---------------------------------------------------------------------------------------------

ORCHESTRATOR_CELLS = [
    ("markdown", "## Orchestratore matrice classificatori\n\n"
     "Costruisce la matrice, mostra conteggi, e puo' avviare/mettere in pausa lo scheduler. "
     "**Non esegue mai il locked test**: quella cella e' assente per costruzione, non solo "
     "disabilitata — la conferma `--confirm-locked-test` vive esclusivamente in "
     "`scripts/finalize_locked_test_stage.py`, eseguito da riga di comando."),
    ("markdown", "### 1. Costruire/ricostruire la matrice (non distruttivo: lo stato e' ricostruito dagli artefatti)"),
    ("code", BOOTSTRAP + '''
import build_classifier_experiment_matrix as build_matrix
import resume_classifier_experiment_matrix as resume_matrix
import status_classifier_experiment_matrix as status_mod

STAGE = 1  # 1 = screening completo; 2 = solo dopo SELECTED_GENERATOR_UNION firmato

payload = build_matrix.build_and_write(PROJECT_ROOT, STAGE)
stage_jobs = [j for j in payload["jobs"] if j["stage"] == STAGE]
print(f"Stage {STAGE}: {len(stage_jobs)} job")
'''),
    ("markdown", "### 2. Conteggi per architettura / regime / profilo risorsa"),
    ("code", '''
report = status_mod.build_report(PROJECT_ROOT, stage=STAGE)
status_mod.print_report(report)
'''),
    ("markdown", "### 3. Avviare lo scheduler (dry-run di default: imposta DRY_RUN=False solo da riga di comando fuori da questo notebook)"),
    ("code", '''
import run_classifier_experiment_matrix as run_matrix
from classifier_gpu_scheduler import query_gpus_live

DRY_RUN = True  # questo notebook non avvia mai training reali; per il vero avvio usa la CLI
TARGET_5060_JOBS = 3
TARGET_3060_JOBS = 2

gpus = query_gpus_live()
result = run_matrix.run(PROJECT_ROOT, STAGE, "auto", TARGET_5060_JOBS, TARGET_3060_JOBS, dry_run=DRY_RUN, gpus=gpus)
print(f"GPU rilevate: {result['gpus']}")
admitted = [p for p in result["plan"] if p["admitted"]]
print(f"job ammessi in questo piano: {len(admitted)} / {len(result['plan'])}")
for p in admitted[:10]:
    print(" ", p)
'''),
    ("markdown", "### 4. Riprendere dopo un crash/riavvio"),
    ("code", '''
resume_result = resume_matrix.resume(PROJECT_ROOT, stage=STAGE)
print(f"scansionati {resume_result['total_scanned']} job, {len(resume_result['changed'])} cambi di stato")
'''),
    ("markdown", "### 5. Finalizzare validation / preparare il lock (mai eseguito automaticamente da questa cella)"),
    ("code", '''
print("Per finalizzare la validation Stage 1 (calcola SELECTED_GENERATOR_UNION):")
print("  python scripts/finalize_validation_stage.py --stage 1")
print("Per preparare (non bloccare) i pannelli Stage 2:")
print("  python scripts/finalize_validation_stage.py --stage 2")
print("Il locked test richiede sempre un comando esplicito separato, mai una cella notebook:")
print("  python scripts/finalize_locked_test_stage.py --confirm-locked-test")
'''),
]


def main() -> None:
    write_notebook("notebooks/4_comparisons_and_test/00y_Analisi_Consumi_e_Sostenibilita.ipynb",
                    "Analisi Consumi e Sostenibilita'", SUSTAINABILITY_CELLS)
    write_notebook("notebooks/4_comparisons_and_test/05_Experiment_Matrix_Status.ipynb",
                    "Experiment Matrix Status", STATUS_CELLS)
    write_notebook("notebooks/3_classifiers/00_Classifier_Matrix_Orchestrator.ipynb",
                    "Classifier Matrix Orchestrator", ORCHESTRATOR_CELLS)
    print("v2 analysis/orchestration notebooks written.")


if __name__ == "__main__":
    main()
