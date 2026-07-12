#!/usr/bin/env python3
"""Generate 04x_v2/04y_v2/04z_v2 (spec sections 14, 16, 17), reading the NEW classifier
experiment matrix instead of the original 22-experiment final_classifier_registry.json, without
touching the existing 04x/04y/04z notebooks or their locked pipeline.

    python scripts/generate_v2_locked_test_notebooks.py

04y_v2 and 04z_v2 both call finalize_locked_test_stage.verify_lock_still_valid() as their very
first executable cell and stop immediately if it fails — no cell below that point can read a
prediction or metric file. Running this generator does not read the test set; the generated
notebooks themselves refuse to read it until a real lock exists and validates.
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
# 04x_v2: leaderboard over the full new matrix (spec section 14)
# ---------------------------------------------------------------------------------------------

LEADERBOARD_V2_CELLS = [
    ("markdown", "## Leaderboard validation — matrice completa (v2)\n\n"
     "Legge `configs/classifier_experiment_matrix.json` e le `validation_metrics.json` per job. "
     "Non apre mai un file di test. Il ranking primario e' PR-AUC; ROC-AUC e' secondario "
     "(spec 7.1)."),
    ("code", BOOTSTRAP + '''
import classifier_metrics as cm
import finalize_validation_stage as fvs

matrix_path = PROJECT_ROOT / "configs/classifier_experiment_matrix.json"
matrix = json.loads(matrix_path.read_text()) if matrix_path.is_file() else {"jobs": []}
print(f"job totali in matrice: {len(matrix['jobs'])}")
'''),
    ("markdown", "### Ranking per architettura, per regime (controlled/full/positive-only) — spec 14.1"),
    ("code", '''
import sys as _sys
sys.path.insert(0, str(PROJECT_ROOT / "notebooks/utility"))
from dataset_variant_registry import load_generator_registry  # noqa

dataset_registry = json.loads((PROJECT_ROOT / "configs/dataset_variant_registry.json").read_text())
variants_by_id = {v["dataset_variant_id"]: v for v in dataset_registry["variants"]}

rows = fvs.load_completed_validations(PROJECT_ROOT, stage=1)
for architecture in ("resnet50", "maxvit512", "mammofm", "raddino"):
    ranking = fvs.rank_by_generator(rows, architecture)
    print(f"\\n{architecture}: {len(ranking)} generatori con almeno un seed validato")
    for entry in ranking[:5]:
        print(f"  #{entry['rank']} {entry['generator_id']}: mean_pr_auc={entry['mean_pr_auc']:.4f} (n_seeds={entry['n_seeds']})")
'''),
    ("markdown", "### Seed stability (spec 14.1)"),
    ("code", '''
by_experiment_family = {}
for row in rows:
    key = (row["architecture"], row["dataset_variant_id"])
    by_experiment_family.setdefault(key, []).append(row["pr_auc"])
for key, values in list(by_experiment_family.items())[:10]:
    if len(values) > 1:
        print(key, cm.seed_stability(values))
'''),
    ("markdown", "### SELECTED_GENERATOR_UNION (spec 3.5 / 14.4) — validation-only, mai dal test"),
    ("code", '''
union_payload = fvs.compute_selected_generator_union(PROJECT_ROOT, stage=1)
print(f"n_completed_jobs_considered: {union_payload['n_completed_jobs_considered']}")
print(f"SELECTED_GENERATOR_UNION: {union_payload['selected_generator_union']}")
print(f"selection_used_test_data: {union_payload['selection_used_test_data']}")
'''),
]

# ---------------------------------------------------------------------------------------------
# 04y_v2: locked test (spec section 16) — gated behind verify_lock_still_valid
# ---------------------------------------------------------------------------------------------

LOCKED_TEST_V2_CELLS = [
    ("markdown", "## Final Test Locked — matrice completa (v2)\n\n"
     "**Ogni cella sotto il controllo del lock si ferma se il lock non e' valido.** Nessuna "
     "predizione o metrica di test viene letta prima che `verify_lock_still_valid` ritorni "
     "`True`. Eseguire questo notebook non equivale a eseguire il test: se il lock manca, il "
     "notebook si arresta immediatamente senza aprire `data/processed/metadata/test.csv`."),
    ("code", BOOTSTRAP + '''
import finalize_locked_test_stage as lock

is_valid, problems = lock.verify_lock_still_valid(PROJECT_ROOT)
print(f"lock valido: {is_valid}")
if not is_valid:
    for p in problems:
        print(" -", p)
    raise SystemExit(
        "Locked test non eseguibile: il lock manca o non e' piu' valido. "
        "Esegui prima 'python scripts/finalize_locked_test_stage.py --confirm-locked-test' "
        "dopo aver congelato Stage 1 e Stage 2."
    )
'''),
    ("markdown", "### Predizioni per seed + ensemble\n\n"
     "**Nota di sicurezza:** in un kernel Jupyter un `SystemExit` in una cella non impedisce "
     "l'esecuzione delle celle successive (a differenza di uno script). Per questo ogni cella "
     "sotto interroga di nuovo `is_valid`, invece di fidarsi del solo arresto della cella precedente."),
    ("code", '''
assert is_valid, "lock non valido: nessuna lettura di dati di test consentita da questa cella"
lock_dir = PROJECT_ROOT / "results/final_evaluation_v2"
secondary_panel = json.loads((lock_dir / "secondary_panel_manifest.json").read_text())
print(f"pannello secondario locked: {len(secondary_panel['experiment_ids'])} esperimenti")

import classifier_metrics as cm
import locked_matrix_inference as locked_inference

TEST_PREDICTIONS_SCHEMA = ["patient_id", "image_id", "label", "prob_seed_17", "prob_seed_42",
                           "prob_seed_73", "prob_ensemble", "predicted_label", "threshold"]
print("Schema predizioni atteso per ogni configurazione:", TEST_PREDICTIONS_SCHEMA)
RUN_LOCKED_TEST = False  # impostare True soltanto nella sessione one-shot autorizzata
if RUN_LOCKED_TEST:
    assert is_valid, "lock non valido"
    locked_manifest = locked_inference.run_locked(PROJECT_ROOT)
    print(json.dumps(locked_manifest, indent=1))
else:
    print("DRY GUARD: inferenza non eseguita; nessun file test aperto da questa cella.")
'''),
]

# ---------------------------------------------------------------------------------------------
# 04z_v2: statistics (spec section 17) — also gated behind the same lock check
# ---------------------------------------------------------------------------------------------

STATS_V2_CELLS = [
    ("markdown", "## Final Statistical Comparison — matrice completa (v2)\n\n"
     "Stesse regole di 04y_v2: nessuna metrica di test viene letta se il lock non e' valido."),
    ("code", BOOTSTRAP + '''
import finalize_locked_test_stage as lock
import classifier_statistics as cs

is_valid, problems = lock.verify_lock_still_valid(PROJECT_ROOT)
print(f"lock valido: {is_valid}")
if not is_valid:
    for p in problems:
        print(" -", p)
    raise SystemExit("Statistiche non calcolabili: nessun lock valido, nessuna metrica di test disponibile.")
'''),
    ("markdown", "### Famiglie Holm separate (spec 17.4): primary_roc_auc, primary_pr_auc, primary_mcnemar, secondary_*\n\n"
     "Come in 04y_v2, ogni cella ricontrolla `is_valid` autonomamente: un `SystemExit` in una "
     "cella precedente non blocca l'esecuzione delle celle successive in un kernel Jupyter."),
    ("code", '''
assert is_valid, "lock non valido: nessuna statistica calcolabile da questa cella"
FAMILIES = ("primary_roc_auc", "primary_pr_auc", "primary_mcnemar",
            "secondary_roc_auc", "secondary_pr_auc", "secondary_mcnemar")
print("Famiglie di correzione multipla dichiarate (mai unite tra loro):", FAMILIES)
print("bootstrap/DeLong/McNemar/Holm sono disponibili in classifier_statistics.py "
      "(paired_stratified_bootstrap, delong_test, mcnemar_test, holm_correction); "
      "questa cella verra' popolata con le vere coppie di confronto primary/secondary "
      "una volta che 04y_v2 avra' scritto predizioni di test reali.")
'''),
]


def main() -> None:
    write_notebook("notebooks/4_comparisons_and_test/04x_v2_Leaderboard_Validation_Matrix.ipynb",
                    "Leaderboard Validation — Matrice Completa (v2)", LEADERBOARD_V2_CELLS)
    write_notebook("notebooks/4_comparisons_and_test/04y_v2_Final_Test_Locked_Matrix.ipynb",
                    "Final Test Locked — Matrice Completa (v2)", LOCKED_TEST_V2_CELLS)
    write_notebook("notebooks/4_comparisons_and_test/04z_v2_Final_Statistical_Comparison_Matrix.ipynb",
                    "Final Statistical Comparison — Matrice Completa (v2)", STATS_V2_CELLS)
    print("v2 leaderboard/locked-test/statistics notebooks written.")


if __name__ == "__main__":
    main()
