#!/usr/bin/env python3
"""Additively extend 00z_Confronto_Diffusori.ipynb to cover all 8 generators (spec section 12).

    python scripts/extend_generator_comparison_notebook.py

00z is hand-assembled (not generator-produced like 04x/04y/04z), so this script never rewrites
existing content: it removes only its own previously-appended "Parte D" cells (marked by
PART_D_MARKER, so re-running is idempotent) and appends a fresh Part D at the end. Parts A/B/C
and their preserved static outputs are never touched.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

PART_D_MARKER = "# PART_D_FULL_G01_G08_V1"
NOTEBOOK_PATH = ROOT / "notebooks/4_comparisons_and_test/00z_Confronto_Diffusori.ipynb"

BOOTSTRAP = '''from pathlib import Path
import json, sys

def find_project_root(start=Path.cwd()):
    for candidate in [start, *start.parents]:
        if (candidate / "configs/final_generator_registry.json").is_file():
            return candidate
    raise FileNotFoundError("Project root not found")

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "notebooks/utility"))
'''

CELLS = [
    ("markdown", "# Parte D - Confronto completo G01-G08 (v2)\n\n"
     f"{PART_D_MARKER}\n\n"
     "Estende sistematicamente il confronto a tutti gli otto generatori registrati in "
     "`configs/final_generator_registry.json`, tramite `generator_comparison_analysis.py`. "
     "Le Parti A/B/C restano l'archivio storico e non vengono modificate."),
    ("code", BOOTSTRAP + '''
import generator_comparison_analysis as gca
import matplotlib.pyplot as plt

tables = gca.write_outputs(PROJECT_ROOT)
for name, rows in tables.items():
    print(f"{name}: {len(rows)} righe -> results/generator_comparison/tables/{name}.json")
'''),
    ("markdown", "## D.1 Confronto classe positiva (8/8 generatori) — spec 12.1"),
    ("code", '''
pos = tables["positive_class_comparison"]
for row in pos:
    fid = row.get("FID")
    print(f"{row['generator_id']:35s} FID={fid if fid is not None else 'N/D':>10} n_generated={row.get('n_generated')}")
'''),
    ("markdown", "## D.2 Confronto due classi (solo generatori two-class) — spec 12.2\n\n"
     "Un generatore con metriche incomplete per una classe resta con `status=incomplete_metrics` "
     "ed e' escluso dalla media: non viene mai mostrato un FID medio calcolato su una sola classe."),
    ("code", '''
two = tables["two_class_comparison"]
ok_rows = [r for r in two if r["status"] == "ok"]
incomplete = [r["generator_id"] for r in two if r["status"] != "ok"]
print(f"generatori con confronto due-classi completo: {len(ok_rows)}")
print(f"generatori con metriche incomplete (esclusi dalla media, non da questa tabella): {incomplete}")

if ok_rows:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([r["generator_id"] for r in ok_rows], [r["FID"] for r in ok_rows])
    ax.set_ylabel("FID medio (negative+positive)/2")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    out_dir = PROJECT_ROOT / "results/generator_comparison/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "two_class_fid_all_generators.png", dpi=200)
'''),
    ("markdown", "## D.3 Tabella finale (G02/G03/G04/G07/G08 + eventuali altri role=final_comparison) — spec 12.3"),
    ("code", '''
final = tables["final_comparison"]
for row in final:
    print(row["generator_id"], "FID=", row.get("FID"), "precision=", row.get("precision"), "recall=", row.get("recall"))
'''),
    ("markdown", "## D.4 Ablation (G01vG02, G02vG03, G02/G03vG04, G05vG06, G07vG08) — spec 12.4"),
    ("code", '''
for row in tables["ablation_comparison"]:
    deltas = {k: v for k, v in row.items() if k.startswith(("positive_delta", "negative_delta"))}
    print(row["ablation"], deltas or "(metriche incomplete per uno dei due generatori)")
'''),
    ("markdown", "## D.5 Sampling ablation G08 (25/50/75/100 step) — spec 12.4"),
    ("code", '''
for row in tables["sampling_ablation"]:
    print(row)
'''),
    ("markdown", "## D.6 Nota metodologica\n\n"
     "Nessun vincitore downstream viene dichiarato da questo notebook: la selezione dei "
     "generatori per Stage 2 avviene esclusivamente su validation dei classificatori "
     "(`scripts/finalize_validation_stage.py --stage 1`), mai da FID/IS/PRDC da soli e mai dal "
     "test set."),
]


def build_part_d_cells():
    import nbformat as nbf
    cells = []
    for kind, source in CELLS:
        cell = nbf.v4.new_code_cell(source) if kind == "code" else nbf.v4.new_markdown_cell(source)
        # Every cell (not just the first) is tagged via metadata, so re-running this script can
        # reliably find and remove *all* previously-appended Part D cells, not just the one whose
        # visible source happens to contain PART_D_MARKER.
        cell["metadata"] = {"part_d_generated": True}
        stable_key = f"part_d:{len(cells)}:{kind}:{source}".encode("utf-8")
        cell["id"] = hashlib.sha256(stable_key).hexdigest()[:12]
        cells.append(cell)
    return cells


def _is_part_d_cell(cell) -> bool:
    return bool(cell.get("metadata", {}).get("part_d_generated"))


def main() -> None:
    import nbformat as nbf

    nb = nbf.read(NOTEBOOK_PATH, as_version=4)
    original_count = len(nb.cells)
    nb.cells = [c for c in nb.cells if not _is_part_d_cell(c)]
    removed = original_count - len(nb.cells)
    nb.cells.extend(build_part_d_cells())
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Part D: removed {removed} old cells, appended {len(CELLS)} new cells. "
          f"Total cells now: {len(nb.cells)}.")


if __name__ == "__main__":
    main()
