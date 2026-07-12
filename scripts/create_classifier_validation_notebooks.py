#!/usr/bin/env python3
"""Generate deterministic validation-only comparison notebooks for the v2 matrix."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "04a_v2_Validation_ResNet50_Matrix.ipynb": ("ResNet-50", "resnet50"),
    "04b_v2_Validation_MaxViT512_Matrix.ipynb": ("MaxViT-512", "maxvit512"),
    "04c_v2_Validation_MammoFM_Matrix.ipynb": ("Mammo-FM", "mammofm"),
    "04d_v2_Validation_RADDINO_Matrix.ipynb": ("RAD-DINO", "raddino"),
    "04e_v2_Validation_CrossArchitecture.ipynb": ("Cross-architecture", None),
    "04f_v2_Dataset_and_Generator_Comparison.ipynb": ("Dataset e generatori", None),
    "04g_v2_Seed_Stability_and_Calibration.ipynb": ("Stabilità seed e calibrazione", None),
    "04h_v2_Performance_vs_Compute.ipynb": ("Performance vs compute", None),
}


def _cell(kind, source, key, index):
    result = {"cell_type": kind, "id": hashlib.sha256(f"{key}:{index}".encode()).hexdigest()[:16],
              "metadata": {}, "source": [line + "\n" for line in source.rstrip().splitlines()]}
    if kind == "code": result.update({"execution_count": None, "outputs": []})
    return result


def build(name, title, architecture):
    code = f'''from pathlib import Path
import json
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs/classifier_experiment_matrix.json").is_file())
ARCHITECTURE = {architecture!r}
paths = sorted((ROOT / "results/classifiers_matrix").glob("*/*/*/ensemble_validation_manifest.json"))
rows = []
for path in paths:
    payload = json.loads(path.read_text())
    if ARCHITECTURE and payload["architecture"] != ARCHITECTURE:
        continue
    vid = payload["dataset_variant_id"]
    regime = ("controlled" if "CONTROLLED" in vid else "full" if "FULL" in vid else
              "positive_only" if "POSITIVE" in vid or vid.startswith("RSP_") else "baseline")
    rows.append({{"architecture": payload["architecture"], "dataset_variant": vid,
                 "regime": regime, "aggregation": "ensemble_3_seed", **payload["metrics"],
                 "pr_auc_seed_std": payload["seed_stability"]["pr_auc"]["std"]}})
print("ensemble validation disponibili:", len(rows))
for regime in ("baseline", "controlled", "full", "positive_only"):
    ranked = sorted((r for r in rows if r["regime"] == regime), key=lambda r: r["pr_auc"], reverse=True)
    print("\\n", regime, "(ensemble only)")
    for row in ranked[:20]: print(row)
assert all(row["aggregation"] == "ensemble_3_seed" for row in rows)
print("Nessun file di test è stato letto.")'''
    key = name.removesuffix(".ipynb")
    return {"cells": [
        _cell("markdown", f"# {title}\n\nConfronto validation-only. Ranking separati per baseline, controlled, full e positive-only; "
              "l’unità primaria è l’ensemble dei tre seed, con mean/std dei seed come stabilità.", key, 0),
        _cell("code", code, key, 1),
        _cell("markdown", "## Confronti preregistrati\n\nR vs RA; R vs RS controlled/full; generatori controlled e full separati; "
              "positive-only, RAS e synthetic-only; confronto cross-architecture sulla stessa variante e delta dalla baseline. "
              "Le sezioni restano vuote, senza valori inventati, finché gli ensemble reali non esistono.", key, 2),
    ], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3"},
                      "mammodiffusion": {"validation_only": True, "architecture": architecture}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    out = ROOT / "notebooks/4_comparisons_and_test"; out.mkdir(parents=True, exist_ok=True)
    for name, (title, architecture) in SPECS.items():
        (out / name).write_text(json.dumps(build(name, title, architecture), ensure_ascii=False,
                                                indent=1, sort_keys=True) + "\n")
    print(f"validation comparison notebooks: {len(SPECS)}")


if __name__ == "__main__": main()
