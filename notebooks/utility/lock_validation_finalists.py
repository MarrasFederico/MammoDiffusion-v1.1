"""Freeze validation-selected finalists without reading any test metric."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from final_classifier_evaluation import (  # noqa: E402
    build_experiment_registry,
    build_locked_finalist_entries,
    build_test_coverage_table,
    lock_finalists_manifest,
    select_validation_finalists,
)

POLICY = {
    "include_baseline_per_architecture": True,
    "include_best_synthetic_per_architecture": True,
    "include_best_augmented_per_architecture": True,
    "include_best_overall_per_architecture": True,
    "max_finalists_total": 10,
}


def main():
    registry = build_experiment_registry(ROOT / "configs/final_classifier_registry.json")
    coverage = build_test_coverage_table(registry, ROOT)
    selected = select_validation_finalists(coverage, POLICY)
    finalists = build_locked_finalist_entries(selected, ROOT)
    output = ROOT / "results/final_evaluation"
    manifest = lock_finalists_manifest(finalists, POLICY, output / "finalists_manifest.json")
    (output / "FINALISTS_LOCKED").write_text(manifest["lock_signature"]["sha256"] + "\n", encoding="utf-8")
    ready_for_inference = [item["experiment_id"] for item in finalists if item["test_status"] == "READY_FOR_TEST_INFERENCE"]
    legacy = [item["experiment_id"] for item in finalists if item.get("blocked_reason") == "unverified_prediction_provenance"]
    blocked = [f"- `{item['experiment_id']}`: {item['blocked_reason']}" for item in finalists if item.get("blocked_reason")]
    conclusions = (
        "# Conclusioni finali — stato corrente della pipeline\n\n"
        "La selezione scientifica è congelata esclusivamente sul validation, ma **non è operativamente completa**. "
        "Non viene dichiarato alcun vincitore e non sono state lette metriche test per questa decisione.\n\n"
        f"Inferenze locked ancora da eseguire: {ready_for_inference or 'nessuna'}. "
        f"Predizioni Mammo-FM legacy che richiedono reinferenza o accettazione esplicita: {legacy or 'nessuna'}.\n\n"
        "## Blocker\n\n" + ("\n".join(blocked) if blocked else "Nessuno.") + "\n\n"
        "Il confronto finale non può ancora coprire correttamente ResNet-50: il finalista validation ResNet non ha un checkpoint recuperabile. "
        "Le predizioni Mammo-FM normalizzate non sono trattate come native.\n\n"
        "## Limiti metodologici\n\nIl test era già stato osservato durante lo sviluppo. I futuri risultati devono riportare CI, confronti paired e Holm; "
        "nessuna differenza puntuale sarà presentata come vittoria senza supporto statistico. Serve inoltre conferma esterna o grouped cross-validation.\n"
    )
    (output / "final_conclusions.md").write_text(conclusions, encoding="utf-8")
    print(json.dumps({
        "scientifically_locked": [x["experiment_id"] for x in finalists],
        "scientific_selection_complete": manifest["scientific_selection_complete"],
        "final_aggregation_complete": manifest["final_aggregation_complete"],
        "operational_blockers": manifest["operational_blockers"],
    }, indent=2))


if __name__ == "__main__":
    main()
