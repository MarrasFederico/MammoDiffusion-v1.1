#wandb_logger.py
"""
Strato di logging WandB lato Samuele (ex blocco "LOGGING WandB" di pipeline_valutazione).

Unica responsabilita': prendere metriche/figure gia' calcolate e riportarle su WandB.
  - log_experiment_to_wandb : 1 run per configurazione (D1+D2+D3)
  - log_riepilogo_progetto  : 1 run "riepilogo" per il prof (tabella confronto + figure)

Dipende solo da wandb. Non importa la pipeline ne' gli evaluator: la pipeline importa
QUESTO modulo, non viceversa (dipendenza a senso unico).
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb non installato")

WANDB_PROJECT = "mammodiffusion"


#costruisce un wandb.Image con didascalia leggibile a partire dal path del PNG
def _wandb_image(p: Path):
    return wandb.Image(str(p), caption=p.stem.replace("_", " ").title())


#costruisce tag wandb a partire dalla config
def build_tags(config: dict) -> list[str]:
    tags = ["mammodiffusion"]
    if "backbone" in config:  tags.append(config["backbone"])
    if "dataset"  in config:  tags.append(config["dataset"])
    if config.get("augmentation"): tags.append("augmented")
    return tags


"""
carica i risultati delle misurazioni su wandb
----------
-run_name: nome del run su wandb (es. il nome della configurazione)
-config_params: iperparametri/config da loggare (da cui si ricavano anche i tag)
-gen_metrics: {"FID", "IS_mean", "IS_std"}
-class_metrics: {"Accuracy", "F1", "Precision", "Recall"}
-sustainability_metrics: dizionario prefissato (sustainability/classifier/inference/co2_kg)
-figure_paths: path PNG da caricare
-project: nome progetto wandb (default="mammodiffusion")
-entity: profilo wandb
"""
def log_experiment_to_wandb(run_name: str, config_params: dict[str, Any], gen_metrics: dict[str, float],
                            class_metrics: dict[str, float], sustainability_metrics: dict[str, float],
                            figure_paths: list[str | Path], project: str | None = None, entity: str | None = None) -> None:

    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb non installato")
    #crea e avvia il run wandb
    run = wandb.init(project=project or WANDB_PROJECT, entity=entity, name=run_name,
                    config=config_params, tags=build_tags(config_params))
    try:
        wandb.log({f"generative/{k}": v     for k, v in gen_metrics.items()}) #metriche generative
        wandb.log({f"classification/{k}": v for k, v in class_metrics.items()}) #metriche classificatore

        if sustainability_metrics:
            wandb.log(sustainability_metrics)

        #carica le figure (confusion matrix) come immagini wandb; le mancanti vengono saltate con warning
        images = {}
        for p in figure_paths:
            p = Path(p)
            if not p.exists():
                print(f"[wandb] WARN figura non trovata, saltata: {p.resolve()}")
                continue
            images[f"figures/{p.stem}"] = _wandb_image(p)
        if images:
            wandb.log(images)

        #risponde alla domanda D3 calcolando il costo totale dell'approccio generativo (training + generation)
        total_gen_co2 = (
            (sustainability_metrics.get("sustainability/generator/training/co2_kg")   or 0.0) +
            (sustainability_metrics.get("sustainability/generator/generation/co2_kg") or 0.0)
        )

        wandb.run.summary.update({
            "best/FID":            gen_metrics.get("FID"),
            "best/IS_mean":        gen_metrics.get("IS_mean"),
            "best/Accuracy":       class_metrics.get("Accuracy"),
            "best/F1":             class_metrics.get("F1"),
            "d3/total_gen_co2_kg": total_gen_co2,
        })

        url = run.url or "(offline)"
        print(f"[wandb] Run '{run_name}' -> {url}")
    finally:
        wandb.finish()


"""
[riepilogo per il prof] logga UN SOLO run wandb con tutto il progetto in una vista:
-tabella di confronto dei classificatori (D2)
-tutte le immagini/plot (confusion matrix + plot di riepilogo) come PNG
-numeri chiave D1 (FID/IS) e D3 (CO2) in evidenza nel summary del run
"""
def log_riepilogo_progetto(summary: dict, figure_paths, project: str | None = None,
                           entity: str | None = None, run_name: str = "RIEPILOGO_PROGETTO") -> None:
    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb non installato")
    run = wandb.init(project=project or WANDB_PROJECT, entity=entity, name=run_name,
                     tags=["mammodiffusion", "riepilogo"], config={"tipo": "riepilogo_progetto"})
    try:
        d1 = summary.get("D1", {})
        d2 = summary.get("D2", {})
        d3 = summary.get("D3", {})

        #tabella di confronto delle configurazioni del classificatore (D2)
        per = d2.get("per_config", {})
        if per:
            cols = ["config", "F1", "Accuracy", "ROC_AUC", "Sensitivity", "Specificity", "delta_F1_vs_baseline"]
            tab = wandb.Table(columns=cols)
            for nome, m in per.items():
                tab.add_data(nome, m.get("F1"), m.get("Accuracy"), m.get("ROC_AUC"),
                             m.get("Sensitivity"), m.get("Specificity"), m.get("delta_F1_vs_baseline"))
            wandb.log({"D2/confronto_classificatori": tab})

        #tutte le figure come immagini (confusion matrix per config + plot di riepilogo)
        for p in figure_paths:
            p = Path(p)
            if p.exists():
                wandb.log({f"figure/{p.stem}": _wandb_image(p)})

        #numeri chiave in evidenza nel summary del run (vista veloce per il prof)
        ranking = d2.get("ranking_per_F1") or []
        wandb.run.summary.update({
            "D1/FID":               d1.get("FID"),
            "D1/IS_mean":           d1.get("IS_mean"),
            "D2/baseline":          d2.get("baseline"),
            "D2/migliore_per_F1":   ranking[0] if ranking else None,
            "D3/generatore_co2_kg": d3.get("generatore_co2_totale_kg"),
        })
        print(f"[wandb] RIEPILOGO_PROGETTO -> {run.url or '(offline)'}")
    finally:
        wandb.finish()
