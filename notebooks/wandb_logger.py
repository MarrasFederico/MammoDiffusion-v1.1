#logging per Weights and Biases per MammoDiffusion
from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb non installato")


WANDB_PROJECT = "mammodiffusion"


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
            images[f"figures/{p.stem}"] = wandb.Image(str(p), caption=p.stem.replace("_"," ").title())
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


#costruisce tag wandb a partire dalla config
def build_tags(config: dict) -> list[str]:
    tags = ["mammodiffusion"]
    if "backbone" in config:  tags.append(config["backbone"])
    if "dataset"  in config:  tags.append(config["dataset"])
    if config.get("augmentation"): tags.append("augmented")
    return tags