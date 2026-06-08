#pipeline che raccoglie i risultati su WandB per MammoDiffusion
#-valuta_configurazione(...) -> valuta una configurazione e logga 1 run WandB
#-valuta_tutte_configurazioni(...) -> valuta una o piu' configurazioni sulle stesse immagini generate
from __future__ import annotations
import os, sys, json
from datetime import datetime
from pathlib import Path
from classifier_evaluator  import ClassifierEvaluator
from eco_tracker           import SustainabilityMetrics
from wandb_logger          import log_experiment_to_wandb
try:
    from generative_evaluator import GenerativeEvaluator
except ImportError:
    GenerativeEvaluator = None  #type: ignore[assignment]
DEFAULT_OUTPUT_DIR = "risultati_finali"


#duplica uno stream su console + file log.txt
class _Tee:
    #streams - sui quali scrivere
    def __init__(self, *streams):
        self.streams = streams
    #scrive stringa su tutti gli stream
    def write(self, s):
        for st in self.streams:
            try: st.write(s)
            except Exception: pass
        return len(s)
    #forza svuotamento dei buffer di tutti gli stream
    def flush(self):
        for st in self.streams:
            try: st.flush()
            except Exception: pass


#calcola FID e IS sulle immagini (solleva un errore se manca lo stack torch)
def compute_generative(real_images_dir: str, fake_images_dir: str, device: str = "auto") -> dict:
    if GenerativeEvaluator is None:
        raise RuntimeError("esegui pip install torch torchvision torchmetrics")
    _device = None if device == "auto" else device
    return GenerativeEvaluator(
        real_dir=real_images_dir,
        generated_dir=fake_images_dir,
        batch_size=16,
        device=_device,
    ).compute()


"""
aggiunge al summary i blocchi con metriche necessarie alle domande D1/D2/D3
- D1: FID / IS
- D2: metriche classificatore per config + delta vs baseline + ranking
- D3: CO2 del generatore (totale) e del classificatore (per config)
"""
def aggiungi_aggregati(summary: dict, configurazioni: dict, eco_diff_train, eco_diff_gen) -> None:

    #somma sicura del co2 di un eco (0 se non fornito)
    def co2(eco):
        return eco.co2_kg if eco is not None else 0.0

    summary["D1"] = dict(summary["generative"])#D1 - copia valori di FID e IS 

    runs = summary["runs"]#i risultati per ogni configurazione
    baseline = "real_only" if "real_only" in runs else next(iter(runs))#config di riferimento per i delta
    base = runs[baseline]["classifier"] #metriche baseline
    per_config = {}#raccoglie metriche + delta per ogni configurazione
    #per ogni config calcola le metriche chiave e i delta rispetto alla baseline
    for nome, ris in runs.items():
        clf = ris["classifier"]#metriche del classificatore di questa config
        d_auc = None #delta ROC-AUC - None se non disponibile
        #il delta AUC si calcola solo se entrambe le AUC esistono
        if clf.get("ROC_AUC") is not None and base.get("ROC_AUC") is not None:
            d_auc = clf["ROC_AUC"] - base["ROC_AUC"]
        #singola riga della config con metriche assolute + delta vs baseline
        per_config[nome] = {
            "F1": clf["F1"], "Accuracy": clf["Accuracy"],
            "ROC_AUC": clf.get("ROC_AUC"),
            "Sensitivity": clf.get("Sensitivity"), "Specificity": clf.get("Specificity"),
            "delta_F1_vs_baseline": clf["F1"] - base["F1"],
            "delta_ROC_AUC_vs_baseline": d_auc,
        }
    ranking = sorted(runs, key=lambda n: runs[n]["classifier"]["F1"], reverse=True)#config ordinate per F1 decrescente

    #D2 - confronto tra le configurazioni
    summary["D2"] = {
        "baseline": baseline, "ranking_per_F1": ranking, "per_config": per_config,
    }

    #D3 - costo CO2 (generatore totale + classificatore per config)
    summary["D3"] = {
        "generatore_co2_totale_kg": co2(eco_diff_train) + co2(eco_diff_gen),
        "classificatore_co2_per_config_kg": {
            nome: co2(c.get("eco_classifier_training")) + co2(c.get("eco_inference"))
            for nome, c in configurazioni.items()
        },
    }


"""
esegue D1+D2+D3 per UNA configurazione e logga 1 run WandB; ritorna un dict
-output_dir: cartella su cui ClassifierEvaluator salva le figure (CM)
-pos_label: etichetta della classe "malato"
-config_params: voci extra del config WandB
-gen_metrics: se fornito il FID/IS non viene ricalcolato e si riusano questi valori 
-project/entity: destinazione WandB
"""
def valuta_configurazione(
    run_name: str,
    #dal generatore
    real_images_dir: str,
    fake_images_dir: str,
    #dal classificatore
    y_true, y_pred, y_prob,
    class_names: list,
    #metriche di sostenibilità
    eco_inference:            SustainabilityMetrics | None = None,
    eco_classifier_training:  SustainabilityMetrics | None = None,
    eco_diffusion_training:   SustainabilityMetrics | None = None,
    eco_diffusion_generation: SustainabilityMetrics | None = None,
    #configurazione
    device: str = "auto",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    pos_label: int = 1,
    config_params: dict | None = None,
    gen_metrics: dict | None = None,
    project: str | None = None,
    entity: str | None = None,
) -> dict:
    print(f"\n{'=' * 60}\n  valutazione: {run_name}\n{'=' * 60}")

    #metriche generative (D1)
    if gen_metrics is None:
        print("\nmetriche generative (FID, IS):")
        gen_results = compute_generative(real_images_dir, fake_images_dir, device)
    else:
        print("\nmetriche generative (valori pre-calcolati):")
        gen_results = gen_metrics
    print(f"  FID: {gen_results['FID']:.4f} | IS mean: {gen_results['IS_mean']:.4f}")

    #metriche classificatore (D2)
    print("\nmetriche classificatore:")
    class_results = ClassifierEvaluator(output_dir=output_dir).evaluate_predictions(
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        label=run_name, class_names=class_names, pos_label=pos_label,
    )
    print(f"  F1: {class_results.f1:.4f} | Accuracy: {class_results.accuracy:.4f}")

    #costruisce il dizionario di sostenibilità e trasforma SustainabilityMetrics nelle chiavi WandB
    def eco_dict(eco: SustainabilityMetrics | None, prefix: str) -> dict:
        if eco is None:
            print(f"    '{prefix}' non fornite -> zeri.")
            return {f"{prefix}/elapsed_s": 0.0, f"{prefix}/peak_ram_mb": 0.0,
                    f"{prefix}/energy_kwh": 0.0, f"{prefix}/co2_kg": 0.0}
        return {f"{prefix}/elapsed_s":   eco.elapsed_seconds,
                f"{prefix}/peak_ram_mb": eco.peak_ram_mb,
                f"{prefix}/energy_kwh":  eco.energy_kwh,
                f"{prefix}/co2_kg":      eco.co2_kg}

    sustainability = {}
    sustainability.update(eco_dict(eco_inference,            "sustainability/classifier/inference"))
    sustainability.update(eco_dict(eco_classifier_training,  "sustainability/classifier/training"))
    sustainability.update(eco_dict(eco_diffusion_training,   "sustainability/generator/training"))
    sustainability.update(eco_dict(eco_diffusion_generation, "sustainability/generator/generation"))

    #logging WandB
    print("\ncaricando su WandB")
    cfg = {"dataset": "RSNA_Breast_Cancer", "image_size": 299}
    if config_params:
        cfg.update(config_params)
    log_experiment_to_wandb(
        run_name=run_name,
        config_params=cfg,
        gen_metrics=gen_results,
        class_metrics=class_results.to_dict(),
        sustainability_metrics=sustainability,
        figure_paths=class_results.figure_paths,
        project=project,
        entity=entity,
    )

    cm = class_results.confusion_matrix
    return {
        "run_name":       run_name,
        "generative":     gen_results,
        "classifier":     class_results.to_dict(),
        #confusion matrix come numeri 
        "confusion_matrix":        cm.tolist() if cm is not None else None,
        "confusion_matrix_classi": list(class_names) if class_names else None,
        #metriche per ogni classe
        "per_class":      class_results.per_class,
        "sustainability": sustainability,
        "figure_paths":   [str(p) for p in class_results.figure_paths],
    }


"""
valuta una o piu' configurazioni del classificatore sulle stesse immagini generate e raccoglie in output_dir
-il FID/IS viene calcolato e riusato per tutti i run
-config: dict {nome_run: cfg}dove ogni cfg e' un dizionario con
    y_true, y_pred, class_names             obbligatori
    y_prob                                  opz per ROC-AUC
    eco_classifier_training, eco_inference  opz SustainabilityMetrics
    pos_label                               opz default=1
    config_params                           opz dict extra per WandB
-project/entity: progetto e account wandb
-output_dir: directory che contiene i risultati
"""
def valuta_tutte_configurazioni(
    real_images_dir: str,
    fake_images_dir: str,
    configurazioni: dict,
    eco_diffusion_training:   SustainabilityMetrics | None = None,
    eco_diffusion_generation: SustainabilityMetrics | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    project: str | None = "mammodiffusion",
    entity: str | None = None,
    device: str = "auto",
) -> dict:
    out = Path(output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    #mirror che copia in locale i run wandb dentro la cartella dei risultati
    os.environ["WANDB_DIR"] = str(out.resolve())

    real_out, real_err = sys.stdout, sys.stderr
    logfile = open(out / "log.txt", "w", encoding="utf-8")
    try:
        #swap degli stream ma se fallisce il finally chiude comunque il file
        sys.stdout, sys.stderr = _Tee(real_out, logfile), _Tee(real_err, logfile)
        ts = datetime.now().isoformat(timespec="seconds")
        print(f"\n{'#' * 64}\nValuta ogni configurazione -  {ts}")
        print(f"#output: {out.resolve()}")
        print(f"#WandB: project={project!r} entity={entity!r} "
              f"(mode={os.environ.get('WANDB_MODE', 'online')})\n{'#' * 64}")

        #FID/IS calcolato una sola volta
        print("\n[D1] calcolo FID/IS (singolo)")
        gen_metrics = compute_generative(real_images_dir, fake_images_dir, device)
        print(f"     FID={gen_metrics['FID']:.4f} | IS_mean={gen_metrics['IS_mean']:.4f} "
              f"| IS_std={gen_metrics['IS_std']:.4f}")

        summary = {
            "timestamp": ts, "output_dir": str(out.resolve()),
            "project": project, "entity": entity,
            "generative": gen_metrics, "runs": {},
        }
        for nome, c in configurazioni.items():
            ris = valuta_configurazione(
                run_name=nome,
                real_images_dir=real_images_dir, fake_images_dir=fake_images_dir,
                y_true=c["y_true"], y_pred=c["y_pred"], y_prob=c.get("y_prob"),
                class_names=c["class_names"],
                eco_inference=c.get("eco_inference"),
                eco_classifier_training=c.get("eco_classifier_training"),
                eco_diffusion_training=eco_diffusion_training,
                eco_diffusion_generation=eco_diffusion_generation,
                device=device, output_dir=out,
                pos_label=c.get("pos_label", 1),
                config_params=c.get("config_params"),
                gen_metrics=gen_metrics,          #ricicla il FID calcolato una volta
                project=project, entity=entity,
            )
            summary["runs"][nome] = ris

        #blocchi aggregati pronti per il report (D1/D2/D3)
        aggiungi_aggregati(summary, configurazioni, eco_diffusion_training, eco_diffusion_generation)

        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        #riepilogo D2 a video ordinato per F1
        print("\n[D2] confronto configurazioni (ordinate per F1):")
        base = summary["D2"]["baseline"]
        for nome in summary["D2"]["ranking_per_F1"]:
            pc = summary["D2"]["per_config"][nome]
            dF1 = pc["delta_F1_vs_baseline"]
            auc_str = f"{pc['ROC_AUC']:.4f}" if pc["ROC_AUC"] is not None else "N/A"
            print(f"     {nome:24s} F1={pc['F1']:.4f}  ROC_AUC={auc_str}  "
                  f"(dF1 vs {base}: {dF1:+.4f})")
        print(f"\nvalutazione terminata raccolta in: {out.resolve()}")
        print(f"     configurazioni: {list(summary['runs'])}")
        return summary
    finally:
        sys.stdout, sys.stderr = real_out, real_err
        logfile.close()


#test con mock data (no torch no WandB online)
if __name__ == "__main__":
    import tempfile

    os.environ.setdefault("WANDB_MODE", "offline")

    class _StubGenerativeEvaluator:
        def __init__(self, *args, **kwargs) -> None: pass
        def compute(self) -> dict:
            return {"FID": 37.5, "IS_mean": 1.92, "IS_std": 0.11}

    GenerativeEvaluator = _StubGenerativeEvaluator 

    def mock_eco(label, t=1.0, ram=512.0, kwh=0.001, co2=0.0004):
        return SustainabilityMetrics(elapsed_seconds=t, peak_ram_mb=ram,
                                     energy_kwh=kwh, co2_kg=co2, label=label)

    summary = valuta_tutte_configurazioni(
        real_images_dir="percorso/immagini/reali",     #ignorato dallo stub
        fake_images_dir="percorso/immagini/generate",   #ignorato dallo stub
        configurazioni={
            "Test_Pipeline_Mock": dict(
                y_true=[0, 1, 1, 0, 1], y_pred=[0, 1, 0, 0, 1],
                y_prob=[[0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [0.7, 0.3], [0.2, 0.8]],
                class_names=["Sano", "Malato"],
                eco_inference=          mock_eco("inference",           t=0.45,   kwh=0.00005),
                eco_classifier_training=mock_eco("classifier_training", t=120.0,  kwh=0.010),
            ),
        },
        eco_diffusion_training=  mock_eco("diffusion_training",  t=3600.0, kwh=0.500),
        eco_diffusion_generation=mock_eco("diffusion_generation", t=180.0, kwh=0.020),
        output_dir=tempfile.mkdtemp(prefix="selftest_pipeline_"),
    )
    print("summary keys:", list(summary))
