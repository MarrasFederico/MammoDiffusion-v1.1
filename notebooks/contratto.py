#contratto.py
"""
Modulo CONDIVISO da tutto il gruppo (generatore + classificatore + Samuele).

Riunisce due responsabilita' che ogni collega usa sempre insieme:

  1) MISURA della sostenibilita' (ex eco_tracker):
     - SustainabilityMetrics  : dataclass coi consumi (tempo, RAM, energia, CO2)
     - measure_sustainability : context manager per misurare un blocco di codice
     - compare_sustainability : confronto fra due misurazioni

  2) SCAMBIO DATI su file (ex io_contratto):
     - eco        : SustainabilityMetrics <-> JSON/JSONL  (salva_eco/carica_eco/aggrega_eco)
     - predizioni : y_true/y_pred/y_prob/class_names <-> .npz
     - metriche   : dizionari accuracy/F1/FID/IS <-> JSON
     - consegna_classificatore / consegna_generatore : UNA chiamata che misura/valuta
       e deposita tutto nella CARTELLA DI CONSEGNA condivisa per Samuele.

DUE cartelle diverse, non confonderle:
  - CARTELLA_CONSEGNE ("consegne/")  = INGRESSO: dove tutti depositano cio' che misurano/valutano
  - risultati_finali/                = USCITA: cio' che produce pipeline_valutazione (figure/log/summary/wandb)

Dipendenze volutamente LEGGERE: psutil + codecarbon (misura) e numpy (I/O .npz).
Nessun import di sklearn/torch a livello di modulo: ClassifierEvaluator viene
importato in modo lazy solo dentro consegna_classificatore.
"""
from __future__ import annotations
import contextlib, json, os, threading, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import psutil
from codecarbon import EmissionsTracker


# ============================================================================
#  PARTE 1 - MISURA DELLA SOSTENIBILITA' (ex eco_tracker)
# ============================================================================

#contiene i consumi misurati per un blocco di codice (tempo, RAM, energia, CO2)
@dataclass
class SustainabilityMetrics:
    elapsed_seconds: float = 0.0
    peak_ram_mb: float = 0.0
    start_ram_mb: float = 0.0
    energy_kwh: float = 0.0
    co2_kg: float = 0.0
    label: str = "run"

    #stringa riassuntiva
    def __str__(self) -> str:
        return (
            f"[{self.label}] "
            f"Tempo: {self.elapsed_seconds:.2f}s | "
            f"RAM picco: {self.peak_ram_mb:.1f} MB | "
            f"Energia: {self.energy_kwh:.6f} kWh | "
            f"CO2: {self.co2_kg:.6f} kg"
        )

    #dizionario per il salvataggio delle metriche
    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_ram_mb": self.peak_ram_mb,
            "energy_kwh": self.energy_kwh,
            "co2_kg": self.co2_kg,
        }


#campiona la RAM del processo per stimarne il picco in MB
class _RamMonitor:
    def __init__(self, interval: float = 0.05) -> None:
        self._interval = interval #secondi tra un campione di RAM e il successivo
        self._process = psutil.Process(os.getpid()) #handle del processo
        self._peak: float = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None #thread di campionamento

    #avvia thread di campionamento
    def start(self) -> None:
        self._peak = self.current_mb() #inizializzato al valore corrente
        self._running = True
        self._thread = threading.Thread(target=self.sample, daemon=True)
        self._thread.start()

    #termina il thread e rende il picco di RAM in MB
    def stop(self) -> float:
        self._running = False
        if self._thread:
            self._thread.join()
        return self._peak

    #RAM attuale del processo in MB
    def current_mb(self) -> float:
        try:
            return self._process.memory_info().rss / (1024 ** 2)
        except psutil.NoSuchProcess:
            return 0.0 #se il processo non esiste piu'

    #procedura del thread che aggiorna il picco finche _running è True
    def sample(self) -> None:
        while self._running:
            current = self.current_mb()
            if current > self._peak:
                self._peak = current
            time.sleep(self._interval)


#coordina cronometro, monitor RAM e codecarbon per singolo blocco
class _EcoTracker:
    #inizializza monitor RAM e tracker codecarbon
    def __init__(self, label: str, sample_interval: float) -> None:
        self.label = label #nome operazione misurata
        self.metrics: Optional[SustainabilityMetrics] = None #popolato con i risultati
        self._monitor = _RamMonitor(interval=sample_interval) #monitor del picco di RAM
        self._carbon_tracker = EmissionsTracker(
            measure_power_secs=sample_interval, #determina intervallo di campionamento
            log_level="error", #silenzia i log di codecarbon (errori)
            save_to_file=False, #niente file emissions.csv su disco
        )
        self._t0: float = 0.0 #istante di inizio (perf_counter)
        self._start_ram: float = 0.0 #RAM all'avvio in MB

    #avvia misurazioni RAM, codecarbon e cronometro
    def start(self) -> None:
        self._start_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        self._monitor.start()
        self._carbon_tracker.start()
        self._t0 = time.perf_counter()

    #ferma thread e popola self.metrics con i valori raccolti
    def stop(self) -> None:
        elapsed   = time.perf_counter() - self._t0 #tempo trascorso in secondi
        peak_ram  = self._monitor.stop()
        co2_kg    = self._carbon_tracker.stop()
        #energia consumata in kWh
        energy_kwh = (
            self._carbon_tracker.final_emissions_data.energy_consumed
            if self._carbon_tracker.final_emissions_data else 0.0
        )
        #incapsula i risultati in un SustainabilityMetrics
        self.metrics = SustainabilityMetrics(
            elapsed_seconds=elapsed,
            peak_ram_mb=peak_ram,
            start_ram_mb=self._start_ram,
            energy_kwh=energy_kwh,
            co2_kg=co2_kg if co2_kg is not None else 0.0,
            label=self.label,
        )


"""
-context manager da utilizzare per monitorare l'operazione
-sample_interval(s) regola campionamento RAM e la misura di potenza di codecarbon.
-di default sample_interval=0.5s per training lunghi per evitare overhead
-per operazioni brevi sample_interval=0.05 per non perdere il picco
"""
@contextlib.contextmanager
def measure_sustainability(label: str = "run", sample_interval: float = 0.5):
    tracker = _EcoTracker(label=label, sample_interval=sample_interval)
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()


"""
confronta due SustainabilityMetrics per calcolare la differenza di consumi
"""
def compare_sustainability(a: SustainabilityMetrics, b: SustainabilityMetrics) -> dict:
    return {
        "label_a": a.label,
        "label_b": b.label,
        "delta_elapsed_s":  round(b.elapsed_seconds - a.elapsed_seconds, 4),
        "delta_ram_mb":     round(b.peak_ram_mb     - a.peak_ram_mb,     4),
        "delta_energy_kwh": round(b.energy_kwh      - a.energy_kwh,      8),
        "delta_co2_kg":     round(b.co2_kg          - a.co2_kg,          8),
        "faster":  a.label if b.elapsed_seconds > a.elapsed_seconds else b.label,
        "greener": a.label if b.co2_kg          > a.co2_kg          else b.label,
    }


# ============================================================================
#  PARTE 2 - SCAMBIO DATI SU FILE (ex io_contratto)
# ============================================================================

#cartella di consegna condivisa: ANCORATA alla posizione di contratto.py (NON al CWD),
#cosi' chi usa le funzioni non deve dire dove salvare e nessuno sposta file a mano:
#salva_*() ci scrive da sola. Sta accanto al tool (es. MammoDiffusion/notebooks/consegne/),
#quindi e' dentro il progetto sincronizzato e arriva a tutti.
#Sovrascrivibile se il gruppo vuole un percorso diverso/condiviso:
#  import contratto; contratto.CARTELLA_CONSEGNE = "/percorso/condiviso/consegne"
CARTELLA_CONSEGNE = str(Path(__file__).resolve().parent / "consegne")


#costruisce il path dentro la cartella di consegna e crea le cartelle mancanti
#(se 'nome' e' un path assoluto, 'cartella' viene ignorata)
def _percorso_consegna(nome: str | Path, cartella: Optional[str | Path] = None) -> Path:
    base = Path(cartella if cartella is not None else CARTELLA_CONSEGNE)
    p = base / nome
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ===================== ECO (sostenibilità) =====================

#legge i record da .json (un dict o una lista) o .jsonl (un dict per riga)
def _leggi_record(path: str | Path) -> list[dict]:
    p = Path(path)
    testo = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonl":
        return [json.loads(riga) for riga in testo.splitlines() if riga.strip()]
    dati = json.loads(testo)
    return dati if isinstance(dati, list) else [dati]


#costruisce un SustainabilityMetrics dai 5 campi noti (ignora gli extra tipo timestamp/status)
def _eco_da_dict(d: dict) -> SustainabilityMetrics:
    return SustainabilityMetrics(
        elapsed_seconds=float(d.get("elapsed_seconds", 0.0)),
        peak_ram_mb=float(d.get("peak_ram_mb", 0.0)),
        start_ram_mb=float(d.get("start_ram_mb", 0.0)),
        energy_kwh=float(d.get("energy_kwh", 0.0)),
        co2_kg=float(d.get("co2_kg", 0.0)),
        label=str(d.get("label", "eco")),
    )


"""
carica record eco in una lista di SustainabilityMetrics
-paths      : UN file o una LISTA di file (.json/.jsonl). I record vengono concatenati
              (utile per fondere piu' file, es. finetuning + scelta del checkpoint)
-solo_status: se dato (es. "completed"), tiene solo i record con quel campo "status"
              (per escludere i run falliti dal conteggio)
"""
def carica_eco(paths, solo_status: Optional[str] = None) -> list[SustainabilityMetrics]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    raw: list[dict] = []
    for p in paths:
        raw.extend(_leggi_record(p))
    if solo_status is not None:
        raw = [d for d in raw if d.get("status") == solo_status]
    return [_eco_da_dict(d) for d in raw]


"""
aggrega piu' SustainabilityMetrics in UNO solo (quello che la pipeline si aspetta)
-tempo / energia / CO2 -> SOMMA (additivi: tutta l'energia spesa)
-RAM picco             -> MAX   (il picco non si somma)
"""
def aggrega_eco(records: list[SustainabilityMetrics], label: str = "aggregato") -> SustainabilityMetrics:
    if not records:
        return SustainabilityMetrics(label=label)
    return SustainabilityMetrics(
        elapsed_seconds=sum(m.elapsed_seconds for m in records),
        peak_ram_mb=max(m.peak_ram_mb for m in records),
        start_ram_mb=max(m.start_ram_mb for m in records),
        energy_kwh=sum(m.energy_kwh for m in records),
        co2_kg=sum(m.co2_kg for m in records),
        label=label,
    )


#comodita': carica uno o piu' file eco e aggrega tutto in un solo SustainabilityMetrics
def carica_e_aggrega_eco(paths, label: str = "aggregato",
                         solo_status: Optional[str] = None) -> SustainabilityMetrics:
    return aggrega_eco(carica_eco(paths, solo_status=solo_status), label=label)


#salva uno o piu' SustainabilityMetrics nella cartella di consegna
#(.jsonl = un record per riga, .json = lista/dict). Ritorna il path salvato.
def salva_eco(metrics, nome: str | Path, cartella: Optional[str | Path] = None) -> Path:
    p = _percorso_consegna(nome, cartella)
    lista = metrics if isinstance(metrics, (list, tuple)) else [metrics]
    record = [m.to_dict() for m in lista]
    if p.suffix.lower() == ".jsonl":
        p.write_text("\n".join(json.dumps(r) for r in record) + "\n", encoding="utf-8")
    else:
        payload = record if len(record) > 1 else record[0]
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p.resolve()


# ===================== PREDIZIONI (classificatore) =====================

"""
salva le predizioni del classificatore in .npz nella cartella di consegna
-y_true/y_pred : obbligatori
-y_prob        : opzionale (serve a ROC-AUC); per il binario matrice (N,2) col 1 = positiva
-class_names   : opzionale, es. ["Sano", "Malato"]
"""
def salva_predizioni(nome: str | Path, y_true, y_pred,
                     y_prob=None, class_names: Optional[list] = None,
                     cartella: Optional[str | Path] = None) -> Path:
    arrays = {"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred)}
    if y_prob is not None:
        arrays["y_prob"] = np.asarray(y_prob)
    if class_names is not None:
        arrays["class_names"] = np.asarray(list(class_names))
    p = _percorso_consegna(nome, cartella)
    np.savez(p, **arrays)
    #np.savez aggiunge .npz se manca: normalizziamo il path ritornato
    return (p if p.suffix == ".npz" else p.with_suffix(".npz")).resolve()


#ricarica le predizioni salvate con salva_predizioni; ritorna dict con y_true/y_pred/y_prob/class_names
def carica_predizioni(path: str | Path) -> dict:
    d = np.load(Path(path), allow_pickle=False)
    return {
        "y_true": d["y_true"],
        "y_pred": d["y_pred"],
        "y_prob": d["y_prob"] if "y_prob" in d.files else None,
        "class_names": [str(c) for c in d["class_names"]] if "class_names" in d.files else None,
    }


#elenca i file presenti nella cartella di consegna (comodo per Samuele a integrazione)
def elenca_consegne(cartella: Optional[str | Path] = None) -> list[Path]:
    base = Path(cartella if cartella is not None else CARTELLA_CONSEGNE)
    return sorted(p for p in base.iterdir() if p.is_file()) if base.is_dir() else []


# ===================== METRICHE (prestazioni: accuracy/F1/FID/IS...) =====================

#salva un dizionario di metriche (accuracy/F1/... oppure FID/IS) in JSON nella cartella di consegna
def salva_metriche(nome: str | Path, metriche: dict, cartella: Optional[str | Path] = None) -> Path:
    p = _percorso_consegna(nome, cartella)
    p.write_text(json.dumps(metriche, indent=2), encoding="utf-8")
    return p.resolve()


#ricarica un dizionario di metriche salvato con salva_metriche
def carica_metriche(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ===================== CONSEGNA IN UN COLPO (osserva + salva tutto) =====================

"""
[lato collega CLASSIFICATORE] UNA chiamata per una configurazione:
- VALUTA il classificatore e STAMPA le metriche (per osservarle),
- SALVA tutto in consegne/: predizioni (.npz), eco training+inferenza (.jsonl),
  metriche (.json) e confusion matrix (.png).
Nomi config: real_only / real_plus_synthetic / synthetic_only / trad_aug.
Ritorna l'EvaluationResult.
"""
def consegna_classificatore(nome_config: str, y_true, y_pred, y_prob, class_names: list,
                            eco_training, eco_inference, pos_label: int = 1,
                            cartella: Optional[str | Path] = None):
    from classifier_evaluator import ClassifierEvaluator  #import lazy: non serve a chi fa il generatore
    base = cartella if cartella is not None else CARTELLA_CONSEGNE
    res = ClassifierEvaluator(output_dir=base).evaluate_predictions(
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        label=nome_config, class_names=class_names, pos_label=pos_label)
    print(res)  #osserva le metriche a video
    salva_predizioni(f"pred_{nome_config}.npz", y_true, y_pred, y_prob, class_names=class_names, cartella=base)
    salva_eco(eco_training,  f"eco_training_{nome_config}.jsonl",  cartella=base)
    salva_eco(eco_inference, f"eco_inference_{nome_config}.jsonl", cartella=base)
    salva_metriche(f"metriche_{nome_config}.json", res.to_dict(), cartella=base)
    print(f"[consegnato in {Path(base)}] config '{nome_config}'")
    return res


"""
[lato collega GENERATORE] UNA chiamata: SALVA in consegne/ gli eco del modello
diffusivo (finetuning, scelta checkpoint opz., generazione) + (opz.) le metriche
FID/IS che hai calcolato per debug. Le immagini generate restano nella loro cartella:
il FID/IS UFFICIALE lo calcola Samuele da quelle.
'eco_finetuning' puo' essere un singolo SustainabilityMetrics o una LISTA (piu' run).
"""
def consegna_generatore(eco_finetuning, eco_generation, eco_checkpoint=None,
                        metriche_fid_is: Optional[dict] = None,
                        cartella: Optional[str | Path] = None):
    base = cartella if cartella is not None else CARTELLA_CONSEGNE
    salva_eco(eco_finetuning, "eco_finetuning.jsonl", cartella=base)
    if eco_checkpoint is not None:
        salva_eco(eco_checkpoint, "eco_checkpoint.jsonl", cartella=base)
    salva_eco(eco_generation, "eco_generation.jsonl", cartella=base)
    if metriche_fid_is is not None:
        salva_metriche("metriche_generatore.json", metriche_fid_is, cartella=base)
    print(f"[consegnato in {Path(base)}] eco generatore")
