#classi e procedure per misurare sostenibilità di qualsiasi blocco di codice (training, inferenza, generazione)
from __future__ import annotations
import contextlib, os, threading, time
from dataclasses import dataclass
from typing import Optional
import psutil
from codecarbon import EmissionsTracker

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