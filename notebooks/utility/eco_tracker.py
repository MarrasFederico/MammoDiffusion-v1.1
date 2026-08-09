"""Measure runtime sustainability data for training, inference, and generation.

CodeCarbon's ``energy_kwh`` and ``co2_kg`` are unreliable on the RTX 5060 Ti
because it has no power model for that GPU. The analysis therefore uses only
``elapsed_seconds`` and estimates energy as hours x 0.170 kW. See
``docs/SUSTAINABILITY_ANALYSIS.md``.
"""
from __future__ import annotations
import contextlib, os, threading, time
from dataclasses import dataclass
from typing import Optional
import psutil
from codecarbon import EmissionsTracker

@dataclass
class SustainabilityMetrics:
    """Runtime, peak memory, energy, and emissions measured for one code block."""
    elapsed_seconds: float = 0.0
    peak_ram_mb: float = 0.0
    start_ram_mb: float = 0.0
    energy_kwh: float = 0.0
    co2_kg: float = 0.0
    label: str = "run"

    def __str__(self) -> str:
        return (
            f"[{self.label}] "
            f"Time: {self.elapsed_seconds:.2f}s | "
            f"Peak RAM: {self.peak_ram_mb:.1f} MB | "
            f"Energy: {self.energy_kwh:.6f} kWh | "
            f"CO2: {self.co2_kg:.6f} kg"
        )

    def to_dict(self) -> dict:
        """Return a serializable metrics record."""
        return {
            "label": self.label,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_ram_mb": self.peak_ram_mb,
            "energy_kwh": self.energy_kwh,
            "co2_kg": self.co2_kg,
        }


class _RamMonitor:
    """Sample process RAM to estimate its peak in MiB."""

    def __init__(self, interval: float = 0.05) -> None:
        self._interval = interval  # Seconds between RAM samples.
        self._process = psutil.Process(os.getpid())
        self._peak: float = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the sampling thread from the current RAM reading."""
        self._peak = self.current_mb()
        self._running = True
        self._thread = threading.Thread(target=self.sample, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        """Stop the sampling thread and return peak RAM in MiB."""
        self._running = False
        if self._thread:
            self._thread.join()
        return self._peak

    def current_mb(self) -> float:
        """Return the process's current resident memory in MiB."""
        try:
            return self._process.memory_info().rss / (1024 ** 2)
        except psutil.NoSuchProcess:
            return 0.0  # The process has already exited.

    def sample(self) -> None:
        """Update the peak reading while monitoring is active."""
        while self._running:
            current = self.current_mb()
            if current > self._peak:
                self._peak = current
            time.sleep(self._interval)


class _EcoTracker:
    """Coordinate a timer, RAM monitor, and CodeCarbon for one code block."""

    def __init__(self, label: str, sample_interval: float) -> None:
        self.label = label
        self.metrics: Optional[SustainabilityMetrics] = None
        self._monitor = _RamMonitor(interval=sample_interval)
        self._carbon_tracker = EmissionsTracker(
            measure_power_secs=sample_interval,
            log_level="error",
            save_to_file=False,
        )
        self._t0: float = 0.0
        self._start_ram: float = 0.0

    def start(self) -> None:
        """Start RAM, CodeCarbon, and elapsed-time measurements."""
        self._start_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        self._monitor.start()
        self._carbon_tracker.start()
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        """Stop all measurements and populate ``metrics``."""
        elapsed   = time.perf_counter() - self._t0
        peak_ram  = self._monitor.stop()
        co2_kg    = self._carbon_tracker.stop()
        energy_kwh = (
            self._carbon_tracker.final_emissions_data.energy_consumed
            if self._carbon_tracker.final_emissions_data else 0.0
        )
        self.metrics = SustainabilityMetrics(
            elapsed_seconds=elapsed,
            peak_ram_mb=peak_ram,
            start_ram_mb=self._start_ram,
            energy_kwh=energy_kwh,
            co2_kg=co2_kg if co2_kg is not None else 0.0,
            label=self.label,
        )


@contextlib.contextmanager
def measure_sustainability(label: str = "run", sample_interval: float = 0.5):
    """Measure one operation as a context manager.

    ``sample_interval`` controls both RAM sampling and CodeCarbon power
    measurement. The 0.5-second default limits overhead during long training;
    use about 0.05 seconds for short operations whose RAM peak could be missed.
    """
    tracker = _EcoTracker(label=label, sample_interval=sample_interval)
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()
