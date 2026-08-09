from __future__ import annotations
import sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import sustainability_registry as sr  # noqa: E402


def event(run_id, phase="classifier_training", status="completed", canonical=True, energy=1.0, co2=0.1,
          seconds=100.0, start="2026-01-01T00:00:00", end="2026-01-01T00:01:00", **extra):
    return {"run_id": run_id, "experiment_id": "exp1", "dataset_variant_id": "R", "architecture": "maxvit512",
            "seed": 17, "phase": phase, "status": status, "parent_run_id": None, "canonical": canonical,
            "reused_artifact": False, "start_time": start, "end_time": end, "elapsed_seconds": seconds,
            "energy_kwh": energy, "co2_kg": co2, "peak_ram_mb": 100, "peak_vram_mb": 1000, "gpu_uuid": "u",
            "gpu_name": "g", "num_images": 1000, "optimizer_updates": 50, "epochs": 1, "source_log": "x",
            "signature": {}, "value_precision": "measured", **extra}


class ValidationTests(unittest.TestCase):
    def test_invalid_phase_rejected(self):
        errors = sr.validate_event(event("r1", phase="not_a_real_phase"))
        self.assertTrue(any("phase" in e for e in errors))

    def test_invalid_status_rejected(self):
        errors = sr.validate_event(event("r1", status="not_a_real_status"))
        self.assertTrue(any("status" in e for e in errors))

    def test_nan_energy_rejected(self):
        errors = sr.validate_event(event("r1", energy=float("nan")))
        self.assertTrue(any("NaN" in e for e in errors))

    def test_missing_run_id_rejected(self):
        e = event("r1"); del e["run_id"]
        errors = sr.validate_event(e)
        self.assertTrue(any("run_id" in e for e in errors))

    def test_valid_event_has_no_errors(self):
        self.assertEqual(sr.validate_event(event("r1")), [])

    def test_append_event_rejects_invalid_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "events.jsonl"
            with self.assertRaises(ValueError):
                sr.append_event(path, event("r1", phase="bogus"))
            self.assertFalse(path.exists())

    def test_append_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "events.jsonl"
            sr.append_event(path, event("r1"))
            sr.append_event(path, event("r2"))
            loaded = sr.load_events(path)
            self.assertEqual(len(loaded), 2)


class DeduplicationTests(unittest.TestCase):
    def test_reused_artifact_events_excluded_from_canonical(self):
        events = [event("r1", status="reused")]
        self.assertEqual(sr.deduplicate_canonical_events(events), [])

    def test_noncanonical_events_excluded(self):
        events = [event("r1", canonical=False)]
        self.assertEqual(sr.deduplicate_canonical_events(events), [])

    def test_failed_events_excluded_from_canonical(self):
        events = [event("r1", status="failed")]
        self.assertEqual(sr.deduplicate_canonical_events(events), [])

    def test_duplicate_run_id_counted_once_keeping_latest(self):
        events = [event("r1", end="2026-01-01T00:01:00"), event("r1", end="2026-01-01T00:02:00")]
        canonical = sr.deduplicate_canonical_events(events)
        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0]["end_time"], "2026-01-01T00:02:00")

class RetiredAggregateApiTests(unittest.TestCase):
    def test_untrusted_codecarbon_aggregates_are_not_exposed(self):
        for name in (
            "actual_vs_canonical",
            "group_by_phase",
            "normalized_metrics",
            "sum_resumed_segments",
            "write_summary_by_experiment",
            "write_summary_by_run",
        ):
            self.assertFalse(hasattr(sr, name), name)


if __name__ == "__main__":
    unittest.main()
