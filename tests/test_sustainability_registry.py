from __future__ import annotations
import json, sys, tempfile, unittest
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

    def test_resumed_segments_of_same_run_id_sum_without_double_counting(self):
        events = [
            event("train1", status="completed", canonical=True, energy=1.0, seconds=100, start="t0", end="t1"),
        ]
        result = sr.sum_resumed_segments(events, "train1")
        self.assertEqual(result["n_segments"], 1)
        self.assertEqual(result["energy_kwh"], 1.0)


class ActualVsCanonicalTests(unittest.TestCase):
    def test_actual_includes_failures_canonical_does_not(self):
        events = [
            event("r1", status="completed", energy=1.0),
            event("r2", status="failed", energy=0.5),
        ]
        result = sr.actual_vs_canonical(events)
        self.assertAlmostEqual(result["actual_project_energy_kwh"], 1.5)
        self.assertAlmostEqual(result["canonical_pipeline_energy_kwh"], 1.0)
        self.assertGreater(result["retry_and_failure_overhead_kwh"], 0)

    def test_canonical_never_exceeds_actual(self):
        events = [event("r1", status="completed", energy=1.0), event("r2", status="failed", energy=5.0)]
        result = sr.actual_vs_canonical(events)
        self.assertLessEqual(result["canonical_pipeline_energy_kwh"], result["actual_project_energy_kwh"])

    def test_exact_duplicate_log_lines_not_double_counted_in_actual(self):
        e = event("r1", status="completed", energy=1.0)
        events = [e, dict(e)]  # literal duplicate log entry
        result = sr.actual_vs_canonical(events)
        self.assertAlmostEqual(result["actual_project_energy_kwh"], 1.0)


class PhaseGroupingTests(unittest.TestCase):
    def test_group_by_phase_separates_training_from_generation(self):
        events = [event("r1", phase="classifier_training", energy=2.0), event("r2", phase="generation", energy=3.0)]
        grouped = sr.group_by_phase(events)
        self.assertAlmostEqual(grouped["classifier_training"]["energy_kwh"], 2.0)
        self.assertAlmostEqual(grouped["generation"]["energy_kwh"], 3.0)


class NormalizedMetricsTests(unittest.TestCase):
    def test_kwh_per_1000_images(self):
        result = sr.normalized_metrics(event("r1", energy=2.0, num_images=1000))
        self.assertAlmostEqual(result["kwh_per_1000_images"], 2.0)

    def test_zero_images_yields_none_not_division_error(self):
        result = sr.normalized_metrics(event("r1", num_images=0))
        self.assertIsNone(result["kwh_per_1000_images"])

    def test_kwh_per_optimizer_update(self):
        result = sr.normalized_metrics(event("r1", energy=1.0, optimizer_updates=100))
        self.assertAlmostEqual(result["kwh_per_optimizer_update"], 0.01)


class CsvOutputTests(unittest.TestCase):
    def test_write_summary_by_run_creates_csv_with_canonical_rows_only(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            events = [event("r1", status="completed"), event("r2", status="failed")]
            out = sr.write_summary_by_run(root, events)
            content = out.read_text()
            self.assertIn("r1", content)
            self.assertNotIn("r2", content)

    def test_write_summary_by_experiment_aggregates_across_runs(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            events = [event("r1", energy=1.0), event("r2", energy=2.0)]
            out = sr.write_summary_by_experiment(root, events)
            content = out.read_text()
            self.assertIn("exp1", content)
            self.assertIn("2", content)  # n_runs=2 somewhere in the row


if __name__ == "__main__":
    unittest.main()
