"""The publication report must be amendment-aware and free of invalid efficiency durations."""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import final_evaluation as fe  # noqa: E402


def _build_report_fixture(tmp: Path) -> Path:
    (tmp / "configs").mkdir(parents=True)
    (tmp / "configs/generator_registry.json").write_text(json.dumps({"generators": [
        {"id": "02_sd21_filtered_100steps", "scientific_family": "finetuned",
         "eligible_for_downstream_selection": True},
        {"id": "07_ldm_sdvae_extra1361", "scientific_family": "from_scratch",
         "eligible_for_downstream_selection": True}]}))
    (tmp / "configs/generator_benchmark_protocol.json").write_text(json.dumps({"study_question": "RQ1: ..."}))
    (tmp / "configs/generator_benchmark_protocol_amendment_v1.json").write_text(json.dumps({
        "selected_policy": "B", "status": "approved_post_benchmark",
        "original_outcome": {"official_candidates_measured": 5, "eligible_under_original_gates": 0}}))
    (tmp / "configs/selected_generators.json").write_text(json.dumps({
        "finetuned": "02_sd21_filtered_100steps", "from_scratch": "07_ldm_sdvae_extra1361",
        "schema_version": 2, "primary_metric": "raddino_kid", "benchmark_run_id": "run_x",
        "active_amendment": "configs/generator_benchmark_protocol_amendment_v1.json",
        "post_benchmark_amendment": True, "test_access": False, "selection_notes": "Option B",
        "selection_identity": {
            "finetuned": {"generator_id": "02_sd21_filtered_100steps", "descriptive_family_rank": 1,
                          "primary_metric": "raddino_kid", "primary_metric_value": 0.199},
            "from_scratch": {"generator_id": "07_ldm_sdvae_extra1361", "descriptive_family_rank": 1,
                             "primary_metric": "raddino_kid", "primary_metric_value": 0.087}}}))
    summary = tmp / "results/publication_v2/generator_benchmark/generator_summary_corrected.csv"
    summary.parent.mkdir(parents=True)
    summary.write_text("generator_id,condition,generation_seconds_per_image,peak_vram_mb,energy_kwh,"
                       "checkpoint_size_bytes,efficiency_source,efficiency_status\n"
                       "02_sd21_filtered_100steps,FILTERED,,,,3463726504,"
                       "results/diffusers/02/metrics/generation_info.json,unavailable_invalid_duration_semantics\n")
    return tmp


class PublicationReportAmendmentTests(unittest.TestCase):
    def test_report_is_amendment_aware(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_report_fixture(Path(tmp))
            report = Path(fe.generate_publication_report(root)).read_text()
        # Original zero-eligible outcome and Option B are both present.
        self.assertIn("Eligible under original gates: 0", report)
        self.assertIn("Option B", report)
        # Selected generators and the post-benchmark amendment declaration.
        self.assertIn("02_sd21_filtered_100steps", report)
        self.assertIn("07_ldm_sdvae_extra1361", report)
        self.assertIn("Post-benchmark amendment: True", report)
        self.assertIn("Test access: False", report)
        # No microsecond-per-image durations shown as available.
        self.assertIn("unavailable_invalid_duration_semantics", report)
        self.assertNotIn("selection_basis", report)
        self.assertNotIn("manual_override", report)
        # No physically impossible microsecond-scale seconds-per-image leaks through.
        self.assertFalse(re.search(r"\b0\.0000\d+\b", report), "an invalid microsecond duration leaked into the report")

    def test_limitations_declare_the_amendment(self):
        self.assertIn("post-benchmark methodological amendment", fe.format_limitations())


if __name__ == "__main__":
    unittest.main()
