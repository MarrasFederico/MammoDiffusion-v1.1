from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import generator_benchmark as gb  # noqa: E402


def audit_counts(*, total: int, feature: int, quality: int, near_black: int = 0,
                 invalid_range: int = 0, corrupt: int = 0, wrong_shape: int = 0) -> dict:
    unique_feature, unique_quality = feature, quality
    return {
        "n_discovered": total,
        "n_readable": total - corrupt,
        "n_corrupt": corrupt,
        "n_wrong_shape": wrong_shape,
        "n_near_black": near_black,
        "n_invalid_range": invalid_range,
        "n_feature_extractable": feature,
        "n_feature_nonextractable": total - feature,
        "feature_extractable_rate": feature / total,
        "n_unique_feature_extractable_content": unique_feature,
        "n_exact_duplicates_among_feature_extractable": feature - unique_feature,
        "n_quality_valid": quality,
        "n_quality_invalid": total - quality,
        "quality_validity_rate": quality / total,
        "n_unique_quality_valid_content": unique_quality,
        "n_exact_duplicates_among_quality_valid": quality - unique_quality,
        "n_technically_valid": quality,
        "n_technically_invalid": total - quality,
        "n_unique_valid_content": unique_quality,
        "n_exact_duplicates_among_valid": quality - unique_quality,
        "technical_validity_rate": quality / total,
    }


def save_clean(path: Path, size: tuple[int, int] = (16, 16)) -> Path:
    pixels = np.full(size, 70, dtype=np.uint8)
    pixels[:4, :] = 130
    Image.fromarray(pixels).save(path)
    return path


class RepresentationAwareTechnicalPolicyTests(unittest.TestCase):
    def test_g07_raw_exact_counts_are_ready_with_quality_warning(self):
        fixture = audit_counts(total=4083, feature=4083, quality=3980,
                               near_black=103, invalid_range=1)
        with mock.patch.object(gb, "technical_audit", return_value=fixture):
            row = gb.technical_validity_row("07_ldm_sdvae_extra1361", "raw", [], minimum_unique=1361)
        self.assertTrue(row["eligible_for_distribution_metrics"])
        self.assertFalse(row["eligible_for_official_ranking"])
        self.assertTrue(row["quality_warning"])
        self.assertEqual(row["warning_reasons"], "near_black; invalid_range")
        self.assertEqual(row["fatal_failure_reasons"], "")
        self.assertEqual((row["n_feature_extractable"], row["n_quality_valid"]), (4083, 3980))

    def test_g07_filtered_exact_counts_are_official_ranking_ready(self):
        fixture = audit_counts(total=1361, feature=1361, quality=1361)
        with mock.patch.object(gb, "technical_audit", return_value=fixture):
            row = gb.technical_validity_row("07_ldm_sdvae_extra1361", "filtered", [], minimum_unique=1361)
        self.assertTrue(row["eligible_for_distribution_metrics"])
        self.assertTrue(row["eligible_for_official_ranking"])
        self.assertFalse(row["quality_warning"])

    def test_constant_black_is_extractable_raw_warning_but_filtered_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            black = Path(temporary) / "black.png"
            Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(black)
            raw = gb.technical_validity_row("g", "raw", [black], minimum_unique=1,
                                            expected_size=(16, 16))
            filtered = gb.technical_validity_row("g", "filtered", [black], minimum_unique=1,
                                                 expected_size=(16, 16))
        self.assertEqual((raw["n_feature_extractable"], raw["n_quality_valid"]), (1, 0))
        self.assertTrue(raw["eligible_for_distribution_metrics"])
        self.assertEqual(raw["warning_reasons"], "near_black; invalid_range")
        self.assertFalse(filtered["eligible_for_official_ranking"])
        self.assertIn("near_black", filtered["fatal_failure_reasons"])
        self.assertIn("invalid_range", filtered["fatal_failure_reasons"])

    def test_raw_fatal_does_not_block_clean_filtered_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corrupt = root / "corrupt.png"
            corrupt.write_text("not an image")
            clean = save_clean(root / "clean.png")
            raw = gb.technical_validity_row("g07", "raw", [corrupt], minimum_unique=1,
                                            expected_size=(16, 16))
            filtered = gb.technical_validity_row("g07", "filtered", [clean], minimum_unique=1,
                                                 expected_size=(16, 16))
        audit = {"generator_id": "g07", "scientific_family": "from_scratch",
                 "candidate_role": "primary_candidate", "eligible_for_benchmark_execution": True,
                 "eligible_for_descriptive_benchmark": True,
                 "eligible_for_official_family_ranking": True, "blockers": []}
        row = gb.representation_preflight_rows([audit], [raw, filtered])[0]
        self.assertFalse(row["raw_descriptive_ready"])
        self.assertTrue(row["filtered_descriptive_ready"])
        self.assertTrue(row["filtered_official_ranking_ready"])

    def test_family_coverage_stops_only_when_an_entire_filtered_family_is_missing(self):
        rows = [
            {"scientific_family": "finetuned", "filtered_official_ranking_ready": True},
            {"scientific_family": "from_scratch", "filtered_official_ranking_ready": True},
        ]
        self.assertEqual(gb.require_official_family_coverage(rows), {"finetuned": 1, "from_scratch": 1})
        with self.assertRaisesRegex(RuntimeError, "from_scratch"):
            gb.require_official_family_coverage(rows[:1])

    def test_raw_sampling_keeps_black_output_and_does_not_modify_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            black = root / "black.png"
            Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(black)
            clean = save_clean(root / "clean.png")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"samples": [black.name, clean.name]}, sort_keys=True))
            before = hashlib.sha256(manifest.read_bytes()).hexdigest()
            selected = gb.deterministic_sample([str(black), str(clean)], 2, 17)
            after = hashlib.sha256(manifest.read_bytes()).hexdigest()
        self.assertIn(str(black), selected)
        self.assertEqual(before, after)

    def test_nonfinite_features_report_sample_id_path_extractor_and_write_no_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [save_clean(root / "a.png"), save_clean(root / "b.png")]
            manifest = root / "manifest.json"
            manifest.write_text('{"version": 1}')
            cache = root / "features.npy"
            with self.assertRaises(gb.NonFiniteEmbeddingError) as caught:
                gb.get_or_extract_embeddings(
                    cache, paths, ["a", "b"], extractor="fixture", preprocessing="fixture",
                    code_version="fixture", source_manifest=str(manifest), feature_dimension=2,
                    extract_fn=lambda values, name: np.array([[1.0, 2.0], [np.nan, 0.0]]),
                )
        failure = caught.exception.failures[0]
        self.assertEqual((failure["sample_id"], failure["extractor"]), ("b", "fixture"))
        self.assertTrue(failure["path"].endswith("b.png"))
        self.assertEqual(failure["cause"], "non_finite_feature_values")
        self.assertFalse(cache.exists())

    def test_protocol_declares_strict_filtered_and_warning_raw_policy(self):
        protocol = gb.load_protocol(ROOT)
        policy = protocol["technical_validity_policy"]
        self.assertEqual(policy["raw"]["near_black"], "warning_and_include")
        self.assertEqual(policy["raw"]["constant_range"], "warning_and_include")
        self.assertEqual(policy["filtered"]["near_black"], "fatal_for_official_ranking")
        self.assertEqual(protocol["selection"]["official_representation"], "filtered")

    def test_notebook_reports_both_validity_categories_and_ranks_filtered_only(self):
        text = (ROOT / "notebooks/3_generator_benchmark/05_Unified_Generator_Benchmark.ipynb").read_text()
        for token in ("raw_feature_extractable_rate", "raw_quality_validity_rate",
                      "filtered_feature_extractable_rate", "filtered_quality_validity_rate",
                      "representation_preflight_rows", "require_official_family_coverage",
                      "feature_extraction_failures", "extractor_preflight"):
            self.assertIn(token, text)
        self.assertIn("row['condition'] == 'FILTERED' and row['eligible_for_official_ranking']", text)


if __name__ == "__main__":
    unittest.main()
