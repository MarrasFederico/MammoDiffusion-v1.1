from __future__ import annotations

import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import downstream_experiment as de  # noqa: E402
import generator_benchmark as gb  # noqa: E402


def image(path: Path, value: int, size=(16, 16)) -> Path:
    pixels = np.full(size, value, dtype=np.uint8)
    pixels[:2, :] = min(255, value + 20)
    Image.fromarray(pixels).save(path)
    return path


class ContentAwareCacheTests(unittest.TestCase):
    def test_cache_reuse_and_every_relevant_invalidation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [image(root / "a.png", 30), image(root / "b.png", 90)]
            manifest = root / "manifest.json"; manifest.write_text('{"version":1}')
            metadata_csv = root / "reference.csv"; metadata_csv.write_text("image_id,label\na,1\n")
            cache = root / "features.npy"; calls = []

            def extract(values, extractor):
                calls.append((tuple(map(str, values)), extractor))
                return np.arange(6, dtype=float).reshape(2, 3) + len(calls)

            kwargs = dict(extractor="rad_dino", preprocessing="resize-v1", code_version="cache-v2",
                          source_manifest=str(manifest), extractor_model_id="microsoft/rad-dino",
                          extractor_weights_identifier="checkpoint-A", feature_dimension=3,
                          metadata_csv=metadata_csv, extract_fn=extract)
            gb.get_or_extract_embeddings(cache, paths, ["a", "b"], **kwargs)
            gb.get_or_extract_embeddings(cache, paths, ["a", "b"], **kwargs)
            self.assertEqual(len(calls), 1)
            image(paths[0], 50)
            gb.get_or_extract_embeddings(cache, paths, ["a", "b"], **kwargs)
            metadata_csv.write_text("image_id,label\na,1\nb,1\n")
            gb.get_or_extract_embeddings(cache, paths, ["a", "b"], **kwargs)
            manifest.write_text('{"version":2}')
            gb.get_or_extract_embeddings(cache, paths, ["a", "b"], **kwargs)
            gb.get_or_extract_embeddings(cache, paths, ["a", "b"], **{**kwargs, "preprocessing": "resize-v2"})
            gb.get_or_extract_embeddings(cache, paths, ["a", "b"], **{**kwargs, "extractor_model_id": "other-model"})
            self.assertEqual(len(calls), 6)
            _, ordered = gb.get_or_extract_embeddings(cache, list(reversed(paths)), ["b", "a"], **kwargs)
            self.assertEqual(ordered["image_ids"], ["b", "a"])
            self.assertEqual(len(calls), 7)


class TechnicalValidityTests(unittest.TestCase):
    def test_invalid_readable_images_are_not_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = image(root / "valid.png", 60)
            wrong = image(root / "wrong.png", 100, (8, 8))
            result = gb.technical_audit([valid, wrong], expected_size=(16, 16))
            self.assertEqual(result["n_exact_duplicates_among_valid"], 0)
            self.assertEqual((result["n_technically_valid"], result["n_technically_invalid"]), (1, 1))

    def test_duplicates_only_among_valid_and_corrupt_is_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = image(root / "one.png", 80); second = root / "two.png"; second.write_bytes(first.read_bytes())
            corrupt = root / "corrupt.png"; corrupt.write_text("not an image")
            result = gb.technical_audit([first, second, corrupt], expected_size=(16, 16))
            self.assertEqual(result["n_exact_duplicates_among_valid"], 1)
            self.assertEqual(result["n_corrupt"], 1)

    def test_near_black_does_not_create_valid_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            black = root / "black.png"; pixels = np.zeros((16, 16), dtype=np.uint8); pixels[0, 0] = 10; Image.fromarray(pixels).save(black)
            paths = [image(root / "a.png", 60), image(root / "b.png", 120), black]
            result = gb.technical_audit(paths, expected_size=(16, 16))
            self.assertEqual(result["n_exact_duplicates_among_valid"], 0)


class ProvenanceAndFilterTests(unittest.TestCase):
    def setUp(self):
        self.protocol = copy.deepcopy(gb.load_protocol(ROOT)); self.protocol["synthetic_pool_target"] = 1

    def _fixture(self, root: Path):
        raw = root / "raw"; filtered = root / "filtered"; raw.mkdir(); filtered.mkdir()
        raw_paths = [image(raw / "r1.png", 40), image(raw / "r2.png", 90)]
        filtered_paths = [filtered / "f1.png"]; filtered_paths[0].write_bytes(raw_paths[0].read_bytes())
        train = root / "train.csv"; train.write_text("image_id,label\nt1,0\n")
        checkpoint = root / "model.pt"; checkpoint.write_bytes(b"weights")
        filter_manifest = root / "filter.csv"
        with filter_manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["raw_path", "selected", "filtered_path"]); writer.writeheader()
            writer.writerow({"raw_path": raw_paths[0], "selected": "true", "filtered_path": filtered_paths[0]})
            writer.writerow({"raw_path": raw_paths[1], "selected": "false", "filtered_path": ""})
        manifest = root / "provenance.json"
        records = lambda values: [{"path": str(path), "sha256": gb.file_sha256(path)} for path in values]
        payload = {"generator_id": "g1", "checkpoint": "model.pt", "training_dataset": "fixture",
                   "training_corpus_manifest": "train.csv",
                   "samples": {"raw": records(raw_paths), "filtered": records(filtered_paths)}}
        manifest.write_text(json.dumps(payload))
        entry = {"id": "g1", "scientific_family": "finetuned", "model_family": "fixture", "model_variant": "v1",
                 "candidate_role": "primary_candidate", "eligible_for_downstream_selection": True,
                 "checkpoint": "model.pt", "provenance_manifest": "provenance.json",
                 "raw_generation_manifest": "provenance.json", "training_corpus_manifest": "train.csv",
                 "filtering_applied": True, "filter_manifest": "filter.csv",
                 "samples": {"raw_positive": "raw", "filtered_positive": "filtered"}}
        return entry, payload, manifest, filter_manifest, filtered_paths

    def test_filter_acceptance_is_independent_of_technical_validity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); entry, _, _, filter_manifest, filtered_paths = self._fixture(root)
            filtering = gb.filter_acceptance_from_manifest(filter_manifest, filtered_paths)
            technical = gb.technical_audit(filtered_paths, expected_size=(16, 16))
            self.assertEqual(filtering["filter_acceptance_rate"], 0.5)
            self.assertEqual(technical["technical_validity_rate"], 1.0)
            self.assertNotEqual(filtering["filter_acceptance_rate"], technical["technical_validity_rate"])
            self.assertTrue(gb.audit_candidate(root, entry, self.protocol)["eligible_for_benchmark_execution"])

    def test_missing_wrong_and_mismatched_provenance_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); entry, payload, manifest, filter_manifest, _ = self._fixture(root)
            manifest.unlink()
            self.assertFalse(gb.audit_candidate(root, entry, self.protocol)["eligible_for_benchmark_execution"])
            manifest.write_text(json.dumps({**payload, "generator_id": "another"}))
            self.assertIn("wrong_generator_id", gb.audit_candidate(root, entry, self.protocol)["provenance_failure_reason"])
            manifest.write_text(json.dumps({**payload, "checkpoint": "wrong.pt"}))
            self.assertIn("wrong_checkpoint", gb.audit_candidate(root, entry, self.protocol)["provenance_failure_reason"])
            manifest.write_text(json.dumps({**payload, "samples": {"raw": payload["samples"]["raw"], "filtered": ["other.png"]}}))
            self.assertIn("sample_set_mismatch", gb.audit_candidate(root, entry, self.protocol)["provenance_failure_reason"])
            manifest.write_text(json.dumps(payload)); filter_manifest.unlink()
            audit = gb.audit_candidate(root, entry, self.protocol)
            self.assertFalse(audit["filter_manifest_valid"]); self.assertFalse(audit["eligible_for_benchmark_execution"])

    def test_descriptive_g05_with_autonomous_provenance_stays_selection_ineligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); entry, _, _, _, _ = self._fixture(root)
            entry["id"] = "05_fixture"; entry["candidate_role"] = "descriptive_baseline"
            entry["eligible_for_downstream_selection"] = False
            payload = json.loads((root / "provenance.json").read_text()); payload["generator_id"] = "05_fixture"
            (root / "provenance.json").write_text(json.dumps(payload))
            audit = gb.audit_candidate(root, entry, self.protocol)
            self.assertTrue(audit["lineage_complete"])
            self.assertTrue(audit["eligible_for_descriptive_benchmark"])
            self.assertFalse(audit["eligible_for_official_family_ranking"])


class StatisticsMemorizationAccountingTests(unittest.TestCase):
    def test_real_subset_varies_shared_plan_and_paired_differences(self):
        protocol = copy.deepcopy(gb.load_protocol(ROOT)); protocol["synthetic_pool_target"] = 20
        protocol["resampling"].update(stability_repetitions=5, nearest_neighbour_k=3)
        size = gb.evaluation_subset_size(30, 73, 20, .8)
        plan = gb.balanced_subsample_indices(73, 30, size, 5, 17, nearest_neighbour_k=3)
        self.assertLess(size, 73); self.assertGreater(len({tuple(row["real_indices"]) for row in plan}), 1)
        rng = np.random.default_rng(1); real = rng.normal(size=(73, 4)); left = rng.normal(size=(30, 4)); right = rng.normal(size=(30, 4))
        left_rows, _ = gb.repeated_distribution_metrics(real, left, protocol, resampling_plan=plan)
        right_rows, _ = gb.repeated_distribution_metrics(real, right, protocol, resampling_plan=plan)
        paired = gb.paired_kid_differences(left_rows, right_rows, "a", "b")
        self.assertEqual(paired, gb.paired_kid_differences(left_rows, right_rows, "a", "b"))
        equivalence = gb.practical_equivalence(paired, protocol)
        self.assertEqual(equivalence["practical_equivalence_margin"], protocol["selection"]["practical_equivalence_margin"])
        self.assertEqual(paired["interval_type"], "repeated-subsampling stability interval")

    def test_entire_training_corpus_finds_negative_exact_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            negative = image(root / "negative.png", 70); positive = image(root / "positive.png", 150)
            synthetic = root / "synthetic.png"; synthetic.write_bytes(negative.read_bytes())
            rows = gb.build_train_memorization_rows(np.array([[10., 10.]]), np.array([[0., 0.], [10., 10.]]),
                ["s"], ["neg", "pos"], {"s": synthetic}, {"neg": negative, "pos": positive},
                {"ssim_gte": .98, "perceptual_hash_distance_lte": 2},
                {"neg": 0, "pos": 1}, {"neg": "real_negative", "pos": "real_positive"})
            self.assertTrue(rows[0]["exact_hash_match"]); self.assertTrue(rows[0]["memorization_flag"])
            self.assertEqual(rows[0]["exact_match_train_id"], "neg")
            self.assertEqual(rows[0]["exact_match_train_label"], 0)
            self.assertEqual(rows[0]["exact_match_train_source"], "real_negative")
            self.assertEqual(rows[0]["exact_match_train_ids"], ["neg"])
            self.assertEqual(rows[0]["nearest_train_label"], 1)

    def test_actual_accounting_stop_and_resume_and_estimate_labels(self):
        first = de.source_accounting([{"label": 0, "source": "real"}, {"label": 1, "source": "finetuned_synthetic"}])
        self.assertEqual((first["real_negative_seen"], first["finetuned_synthetic_seen"]), (1, 1))
        resumed = de.source_accounting([{"label": 1, "source": "traditional_augmentation"}], first)
        self.assertEqual(resumed["total_samples_seen"], 3); self.assertEqual(resumed["accounting_mode"], "actual")
        estimate = de.proportional_source_accounting({"source_distribution": {"real": 2}}, 1, 2)
        self.assertEqual(estimate["accounting_mode"], "proportional_estimate")
        self.assertFalse(any(key.endswith("_seen") for key in estimate))

    def test_publication_adapter_does_not_write_legacy_namespace(self):
        adapter = (ROOT / "notebooks/utility/classifier_architecture_adapters.py").read_text()
        downstream = (ROOT / "notebooks/utility/downstream_experiment.py").read_text()
        self.assertNotIn("results/downstream_classifiers", adapter + downstream)
        self.assertIn("results/publication_v2/downstream", downstream)


if __name__ == "__main__":
    unittest.main()
