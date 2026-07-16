"""Classifier synthetic sets must come from the signed FILTERED manifest, never a directory scan."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import classifier_dataset_builder as builder  # noqa: E402
import classifier_architecture_adapters as adapters  # noqa: E402

POOL = "data/synthetic/g/positive"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_tree(tmp: Path, *, count: int = 6, extra_directory_file: bool = False,
               test_path_row: bool = False, duplicate: str | None = None,
               bad_size: bool = False, bad_sha: bool = False, missing_file: bool = False) -> tuple[Path, dict]:
    root = Path(tmp)
    pool = root / POOL
    pool.mkdir(parents=True)
    rows = []
    for index in range(count):
        data = f"image-{index}".encode() * (index + 1)
        rel = f"{POOL}/pos_{index:04d}.png"
        (root / rel).write_bytes(data)
        rows.append({"sample_id": f"g::filtered::{rel}", "relative_path": rel,
                     "file_size": len(data), "sha256": _sha(data), "selection_rank": index,
                     "source_raw_sample_id": f"g::raw::src_{index}.png"})
    if extra_directory_file:
        (root / f"{POOL}/pos_extra.png").write_bytes(b"extra-not-in-manifest")
    if test_path_row:
        rows[0]["relative_path"] = "data/processed/test/1/leak.png"
        rows[0]["sample_id"] = "g::filtered::data/processed/test/1/leak.png"
    if duplicate == "sample_id":
        rows[1]["sample_id"] = rows[0]["sample_id"]
    if duplicate == "relative_path":
        rows[1]["relative_path"] = rows[0]["relative_path"]
    if duplicate == "rank":
        rows[1]["selection_rank"] = rows[0]["selection_rank"]
    if bad_size:
        rows[0]["file_size"] = rows[0]["file_size"] + 1
    if bad_sha:
        rows[0]["sha256"] = "0" * 64
    if missing_file:
        (root / rows[0]["relative_path"]).unlink()

    manifest_rel = "results/publication_v2/generator_provenance/runtime/G/filtered_samples.csv"
    manifest_path = root / manifest_rel
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "relative_path", "file_size", "sha256",
                                                    "selection_rank", "source_raw_sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    manifest_sha = _sha(manifest_path.read_bytes())

    (root / "configs").mkdir(exist_ok=True)
    (root / "configs/prov").mkdir(parents=True, exist_ok=True)
    (root / "configs/prov/G.json").write_text(json.dumps({
        "model_identity_sha256": "m", "generation_identity_sha256": "gen",
        "filtered_sample_manifest": manifest_rel, "manifest_sha256": {"filtered_samples": manifest_sha}}))
    (root / "configs/generator_registry.json").write_text(json.dumps({"generators": [
        {"id": "G", "scientific_family": "finetuned", "eligible_for_downstream_selection": True,
         "provenance_manifest": "configs/prov/G.json"},
        {"id": "OTHER", "scientific_family": "from_scratch", "eligible_for_downstream_selection": True,
         "provenance_manifest": "configs/prov/G.json"}]}))
    (root / "configs/selected_generators.json").write_text(json.dumps({
        "finetuned": "G", "from_scratch": "OTHER", "schema_version": 2,
        "selection_identity": {"finetuned": {
            "generator_id": "G", "descriptive_family_rank": 1, "filtered_manifest_path": manifest_rel,
            "filtered_manifest_sha256": manifest_sha, "filtered_image_count": count}}}))
    variant = {"dataset_variant_id": "real_plus_best_finetuned_positive", "status": "ready",
               "real_source": False, "augmentation_source": False,
               "synthetic_generator_id": "G", "synthetic_count_by_class": {"positive": count}, "seed": 42}
    return root, variant


class SignedManifestConsumptionTests(unittest.TestCase):
    def test_directory_extra_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, variant = build_tree(Path(tmp), count=6, extra_directory_file=True)
            file_list = builder.build_file_list(root, variant)
            payload = json.loads((root / "configs/selected_generators.json").read_text())
            records = builder.load_selected_filtered_records(root, payload, "finetuned")
            audit = builder.audit_built_synthetic_set(root, file_list, records)
        synthetic = [e for e in file_list["positive"] if e.get("source") == "synthetic"]
        self.assertEqual(len(synthetic), 6)
        self.assertEqual(audit["extra_files"], 0)
        self.assertTrue(audit["exact_path_set_match"])
        self.assertTrue(audit["exact_sha256_set_match"])
        self.assertNotIn(f"{POOL}/pos_extra.png", {e["path"] for e in synthetic})

    def test_exact_set_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, variant = build_tree(Path(tmp), count=5)
            file_list = builder.build_file_list(root, variant)
            payload = json.loads((root / "configs/selected_generators.json").read_text())
            records = builder.load_selected_filtered_records(root, payload, "finetuned")
        built = [e for e in file_list["positive"] if e.get("source") == "synthetic"]
        self.assertEqual({e["path"] for e in built}, {r["relative_path"] for r in records})
        self.assertEqual({e["file_sha256"] for e in built}, {r["sha256"] for r in records})
        self.assertEqual({e["sample_id"] for e in built}, {r["sample_id"] for r in records})

    def test_build_flatten_and_adapter_accounting_preserve_signed_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, variant = build_tree(Path(tmp), count=5)
            file_list = builder.build_file_list(root, variant)
            rows = builder.rows_from_file_list(root, file_list)
            metadata = adapters._accounting_metadata(rows)

        synthetic = [row for row in rows if row["source"] == "synthetic"]
        self.assertEqual(len(synthetic), 5)
        for row in synthetic:
            self.assertEqual(row["synthetic_family"], "finetuned")
            self.assertEqual(row["generator_id"], "G")
            for field in ("sample_id", "source_raw_sample_id", "manifest_sha256",
                          "file_sha256", "selection_rank"):
                self.assertIn(field, row)
        self.assertEqual(
            {item["accounting_field"] for item in metadata},
            {"finetuned_synthetic_seen"},
        )

    def test_synthetic_without_valid_family_is_rejected_by_adapter(self):
        with self.assertRaisesRegex(ValueError, "invalid synthetic_family"):
            adapters._accounting_metadata([{"source": "synthetic", "label": 1}])

    def test_no_sampling_functions_are_called_for_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, variant = build_tree(Path(tmp), count=4)
            with mock.patch.object(builder, "deterministic_sample_signature",
                                   side_effect=AssertionError("must not sample")), \
                 mock.patch.object(builder, "_synthetic_candidate_files",
                                   side_effect=AssertionError("must not scan")):
                file_list = builder.build_file_list(root, variant)
        self.assertEqual(sum(1 for e in file_list["positive"] if e.get("source") == "synthetic"), 4)

    def _expect_rejected(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            root, variant = build_tree(Path(tmp), count=6, **kwargs)
            with self.assertRaises((ValueError, FileNotFoundError)):
                builder.build_file_list(root, variant)

    def test_modified_or_wrong_sha_rejected(self):
        self._expect_rejected(bad_sha=True)

    def test_wrong_size_rejected(self):
        self._expect_rejected(bad_size=True)

    def test_missing_file_rejected(self):
        self._expect_rejected(missing_file=True)

    def test_test_path_rejected(self):
        self._expect_rejected(test_path_row=True)

    def test_duplicate_sample_id_rejected(self):
        self._expect_rejected(duplicate="sample_id")

    def test_duplicate_relative_path_rejected(self):
        self._expect_rejected(duplicate="relative_path")

    def test_noncontiguous_rank_rejected(self):
        self._expect_rejected(duplicate="rank")

    def test_manifest_file_bytes_changed_rejected(self):
        # Same manifest path, different bytes -> manifest SHA-256 no longer matches the selection record.
        with tempfile.TemporaryDirectory() as tmp:
            root, variant = build_tree(Path(tmp), count=6)
            manifest = root / "results/publication_v2/generator_provenance/runtime/G/filtered_samples.csv"
            with manifest.open("a") as stream:
                stream.write("g::filtered::x,data/synthetic/g/positive/x.png,1,ff,999,g::raw::x\n")
            payload = json.loads((root / "configs/selected_generators.json").read_text())
            with self.assertRaises(ValueError):
                builder.load_selected_filtered_records(root, payload, "finetuned")

    def test_count_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, variant = build_tree(Path(tmp), count=6)
            variant["synthetic_count_by_class"] = {"positive": 5}
            with self.assertRaises(ValueError):
                builder.build_file_list(root, variant)

    def test_manifest_path_traversal_outside_root_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = build_tree(Path(tmp), count=6)
            payload = json.loads((root / "configs/selected_generators.json").read_text())
            # A '..' traversal that resolves outside the repository must be rejected.
            payload["selection_identity"]["finetuned"]["filtered_manifest_path"] = "../outside/filtered_samples.csv"
            with self.assertRaisesRegex(ValueError, "escapes the project root"):
                builder.load_selected_filtered_records(root, payload, "finetuned")


class LegacyAndRealIndependenceTests(unittest.TestCase):
    def test_csv_string_labels_build_real_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "data/processed/metadata/train.csv"
            metadata.parent.mkdir(parents=True)
            with metadata.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["patient_id", "image_id", "label", "processed_path"])
                writer.writeheader()
                for label in ("0", "1"):
                    path = root / f"data/processed/train/{label}/i{label}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(label.encode())
                    writer.writerow({"patient_id": f"p{label}", "image_id": f"i{label}",
                                     "label": label, "processed_path": path.relative_to(root)})
            variant = {"dataset_variant_id": "real_only", "status": "ready", "real_source": True,
                       "augmentation_source": False, "synthetic_generator_id": None,
                       "synthetic_count_by_class": {}}
            rows = builder.rows_from_file_list(root, builder.build_file_list(root, variant))
        self.assertEqual(sorted(row["label"] for row in rows), [0, 1])

    def test_no_selection_file_means_legacy_family_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            self.assertIsNone(builder.selected_family_for_generator(root, "G"))

    def test_real_only_variant_needs_no_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            variant = {"dataset_variant_id": "real_only", "status": "ready", "real_source": True,
                       "augmentation_source": False, "synthetic_generator_id": None,
                       "synthetic_count_by_class": {}}
            # No metadata/train.csv here, but the synthetic branch must be skipped entirely.
            with self.assertRaises(FileNotFoundError):
                builder.build_file_list(root, variant)  # fails only on missing real metadata, not selection


if __name__ == "__main__":
    unittest.main()
