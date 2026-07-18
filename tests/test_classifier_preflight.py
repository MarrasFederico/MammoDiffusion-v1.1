"""Metadata-only downstream preflight audit: integrity checks with fixtures."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import classifier_preflight as pre  # noqa: E402


def _write_metadata(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "image_id", "label"])
        writer.writerows(rows)


def _write_manifest(path: Path, paths: list[str], *, duplicate_id: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "relative_path", "source_raw_sample_id"])
        for index, rel in enumerate(paths):
            sample_id = "dup" if duplicate_id else f"s{index}"
            writer.writerow([sample_id, rel, f"src{index}"])


class RealPatientAuditTests(unittest.TestCase):
    def test_detects_no_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_metadata(root / pre.TRAIN_METADATA, [("p1", "i1", "1"), ("p2", "i2", "0")])
            _write_metadata(root / pre.VAL_METADATA, [("p3", "i3", "1")])
            audit = pre.real_patient_audit(root)
        self.assertEqual(audit["train_validation_patient_overlap"], 0)
        self.assertEqual(audit["train_positive_images"], 1)
        self.assertEqual(audit["validation_positive_images"], 1)

    def test_detects_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_metadata(root / pre.TRAIN_METADATA, [("shared", "i1", "1")])
            _write_metadata(root / pre.VAL_METADATA, [("shared", "i2", "1")])
            audit = pre.real_patient_audit(root)
        self.assertEqual(audit["train_validation_patient_overlap"], 1)


class SyntheticManifestAuditTests(unittest.TestCase):
    def _root(self, tmp: Path, paths: list[str], *, duplicate_id: bool = False,
              create_files: bool = True) -> Path:
        root = Path(tmp)
        manifest_rel = "results/2_diffusers/provenance/runtime/G/filtered_samples.csv"
        _write_manifest(root / manifest_rel, paths, duplicate_id=duplicate_id)
        (root / "configs/prov").mkdir(parents=True, exist_ok=True)
        (root / "configs/prov/G.json").write_text(json.dumps({"filtered_sample_manifest": manifest_rel}))
        (root / "configs/generator_registry.json").write_text(json.dumps({"generators": [
            {"id": "G", "provenance_manifest": "configs/prov/G.json"}]}))
        if create_files:
            for rel in paths:
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"x")
        return root

    def test_clean_manifest(self):
        paths = [f"data/synthetic/g/positive/pos_{i}.png" for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            audit = pre.synthetic_manifest_audit(self._root(tmp, paths), "G")
        self.assertEqual(audit["synthetic_sample_count"], 5)
        self.assertEqual(audit["class_counts"], {"positive": 5, "negative": 0})
        self.assertEqual(audit["test_paths_found"], 0)
        self.assertEqual(audit["duplicate_sample_ids"], 0)
        self.assertEqual(audit["missing_files"], 0)

    def test_test_path_is_flagged(self):
        paths = ["data/synthetic/g/positive/pos_0.png", "data/processed/test/1/leak.png"]
        with tempfile.TemporaryDirectory() as tmp:
            audit = pre.synthetic_manifest_audit(self._root(tmp, paths), "G")
        self.assertEqual(audit["test_paths_found"], 1)

    def test_duplicate_ids_and_missing_files(self):
        paths = [f"data/synthetic/g/positive/pos_{i}.png" for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            audit = pre.synthetic_manifest_audit(self._root(tmp, paths, duplicate_id=True,
                                                            create_files=False), "G")
        self.assertGreater(audit["duplicate_sample_ids"], 0)
        self.assertEqual(audit["missing_files"], 3)


if __name__ == "__main__":
    unittest.main()
