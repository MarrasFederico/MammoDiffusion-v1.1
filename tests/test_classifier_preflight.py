"""Metadata-only classifier preflight audit: integrity checks with fixtures."""
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


class SyntheticPoolAuditTests(unittest.TestCase):
    def _root(self, tmp: Path, pool_files: list[str], *, extra_files: list[str] | None = None) -> Path:
        root = Path(tmp)
        pool_dir = "data/synthetic/G/positive"
        (root / "configs").mkdir(parents=True, exist_ok=True)
        (root / "configs/generator_registry.json").write_text(json.dumps({"generators": [
            {"id": "G", "scientific_family": "finetuned",
             "samples": {"filtered_positive": pool_dir}}]}))
        for rel in pool_files:
            target = root / pool_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
        for rel in extra_files or []:
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
        return root

    def test_clean_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = pre.synthetic_pool_audit(self._root(tmp, [f"pos_{i}.png" for i in range(5)]), "G")
        self.assertEqual(audit["synthetic_sample_count"], 5)
        self.assertEqual(audit["class_counts"], {"positive": 5, "negative": 0})
        self.assertEqual(audit["test_paths_found"], 0)
        self.assertEqual(audit["duplicate_relative_paths"], 0)
        self.assertEqual(audit["unreadable_files"], 0)

    def test_missing_pool_reports_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = pre.synthetic_pool_audit(self._root(tmp, []), "G")
        self.assertEqual(audit["synthetic_sample_count"], 0)


if __name__ == "__main__":
    unittest.main()
