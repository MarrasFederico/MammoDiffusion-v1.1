from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "utility"))

from processed_dataset_reuse import audit_processed_dataset  # noqa: E402


class ProcessedDatasetReuseAuditTests(unittest.TestCase):
    def _complete_cohort(self, root: Path) -> tuple[Path, pd.DataFrame]:
        processed_dir = root / "data" / "processed"
        metadata_dir = processed_dir / "metadata"
        metadata_dir.mkdir(parents=True)
        rows = []
        for split_index, split in enumerate(("train", "val", "test")):
            for label in (0, 1):
                patient_id = f"p{split_index}{label}"
                image_id = f"i{split_index}{label}"
                relative_path = Path("data") / "processed" / split / str(label) / f"{image_id}.png"
                image_path = root / relative_path
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((8, 8), 32 + 96 * label, dtype=np.uint8)).save(image_path)
                rows.append({
                    "patient_id": patient_id,
                    "image_id": image_id,
                    "laterality": "L",
                    "view": "MLO",
                    "label": label,
                    "cancer": label,
                    "patient_label": label,
                    "split": split,
                    "source": "real",
                    "processed_path": relative_path.as_posix(),
                    "visual_side_before": "left",
                    "visual_side_after": "left",
                    "left_ratio_before": 0.9,
                    "right_ratio_before": 0.1,
                    "flipped_by_visual_rule": False,
                    "normalized_tissue_side": "left",
                    "normalized_laterality": "L",
                })
        manifest = pd.DataFrame(rows)
        manifest.to_csv(metadata_dir / "all_processed.csv", index=False)
        for split in ("train", "val", "test"):
            manifest[manifest["split"] == split].to_csv(
                metadata_dir / f"{split}.csv", index=False
            )
        return processed_dir, manifest

    def test_complete_partitioned_cohort_is_accepted_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed_dir, manifest = self._complete_cohort(root)
            before = {
                path.relative_to(root): (path.stat().st_size, path.stat().st_mtime_ns)
                for path in root.rglob("*")
                if path.is_file()
            }
            audit = audit_processed_dataset(root, processed_dir)
            after = {
                path.relative_to(root): (path.stat().st_size, path.stat().st_mtime_ns)
                for path in root.rglob("*")
                if path.is_file()
            }

        self.assertTrue(audit["ready"], audit["reasons"])
        self.assertEqual(audit["row_count"], len(manifest))
        self.assertEqual(audit["missing_image_count"], 0)
        self.assertEqual(audit["split_counts"], {"train": 2, "val": 2, "test": 2})
        self.assertEqual(before, after)

    def test_missing_image_rejects_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed_dir, manifest = self._complete_cohort(root)
            missing = root / manifest.iloc[0]["processed_path"]
            missing.unlink()
            audit = audit_processed_dataset(root, processed_dir)

        self.assertFalse(audit["ready"])
        self.assertEqual(audit["missing_image_count"], 1)
        self.assertTrue(any("processed images are missing" in reason for reason in audit["reasons"]))

    def test_split_manifest_mismatch_rejects_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed_dir, _ = self._complete_cohort(root)
            train_path = processed_dir / "metadata" / "train.csv"
            pd.read_csv(train_path).iloc[:1].to_csv(train_path, index=False)
            audit = audit_processed_dataset(root, processed_dir)

        self.assertFalse(audit["ready"])
        self.assertTrue(any("does not reconcile" in reason for reason in audit["reasons"]))


class PreprocessingNotebookReuseWiringTests(unittest.TestCase):
    def test_notebook_has_audited_reuse_and_preserves_raw_mode(self):
        path = ROOT / "notebooks/1_preprocessing/01_Preprocessing_RSNA_512_gray_MLO.ipynb"
        notebook = nbformat.read(path, as_version=4)
        code = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        markdown = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "markdown"
        )

        for token in (
            "audit_processed_dataset",
            "ALLOW_COMPLETE_PROCESSED_REUSE = True",
            "REUSE_EXISTING_PROCESSED_DATA",
            "and not RAW_DATASET_READY",
            "Verified manifests remain unchanged during read-only reuse.",
            "Preserved published train/validation/test assignments.",
            '"data_source_mode"',
        ):
            self.assertIn(token, code)
        self.assertIn("elif FORCE_REDOWNLOAD_DATASET or not RAW_DATASET_READY", code)
        self.assertIn("local-reuse mode", markdown)
        self.assertIn("read-only", markdown)


if __name__ == "__main__":
    unittest.main()
