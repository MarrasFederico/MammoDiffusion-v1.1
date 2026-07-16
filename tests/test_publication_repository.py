from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "notebooks").rglob("*.ipynb"))


class PublicationRepositoryTests(unittest.TestCase):
    def test_notebooks_validate_and_code_cells_compile(self):
        transformer = TransformerManager()
        for relative in NOTEBOOKS:
            notebook = nbformat.read(ROOT / relative, as_version=4); nbformat.validate(notebook)
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    compile(transformer.transform_cell(cell.source), f"{relative}:cell-{index}", "exec")

    def test_main_publication_files_exist(self):
        required = (
            "notebooks/04_classifiers/07_MaxViT512.ipynb",
            "notebooks/04_classifiers/08_MammoFM.ipynb",
            "notebooks/04_classifiers/09_Validation_Comparison.ipynb",
            "notebooks/04_classifiers/10_Final_Evaluation_and_Report.ipynb",
            "notebooks/utility/classifier_experiment.py",
            "notebooks/utility/classifier_architecture_adapters.py",
        )
        self.assertTrue(all((ROOT / path).is_file() for path in required))

    def test_final_evaluation_guard_exists(self):
        final = (ROOT / "notebooks/04_classifiers/10_Final_Evaluation_and_Report.ipynb").read_text()
        self.assertIn("RUN_FINAL_EVALUATION = False", final)
        self.assertIn("PLANNED_COMPARISONS", final)

    def test_selection_is_a_transparent_post_benchmark_amendment(self):
        # Selection exists only after the benchmark + human-approved Option B amendment.
        import json
        selection = ROOT / "configs/selected_generators.json"
        self.assertTrue(selection.exists())
        payload = json.loads(selection.read_text())
        self.assertEqual(payload["finetuned"], "02_sd21_filtered_100steps")
        self.assertEqual(payload["from_scratch"], "07_ldm_sdvae_extra1361")
        self.assertTrue(payload["post_benchmark_amendment"])
        self.assertFalse(payload["test_access"])
        self.assertEqual(payload["active_amendment"], "configs/generator_benchmark_protocol_amendment_v1.json")

    def test_no_tracked_model_cache_or_archive_artifacts(self):
        if not (ROOT / ".git").exists():
            self.skipTest("Git metadata unavailable in source archive")
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
        forbidden = (".pt", ".pth", ".ckpt", ".keras", ".h5", ".safetensors", ".zip", ".tar.gz", ".tgz", ".pyc", ".pyo")
        self.assertEqual([path for path in tracked if path.endswith(forbidden)], [])
        self.assertFalse(any("__pycache__" in path for path in tracked))

if __name__ == "__main__": unittest.main()
