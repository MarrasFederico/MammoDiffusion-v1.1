from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


class PublicationRepositoryTests(unittest.TestCase):
    def test_final_notebooks_exist_and_validate(self):
        paths = [
            "notebooks/3_generator_benchmark/05_Unified_Generator_Benchmark.ipynb",
            "notebooks/3_generator_benchmark/06_Generator_Selection.ipynb",
            "notebooks/4_downstream_classifiers/07_MaxViT512_Downstream.ipynb",
            "notebooks/4_downstream_classifiers/08_MammoFM_Downstream.ipynb",
            "notebooks/4_downstream_classifiers/09_Downstream_Validation_Comparison.ipynb",
            "notebooks/4_downstream_classifiers/10_Locked_Test_and_Final_Report.ipynb",
        ]
        for relative in paths:
            nbformat.validate(nbformat.read(ROOT / relative, as_version=4))

    def test_obsolete_matrix_notebooks_are_retired(self):
        self.assertFalse((ROOT / "notebooks/3_classifiers_matrix").exists())
        self.assertFalse((ROOT / "notebooks/3_classifiers").exists())

    def test_required_research_questions_are_documented(self):
        for relative in ("README.md", "docs/publication_experimental_design.md", "docs/experimental_protocol.md"):
            text = (ROOT / relative).read_text()
            for question in ("RQ1", "RQ2", "RQ3"):
                self.assertIn(question, text)

    def test_no_approved_generator_is_hardcoded_before_benchmark(self):
        self.assertFalse((ROOT / "configs/approved_generators.json").exists())

    def test_no_tracked_model_or_archive_artifacts(self):
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
        forbidden = (".pt", ".pth", ".ckpt", ".keras", ".h5", ".safetensors", ".zip", ".tar.gz", ".tgz", ".pyc", ".pyo")
        self.assertEqual([path for path in tracked if path.endswith(forbidden)], [])

    def test_mammofm_license_note_and_gitignore(self):
        note = (ROOT / "docs/mammo_fm_license_note.md").read_text()
        ignore = (ROOT / ".gitignore").read_text()
        self.assertIn("Custom Academic License", note)
        self.assertIn("must not contain Mammo-FM checkpoints", note)
        self.assertIn("*Mammo-FM*Trained*.tar", ignore)


if __name__ == "__main__":
    unittest.main()
