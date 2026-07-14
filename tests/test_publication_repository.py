from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "notebooks/3_generator_benchmark/05_Unified_Generator_Benchmark.ipynb",
    "notebooks/3_generator_benchmark/06_Generator_Selection.ipynb",
    "notebooks/4_downstream_classifiers/07_MaxViT512_Downstream.ipynb",
    "notebooks/4_downstream_classifiers/08_MammoFM_Downstream.ipynb",
    "notebooks/4_downstream_classifiers/09_Downstream_Validation_Comparison.ipynb",
    "notebooks/4_downstream_classifiers/10_Final_Evaluation_and_Report.ipynb",
]


class PublicationRepositoryTests(unittest.TestCase):
    def test_notebooks_validate_and_code_cells_compile(self):
        for relative in NOTEBOOKS:
            notebook = nbformat.read(ROOT / relative, as_version=4); nbformat.validate(notebook)
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code": compile(cell.source, f"{relative}:cell-{index}", "exec")

    def test_no_subprocess_and_direct_utility_imports(self):
        for relative in NOTEBOOKS:
            text = (ROOT / relative).read_text()
            self.assertNotIn("subprocess", text, relative)
            self.assertIn("notebooks.utility", text, relative)

    def test_required_sections_exist(self):
        expected = {
            NOTEBOOKS[0]: ["Protocol configuration", "Candidate discovery", "Candidate eligibility", "RAW/FILTERED image counts",
                           "Real reference set", "Technical validity", "Feature extraction", "Distribution metrics", "Diversity analysis",
                           "Duplicate analysis", "Train memorization analysis", "Validation similarity analysis", "Bootstrap/repeated subsampling",
                           "Results tables", "Pareto analysis", "Visual panels", "Family-specific conclusions"],
            NOTEBOOKS[2]: ["Experiment configuration", "Environment and GPU", "Dataset audit", "Training", "Validation inference", "Interpretability", "Saved artifacts"],
            NOTEBOOKS[3]: ["Experiment configuration", "Environment and GPU", "Dataset audit", "Training", "Validation inference", "Interpretability", "Saved artifacts"],
        }
        for relative, sections in expected.items():
            sources = "\n".join(cell.source for cell in nbformat.read(ROOT / relative, 4).cells)
            for section in sections: self.assertIn(section, sources)

    def test_configuration_cells_and_final_guard_exist(self):
        for relative in NOTEBOOKS[2:4]:
            text = (ROOT / relative).read_text()
            for token in ('CONDITION =', 'SEED =', 'GPU =', 'RESUME ='):
                self.assertIn(token, text)
        final = (ROOT / NOTEBOOKS[-1]).read_text()
        self.assertIn("RUN_FINAL_EVALUATION = False", final)
        self.assertIn("PLANNED_COMPARISONS", final)

    def test_removed_wrappers_and_old_locked_notebook_are_absent(self):
        removed = ("run_generator_benchmark.py", "approve_generator_selection.py", "run_downstream_classifier.py",
                   "list_downstream_jobs.py", "status_downstream_classifiers.py", "build_downstream_ensembles.py",
                   "finalize_downstream_validation.py", "lock_downstream_test.py", "run_downstream_locked_test.py",
                   "finalize_publication_report.py")
        for name in removed: self.assertFalse((ROOT / "scripts" / name).exists(), name)
        self.assertFalse((ROOT / "notebooks/4_downstream_classifiers/10_Locked_Test_and_Final_Report.ipynb").exists())

    def test_required_research_questions_are_documented(self):
        for relative in ("README.md", "docs/publication_experimental_design.md", "docs/experimental_protocol.md"):
            text = (ROOT / relative).read_text()
            for question in ("RQ1", "RQ2", "RQ3"): self.assertIn(question, text)

    def test_no_selection_is_hardcoded_before_benchmark(self):
        self.assertFalse((ROOT / "configs/selected_generators.json").exists())

    def test_no_tracked_model_cache_or_archive_artifacts(self):
        if not (ROOT / ".git").exists():
            self.skipTest("Git metadata unavailable in source archive")
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
        forbidden = (".pt", ".pth", ".ckpt", ".keras", ".h5", ".safetensors", ".zip", ".tar.gz", ".tgz", ".pyc", ".pyo")
        self.assertEqual([path for path in tracked if path.endswith(forbidden)], [])
        self.assertFalse(any("__pycache__" in path for path in tracked))

    def test_logo_and_final_dataset_status(self):
        self.assertTrue((ROOT / "assets/logo_MammoDiffusion.png").is_file())
        status = (ROOT / "docs/final_evaluation_dataset_status.md").read_text()
        self.assertIn("historically reused internal evaluation set", status)
        self.assertIn("not an independent external confirmation", status)

    def test_gitignore_has_required_hygiene(self):
        text = (ROOT / ".gitignore").read_text()
        for token in ("__pycache__/", "*.pyc", "*.zip", "*.tar.gz", ".coverage", "htmlcov/", ".pytest_cache/",
                      ".mypy_cache/", ".ruff_cache/", ".ipynb_checkpoints/", "embedding_cache"):
            self.assertIn(token, text)


if __name__ == "__main__": unittest.main()
