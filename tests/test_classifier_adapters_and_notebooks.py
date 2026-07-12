from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
from classifier_architecture_adapters import TinyAdapter, _torch_payload, get_adapter  # noqa: E402


class AdapterContractTests(unittest.TestCase):
    def test_registry_resolves_all_four_architectures_without_framework_imports(self):
        for architecture in ("resnet50", "maxvit512", "mammofm", "raddino"):
            adapter = get_adapter(architecture, {"positive_class": 1}, ROOT, tiny=True)
            self.assertIsInstance(adapter, TinyAdapter)
            self.assertEqual(adapter.policy["positive_class"], 1)

    def test_tiny_checkpoint_roundtrip_prediction_shape_and_locked_test_refusal(self):
        with tempfile.TemporaryDirectory() as t:
            adapter = TinyAdapter("mammofm", {"positive_class": 1}, Path(t))
            path = Path(t) / "model.pt"
            rows = [{"label": 0, "image_id": "a"}, {"label": 1, "image_id": "b"}]
            adapter.train(rows, rows, path, seed=17)
            ok, reason = adapter.validate_checkpoint_compatibility(path)
            self.assertTrue(ok, reason)
            prediction = adapter.predict_validation(path, rows)
            self.assertEqual(prediction["labels"], [0, 1])
            self.assertEqual(len(prediction["probabilities"]), 2)
            with self.assertRaises(PermissionError): adapter.predict_locked_test([])

    def test_torch_checkpoint_wrappers_and_uniform_module_prefix(self):
        self.assertEqual(_torch_payload({"state_dict": {"module.a": 1}}), {"a": 1})
        self.assertEqual(_torch_payload({"model_state_dict": {"a": 1}}), {"a": 1})
        self.assertEqual(_torch_payload({"model": {"a": 1}}), {"a": 1})
        self.assertEqual(_torch_payload({"a": 1}), {"a": 1})
        with self.assertRaises(ValueError): _torch_payload({"module.a": 1, "b": 2})


class NotebookGeneratorTests(unittest.TestCase):
    def test_notebooks_are_standalone_and_have_required_reporting_sections(self):
        required = [f"## {i} —" for i in range(4, 18)]
        for path in (ROOT / "notebooks/3_classifiers_matrix").glob("*/*.ipynb"):
            text = path.read_text()
            for heading in required:
                self.assertIn(heading, text, f"{path}: {heading}")
            self.assertIn("Grad-CAM / Gradient-based attribution", text)
            self.assertIn("RESUME = True", text)
            self.assertNotIn("CodeCarbon", text)

    def test_real_only_notebook_has_no_synthetic_attribution_section(self):
        text = (ROOT / "notebooks/3_classifiers_matrix/resnet50/R50_R.ipynb").read_text()
        self.assertIn("sintetici: **False**", text)

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("matrix_notebooks", ROOT / "scripts/create_classifier_matrix_notebooks.py")
        cls.generator = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.generator)

    def test_exactly_112_dedicated_stage1_notebooks_and_no_orphans(self):
        paths = sorted((ROOT / "notebooks/3_classifiers_matrix").glob("*/*.ipynb"))
        self.assertEqual(len(paths), 112)
        inventory = json.loads((ROOT / "results/notebook_inventory/notebook_inventory.json").read_text())
        stage1 = [row for row in inventory if int(row["stage"]) == 1]
        self.assertEqual(len(stage1), 112)
        self.assertEqual({row["path"] for row in stage1}, {str(path.relative_to(ROOT)) for path in paths})

    def test_notebook_cells_compile_and_ids_are_deterministic(self):
        for path in (ROOT / "notebooks/3_classifiers_matrix").glob("*/*.ipynb"):
            payload = json.loads(path.read_text())
            self.assertEqual(len({cell["id"] for cell in payload["cells"]}), len(payload["cells"]))
            for cell in payload["cells"]:
                if cell["cell_type"] == "code": compile("".join(cell["source"]), str(path), "exec")
            source = path.read_text()
            rebuilt = self.generator.notebook(
                payload["metadata"]["mammodiffusion"]["architecture"],
                next(v for v in json.loads((ROOT / "configs/dataset_variant_registry.json").read_text())["variants"]
                     if v["dataset_variant_id"] == payload["metadata"]["mammodiffusion"]["dataset_variant_id"]),
                "BLOCKED" if 'DATASET_STATUS = \'BLOCKED\'' in source else "READY",
                next((row["note_blocker"] for row in json.loads((ROOT / "results/notebook_inventory/notebook_inventory.json").read_text())
                      if row["path"] == str(path.relative_to(ROOT))), None),
            )
            self.assertEqual(payload, rebuilt)


if __name__ == "__main__": unittest.main()
