"""Static guarantees that notebook 05 only ever extracts through the configured
local encoders, and that the frozen extractor exposes the reuse/close contract.

These checks parse the notebook as text and introspect the module; they never
load a model, so they run in any environment.
"""
from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import generator_benchmark as gb  # noqa: E402

NOTEBOOK = ROOT / "notebooks/3_generator_benchmark/05_Unified_Generator_Benchmark.ipynb"


def _code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text())
    return ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"]


class NotebookLocalEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cells = _code_cells()
        self.code = "\n".join(self.cells)

    def test_all_real_extractions_go_through_the_centralized_local_function(self):
        # After centralisation the notebook never calls extract_features() directly;
        # every extraction and every cache-miss extract_fn is extract_local_features.
        for cell in self.cells:
            self.assertNotIn("extract_features(", cell,
                             "operational extract_features() call must route through extract_local_features")
        self.assertIn("def extract_local_features(paths, feature_space):", self.code)
        self.assertIn("first = extract_local_features(preflight_paths, extractor_name)", self.code)
        for occurrence in self.code.split("extract_fn=")[1:]:
            self.assertTrue(occurrence.startswith("extract_local_features"),
                            "every extract_fn must be the centralized extract_local_features")

    def test_inception_uses_inception_path_and_rad_dino_uses_rad_path(self):
        self.assertIn("inception_path, rad_path = Path(INCEPTION_CHECKPOINT_PATH), Path(RAD_DINO_SNAPSHOT_PATH)",
                      self.code)
        self.assertIn("LOCAL_ENCODER_PATHS = {'inception_v3': inception_path, 'rad_dino': rad_path}", self.code)
        self.assertIn("FrozenLocalFeatureExtractor(feature_space, LOCAL_ENCODER_PATHS[feature_space]", self.code)

    def test_downloads_stay_disabled_during_the_benchmark(self):
        self.assertNotIn("allow_model_download=True", self.code)

    def test_training_corpus_uses_shared_content_addressed_cache(self):
        self.assertIn("training_manifest_sha256 = file_sha256(ROOT / training_manifest)", self.code)
        self.assertIn("embedding_cache_root / 'shared_training_corpora' / training_manifest_sha256 / 'rad_dino.npy'",
                      self.code)
        # The stale per-generator '_training_corpus' cache directory must be gone.
        self.assertNotIn("'_training_corpus'", self.code)
        self.assertIn("close_local_encoders()", self.code)


class FrozenLocalFeatureExtractorContractTests(unittest.TestCase):
    def test_class_is_exported_with_the_reuse_and_close_contract(self):
        self.assertIn("FrozenLocalFeatureExtractor", gb.__all__)
        cls = gb.FrozenLocalFeatureExtractor
        init = inspect.signature(cls.__init__)
        for name in ("feature_space", "local_model_path", "device", "batch_size"):
            self.assertIn(name, init.parameters)
        self.assertEqual(init.parameters["batch_size"].default, 16)
        for method in ("extract", "close", "__enter__", "__exit__"):
            self.assertTrue(callable(getattr(cls, method)))

    def test_extractor_never_downloads_and_stays_frozen(self):
        source = inspect.getsource(gb.FrozenLocalFeatureExtractor)
        self.assertIn("local_files_only=True", source)          # RAD-DINO stays offline
        self.assertIn("Inception_V3_Weights.IMAGENET1K_V1", source)  # official 1000-class enum
        self.assertIn("weights=None", source)                   # no implicit torchvision download
        self.assertIn("model.eval()", source)
        self.assertIn("requires_grad_(False)", source)
        self.assertIn("torch.inference_mode()", source)
        self.assertIn("cuda.empty_cache()", source)


if __name__ == "__main__":
    unittest.main()
