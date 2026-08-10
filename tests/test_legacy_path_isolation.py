"""Legacy recorded paths must never reach another repository.

Frozen manifests and caches store absolute paths under a project root that no
longer exists at that location. On the original workstation the historical
prefix is now a symlink to an unrelated repository, so a consumer that follows
such a string verbatim reads someone else's files, and a consumer that compares
it literally declares a still-valid artifact incompatible.

These tests build that exact trap -- a sibling ``MammoDiffusion`` symlink
pointing at a decoy repository seeded with canary files -- and assert that every
shared path utility resolves into the checkout, or fails, and never touches the
decoy.
"""
from __future__ import annotations

import builtins
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import parallel_generation_utils as pgu  # noqa: E402
import project_paths  # noqa: E402
from classifier_dataset_builder import resolve_project_path as builder_resolve  # noqa: E402
from processed_dataset_reuse import _resolve_project_path as reuse_resolve  # noqa: E402

LEGACY_ROOT = "/mnt/MammoDiffusion/MammoDiffusion"


def build_trap(base: Path) -> tuple[Path, Path]:
    """A checkout, a decoy repository, and a bare-name symlink pointing at the decoy."""
    decoy = base / "MammoDiffusion-v2"
    for relative in ("results", "configs", "data/processed/metadata", "experiments"):
        (decoy / relative).mkdir(parents=True, exist_ok=True)
    (decoy / "DO_NOT_READ_FROM_V2").write_text("canary", encoding="utf-8")
    (decoy / "data/processed/metadata/val.csv").write_text("decoy", encoding="utf-8")

    checkout = base / "MammoDiffusion-v1.1"
    for relative in ("results", "configs", "data/processed/metadata", "experiments", "notebooks"):
        (checkout / relative).mkdir(parents=True, exist_ok=True)
    (checkout / ".git").mkdir(exist_ok=True)
    (checkout / "data/processed/metadata/val.csv").write_text("real", encoding="utf-8")

    link = base / "MammoDiffusion"
    if not link.exists():
        link.symlink_to(decoy)
    return checkout, decoy


class CanaryRecorder:
    """Fail the test if anything opens a file inside the decoy repository."""

    def __init__(self, decoy: Path):
        self.decoy = decoy.resolve()
        self.touched: list[str] = []
        self._open = builtins.open

    def __enter__(self):
        recorder = self

        def guarded(file, *args, **kwargs):
            try:
                resolved = Path(file).resolve()
            except (TypeError, ValueError, OSError):
                return recorder._open(file, *args, **kwargs)
            if resolved == recorder.decoy or recorder.decoy in resolved.parents:
                recorder.touched.append(str(resolved))
            return recorder._open(file, *args, **kwargs)

        builtins.open = guarded
        return self

    def __exit__(self, *exc):
        builtins.open = self._open
        return False


class ProjectPathRuleTests(unittest.TestCase):
    def test_a_legacy_prefix_is_rerooted_onto_this_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout, decoy = build_trap(Path(temporary))
            legacy = f"{LEGACY_ROOT}/data/processed/metadata/val.csv"
            resolved = project_paths.resolve_project_path(checkout, legacy)
            self.assertEqual(resolved, checkout / "data/processed/metadata/val.csv")
            self.assertNotIn("MammoDiffusion-v2", str(resolved))
            self.assertEqual(resolved.read_text(encoding="utf-8"), "real")

    def test_the_symlinked_decoy_is_never_the_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout, decoy = build_trap(Path(temporary))
            with CanaryRecorder(decoy) as canary:
                for suffix in ("results", "configs", "data/processed/metadata/val.csv",
                               "experiments/diffusers/x"):
                    for resolver in (project_paths.resolve_project_path,
                                     builder_resolve,
                                     reuse_resolve):
                        resolved = Path(resolver(checkout, f"{LEGACY_ROOT}/{suffix}"))
                        self.assertNotIn("MammoDiffusion-v2", str(resolved))
                        self.assertTrue(str(resolved).startswith(str(checkout)),
                                        f"{resolver.__name__} escaped: {resolved}")
            self.assertEqual(canary.touched, [], "a resolver read from the decoy repository")

    def test_an_unrecognised_absolute_path_is_refused_or_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout, _ = build_trap(Path(temporary))
            with self.assertRaises(ValueError):
                project_paths.resolve_project_path(checkout, "/elsewhere/opaque/file.png")
            with self.assertRaises(ValueError):
                builder_resolve(checkout, "/elsewhere/opaque/file.png")
            # The reuse audit reports rather than raises, so its caller can explain why.
            self.assertEqual(Path(reuse_resolve(checkout, "/elsewhere/opaque/file.png")),
                             Path("/elsewhere/opaque/file.png"))

    def test_traversal_cannot_climb_out_of_the_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout, _ = build_trap(Path(temporary))
            with self.assertRaises(ValueError):
                project_paths.resolve_project_path(checkout, "../MammoDiffusion-v2/results")

    def test_a_symlinked_data_directory_still_counts_as_inside(self):
        """data/ or experiments/ is often a symlink to a larger volume."""
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            volume = base / "big-volume" / "processed"
            (volume / "metadata").mkdir(parents=True)
            (volume / "metadata" / "val.csv").write_text("real", encoding="utf-8")
            checkout = base / "checkout"
            (checkout / "data").mkdir(parents=True)
            (checkout / "data" / "processed").symlink_to(volume)
            resolved = project_paths.resolve_project_path(
                checkout, "data/processed/metadata/val.csv")
            self.assertEqual(resolved.read_text(encoding="utf-8"), "real")
            self.assertTrue(str(resolved).startswith(str(checkout)))


class PathIdentityTests(unittest.TestCase):
    def test_paths_equivalent_ignores_the_recorded_root(self):
        self.assertTrue(project_paths.paths_equivalent(
            f"{LEGACY_ROOT}/experiments/diffusers/x",
            "/somewhere/else/MammoDiffusion-v1.1/experiments/diffusers/x"))
        self.assertFalse(project_paths.paths_equivalent(
            f"{LEGACY_ROOT}/experiments/diffusers/x",
            f"{LEGACY_ROOT}/experiments/diffusers/y"))

    def test_a_renamed_checkout_does_not_invalidate_a_frozen_signature(self):
        cached = {"path": f"{LEGACY_ROOT}/data/processed/metadata/val.csv",
                  "size": 80527, "sha256": "abc"}
        current = {"path": "/new/MammoDiffusion-v1.1/data/processed/metadata/val.csv",
                   "size": 80527, "sha256": "abc"}
        self.assertTrue(pgu.signature_matches(cached, current))
        self.assertTrue(pgu.records_equivalent(cached, current))

    def test_changed_content_still_invalidates(self):
        cached = {"path": "/a/data/x", "size": 1, "sha256": "abc"}
        self.assertFalse(pgu.records_equivalent(cached, {"path": "/b/data/x", "size": 2, "sha256": "abc"}))
        self.assertFalse(pgu.records_equivalent(cached, {"path": "/b/data/x", "size": 1, "sha256": "def"}))

    def test_a_record_carrying_only_a_path_never_matches(self):
        self.assertFalse(pgu.records_equivalent({"path": "/a/data/x"}, {"path": "/b/data/x"}))
        self.assertFalse(pgu.signature_matches({"path": "/a"}, {"path": "/b"}))

    def test_nested_signatures_are_compared_by_content(self):
        cached = {"config": {"base_model": {"path": f"{LEGACY_ROOT}/notebooks/pretrained_model",
                                            "files": [{"name": "a", "size": 1}]},
                             "steps": 100}}
        current = {"config": {"base_model": {"path": "/new/x/notebooks/pretrained_model",
                                             "files": [{"name": "a", "size": 1}]},
                              "steps": 100}}
        self.assertTrue(pgu.records_equivalent(cached, current))
        current["config"]["steps"] = 50
        self.assertFalse(pgu.records_equivalent(cached, current))


class FrozenCacheStillValidTests(unittest.TestCase):
    """The shipped caches must remain usable in this renamed checkout."""

    CACHE = ROOT / ("results/2_diffusers/02_sd21_filtered_100steps/metrics/"
                    "checkpoint_validation_cache_v2.json")

    @unittest.skipUnless(CACHE.is_file(), "benchmark cache is not part of a source-only clone")
    def test_the_shipped_checkpoint_cache_is_recognised_after_the_rename(self):
        payload = json.loads(self.CACHE.read_text(encoding="utf-8"))
        validation_csv = ROOT / "data/processed/metadata/val.csv"
        if not validation_csv.is_file():
            self.skipTest("the image cohort is not present in a source-only clone")
        self.assertTrue(
            pgu.sd_metrics_cache_compatible(payload, payload["config"], validation_csv),
            "the frozen cache is rejected in this checkout; a recorded path is being "
            "compared literally again",
        )
        recorded = payload["validation_csv_signature"]["path"]
        self.assertNotEqual(recorded, str(validation_csv.resolve()),
                            "this test is meaningless unless the recorded path really differs")


if __name__ == "__main__":
    unittest.main()
