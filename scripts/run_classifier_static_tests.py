#!/usr/bin/env python3
"""Run the classifier fixture/mock/static suite, explicitly excluding real integration tiers."""
from __future__ import annotations

import os
import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

MODULES = (
    "tests.test_artifact_phase_planner",
    "tests.test_classifier_adapters_and_notebooks",
    "tests.test_classifier_blocker_fixes_18",
    "tests.test_classifier_blocking_fixes",
    "tests.test_classifier_experiment_matrix",
    "tests.test_classifier_final_patch_19",
    "tests.test_classifier_gpu_scheduler",
    "tests.test_classifier_pipeline_completion",
    "tests.test_classifier_runner_resume",
    "tests.test_classifier_seed_ensemble",
    "tests.test_dataset_variant_registry",
    "tests.test_final_classifier_evaluation",
    "tests.test_final_matrix_lock",
    "tests.test_final_matrix_statistics",
    "tests.test_validation_matrix_selection",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="emit dots plus the final summary")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    result = unittest.TextTestRunner(verbosity=1 if args.quiet else 2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
