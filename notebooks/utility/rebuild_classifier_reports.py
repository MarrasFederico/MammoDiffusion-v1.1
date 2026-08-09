"""Rebuild every derived classifier report from frozen prediction CSV files.

The order is intentional: validation ensembles and their frozen thresholds are
rebuilt first, followed by held-out-test reports. The command never opens an
image, checkpoint, model, or GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .classifier_analysis import (
        build_all_validation_ensembles,
        compare_validation,
        rebuild_all_validation_seed_reports,
    )
    from .regenerate_classifier_metrics import regenerate
except ImportError:
    from classifier_analysis import (
        build_all_validation_ensembles,
        compare_validation,
        rebuild_all_validation_seed_reports,
    )
    from regenerate_classifier_metrics import regenerate


def rebuild(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    validation_seed_reports = rebuild_all_validation_seed_reports(root)
    validation_ensembles = build_all_validation_ensembles(root, split="validation")
    validation = compare_validation(
        root, validation_ensembles, split="validation"
    )
    test = regenerate(root)
    return {
        **test,
        "test_source": test["source"],
        "source": "saved validation_predictions.csv and test_predictions.csv only",
        "validation_seed_reports_written": validation_seed_reports,
        "validation_ensemble_reports_written": len(validation_ensembles),
        "validation_comparisons_written": len(validation["comparisons"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    print(json.dumps(rebuild(parse_args().root), indent=2))


if __name__ == "__main__":
    main()
