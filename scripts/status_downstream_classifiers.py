#!/usr/bin/env python3
"""Report compact downstream job, ensemble, approval, and lock status."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

from downstream_lifecycle import inventory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = inventory(ROOT)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"Jobs: {report['job_counts']}")
    print(f"Ensembles: {report['ensemble_counts']}")
    print(f"Approved generators: {report['approved_generators']}")
    print(f"Locked test: {report['locked_test_status']}")


if __name__ == "__main__":
    main()
