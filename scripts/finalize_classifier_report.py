#!/usr/bin/env python3
"""Build the v2 patient-level locked metrics, comparisons, tables, figures and final report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
from classifier_final_report import build_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    report = build_report(Path(args.project_root), n_bootstrap=args.bootstrap)
    print(json.dumps({"final_aggregation_complete": report["final_aggregation_complete"],
                      "report_signature": report["signature"]}, indent=1))


if __name__ == "__main__":
    main()
