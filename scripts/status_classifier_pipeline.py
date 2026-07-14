#!/usr/bin/env python3
"""Read-only global classifier-matrix v2 status in human or machine-readable JSON form."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
from classifier_pipeline_status import build_status, format_status  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(); report = build_status(Path(args.project_root))
    print(json.dumps(report, ensure_ascii=False, indent=1) if args.json else format_status(report))


if __name__ == "__main__":
    main()
