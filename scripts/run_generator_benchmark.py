#!/usr/bin/env python3
"""Audit or explicitly execute the validation-only unified generator benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

from generator_benchmark import build_execution_plan, run_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="read-only registry, lineage and sample-count audit")
    mode.add_argument("--execute", action="store_true", help="run real feature extraction and metrics")
    parser.add_argument("--confirm", action="store_true", help="required for --execute")
    parser.add_argument("--device")
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(build_execution_plan(ROOT), indent=2))
        return
    if not args.confirm:
        parser.error("--execute requires --confirm")
    result = run_benchmark(ROOT, allow_model_download=args.allow_model_download, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
