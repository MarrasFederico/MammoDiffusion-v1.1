#!/usr/bin/env python3
"""Read-only, deterministic classifier Stage-1 preflight. Never opens the locked test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
from classifier_preflight import build_preflight, format_preflight  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--shallow", action="store_true", help="skip train/validation file and signature checks")
    args = parser.parse_args()
    report = build_preflight(Path(args.project_root), deep_dataset_check=not args.shallow)
    print(json.dumps(report, indent=1, sort_keys=True) if args.json else format_preflight(report))
    if report["readiness"] == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
