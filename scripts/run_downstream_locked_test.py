#!/usr/bin/env python3
"""Perform the one-shot locked inference after verifying every frozen artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
from downstream_lifecycle import run_locked_test  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm: parser.error("locked inference requires --confirm")
    print(json.dumps(run_locked_test(ROOT), indent=2))
