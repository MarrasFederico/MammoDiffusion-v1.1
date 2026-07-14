#!/usr/bin/env python3
"""Freeze the final protocol and test-manifest hash after validation is complete."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
from downstream_lifecycle import create_test_lock  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm: parser.error("locking the test requires --confirm")
    print(json.dumps(create_test_lock(ROOT), indent=2))
