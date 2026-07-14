#!/usr/bin/env python3
"""Finalize patient-level validation and the eight preregistered comparisons."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
from downstream_lifecycle import finalize_validation  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(finalize_validation(ROOT), indent=2))
