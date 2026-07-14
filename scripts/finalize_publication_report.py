#!/usr/bin/env python3
"""Create the publication report from finalized validation and locked-test artifacts."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
from downstream_lifecycle import finalize_publication_report  # noqa: E402

if __name__ == "__main__":
    print(finalize_publication_report(ROOT))
