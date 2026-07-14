#!/usr/bin/env python3
"""List the 24 manually runnable downstream jobs and their current state."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

from downstream_lifecycle import inventory  # noqa: E402


def main() -> None:
    for job in inventory(ROOT)["jobs"]:
        print(f"{job['experiment_id']}\t{job['architecture']}\t{job['condition']}\t{job['seed']}\t"
              f"{job['status']}\t{job['suggested_command']}")


if __name__ == "__main__":
    main()
