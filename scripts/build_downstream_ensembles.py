#!/usr/bin/env python3
"""Build the eight exact-seed mean-probability validation ensembles."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

from classifier_experiment_runner import build_ensemble_if_ready  # noqa: E402
from downstream_protocol import ARCHITECTURES, CONDITIONS  # noqa: E402


def main() -> None:
    results = [{"architecture": architecture, "condition": condition,
                **build_ensemble_if_ready(ROOT, architecture, condition)}
               for architecture in ARCHITECTURES for condition in CONDITIONS]
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
