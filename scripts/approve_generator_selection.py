#!/usr/bin/env python3
"""Explicitly approve a signed generator-selection proposal for downstream validation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

from downstream_protocol import approval_payload, atomic_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        parser.error("approval requires --confirm")
    proposal_path = (ROOT / args.proposal).resolve() if not Path(args.proposal).is_absolute() else Path(args.proposal)
    proposal = json.loads(proposal_path.read_text())
    payload = approval_payload(ROOT, proposal)
    output = atomic_json(ROOT / "configs/approved_generators.json", payload)
    print(json.dumps({"status": "approved", "output": str(output),
                      "approval_signature": payload["signature"]}, indent=2))


if __name__ == "__main__":
    main()
