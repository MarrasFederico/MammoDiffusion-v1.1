"""Regenerate the canonical generator ranking from the saved summary CSV.

``generator_summary.csv`` is a required scientific input. This utility never
reconstructs or modifies that summary and never opens images, models, caches,
datasets, or checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from generator_benchmark import rank_generator_family, write_csv_rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required generator summary is missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"required generator summary is empty: {path}")
    return rows


def rebuild_ranking(source_root: str | Path,
                    output_root: str | Path | None = None) -> Path:
    """Write ``generator_ranking.csv`` from the required saved summary."""
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve() if output_root else source_root
    summary_path = source_root / "results/2_diffusers/benchmark/generator_summary.csv"
    protocol_path = source_root / "configs/generator_benchmark_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    summary_rows = _read_csv(summary_path)
    filtered_candidates = [
        row for row in summary_rows
        if row.get("condition") == "FILTERED"
        and str(row.get("eligible_for_selection", "")).lower() == "true"
    ]
    ranking: list[dict[str, object]] = []
    for family in ("finetuned", "from_scratch"):
        ranking.extend(
            rank_generator_family(
                filtered_candidates, family, protocol["eligibility_gates"]
            )
        )

    destination = output_root / "results/2_diffusers/benchmark/generator_ranking.csv"
    return write_csv_rows(destination, ranking)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(rebuild_ranking(args.source_root, args.output_root))
