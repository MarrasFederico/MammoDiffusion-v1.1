"""Regression test for the distribution-metrics guard in benchmark notebook 01.

Excluded generators (for example the G06 generation-pool ablation, a provenance mismatch) are
absent from ``technical_rows``.  The distribution-metrics loop must skip them without raising a
``KeyError`` and must never emit a metric row for them.  This exercises the *behaviour* of the
actual notebook cell source (executed with stubbed heavy dependencies), not the presence of the
``.get`` string.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import generator_benchmark as gb  # noqa: E402

NOTEBOOK = ROOT / "notebooks/3_generator_benchmark/01_Unified_Generator_Benchmark.ipynb"


def _distribution_metrics_cell_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    matches = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "eligibility = {" in "".join(cell["source"])
        and "repeated_distribution_metrics(" in "".join(cell["source"])
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one distribution-metrics cell, found {len(matches)}")
    return matches[0]


class _NullLookup:
    """Return ``None`` for any subscript so argument evaluation never needs real features."""

    def __getitem__(self, key):
        return None


class _ListLookup(dict):
    def __missing__(self, key):
        return []


def _stub_summary() -> dict:
    """Minimal but structurally complete summary consumed by the append expression."""
    return {
        "full_pool_distribution_estimates": {},
        "balanced_prdc_point_estimates": {},
        "stability_estimates": {},
        "full_pool_real_count": 0,
        "full_pool_synthetic_count": 0,
        "balanced_prdc_point_real_count": 0,
        "balanced_prdc_point_synthetic_count": 0,
        "stability_subset_size": 0,
        "stability_interval_type": "repeated-subsampling stability interval",
        "full_pool_distribution_policy": "stub",
        "fid_full_pool_caveat": "stub",
    }


class DistributionMetricsGuardTests(unittest.TestCase):
    def _run_cell(self, technical_rows, candidate_audits):
        calls: list[tuple] = []

        def fake_repeated_distribution_metrics(reference, candidate, protocol, *, resampling_plan=None):
            # Records is empty so the extend() generator never indexes validation/candidate ids.
            return [], _stub_summary()

        namespace = {
            "RUN_REAL_BENCHMARK": True,
            "evaluation_subset_size": lambda *a, **k: 73,
            "balanced_subsample_indices": lambda *a, **k: [{"real_indices": [], "synthetic_indices": []}],
            "save_resampling_plan": lambda *a, **k: None,
            "repeated_distribution_metrics": fake_repeated_distribution_metrics,
            "write_csv_rows": lambda path, rows, *a, **k: rows,
            "protocol": {
                "synthetic_pool_target": 1361,
                "resampling": {"subsampling_fraction": 0.8, "stability_repetitions": 200,
                               "nearest_neighbour_k": 5},
                "sampling": {"seed": 17},
            },
            "validation_ids": list(range(73)),
            "REPRESENTATIONS": gb.REPRESENTATIONS,
            "FEATURE_SPACES": ["rad_dino"],
            "reference_features": _NullLookup(),
            "candidate_features": _NullLookup(),
            "candidate_ids": _ListLookup(),
            "OUTPUT_ROOT": Path("/tmp/nonexistent_benchmark_guard"),
            "technical_rows": technical_rows,
            "candidate_audits": candidate_audits,
        }

        # Track which (generator_id, representation) actually reach metric computation by wrapping
        # the stub to append after the guard has already let the pair through.
        def tracking(reference, candidate, protocol, *, resampling_plan=None):
            return [], _stub_summary()

        namespace["repeated_distribution_metrics"] = tracking

        exec(compile(_distribution_metrics_cell_source(), "<cell-16>", "exec"), namespace)  # noqa: S102
        return namespace["distribution_summaries"]

    def _technical_row(self, generator_id, condition, eligible):
        return {"generator_id": generator_id, "condition": condition,
                "eligible_for_distribution_metrics": eligible}

    def _audit(self, generator_id):
        return {"generator_id": generator_id}

    def test_excluded_generator_absent_from_eligibility_is_skipped(self):
        technical_rows = [
            self._technical_row("G02", "RAW", True),
            self._technical_row("G02", "FILTERED", True),
            self._technical_row("G07", "RAW", True),
            self._technical_row("G07", "FILTERED", True),
        ]
        # G06 is present as a candidate audit but absent from technical_rows (excluded).
        candidate_audits = [self._audit("G02"), self._audit("G06"), self._audit("G07")]

        summaries = self._run_cell(technical_rows, candidate_audits)

        produced = {(row["generator_id"], row["condition"]) for row in summaries}
        # G02 and G07 processed for both representations.
        self.assertIn(("G02", "RAW"), produced)
        self.assertIn(("G02", "FILTERED"), produced)
        self.assertIn(("G07", "RAW"), produced)
        self.assertIn(("G07", "FILTERED"), produced)
        # G06 never produces a row.
        self.assertFalse(any(gid == "G06" for gid, _ in produced),
                         f"excluded generator produced rows: {produced}")

    def test_ineligible_condition_is_skipped_without_error(self):
        technical_rows = [
            self._technical_row("G02", "RAW", False),
            self._technical_row("G02", "FILTERED", True),
        ]
        candidate_audits = [self._audit("G02"), self._audit("G06")]

        summaries = self._run_cell(technical_rows, candidate_audits)
        produced = {(row["generator_id"], row["condition"]) for row in summaries}
        self.assertEqual(produced, {("G02", "FILTERED")})


if __name__ == "__main__":
    unittest.main()
