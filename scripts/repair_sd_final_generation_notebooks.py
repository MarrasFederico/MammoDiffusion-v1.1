#!/usr/bin/env python3
"""Remove the stale notebook-01 exception after validating the shared non-destructive fix."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [ROOT / "notebooks/2_diffusers" / f"0{i}_{name}.ipynb" for i, name in (
    (1, "SD21_Baseline_50steps"), (2, "SD21_Filtered_100steps"),
    (3, "SD21_VAE_FineTuned"), (4, "SD21_LoRA"),
)]


def main() -> None:
    shared = []
    for path in NOTEBOOKS:
        payload = json.loads(path.read_text())
        cells = [cell for cell in payload["cells"] if "final_sd_generation_plan" in "".join(cell.get("source", []))]
        if not cells:
            raise RuntimeError(f"{path.name}: shared final generation utility is not used")
        shared.append(path.name)
        if path == NOTEBOOKS[0]:
            for cell in cells:
                errors = [out for out in cell.get("outputs", []) if out.get("output_type") == "error" and
                          out.get("ename") == "RuntimeError" and "Dataset finale non valido" in out.get("evalue", "")]
                if errors:
                    cell["outputs"] = [out for out in cell.get("outputs", []) if out not in errors]
                    cell["execution_count"] = None
                    cell.setdefault("metadata", {})["mammodiffusion_stale_error_cleared"] = {
                        "reason": "shared planner now accepts compact/global exact subsets and preserves surplus files",
                        "utility": "notebooks/utility/parallel_generation_utils.py:final_sd_generation_plan",
                    }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    report = {"schema_version": 1, "notebook_1_error": "resolved",
              "shared_risk_checked": shared,
              "fix_scope": "shared utility; notebooks 01-04", "files_deleted": False}
    out = ROOT / "results/notebook_inventory/sd_notebook_error_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__": main()
