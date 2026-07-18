from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def csv_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise AssertionError(f"expected an explicit CSV boolean, found {value!r}")
    return normalized == "true"


class GeneratorCandidateAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads((ROOT / "configs/generator_registry.json").read_text(encoding="utf-8"))
        cls.index = json.loads((ROOT / "configs/generator_provenance_index.json").read_text(encoding="utf-8"))
        with (ROOT / "results/2_diffusers/benchmark/candidate_audit.csv").open(
                newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            cls.audit_fields = set(reader.fieldnames or [])
            cls.audit_rows = list(reader)

    def test_registry_index_and_candidate_audit_identities_are_consistent(self):
        registry = {row["id"]: row for row in self.registry["generators"]}
        index = {row["generator_id"]: row for row in self.index["generators"]}
        audit = {row["generator_id"]: row for row in self.audit_rows}
        self.assertEqual(set(registry), set(index))
        self.assertEqual(set(registry), set(audit))
        required = {
            "generator_id", "scientific_family", "candidate_role", "parent_generator_id",
            "model_identity_sha256", "generation_identity_sha256", "duplicate_model_group",
            "distinct_generator_for_ranking", "audit_mode", "audit_generated_at",
            "project_root_independent_paths", "provenance_record_schema_valid",
            "provenance_index_consistent", "runtime_manifest_hashes_declared",
            "runtime_manifest_contents_verified", "runtime_assets_verified",
            "runtime_assets_unavailable", "runtime_assets_mismatch",
            "eligible_for_descriptive_benchmark", "eligible_for_official_family_ranking",
            "raw_count", "filtered_count", "filter_acceptance_rate_descriptive",
            "provenance_manifest_exists", "provenance_manifest_valid", "lineage_complete",
            "raw_manifest_valid", "filter_manifest_valid", "sample_set_matches_manifest",
            "training_corpus_manifest", "training_corpus_manifest_valid",
            "provenance_failure_reason", "block_reasons",
        }
        self.assertTrue(required <= self.audit_fields)
        self.assertTrue({"canonical_manifests_internally_valid", "mapping_complete"}.isdisjoint(self.audit_fields))
        official = set()
        for generator_id, registry_row in registry.items():
            index_row, audit_row = index[generator_id], audit[generator_id]
            for field in ("candidate_role", "parent_generator_id", "model_identity_sha256",
                          "generation_identity_sha256"):
                expected = "" if registry_row.get(field) is None else str(registry_row.get(field))
                self.assertEqual(str(index_row.get(field) or ""), expected, (generator_id, field, "index"))
                self.assertEqual(audit_row[field], expected, (generator_id, field, "audit"))
            distinct = bool(registry_row["distinct_generator_for_ranking"])
            self.assertEqual(bool(index_row["distinct_generator_for_ranking"]), distinct)
            self.assertEqual(csv_bool(audit_row["distinct_generator_for_ranking"]), distinct)
            expected_official = bool(registry_row["eligible_for_downstream_selection"]) and \
                registry_row["candidate_role"] == "primary_candidate" and distinct
            self.assertEqual(bool(index_row["eligible_for_downstream_selection"]), expected_official)
            self.assertEqual(csv_bool(audit_row["eligible_for_official_family_ranking"]), expected_official)
            if expected_official:
                self.assertTrue(csv_bool(audit_row["eligible_for_descriptive_benchmark"]))
                official.add(generator_id)
        self.assertEqual(official, {
            "02_sd21_filtered_100steps", "03_sd21_vae_finetuned", "04_sd21_lora",
            "07_ldm_sdvae_extra1361", "08_ldm_v3_sdvae_fromscratch",
        })

    def test_g06_shared_model_pool_ablation_cannot_be_primary(self):
        audit = {row["generator_id"]: row for row in self.audit_rows}
        g05, g06 = audit["05_ldm_basic_fromscratch"], audit["06_ldm_extra1361_fromscratch"]
        self.assertEqual(g06["candidate_role"], "generation_pool_ablation")
        self.assertNotEqual(g06["candidate_role"], "primary_candidate")
        self.assertEqual(g06["parent_generator_id"], "05_ldm_basic_fromscratch")
        self.assertEqual(g06["model_identity_sha256"], g05["model_identity_sha256"])
        self.assertFalse(csv_bool(g06["distinct_generator_for_ranking"]))
        self.assertFalse(csv_bool(g06["eligible_for_descriptive_benchmark"]))
        self.assertFalse(csv_bool(g06["eligible_for_official_family_ranking"]))
        self.assertTrue(csv_bool(g06["runtime_assets_mismatch"]))
        self.assertIn("synth_04082.png->synth_filtered_0032.png", g06["block_reasons"])

    def test_notebook_has_separate_metadata_only_audit_refresh(self):
        notebook = nbformat.read(ROOT / "notebooks/3_generator_benchmark/01_Unified_Generator_Benchmark.ipynb", 4)
        source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        self.assertIn("REFRESH_CANDIDATE_AUDIT = True", source)
        self.assertIn("candidate_audit_document_rows(candidate_audits)", source)
        self.assertIn("if REFRESH_CANDIDATE_AUDIT:", source)
        self.assertNotIn("'family': row['scientific_family']", source)
        self.assertNotIn("'role': row['candidate_role']", source)


if __name__ == "__main__":
    unittest.main()
