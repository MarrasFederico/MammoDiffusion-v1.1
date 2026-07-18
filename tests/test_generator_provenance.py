from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import generator_benchmark as gb  # noqa: E402


def make_image(path: Path, value: int) -> Path:
    pixels = np.full((16, 16), value, dtype=np.uint8); pixels[0, :] = min(255, value + 20)
    path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(pixels).save(path); return path


class CanonicalProvenanceTests(unittest.TestCase):
    def _fixture(self, root: Path):
        train_image = make_image(root / "data/train/real.png", 40)
        with (root / "training.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["file_name", "label", "patient_id", "image_id", "source"])
            writer.writeheader(); writer.writerow({"file_name": "data/train/real.png", "label": 0,
                                                    "patient_id": "p1", "image_id": "i1", "source": "real"})
        raw = [make_image(root / f"raw/r{index}.png", 60 + index * 30) for index in range(3)]
        filtered = root / "filtered/f0.png"; filtered.parent.mkdir(); filtered.write_bytes(raw[0].read_bytes())
        with (root / "filter.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["source_path", "source_name", "selected", "selection_rank", "reject_reason", "filtered_path"])
            writer.writeheader()
            writer.writerow({"source_path": raw[0], "source_name": raw[0].name, "selected": True,
                             "selection_rank": 0, "reject_reason": "", "filtered_path": filtered})
            for item in raw[1:]: writer.writerow({"source_path": item, "source_name": item.name, "selected": False,
                                                  "selection_rank": "", "reject_reason": "not_top_k", "filtered_path": ""})
        checkpoint = root / "model/checkpoint.bin"; checkpoint.parent.mkdir(); checkpoint.write_bytes(b"checkpoint")
        component_hash = gb.file_sha256(checkpoint)
        component_roles = ("unet", "unet_config", "base_model_config", "vae", "vae_config", "text_encoder",
                           "text_encoder_config", "tokenizer_config", "tokenizer_vocab", "scheduler_config")
        sources = {"generator_id": "fixture_generator", "scientific_family": "finetuned",
                   "candidate_role": "primary_candidate", "checkpoint": "model/checkpoint.bin",
                   "training_metadata": "training.csv", "training_dataset_identifier": "fixture_train",
                   "training_notebook_config_identifier": "fixture.ipynb:cell1", "raw_directory": "raw",
                   "filtered_directory": "filtered", "filter_report": "filter.csv", "filtered_target": 1,
                   "sampling_steps": 100, "generation_configuration": {"base_seed": 42, "seed_strategy": "stateless_seed_per_image_v1"},
                   "guidance_conditioning_configuration": {"guidance_scale": 7.5, "prompt": "fixture"},
                   "filter_configuration": {"selection": "top_k"}, "scheduler": "PNDMScheduler",
                   "generation_code_signature": component_hash,
                   "model_configuration": {"model_type": "stable_diffusion_full_unet",
                                           "base_model_identifier": "fixture", "scheduler": "PNDMScheduler"},
                   "model_components": [{"role": role, "identifier": f"fixture-{role}",
                                          "path": "model/checkpoint.bin", "sha256": component_hash,
                                          "source_type": "local"} for role in component_roles]}
        return sources, checkpoint, raw, filtered, train_image

    def test_build_and_audit_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); sources, checkpoint, raw, filtered, _ = self._fixture(root)
            payload = gb.build_canonical_generator_provenance(root, sources)
            output = root / "results/2_diffusers/provenance/fixture_generator"
            runtime = root / "results/2_diffusers/provenance/runtime/fixture_generator"
            self.assertEqual(payload["checkpoint_sha256"], gb.file_sha256(checkpoint))
            self.assertTrue((output / "provenance.json").is_file())
            self.assertTrue((root / gb.SHARED_TRAINING_CORPUS).is_file())
            for name in ("raw_samples.csv", "filtered_samples.csv", "filter_mapping.csv"):
                self.assertTrue((runtime / name).is_file())
            self.assertFalse((output / "training_corpus.csv").exists())
            paths, ids, labels, source_labels = gb.training_corpus_from_manifest(root, payload["training_corpus_manifest"])
            self.assertEqual((len(paths), len(ids), labels[ids[0]], source_labels[ids[0]]), (1, 1, 0, "real"))
            entry = {"id": "fixture_generator", "scientific_family": "finetuned", "model_family": "fixture",
                     "model_variant": "v1", "candidate_role": "primary_candidate", "eligible_for_downstream_selection": True,
                     "checkpoint": "model/checkpoint.bin", "provenance_manifest": str((output / "provenance.json").relative_to(root)),
                     "raw_generation_manifest": payload["raw_sample_manifest"], "training_corpus_manifest": payload["training_corpus_manifest"],
                     "filtering_applied": True, "filter_manifest": payload["filter_mapping_manifest"],
                     "samples": {"raw_positive": "raw", "filtered_positive": "filtered"}}
            protocol = copy.deepcopy(gb.load_protocol(ROOT)); protocol["synthetic_pool_target"] = 1
            audit = gb.audit_candidate(root, entry, protocol)
            self.assertTrue(audit["provenance_manifest_valid"] and audit["lineage_complete"])
            self.assertTrue(audit["runtime_manifest_contents_verified"] and audit["runtime_assets_verified"])
            self.assertTrue(audit["eligible_for_descriptive_benchmark"] and audit["eligible_for_official_family_ranking"])
            self.assertEqual((audit["canonical_raw_count"], audit["canonical_filtered_count"]), (3, 1))
            checkpoint.write_bytes(b"changed")
            self.assertFalse(gb.audit_candidate(root, entry, protocol)["provenance_manifest_valid"])

    def test_shared_corpus_dependencies_deduplicate_and_invalidate_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); shared = root / "runtime/shared.csv"; other = root / "runtime/other.csv"
            shared.parent.mkdir(parents=True); shared.write_text("sample_id\na\n"); other.write_text("sample_id\nb\n")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            index = {"generators": [
                {"generator_id": "same_a", "training_corpus_manifest": "runtime/shared.csv", "training_corpus_manifest_sha256": digest(shared)},
                {"generator_id": "same_b", "training_corpus_manifest": "runtime/shared.csv", "training_corpus_manifest_sha256": digest(shared)},
                {"generator_id": "different", "training_corpus_manifest": "runtime/other.csv", "training_corpus_manifest_sha256": digest(other)},
            ]}
            self.assertEqual(index["generators"][0]["training_corpus_manifest"], index["generators"][1]["training_corpus_manifest"])
            self.assertNotEqual(index["generators"][0]["training_corpus_manifest"], index["generators"][2]["training_corpus_manifest"])
            self.assertEqual(gb.audit_training_corpus_dependencies(root, index), {"same_a": True, "same_b": True, "different": True})
            shared.write_text("sample_id\nchanged\n")
            self.assertEqual(gb.audit_training_corpus_dependencies(root, index), {"same_a": False, "same_b": False, "different": True})

    def test_repository_shared_corpus_declaration_matches_training_evidence(self):
        sources = json.loads((ROOT / "configs/generator_provenance_sources.json").read_text())
        shared = sources["shared_training_corpus"]
        self.assertEqual(set(shared["dependent_generators"]), {row["generator_id"] for row in sources["generators"]})
        self.assertIn("AUGMENTED_METADATA_PATH", (ROOT / "notebooks/utility/train_ldm.py").read_text())
        self.assertIn("augmented_df", (ROOT / "notebooks/utility/train_ldm.py").read_text())
        metadata = ROOT / sources["training_corpus_evidence"]
        if not metadata.is_file():
            self.skipTest("runtime assets unavailable: shared training metadata is not included in source archives")
        self.assertEqual(hashlib.sha256(metadata.read_bytes()).hexdigest(), shared["source_metadata_sha256"])
        with metadata.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3061)
        counts = {(label, source): sum(row["label"] == label and row["source"] == source for row in rows)
                  for label, source in (("0", "real"), ("1", "real"), ("1", "positive_augmentation"))}
        self.assertEqual(counts, {("0", "real"): 1701, ("1", "real"): 340, ("1", "positive_augmentation"): 1020})

    def test_source_audit_never_claims_runtime_assets(self):
        protocol = gb.load_protocol(ROOT); registry = gb.load_registry(ROOT); index = gb.load_provenance_index(ROOT)
        rows = [gb.audit_source_generator_metadata(ROOT, entry, protocol, index) for entry in registry["generators"]]
        by_id = {row["generator_id"]: row for row in rows}
        for generator_id in ("01_sd21_baseline_50steps", "02_sd21_filtered_100steps", "03_sd21_vae_finetuned",
                             "04_sd21_lora", "05_ldm_basic_fromscratch", "07_ldm_sdvae_extra1361", "08_ldm_v3_sdvae_fromscratch"):
            self.assertTrue(by_id[generator_id]["provenance_recorded"])
            self.assertTrue(by_id[generator_id]["provenance_record_schema_valid"])
            self.assertTrue(by_id[generator_id]["provenance_index_consistent"])
            self.assertTrue(by_id[generator_id]["runtime_manifest_hashes_declared"])
            self.assertFalse(by_id[generator_id]["runtime_manifest_contents_verified"])
            self.assertFalse(by_id[generator_id]["runtime_assets_verified"])
            self.assertTrue(by_id[generator_id]["runtime_assets_unavailable"])
        self.assertTrue(by_id["06_ldm_extra1361_fromscratch"]["provenance_record_schema_valid"])
        self.assertTrue(by_id["06_ldm_extra1361_fromscratch"]["provenance_index_consistent"])
        self.assertFalse(by_id["06_ldm_extra1361_fromscratch"]["runtime_manifest_hashes_declared"])
        self.assertFalse(by_id["06_ldm_extra1361_fromscratch"]["runtime_manifest_contents_verified"])
        self.assertTrue(by_id["06_ldm_extra1361_fromscratch"]["runtime_assets_mismatch"])

    def test_sd_name_preserved_maps_and_missing_mapping_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); raw = make_image(root / "raw/a.png", 80)
            filtered = root / "filtered/a.png"; filtered.parent.mkdir(); filtered.write_bytes(raw.read_bytes())
            report = root / "report.csv"
            report.write_text("source_path,source_name,selected,selection_rank,reject_reason\n" + f"{raw},a.png,True,1,\n")
            self.assertTrue(gb.filter_acceptance_from_manifest(report, [filtered], [raw])["filter_manifest_valid"])
            filtered.rename(root / "filtered/renamed.png")
            result = gb.filter_acceptance_from_manifest(report, [root / "filtered/renamed.png"], [raw])
            self.assertFalse(result["filter_manifest_valid"])

    def test_sample_identity_never_uses_basename_only(self):
        left = [{"sample_id": "s", "relative_path": "one/a.png", "sha256": "h"}]
        self.assertFalse(gb.sample_sets_equivalent(left, [{"sample_id": "s", "relative_path": "two/a.png", "sha256": "h"}]))
        self.assertFalse(gb.sample_sets_equivalent(left, [{"sample_id": "s", "relative_path": "one/a.png", "sha256": "other"}]))
        self.assertTrue(gb.sample_sets_equivalent(left, [dict(left[0])]))


class GeneratorIdentityTests(unittest.TestCase):
    def _record(self, generator_id="02_sd21_filtered_100steps"):
        return json.loads((ROOT / "results/2_diffusers/provenance" / generator_id / "provenance.json").read_text())

    def test_schema_and_record_versions_are_explicit(self):
        schema = json.loads((ROOT / "configs/generator_provenance_schema.json").read_text())
        self.assertEqual(schema["schema_version"], 2)
        record = self._record()
        gb.validate_generator_provenance_record(ROOT, record, schema)
        for version in (1, 3, None):
            changed = copy.deepcopy(record)
            if version is None: changed.pop("schema_version")
            else: changed["schema_version"] = version
            with self.assertRaisesRegex(ValueError, "unsupported provenance record version"):
                gb.validate_generator_provenance_record(ROOT, changed, schema)

    def test_primary_candidate_missing_vae_identity_fails(self):
        record = self._record()
        record["model_components"] = [row for row in record["model_components"] if row["role"] != "vae"]
        with self.assertRaisesRegex(ValueError, "vae"):
            gb.validate_generator_provenance_record(ROOT, record)

    def test_lora_missing_base_model_identity_fails(self):
        record = self._record("04_sd21_lora")
        record["model_components"] = [row for row in record["model_components"] if row["role"] != "base_unet"]
        with self.assertRaisesRegex(ValueError, "base_unet"):
            gb.validate_generator_provenance_record(ROOT, record)

    def test_ldm_missing_latent_or_vae_identity_fails(self):
        record = self._record("07_ldm_sdvae_extra1361")
        for missing_role in ("latent_stats", "latents_manifest", "vae"):
            changed = copy.deepcopy(record)
            changed["model_components"] = [row for row in changed["model_components"] if row["role"] != missing_role]
            with self.assertRaises(ValueError):
                gb.validate_generator_provenance_record(ROOT, changed)

    def test_component_order_is_irrelevant_but_hash_change_changes_identity(self):
        record = self._record()
        original = gb.model_identity_sha256(record["model_components"], record["model_configuration"])
        self.assertEqual(original, gb.model_identity_sha256(list(reversed(record["model_components"])), record["model_configuration"]))
        changed = copy.deepcopy(record["model_components"])
        changed[0]["sha256"] = "0" * 64
        self.assertNotEqual(original, gb.model_identity_sha256(changed, record["model_configuration"]))

    def test_same_model_different_sampling_has_distinct_generation_identity(self):
        g01, g02 = self._record("01_sd21_baseline_50steps"), self._record("02_sd21_filtered_100steps")
        self.assertEqual(g01["model_identity_sha256"], g02["model_identity_sha256"])
        self.assertNotEqual(g01["generation_identity_sha256"], g02["generation_identity_sha256"])

    def test_duplicate_model_cannot_rank_twice(self):
        identity = self._record()["model_identity_sha256"]
        rows = [{"generator_id": "a", "model_identity_sha256": identity, "candidate_role": "primary_candidate",
                 "eligible_for_official_family_ranking": True},
                {"generator_id": "b", "model_identity_sha256": identity, "candidate_role": "primary_candidate",
                 "eligible_for_official_family_ranking": True}]
        with self.assertRaisesRegex(ValueError, "cannot enter official ranking twice"):
            gb.detect_duplicate_generator_identities(rows)

    def test_g05_g06_same_identity_is_pool_ablation(self):
        g05, g06 = self._record("05_ldm_basic_fromscratch"), self._record("06_ldm_extra1361_fromscratch")
        self.assertEqual(g05["model_identity_sha256"], g06["model_identity_sha256"])
        classification = gb.classify_generation_pool_ablation(g05, g06)
        self.assertEqual(classification["candidate_role"], "generation_pool_ablation")
        self.assertFalse(classification["eligible_for_downstream_selection"])

    def test_different_complete_identity_remains_distinct_but_blocked(self):
        g05, candidate = self._record("05_ldm_basic_fromscratch"), self._record("06_ldm_extra1361_fromscratch")
        candidate["model_identity_sha256"] = "f" * 64
        classification = gb.classify_generation_pool_ablation(g05, candidate)
        self.assertFalse(classification["same_generator_identity"])
        self.assertTrue(classification["distinct_generator_for_ranking"])
        self.assertFalse(classification["eligible_for_downstream_selection"])


class GateMetricAndCacheTests(unittest.TestCase):
    def _eligible_row(self, count=1361):
        return {"valid_positive_images": count, "synthetic_exact_duplicate_rate": 0,
                "perceptual_hash_duplicate_rate": 0, "train_memorization_rate": 0,
                "raddino_coverage": .8, "filter_acceptance_rate": 1 / 3,
                "filter_manifest_valid": True, "filter_provenance_complete": True, "n_corrupt": 0,
                "metrics_complete": True, "test_access": False, "lineage_complete": True,
                "provenance_manifest_valid": True, "training_corpus_manifest_valid": True,
                "eligible_for_selection": True}

    def test_one_third_acceptance_is_descriptive_but_target_is_mandatory(self):
        gates = gb.load_protocol(ROOT)["eligibility_gates"]
        self.assertNotIn("filter_acceptance_rate", gb.eligibility_failures(self._eligible_row(), gates))
        self.assertIn("valid_positive_images", gb.eligibility_failures(self._eligible_row(1360), gates))

    def test_distribution_outputs_have_distinct_balanced_and_full_pool_sections(self):
        protocol = copy.deepcopy(gb.load_protocol(ROOT)); protocol["synthetic_pool_target"] = 20
        protocol["resampling"].update(stability_repetitions=3, nearest_neighbour_k=3)
        rng = np.random.default_rng(4)
        rows, summary = gb.repeated_distribution_metrics(rng.normal(size=(9, 5)), rng.normal(size=(20, 5)), protocol)
        self.assertEqual((summary["full_pool_real_count"], summary["full_pool_synthetic_count"]), (9, 20))
        self.assertEqual((summary["balanced_prdc_point_real_count"], summary["balanced_prdc_point_synthetic_count"]), (9, 9))
        self.assertTrue(all(len(row["real_indices"]) == len(row["synthetic_indices"]) for row in rows))
        self.assertEqual(set(summary) & {"full_reference_estimates"}, set())

    def test_weight_and_processor_identity_changes_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); paths = [make_image(root / "a.png", 40), make_image(root / "b.png", 90)]
            manifest = root / "manifest.csv"; manifest.write_text("sample_id\na\n")
            calls = []
            def extract(values, name): calls.append(name); return np.ones((2, 3)) * len(calls)
            kwargs = {"extractor": "rad_dino", "preprocessing": "processor", "code_version": "v4",
                      "source_manifest": str(manifest), "extractor_model_id": "microsoft/rad-dino",
                      "extractor_weights_identifier": "identity", "feature_dimension": 3, "extract_fn": extract}
            cache = root / "cache.npy"; identity = {"weight_composite_sha256": "w1", "processor_config_sha256": "p1"}
            gb.get_or_extract_embeddings(cache, paths, ["a", "b"], extractor_identity=identity, **kwargs)
            gb.get_or_extract_embeddings(cache, paths, ["a", "b"], extractor_identity=identity, **kwargs)
            gb.get_or_extract_embeddings(cache, paths, ["a", "b"], extractor_identity={**identity, "weight_composite_sha256": "w2"}, **kwargs)
            gb.get_or_extract_embeddings(cache, paths, ["a", "b"], extractor_identity={**identity, "processor_config_sha256": "p2"}, **kwargs)
            self.assertEqual(len(calls), 3)


if __name__ == "__main__": unittest.main()
