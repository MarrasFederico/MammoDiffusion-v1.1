from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import dataset_variant_registry as dvr  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def build_fixture_project(root: Path, *, include_g_full_json: bool = True) -> None:
    """A minimal but schema-faithful project: two two-class generators (one with a direct
    produced_per_class count, one that must be resolved from a per_class metrics file), one
    positive-only generator, and one generator that cannot be resolved at all.
    """
    generators = [
        {"id": "gA_direct", "family": "fam", "notebook": "n", "classes": ["negative", "positive"],
         "status": "completed", "role": "final_comparison", "final_dataset": True, "produced_per_class": 100},
        {"id": "gB_metrics", "family": "fam", "notebook": "n", "classes": ["negative", "positive"],
         "status": "completed", "role": "final_comparison", "final_dataset": True,
         "metrics": "results/diffusers/gB_metrics/metrics/final_test_metrics.json"},
        {"id": "gC_positive_only", "family": "fam", "notebook": "n", "classes": ["positive"],
         "status": "completed_positive_only", "role": "baseline", "final_dataset": False,
         "metrics": "results/diffusers/gC_positive_only/metrics/final_filtered_vs_test.json"},
        {"id": "gD_unresolvable", "family": "fam", "notebook": "n", "classes": ["negative", "positive"],
         "status": "completed", "role": "ablation", "final_dataset": False},
        {"id": "gE_not_usable", "family": "fam", "notebook": "n", "classes": ["negative", "positive"],
         "status": "planned", "role": "ablation", "final_dataset": False, "produced_per_class": 999},
    ]
    write_json(root / "configs/final_generator_registry.json", {
        "schema_version": 1, "selection_policy": "x",
        "canonical_assets": {"diffusers_revision": "x", "diffusers_path": "x", "sd21_base_path": "x", "sd21_base_sha256": "x"},
        "generators": generators,
    })
    write_json(root / "configs/final_classifier_registry.json", {
        "schema_version": 1, "experiments": [
            {"experiment_id": "arch_legacy_partial", "architecture": "Arch", "training_dataset_variant": "real_plus_synthetic",
             "synthetic_source": "stable_diffusion_finetuned", "training_mode": "partial_finetuning"},
        ],
    })

    metrics_dir = root / "results/diffusers/gB_metrics/metrics"
    write_json(metrics_dir / "final_test_metrics.json", {
        "per_class": {"negative": {"n_generated": 80}, "positive": {"n_generated": 80}},
    })
    if include_g_full_json:
        write_json(root / "results/diffusers/gC_positive_only/metrics/final_filtered_vs_test.json", {
            "metrics": {"n_synthetic_filtered": 50, "target_label": 1},
        })

    real_csv = root / dvr.REAL_METADATA
    real_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = ["patient_id,image_id,label"]
    for i in range(60):
        rows.append(f"{1000+i},{2000+i},0")
    for i in range(20):
        rows.append(f"{3000+i},{4000+i},1")
    real_csv.write_text("\n".join(rows) + "\n")

    aug_dir = root / dvr.AUGMENTED_DIR
    aug_dir.mkdir(parents=True, exist_ok=True)
    for i in range(15):
        (aug_dir / f"mammo_{i:06d}_label1_aug0.png").write_bytes(b"png")


class DatasetVariantRegistryTests(unittest.TestCase):
    def test_real_and_augmented_counts_resolved_from_fixtures(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            self.assertEqual(dvr.real_count_by_class(root), {"negative": 60, "positive": 20})
            self.assertEqual(dvr.augmented_count_by_class(root), {"negative": 0, "positive": 15})

    def test_direct_and_metrics_based_counts_resolve(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            gen = dvr.load_generator_registry(root)["generators"]
            by_id = {g["id"]: g for g in gen}
            direct = dvr.resolve_generator_class_count(root, by_id["gA_direct"], "negative")
            self.assertEqual(direct["count"], 100); self.assertEqual(direct["source_precision"], "registry_direct")
            metrics = dvr.resolve_generator_class_count(root, by_id["gB_metrics"], "positive")
            self.assertEqual(metrics["count"], 80); self.assertEqual(metrics["source_precision"], "metrics_per_class")
            unresolved = dvr.resolve_generator_class_count(root, by_id["gD_unresolvable"], "negative")
            self.assertIsNone(unresolved["count"]); self.assertEqual(unresolved["source_precision"], "unresolved")

    def test_reroot_under_project_handles_foreign_absolute_path(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            foreign = "/home/someone/elsewhere/MammoDiffusion/data/synthetic/x/positive"
            rerooted = dvr._reroot_under_project(foreign, root)
            self.assertEqual(rerooted, root / "data/synthetic/x/positive")

    def test_directory_scan_fallback_finds_sibling_class_dir_via_reroot(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            # gC's json only reports positive; simulate a generator whose metrics embed a
            # foreign absolute path but whose *actual* sibling negative/ dir exists locally.
            write_json(root / "results/diffusers/gF/metrics/final_filtered_vs_test.json", {
                "config": {"synthetic_dir": "/foreign/mount/data/synthetic/gF/positive"},
                "metrics": {"n_synthetic_filtered": 33, "target_label": 1},
            })
            neg_dir = root / "data/synthetic/gF/negative"; neg_dir.mkdir(parents=True)
            for i in range(12): (neg_dir / f"img_{i}.png").write_bytes(b"png")
            entry = {"id": "gF", "classes": ["negative", "positive"], "status": "completed",
                     "metrics": "results/diffusers/gF/metrics/final_filtered_vs_test.json"}
            result = dvr.resolve_generator_class_count(root, entry, "negative")
            self.assertEqual(result["count"], 12); self.assertEqual(result["source_precision"], "directory_scan")

    def test_g05_style_positive_only_never_gets_negative_class(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            registry = dvr.build_stage1_registry(root)
            for v in registry["variants"]:
                if v["synthetic_generator_id"] == "gC_positive_only":
                    self.assertNotIn("negative", v["classes"])

    def test_positive_only_synthetic_keeps_both_real_classes(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            registry = dvr.build_stage1_registry(root)
            variants = [v for v in registry["variants"] if v["synthetic_generator_id"] == "gC_positive_only"]
            self.assertTrue(variants)
            for variant in variants:
                self.assertEqual(variant["real_count_by_class"], {"negative": 60, "positive": 20})
                self.assertNotIn("negative", variant["synthetic_count_by_class"])

    def test_no_duplicate_variant_ids(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            registry = dvr.build_stage1_registry(root)
            ids = [v["dataset_variant_id"] for v in registry["variants"]]
            self.assertEqual(len(ids), len(set(ids)))

    def test_controlled_budget_identical_across_generators(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            registry = dvr.build_stage1_registry(root)
            controlled = [v for v in registry["variants"] if v["dataset_variant_id"].startswith("RSB_CONTROLLED_")]
            self.assertTrue(controlled)
            counts = {frozenset(v["synthetic_count_by_class"].items()) for v in controlled}
            self.assertEqual(len(counts), 1, f"controlled variants disagree: {counts}")
            # budget must be min(gA=100, gB=80) = 80, never inflated to the larger generator's count
            only = next(iter(counts))
            self.assertEqual(dict(only), {"negative": 80, "positive": 80})

    def test_unresolvable_generator_excluded_from_common_budget_not_silently_zero(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            registry = dvr.build_stage1_registry(root)
            self.assertIn("gD_unresolvable", registry["budgets"]["unresolved_two_class_generators"])
            self.assertNotIn("RSB_CONTROLLED_gD_unresolvable", [v["dataset_variant_id"] for v in registry["variants"]])
            full = next(v for v in registry["variants"] if v["dataset_variant_id"] == "RSB_FULL_gD_unresolvable")
            self.assertEqual(full["status"], "invalid")

    def test_planned_generator_excluded_entirely(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            registry = dvr.build_stage1_registry(root)
            self.assertFalse(any("gE_not_usable" in v["dataset_variant_id"] for v in registry["variants"]))

    def test_deterministic_sample_signature_reproducible_and_order_independent(self):
        names = [f"img_{i:03d}.png" for i in range(50)]
        shuffled = list(reversed(names))
        sig1 = dvr.deterministic_sample_signature(names, 10, seed=7)
        sig2 = dvr.deterministic_sample_signature(shuffled, 10, seed=7)
        self.assertEqual(sig1["sha256"], sig2["sha256"])
        sig3 = dvr.deterministic_sample_signature(names, 10, seed=8)
        self.assertNotEqual(sig1["sha256"], sig3["sha256"])

    def test_deterministic_sample_signature_rejects_oversized_budget(self):
        with self.assertRaises(ValueError):
            dvr.deterministic_sample_signature(["a", "b"], 5, seed=1)

    def test_train_only_always_true(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            registry = dvr.build_stage1_registry(root)
            self.assertTrue(all(v["train_only"] for v in registry["variants"]))

    def test_legacy_variant_emitted_only_when_experiment_really_exists(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            registry = dvr.build_stage1_registry(root)
            ids = [v["dataset_variant_id"] for v in registry["variants"]]
            # fixture registers only a partial_finetuning legacy experiment for G02-equivalent;
            # RS_FULL_G02 (full_finetuning) has no matching experiment and must not be invented.
            self.assertNotIn("RS_FULL_G02", ids)
            self.assertNotIn("SYNTHETIC_ONLY_PARTIAL_G02", ids)  # fixture uses a different synthetic_source id than 02_...

    def test_legacy_variant_id_maps_to_02_sd21_filtered_generator(self):
        # This test uses the *real* project layout deliberately: it only reads
        # configs/final_classifier_registry.json + final_generator_registry.json (both
        # tracked in git), never data/ (gitignored), so it is safe to run against the repo.
        registry = dvr.load_classifier_registry(ROOT)
        gen_registry = dvr.load_generator_registry(ROOT)
        gen_ids = {g["id"] for g in gen_registry["generators"]}
        for exp in registry["experiments"]:
            if exp.get("synthetic_source") == "stable_diffusion_finetuned":
                self.assertIn("02_sd21_filtered_100steps", gen_ids)

    def test_stage2_variants_skip_g05_and_use_only_selected_union(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            stage2 = dvr.build_stage2_variants(root, ["gA_direct", "gC_positive_only"])
            ids = [v["dataset_variant_id"] for v in stage2]
            self.assertTrue(any("gA_direct" in i for i in ids))
            self.assertFalse(any("gC_positive_only" in i and i.startswith("S_ONLY") for i in ids))
            not_selected = dvr.build_stage2_variants(root, ["gB_metrics"])
            self.assertTrue(all("gB_metrics" in v["dataset_variant_id"] for v in not_selected))

    def test_validate_registry_flags_duplicate_ids(self):
        registry = {"variants": [
            {"dataset_variant_id": "X", "train_only": True, "classes": ["negative", "positive"],
             "budget_regime": "not_applicable", "status": "ready", "signature": {"a": 1}, "regime": "base",
             "synthetic_generator_id": None, "synthetic_count_by_class": {}},
            {"dataset_variant_id": "X", "train_only": True, "classes": ["negative", "positive"],
             "budget_regime": "not_applicable", "status": "ready", "signature": {"a": 1}, "regime": "base",
             "synthetic_generator_id": None, "synthetic_count_by_class": {}},
        ]}
        errors = dvr.validate_registry(registry)
        self.assertTrue(any("duplicate" in e for e in errors))

    def test_validate_registry_flags_g05_with_negative_class(self):
        registry = {"variants": [
            {"dataset_variant_id": "BAD", "train_only": True, "classes": ["negative", "positive"],
             "budget_regime": "not_applicable", "status": "ready", "signature": {"a": 1}, "regime": "base",
             "synthetic_generator_id": "05_ldm_basic_fromscratch", "synthetic_count_by_class": {}},
        ]}
        errors = dvr.validate_registry(registry)
        self.assertTrue(any("positive-only" in e for e in errors))

    def test_validate_registry_flags_nonuniform_controlled_budget(self):
        registry = {"variants": [
            {"dataset_variant_id": "A", "train_only": True, "classes": ["negative", "positive"],
             "budget_regime": "controlled", "status": "ready", "signature": {"a": 1}, "regime": "stage1_screening",
             "synthetic_generator_id": "g1", "synthetic_count_by_class": {"negative": 80, "positive": 80}},
            {"dataset_variant_id": "B", "train_only": True, "classes": ["negative", "positive"],
             "budget_regime": "controlled", "status": "ready", "signature": {"a": 1}, "regime": "stage1_screening",
             "synthetic_generator_id": "g2", "synthetic_count_by_class": {"negative": 50, "positive": 50}},
        ]}
        errors = dvr.validate_registry(registry)
        self.assertTrue(any("disagree" in e for e in errors))

    def test_build_and_write_produces_valid_registry_on_disk(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            registry = dvr.build_and_write(root)
            self.assertEqual(registry["validation_errors"], [])
            on_disk = json.loads((root / "configs/dataset_variant_registry.json").read_text())
            self.assertEqual(on_disk["variants"], registry["variants"])


if __name__ == "__main__":
    unittest.main()
