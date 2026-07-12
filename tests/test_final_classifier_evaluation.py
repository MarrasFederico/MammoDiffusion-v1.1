import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

from final_classifier_evaluation import (  # noqa: E402
    ARCHITECTURE_FAMILY,
    PREDICTION_COLUMNS,
    aggregate_predictions_by_patient,
    build_experiment_registry,
    build_locked_finalist_entries,
    build_test_coverage_table,
    build_test_dataset_manifest,
    canonical_test_prediction_paths,
    checkpoint_key_mismatch,
    compare_patient_sets,
    compute_binary_metrics,
    content_signature,
    delong_roc_test,
    holm_adjustment,
    lock_finalists_manifest,
    mcnemar_test,
    paired_stratified_bootstrap,
    select_validation_finalists,
    standardize_prediction_dataframe,
    unwrap_checkpoint_state_dict,
    validate_locked_finalists_manifest,
    validate_prediction_cache,
    write_prediction_manifest,
    strict_json_dumps,
)


class FinalClassifierEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = build_experiment_registry(ROOT / "configs/final_classifier_registry.json")
        cls.coverage = build_test_coverage_table(cls.registry, ROOT)

    def sample_rows(self):
        return pd.DataFrame([
            {"experiment_id": "a_base", "architecture": "A", "training_dataset_variant": "real_only", "validation_roc_auc": .60, "validation_pr_auc": .30, "eligible_for_final_selection": True, "required_for_primary_comparison": False},
            {"experiment_id": "a_syn", "architecture": "A", "training_dataset_variant": "real_plus_synthetic", "validation_roc_auc": .70, "validation_pr_auc": .35, "eligible_for_final_selection": True, "required_for_primary_comparison": False},
            {"experiment_id": "a_aug", "architecture": "A", "training_dataset_variant": "real_plus_augmented", "validation_roc_auc": .65, "validation_pr_auc": .34, "eligible_for_final_selection": True, "required_for_primary_comparison": False},
        ])

    def canonical_frame(self, scores=(.1, .8), patients=("1", "2")):
        return pd.DataFrame({"patient_id": patients, "image_id": ("11", "22"), "path": ("1_11.png", "2_22.png"), "y_true": (0, 1), "y_score": scores, "y_pred": (0, 1), "threshold": (.5, .5), "threshold_method": ("youden_validation",) * 2, "split": ("test",) * 2, "experiment_id": ("e",) * 2, "architecture": ("A",) * 2, "dataset_variant": ("real_only",) * 2, "synthetic_source": ("none",) * 2, "training_mode": ("partial_finetuning",) * 2, "checkpoint_path": ("x.pt",) * 2})

    def skip_unless_experiments_available(self):
        # The light code-audit package (scripts/package_code_audit.sh) never includes
        # experiments/ (checkpoints, generated images): any assertion that genuinely needs those
        # real bytes must skip there instead of failing, rather than pretending to have verified
        # something that wasn't actually shipped.
        if not (ROOT / "experiments").is_dir():
            self.skipTest("requires experiments/ (checkpoints), excluded from the light audit package")

    # Audit and registry (1-7)
    def test_01_registry_valid(self): self.assertEqual(len(self.registry), 22)
    def test_02_incomplete_experiment_has_reason(self):
        rows = [x for x in self.registry if not x["eligible_for_final_selection"]]
        self.assertTrue(all(x["exclusion_reason"] for x in rows))
    def test_03_audit_detects_model_without_test(self):
        self.skip_unless_experiments_available()
        self.assertTrue((self.coverage.status == "READY_FOR_TEST_INFERENCE").any())
    def test_04_audit_distinguishes_aggregate_metrics(self):
        self.skip_unless_experiments_available()
        row = self.coverage.set_index("experiment_id").loc["maxvit512_02i_real_aug_synth_fromscratch"]
        self.assertTrue(row.test_metrics_available); self.assertFalse(row.test_predictions_available)
    def test_05_validation_loser_does_not_get_test_notebook(self):
        row = next(x for x in self.registry if x["experiment_id"] == "resnet50_01c_real_synth_full")
        self.assertIsNone(row["test_notebook"])
    def test_06_status_coherent(self): self.assertTrue(set(self.coverage.status) <= {"COMPLETE", "READY_FOR_TEST_INFERENCE", "TEST_PREDICTIONS_AVAILABLE", "INCOMPLETE", "EXCLUDED_ON_VALIDATION", "NOT_REQUIRED", "INVALID_PROVENANCE"})
    def test_07_synthetic_only_not_training_mode(self):
        row = next(x for x in self.registry if x["experiment_id"] == "maxvit512_02d_synthetic_only")
        self.assertEqual(row["training_dataset_variant"], "synthetic_only"); self.assertNotEqual(row["training_mode"], "synthetic_only")

    # Validation selection (8-20)
    def test_08_selection_uses_validation(self): self.assertEqual(select_validation_finalists(self.sample_rows())[0]["experiment_id"], "a_syn")
    def test_09_test_metrics_ignored(self):
        rows = self.sample_rows(); rows["test_roc_auc"] = [1, 0, 0]
        self.assertEqual(select_validation_finalists(rows)[0]["experiment_id"], "a_syn")
    def test_10_baseline_included(self): self.assertIn("a_base", {x["experiment_id"] for x in select_validation_finalists(self.sample_rows())})
    def test_11_best_synthetic_included(self): self.assertIn("a_syn", {x["experiment_id"] for x in select_validation_finalists(self.sample_rows())})
    def test_12_best_augmentation_included(self): self.assertIn("a_aug", {x["experiment_id"] for x in select_validation_finalists(self.sample_rows())})
    def test_13_selection_deduplicates(self):
        selected = select_validation_finalists(self.sample_rows()); self.assertEqual(len(selected), len({x["experiment_id"] for x in selected}))
    def test_14_max_finalists(self): self.assertLessEqual(len(select_validation_finalists(self.coverage, {"max_finalists_total": 8})), 8)
    def test_15_02w_one_winner(self):
        candidates = self.coverage[self.coverage.experiment_id.str.contains("02[gh]_fromscratch")]
        eligible = set(candidates.loc[candidates.eligible_for_final_selection, "experiment_id"])
        self.assertEqual(eligible, {"maxvit512_02h_fromscratch_synthetic_full"})
    def test_16_raddino_policy_both(self):
        selected = {x["experiment_id"] for x in select_validation_finalists(self.coverage)}
        self.assertTrue({"raddino_04a_real_only", "raddino_04b_real_synth"} <= selected)
    def test_17_raddino_winner_only_supported(self):
        rows = self.coverage.copy(); rows.loc[rows.experiment_id.eq("raddino_04b_real_synth"), ["eligible_for_final_selection", "scientifically_eligible", "required_for_primary_comparison"]] = False
        selected = {x["experiment_id"] for x in select_validation_finalists(rows)}; self.assertNotIn("raddino_04b_real_synth", selected)
    def test_18_manifest_locked(self): self.assertTrue(lock_finalists_manifest([], {})["locked"])
    def test_19_lock_signature_verified(self): self.assertTrue(validate_locked_finalists_manifest(lock_finalists_manifest([], {}))["locked"])
    def test_20_nonlocked_refused(self):
        with self.assertRaises(ValueError): validate_locked_finalists_manifest({"locked": False})

    # Predictions (21-32)
    def test_21_checkpoint_signature_locked(self):
        self.skip_unless_experiments_available()
        manifest = json.loads((ROOT / "results/final_evaluation/finalists_manifest.json").read_text()); item = manifest["finalists"][0]
        self.assertEqual(item["checkpoint_signature"], content_signature(ROOT / item["checkpoint_path"]))
    def test_22_threshold_is_validation_derived(self):
        self.assertTrue(all("validation" in x["threshold_method"] for x in json.loads((ROOT / "results/final_evaluation/finalists_manifest.json").read_text())["finalists"]))
    def test_23_duplicate_patient_rejected(self):
        raw = pd.DataFrame({"path": ["1_1.png", "1_2.png"], "y_true": [0, 0], "y_score": [.1, .2]})
        with self.assertRaises(ValueError): standardize_prediction_dataframe(raw, experiment={"experiment_id":"e"}, threshold=.5, threshold_method="youden_validation")
    def test_24_prediction_schema_valid(self): self.assertEqual(list(self.canonical_frame().columns), PREDICTION_COLUMNS)
    def test_25_patient_sets_identical(self): self.assertEqual(len(compare_patient_sets({"a": self.canonical_frame(), "b": self.canonical_frame()})), 2)
    def test_26_patient_order_realigned(self):
        out = compare_patient_sets({"a": self.canonical_frame(), "b": self.canonical_frame().iloc[::-1]}); self.assertEqual(out["a"].patient_id.tolist(), out["b"].patient_id.tolist())
    def test_27_missing_patient_detected(self):
        with self.assertRaises(ValueError): compare_patient_sets({"a": self.canonical_frame(), "b": self.canonical_frame().iloc[:1]})
    def test_28_mammofm_predictions_reused(self):
        # 03e is now the registered test_notebook for this experiment (see section 4), so it is
        # ready to have inference run -- but the legacy predictions on disk are still unverified,
        # so it must NOT read as already having verified test predictions.
        self.skip_unless_experiments_available()
        row = self.coverage.set_index("experiment_id").loc["mammofm_03a_real_only"]
        self.assertEqual(row.status, "READY_FOR_TEST_INFERENCE"); self.assertEqual(row.provenance_level, "legacy_normalized_unverified")
        self.assertTrue(row.inference_ready); self.assertFalse(row.final_aggregation_ready)
    def test_29_aggregate_only_incomplete_for_paired(self):
        row = self.coverage.set_index("experiment_id").loc["maxvit512_02i_real_aug_synth_fromscratch"]; self.assertFalse(row.provenance_valid)
    def test_30_locked_test_has_coverage_guard(self):
        text = (ROOT / "notebooks/4_comparisons_and_test/04y_Final_Test_Locked.ipynb").read_text(); self.assertIn("Copertura test incompleta", text)
    def test_31_classifier_dry_run_no_write_branch(self):
        text = (ROOT / "notebooks/3_classifiers/02k_MaxViT512_LockedFinalTest.ipynb").read_text(); self.assertIn("if not DRY_RUN", text)
    def test_32_final_dry_run_no_write_branch(self):
        text = (ROOT / "notebooks/4_comparisons_and_test/04x_Leaderboard_Validation_All_Classifiers.ipynb").read_text(); self.assertIn("if not DRY_RUN", text)

    # Cache (33-38)
    def cache_case(self, mutation=None):
        tmp = tempfile.TemporaryDirectory(); root = Path(tmp.name); pred = root / "test_predictions.csv"; pred.write_text("x\n1\n")
        expected = {"checkpoint_signature": {"sha256": "a"}, "validation_threshold": .5, "test_csv_signature": {"sha256": "b"}, "preprocessing": {"size": 1}}
        write_prediction_manifest(root / "test_predictions.manifest.json", expected, pred)
        changed = json.loads(json.dumps(expected));
        if mutation: mutation(changed)
        return tmp, validate_prediction_cache(root / "test_predictions.manifest.json", changed, pred)
    def test_33_cache_checkpoint_change(self):
        tmp, valid = self.cache_case(lambda x: x.update(checkpoint_signature={"sha256":"z"})); self.addCleanup(tmp.cleanup); self.assertFalse(valid)
    def test_34_cache_threshold_change(self):
        tmp, valid = self.cache_case(lambda x: x.update(validation_threshold=.6)); self.addCleanup(tmp.cleanup); self.assertFalse(valid)
    def test_35_cache_test_csv_change(self):
        tmp, valid = self.cache_case(lambda x: x.update(test_csv_signature={"sha256":"z"})); self.addCleanup(tmp.cleanup); self.assertFalse(valid)
    def test_36_cache_image_change(self):
        tmp, valid = self.cache_case(lambda x: x.update(image_signatures={"x":"z"})); self.addCleanup(tmp.cleanup); self.assertFalse(valid)
    def test_37_cache_preprocessing_change(self):
        tmp, valid = self.cache_case(lambda x: x.update(preprocessing={"size":2})); self.addCleanup(tmp.cleanup); self.assertFalse(valid)
    def test_38_cache_prediction_tamper(self):
        tmp, valid = self.cache_case(); root = Path(tmp.name); (root / "test_predictions.csv").write_text("x\n2\n"); self.addCleanup(tmp.cleanup); self.assertFalse(validate_prediction_cache(root / "test_predictions.manifest.json", {"checkpoint_signature":{"sha256":"a"}, "validation_threshold":.5, "test_csv_signature":{"sha256":"b"}, "preprocessing":{"size":1}}, root / "test_predictions.csv"))

    # Statistics (39-48)
    def test_39_bootstrap_paired_samples(self):
        out = paired_stratified_bootstrap([0,0,1,1], [.1,.2,.8,.9], [.2,.3,.7,.8], iterations=100); self.assertEqual(out["valid_iterations"], 100)
    def test_40_bootstrap_direct_difference(self):
        out = paired_stratified_bootstrap([0,0,1,1], [.1,.2,.8,.9], [.9,.8,.2,.1], iterations=100); self.assertGreater(out["mean_difference"], .9)
    def test_41_bootstrap_ci_synthetic(self):
        out = paired_stratified_bootstrap([0,0,1,1], [.1,.2,.8,.9], [.9,.8,.2,.1], iterations=100); self.assertGreater(out["ci_lower"], 0)
    def test_42_delong_known_case(self): self.assertEqual(delong_roc_test([0,0,0,1,1,1],[.1,.2,.7,.6,.8,.9],[.2,.3,.4,.5,.6,.7])["status"], "ok")
    def test_43_delong_degenerate(self): self.assertEqual(delong_roc_test([0,1],[.1,.9],[.2,.8])["status"], "not_computable")
    def test_44_mcnemar_known_table(self):
        out = mcnemar_test([0]*30, [0]*20+[1]*10, [1]*5+[0]*25); self.assertEqual((out["b"], out["c"]), (5,10))
    def test_45_mcnemar_exact(self): self.assertEqual(mcnemar_test([0,0,0],[0,0,1],[0,1,0])["method"], "exact_binomial")
    def test_46_holm(self): np.testing.assert_allclose(holm_adjustment([.01,.03,.04]), [.03,.06,.06])
    def test_47_aggregate_metrics_not_prediction_frame(self):
        with self.assertRaises(ValueError): compare_patient_sets({"a": pd.DataFrame({"roc_auc":[.5]})})
    def test_48_labels_must_match(self):
        other = self.canonical_frame(); other["y_true"] = [1,0]
        with self.assertRaises(ValueError): compare_patient_sets({"a": self.canonical_frame(), "b": other})

    # Notebook integrity (49-54)
    def notebook_paths(self):
        return [ROOT / p for p in [
            "notebooks/4_comparisons_and_test/01x_ResNet50_Val_PartialVSFull.ipynb", "notebooks/4_comparisons_and_test/01y_ResNet50_Test.ipynb", "notebooks/4_comparisons_and_test/01z_ResNet50_Confronto_Configurazioni.ipynb",
            "notebooks/4_comparisons_and_test/02v_MaxViT512_Val_PartialVSFull.ipynb", "notebooks/4_comparisons_and_test/02w_MaxViT512_Val_FromScratch_AllVSPart.ipynb", "notebooks/4_comparisons_and_test/02x_MaxViT512_Test.ipynb", "notebooks/4_comparisons_and_test/02y_MaxViT512_Confronto_AugVSSynth.ipynb", "notebooks/4_comparisons_and_test/02z_MaxViT512_Confronto_FromScratchVSFineTuned.ipynb", "notebooks/4_comparisons_and_test/03z_Confronto_ResNet50_vs_MaxViT512.ipynb",
            "notebooks/3_classifiers/02k_MaxViT512_LockedFinalTest.ipynb", "notebooks/3_classifiers/03e_MammoFM_LockedFinalTest.ipynb", "notebooks/3_classifiers/04c_RADDINO_LockedFinalTest.ipynb", "notebooks/4_comparisons_and_test/04x_Leaderboard_Validation_All_Classifiers.ipynb", "notebooks/4_comparisons_and_test/04y_Final_Test_Locked.ipynb", "notebooks/4_comparisons_and_test/04z_Final_Statistical_Comparison.ipynb"]]
    def test_49_all_notebooks_nbformat_valid(self):
        for path in self.notebook_paths(): nbformat.validate(nbformat.read(path, as_version=4))
    def test_50_04x_valid(self): nbformat.validate(nbformat.read(ROOT / "notebooks/4_comparisons_and_test/04x_Leaderboard_Validation_All_Classifiers.ipynb", as_version=4))
    def test_51_04y_valid(self): nbformat.validate(nbformat.read(ROOT / "notebooks/4_comparisons_and_test/04y_Final_Test_Locked.ipynb", as_version=4))
    def test_52_04z_valid(self): nbformat.validate(nbformat.read(ROOT / "notebooks/4_comparisons_and_test/04z_Final_Statistical_Comparison.ipynb", as_version=4))
    def test_53_no_home_fede_in_new_notebooks(self):
        # 6 locked/final notebooks now (02k, 03e, 04c, 04x, 04y, 04z) since 03e was added.
        for path in self.notebook_paths()[-6:]: self.assertNotIn("/home/fede", path.read_text())
    def test_54_selection_code_has_no_test_metric(self):
        notebook = nbformat.read(ROOT / "notebooks/4_comparisons_and_test/04x_Leaderboard_Validation_All_Classifiers.ipynb", as_version=4); code = "\n".join("".join(c.source) for c in notebook.cells if c.cell_type == "code")
        self.assertNotIn("test_roc_auc", code); self.assertNotIn("test_metrics_path", code)

    def test_55_required_artifacts_exist(self):
        for relative in ["configs/final_classifier_registry.json", "results/final_evaluation/classifier_test_coverage.csv", "results/final_evaluation/classifier_test_coverage.json", "results/final_evaluation/validation_leaderboard.csv", "results/final_evaluation/validation_leaderboard.json", "results/final_evaluation/finalists_manifest.json", "results/final_evaluation/FINALISTS_LOCKED", "results/final_evaluation/test_dataset_manifest.json"]:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_56_all_json_is_strict(self):
        def reject(value): raise ValueError(value)
        paths = list((ROOT / "results/final_evaluation").rglob("*.json")) + [ROOT / "configs/final_classifier_registry.json"]
        for path in paths:
            json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
            self.assertNotRegex(path.read_text(encoding="utf-8"), r"(?:NaN|[-+]?Infinity)")

    def test_57_precision_beyond_four_decimals_controls_ranking(self):
        rows = pd.DataFrame([
            {"experiment_id":"lower", "architecture":"A", "training_dataset_variant":"real_only", "validation_roc_auc":.70001, "validation_pr_auc":.3, "scientifically_eligible":True},
            {"experiment_id":"higher", "architecture":"A", "training_dataset_variant":"real_only", "validation_roc_auc":.70009, "validation_pr_auc":.3, "scientifically_eligible":True},
        ])
        self.assertEqual(select_validation_finalists(rows, {"max_finalists_total":1})[0]["experiment_id"], "higher")

    def test_58_mandatory_finalists_cannot_be_truncated(self):
        rows = pd.DataFrame([{"experiment_id":f"e{i}", "architecture":"A", "training_dataset_variant":"real_only", "validation_roc_auc":.8-i*.01, "validation_pr_auc":.3, "scientifically_eligible":True, "required_for_primary_comparison":True} for i in range(3)])
        with self.assertRaisesRegex(RuntimeError, "mandatory finalists exceed"): select_validation_finalists(rows, {"max_finalists_total":2})

    def test_59_blocked_scientific_lock_is_incomplete(self):
        locked = lock_finalists_manifest([{"experiment_id":"x", "operationally_ready":False, "blocked_reason":"checkpoint"}], {})
        self.assertFalse(locked["selection_complete"])
        with self.assertRaises(RuntimeError): validate_locked_finalists_manifest(locked)

    def test_60_test_manifest_canonical_counts(self):
        payload = json.loads((ROOT / "results/final_evaluation/test_dataset_manifest.json").read_text())
        self.assertEqual((payload["n_patients"], payload["n_positive"], payload["n_negative"]), (438, 73, 365))

    def test_61_finalists_marker_matches_signature(self):
        payload = json.loads((ROOT / "results/final_evaluation/finalists_manifest.json").read_text())
        self.assertEqual((ROOT / "results/final_evaluation/FINALISTS_LOCKED").read_text().strip(), payload["lock_signature"]["sha256"])

    def test_62_registry_evidence_and_separation(self):
        self.assertTrue(all("scientifically_eligible" in row and "operationally_ready" in row and "validation_metrics_source" in row for row in self.registry))
        resnet = next(row for row in self.registry if row["experiment_id"] == "resnet50_01b_real_synth_partial")
        self.assertTrue(resnet["scientifically_eligible"]); self.assertFalse(resnet["operationally_ready"])

    def test_63_stats_notebook_declares_complete_outputs(self):
        text = (ROOT / "notebooks/4_comparisons_and_test/04z_Final_Statistical_Comparison.ipynb").read_text()
        for token in ["pr_auc", "calibration_curves.png", "confusion_matrices.png", "bootstrap_auc_differences.png", "Brier", "ECE"]:
            self.assertIn(token, text)

    def test_64_full_precision_threshold_reconstructed_from_validation(self):
        row = next(item for item in self.registry if item["experiment_id"] == "maxvit512_02c_real_synth_full")
        self.assertEqual(row["validation_threshold"], 0.2750791)
        self.assertEqual(row["validation_threshold_source"]["source_precision"], "full_recomputed")

    def test_65_incompatible_resnet_reconstruction_not_promoted(self):
        row = next(item for item in self.registry if item["experiment_id"] == "resnet50_01b_real_synth_partial")
        self.assertEqual(row["validation_threshold"], 0.2689)
        self.assertIn("validation_reconstruction_mismatch", row)

    # Scientific vs. operational separation (66-70)
    def test_66_scientific_validation_passes_incomplete_lock(self):
        locked = lock_finalists_manifest([{
            "experiment_id": "x", "scientifically_eligible": True, "selected_by_validation": True,
            "final_aggregation_ready": False, "blocked_reason": "checkpoint",
        }], {})
        self.assertFalse(locked["final_aggregation_complete"])
        validated = validate_locked_finalists_manifest(locked, require_operational_complete=False)
        self.assertTrue(validated["locked"])
        with self.assertRaises(RuntimeError):
            validate_locked_finalists_manifest(locked, require_operational_complete=True)

    def test_67_readiness_tiers_independently_correct(self):
        # Split in two: the Mammo-FM half needs the real checkpoint (experiments/, excluded from
        # the light audit package) to reach inference_ready=True; the ResNet half holds regardless
        # since "no checkpoint at all" is true both in the real repo and the light package.
        self.skip_unless_experiments_available()
        rows = self.coverage.set_index("experiment_id")
        for eid in ["mammofm_03a_real_only", "mammofm_03b_real_synth_finetuned", "mammofm_03d_real_augmented"]:
            self.assertTrue(rows.loc[eid, "inference_ready"], eid)
            self.assertFalse(rows.loc[eid, "final_aggregation_ready"], eid)

    def test_67b_resnet_readiness_tiers_blocked_regardless_of_package(self):
        rows = self.coverage.set_index("experiment_id")
        for eid in ["resnet50_01a_real_only", "resnet50_01b_real_synth_partial"]:
            self.assertFalse(rows.loc[eid, "inference_ready"], eid)
            self.assertFalse(rows.loc[eid, "final_aggregation_ready"], eid)

    def test_68_manifest_completeness_fields_on_real_lock(self):
        manifest = json.loads((ROOT / "results/final_evaluation/finalists_manifest.json").read_text())
        self.assertTrue(manifest["scientific_selection_complete"])
        self.assertFalse(manifest["final_aggregation_complete"])
        self.assertEqual(manifest["selection_complete"], manifest["final_aggregation_complete"])

    def test_69_build_locked_finalist_entries_missing_checkpoint_no_crash(self):
        entries = build_locked_finalist_entries([{
            "experiment_id": "resnet50_01a_real_only",
            "architecture": "ResNet-50",
            "checkpoint_path": "experiments/classifiers/resnet50/01a_real_only/baseline_resnet50_final_best.keras",
            "scientifically_eligible": True, "selected_by_validation": True, "operationally_ready": False,
            "inference_ready": False, "test_predictions_ready": False, "final_aggregation_ready": False,
            "blocked_reason": "checkpoint missing", "status": "INCOMPLETE", "validation_threshold_method": "youden_validation",
        }], ROOT)
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0]["checkpoint_signature"])
        self.assertEqual(entries[0]["test_status"], "INCOMPLETE")
        self.assertEqual(entries[0]["threshold_method"], "youden_validation")

    def test_70_build_locked_finalist_entries_preserves_full_field_set(self):
        item = {
            "experiment_id": "e", "architecture": "MaxViT-512", "selection_reason": ["baseline_per_architecture"], "checkpoint_path": None,
            "validation_threshold": .5, "validation_threshold_method": "youden_validation", "status": "INCOMPLETE",
            "test_notebook": None, "test_predictions_path": None, "test_predictions_manifest_path": None,
            "scientifically_eligible": True, "selected_by_validation": True, "operationally_ready": False,
            "inference_ready": False, "test_predictions_ready": False, "final_aggregation_ready": False,
            "blocked_reason": "checkpoint_missing", "validation_metrics_source": {"path": "x"},
            "validation_threshold_source": {"path": "y"}, "provenance_level": "invalid",
        }
        entry = build_locked_finalist_entries([item], ROOT)[0]
        for field in ["scientifically_eligible", "selected_by_validation", "operationally_ready", "inference_ready",
                      "test_predictions_ready", "final_aggregation_ready", "blocked_reason",
                      "validation_metrics_source", "validation_threshold_source", "provenance_level"]:
            self.assertIn(field, entry, field)
        self.assertEqual(entry["validation_metrics_source"], {"path": "x"})

    # Finalist policy at the production max_finalists_total=10 (71-73)
    def production_policy(self):
        return {"include_baseline_per_architecture": True, "include_best_synthetic_per_architecture": True,
                "include_best_augmented_per_architecture": True, "include_best_overall_per_architecture": True,
                "max_finalists_total": 10}

    def test_71_baseline_resnet_and_maxvit_survive_production_policy(self):
        selected_ids = {x["experiment_id"] for x in select_validation_finalists(self.coverage, self.production_policy())}
        self.assertIn("resnet50_01a_real_only", selected_ids)
        self.assertIn("maxvit512_02a_real_only", selected_ids)

    def test_72_all_ten_production_finalists_present(self):
        selected_ids = {x["experiment_id"] for x in select_validation_finalists(self.coverage, self.production_policy())}
        self.assertEqual(selected_ids, {
            "resnet50_01a_real_only", "resnet50_01b_real_synth_partial",
            "maxvit512_02a_real_only", "maxvit512_02c_real_synth_full", "maxvit512_02j_real_aug_synth_finetuned",
            "mammofm_03a_real_only", "mammofm_03b_real_synth_finetuned", "mammofm_03d_real_augmented",
            "raddino_04a_real_only", "raddino_04b_real_synth",
        })

    def test_73_primary_comparison_pair_present_together(self):
        selected_ids = {x["experiment_id"] for x in select_validation_finalists(self.coverage, self.production_policy())}
        self.assertTrue({"raddino_04a_real_only", "raddino_04b_real_synth"} <= selected_ids)

    def test_74_architecture_family_exhaustive(self):
        architectures = {x["architecture"] for x in self.registry}
        self.assertEqual(architectures, set(ARCHITECTURE_FAMILY.keys()))
        self.assertEqual(set(ARCHITECTURE_FAMILY.values()), {"resnet50", "maxvit512", "mammofm", "raddino"})

    # Mammo-FM strict checkpoint loading (75-77)
    def test_75_unwrap_checkpoint_state_dict_variants(self):
        raw = {"a.weight": 1, "b.bias": 2}
        self.assertEqual(unwrap_checkpoint_state_dict(raw), raw)
        self.assertEqual(unwrap_checkpoint_state_dict({"state_dict": raw}), raw)
        self.assertEqual(unwrap_checkpoint_state_dict({"model_state_dict": raw}), raw)
        prefixed = {f"module.{k}": v for k, v in raw.items()}
        self.assertEqual(unwrap_checkpoint_state_dict(prefixed), raw)
        partially_prefixed = {"module.a.weight": 1, "b.bias": 2}
        self.assertEqual(unwrap_checkpoint_state_dict(partially_prefixed), partially_prefixed)

    def test_76_checkpoint_key_mismatch_allowlist(self):
        mismatch = checkpoint_key_mismatch(["a", "b"], ["c"], allowed_mismatches=["a"])
        self.assertEqual(mismatch, {"unexplained_missing": ["b"], "unexplained_unexpected": ["c"]})
        clean = checkpoint_key_mismatch([], [])
        self.assertEqual(clean, {"unexplained_missing": [], "unexplained_unexpected": []})

    def test_77_mammofm_strict_load_via_tiny_module(self):
        import torch
        model = torch.nn.Linear(4, 1)
        matching = {k: v.clone() for k, v in model.state_dict().items()}
        for candidate in (matching, {"state_dict": matching}, {"model_state_dict": matching},
                          {f"module.{k}": v for k, v in matching.items()}):
            missing, unexpected = model.load_state_dict(unwrap_checkpoint_state_dict(candidate), strict=False)
            mismatch = checkpoint_key_mismatch(missing, unexpected)
            self.assertEqual(mismatch, {"unexplained_missing": [], "unexplained_unexpected": []})
        incompatible = {"totally.wrong.key": torch.zeros(1)}
        missing, unexpected = model.load_state_dict(unwrap_checkpoint_state_dict(incompatible), strict=False)
        mismatch = checkpoint_key_mismatch(missing, unexpected)
        self.assertTrue(mismatch["unexplained_missing"] or mismatch["unexplained_unexpected"])

    def test_78_p_bootstrap_matches_documented_two_tailed_formula(self):
        # p_bootstrap must be reproducible from the documented seeded-resampling formula
        # (2 * min(P(diff<=0), P(diff>=0)), continuity-corrected) -- not merely derived from
        # whether zero falls inside/outside the CI.
        from sklearn.metrics import roc_auc_score
        seed, iterations = 7, 300
        y = [0, 0, 1, 1, 0, 1, 0, 1]
        sa = [.1, .2, .8, .9, .15, .85, .25, .95]
        sb = [.9, .8, .2, .1, .85, .15, .75, .05]
        result = paired_stratified_bootstrap(y, sa, sb, "roc_auc", iterations=iterations, seed=seed)
        y_arr, a_arr, b_arr = np.asarray(y), np.asarray(sa), np.asarray(sb)
        pos, neg = np.flatnonzero(y_arr == 1), np.flatnonzero(y_arr == 0)
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(iterations):
            idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
            diffs.append(float(roc_auc_score(y_arr[idx], a_arr[idx]) - roc_auc_score(y_arr[idx], b_arr[idx])))
        values = np.asarray(diffs)
        expected_p = min(1.0, 2 * min((np.sum(values <= 0) + 1) / (len(values) + 1), (np.sum(values >= 0) + 1) / (len(values) + 1)))
        self.assertEqual(result["valid_iterations"], iterations)
        self.assertAlmostEqual(result["p_bootstrap"], expected_p, places=12)
        self.assertGreater(result["ci_lower"], 0)  # confirms this is a non-trivial, one-sided case

    # Notebook regressions (78-82)
    def _exec_first_code_cell(self, relative_path):
        notebook = nbformat.read(ROOT / relative_path, as_version=4)
        cell = next(c for c in notebook.cells if c.cell_type == "code")
        namespace = {}
        exec(compile(cell.source, relative_path, "exec"), namespace)  # noqa: S102
        return namespace

    def test_79_03e_valid(self):
        nbformat.validate(nbformat.read(ROOT / "notebooks/3_classifiers/03e_MammoFM_LockedFinalTest.ipynb", as_version=4))

    def test_80_02k_selects_maxvit_ready_members_despite_other_blockers(self):
        namespace = self._exec_first_code_cell("notebooks/3_classifiers/02k_MaxViT512_LockedFinalTest.ipynb")
        self.assertEqual(set(namespace["EXPERIMENT_IDS"]),
                          {"maxvit512_02a_real_only", "maxvit512_02c_real_synth_full", "maxvit512_02j_real_aug_synth_finetuned"})

    def test_81_04c_selects_raddino_ready_members_despite_other_blockers(self):
        namespace = self._exec_first_code_cell("notebooks/3_classifiers/04c_RADDINO_LockedFinalTest.ipynb")
        self.assertEqual(set(namespace["EXPERIMENT_IDS"]), {"raddino_04a_real_only", "raddino_04b_real_synth"})

    def test_82_03e_selects_mammofm_ready_members_despite_provenance_blocker(self):
        namespace = self._exec_first_code_cell("notebooks/3_classifiers/03e_MammoFM_LockedFinalTest.ipynb")
        self.assertEqual(set(namespace["EXPERIMENT_IDS"]),
                          {"mammofm_03a_real_only", "mammofm_03b_real_synth_finetuned", "mammofm_03d_real_augmented"})

    def test_83_04x_uses_strict_json_dumps_for_leaderboard(self):
        text = (ROOT / "notebooks/4_comparisons_and_test/04x_Leaderboard_Validation_All_Classifiers.ipynb").read_text()
        self.assertIn("strict_json_dumps", text)
        self.assertNotIn('json.dumps(leaderboard', text)

    def test_84_04y_uses_canonical_prediction_paths(self):
        text = (ROOT / "notebooks/4_comparisons_and_test/04y_Final_Test_Locked.ipynb").read_text()
        self.assertIn("canonical_test_prediction_paths", text)
        self.assertNotIn('PROJECT_ROOT / "results/classifiers" / family', text)

    def test_85_04z_verifies_central_manifest_before_reading_csv(self):
        text = (ROOT / "notebooks/4_comparisons_and_test/04z_Final_Statistical_Comparison.ipynb").read_text()
        for token in ["prediction_file_signature", "finalists_lock_signature", "test_dataset_manifest_signature",
                      "compare_patient_sets(frames, canonical_ids)"]:
            self.assertIn(token, text)

    def test_86_locked_registry_entries_have_canonical_prediction_paths(self):
        locked_ids = {
            "resnet50_01a_real_only", "resnet50_01b_real_synth_partial",
            "maxvit512_02a_real_only", "maxvit512_02c_real_synth_full",
            "maxvit512_02j_real_aug_synth_finetuned", "mammofm_03a_real_only",
            "mammofm_03b_real_synth_finetuned", "mammofm_03d_real_augmented",
            "raddino_04a_real_only", "raddino_04b_real_synth",
        }
        for experiment in self.registry:
            if experiment["experiment_id"] in locked_ids:
                self.assertEqual(
                    {key: experiment[key] for key in ("test_predictions_path", "test_predictions_manifest_path")},
                    canonical_test_prediction_paths(experiment),
                )

    def test_87_ready_to_complete_end_to_end_and_content_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"; checkpoint.write_bytes(b"checkpoint-v1")
            validation = root / "validation.json"; validation.write_text('{"validation_threshold": 0.5}')
            image = root / "data/image.bin"; image.parent.mkdir(parents=True); image.write_bytes(b"AAAA")
            test_csv = root / "data/test.csv"
            pd.DataFrame([{"patient_id": "p1", "image_id": "i1", "label": 1,
                           "processed_path": "data/image.bin"}]).to_csv(test_csv, index=False)
            experiment = {
                "experiment_id": "maxvit512_synthetic_e2e", "architecture": "MaxViT-512",
                "training_dataset_variant": "real_only", "training_mode": "partial_finetuning",
                "synthetic_source": "none", "checkpoint_path": "checkpoint.pt",
                "validation_metrics_path": "validation.json", "validation_threshold": 0.5,
                "validation_threshold_method": "youden_validation", "test_csv": "data/test.csv",
                "test_notebook": "02k.ipynb", "required_for_final_pipeline": True,
                "eligible_for_final_selection": True, "scientifically_eligible": True,
                "selected_by_validation": True, "validation_roc_auc": 0.8, "validation_pr_auc": 0.7,
            }
            experiment.update(canonical_test_prediction_paths(experiment))
            registry = build_experiment_registry([experiment])
            dataset_manifest = build_test_dataset_manifest(test_csv, project_root=root,
                                                           preprocessing={"resolution": 512})
            dataset_path = root / "results/final_evaluation/test_dataset_manifest.json"
            dataset_path.parent.mkdir(parents=True)
            dataset_path.write_text(strict_json_dumps(dataset_manifest, indent=2) + "\n")

            initial = build_test_coverage_table(registry, root).iloc[0]
            self.assertEqual(initial.status, "READY_FOR_TEST_INFERENCE")
            self.assertTrue(initial.inference_ready)
            self.assertFalse(initial.test_predictions_ready)
            self.assertFalse(initial.final_aggregation_ready)

            paths = canonical_test_prediction_paths(experiment)
            prediction_path = root / paths["test_predictions_path"]
            manifest_path = root / paths["test_predictions_manifest_path"]
            prediction_path.parent.mkdir(parents=True)
            pd.DataFrame([{"patient_id": "p1", "y_true": 1, "y_score": 0.9}]).to_csv(prediction_path, index=False)

            def write_compatible_manifest(current_experiment):
                manifest = {
                    "experiment_id": current_experiment["experiment_id"],
                    "checkpoint_signature": content_signature(checkpoint),
                    "validation_metrics_signature": content_signature(validation),
                    "validation_threshold": current_experiment["validation_threshold"],
                    "threshold_method": current_experiment["validation_threshold_method"],
                    "test_dataset_manifest_signature": content_signature(dataset_path),
                    "patient_ids_hash": dataset_manifest["patient_ids_hash"],
                    "preprocessing": {"resolution": 512}, "model_config": {"model": "fixture"},
                    "pipeline_schema_version": 1, "provenance_level": "verified_recomputed",
                    "test_used_for_selection": False,
                }
                write_prediction_manifest(manifest_path, manifest, prediction_path)

            write_compatible_manifest(experiment)
            completed = build_test_coverage_table(registry, root).iloc[0]
            self.assertTrue(completed.test_predictions_ready)
            self.assertTrue(completed.final_aggregation_ready)
            self.assertIsNone(completed.blocked_reason)
            entries = build_locked_finalist_entries([completed.to_dict()], root)
            lock = lock_finalists_manifest(entries, {"fixture": True})
            self.assertTrue(lock["final_aggregation_complete"])
            self.assertNotIn(experiment["experiment_id"], {x["experiment_id"] for x in lock["operational_blockers"]})

            checkpoint.write_bytes(b"checkpoint-v2")
            self.assertFalse(build_test_coverage_table(registry, root).iloc[0].final_aggregation_ready)
            write_compatible_manifest(experiment)
            changed_threshold = dict(experiment, validation_threshold=0.6)
            changed_registry = build_experiment_registry([changed_threshold])
            self.assertFalse(build_test_coverage_table(changed_registry, root).iloc[0].final_aggregation_ready)
            write_compatible_manifest(changed_threshold)
            timestamp = image.stat().st_mtime_ns
            image.write_bytes(b"BBBB")
            os.utime(image, ns=(timestamp, timestamp))
            self.assertEqual(image.stat().st_size, 4)
            self.assertFalse(build_test_coverage_table(changed_registry, root).iloc[0].final_aggregation_ready)

    def test_88_readiness_invariants_hold_for_every_registry_row(self):
        for row in self.coverage.itertuples():
            self.assertFalse(row.final_aggregation_ready and not row.test_predictions_ready)
            self.assertFalse(row.status == "TEST_PREDICTIONS_AVAILABLE" and not row.test_predictions_ready)
            self.assertFalse(row.status == "READY_FOR_TEST_INFERENCE" and not row.inference_ready)

    def test_89_holm_families_are_metric_and_scope_specific(self):
        source = (ROOT / "notebooks/utility/create_final_classifier_notebooks.py").read_text()
        self.assertIn('family = "primary" if ga and ga == gb else "secondary"', source)
        self.assertIn('comparisons["holm_family"]', source)
        self.assertIn('groupby("holm_family")', source)

    def test_90_generated_notebook_cell_ids_are_deterministic(self):
        source = (ROOT / "notebooks/utility/create_final_classifier_notebooks.py").read_text()
        self.assertIn('hashlib.sha256(stable_key).hexdigest()[:12]', source)
        for path in self.notebook_paths()[-6:]:
            notebook = nbformat.read(path, as_version=4)
            self.assertEqual(len(notebook.cells), len({cell.id for cell in notebook.cells}))

    def test_91_bootstrap_pr_auc_is_seed_reproducible(self):
        args = ([0, 0, 1, 1, 0, 1], [.1, .3, .8, .9, .2, .7], [.2, .4, .7, .8, .3, .6])
        first = paired_stratified_bootstrap(*args, metric="pr_auc", iterations=200, seed=19)
        second = paired_stratified_bootstrap(*args, metric="pr_auc", iterations=200, seed=19)
        self.assertEqual(first, second)

    def test_92_delong_equal_auc_is_explicitly_not_computable(self):
        result = delong_roc_test([0, 0, 1, 1], [.1, .2, .8, .9], [.1, .2, .8, .9])
        self.assertEqual(result["status"], "not_computable")
        self.assertEqual(result["difference"], 0.0)

    def test_93_delong_finite_when_variance_is_positive(self):
        result = delong_roc_test([0, 0, 0, 1, 1, 1], [.1, .2, .7, .6, .8, .9], [.2, .3, .4, .5, .6, .7])
        self.assertEqual(result["status"], "ok")
        self.assertTrue(np.isfinite(result["p_value"]))

    def test_94_mcnemar_large_discordance_uses_corrected_chi_square(self):
        y = np.zeros(40, dtype=int)
        a = np.r_[np.zeros(30, dtype=int), np.ones(10, dtype=int)]
        b = np.r_[np.ones(20, dtype=int), np.zeros(20, dtype=int)]
        result = mcnemar_test(y, a, b)
        self.assertEqual((result["b"], result["c"]), (20, 10))
        self.assertEqual(result["method"], "chi_square_continuity_corrected")
        self.assertAlmostEqual(result["statistic"], (abs(20 - 10) - 1) ** 2 / 30)

    def test_95_artifact_builder_never_overwrites_canonical_predictions(self):
        import build_final_classifier_artifacts as builder
        experiment = {
            "experiment_id": "mammofm_fixture", "architecture": "Mammo-FM",
            "legacy_test_predictions_path": "legacy/test_predictions.csv",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = canonical_test_prediction_paths(experiment)
            prediction = root / paths["test_predictions_path"]
            manifest = root / paths["test_predictions_manifest_path"]
            prediction.parent.mkdir(parents=True)
            prediction.write_text("verified-predictions")
            manifest.write_text('{"provenance_level":"verified_recomputed"}')
            with patch.object(builder, "ROOT", root):
                builder.normalize_mammofm_predictions(experiment)
            self.assertEqual(prediction.read_text(), "verified-predictions")
            self.assertEqual(manifest.read_text(), '{"provenance_level":"verified_recomputed"}')


if __name__ == "__main__":
    unittest.main()
