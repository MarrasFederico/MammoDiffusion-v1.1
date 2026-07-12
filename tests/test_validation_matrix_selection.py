from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import finalize_validation_stage as fvs  # noqa: E402


def write_matrix_and_metrics(root: Path, jobs_with_metrics: list[tuple[dict, dict | None]]) -> None:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    jobs = []
    for job, metrics in jobs_with_metrics:
        run = root / "experiments/classifiers_matrix" / job["architecture"] / job["dataset_variant_id"] / "p" / f"seed_{job['seed']}"
        run.mkdir(parents=True, exist_ok=True)
        job = {**job, "validation_predictions_path": str((run / "validation_predictions.json").relative_to(root))}
        if metrics is not None:
            (run / "validation_metrics.json").write_text(json.dumps(metrics))
        jobs.append(job)
    (root / "configs/classifier_experiment_matrix.json").write_text(json.dumps({"schema_version": 1, "jobs": jobs}))


def job(architecture, variant, seed, status="VALIDATED"):
    return {"experiment_id": f"{architecture}__{variant}__seed{seed}", "stage": 1, "architecture": architecture,
            "dataset_variant_id": variant, "seed": seed, "status": status}


def write_ensemble(root: Path, architecture: str, variant: str, pr_auc: float, roc_auc: float) -> Path:
    path = root / "results/classifiers_matrix" / architecture / variant / f"{architecture}_standard" / "ensemble_validation_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"architecture": architecture, "dataset_variant_id": variant,
                                "seeds": [17, 42, 73], "test_access": False,
                                "metrics": {"pr_auc": pr_auc, "roc_auc": roc_auc},
                                "signature": f"{architecture}-{variant}-{pr_auc}"}))
    return path


class LoadCompletedValidationsTests(unittest.TestCase):
    def test_only_complete_three_seed_controlled_ensembles_are_loaded(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            write_matrix_and_metrics(root, [])
            write_ensemble(root, "maxvit512", "RSB_CONTROLLED_gA", 0.8, 0.9)
            write_ensemble(root, "maxvit512", "RSB_FULL_gA", 0.99, 0.99)
            rows = fvs.load_completed_validations(root, stage=1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["aggregation"], "ensemble")

    def test_never_reads_test_metrics_field(self):
        # regression guard: the loader must only ever look at validation_metrics.json,
        # and must not accept a differently-named "test_metrics.json" as a substitute.
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            write_matrix_and_metrics(root, [(job("maxvit512", "R", 17), None)])
            run = root / "experiments/classifiers_matrix/maxvit512/R/p/seed_17"
            (run / "test_metrics.json").write_text(json.dumps({"pr_auc": 1.0}))
            rows = fvs.load_completed_validations(root, stage=1)
            self.assertEqual(len(rows), 0)


class RankingTests(unittest.TestCase):
    def test_generator_extracted_from_variant_id_prefix(self):
        self.assertEqual(fvs._generator_of("RSB_CONTROLLED_02_sd21_filtered_100steps"), "02_sd21_filtered_100steps")
        self.assertIsNone(fvs._generator_of("RSP_FULL_05_ldm_basic_fromscratch"))
        self.assertIsNone(fvs._generator_of("RSB_FULL_02_sd21_filtered_100steps"))
        self.assertIsNone(fvs._generator_of("R"))
        self.assertIsNone(fvs._generator_of("RA"))

    def test_ranking_uses_pr_auc_as_primary_key(self):
        rows = [
            {"architecture": "maxvit512", "dataset_variant_id": "RSB_CONTROLLED_gA", "pr_auc": 0.70, "roc_auc": 0.99},
            {"architecture": "maxvit512", "dataset_variant_id": "RSB_CONTROLLED_gB", "pr_auc": 0.90, "roc_auc": 0.50},
        ]
        ranking = fvs.rank_by_generator(rows, "maxvit512")
        self.assertEqual(ranking[0]["generator_id"], "gB")  # higher PR-AUC wins despite lower ROC-AUC

    def test_ranking_does_not_mix_full_or_positive_regimes(self):
        rows = [
            {"architecture": "maxvit512", "dataset_variant_id": "RSB_CONTROLLED_gA", "pr_auc": 0.6, "roc_auc": 0.6,
             "aggregation": "ensemble"},
            {"architecture": "maxvit512", "dataset_variant_id": "RSB_FULL_gB", "pr_auc": 1.0, "roc_auc": 1.0},
            {"architecture": "maxvit512", "dataset_variant_id": "RSP_CONTROLLED_gC", "pr_auc": 1.0, "roc_auc": 1.0},
        ]
        ranking = fvs.rank_by_generator(rows, "maxvit512")
        self.assertEqual([row["generator_id"] for row in ranking], ["gA"])
        self.assertEqual(ranking[0]["aggregation"], "ensemble")


class SelectedGeneratorUnionTests(unittest.TestCase):
    def test_union_includes_family_winners_even_outside_global_top_k(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            rows = []
            # gA wins 3 of 4 families comfortably; gD is worst everywhere except it uniquely
            # wins raddino by a hair -> must still enter the union even though it's not top-3 globally.
            data = {
                "resnet50": {"gA": 0.9, "gB": 0.5, "gC": 0.4, "gD": 0.1},
                "maxvit512": {"gA": 0.9, "gB": 0.5, "gC": 0.4, "gD": 0.1},
                "mammofm": {"gA": 0.9, "gB": 0.5, "gC": 0.4, "gD": 0.1},
                "raddino": {"gA": 0.2, "gB": 0.3, "gC": 0.4, "gD": 0.5},
            }
            for arch, gens in data.items():
                for gid, score in gens.items():
                    rows.append((arch, gid, score))
            write_matrix_and_metrics(root, [])
            for arch, gid, score in rows:
                write_ensemble(root, arch, f"RSB_CONTROLLED_{gid}", score, score)
            payload = fvs.compute_selected_generator_union(root, stage=1)
            self.assertIn("gD", payload["selected_generator_union"])
            self.assertIn("gA", payload["selected_generator_union"])

    def test_union_never_reads_test_data_flag(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            write_matrix_and_metrics(root, [])
            write_ensemble(root, "maxvit512", "RSB_CONTROLLED_gA", 0.8, 0.9)
            payload = fvs.compute_selected_generator_union(root, stage=1)
            self.assertFalse(payload["selection_used_test_data"])

    def test_union_is_empty_and_honest_when_nothing_completed(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            write_matrix_and_metrics(root, [(job("maxvit512", "RSB_CONTROLLED_gA", 17, status="PENDING"), None)])
            payload = fvs.compute_selected_generator_union(root, stage=1)
            self.assertEqual(payload["selected_generator_union"], [])
            self.assertEqual(payload["n_completed_jobs_considered"], 0)

    def test_signature_changes_if_any_input_row_changes(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            write_matrix_and_metrics(root, [])
            write_ensemble(root, "maxvit512", "RSB_CONTROLLED_gA", 0.8, 0.9)
            p1 = fvs.compute_selected_generator_union(root, stage=1)
            write_ensemble(root, "maxvit512", "RSB_CONTROLLED_gA", 0.5, 0.9)
            p2 = fvs.compute_selected_generator_union(root, stage=1)
            self.assertNotEqual(p1["signature"], p2["signature"])

    def test_write_selected_union_persists_to_canonical_path(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            write_matrix_and_metrics(root, [])
            write_ensemble(root, "maxvit512", "RSB_CONTROLLED_gA", 0.8, 0.9)
            payload = fvs.compute_selected_generator_union(root, stage=1)
            out = fvs.write_selected_union(root, payload)
            self.assertEqual(out, root / "results/generator_comparison/selected_generator_union.json")
            self.assertEqual(json.loads(out.read_text())["signature"], payload["signature"])


if __name__ == "__main__":
    unittest.main()
