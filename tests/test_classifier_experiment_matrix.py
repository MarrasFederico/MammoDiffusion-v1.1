from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_classifier_experiment_matrix as build_matrix  # noqa: E402
import resume_classifier_experiment_matrix as resume_matrix  # noqa: E402
import status_classifier_experiment_matrix as status_matrix  # noqa: E402
import classifier_checkpoint_io as ckio  # noqa: E402
import classifier_run_manifest as crm  # noqa: E402


def build_fixture_project(root: Path) -> None:
    (root / "configs").mkdir(parents=True)
    (root / "configs/dataset_variant_registry.json").write_text(json.dumps({"variants": [
        {"dataset_variant_id": "R", "regime": "base", "status": "ready"},
        {"dataset_variant_id": "RSB_CONTROLLED_gA", "regime": "stage1_screening", "status": "ready"},
        {"dataset_variant_id": "BROKEN", "regime": "stage1_screening", "status": "invalid"},
    ]}))
    (root / "configs/classifier_training_protocols.json").write_text(json.dumps({"policies": {
        "maxvit512": {"framework": "pytorch_timm", "resource_profile_by_phase": {"full_run": "heavy"}},
        "resnet50": {"framework": "tensorflow_keras", "resource_profile_by_phase": {"head_training": "light", "fine_tuning": "medium"}},
    }}))


class BuildMatrixTests(unittest.TestCase):
    def test_stage2_build_persists_variants_for_runner_resolution(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            with patch.object(build_matrix, "build_stage2_variants", return_value=[{
                "dataset_variant_id": "RAS_CONTROLLED_gA", "regime": "stage2_advanced", "status": "ready"
            }]):
                build_matrix.build_and_write(root, stage=2, selected_union=["gA"])
            registry = json.loads((root / "configs/dataset_variant_registry.json").read_text())
            self.assertIn("RAS_CONTROLLED_gA", {v["dataset_variant_id"] for v in registry["variants"]})
    def test_stage1_builds_one_job_per_architecture_variant_seed(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            jobs = build_matrix.build_jobs(root, stage=1)
            # 2 architectures x 2 ready/legacy variants x 3 seeds = 12 (BROKEN excluded)
            self.assertEqual(len(jobs), 12)
            self.assertTrue(all(j["dataset_variant_id"] != "BROKEN" for j in jobs))

    def test_seeds_are_exactly_17_42_73(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            jobs = build_matrix.build_jobs(root, stage=1)
            seeds = {j["seed"] for j in jobs if j["dataset_variant_id"] == "R" and j["architecture"] == "maxvit512"}
            self.assertEqual(seeds, {17, 42, 73})

    def test_heavy_profile_gets_5060_only_eligibility(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            jobs = build_matrix.build_jobs(root, stage=1)
            maxvit_job = next(j for j in jobs if j["architecture"] == "maxvit512")
            self.assertEqual(maxvit_job["resource_profile"], "heavy")
            self.assertEqual(maxvit_job["gpu_eligibility"], ["rtx_5060_ti_16gb"])

    def test_medium_profile_gets_both_gpus_eligible(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            jobs = build_matrix.build_jobs(root, stage=1)
            resnet_job = next(j for j in jobs if j["architecture"] == "resnet50")
            self.assertEqual(resnet_job["resource_profile"], "medium")
            self.assertIn("rtx_3060_12gb", resnet_job["gpu_eligibility"])

    def test_stage2_refuses_without_selected_union(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            with self.assertRaises(ValueError):
                build_matrix.build_jobs(root, stage=2, selected_union=None)

    def test_experiment_ids_are_globally_unique(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            jobs = build_matrix.build_jobs(root, stage=1)
            ids = [j["experiment_id"] for j in jobs]
            self.assertEqual(len(ids), len(set(ids)))

    def test_rebuild_preserves_reconstructed_status_not_blind_pending(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            jobs = build_matrix.build_jobs(root, stage=1)
            job = next(j for j in jobs if j["dataset_variant_id"] == "R" and j["architecture"] == "maxvit512" and j["seed"] == 17)
            run = root / job["manifest_path"]; run = run.parent
            run.mkdir(parents=True)
            (run / "model.pt").write_bytes(b"weights")
            ckio.write_checkpoint_metadata(run, architecture="maxvit512", dataset_variant_id="R", training_policy="p",
                                            seed=17, checkpoint=run / "model.pt", dataset_manifest_sha256="x", protocol_signature="y")
            jobs2 = build_matrix.build_jobs(root, stage=1)
            job2 = next(j for j in jobs2 if j["dataset_variant_id"] == "R" and j["architecture"] == "maxvit512" and j["seed"] == 17)
            self.assertEqual(job2["status"], "TRAINED")

    def test_build_and_write_only_touches_its_own_stage(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            payload1 = build_matrix.build_and_write(root, stage=1)
            n_stage1 = len(payload1["jobs"])
            # simulate stage 2 by writing directly (bypassing the union requirement, this is a
            # unit test of build_and_write's merge behavior, not of the stage-2 gate)
            union_path = root / "configs/dataset_variant_registry.json"
            registry = json.loads(union_path.read_text())
            registry["variants"].append({"dataset_variant_id": "05_ldm_basic_fromscratch", "regime": "base",
                                          "status": "ready", "classes": ["positive"]})
            union_path.write_text(json.dumps(registry))

            def fake_stage2(root_, union):
                return [{"dataset_variant_id": "RAS_FULL_gA", "status": "ready"}]

            original = build_matrix.build_stage2_variants
            build_matrix.build_stage2_variants = fake_stage2
            try:
                payload2 = build_matrix.build_and_write(root, stage=2, selected_union=["gA"])
            finally:
                build_matrix.build_stage2_variants = original
            stage1_jobs_after = [j for j in payload2["jobs"] if j["stage"] == 1]
            self.assertEqual(len(stage1_jobs_after), n_stage1)


class ResumeAndStatusTests(unittest.TestCase):
    def test_resume_reports_state_transitions(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            build_matrix.build_and_write(root, stage=1)
            matrix = json.loads((root / "configs/classifier_experiment_matrix.json").read_text())
            job = matrix["jobs"][0]
            run = root / job["manifest_path"]; run = run.parent
            run.mkdir(parents=True, exist_ok=True)
            (run / "model.pt").write_bytes(b"weights") if "model.pt" == ckio.checkpoint_path(run, "pytorch_timm").name else (run / "model.keras").write_bytes(b"weights")
            protocols = json.loads((root / "configs/classifier_training_protocols.json").read_text())["policies"]
            framework = protocols[job["architecture"]]["framework"]
            ckio.write_checkpoint_metadata(run, architecture=job["architecture"], dataset_variant_id=job["dataset_variant_id"],
                                            training_policy=job["training_policy"], seed=job["seed"],
                                            checkpoint=ckio.checkpoint_path(run, framework), dataset_manifest_sha256="x", protocol_signature="y")
            result = resume_matrix.resume(root, stage=1)
            self.assertTrue(any(c["experiment_id"] == job["experiment_id"] and c["to"] == "TRAINED" for c in result["changed"]))

    def test_status_report_groups_by_status_and_architecture(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            build_matrix.build_and_write(root, stage=1)
            report = status_matrix.build_report(root, stage=1)
            self.assertEqual(report["total"], 12)
            self.assertEqual(sum(report["by_status"].values()), 12)
            self.assertEqual(sum(report["by_architecture"].values()), 12)

    def test_status_report_empty_when_no_matrix_built_yet(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build_fixture_project(root)
            report = status_matrix.build_report(root)
            self.assertEqual(report["jobs"], [])


if __name__ == "__main__":
    unittest.main()
