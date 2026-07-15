"""Content-aware selection validation: silent post-selection edits must be rejected."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))
import downstream_protocol as dp  # noqa: E402

BENCHMARK = "results/publication_v2/generator_benchmark"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _write_manifest_csv(path: Path, count: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "relative_path"])
        for index in range(count):
            writer.writerow([f"s{index:05d}", f"data/synthetic/g/positive/pos_{index:05d}.png"])
    return path


def build_valid_tree(tmp: Path, *, finetuned_count: int = 1361, fromscratch_count: int = 1361,
                     fromscratch_rank: str = "1", fromscratch_eligible: str = "True") -> Path:
    generators = {"finetuned": ("GF", "finetuned"), "from_scratch": ("GS", "from_scratch")}
    registry = {"generators": []}
    identity = {}
    provenance_hashes = {}
    for family, (gid, fam) in generators.items():
        count = finetuned_count if family == "finetuned" else fromscratch_count
        manifest_rel = f"results/publication_v2/generator_provenance/runtime/{gid}/filtered_samples.csv"
        manifest_path = _write_manifest_csv(tmp / manifest_rel, count)
        manifest_sha = _sha(manifest_path)
        model_sha = hashlib.sha256(f"model-{gid}".encode()).hexdigest()
        gen_sha = hashlib.sha256(f"gen-{gid}".encode()).hexdigest()
        provenance_rel = f"configs/provenance/{gid}.json"
        _write(tmp / provenance_rel, json.dumps({
            "model_identity_sha256": model_sha, "generation_identity_sha256": gen_sha,
            "filtered_sample_manifest": manifest_rel,
            "manifest_sha256": {"filtered_samples": manifest_sha}}))
        registry["generators"].append({"id": gid, "scientific_family": fam,
                                       "eligible_for_downstream_selection": True,
                                       "provenance_manifest": provenance_rel})
        identity[family] = {"generator_id": gid, "family": fam, "descriptive_family_rank": 1,
                            "primary_metric": "raddino_kid", "primary_metric_value": 0.1,
                            "model_identity_sha256": model_sha, "generation_identity_sha256": gen_sha,
                            "filtered_manifest_path": manifest_rel, "filtered_manifest_sha256": manifest_sha,
                            "filtered_image_count": count}
        provenance_hashes[gid] = (model_sha, gen_sha, manifest_sha)
    _write(tmp / "configs/generator_registry.json", json.dumps(registry))

    amendment_path = _write(tmp / "configs/generator_benchmark_protocol_amendment_v1.json",
                            json.dumps({"amendment_version": "v1", "selected_policy": "B"}))
    summary_path = _write(tmp / f"{BENCHMARK}/generator_summary_corrected.csv", "generator_id,condition\nGF,FILTERED\n")
    gate_path = tmp / f"{BENCHMARK}/gate_audit/amended_gate_results.csv"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    with gate_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["full_generator_id", "amended_safety_gate_eligible",
                                                    "descriptive_family_rank"])
        writer.writeheader()
        writer.writerow({"full_generator_id": "GF", "amended_safety_gate_eligible": "True",
                         "descriptive_family_rank": "1"})
        writer.writerow({"full_generator_id": "GS", "amended_safety_gate_eligible": fromscratch_eligible,
                         "descriptive_family_rank": fromscratch_rank})

    evidence = {
        "active_amendment_sha256": _sha(amendment_path), "test_access": False,
        "selected_generators": {"finetuned": "GF", "from_scratch": "GS"},
        "amended_safety_gate_results": [
            {"generator_id": "GF", "amended_safety_gate_eligible": True},
            {"generator_id": "GS", "amended_safety_gate_eligible": fromscratch_eligible in (True, "True")}],
        "descriptive_family_ranking": {
            "finetuned": [{"generator_id": "GF", "descriptive_family_rank": 1}],
            "from_scratch": [{"generator_id": "GS", "descriptive_family_rank": int(fromscratch_rank)}]}}
    evidence_path = _write(tmp / "configs/generator_selection_evidence_v1.json", json.dumps(evidence))
    payload = {
        "finetuned": "GF", "from_scratch": "GS", "schema_version": 2, "primary_metric": "raddino_kid",
        "benchmark_HEAD": "abc", "benchmark_run_id": "run",
        "benchmark_summary_path": f"{BENCHMARK}/generator_summary_corrected.csv",
        "benchmark_summary_sha256": _sha(summary_path),
        "amended_gate_results_path": f"{BENCHMARK}/gate_audit/amended_gate_results.csv",
        "amended_gate_results_sha256": _sha(gate_path),
        "active_amendment": "configs/generator_benchmark_protocol_amendment_v1.json",
        "active_amendment_sha256": _sha(amendment_path),
        "selection_evidence_path": "configs/generator_selection_evidence_v1.json",
        "selection_evidence_sha256": _sha(evidence_path),
        "original_protocol_result": {"eligible_under_original_gates": 0},
        "post_benchmark_amendment": True, "test_access": False, "selection_notes": "n",
        "selection_identity": identity}
    _write(tmp / "configs/selected_generators.json", json.dumps(payload))
    return tmp


class SilentModificationTests(unittest.TestCase):
    def _payload(self, root: Path) -> dict:
        return json.loads((root / "configs/selected_generators.json").read_text())

    def test_valid_tree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_valid_tree(Path(tmp))
            dp.validate_selection_content(root, self._payload(root))  # must not raise

    def test_amendment_content_changed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_valid_tree(Path(tmp))
            (root / "configs/generator_benchmark_protocol_amendment_v1.json").write_text('{"tampered": true}')
            with self.assertRaises(ValueError):
                dp.validate_selection_content(root, self._payload(root))

    def test_summary_changed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_valid_tree(Path(tmp))
            (root / f"{BENCHMARK}/generator_summary_corrected.csv").write_text("generator_id,condition\nX,Y\n")
            with self.assertRaises(ValueError):
                dp.validate_selection_content(root, self._payload(root))

    def test_gate_results_changed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_valid_tree(Path(tmp))
            (root / f"{BENCHMARK}/gate_audit/amended_gate_results.csv").write_text("full_generator_id\nGF\n")
            with self.assertRaises(ValueError):
                dp.validate_selection_content(root, self._payload(root))

    def test_provenance_identity_changed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_valid_tree(Path(tmp))
            prov_path = root / "configs/provenance/GF.json"
            prov = json.loads(prov_path.read_text())
            prov["model_identity_sha256"] = "deadbeef"
            prov_path.write_text(json.dumps(prov))
            with self.assertRaises(ValueError):
                dp.validate_selection_content(root, self._payload(root))

    def test_manifest_same_path_different_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_valid_tree(Path(tmp))
            manifest = root / "results/publication_v2/generator_provenance/runtime/GF/filtered_samples.csv"
            # Same path, different CSV content -> file hash no longer matches provenance/selection record.
            with manifest.open("a") as stream:
                stream.write("s99999,data/synthetic/g/positive/pos_99999.png\n")
            with self.assertRaises(ValueError):
                dp.validate_selection_content(root, self._payload(root))

    def test_wrong_filtered_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_valid_tree(Path(tmp), finetuned_count=1360)
            with self.assertRaises(ValueError):
                dp.validate_selection_content(root, self._payload(root))
        with tempfile.TemporaryDirectory() as tmp:
            root = build_valid_tree(Path(tmp), fromscratch_count=1362)
            with self.assertRaises(ValueError):
                dp.validate_selection_content(root, self._payload(root))

    def test_safety_eligible_but_rank_two_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_valid_tree(Path(tmp), fromscratch_rank="2")
            with self.assertRaises(ValueError):
                dp.validate_selection_content(root, self._payload(root))


def _build_portable_tree(tmp: Path, *, evidence_eligible: bool = True, evidence_rank: int = 1) -> Path:
    """A clean-archive-style tree: only committed configs, no runtime CSVs or provenance."""
    root = Path(tmp)
    (root / "configs").mkdir(parents=True)
    (root / "configs/generator_registry.json").write_text(json.dumps({"generators": [
        {"id": "GF", "scientific_family": "finetuned", "eligible_for_downstream_selection": True},
        {"id": "GS", "scientific_family": "from_scratch", "eligible_for_downstream_selection": True}]}))
    amendment_path = _write(root / "configs/generator_benchmark_protocol_amendment_v1.json",
                            json.dumps({"selected_policy": "B", "status": "approved_post_benchmark"}))
    amendment_sha = _sha(amendment_path)
    evidence = {
        "active_amendment_sha256": amendment_sha, "test_access": False,
        "selected_generators": {"finetuned": "GF", "from_scratch": "GS"},
        "amended_safety_gate_results": [
            {"generator_id": "GF", "amended_safety_gate_eligible": evidence_eligible},
            {"generator_id": "GS", "amended_safety_gate_eligible": True}],
        "descriptive_family_ranking": {
            "finetuned": [{"generator_id": "GF", "descriptive_family_rank": evidence_rank}],
            "from_scratch": [{"generator_id": "GS", "descriptive_family_rank": 1}]}}
    evidence_path = _write(root / "configs/generator_selection_evidence_v1.json", json.dumps(evidence))
    _write(root / "configs/selected_generators.json", json.dumps({
        "finetuned": "GF", "from_scratch": "GS", "schema_version": 2, "test_access": False,
        "active_amendment": "configs/generator_benchmark_protocol_amendment_v1.json",
        "active_amendment_sha256": amendment_sha,
        "selection_evidence_path": "configs/generator_selection_evidence_v1.json",
        "selection_evidence_sha256": _sha(evidence_path),
        "selection_identity": {
            "finetuned": {"generator_id": "GF", "descriptive_family_rank": 1},
            "from_scratch": {"generator_id": "GS", "descriptive_family_rank": 1}}}))
    return root


class PortableDecisionTests(unittest.TestCase):
    def _payload(self, root: Path) -> dict:
        return json.loads((root / "configs/selected_generators.json").read_text())

    def test_decision_validates_from_committed_evidence_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_portable_tree(Path(tmp))
            # No runtime CSVs or provenance in this tree: the decision must still validate.
            dp.validate_selection_decision(root, self._payload(root))

    def test_tampered_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_portable_tree(Path(tmp))
            (root / "configs/generator_selection_evidence_v1.json").write_text('{"tampered": true}')
            with self.assertRaises(ValueError):
                dp.validate_selection_decision(root, self._payload(root))

    def test_evidence_not_safety_eligible_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_portable_tree(Path(tmp), evidence_eligible=False)
            with self.assertRaises(ValueError):
                dp.validate_selection_decision(root, self._payload(root))

    def test_evidence_rank_not_one_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_portable_tree(Path(tmp), evidence_rank=2)
            with self.assertRaises(ValueError):
                dp.validate_selection_decision(root, self._payload(root))

    def test_real_repository_decision_is_portable(self):
        # The committed real selection + evidence + amendment validate without runtime CSVs.
        selection = ROOT / "configs/selected_generators.json"
        evidence = ROOT / "configs/generator_selection_evidence_v1.json"
        if not (selection.is_file() and evidence.is_file()):
            self.skipTest("committed selection/evidence not present")
        dp.validate_selection_decision(ROOT, json.loads(selection.read_text()))


class RealConditionsNeedNoSelectionTests(unittest.TestCase):
    def test_real_only_and_real_augmented_pass_without_selection_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Provide only the downstream protocol; no selected_generators.json at all.
            shutil_src = ROOT / "configs/downstream_classifier_protocol.json"
            (tmp / "configs").mkdir(parents=True)
            (tmp / "configs/downstream_classifier_protocol.json").write_text(shutil_src.read_text())
            self.assertFalse((tmp / "configs/selected_generators.json").exists())
            for condition in ("real_only", "real_augmented"):
                resolved = dp.resolve_condition(tmp, condition)
                self.assertEqual(resolved["synthetic_count_by_class"], {})
                self.assertIsNone(resolved["synthetic_generator_id"])


if __name__ == "__main__":
    unittest.main()
