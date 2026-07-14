"""Lightweight regression tests for multi-GPU generation planning (no model loading)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
import ast
import argparse
import importlib
import importlib.util
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "utility"))

from parallel_generation_utils import (  # noqa: E402
    SD_SEED_OFFSETS,
    SD_SEED_STRATEGY,
    GENERATED_PNG_PATTERN,
    acquire_parallel_generation_lock,
    copy_validated_sd_evaluation_images,
    checkpoint_content_signature,
    create_parallel_run_dir,
    file_content_signature,
    final_sd_generation_plan,
    filtered_selection_cache_matches,
    exact_filtered_png_paths,
    ldm_raw_png_paths,
    metric_image_paths,
    missing_named_png_indices,
    partition_indices,
    prepare_sd_manifest,
    print_gpu_resolution_dry_run,
    png_content_signature,
    readable_png_paths,
    release_parallel_generation_lock,
    resolve_generation_gpu_devices,
    run_dynamic_gpu_jobs,
    run_sd_generation_jobs,
    sd_seed,
    sd_metrics_cache_config,
    sd_metrics_cache_compatible,
    sd_metrics_cache_entry_matches,
    sd_base_model_signature,
    worker_log_path,
    DEFAULT_GENERATION_RESERVATION_SIZE,
    DEFAULT_GENERATION_SCHEDULER,
    claim_index,
    claim_next_chunk,
    complete_claimed_chunk,
    dynamic_chunks,
    prepare_dynamic_queue,
    release_index_claim,
    validate_queue_indices,
)
import sd_generation_worker  # noqa: E402
import sd_vae_utils  # noqa: E402
from run_adaptive_filter import candidate_png_paths  # noqa: E402
import generate_ldm_v2  # noqa: E402
import evaluate_ldm_v2  # noqa: E402
import evaluate_filtered_ldm_v2  # noqa: E402


def png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (2, 2), color=1).save(path)


class ParallelGenerationTests(unittest.TestCase):
    def test_final_plan_accepts_global_slot_layout_and_preserves_surplus(self):
        with tempfile.TemporaryDirectory() as t:
            directory = Path(t)
            for index in range(2):
                png(directory / f"eval_neg_{index:04d}.png")
            (directory / ".evaluation_reuse.json").write_text(json.dumps({"files": [{}, {}]}))
            for index in range(2, 5):
                png(directory / f"gen_{index:04d}.png")
            plan = final_sd_generation_plan(directory, target_total=5, reused_prefix="eval_neg")
            self.assertTrue(plan["complete"])
            self.assertEqual(plan["gen_index_layout"], "global_target_slots")
            self.assertEqual(plan["excluded_surplus_gen_files"], [])
    def test_dynamic_reservation_defaults_and_last_chunk(self):
        self.assertEqual(DEFAULT_GENERATION_SCHEDULER, "dynamic_reservations")
        self.assertEqual(DEFAULT_GENERATION_RESERVATION_SIZE, 4)
        self.assertEqual(dynamic_chunks(range(11), 4), [[0,1,2,3], [4,5,6,7], [8,9,10]])

    def test_dynamic_queue_handles_arbitrary_holes_not_parity(self):
        valid = [0, 2, 4, 5, 8, 10, 11, 14, 18]
        missing = sorted(set(range(20)) - set(valid))
        self.assertEqual(dynamic_chunks(validate_queue_indices(missing, 20, valid), 4), [[1,3,6,7], [9,12,13,15], [16,17,19]])

    def test_dynamic_queue_rejects_duplicates_range_and_valid(self):
        with self.assertRaises(ValueError): validate_queue_indices([1,1], 3)
        with self.assertRaises(ValueError): validate_queue_indices([3], 3)
        with self.assertRaises(ValueError): validate_queue_indices([1], 3, [1])

    def test_seed_is_identical_across_reservation_sizes(self):
        expected = {index: sd_seed(42, "final_new", "positive", index) for index in range(17)}
        for size in (1, 4, 8):
            ordered = [index for chunk in dynamic_chunks(range(17), size) for index in chunk]
            self.assertEqual({index: sd_seed(42, "final_new", "positive", index) for index in ordered}, expected)

    def test_dynamic_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / "queue"
            prepare_dynamic_queue(queue, [1,3], target_count=4, valid_indices=[0,2], output_dir=Path(tmp), dry_run=True)
            self.assertFalse(queue.exists())

    def test_chunk_and_index_claims_are_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / "queue"
            prepare_dynamic_queue(queue, [0,1,2,3], target_count=4, output_dir=Path(tmp))
            first = claim_next_chunk(queue, "a"); self.assertIsNotNone(first)
            self.assertIsNone(claim_next_chunk(queue, "b"))
            reservation = claim_index(queue, 0, 0, "a", "0")
            self.assertIsNotNone(reservation); self.assertIsNone(claim_index(queue, 0, 0, "b", "1"))
            release_index_claim(reservation); complete_claimed_chunk(queue, *first)

    def test_stale_reservation_recovered_but_live_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / "queue"; prepare_dynamic_queue(queue, [0], target_count=1, output_dir=Path(tmp))
            stale = queue / "reservations/index_000000.claim"; stale.write_text(json.dumps({"pid": 99999999}))
            claimed = claim_index(queue, 0, 0, "new", "0"); self.assertIsNotNone(claimed); release_index_claim(claimed)
            live = claim_index(queue, 0, 0, "live", "0"); self.assertIsNotNone(live)
            self.assertIsNone(claim_index(queue, 0, 0, "other", "1")); release_index_claim(live)

    def test_mock_fast_worker_claims_more_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / "queue"; prepare_dynamic_queue(queue, range(40), target_count=40, output_dir=Path(tmp), reservation_size=4)
            completed, stats, guard = [], {"fast":0, "slow":0}, threading.Lock()
            def worker(name, delay):
                while True:
                    claimed = claim_next_chunk(queue, name)
                    if claimed is None: return
                    path, chunk = claimed
                    for index in chunk["indices"]:
                        reservation = claim_index(queue, index, chunk["chunk_id"], name, name)
                        self.assertIsNotNone(reservation); time.sleep(delay)
                        with guard: completed.append(index); stats[name] += 1
                        release_index_claim(reservation)
                    complete_claimed_chunk(queue, path, chunk)
            threads = [threading.Thread(target=worker, args=("fast", .001)), threading.Thread(target=worker, args=("slow", .005))]
            [thread.start() for thread in threads]; [thread.join() for thread in threads]
            self.assertEqual(sorted(completed), list(range(40))); self.assertEqual(len(completed), len(set(completed)))
            self.assertGreater(stats["fast"], stats["slow"])
            self.assertFalse(list((queue / "reservations").glob("*.claim")))
    def test_final_total_counts_reused_and_gen_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            png(directory / "eval_neg_0000.png")
            png(directory / "eval_neg_0001.png")
            png(directory / "gen_0000.png")
            plan = final_sd_generation_plan(directory, target_total=5, reused_prefix="eval_neg")
            self.assertEqual(plan["n_valid_reused"], 2)
            self.assertEqual(plan["n_new_required"], 3)
            self.assertEqual(plan["missing_gen_indices"], [1, 2])
            self.assertEqual(2 + 1 + len(plan["missing_gen_indices"]), 5)

    def test_holes_corruption_tmp_and_foreign_png_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            png(directory / "gen_0000.png")
            (directory / "gen_0001.png").write_bytes(b"not a png")
            png(directory / ".tmp_gen_0001_1234.png")
            png(directory / "foreign.png")
            self.assertEqual(missing_named_png_indices(directory, 3), [1, 2])

    def test_seed_namespaces_are_disjoint_and_worker_independent(self) -> None:
        namespaces = [
            {sd_seed(123, phase, cls, i) for i in range(10_000)}
            for phase, cls in (("evaluation", "negative"), ("evaluation", "positive"), ("final_new", "negative"), ("final_new", "positive"))
        ]
        for left in range(len(namespaces)):
            for right in range(left + 1, len(namespaces)):
                self.assertFalse(namespaces[left] & namespaces[right])
        self.assertEqual(sd_seed(123, "final_new", "positive", 77), 123 + SD_SEED_OFFSETS["final_new:positive"] + 77)

    def test_partition_and_logs_do_not_overlap(self) -> None:
        shards = partition_indices([8, 1, 3, 1, 7], 2)
        self.assertEqual(sorted(value for shard in shards for value in shard), [1, 3, 7, 8])
        root = Path("/tmp/log-test")
        self.assertNotEqual(
            worker_log_path(root, "final_negative_worker_0", "0"),
            worker_log_path(root, "final_positive_worker_0", "0"),
        )

    def test_seed_manifest_rejects_legacy_pngs_and_records_v2(self) -> None:
        request = {
            "out_dir": "",
            "seed": 99,
            "phase": "final_new",
            "class_name": "negative",
            "class_offset": SD_SEED_OFFSETS["final_new:negative"],
            "prompt": "prompt",
            "inference_steps": 10,
            "guidance_scale": 1.5,
            "resolution": 512,
        }
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            request["out_dir"] = str(directory)
            png(directory / "gen_0000.png")
            with self.assertRaises(RuntimeError):
                prepare_sd_manifest(request, "checkpoint", False)
            prepare_sd_manifest(request, "checkpoint", True)
            manifest = __import__("json").loads((directory / ".generation_manifest.json").read_text())
            self.assertEqual(manifest["seed_strategy"], "mixed_legacy_and_per_image_seed_v2")
            self.assertTrue(manifest["legacy_seed_mix"])
            self.assertEqual(manifest["legacy_files"], ["gen_0000.png"])
            self.assertEqual(manifest["class_offset"], SD_SEED_OFFSETS["final_new:negative"])

    @staticmethod
    def _request(out_dir: Path, phase: str = "evaluation") -> dict:
        class_name = "negative"
        return {
            "out_dir": str(out_dir), "seed": 99, "phase": phase,
            "class_name": class_name,
            "class_offset": SD_SEED_OFFSETS[f"{phase}:{class_name}"],
            "prompt": "prompt", "inference_steps": 10,
            "guidance_scale": 1.5, "resolution": 512,
            "count": 3,
            "checkpoint_type": "full_unet",
            "base_model_dir": str((out_dir.parent / "base_model").resolve()),
        }

    @staticmethod
    def _copy_kwargs(root: Path, **overrides) -> dict:
        values = {
            "base_seed": 99,
            "num_inference_steps": 10,
            "guidance_scale": 1.5,
            "resolution": 512,
            "checkpoint_type": "full_unet",
            "base_model_dir": root / "base_model",
        }
        values.update(overrides)
        return values

    def test_evaluation_v2_copy_then_final_manifest_and_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation = Path(tmp) / "evaluation"
            final = Path(tmp) / "final"
            evaluation.mkdir()
            prepare_sd_manifest(self._request(evaluation), "checkpoint", False)
            png(evaluation / "gen_0000.png")
            png(evaluation / "gen_0001.png")
            copy_validated_sd_evaluation_images(
                evaluation, final, 2, "eval_neg", "checkpoint", "negative", "prompt", False,
                **self._copy_kwargs(Path(tmp)),
            )
            final_request = self._request(final, "final_new")
            prepare_sd_manifest(final_request, "checkpoint", False)
            manifest = json.loads((final / ".generation_manifest.json").read_text())
            plan = final_sd_generation_plan(final, 5, "eval_neg")
            self.assertEqual(manifest["seed_strategy"], SD_SEED_STRATEGY)
            self.assertEqual(manifest["groups"]["evaluation_reused"]["source_directory"], str(evaluation.resolve()))
            self.assertEqual(manifest["groups"]["final_new"]["class_offset"], SD_SEED_OFFSETS["final_new:negative"])
            self.assertEqual(plan["missing_gen_indices"], [0, 1, 2])

    def test_legacy_evaluation_rejected_and_override_is_marked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation = Path(tmp) / "evaluation"
            final = Path(tmp) / "final"
            evaluation.mkdir()
            png(evaluation / "gen_0000.png")
            with self.assertRaises(RuntimeError):
                copy_validated_sd_evaluation_images(
                    evaluation, final, 1, "eval_neg", "checkpoint", "negative", "prompt", False,
                    **self._copy_kwargs(Path(tmp)),
                )
            copy_validated_sd_evaluation_images(
                evaluation, final, 1, "eval_neg", "checkpoint", "negative", "prompt", True,
                **self._copy_kwargs(Path(tmp)),
            )
            request = self._request(final, "final_new")
            prepare_sd_manifest(request, "checkpoint", True)
            manifest = json.loads((final / ".generation_manifest.json").read_text())
            self.assertTrue(manifest["legacy_seed_mix"])
            self.assertIn("eval_neg_0000.png", manifest["legacy_files"])
            self.assertEqual(manifest["seed_strategy"], "mixed_legacy_and_per_image_seed_v2")

    def test_raw_count_equal_target_does_not_skip_corrupt_or_foreign(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            png(directory / "eval_neg_0000.png")
            (directory / "gen_0000.png").write_bytes(b"corrupt")
            png(directory / "foreign.png")
            self.assertEqual(len(list(directory.glob("*.png"))), 3)
            plan = final_sd_generation_plan(directory, 3, "eval_neg")
            self.assertFalse(plan["complete"])
            self.assertIn("gen_0000.png", plan["corrupt_files"])
            self.assertIn("foreign.png", plan["extra_files"])

    def test_all_to_test_command_is_accepted_by_both_real_parsers(self) -> None:
        paths = SimpleNamespace(
            project_root=ROOT, experiment_dir=ROOT / "experiments" / "test",
            synthetic_raw_dir=ROOT / "raw", synthetic_filtered_dir=ROOT / "filtered",
        )
        with patch.object(sys, "argv", ["generate_ldm_v2.py", "--mode", "all"]):
            parent = generate_ldm_v2.parse_args()
        test_child = generate_ldm_v2.child_command(parent, paths, "test")
        with patch.object(sys, "argv", [test_child[1], *test_child[2:]]):
            child_args = generate_ldm_v2.parse_args()
        filtered = paths.synthetic_filtered_dir
        metrics_command = generate_ldm_v2.evaluate_filtered_command(child_args, paths, filtered)
        with patch.object(sys, "argv", [metrics_command[1], *metrics_command[2:]]):
            parsed = evaluate_filtered_ldm_v2.parse_args()
        self.assertEqual(parsed.target_label, parent.target_label)
        self.assertNotIn("--generation-gpus", metrics_command)
        self.assertNotIn("--max-generation-workers", metrics_command)

    def test_generate_child_and_worker_preserve_gpu_configuration(self) -> None:
        paths = SimpleNamespace(project_root=ROOT, experiment_dir=ROOT / "experiment")
        for devices, maximum in (("1", 1), ("0,1", 1), ("off", None), ("auto", None)):
            with patch.object(sys, "argv", ["generate_ldm_v2.py"]):
                args = generate_ldm_v2.parse_args()
            args.generation_gpus = devices
            args.max_generation_workers = maximum
            child = generate_ldm_v2.child_command(args, paths, "generate")
            self.assertEqual(child[child.index("--generation-gpus") + 1], devices)
            if maximum is not None:
                self.assertEqual(child[child.index("--max-generation-workers") + 1], str(maximum))
            worker = generate_ldm_v2.generation_child_command(args, paths, Path("indices.json"), "7")
            self.assertEqual(worker[worker.index("--generation-gpus") + 1], "off")
            self.assertEqual(worker[worker.index("--gpu-visible-devices") + 1], "7")
            if maximum is not None:
                self.assertEqual(worker[worker.index("--max-generation-workers") + 1], str(maximum))

    def test_metric_and_filter_lists_exclude_tmp_corrupt_and_foreign(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            png(directory / "gen_0000.png")
            png(directory / ".tmp_gen_0001_123.png")
            png(directory / "foreign.png")
            (directory / "gen_0002.png").write_bytes(b"corrupt")
            metric_names = [path.name for path in metric_image_paths(directory, (".png",))]
            filter_names = [path.name for path in candidate_png_paths(directory)]
            self.assertEqual(metric_names, ["gen_0000.png"])
            self.assertEqual(filter_names, ["gen_0000.png"])

    def _complete_evaluation(self, root: Path, count: int = 3) -> tuple[Path, dict]:
        evaluation = root / "evaluation"
        evaluation.mkdir()
        request = self._request(evaluation)
        request["count"] = count
        prepare_sd_manifest(request, "checkpoint", False)
        for index in range(count):
            png(evaluation / f"gen_{index:04d}.png")
        return evaluation, request

    def test_complete_evaluation_rejects_incompatible_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluation, request = self._complete_evaluation(Path(tmp))
            changed = {**request, "inference_steps": 100, "guidance_scale": 7.5}
            with self.assertRaisesRegex(RuntimeError, "Incompatible generation manifest"):
                prepare_sd_manifest(changed, "checkpoint", False)

    def test_complete_evaluation_without_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            for index in range(3):
                png(evaluation / f"gen_{index:04d}.png")
            with self.assertRaises(RuntimeError):
                prepare_sd_manifest(self._request(evaluation), "checkpoint", False)

    def test_complete_final_without_lineage_manifests_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "final"
            final.mkdir()
            for index in range(2):
                png(final / f"eval_neg_{index:04d}.png")
            for index in range(3):
                png(final / f"gen_{index:04d}.png")
            self.assertTrue(final_sd_generation_plan(final, 5, "eval_neg")["complete"])
            with self.assertRaises(RuntimeError):
                prepare_sd_manifest(self._request(final, "final_new"), "checkpoint", False)

    def test_complete_final_with_compatible_manifest_can_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation, _ = self._complete_evaluation(root, count=2)
            final = root / "final"
            copy_validated_sd_evaluation_images(
                evaluation, final, 2, "eval_neg", "checkpoint", "negative", "prompt", False,
                **self._copy_kwargs(root),
            )
            final_request = self._request(final, "final_new")
            prepare_sd_manifest(final_request, "checkpoint", False)
            for index in range(3):
                png(final / f"gen_{index:04d}.png")
            prepare_sd_manifest(final_request, "checkpoint", False)
            self.assertTrue(final_sd_generation_plan(final, 5, "eval_neg")["complete"])

    def test_missing_reused_image_is_reported_and_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation, _ = self._complete_evaluation(root)
            final = root / "final"
            kwargs = self._copy_kwargs(root)
            copy_validated_sd_evaluation_images(evaluation, final, 3, "eval_neg", "checkpoint", "negative", "prompt", False, **kwargs)
            (final / "eval_neg_0001.png").unlink()
            self.assertEqual(final_sd_generation_plan(final, 5, "eval_neg")["missing_reused_indices"], [1])
            copy_validated_sd_evaluation_images(evaluation, final, 3, "eval_neg", "checkpoint", "negative", "prompt", False, **kwargs)
            self.assertEqual(final_sd_generation_plan(final, 5, "eval_neg")["missing_reused_indices"], [])

    def test_corrupt_reused_image_is_reported_and_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation, _ = self._complete_evaluation(root)
            final = root / "final"
            kwargs = self._copy_kwargs(root)
            copy_validated_sd_evaluation_images(evaluation, final, 3, "eval_neg", "checkpoint", "negative", "prompt", False, **kwargs)
            (final / "eval_neg_0001.png").write_bytes(b"corrupt")
            self.assertEqual(final_sd_generation_plan(final, 5, "eval_neg")["corrupt_reused_indices"], [1])
            copy_validated_sd_evaluation_images(evaluation, final, 3, "eval_neg", "checkpoint", "negative", "prompt", False, **kwargs)
            self.assertEqual(final_sd_generation_plan(final, 5, "eval_neg")["corrupt_reused_indices"], [])

    def _assert_evaluation_mismatch(self, **overrides) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation, _ = self._complete_evaluation(root)
            with self.assertRaises(RuntimeError):
                copy_validated_sd_evaluation_images(
                    evaluation, root / "final", 3, "eval_neg", "checkpoint", "negative", "prompt", False,
                    **self._copy_kwargs(root, **overrides),
                )

    def test_evaluation_seed_mismatch_is_rejected(self) -> None:
        self._assert_evaluation_mismatch(base_seed=100)

    def test_evaluation_guidance_mismatch_is_rejected(self) -> None:
        self._assert_evaluation_mismatch(guidance_scale=7.5)

    def test_evaluation_steps_mismatch_is_rejected(self) -> None:
        self._assert_evaluation_mismatch(num_inference_steps=100)

    def test_evaluation_resolution_mismatch_is_rejected(self) -> None:
        self._assert_evaluation_mismatch(resolution=1024)

    def test_evaluation_base_model_or_vae_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._assert_evaluation_mismatch(base_model_dir=Path(tmp) / "other_model")

    def test_metrics_cache_is_invalidated_when_png_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.keras"
            checkpoint.write_bytes(b"checkpoint")
            negative, positive = root / "negative", root / "positive"
            png(negative / "gen_0000.png")
            png(positive / "gen_0000.png")
            entry = {
                "checkpoint_signature": checkpoint_content_signature(checkpoint),
                "negative_image_signature": png_content_signature(negative),
                "positive_image_signature": png_content_signature(positive),
                "metrics": {"FID": 1.0},
            }
            self.assertTrue(sd_metrics_cache_entry_matches(entry, checkpoint, negative, positive))
            Image.new("L", (3, 3), color=2).save(negative / "gen_0000.png")
            self.assertFalse(sd_metrics_cache_entry_matches(entry, checkpoint, negative, positive))

    def test_metrics_cache_is_invalidated_when_guidance_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation = root / "validation.csv"
            validation.write_text("label,path\n", encoding="utf-8")
            base = root / "base_model"
            config = sd_metrics_cache_config(eval_seed=42, inference_steps=100, guidance_scale=1.0, resolution=512, n_gen_per_class=100, checkpoint_type="full_unet", base_model_dir=base)
            payload = {"schema_version": 2, "config": config, "validation_csv_signature": file_content_signature(validation), "checkpoints": {}}
            self.assertTrue(sd_metrics_cache_compatible(payload, config, validation))
            changed = sd_metrics_cache_config(eval_seed=42, inference_steps=100, guidance_scale=7.5, resolution=512, n_gen_per_class=100, checkpoint_type="full_unet", base_model_dir=base)
            self.assertFalse(sd_metrics_cache_compatible(payload, changed, validation))

    def test_parallel_run_directories_never_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = create_parallel_run_dir(Path(tmp) / "logs")
            second = create_parallel_run_dir(Path(tmp) / "logs")
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir() and second.is_dir())

    def test_ldm_log_labels_include_experiment_class_phase_worker_and_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = SimpleNamespace(project_root=ROOT, experiment_dir=root / "experiment_x", logs_dir=root / "logs")
            labels = []
            def capture(**kwargs):
                labels.extend((job["label"], gpu) for job, gpu in zip(kwargs["jobs"], kwargs["devices"]))
            for target_label in (0, 1):
                with patch.object(sys, "argv", ["generate_ldm_v2.py"]):
                    args = generate_ldm_v2.parse_args()
                args.target_label = target_label
                args.generation_gpus = "0"
                args.max_generation_workers = 1
                args.dry_run = True
                with patch.object(generate_ldm_v2, "run_dynamic_gpu_jobs", side_effect=capture):
                    generate_ldm_v2.orchestrate_parallel_raw_generation(args, paths, root / "raw", [0])
            self.assertNotEqual(labels[0][0], labels[1][0])
            for label, gpu in labels:
                self.assertIn("experiment_x", label)
                self.assertIn("raw_generation", label)
                self.assertIn("worker_0", label)
                self.assertEqual(gpu, "0")

    def test_generation_info_uses_shared_v2_strategy(self) -> None:
        for name in (
            "01_SD21_Baseline_50steps.ipynb", "02_SD21_Filtered_100steps.ipynb",
            "03_SD21_VAE_FineTuned.ipynb", "04_SD21_LoRA.ipynb",
        ):
            notebook = json.loads((ROOT / "notebooks" / "2_diffusers" / name).read_text(encoding="utf-8"))
            final_cell = next(
                "".join(cell.get("source", [])) for cell in notebook["cells"]
                if "# Generazione finale con il miglior checkpoint" in "".join(cell.get("source", []))
            )
            compile(final_cell, name, "exec")
            self.assertIn('"seed_strategy": SD_SEED_STRATEGY', final_cell)
            self.assertNotIn('"seed_strategy": "per_image_seed_v1"', final_cell)
            self.assertIn("copy_validated_sd_evaluation_images", final_cell)
            self.assertIn("_sd_parallel_generate_final", final_cell)
            self.assertNotIn('if _final_plan(cfg)["complete"]:', final_cell)
            self.assertNotIn('count_pngs(cfg["final_dir"]) ==', final_cell)

    def test_notebook_evaluation_validates_manifest_before_missing_scan(self) -> None:
        for name in (
            "01_SD21_Baseline_50steps.ipynb", "02_SD21_Filtered_100steps.ipynb",
            "03_SD21_VAE_FineTuned.ipynb", "04_SD21_LoRA.ipynb",
        ):
            notebook = json.loads((ROOT / "notebooks" / "2_diffusers" / name).read_text(encoding="utf-8"))
            cell = next("".join(item.get("source", [])) for item in notebook["cells"] if "def _sd_parallel_generate_checkpoint_jobs" in "".join(item.get("source", [])))
            compile(cell, name, "exec")
            function_source = cell[cell.index("def _sd_parallel_generate_checkpoint_jobs"):cell.index("def _sd_parallel_generate_final")]
            self.assertLess(function_source.index("prepare_sd_manifest(request"), function_source.index("missing = missing_named_png_indices"))
            final_function = cell[cell.index("def _sd_parallel_generate_final"):]
            self.assertLess(final_function.index("prepare_sd_manifest(request"), final_function.index("plan = final_sd_generation_plan"))

    def test_notebooks_02_to_04_use_signature_cache_v2(self) -> None:
        for name in (
            "02_SD21_Filtered_100steps.ipynb", "03_SD21_VAE_FineTuned.ipynb", "04_SD21_LoRA.ipynb",
        ):
            notebook = json.loads((ROOT / "notebooks" / "2_diffusers" / name).read_text(encoding="utf-8"))
            cell = next("".join(item.get("source", [])) for item in notebook["cells"] if "checkpoint_validation_cache_v2.json" in "".join(item.get("source", [])))
            compile(cell, name, "exec")
            for field in ("schema_version", "validation_csv_signature", "checkpoint_signature", "negative_image_signature", "positive_image_signature", "sd_metrics_cache_config", "n_validation_images_per_class", "prdc_nearest_k", "evaluator_batch_size", "evaluator_num_workers", "metric_backend", "metric_backend_version"):
                self.assertIn(field, cell)

    def test_notebook_commands_and_training_gpu_are_separate(self) -> None:
        for name in ("07_LDM_SDVAE_Extra1361.ipynb", "08_LDM_v3_SDVAE_FromScratch.ipynb"):
            source = (ROOT / "notebooks" / "2_diffusers" / name).read_text(encoding="utf-8")
            self.assertIn("add_generation_parallel_args(pos_cmd)", source)
            self.assertIn("add_generation_parallel_args(neg_base_cmd)", source)
            self.assertIn("TRAIN_GPU_VISIBLE_DEVICES", source)

    def test_07_08_negative_parallel_flags_match_positive_configuration(self) -> None:
        for name in ("07_LDM_SDVAE_Extra1361.ipynb", "08_LDM_v3_SDVAE_FromScratch.ipynb"):
            notebook = __import__("json").loads((ROOT / "notebooks" / "2_diffusers" / name).read_text())
            config_source = next("".join(cell.get("source", [])) for cell in notebook["cells"] if "def add_generation_parallel_args" in "".join(cell.get("source", [])))
            module = ast.parse(config_source)
            function = next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == "add_generation_parallel_args")
            namespace: dict = {}
            exec(compile(ast.Module(body=[function], type_ignores=[]), "<notebook>", "exec"), namespace)
            add = namespace["add_generation_parallel_args"]
            for enabled, devices, maximum, expected in (
                (False, "0", None, ["--generation-gpus", "off"]),
                (True, "0", None, ["--generation-gpus", "0"]),
                (True, "0,1", 1, ["--generation-gpus", "0,1", "--max-generation-workers", "1"]),
            ):
                add.__globals__.update(PARALLEL_GENERATION=enabled, GENERATION_GPU_DEVICES=devices, GENERATION_MAX_WORKERS=maximum)
                positive, negative = add([]), add([])
                self.assertEqual(positive, expected)
                self.assertEqual(negative, expected)

    def test_parallel_parent_manifest_preserves_serial_fields(self) -> None:
        source = (ROOT / "notebooks" / "utility" / "generate_ldm_v2.py").read_text(encoding="utf-8")
        for field in (
            '"sample_steps"', '"guidance_scale"', '"vae_backend"', '"sd_vae_model"',
            '"model_path"', '"state_json"', '"canonical_run"', '"generation_worker_count"',
            '"generation_gpu_devices_resolved"', '"wall_clock_seconds"',
        ):
            self.assertIn(field, source)

    def test_sd_checkpoint_content_change_rejects_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.keras"
            checkpoint.write_bytes(b"first weights")
            request = self._request(root / "evaluation")
            prepare_sd_manifest(request, str(checkpoint), False)
            checkpoint.write_bytes(b"second weights")
            with self.assertRaisesRegex(RuntimeError, "checkpoint_signature"):
                prepare_sd_manifest(request, str(checkpoint), False)

    def test_sd_base_vae_content_change_invalidates_manifest_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            vae = base / "vae" / "diffusion_pytorch_model.safetensors"
            vae.parent.mkdir(parents=True)
            vae.write_bytes(b"vae first")
            checkpoint = root / "checkpoint.keras"
            checkpoint.write_bytes(b"weights")
            request = self._request(root / "evaluation")
            request["base_model_dir"] = str(base)
            prepare_sd_manifest(request, str(checkpoint), False)
            before = sd_base_model_signature(base)
            config = sd_metrics_cache_config(
                eval_seed=42, inference_steps=10, guidance_scale=1.5, resolution=512,
                n_gen_per_class=3, checkpoint_type="full_unet", base_model_dir=base,
            )
            validation = root / "validation.csv"
            validation.write_text("label,path\n", encoding="utf-8")
            cache = {"schema_version": 2, "config": config, "validation_csv_signature": file_content_signature(validation), "checkpoints": {}}
            vae.write_bytes(b"vae second")
            self.assertNotEqual(before, sd_base_model_signature(base))
            with self.assertRaisesRegex(RuntimeError, "base_model"):
                prepare_sd_manifest(request, str(checkpoint), False)
            changed = sd_metrics_cache_config(
                eval_seed=42, inference_steps=10, guidance_scale=1.5, resolution=512,
                n_gen_per_class=3, checkpoint_type="full_unet", base_model_dir=base,
            )
            self.assertFalse(sd_metrics_cache_compatible(cache, changed, validation))

    def test_final_sd_checkpoint_content_change_rejects_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.keras"
            checkpoint.write_bytes(b"first")
            request = self._request(root / "final", "final_new")
            prepare_sd_manifest(request, str(checkpoint), False)
            checkpoint.write_bytes(b"second")
            with self.assertRaisesRegex(RuntimeError, "checkpoint_signature"):
                prepare_sd_manifest(request, str(checkpoint), False)

    def test_mixed_sd_manifest_accepts_second_override_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request = self._request(Path(tmp) / "final", "final_new")
            png(Path(request["out_dir"]) / "gen_0000.png")
            prepare_sd_manifest(request, "checkpoint", True)
            # The audited legacy set remains immutable across resumes.
            prepare_sd_manifest(request, "checkpoint", True)
            manifest = json.loads((Path(request["out_dir"]) / ".generation_manifest.json").read_text())
            self.assertEqual(manifest["seed_strategy"], "mixed_legacy_and_per_image_seed_v2")
            with self.assertRaises(RuntimeError):
                prepare_sd_manifest(request, "checkpoint", False)

    @staticmethod
    def _raw_args(**overrides):
        values = dict(
            seed=42, target_label=1, n_raw=2, sample_steps=100,
            guidance_scale=1.5, parameterization="eps", unet_version="v2",
            vae_backend="sd", vae_source="sd_vae_original", sd_vae_model=None,
            model_path=None, force_recompute=False, decode_on_cpu=False,
            sd_vae_batch_size=1,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _sweep_args(**overrides):
        values = dict(
            seed=42, n_gen_per_class=2, sample_steps=100, guidance_scale=1.5,
            parameterization="eps", unet_version="v2", vae_backend="keras",
            vae_source="sd_vae_original", sd_vae_model=None, force_recompute=False,
            decode_on_cpu=False, sd_vae_batch_size=1, generation_worker=False,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_raw_ldm_complete_directory_rejects_guidance_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "ldm.keras"
            checkpoint.write_bytes(b"weights")
            raw = root / "raw"
            paths = SimpleNamespace(experiment_dir=root, project_root=root, models_dir=root / "models")
            with patch.object(generate_ldm_v2, "resolve_model_path", return_value=checkpoint), patch.object(generate_ldm_v2, "ldm_vae_signature", return_value={"vae": 1}):
                generate_ldm_v2.prepare_raw_generation_manifest(self._raw_args(), paths, raw, parent=True)
                png(raw / "synth_00000.png")
                png(raw / "synth_00001.png")
                with self.assertRaisesRegex(RuntimeError, "Incompatible RAW"):
                    generate_ldm_v2.prepare_raw_generation_manifest(self._raw_args(guidance_scale=2.0), paths, raw, parent=True)

    def test_raw_ldm_pngs_without_manifest_refuse_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "ldm.keras"
            checkpoint.write_bytes(b"weights")
            raw = root / "raw"
            png(raw / "synth_00000.png")
            paths = SimpleNamespace(experiment_dir=root, project_root=root, models_dir=root / "models")
            with patch.object(generate_ldm_v2, "resolve_model_path", return_value=checkpoint), patch.object(generate_ldm_v2, "ldm_vae_signature", return_value={"vae": 1}):
                with self.assertRaisesRegex(RuntimeError, "without .generation_manifest"):
                    generate_ldm_v2.prepare_raw_generation_manifest(self._raw_args(), paths, raw, parent=True)

    def test_raw_ldm_partial_directory_rejects_seed_or_model_change_and_allows_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "ldm.keras"
            checkpoint.write_bytes(b"first")
            raw = root / "raw"
            paths = SimpleNamespace(experiment_dir=root, project_root=root, models_dir=root / "models")
            with patch.object(generate_ldm_v2, "resolve_model_path", return_value=checkpoint), patch.object(generate_ldm_v2, "ldm_vae_signature", return_value={"vae": 1}):
                args = self._raw_args()
                generate_ldm_v2.prepare_raw_generation_manifest(args, paths, raw, parent=True)
                png(raw / "synth_00000.png")
                # Same contract permits filling only the missing index.
                generate_ldm_v2.prepare_raw_generation_manifest(args, paths, raw, parent=True)
                with self.assertRaisesRegex(RuntimeError, "Incompatible RAW"):
                    generate_ldm_v2.prepare_raw_generation_manifest(self._raw_args(seed=43), paths, raw, parent=True)
                checkpoint.write_bytes(b"second")
                with self.assertRaisesRegex(RuntimeError, "Incompatible RAW"):
                    generate_ldm_v2.prepare_raw_generation_manifest(args, paths, raw, parent=True)

    def test_sweep_checkpoint_change_invalidates_generation_and_metrics_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "ldm.keras"
            checkpoint.write_bytes(b"first")
            paths = SimpleNamespace(project_root=root, evaluation_dir=root / "evaluation", models_dir=root / "models")
            args = self._sweep_args()
            candidate = {"checkpoint_id": "step_1", "kind": "step", "step": 1, "path": checkpoint}
            with patch.object(evaluate_ldm_v2, "sweep_vae_signature", return_value={"vae": 1}):
                for cls in (0, 1):
                    evaluate_ldm_v2.prepare_sweep_generation_manifest(args, paths, candidate, cls)
                    png(evaluate_ldm_v2.fake_output_dir(paths, args, "step_1", cls) / "0000.png")
                    png(evaluate_ldm_v2.fake_output_dir(paths, args, "step_1", cls) / "0001.png")
                payload = {
                    "schema_version": 2,
                    "candidate_signature": evaluate_ldm_v2.build_candidate_signature([candidate]),
                    "generation_image_signature": evaluate_ldm_v2.sweep_generated_image_signature(args, paths, [candidate]),
                }
                checkpoint.write_bytes(b"second")
                with self.assertRaisesRegex(RuntimeError, "Incompatible sweep"):
                    evaluate_ldm_v2.prepare_sweep_generation_manifest(args, paths, candidate, 0)
                self.assertFalse(evaluate_ldm_v2.ldm_metrics_cache_compatible(payload, args, paths, [candidate]))

    def test_ldm_metrics_cache_is_invalidated_when_sweep_png_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "ldm.keras"
            checkpoint.write_bytes(b"weights")
            paths = SimpleNamespace(project_root=root, evaluation_dir=root / "evaluation", models_dir=root / "models")
            args = self._sweep_args()
            candidate = {"checkpoint_id": "step_1", "kind": "step", "step": 1, "path": checkpoint}
            with patch.object(evaluate_ldm_v2, "sweep_vae_signature", return_value={"vae": 1}):
                for cls in (0, 1):
                    evaluate_ldm_v2.prepare_sweep_generation_manifest(args, paths, candidate, cls)
                    png(evaluate_ldm_v2.fake_output_dir(paths, args, "step_1", cls) / "0000.png")
                payload = {"schema_version": 2, "candidate_signature": evaluate_ldm_v2.build_candidate_signature([candidate]), "generation_image_signature": evaluate_ldm_v2.sweep_generated_image_signature(args, paths, [candidate])}
                Image.new("L", (3, 3), color=2).save(evaluate_ldm_v2.fake_output_dir(paths, args, "step_1", 0) / "0000.png")
                self.assertFalse(evaluate_ldm_v2.ldm_metrics_cache_compatible(payload, args, paths, [candidate]))

    def test_sd_cache_invalidates_validation_reference_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation = root / "validation.csv"
            validation.write_text("label,path\n", encoding="utf-8")
            common = dict(eval_seed=42, inference_steps=100, guidance_scale=1.5, resolution=512, n_gen_per_class=100, checkpoint_type="full_unet", base_model_dir=root / "base", n_validation_images_per_class=73, prdc_nearest_k=5, evaluator_batch_size=8, evaluator_num_workers=0, metric_backend="GenerativeEvaluator", metric_backend_version="v1")
            config = sd_metrics_cache_config(**common)
            payload = {"schema_version": 2, "config": config, "validation_csv_signature": file_content_signature(validation), "checkpoints": {}}
            self.assertFalse(sd_metrics_cache_compatible(payload, sd_metrics_cache_config(**{**common, "n_validation_images_per_class": 72}), validation))

    def test_sd_cache_invalidates_prdc_nearest_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation = root / "validation.csv"
            validation.write_text("label,path\n", encoding="utf-8")
            common = dict(eval_seed=42, inference_steps=100, guidance_scale=1.5, resolution=512, n_gen_per_class=100, checkpoint_type="full_unet", base_model_dir=root / "base", n_validation_images_per_class=73, prdc_nearest_k=5, evaluator_batch_size=8, evaluator_num_workers=0, metric_backend="GenerativeEvaluator", metric_backend_version="v1")
            config = sd_metrics_cache_config(**common)
            payload = {"schema_version": 2, "config": config, "validation_csv_signature": file_content_signature(validation), "checkpoints": {}}
            self.assertFalse(sd_metrics_cache_compatible(payload, sd_metrics_cache_config(**{**common, "prdc_nearest_k": 3}), validation))

    def test_filtered_ldm_real_name_discovery_excludes_invalid_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            png(directory / "synth_filtered_0000.png")
            png(directory / ".tmp_synth_filtered_0001.png")
            png(directory / "foreign.png")
            (directory / "synth_filtered_0002.png").write_bytes(b"corrupt")
            self.assertEqual(
                readable_png_paths(directory, GENERATED_PNG_PATTERN),
                [directory / "synth_filtered_0000.png"],
            )

    def test_mode_all_filter_outputs_are_discoverable_by_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filtered = Path(tmp) / "filtered"
            png(filtered / "synth_filtered_0000.png")
            png(filtered / "synth_filtered_0001.png")
            self.assertEqual(
                [p.name for p in readable_png_paths(filtered, GENERATED_PNG_PATTERN)],
                ["synth_filtered_0000.png", "synth_filtered_0001.png"],
            )

    def test_sweep_force_is_parsed_by_real_worker_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                **self._sweep_args(force_recompute=True).__dict__, cuda_root=root,
                min_step=1, classes="0,1", vram_log_every=10,
                results_stage_name="test", generation_gpus="auto",
                uses_vae_ft_from_03c=False, notebook_name=None,
            )
            paths = SimpleNamespace(project_root=root, experiment_dir=root / "experiment")
            command = evaluate_ldm_v2.child_generate_command(args, paths, "step_1", "0")
            with patch.object(sys, "argv", [command[1], *command[2:]]):
                parsed = evaluate_ldm_v2.parse_args()
            self.assertTrue(parsed.force_recompute)
            self.assertTrue(parsed.generation_worker)

    def test_sweep_force_worker_clears_compatible_complete_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "ldm.keras"; checkpoint.write_bytes(b"weights")
            paths = SimpleNamespace(project_root=root, experiment_dir=root, evaluation_dir=root / "evaluation", models_dir=root / "models", latents_dir=root / "latents")
            args = self._sweep_args()
            candidate = {"checkpoint_id": "step_1", "kind": "step", "step": 1, "path": checkpoint}
            with patch.object(evaluate_ldm_v2, "sweep_vae_signature", return_value={"vae": 1}):
                evaluate_ldm_v2.prepare_sweep_generation_manifest(args, paths, candidate, 0)
                out = evaluate_ldm_v2.fake_output_dir(paths, args, "step_1", 0)
                png(out / "0000.png"); png(out / "0001.png")
                forced = self._sweep_args(force_recompute=True, generation_worker=True)
                evaluate_ldm_v2.prepare_sweep_generation_manifest(forced, paths, candidate, 0)
                self.assertEqual(evaluate_ldm_v2.missing_fake_image_indices(paths, forced, "step_1", 0), [0, 1])

    def test_sweep_force_worker_replaces_incompatible_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); checkpoint = root / "ldm.keras"; checkpoint.write_bytes(b"weights")
            paths = SimpleNamespace(project_root=root, experiment_dir=root, evaluation_dir=root / "evaluation", models_dir=root / "models", latents_dir=root / "latents")
            candidate = {"checkpoint_id": "step_1", "kind": "step", "step": 1, "path": checkpoint}
            with patch.object(evaluate_ldm_v2, "sweep_vae_signature", return_value={"vae": 1}):
                evaluate_ldm_v2.prepare_sweep_generation_manifest(self._sweep_args(), paths, candidate, 0)
                out = evaluate_ldm_v2.fake_output_dir(paths, self._sweep_args(), "step_1", 0); png(out / "0000.png")
                forced = self._sweep_args(decode_on_cpu=True, force_recompute=True, generation_worker=True)
                expected = evaluate_ldm_v2.prepare_sweep_generation_manifest(forced, paths, candidate, 0)
                self.assertFalse((out / "0000.png").exists())
                self.assertEqual(json.loads((out / ".generation_manifest.json").read_text()), expected)

    def test_sweep_metrics_force_does_not_delete_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); checkpoint = root / "ldm.keras"; checkpoint.write_bytes(b"weights")
            paths = SimpleNamespace(project_root=root, experiment_dir=root, evaluation_dir=root / "evaluation", models_dir=root / "models", latents_dir=root / "latents")
            args = self._sweep_args(); candidate = {"checkpoint_id": "step_1", "kind": "step", "step": 1, "path": checkpoint}
            with patch.object(evaluate_ldm_v2, "sweep_vae_signature", return_value={"vae": 1}):
                evaluate_ldm_v2.prepare_sweep_generation_manifest(args, paths, candidate, 0)
                image = evaluate_ldm_v2.fake_output_dir(paths, args, "step_1", 0) / "0000.png"; png(image)
                evaluate_ldm_v2.prepare_sweep_generation_manifest(self._sweep_args(force_recompute=True), paths, candidate, 0)
                self.assertTrue(image.exists())

    def test_mode_all_force_requests_full_raw_regeneration_once(self) -> None:
        args = SimpleNamespace(force_recompute=True, results_stage_name="test")
        with patch.object(generate_ldm_v2, "run_generate") as generate, patch.object(
            generate_ldm_v2, "resolve_image_dirs", return_value=(Path("raw"), Path("filtered"))
        ), patch.dict(sys.modules, {"adaptive_mammography_filter": SimpleNamespace(filter_generated_directory=lambda **kwargs: {"cached": False})}):
            paths = SimpleNamespace(project_root=ROOT, results_stage_name="x")
            with patch.object(generate_ldm_v2, "get_results_paths", side_effect=RuntimeError("after generation")):
                with self.assertRaises(RuntimeError):
                    generate_ldm_v2.run_filter(args, paths)
            generate.assert_called_once()
            self.assertFalse(generate.call_args.args[0].force_recompute)

    def test_raw_manifest_changes_with_latent_stats_and_decode_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "latents").mkdir(); stats = root / "latents" / "latent_stats.npz"; stats.write_bytes(b"one")
            checkpoint = root / "ldm.keras"; checkpoint.write_bytes(b"weights")
            paths = SimpleNamespace(project_root=root, experiment_dir=root, models_dir=root / "models", latents_dir=root / "latents")
            with patch.object(generate_ldm_v2, "resolve_model_path", return_value=checkpoint), patch.object(generate_ldm_v2, "ldm_vae_signature", return_value={"vae": 1}):
                first = generate_ldm_v2.raw_generation_manifest_payload(self._raw_args(), paths, root / "raw")
                stats.write_bytes(b"two")
                second = generate_ldm_v2.raw_generation_manifest_payload(self._raw_args(), paths, root / "raw")
                third = generate_ldm_v2.raw_generation_manifest_payload(self._raw_args(decode_on_cpu=True), paths, root / "raw")
            self.assertNotEqual(first["latent_stats_signature"], second["latent_stats_signature"])
            self.assertNotEqual(second, third)

    @staticmethod
    def _metrics_signature_fixture(root: Path):
        metadata = root / "metadata"; data = root / "data"; models = root / "models"; latents = root / "latents"
        metadata.mkdir(); (data / "val" / "1").mkdir(parents=True); models.mkdir(); latents.mkdir()
        (metadata / "val.csv").write_text("processed_path,split,label\nreal.png,val,1\n", encoding="utf-8")
        png(data / "val" / "1" / "real.png")
        (models / "vae_decoder_best.keras").write_bytes(b"vae-one")
        (latents / "latent_stats.npz").write_bytes(b"stats")
        paths = SimpleNamespace(project_root=root, experiment_dir=root, metadata_dir=metadata, data_processed_dir=data, models_dir=models, latents_dir=latents)
        args = ParallelGenerationTests._sweep_args()
        frame = SimpleNamespace(iterrows=lambda: iter([(0, {"processed_path": "real.png", "split": "val", "label": 1})]))
        pandas_stub = SimpleNamespace(read_csv=lambda path: frame)
        return args, paths, pandas_stub

    def test_ldm_metrics_cache_signature_changes_with_vae_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, paths, pandas_stub = self._metrics_signature_fixture(Path(tmp))
            with patch.dict(sys.modules, {"pandas": pandas_stub}):
                first = evaluate_ldm_v2.sweep_metrics_input_signature(args, paths)
                (paths.models_dir / "vae_decoder_best.keras").write_bytes(b"vae-two")
                second = evaluate_ldm_v2.sweep_metrics_input_signature(args, paths)
            self.assertNotEqual(first, second)

    def test_ldm_metrics_cache_signature_changes_with_validation_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, paths, pandas_stub = self._metrics_signature_fixture(Path(tmp))
            with patch.dict(sys.modules, {"pandas": pandas_stub}):
                first = evaluate_ldm_v2.sweep_metrics_input_signature(args, paths)
                (paths.metadata_dir / "val.csv").write_text("processed_path,split,label\nreal.png,val,1\n#changed\n", encoding="utf-8")
                second = evaluate_ldm_v2.sweep_metrics_input_signature(args, paths)
            self.assertNotEqual(first, second)

    def test_ldm_metrics_cache_signature_changes_with_validation_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, paths, pandas_stub = self._metrics_signature_fixture(Path(tmp))
            with patch.dict(sys.modules, {"pandas": pandas_stub}):
                first = evaluate_ldm_v2.sweep_metrics_input_signature(args, paths)
                Image.new("L", (3, 3), color=2).save(paths.data_processed_dir / "val" / "1" / "real.png")
                second = evaluate_ldm_v2.sweep_metrics_input_signature(args, paths)
            self.assertNotEqual(first, second)

    def test_filter_excludes_foreign_raw_and_raw_change_invalidates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / "raw"; filtered = root / "filtered"
            png(raw / "synth_00000.png"); png(raw / "foreign.png")
            png(raw / ".tmp_synth_00001.png"); (raw / "synth_00002.png").write_bytes(b"corrupt")
            self.assertEqual([p.name for p in ldm_raw_png_paths(raw)], ["synth_00000.png"])
            png(filtered / "synth_filtered_0000.png")
            summary = root / "synthetic_filter_summary.json"
            signature = {"raw_files": [{"name": "synth_00000.png", "size": 1}]}
            summary.write_text(json.dumps({"input_signature": signature}), encoding="utf-8")
            self.assertTrue(filtered_selection_cache_matches(summary, signature, filtered, 1))
            changed = {"raw_files": [{"name": "synth_00000.png", "size": 2}]}
            self.assertFalse(filtered_selection_cache_matches(summary, changed, filtered, 1))

    def test_filtered_cache_rejects_corrupt_or_extra_canonical_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); filtered = root / "filtered"; summary = root / "summary.json"
            signature = {"raw_files": []}
            png(filtered / "synth_filtered_0000.png")
            (filtered / "synth_filtered_0001.png").write_bytes(b"corrupt")
            summary.write_text(json.dumps({"input_signature": signature}), encoding="utf-8")
            self.assertFalse(filtered_selection_cache_matches(summary, signature, filtered, 1))

    def test_filter_rejects_missing_raw_index_despite_out_of_range_png(self) -> None:
        module_path = ROOT / "notebooks" / "utility" / "adaptive_mammography_filter.py"
        spec = importlib.util.spec_from_file_location("_filter_raw_set_test", module_path)
        module = importlib.util.module_from_spec(spec)
        import numpy  # Keep its extension module registered across patch.dict cleanup.
        fake_ndi = SimpleNamespace()
        with patch.dict(sys.modules, {"pandas": SimpleNamespace(), "scipy": SimpleNamespace(ndimage=fake_ndi), "scipy.ndimage": fake_ndi}):
            assert spec.loader is not None
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / "raw"; png(raw / "synth_00000.png"); png(raw / "synth_99999.png")
            with self.assertRaisesRegex(RuntimeError, r"\[1\]"):
                module.filter_generated_directory(raw, root / "filtered", [], n_raw=2, n_selected=1, verbose=False)

    def test_sd_worker_never_removes_another_workers_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"; out_dir.mkdir()
            active_temp = out_dir / ".tmp_gen_0000_99999.png"; active_temp.write_bytes(b"active")
            job_path = Path(tmp) / "job.json"
            job_path.write_text(json.dumps({"requests": [{"out_dir": str(out_dir), "count": 0}]}), encoding="utf-8")
            with patch.object(sys, "argv", ["sd_generation_worker.py", "--job-file", str(job_path)]):
                sd_generation_worker.main()
            self.assertTrue(active_temp.exists())

    def test_sd_parent_cleans_shared_output_once_before_scheduling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out_dir = root / "out"; events = []
            request = self._request(out_dir)
            job = {
                "label": "shared", "checkpoint_path": str(root / "checkpoint"),
                "checkpoint_type": "full_unet", "base_model_dir": str(root / "base"),
                "requests": [request, {**request}],
            }
            with patch("parallel_generation_utils.resolve_generation_gpu_devices", return_value=["0"]), patch(
                "parallel_generation_utils.remove_stale_generation_temps",
                side_effect=lambda directory: events.append(("clean", Path(directory))),
            ), patch("parallel_generation_utils.run_dynamic_gpu_jobs", side_effect=lambda **kwargs: events.append(("launch", None))):
                run_sd_generation_jobs([job], "0", 1, root / "logs", root)
            self.assertEqual(events, [("clean", out_dir), ("launch", None)])

    def test_sd_dry_run_does_not_create_manifest_jobs_or_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out_dir = root / "out"; out_dir.mkdir()
            def tree_snapshot():
                return {
                    path.relative_to(root): ("directory", None) if path.is_dir() else ("file", path.read_bytes())
                    for path in root.rglob("*")
                }
            before = tree_snapshot()
            request = self._request(out_dir)
            job = {"label": "dry", "checkpoint_path": str(root / "checkpoint"), "requests": [request]}
            with patch("parallel_generation_utils.resolve_generation_gpu_devices", return_value=["0"]):
                run_sd_generation_jobs([job], "0", 1, root / "logs", root, dry_run=True)
            self.assertEqual(before, tree_snapshot())

    def test_worker_failure_is_propagated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                run_dynamic_gpu_jobs(
                    jobs=[{"label": "intentional_failure"}],
                    devices=["0"],
                    command_for_job=lambda job, gpu: [sys.executable, "-c", "raise SystemExit(9)"],
                    logs_dir=Path(tmp),
                    cwd=ROOT,
                )

    def test_dry_run_force_manifest_is_byte_identical_and_non_destructive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / "raw"; raw.mkdir(); png(raw / "synth_00000.png")
            manifest = raw / ".generation_manifest.json"; manifest.write_bytes(b'{"old": true}\n')
            before = {path.name: path.read_bytes() for path in raw.iterdir()}
            checkpoint = root / "model.keras"; checkpoint.write_bytes(b"model")
            paths = SimpleNamespace(project_root=root, experiment_dir=root, models_dir=root / "models", latents_dir=root / "latents")
            with patch.object(generate_ldm_v2, "resolve_model_path", return_value=checkpoint), patch.object(generate_ldm_v2, "ldm_vae_signature", return_value={"vae": 1}):
                generate_ldm_v2.prepare_raw_generation_manifest(self._raw_args(force_recompute=True), paths, raw, parent=True, dry_run=True)
            self.assertEqual(before, {path.name: path.read_bytes() for path in raw.iterdir()})

    def test_dry_run_sweep_manifest_does_not_create_or_clear_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); checkpoint = root / "ldm.keras"; checkpoint.write_bytes(b"weights")
            paths = SimpleNamespace(project_root=root, evaluation_dir=root / "evaluation", models_dir=root / "models")
            args = self._sweep_args(force_recompute=True, generation_worker=True)
            candidate = {"checkpoint_id": "step_1", "kind": "step", "step": 1, "path": checkpoint}
            before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            with patch.object(evaluate_ldm_v2, "sweep_vae_signature", return_value={"vae": 1}):
                evaluate_ldm_v2.prepare_sweep_generation_manifest(args, paths, candidate, 0, dry_run=True)
            self.assertEqual(before, {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()})
            self.assertFalse(paths.evaluation_dir.exists())

    def test_exact_filtered_scanner_ignores_foreign_and_rejects_wrong_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); png(directory / "synth_filtered_0000.png"); png(directory / "foreign.png")
            self.assertEqual(exact_filtered_png_paths(directory, 1), [directory / "synth_filtered_0000.png"])
            png(directory / "synth_filtered_0002.png")
            with self.assertRaisesRegex(RuntimeError, "extra"):
                exact_filtered_png_paths(directory, 1)
            (directory / "synth_filtered_0002.png").unlink()
            with self.assertRaisesRegex(RuntimeError, "0001"):
                exact_filtered_png_paths(directory, 2)

    def test_final_evaluator_command_passes_exact_selected_count(self) -> None:
        paths = SimpleNamespace(project_root=ROOT, experiment_dir=ROOT / "experiment", synthetic_raw_dir=ROOT / "raw", synthetic_filtered_dir=ROOT / "filtered")
        with patch.object(sys, "argv", ["generate_ldm_v2.py", "--mode", "all", "--n-selected", "7"]):
            args = generate_ldm_v2.parse_args()
        command = generate_ldm_v2.evaluate_filtered_command(args, paths, paths.synthetic_filtered_dir)
        self.assertEqual(command[command.index("--expected-synthetic-count") + 1], "7")

    def test_sweep_out_of_range_index_does_not_fill_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); paths = SimpleNamespace(project_root=root, evaluation_dir=root / "evaluation")
            args = self._sweep_args(); out = evaluate_ldm_v2.fake_output_dir(paths, args, "step_1", 0)
            png(out / "0000.png"); png(out / "9999.png")
            self.assertEqual(evaluate_ldm_v2.missing_fake_image_indices(paths, args, "step_1", 0), [1])

    def test_sd_vae_worker_enables_tensorflow_memory_growth_environment(self) -> None:
        source = (ROOT / "notebooks" / "utility" / "generate_ldm_v2.py").read_text(encoding="utf-8")
        keras_source = (ROOT / "notebooks" / "utility" / "ldm_keras_utils.py").read_text(encoding="utf-8")
        self.assertIn('TF_FORCE_GPU_ALLOW_GROWTH", "true"', source)
        self.assertIn("set_memory_growth(gpu, True)", keras_source)
        self.assertIn('args.vae_backend == "sd" and not args.decode_on_cpu', source)

    def test_sd_vae_lazy_tensor_annotation_resolves_without_torch_import(self) -> None:
        import typing
        import sd_vae_utils
        self.assertIn("return", typing.get_type_hints(sd_vae_utils.image_batch_to_sd_tensor))

    def test_downstream_comparison_notebook_compiles_and_does_not_open_test(self) -> None:
        notebook = json.loads((ROOT / "notebooks" / "4_downstream_classifiers" / "09_Downstream_Validation_Comparison.ipynb").read_text(encoding="utf-8"))
        cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        joined = "\n".join(cells)
        self.assertIn("finalize_downstream_validation.py", joined)
        self.assertNotIn("metadata/test.csv", joined)
        for index, source in enumerate(cells):
            compile(source, f"09-cell-{index}", "exec")

    def test_ldm_preview_notebooks_guard_both_sd_layout_globals(self) -> None:
        for name in ("05_LDM_Basic_FromScratch.ipynb", "06_LDM_Extra1361_FromScratch.ipynb"):
            payload = json.loads((ROOT / "notebooks" / "2_diffusers" / name).read_text(encoding="utf-8"))
            text_value = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
            self.assertIn('"EVAL_DIR" in globals()', text_value)

    def test_filter_recompute_path_always_clears_canonical_outputs(self) -> None:
        source = (ROOT / "notebooks" / "utility" / "adaptive_mammography_filter.py").read_text(encoding="utf-8")
        cache_end = source.index("reference = compute_reference_statistics")
        prefix = source[:cache_end]
        self.assertIn("_clear_filtered_outputs(filtered_dir)", prefix)
        self.assertNotIn("if force_recompute:\n        _clear_filtered_outputs", prefix)

    def test_cached_best_model_is_written_only_to_current_canonical_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); evaluation = root / "evaluation"; checkpoints = root / "checkpoints"
            evaluation.mkdir(); checkpoints.mkdir(); best = root / "best.keras"; best.write_bytes(b"best-weights")
            old = root / "old-project" / "ldm_unet_best_eval.keras"
            args = SimpleNamespace(force_recompute=False)
            paths = SimpleNamespace(evaluation_dir=evaluation, checkpoints_dir=checkpoints)
            candidates = [{"checkpoint_id": "best", "kind": "best", "step": 1, "path": best}]
            config = {"config": True}; candidate_signature = [{"candidate": True}]
            payload = {
                "schema_version": 2, "config": config, "candidate_signature": candidate_signature,
                "selection": {
                    **evaluate_ldm_v2.selection_policy(), "best_checkpoint": str(best),
                    "best_model": str(old), "best_checkpoint_id": "best",
                },
            }
            (evaluation / "checkpoint_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(evaluate_ldm_v2, "build_eval_config", return_value=config), patch.object(
                evaluate_ldm_v2, "build_candidate_signature", return_value=candidate_signature
            ), patch.object(evaluate_ldm_v2, "normalize_eval_config", return_value=config), patch.object(
                evaluate_ldm_v2, "ldm_metrics_cache_compatible", return_value=True
            ), patch.object(
                evaluate_ldm_v2, "refresh_artifacts_from_cache", return_value=True
            ):
                self.assertTrue(evaluate_ldm_v2.use_cached_metrics_if_valid(args, paths, candidates))
            canonical = checkpoints / "ldm_unet_best_eval.keras"
            self.assertEqual(canonical.read_bytes(), best.read_bytes())
            self.assertFalse(old.exists())
            self.assertEqual(payload["selection"].get("best_model"), str(old))

    # -- Multi-GPU smoke: CUDA_VISIBLE_DEVICES diagnostics (item 2) -----------------

    def test_auto_gpu_resolution_respects_inherited_cuda_visible_devices(self) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            devices = resolve_generation_gpu_devices("auto")
        self.assertEqual(devices, ["0"])

    def test_explicit_gpu_list_ignores_inherited_cuda_visible_devices(self) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            devices = resolve_generation_gpu_devices("0,1")
        self.assertEqual(devices, ["0", "1"])

    def test_print_gpu_resolution_dry_run_reports_physical_requested_resolved_worker_count(self) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False), patch(
            "parallel_generation_utils._devices_from_nvidia_smi", return_value=["0", "1"]
        ):
            devices = print_gpu_resolution_dry_run("auto")
        self.assertEqual(devices, ["0"])

    # -- Diffusers model signature: FP16 / sharded discovery (item 3) ---------------

    def test_sd_base_model_signature_detects_fp16_vae_weight_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            vae_dir = base / "vae"
            vae_dir.mkdir(parents=True)
            (vae_dir / "config.json").write_text("{}", encoding="utf-8")
            weight = vae_dir / "diffusion_pytorch_model.fp16.safetensors"
            weight.write_bytes(b"vae fp16 weights v1")
            before = sd_base_model_signature(base)
            self.assertIn("vae/diffusion_pytorch_model.fp16.safetensors", before["components"])
            weight.write_bytes(b"vae fp16 weights v2 - different content")
            after = sd_base_model_signature(base)
            self.assertNotEqual(before, after)

    def test_sd_base_model_signature_detects_fp16_unet_weight_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            unet_dir = base / "unet"
            unet_dir.mkdir(parents=True)
            (unet_dir / "config.json").write_text("{}", encoding="utf-8")
            weight = unet_dir / "diffusion_pytorch_model.fp16.safetensors"
            weight.write_bytes(b"unet fp16 weights v1")
            before = sd_base_model_signature(base)
            self.assertIn("unet/diffusion_pytorch_model.fp16.safetensors", before["components"])
            weight.write_bytes(b"unet fp16 weights v2 - different content")
            after = sd_base_model_signature(base)
            self.assertNotEqual(before, after)

    def test_sd_base_model_signature_detects_sharded_weight_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            unet_dir = base / "unet"
            unet_dir.mkdir(parents=True)
            (unet_dir / "config.json").write_text("{}", encoding="utf-8")
            (unet_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
                '{"weight_map": {}}', encoding="utf-8"
            )
            shard_1 = unet_dir / "diffusion_pytorch_model-00001-of-00002.safetensors"
            shard_2 = unet_dir / "diffusion_pytorch_model-00002-of-00002.safetensors"
            shard_1.write_bytes(b"shard one")
            shard_2.write_bytes(b"shard two")
            before = sd_base_model_signature(base)
            shard_2.write_bytes(b"shard two changed")
            after = sd_base_model_signature(base)
            self.assertNotEqual(before, after)
            key_1 = "unet/diffusion_pytorch_model-00001-of-00002.safetensors"
            key_2 = "unet/diffusion_pytorch_model-00002-of-00002.safetensors"
            self.assertEqual(before["components"][key_1], after["components"][key_1])
            self.assertNotEqual(before["components"][key_2], after["components"][key_2])

    def test_sd_base_model_signature_still_recognizes_standard_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            (base / "vae").mkdir(parents=True)
            (base / "scheduler").mkdir(parents=True)
            (base / "model_index.json").write_text("{}", encoding="utf-8")
            (base / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
            (base / "vae" / "config.json").write_text("{}", encoding="utf-8")
            (base / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"vae standard")
            signature = sd_base_model_signature(base)
            expected = {
                "model_index.json", "scheduler/scheduler_config.json",
                "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
            }
            self.assertTrue(expected.issubset(signature["components"].keys()))
            self.assertTrue(all(signature["components"][key] is not None for key in expected))

    def test_standalone_vae_recognizes_fp16_and_sharded_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            standard = Path(tmp) / "standard"; standard.mkdir()
            (standard / "config.json").write_text("{}", encoding="utf-8")
            (standard / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
            self.assertTrue(sd_vae_utils._has_vae_config(standard))

            fp16 = Path(tmp) / "fp16"; fp16.mkdir()
            (fp16 / "config.json").write_text("{}", encoding="utf-8")
            (fp16 / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"fp16 weights")
            self.assertTrue(sd_vae_utils._has_vae_config(fp16))

            sharded = Path(tmp) / "sharded"; sharded.mkdir()
            (sharded / "config.json").write_text("{}", encoding="utf-8")
            (sharded / "diffusion_pytorch_model.safetensors.index.json").write_text(
                '{"weight_map": {}}', encoding="utf-8"
            )
            (sharded / "diffusion_pytorch_model-00001-of-00002.safetensors").write_bytes(b"shard")
            self.assertTrue(sd_vae_utils._has_vae_config(sharded))

            config_only = Path(tmp) / "config_only"; config_only.mkdir()
            (config_only / "config.json").write_text("{}", encoding="utf-8")
            self.assertFalse(sd_vae_utils._has_vae_config(config_only))

            no_config = Path(tmp) / "no_config"; no_config.mkdir()
            (no_config / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
            self.assertFalse(sd_vae_utils._has_vae_config(no_config))

    # -- Content-aware filter/validation signatures (item 4) ------------------------

    def test_filter_image_signature_is_content_aware_despite_matching_stat(self) -> None:
        module_path = ROOT / "notebooks" / "utility" / "adaptive_mammography_filter.py"
        spec = importlib.util.spec_from_file_location("_filter_signature_test", module_path)
        module = importlib.util.module_from_spec(spec)
        import numpy  # Keep its extension module registered across patch.dict cleanup.
        fake_ndi = SimpleNamespace()
        with patch.dict(sys.modules, {"pandas": SimpleNamespace(), "scipy": SimpleNamespace(ndimage=fake_ndi), "scipy.ndimage": fake_ndi}):
            assert spec.loader is not None
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synth_00000.png"
            png(path)
            stat_before = path.stat()
            before = module._image_signature([path])
            data = bytearray(path.read_bytes())
            data[-1] ^= 0xFF
            path.write_bytes(bytes(data))
            os.utime(path, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns))
            self.assertEqual(path.stat().st_size, stat_before.st_size)
            self.assertEqual(path.stat().st_mtime_ns, stat_before.st_mtime_ns)
            after = module._image_signature([path])
            self.assertNotEqual(before, after)

    def test_validate_file_signature_is_content_aware_despite_matching_stat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synth_00000.png"
            png(path)
            stat_before = path.stat()
            before = generate_ldm_v2.file_signature([path])
            data = bytearray(path.read_bytes())
            data[-1] ^= 0xFF
            path.write_bytes(bytes(data))
            os.utime(path, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns))
            self.assertEqual(path.stat().st_size, stat_before.st_size)
            self.assertEqual(path.stat().st_mtime_ns, stat_before.st_mtime_ns)
            after = generate_ldm_v2.file_signature([path])
            self.assertNotEqual(before, after)

    # -- Concurrent-orchestration lock (item 5) --------------------------------------

    def test_parallel_generation_lock_blocks_concurrent_orchestration_on_same_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            lock_path = acquire_parallel_generation_lock(out_dir)
            try:
                with self.assertRaisesRegex(RuntimeError, "orchestrazione"):
                    acquire_parallel_generation_lock(out_dir)
            finally:
                release_parallel_generation_lock(lock_path)

    def test_parallel_generation_lock_removes_stale_lock_from_dead_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            out_dir.mkdir(parents=True)
            dead = subprocess.Popen([sys.executable, "-c", "pass"])
            dead_pid = dead.pid
            self.assertEqual(dead.wait(), 0)
            lock_path = out_dir / ".parallel_generation.lock"
            lock_path.write_text(json.dumps({"pid": dead_pid, "timestamp": 0}), encoding="utf-8")
            acquired = acquire_parallel_generation_lock(out_dir)
            try:
                self.assertEqual(json.loads(acquired.read_text())["pid"], os.getpid())
            finally:
                release_parallel_generation_lock(acquired)

    def test_parallel_generation_lock_released_in_finally_even_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out_dir = root / "out"
            request = self._request(out_dir)
            job = {
                "label": "shared", "checkpoint_path": str(root / "checkpoint"),
                "checkpoint_type": "full_unet", "base_model_dir": str(root / "base"),
                "requests": [request],
            }
            with patch("parallel_generation_utils.resolve_generation_gpu_devices", return_value=["0"]), patch(
                "parallel_generation_utils.run_dynamic_gpu_jobs", side_effect=RuntimeError("boom")
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    run_sd_generation_jobs([job], "0", 1, root / "logs", root)
            self.assertFalse((out_dir / ".parallel_generation.lock").exists())

    def test_sd_dry_run_does_not_create_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out_dir = root / "out"; out_dir.mkdir()
            request = self._request(out_dir)
            job = {"label": "dry", "checkpoint_path": str(root / "checkpoint"), "requests": [request]}
            with patch("parallel_generation_utils.resolve_generation_gpu_devices", return_value=["0"]):
                run_sd_generation_jobs([job], "0", 1, root / "logs", root, dry_run=True)
            self.assertFalse((out_dir / ".parallel_generation.lock").exists())

    # -- Notebook 07/08 multi-GPU smoke cell (item 1 + item 7 checklist) ------------

    @staticmethod
    def _smoke_cell_source(notebook_name: str) -> str:
        notebook = json.loads((ROOT / "notebooks" / "2_diffusers" / notebook_name).read_text(encoding="utf-8"))
        return next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code" and "RUN_MULTI_GPU_SMOKE = False" in "".join(cell.get("source", []))
        )

    def test_smoke_cell_disabled_by_default_and_compiles(self) -> None:
        for name in ("07_LDM_SDVAE_Extra1361.ipynb", "08_LDM_v3_SDVAE_FromScratch.ipynb"):
            cell = self._smoke_cell_source(name)
            self.assertIn("RUN_MULTI_GPU_SMOKE = False", cell)
            compile(cell, name, "exec")

    def test_smoke_command_uses_two_workers_on_explicit_gpu_list(self) -> None:
        for name in ("07_LDM_SDVAE_Extra1361.ipynb", "08_LDM_v3_SDVAE_FromScratch.ipynb"):
            cell = self._smoke_cell_source(name)
            self.assertIn('SMOKE_GENERATION_GPU_DEVICES = "0,1"', cell)
            self.assertIn("SMOKE_MAX_WORKERS = 2", cell)
            self.assertIn('"--generation-gpus", SMOKE_GENERATION_GPU_DEVICES', cell)
            self.assertIn('"--max-generation-workers", str(SMOKE_MAX_WORKERS)', cell)
            self.assertIn('"--mode", "generate"', cell)

    def test_smoke_command_uses_isolated_directories(self) -> None:
        for name in ("07_LDM_SDVAE_Extra1361.ipynb", "08_LDM_v3_SDVAE_FromScratch.ipynb"):
            cell = self._smoke_cell_source(name)
            self.assertIn('EXPERIMENT_DIR / "smoke_multi_gpu_dynamic" / f"run_{time.time_ns()}"', cell)
            self.assertIn('SMOKE_GENERATION_SCHEDULER = "dynamic_reservations"', cell)
            self.assertIn("SMOKE_RESERVATION_SIZE = 4", cell)
            self.assertIn('"--raw-dir", str(SMOKE_RAW_DIR)', cell)
            self.assertIn('"--filtered-dir", str(SMOKE_FILTERED_DIR)', cell)
            # Must not reference the canonical/negative dataset directories of the real run.
            for canonical_ref in ("NEG_RAW_DIR", "NEG_FILTERED_DIR", "GEN_N_RAW", "GEN_MODEL_PATH"):
                self.assertNotIn(canonical_ref, cell)

    def test_smoke_fails_explicitly_when_fewer_than_two_gpus_resolved(self) -> None:
        for name in ("07_LDM_SDVAE_Extra1361.ipynb", "08_LDM_v3_SDVAE_FromScratch.ipynb"):
            cell = self._smoke_cell_source(name)
            guard_index = cell.index("len(smoke_devices) < 2")
            raise_index = cell.index("raise RuntimeError", guard_index)
            self.assertLess(guard_index, raise_index)
            self.assertLess(raise_index, cell.index("SMOKE MULTI-GPU SUPERATO"))

    def test_smoke_prints_success_marker_verbatim(self) -> None:
        for name in ("07_LDM_SDVAE_Extra1361.ipynb", "08_LDM_v3_SDVAE_FromScratch.ipynb"):
            cell = self._smoke_cell_source(name)
            self.assertIn('print("SMOKE MULTI-GPU SUPERATO")', cell)

    def test_notebooks_document_cuda_visible_devices_inheritance_before_generation(self) -> None:
        for name in ("07_LDM_SDVAE_Extra1361.ipynb", "08_LDM_v3_SDVAE_FromScratch.ipynb"):
            source = (ROOT / "notebooks" / "2_diffusers" / name).read_text(encoding="utf-8")
            self.assertIn('CUDA_VISIBLE_DEVICES ereditato', source)
            self.assertIn('GENERATION_GPU_DEVICES richiesto', source)
            self.assertIn("print_gpu_resolution_dry_run", source)
            diagnostic_index = source.index("CUDA_VISIBLE_DEVICES ereditato")
            first_real_generation_index = source.index("GEN_N_RAW = 4083")
            self.assertLess(diagnostic_index, first_real_generation_index)


if __name__ == "__main__":
    unittest.main(verbosity=2)
