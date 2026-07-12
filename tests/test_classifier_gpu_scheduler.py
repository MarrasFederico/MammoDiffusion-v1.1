from __future__ import annotations
import sys, unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import classifier_gpu_scheduler as sched  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import run_classifier_experiment_matrix as matrix_runner  # noqa: E402


def gpus_5060_and_3060(free_5060=15000, free_3060=11000):
    return [
        {"index": 1, "name": "NVIDIA GeForce RTX 5060 Ti", "uuid": "gpu-5060", "total_vram_mb": 16384, "free_vram_mb": free_5060},
        {"index": 0, "name": "NVIDIA GeForce RTX 3060", "uuid": "gpu-3060", "total_vram_mb": 12288, "free_vram_mb": free_3060},
    ]


def profiles(peak_mb=2000.0):
    return {"maxvit512": {"peak_allocated_mb": peak_mb}, "resnet50": {"peak_allocated_mb": peak_mb},
            "mammofm": {"peak_allocated_mb": peak_mb}, "raddino": {"peak_allocated_mb": peak_mb}}


def job(eid, architecture="maxvit512", profile="medium", eligibility=None):
    return {"experiment_id": eid, "architecture": architecture, "resource_profile": profile,
            "gpu_eligibility": eligibility or ["rtx_5060_ti_16gb", "rtx_3060_12gb"]}


class GpuIdentificationTests(unittest.TestCase):
    def test_classify_by_name_not_index(self):
        # index 0 is the 3060 here (as it genuinely is on the dev machine) — classification
        # must not assume index 0 == the bigger card.
        gpus = gpus_5060_and_3060()
        self.assertEqual(sched.classify_gpu(gpus[0]), "rtx_5060_ti_16gb")
        self.assertEqual(sched.classify_gpu(gpus[1]), "rtx_3060_12gb")

    def test_unrecognized_gpu_name_raises_instead_of_guessing(self):
        with self.assertRaises(ValueError):
            sched.Scheduler([{"index": 0, "name": "Some Unknown GPU", "uuid": "x", "total_vram_mb": 8000, "free_vram_mb": 8000}])


class AdmissionControlTests(unittest.TestCase):
    def test_three_slots_on_5060_two_on_3060(self):
        s = sched.Scheduler(gpus_5060_and_3060(free_5060=60000, free_3060=60000), profiles())
        jobs = [job(f"a{i}", eligibility=["rtx_5060_ti_16gb"]) for i in range(4)] + \
               [job(f"b{i}", eligibility=["rtx_3060_12gb"]) for i in range(3)]
        plan = s.schedule_batch(jobs)
        admitted_5060 = [p for p in plan if p["admitted"] and p["gpu_key"] == "rtx_5060_ti_16gb"]
        admitted_3060 = [p for p in plan if p["admitted"] and p["gpu_key"] == "rtx_3060_12gb"]
        self.assertEqual(len(admitted_5060), 3)
        self.assertEqual(len(admitted_3060), 2)

    def test_insufficient_vram_rejects_admission(self):
        s = sched.Scheduler(gpus_5060_and_3060(free_5060=1000, free_3060=1000), profiles(peak_mb=5000))
        result = s.try_admit(job("big", profile="heavy"))
        self.assertFalse(result["admitted"])
        self.assertIn("VRAM", result["reason"])

    def test_exclusive_profile_requires_fully_idle_eligible_gpu(self):
        s = sched.Scheduler(gpus_5060_and_3060(free_5060=60000, free_3060=60000), profiles())
        s.try_admit(job("occupant", eligibility=["rtx_5060_ti_16gb"]))
        result = s.try_admit(job("excl", profile="exclusive", eligibility=["rtx_5060_ti_16gb"]))
        self.assertFalse(result["admitted"])

    def test_exclusive_profile_admits_onto_idle_gpu(self):
        s = sched.Scheduler(gpus_5060_and_3060(free_5060=60000, free_3060=60000), profiles())
        result = s.try_admit(job("excl", profile="exclusive", eligibility=["rtx_3060_12gb"]))
        self.assertTrue(result["admitted"])

    def test_host_max_concurrent_jobs_caps_total_regardless_of_free_vram(self):
        s = sched.Scheduler(gpus_5060_and_3060(free_5060=999999, free_3060=999999), profiles(peak_mb=10))
        jobs = [job(f"j{i}") for i in range(sched.HOST_MAX_CONCURRENT_JOBS + 3)]
        plan = s.schedule_batch(jobs)
        self.assertEqual(sum(1 for p in plan if p["admitted"]), sched.HOST_MAX_CONCURRENT_JOBS)

    def test_release_frees_the_slot_for_a_later_job(self):
        # free=5000 admits exactly one peak=2000 job (needs 2000*1.20+1800=4200); after that
        # job is running, 5000-2000=3000 free is not enough for a second (still needs 4200).
        s = sched.Scheduler(gpus_5060_and_3060(free_5060=5000, free_3060=100), profiles(peak_mb=2000))
        first = s.try_admit(job("first", eligibility=["rtx_5060_ti_16gb"]))
        self.assertTrue(first["admitted"])
        blocked = s.try_admit(job("second", eligibility=["rtx_5060_ti_16gb"]))
        self.assertFalse(blocked["admitted"])
        s.release(job("first", eligibility=["rtx_5060_ti_16gb"]), "rtx_5060_ti_16gb")
        retried = s.try_admit(job("second", eligibility=["rtx_5060_ti_16gb"]))
        self.assertTrue(retried["admitted"])

    def test_priority_orders_exclusive_and_heavy_before_light_and_medium(self):
        jobs = [job("light1", profile="light"), job("exclusive1", profile="exclusive"), job("medium1", profile="medium")]
        s = sched.Scheduler(gpus_5060_and_3060(free_5060=60000, free_3060=60000), profiles())
        # exclusive1 must be attempted (and admitted) before any other job takes its GPU
        plan = s.schedule_batch(jobs)
        ids_in_order = [p["experiment_id"] for p in plan]
        self.assertEqual(ids_in_order[0], "exclusive1")

    def test_missing_vram_profile_uses_conservative_fallback_not_zero(self):
        s = sched.Scheduler(gpus_5060_and_3060(), vram_profiles={})
        estimated = s._estimated_peak_mb(job("x", profile="heavy"))
        self.assertGreater(estimated, 0)


class OomStateMachineTests(unittest.TestCase):
    def test_first_oom_halves_batch_and_doubles_accumulation_preserving_effective_batch(self):
        state = sched.OomState(physical_batch_size=16, gradient_accumulation_steps=1, effective_batch_size=16)
        state.record_oom()
        self.assertEqual(state.physical_batch_size, 8)
        self.assertEqual(state.gradient_accumulation_steps, 2)
        self.assertEqual(state.physical_batch_size * state.gradient_accumulation_steps, 16)
        self.assertFalse(state.forced_exclusive)
        self.assertTrue(state.should_retry())

    def test_second_oom_forces_one_exclusive_retry(self):
        state = sched.OomState(physical_batch_size=16, gradient_accumulation_steps=1, effective_batch_size=16)
        state.record_oom()
        state.record_oom()
        self.assertTrue(state.forced_exclusive)
        self.assertTrue(state.should_retry())
        state.record_oom()
        self.assertFalse(state.should_retry())

    def test_oom_history_is_never_silent(self):
        state = sched.OomState(physical_batch_size=16, gradient_accumulation_steps=1, effective_batch_size=16)
        state.record_oom()
        self.assertEqual(len(state.history), 1)
        self.assertIn("previous_physical_batch_size", state.history[0])


class SchedulerExecutionTests(unittest.TestCase):
    def test_preview_does_not_consume_slots(self):
        gpu = [{"index": 0, "name": "NVIDIA GeForce RTX 3060", "uuid": "gpu-a", "total_vram_mb": 12000, "free_vram_mb": 12000}]
        scheduler = sched.Scheduler(gpu)
        candidate = job("preview", architecture="resnet50", profile="light", eligibility=["rtx_3060_12gb"])
        self.assertTrue(scheduler.preview_batch([candidate])[0]["admitted"])
        self.assertTrue(scheduler.try_admit(candidate)["admitted"])

    def test_non_dry_run_launches_a_child(self):
        gpu = [{"index": 0, "name": "NVIDIA GeForce RTX 3060", "uuid": "gpu-a", "total_vram_mb": 12000, "free_vram_mb": 12000}]
        candidate = {**job("resnet50__R__seed17", architecture="resnet50", profile="light", eligibility=["rtx_3060_12gb"]),
                     "dataset_variant_id": "R", "training_policy": "resnet50_standard", "seed": 17}
        class Process:
            returncode = 0
            def poll(self): return 0
        with patch.object(matrix_runner, "pending_jobs", side_effect=[[candidate], []]), \
             patch.object(matrix_runner, "launch_job", return_value=Process()) as launch, \
             patch.object(matrix_runner.run_manifest, "reconstruct_state", return_value={"state": "TRAINED"}), \
             patch("time.sleep"):
            result = matrix_runner.run(ROOT, 1, "auto", 1, 1, False, gpus=gpu, poll_interval=0)
        self.assertTrue(launch.called)
        self.assertEqual(result["completed"][0]["experiment_id"], candidate["experiment_id"])


if __name__ == "__main__":
    unittest.main()
