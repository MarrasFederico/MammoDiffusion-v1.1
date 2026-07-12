"""Adaptive multi-GPU scheduler for the classifier experiment matrix (spec section 10).

GPUs are identified by *name* (and reported alongside their UUID for logging/provenance),
never by CUDA device index: index order is not guaranteed stable across drivers/reboots, and is
verified different from how this project's own operating prompt labelled the two cards on this
exact machine (nvidia-smi here reports index 0 = RTX 3060, index 1 = RTX 5060 Ti). A job only
ever receives a single physical device via CUDA_VISIBLE_DEVICES; it never sees both GPUs.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field

GPU_TARGETS = {
    "rtx_5060_ti_16gb": {"name_match": "5060", "target_max_jobs": 3, "reserve_vram_mb": 1800},
    "rtx_3060_12gb": {"name_match": "3060", "target_max_jobs": 2, "reserve_vram_mb": 1500},
}

# Soft ceiling on host-wide concurrent jobs regardless of how many GPU slots look free,
# to avoid oversubscribing dataloader worker processes / CPU threads (spec 10.7).
HOST_MAX_CONCURRENT_JOBS = 5
DEFAULT_DATALOADER_WORKERS = 3
DEFAULT_CPU_THREADS_PER_JOB = 4
MIN_RAM_RESERVE_MB = 12 * 1024

# Priority: exclusive/heavy jobs go first so they land while a GPU is emptiest; light/medium
# jobs backfill remaining slots (spec 10.8).
_PROFILE_PRIORITY = {"exclusive": 0, "heavy": 1, "medium": 2, "light": 3}

VRAM_SAFETY_MARGIN = 1.20  # admit only if free_vram >= estimated_peak * 1.20 + reserve


def query_gpus_live() -> list[dict]:
    """Real nvidia-smi query. Not called by anything importable/testable by default — pass an
    explicit `gpus` list to Scheduler in tests and in any dry-run to avoid depending on hardware.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    gpus = []
    for line in out.strip().splitlines():
        index, name, uuid, total, free = (part.strip() for part in line.split(","))
        gpus.append({"index": int(index), "name": name, "uuid": uuid, "total_vram_mb": int(total), "free_vram_mb": int(free)})
    return gpus


def classify_gpu(gpu: dict) -> str | None:
    for key, spec in GPU_TARGETS.items():
        if spec["name_match"] in gpu["name"]:
            return key
    return None


def admission_check(gpu_key: str, jobs_on_gpu: int, estimated_peak_mb: float, free_vram_mb: float) -> tuple[bool, str]:
    spec = GPU_TARGETS[gpu_key]
    if jobs_on_gpu >= spec["target_max_jobs"]:
        return False, f"target_max_jobs ({spec['target_max_jobs']}) reached on {gpu_key}"
    required = estimated_peak_mb * VRAM_SAFETY_MARGIN + spec["reserve_vram_mb"]
    if free_vram_mb < required:
        return False, f"insufficient free VRAM on {gpu_key}: need {required:.0f}MB (peak*{VRAM_SAFETY_MARGIN}+reserve), have {free_vram_mb:.0f}MB"
    return True, "admitted"


@dataclass
class OomState:
    physical_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    oom_count: int = 0
    forced_exclusive: bool = False
    history: list[dict] = field(default_factory=list)

    def record_oom(self) -> "OomState":
        """Spec 10.6: on first OOM halve the physical batch, double accumulation to keep the
        effective batch size fixed, and reduce concurrency; retry once; a second OOM forces
        the job to `exclusive`. Every change is recorded, never silent.
        """
        self.oom_count += 1
        event = {"oom_number": self.oom_count, "previous_physical_batch_size": self.physical_batch_size}
        if self.oom_count == 1:
            new_physical = max(1, self.physical_batch_size // 2)
            new_accum = self.gradient_accumulation_steps * (self.physical_batch_size // new_physical)
            self.physical_batch_size = new_physical
            self.gradient_accumulation_steps = new_accum
        else:
            self.forced_exclusive = True
        event["new_physical_batch_size"] = self.physical_batch_size
        event["new_gradient_accumulation_steps"] = self.gradient_accumulation_steps
        event["forced_exclusive"] = self.forced_exclusive
        assert self.physical_batch_size * self.gradient_accumulation_steps == self.effective_batch_size, \
            "effective batch size must never change silently during OOM recovery"
        self.history.append(event)
        return self

    def should_retry(self) -> bool:
        # first retry uses smaller physical batch; second retry is exclusive; third is final.
        return self.oom_count <= 2


class Scheduler:
    """Pure, injectable admission-control scheduler: `gpus` and `vram_profiles` are plain data
    so this class needs no real hardware or subprocess calls to test (spec 19.3).
    """

    def __init__(self, gpus: list[dict], vram_profiles: dict[str, dict] | None = None):
        self.gpus = [dict(g, gpu_key=classify_gpu(g)) for g in gpus]
        for g in self.gpus:
            if g["gpu_key"] is None:
                raise ValueError(f"unrecognized GPU name (identify by UUID/name, never assume index): {g['name']}")
        self.vram_profiles = vram_profiles or {}
        self._running: dict[str, list[dict]] = {g["gpu_key"]: [] for g in self.gpus}

    def _estimated_peak_mb(self, job: dict) -> float:
        key = f"{job['architecture']}::{job['resource_profile']}"
        profile = self.vram_profiles.get(key) or self.vram_profiles.get(job["architecture"])
        if profile and "peak_allocated_mb" in profile:
            return float(profile["peak_allocated_mb"])
        # Conservative fallback when no probe exists yet: never admit heavy/exclusive jobs
        # blindly, but let light/medium jobs through with a cautious guess.
        fallback = {"light": 3000.0, "medium": 6000.0, "heavy": 11000.0, "exclusive": 15000.0}
        return fallback.get(job["resource_profile"], 8000.0)

    def total_running_jobs(self) -> int:
        return sum(len(v) for v in self._running.values())

    def eligible_gpus_for(self, job: dict) -> list[dict]:
        return [g for g in self.gpus if g["gpu_key"] in job.get("gpu_eligibility", [g["gpu_key"] for g in self.gpus])]

    def try_admit(self, job: dict) -> dict:
        if self.total_running_jobs() >= HOST_MAX_CONCURRENT_JOBS:
            return {"admitted": False, "reason": f"host_max_concurrent_jobs ({HOST_MAX_CONCURRENT_JOBS}) reached"}

        if job["resource_profile"] == "exclusive":
            for g in self.eligible_gpus_for(job):
                if self._running[g["gpu_key"]]:
                    return {"admitted": False, "reason": f"exclusive job requires an idle GPU; {g['gpu_key']} is busy"}

        estimated = self._estimated_peak_mb(job)
        candidates = []
        for g in self.eligible_gpus_for(job):
            jobs_on_gpu = len(self._running[g["gpu_key"]])
            used = sum(self._estimated_peak_mb(j) for j in self._running[g["gpu_key"]])
            free = max(0.0, g["free_vram_mb"] - used)
            ok, reason = admission_check(g["gpu_key"], jobs_on_gpu, estimated, free)
            if ok:
                candidates.append((g, free))
        if not candidates:
            return {"admitted": False, "reason": "no eligible GPU has enough free VRAM / slot capacity"}

        # Prefer the GPU that would end up most utilized (bin-packing) rather than spreading thin,
        # unless spreading across distinct frameworks reduces TensorFlow/PyTorch preallocation contention.
        chosen, _free = min(candidates, key=lambda pair: pair[1])
        self._running[chosen["gpu_key"]].append(job)
        return {"admitted": True, "gpu_key": chosen["gpu_key"], "gpu_uuid": chosen["uuid"], "estimated_peak_mb": estimated}

    def release(self, job: dict, gpu_key: str) -> None:
        self._running[gpu_key] = [j for j in self._running[gpu_key] if j["experiment_id"] != job["experiment_id"]]

    def schedule_batch(self, jobs: list[dict]) -> list[dict]:
        """Order-independent priority pass: exclusive/heavy first, then backfill light/medium."""
        ordered = sorted(jobs, key=lambda j: _PROFILE_PRIORITY.get(j["resource_profile"], 2))
        results = []
        for job in ordered:
            results.append({"experiment_id": job["experiment_id"], **self.try_admit(job)})
        return results


def load_vram_profiles(path) -> dict:
    from pathlib import Path
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text())
