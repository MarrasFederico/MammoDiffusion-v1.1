#!/usr/bin/env python3
"""Adaptive multi-GPU orchestrator for the classifier experiment matrix (spec sections 9/10).

    python scripts/run_classifier_experiment_matrix.py --stage 1 --mode auto \\
        --target-5060-jobs 3 --target-3060-jobs 2

    # Inspect the plan without launching anything:
    python scripts/run_classifier_experiment_matrix.py --stage 1 --mode auto --dry-run

Each admitted job is launched as its own subprocess with CUDA_VISIBLE_DEVICES pinned to a
single physical GPU resolved by UUID (never by index — see classifier_gpu_scheduler docstring),
so no job ever sees both cards. This script never trains anything itself; it only ever shells
out to classifier_experiment_runner.py, exactly the module already covered by
test_classifier_runner_resume.py.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

from classifier_gpu_scheduler import GPU_TARGETS, OomState, Scheduler, load_vram_profiles, query_gpus_live  # noqa: E402
import classifier_run_manifest as run_manifest  # noqa: E402
import classifier_checkpoint_io as checkpoint_io  # noqa: E402


def pending_jobs(root: Path, stage: int) -> list[dict]:
    matrix = json.loads((root / "configs/classifier_experiment_matrix.json").read_text())
    protocols = json.loads((root / "configs/classifier_training_protocols.json").read_text())["policies"]
    pending = []
    for job in matrix["jobs"]:
        if job["stage"] != stage:
            continue
        run = checkpoint_io.run_dir(root, job["architecture"], job["dataset_variant_id"],
                                    job["training_policy"], job["seed"])
        state = run_manifest.reconstruct_state(run, protocols[job["architecture"]]["framework"])["state"]
        if state in ("PENDING", "INTERRUPTED_RESUMABLE", "TRAINED", "VALIDATING", "VALIDATED", "FAILED_RETRYABLE"):
            pending.append({**job, "status": state})
    return pending


THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4", "TF_NUM_INTRAOP_THREADS": "4", "TF_NUM_INTEROP_THREADS": "2",
}


def launch_job(root: Path, job: dict, gpu_index: int, mode: str = "auto") -> subprocess.Popen:
    import os
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    # With up to 5 concurrent jobs (spec 10.7 host cap), each process defaulting to
    # framework-detected thread counts (often == host core count) oversubscribes the CPU by
    # 5x; pin every launched worker to the policy's declared thread budget instead.
    env.update(THREAD_ENV_VARS)
    cmd = [sys.executable, "-m", "notebooks.utility.classifier_experiment_runner",
           "--experiment-id", job["experiment_id"], "--mode", mode, "--project-root", str(root)]
    return subprocess.Popen(cmd, cwd=str(root), env=env)


def load_or_create_oom_state(run_dir: Path, job: dict, protocol: dict) -> OomState:
    """Seed OomState from a prior session's on-disk oom_override.json when present, so a
    scheduler restart (Ctrl+C -> new process -> same job) never resets retry/exclusive
    progress back to "first OOM".
    """
    override_path = run_dir / "oom_override.json"
    if override_path.is_file():
        try:
            prior = json.loads(override_path.read_text())
            return OomState(
                physical_batch_size=int(prior["physical_batch_size"]),
                gradient_accumulation_steps=int(prior["gradient_accumulation_steps"]),
                effective_batch_size=int(prior["effective_batch_size"]),
                oom_count=int(prior.get("oom_count", 0)),
                forced_exclusive=bool(prior.get("forced_exclusive", False)),
                history=list(prior.get("history", [])),
            )
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            pass  # corrupt override file: fall through to a fresh state below
    return OomState(
        physical_batch_size=int(job.get("physical_batch_size", protocol["physical_batch_size"])),
        gradient_accumulation_steps=int(job.get("gradient_accumulation_steps", protocol["gradient_accumulation_steps"])),
        effective_batch_size=int(job.get("effective_batch_size", protocol["effective_batch_size"])))


def run(root: Path, stage: int, mode: str, target_5060: int, target_3060: int, dry_run: bool,
        gpus: list[dict] | None = None, poll_interval: float = 5.0) -> dict:
    GPU_TARGETS["rtx_5060_ti_16gb"]["target_max_jobs"] = target_5060
    GPU_TARGETS["rtx_3060_12gb"]["target_max_jobs"] = target_3060

    gpus = gpus if gpus is not None else query_gpus_live()
    vram_profiles = load_vram_profiles(root / "results/runtime_profiles/classifier_vram_profiles.json")
    scheduler = Scheduler(gpus, vram_profiles)
    index_by_uuid = {g["uuid"]: g["index"] for g in gpus}

    jobs = pending_jobs(root, stage)
    if dry_run:
        plan = scheduler.preview_batch(jobs)
        return {"mode": mode, "dry_run": True, "gpus": [g["name"] for g in gpus], "plan": plan}

    completed = []
    oom_states = {}
    while jobs:
        batch_plan = scheduler.schedule_batch(jobs)
        processes: dict[str, tuple[subprocess.Popen, str, dict]] = {}
        decisions = {row["experiment_id"]: row for row in batch_plan}
        for job in jobs:
            decision = decisions[job["experiment_id"]]
            if not decision["admitted"]:
                continue
            gpu_index = index_by_uuid[decision["gpu_uuid"]]
            proc = launch_job(root, job, gpu_index, mode="auto")
            processes[job["experiment_id"]] = (proc, decision["gpu_key"], job)
        if not processes:
            break
        attempted_ids = set(processes)
        try:
            while processes:
                time.sleep(poll_interval)
                for experiment_id, (proc, gpu_key, job) in list(processes.items()):
                    if proc.poll() is not None:
                        run_dir = checkpoint_io.run_dir(root, job["architecture"], job["dataset_variant_id"], job["training_policy"], job["seed"])
                        state_payload = run_manifest.read_manifest(run_dir) or {}
                        error = str(state_payload.get("error", "")).lower()
                        is_oom = proc.returncode != 0 and ("out of memory" in error or "resourceexhausted" in error)
                        if is_oom:
                            protocol = json.loads((root / "configs/classifier_training_protocols.json").read_text())["policies"][job["architecture"]]
                            if experiment_id not in oom_states:
                                oom_states[experiment_id] = load_or_create_oom_state(run_dir, job, protocol)
                            oom = oom_states[experiment_id]
                            oom.record_oom()
                            override = {"physical_batch_size": oom.physical_batch_size,
                                        "gradient_accumulation_steps": oom.gradient_accumulation_steps,
                                        "effective_batch_size": oom.effective_batch_size,
                                        "oom_count": oom.oom_count, "forced_exclusive": oom.forced_exclusive,
                                        "history": oom.history}
                            (run_dir / "oom_override.json").write_text(json.dumps(override, indent=2) + "\n")
                            if oom.should_retry():
                                job.update(override)
                                if oom.forced_exclusive: job["resource_profile"] = "exclusive"
                            else:
                                run_manifest.write_state(run_dir, "FAILED_FINAL", error="OOM retry limit exhausted", oom=override)
                        completed.append({"experiment_id": experiment_id, "returncode": proc.returncode,
                                          "state": run_manifest.reconstruct_state(
                                              checkpoint_io.run_dir(root, job["architecture"], job["dataset_variant_id"],
                                                                    job["training_policy"], job["seed"]),
                                              json.loads((root / "configs/classifier_training_protocols.json").read_text())
                                                  ["policies"][job["architecture"]]["framework"])["state"]})
                        scheduler.release(job, gpu_key)
                        del processes[experiment_id]
        except KeyboardInterrupt:
            # SIGTERM gives the shared runner a chance to checkpoint and mark resumable.
            for proc, _gpu_key, _job in processes.values():
                if proc.poll() is None: proc.send_signal(signal.SIGTERM)
            for proc, _gpu_key, _job in processes.values():
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait(timeout=10)
            for _proc, gpu_key, job in processes.values():
                scheduler.release(job, gpu_key)
            return {"mode": mode, "interrupted": True, "message": "children stopped; rerun the same command to resume"}
        jobs = pending_jobs(root, stage)
        for job in jobs:
            oom = oom_states.get(job["experiment_id"])
            if oom:
                job.update({"physical_batch_size": oom.physical_batch_size,
                            "gradient_accumulation_steps": oom.gradient_accumulation_steps,
                            "effective_batch_size": oom.effective_batch_size})
                if oom.forced_exclusive: job["resource_profile"] = "exclusive"
        # A failed job remains retryable for resume/OOM handling; do not spin forever here.
        jobs = [job for job in jobs if job["experiment_id"] not in attempted_ids or
                (job["experiment_id"] in oom_states and oom_states[job["experiment_id"]].should_retry())]
    return {"mode": mode, "dry_run": False, "gpus": [g["name"] for g in gpus], "completed": completed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--mode", choices=("auto", "plan"), default="auto")
    parser.add_argument("--target-5060-jobs", type=int, default=3)
    parser.add_argument("--target-3060-jobs", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args()

    root = Path(args.project_root)
    dry_run = args.dry_run or args.mode == "plan"
    result = run(root, args.stage, args.mode, args.target_5060_jobs, args.target_3060_jobs, dry_run)
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
