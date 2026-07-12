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
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

from classifier_gpu_scheduler import GPU_TARGETS, Scheduler, load_vram_profiles, query_gpus_live  # noqa: E402
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
        if state in ("PENDING", "TRAINED", "VALIDATING", "FAILED_RETRYABLE"):
            pending.append({**job, "status": state})
    return pending


def launch_job(root: Path, job: dict, gpu_index: int, mode: str = "auto") -> subprocess.Popen:
    import os
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    cmd = [sys.executable, "-m", "notebooks.utility.classifier_experiment_runner",
           "--experiment-id", job["experiment_id"], "--mode", mode, "--project-root", str(root)]
    return subprocess.Popen(cmd, cwd=str(root), env=env)


def run(root: Path, stage: int, mode: str, target_5060: int, target_3060: int, dry_run: bool,
        gpus: list[dict] | None = None, poll_interval: float = 5.0) -> dict:
    GPU_TARGETS["rtx_5060_ti_16gb"]["target_max_jobs"] = target_5060
    GPU_TARGETS["rtx_3060_12gb"]["target_max_jobs"] = target_3060

    gpus = gpus if gpus is not None else query_gpus_live()
    vram_profiles = load_vram_profiles(root / "results/runtime_profiles/classifier_vram_profiles.json")
    scheduler = Scheduler(gpus, vram_profiles)
    index_by_uuid = {g["uuid"]: g["index"] for g in gpus}

    jobs = pending_jobs(root, stage)
    plan = scheduler.schedule_batch(jobs)

    if dry_run:
        return {"mode": mode, "dry_run": True, "gpus": [g["name"] for g in gpus], "plan": plan}

    completed = []
    while jobs:
        batch_plan = scheduler.schedule_batch(jobs)
        processes: dict[str, tuple[subprocess.Popen, str, dict]] = {}
        for job, decision in zip(sorted(jobs, key=lambda j: j["experiment_id"]), batch_plan):
            if not decision["admitted"]:
                continue
            gpu_index = index_by_uuid[decision["gpu_uuid"]]
            proc = launch_job(root, job, gpu_index, mode="auto")
            processes[job["experiment_id"]] = (proc, decision["gpu_key"], job)
        if not processes:
            break
        attempted_ids = set(processes)
        while processes:
            time.sleep(poll_interval)
            for experiment_id, (proc, gpu_key, job) in list(processes.items()):
                if proc.poll() is not None:
                    completed.append({"experiment_id": experiment_id, "returncode": proc.returncode,
                                      "state": run_manifest.reconstruct_state(
                                          checkpoint_io.run_dir(root, job["architecture"], job["dataset_variant_id"],
                                                                job["training_policy"], job["seed"]),
                                          json.loads((root / "configs/classifier_training_protocols.json").read_text())
                                              ["policies"][job["architecture"]]["framework"])["state"]})
                    scheduler.release(job, gpu_key)
                    del processes[experiment_id]
        jobs = pending_jobs(root, stage)
        # A failed job remains retryable for resume/OOM handling; do not spin forever here.
        jobs = [job for job in jobs if job["experiment_id"] not in attempted_ids]
    return {"mode": mode, "dry_run": False, "gpus": [g["name"] for g in gpus], "plan": plan, "completed": completed}


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
