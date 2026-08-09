#!/usr/bin/env python3
"""CUDA-isolated Stable Diffusion generation worker.

The parent writes one JSON job per checkpoint (evaluation) or per index shard
(final generation). This process sees exactly one GPU through
``CUDA_VISIBLE_DEVICES`` and therefore always uses logical ``cuda:0``.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MammoDiffusion Stable Diffusion generation worker")
    parser.add_argument("--job-file", type=Path, required=True)
    return parser.parse_args()


def valid_png(path: Path) -> bool:
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def save_atomic(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_gen_{path.stem}_{os.getpid()}.png")
    image.save(tmp)
    if not valid_png(tmp):
        raise RuntimeError(f"Temporary PNG is unreadable: {tmp}")
    os.replace(tmp, path)


def missing_indices(directory: Path, count: int) -> list[int]:
    missing = []
    for index in range(count):
        path = directory / f"gen_{index:04d}.png"
        if not valid_png(path):
            missing.append(index)
    return missing


def load_pipeline(job: dict):
    import torch
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel

    dtype = torch.float16
    checkpoint = Path(job["checkpoint_path"])
    base_model = str(job["base_model_dir"])
    if job["checkpoint_type"] == "lora":
        pipeline = StableDiffusionPipeline.from_pretrained(
            base_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
        )
        pipeline.load_lora_weights(str(checkpoint))
    else:
        unet = UNet2DConditionModel.from_pretrained(str(checkpoint / "unet"), torch_dtype=dtype)
        pipeline = StableDiffusionPipeline.from_pretrained(
            base_model, unet=unet, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
        )
    pipeline = pipeline.to("cuda:0")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def main() -> None:
    args = parse_args()
    job = json.loads(args.job_file.read_text(encoding="utf-8"))
    requests = job["requests"]
    pending = []
    for request in requests:
        out_dir = Path(request["out_dir"])
        indices = request.get("indices")
        if request.get("dynamic_queue_dir"):
            indices = [0] if any(Path(request["dynamic_queue_dir"]).joinpath("pending").glob("chunk_*.pending.json")) else []
        elif indices is None:
            indices = missing_indices(out_dir, int(request["count"]))
        else:
            indices = [int(index) for index in indices if not valid_png(out_dir / f"gen_{int(index):04d}.png")]
        pending.append((request, indices))
    if not any(indices for _, indices in pending):
        print("No missing images assigned; worker exits without loading Stable Diffusion.")
        return

    import torch
    pipeline = load_pipeline(job)
    started = time.perf_counter()
    stats = {"worker_id": job.get("worker_id", os.getpid()), "gpu_physical_id": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
             "chunks_claimed": 0, "chunks_completed": 0, "indices_reserved": 0,
             "images_generated": 0, "images_already_completed": 0, "images_failed": 0}
    try:
        for request, indices in pending:
            queue_dir = request.get("dynamic_queue_dir")
            if queue_dir:
                from parallel_generation_utils import claim_index, claim_next_chunk, complete_claimed_chunk, release_index_claim
                while True:
                    claimed = claim_next_chunk(Path(queue_dir), str(stats["worker_id"]))
                    if claimed is None:
                        break
                    claimed_path, chunk = claimed; stats["chunks_claimed"] += 1
                    try:
                        for index in chunk["indices"]:
                            final = Path(request["out_dir"]) / f"gen_{int(index):04d}.png"
                            if valid_png(final):
                                stats["images_already_completed"] += 1
                                continue
                            reservation = claim_index(Path(queue_dir), int(index), int(chunk["chunk_id"]), str(stats["worker_id"]), stats["gpu_physical_id"])
                            if reservation is None:
                                if valid_png(final): stats["images_already_completed"] += 1
                                continue
                            stats["indices_reserved"] += 1
                            try:
                                seed = int(request["seed"]) + int(request.get("class_offset", 0)) + int(index)
                                generator = torch.Generator(device="cuda:0").manual_seed(seed)
                                image = pipeline(request["prompt"], num_inference_steps=int(request["inference_steps"]),
                                    guidance_scale=float(request["guidance_scale"]), height=int(request["resolution"]),
                                    width=int(request["resolution"]), generator=generator).images[0]
                                save_atomic(image, final); stats["images_generated"] += 1
                                print(f"generated {request.get('name', 'class')} index={index} chunk={chunk['chunk_id']}", flush=True)
                            except Exception:
                                stats["images_failed"] += 1
                                raise
                            finally:
                                release_index_claim(reservation)
                        complete_claimed_chunk(Path(queue_dir), claimed_path, chunk); stats["chunks_completed"] += 1
                    except Exception:
                        raise
                continue
            for index in indices:
                # Per-image seed: independent of process, GPU count and execution order.
                seed = int(request["seed"]) + int(request.get("class_offset", 0)) + index
                generator = torch.Generator(device="cuda:0").manual_seed(seed)
                image = pipeline(
                    request["prompt"],
                    num_inference_steps=int(request["inference_steps"]),
                    guidance_scale=float(request["guidance_scale"]),
                    height=int(request["resolution"]),
                    width=int(request["resolution"]),
                    generator=generator,
                ).images[0]
                save_atomic(image, Path(request["out_dir"]) / f"gen_{index:04d}.png")
                print(f"generated {request.get('name', 'class')} index={index}", flush=True)
    finally:
        stats["wall_clock_seconds"] = time.perf_counter() - started
        stats["images_per_second"] = stats["images_generated"] / stats["wall_clock_seconds"] if stats["wall_clock_seconds"] else 0.0
        queue_dirs = {request.get("dynamic_queue_dir") for request, _ in pending if request.get("dynamic_queue_dir")}
        for queue_dir in queue_dirs:
            path = Path(queue_dir) / "worker_stats" / f"worker_{stats['worker_id']}_{os.getpid()}.json"
            path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
