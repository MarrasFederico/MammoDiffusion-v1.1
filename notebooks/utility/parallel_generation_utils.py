"""Utilities shared by the image-generation orchestrators.

The module deliberately has no TensorFlow or PyTorch imports at module import time:
the parent process must stay CUDA-free while it schedules one child process per GPU.
"""
from __future__ import annotations

import os
import json
import re
import subprocess
import sys
import time
import hashlib
from collections import deque
from pathlib import Path
from typing import Callable, Iterable


OFF_VALUES = {"", "off", "none", "false", "0-off"}
GENERATION_SCHEDULERS = {"dynamic_reservations", "round_robin", "auto", "checkpoint_queue"}
DEFAULT_GENERATION_SCHEDULER = "dynamic_reservations"
DEFAULT_GENERATION_RESERVATION_SIZE = 4
SD_SEED_STRATEGY = "per_image_seed_v2"
SD_SEED_OFFSETS = {
    "evaluation:negative": 0,
    "evaluation:positive": 1_000_000,
    "final_new:negative": 2_000_000,
    "final_new:positive": 3_000_000,
}


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def parse_generation_gpu_devices(requested_devices: str | None) -> tuple[str, list[str] | None]:
    """Parse ``auto``, ``off`` and comma-separated CUDA identifiers.

    CUDA identifiers intentionally remain strings: NVIDIA UUIDs are valid values for
    ``CUDA_VISIBLE_DEVICES`` and must not be coerced to integer ordinals.
    """
    if requested_devices is None:
        return "off", None
    value = str(requested_devices).strip()
    if value.lower() in OFF_VALUES:
        return "off", None
    if value.lower() == "auto":
        return "auto", None
    devices = _unique(value.split(","))
    if not devices:
        raise ValueError("--generation-gpus must be auto, off, or a non-empty comma-separated list")
    return "explicit", devices


def _devices_from_nvidia_smi() -> list[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    if completed.returncode:
        return []
    return _unique(completed.stdout.splitlines())


def _devices_from_available_framework() -> list[str]:
    """Best-effort framework fallback without requiring both frameworks."""
    try:
        import torch  # type: ignore

        count = int(torch.cuda.device_count())
        if count:
            return [str(index) for index in range(count)]
    except Exception:
        pass
    try:
        import tensorflow as tf  # type: ignore

        count = len(tf.config.list_physical_devices("GPU"))
        if count:
            return [str(index) for index in range(count)]
    except Exception:
        pass
    return []


def resolve_generation_gpu_devices(
    requested_devices: str | None,
    max_workers: int | None = None,
) -> list[str]:
    """Resolve requested CUDA devices and fail clearly when no CUDA GPU is usable."""
    if max_workers is not None and max_workers <= 0:
        raise ValueError("--max-generation-workers must be positive")
    mode, explicit = parse_generation_gpu_devices(requested_devices)
    inherited = _unique(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
    if mode == "explicit":
        devices = explicit or []
    elif mode == "auto":
        devices = inherited or _devices_from_nvidia_smi() or _devices_from_available_framework()
    else:
        # Parallelism is disabled, not CUDA. Pick exactly one already-selected or
        # discovered device so a CPU-only run is never silently attempted.
        devices = (inherited or _devices_from_nvidia_smi() or _devices_from_available_framework())[:1]
    devices = _unique(devices)
    if max_workers is not None:
        devices = devices[:max_workers]
    if not devices:
        raise RuntimeError(
            "No CUDA GPU available for generation. Set --generation-gpus to a valid "
            "CUDA device list or make at least one NVIDIA GPU visible."
        )
    return devices


def print_generation_diagnostics(requested_devices: str | None, devices: list[str]) -> None:
    mode = "off" if requested_devices is None else str(requested_devices)
    print(f"Generation GPU mode: {mode}")
    print(f"Resolved generation GPUs: {devices}")
    print(f"Generation workers: {len(devices)}")


def print_gpu_resolution_dry_run(
    requested_devices: str | None,
    max_workers: int | None = None,
) -> list[str]:
    """Explain GPU resolution without launching anything.

    ``auto`` silently respects an inherited ``CUDA_VISIBLE_DEVICES`` (see
    ``resolve_generation_gpu_devices``), which can look like "only one GPU is
    ever used" inside a Jupyter kernel that already restricts CUDA visibility.
    This prints physical/requested/resolved GPUs and the worker count so that
    behaviour is visible before a real run starts, instead of changing it.
    """
    print("GPU fisiche (nvidia-smi):", _devices_from_nvidia_smi() or "nessuna rilevata")
    print("CUDA_VISIBLE_DEVICES ereditato:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("GPU richieste (--generation-gpus):", requested_devices)
    try:
        devices = resolve_generation_gpu_devices(requested_devices, max_workers)
    except RuntimeError as exc:
        print("GPU risolte: ERRORE ->", exc)
        return []
    print("GPU risolte:", devices)
    print("Numero worker:", len(devices))
    return devices


def partition_indices(indices: Iterable[int], worker_count: int) -> list[list[int]]:
    """Round-robin partition with neither duplicate nor dropped indices."""
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    shards = [[] for _ in range(worker_count)]
    for position, index in enumerate(sorted(set(int(value) for value in indices))):
        shards[position % worker_count].append(index)
    return shards


def validate_queue_indices(indices: Iterable[int], target_count: int, valid_indices: Iterable[int] = ()) -> list[int]:
    values = [int(index) for index in indices]
    if any(index < 0 or index >= target_count for index in values):
        raise ValueError("queue contains out-of-range indices")
    if len(values) != len(set(values)):
        raise ValueError("queue contains duplicate indices")
    overlap = set(values) & {int(index) for index in valid_indices}
    if overlap:
        raise ValueError(f"queue contains already-valid indices: {sorted(overlap)[:20]}")
    return sorted(values)


def dynamic_chunks(indices: Iterable[int], reservation_size: int = DEFAULT_GENERATION_RESERVATION_SIZE) -> list[list[int]]:
    if reservation_size <= 0:
        raise ValueError("generation reservation size must be positive")
    values = list(indices)
    return [values[offset:offset + reservation_size] for offset in range(0, len(values), reservation_size)]


def prepare_dynamic_queue(
    run_dir: Path, indices: Iterable[int], *, target_count: int,
    valid_indices: Iterable[int] = (), corrupt_indices: Iterable[int] = (),
    reservation_size: int = DEFAULT_GENERATION_RESERVATION_SIZE,
    output_dir: Path, metadata: dict | None = None, dry_run: bool = False,
) -> dict:
    queue_indices = validate_queue_indices(indices, target_count, valid_indices)
    chunks = dynamic_chunks(queue_indices, reservation_size)
    summary = {
        "schema_version": 1, "scheduler": DEFAULT_GENERATION_SCHEDULER,
        "reservation_size": reservation_size, "target_count": target_count,
        "valid_indices_at_start": sorted(set(map(int, valid_indices))),
        "missing_indices_at_start": queue_indices,
        "corrupt_indices_at_start": sorted(set(map(int, corrupt_indices))),
        "queue_indices_hash": hashlib.sha256(json.dumps(queue_indices, separators=(",", ":")).encode()).hexdigest(),
        "output_dir": str(Path(output_dir).resolve()), "chunk_count": len(chunks), **(metadata or {}),
    }
    if dry_run:
        print(f"dynamic queue: target={target_count} valid={len(summary['valid_indices_at_start'])} "
              f"corrupt={len(summary['corrupt_indices_at_start'])} missing={len(queue_indices)} "
              f"chunks={len(chunks)} reservation_size={reservation_size} indices={queue_indices}")
        return summary
    for name in ("pending", "claimed", "done", "reservations", "worker_stats"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    _atomic_json(run_dir / "queue_manifest.json", summary)
    for chunk_id, chunk in enumerate(chunks):
        _atomic_json(run_dir / "pending" / f"chunk_{chunk_id:06d}.pending.json", {"chunk_id": chunk_id, "indices": chunk})
    return summary


def claim_next_chunk(queue_dir: Path, worker_id: str, pid: int | None = None) -> tuple[Path, dict] | None:
    pid = int(pid or os.getpid())
    for pending in sorted((Path(queue_dir) / "pending").glob("chunk_*.pending.json")):
        claimed = Path(queue_dir) / "claimed" / pending.name.replace(".pending.json", f".claimed.{pid}.{worker_id}.json")
        try:
            os.replace(pending, claimed)
        except FileNotFoundError:
            continue
        return claimed, json.loads(claimed.read_text(encoding="utf-8"))
    return None


def claim_index(queue_dir: Path, index: int, chunk_id: int, worker_id: str, gpu: str, pid: int | None = None) -> Path | None:
    pid = int(pid or os.getpid()); path = Path(queue_dir) / "reservations" / f"index_{int(index):06d}.claim"
    payload = {"index": int(index), "pid": pid, "worker_id": str(worker_id), "gpu": str(gpu), "timestamp": time.time(), "chunk_id": int(chunk_id)}
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try: owner = int(json.loads(path.read_text(encoding="utf-8")).get("pid", -1))
        except (OSError, ValueError, json.JSONDecodeError): owner = -1
        if owner > 0 and _pid_is_running(owner):
            return None
        try: path.unlink()
        except FileNotFoundError: pass
        try: fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError: return None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    return path


def release_index_claim(path: Path | None, pid: int | None = None) -> None:
    if path is None: return
    pid = int(pid or os.getpid())
    try:
        if int(json.loads(Path(path).read_text(encoding="utf-8")).get("pid", -1)) == pid:
            Path(path).unlink()
    except (OSError, ValueError, json.JSONDecodeError):
        pass


def complete_claimed_chunk(queue_dir: Path, claimed_path: Path, payload: dict) -> Path:
    done = Path(queue_dir) / "done" / f"chunk_{int(payload['chunk_id']):06d}.done.json"
    os.replace(claimed_path, done)
    return done


def sd_seed(base_seed: int, phase: str, class_name: str, logical_index: int) -> int:
    """Versioned, disjoint Stable Diffusion seed namespaces."""
    return int(base_seed) + SD_SEED_OFFSETS[f"{phase}:{class_name}"] + int(logical_index)


def _valid_png(path: Path) -> bool:
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def readable_png_paths(directory: Path, name_pattern: str | None = None) -> list[Path]:
    """Return readable, non-temporary PNGs, optionally restricted by regex."""
    expected = re.compile(name_pattern) if name_pattern else None
    paths = []
    for path in sorted(Path(directory).glob("*.png")):
        if path.name.startswith(".tmp_") or (expected and not expected.fullmatch(path.name)):
            continue
        if _valid_png(path):
            paths.append(path)
    return paths


def ldm_raw_png_paths(directory: Path) -> list[Path]:
    """Return only readable canonical RAW LDM files (synth_NNNNN.png)."""
    return readable_png_paths(directory, r"synth_\d{5}\.png")


def exact_filtered_png_paths(directory: Path, count: int) -> list[Path]:
    """Return the exact zero-based filtered set or raise with missing/extra names."""
    directory = Path(directory)
    readable = readable_png_paths(directory, r"synth_filtered_\d{4}\.png")
    expected = [directory / f"synth_filtered_{index:04d}.png" for index in range(int(count))]
    canonical = {path.name for path in directory.glob("synth_filtered_*.png")}
    expected_names = {path.name for path in expected}
    if readable != expected or canonical != expected_names:
        missing = sorted(expected_names - {path.name for path in readable})
        extra = sorted(canonical - expected_names)
        raise RuntimeError(f"Sequenza FILTERED non valida: missing={missing}, extra={extra}")
    return expected


def filtered_selection_cache_matches(
    summary_path: Path, input_signature: dict, filtered_dir: Path, n_selected: int
) -> bool:
    """Validate filter inputs and every expected readable filtered output."""
    try:
        payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("input_signature") != input_signature:
        return False
    try:
        exact_filtered_png_paths(filtered_dir, n_selected)
    except (OSError, RuntimeError):
        return False
    return True


GENERATED_PNG_PATTERN = (
    r"(?:gen_\d+|eval_(?:neg|pos)_\d+|synth_\d+|synth_filtered_\d+|selected_[A-Za-z0-9_]+_\d+)\.png"
)


def metric_image_paths(directory: Path, extensions: Iterable[str]) -> list[Path]:
    """Build the exact readable image list consumed by generative metrics."""
    extensions = {str(value).lower() for value in extensions}
    candidates = []
    for path in sorted(Path(directory).iterdir()):
        if path.name.startswith(".tmp_") or path.suffix.lower() not in extensions:
            continue
        if _valid_image(path):
            candidates.append(path)
    expected = re.compile(GENERATED_PNG_PATTERN, re.IGNORECASE)
    if any(expected.fullmatch(path.name) for path in candidates):
        return [path for path in candidates if expected.fullmatch(path.name)]
    return candidates


def _valid_image(path: Path) -> bool:
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def scan_named_png_set(directory: Path, count: int, prefix: str = "gen_") -> dict:
    """Scan an exact indexed PNG set without letting foreign/corrupt files count."""
    directory = Path(directory)
    expected_names = {f"{prefix}{index:04d}.png" for index in range(int(count))}
    valid, missing, corrupt = [], [], []
    for index in range(int(count)):
        path = directory / f"{prefix}{index:04d}.png"
        if not path.exists():
            missing.append(index)
        elif _valid_png(path):
            valid.append(index)
        else:
            corrupt.append(index)
    extra = [
        path.name for path in sorted(directory.glob("*.png"))
        if not path.name.startswith(".tmp_") and path.name not in expected_names
    ]
    return {
        "valid_indices": valid,
        "missing_indices": missing,
        "corrupt_indices": corrupt,
        "extra_files": extra,
        "complete": not missing and not corrupt and not extra,
    }


def valid_named_png_indices(directory: Path, count: int, prefix: str = "gen_") -> list[int]:
    """Only valid files matching ``prefixNNNN.png`` are dataset images."""
    return [
        index for index in range(int(count))
        if _valid_png(directory / f"{prefix}{index:04d}.png")
    ]


def final_sd_generation_plan(
    final_dir: Path,
    target_total: int,
    reused_prefix: str,
) -> dict:
    """Compute exact new ``gen_*`` slots after valid reused evaluation files."""
    reused = []
    corrupt_reused = []
    reused_indices = []
    expected = re.compile(rf"^{re.escape(reused_prefix)}_(\d{{4}})\.png$")
    for path in sorted(final_dir.glob(f"{reused_prefix}_*.png")):
        if path.name.startswith(".tmp_"):
            continue
        match = expected.fullmatch(path.name)
        if not match:
            continue
        index = int(match.group(1))
        if _valid_png(path):
            reused.append(path)
            reused_indices.append(index)
        else:
            corrupt_reused.append(path.name)
    reuse_record_path = Path(final_dir) / ".evaluation_reuse.json"
    expected_reused_count = 0
    if reuse_record_path.is_file():
        try:
            expected_reused_count = len(json.loads(reuse_record_path.read_text(encoding="utf-8")).get("files", []))
        except Exception:
            expected_reused_count = 0
    if not expected_reused_count and (reused_indices or corrupt_reused):
        all_indices = reused_indices + [
            int(match.group(1)) for name in corrupt_reused
            if (match := expected.fullmatch(name))
        ]
        expected_reused_count = max(all_indices, default=-1) + 1
    if expected_reused_count > target_total:
        raise RuntimeError(f"Too many reused slots in {final_dir}: {expected_reused_count}/{target_total}")
    valid_reused_set = set(reused_indices)
    corrupt_reused_indices = sorted(
        int(match.group(1)) for name in corrupt_reused if (match := expected.fullmatch(name))
    )
    missing_reused_indices = [
        index for index in range(expected_reused_count)
        if index not in valid_reused_set and index not in set(corrupt_reused_indices)
    ]
    n_new_required = target_total - expected_reused_count
    gen_pattern = re.compile(r"^gen_(\d{4,})\.png$")
    valid_gen, corrupt_gen, foreign = [], [], []
    for path in sorted(final_dir.glob("*.png")):
        if path.name.startswith(".tmp_") or path in reused:
            continue
        match = gen_pattern.fullmatch(path.name)
        if match is None:
            foreign.append(path.name)
        elif _valid_png(path):
            valid_gen.append(int(match.group(1)))
        else:
            corrupt_gen.append(path.name)

    # The final target depends on the number of images, not on whether a valid
    # historical run started gen_* numbering at 0 or after the reused eval set.
    # Generate only the numerical deficit and pick free indices deterministically.
    missing_count = max(0, n_new_required - len(valid_gen))
    occupied = set(valid_gen)
    missing_gen, candidate = [], 0
    while len(missing_gen) < missing_count:
        if candidate not in occupied:
            missing_gen.append(candidate)
        candidate += 1
    surplus_gen = [f"gen_{index:04d}.png" for index in valid_gen[n_new_required:]]
    gen_index_layout = "indexed_count"
    return {
        "valid_reused_paths": reused,
        "valid_reused_indices": reused_indices,
        "missing_reused_indices": missing_reused_indices,
        "corrupt_reused_indices": corrupt_reused_indices,
        "expected_reused_count": expected_reused_count,
        "n_valid_reused": len(reused),
        "n_new_required": n_new_required,
        "gen_index_layout": gen_index_layout,
        "valid_gen_indices": valid_gen,
        "missing_gen_indices": missing_gen,
        "corrupt_files": corrupt_reused + corrupt_gen,
        "extra_files": foreign + surplus_gen,
        "excluded_surplus_gen_files": [],
        "complete": (
            len(reused) == expected_reused_count
            and expected_reused_count + len(valid_gen) == int(target_total)
            and not missing_reused_indices
            and not corrupt_reused
            and not corrupt_gen
            and not missing_gen
            and not foreign
            and not surplus_gen
        ),
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{path.name}.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_content_signature(path: Path) -> dict:
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _bounded_content_signature(path: Path) -> dict:
    """Fingerprint a potentially large model file without hashing its full body.

    Metadata catches ordinary replacements; the first/last MiB digest additionally
    distinguishes common in-place weight updates while keeping checkpoint checks
    cheap enough to run before a resume.
    """
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    block_size = 1024 * 1024
    with path.open("rb") as handle:
        first = handle.read(block_size)
        digest.update(first)
        if stat.st_size > block_size:
            handle.seek(max(0, stat.st_size - block_size))
            digest.update(handle.read(block_size))
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "edge_sha256": digest.hexdigest(),
    }


def _directory_content_signature(root: Path, files: Iterable[Path]) -> list[dict]:
    root = Path(root)
    return [
        {"name": str(path.relative_to(root)), **_bounded_content_signature(path)}
        for path in sorted({Path(path) for path in files})
        if path.is_file()
    ]


DIFFUSERS_COMPONENT_DIRS = ("vae", "unet", "text_encoder", "text_encoder_2")
DIFFUSERS_WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.index.json")


def sd_base_model_signature(base_model_dir: Path) -> dict:
    """Fingerprint SD configs and effective VAE/base-model weights.

    In particular, a fine-tuned VAE overwritten in place must invalidate all
    generation manifests and metric caches that depend on it. Component weight
    files are discovered dynamically (full precision, FP16, or sharded, e.g.
    ``diffusion_pytorch_model.fp16-00001-of-00002.safetensors`` plus its
    ``*.index.json``) instead of a fixed filename list, so any replacement
    under ``vae/``, ``unet/``, ``text_encoder/`` or ``text_encoder_2/`` is
    caught without having to enumerate every naming convention up front.
    """
    root = Path(base_model_dir).expanduser().resolve()
    components = {}
    for relative in ("model_index.json", "scheduler/scheduler_config.json"):
        path = root / relative
        components[relative] = _bounded_content_signature(path) if path.is_file() else None
    for component_dir in DIFFUSERS_COMPONENT_DIRS:
        directory = root / component_dir
        if not directory.is_dir():
            continue
        component_files = {directory / "config.json"} if (directory / "config.json").is_file() else set()
        for pattern in DIFFUSERS_WEIGHT_GLOBS:
            component_files.update(directory.glob(pattern))
        for path in sorted(path for path in component_files if path.is_file()):
            components[f"{component_dir}/{path.name}"] = _bounded_content_signature(path)
    return {"path": str(root), "components": components}


def checkpoint_content_signature(checkpoint: Path) -> dict:
    """Content-aware identity for SD checkpoints (LoRA or full U-Net)."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if checkpoint.is_file():
        return {"path": str(checkpoint), "file": _bounded_content_signature(checkpoint)}
    unet = checkpoint / "unet"
    lora_files = [
        checkpoint / "pytorch_lora_weights.safetensors",
        checkpoint / "adapter_config.json",
    ]
    if unet.is_dir():
        relevant = [path for path in unet.rglob("*") if path.is_file()]
    elif any(path.is_file() for path in lora_files):
        relevant = [path for path in lora_files if path.is_file()]
    else:
        # Retain useful behavior for simple .keras checkpoints and unfamiliar
        # layouts while making every recorded entry content-aware.
        relevant = [path for path in checkpoint.rglob("*") if path.is_file()] if checkpoint.is_dir() else []
    return {
        "path": str(checkpoint),
        "files": _directory_content_signature(checkpoint, relevant),
    }


def png_content_signature(directory: Path, pattern: str = GENERATED_PNG_PATTERN) -> list[dict]:
    return [
        {"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)}
        for path in readable_png_paths(Path(directory), pattern)
    ]


def sd_metrics_cache_config(**values) -> dict:
    """Normalize the scientific inputs that make an SD metrics cache reusable."""
    required = (
        "eval_seed", "inference_steps", "guidance_scale", "resolution",
        "n_gen_per_class", "checkpoint_type", "base_model_dir",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"Missing SD metrics cache fields: {missing}")
    optional_defaults = {
        "n_validation_images_per_class": None,
        "prdc_nearest_k": None,
        "evaluator_batch_size": None,
        "evaluator_num_workers": None,
        "metric_backend": None,
        "metric_backend_version": None,
    }
    return {
        "seed_strategy": SD_SEED_STRATEGY,
        "eval_seed": int(values["eval_seed"]),
        "inference_steps": int(values["inference_steps"]),
        "guidance_scale": float(values["guidance_scale"]),
        "resolution": int(values["resolution"]),
        "n_gen_per_class": int(values["n_gen_per_class"]),
        "checkpoint_type": str(values["checkpoint_type"]),
        "base_model": sd_base_model_signature(Path(values["base_model_dir"])),
        **{key: values.get(key, default) for key, default in optional_defaults.items()},
    }


def sd_metrics_cache_compatible(payload: object, config: dict, validation_csv: Path) -> bool:
    return bool(
        isinstance(payload, dict)
        and payload.get("schema_version") == 2
        and payload.get("config") == config
        and payload.get("validation_csv_signature") == file_content_signature(validation_csv)
        and isinstance(payload.get("checkpoints"), dict)
    )


def sd_metrics_cache_entry_matches(entry: object, checkpoint: Path, negative_dir: Path, positive_dir: Path) -> bool:
    if not isinstance(entry, dict) or "metrics" not in entry:
        return False
    expected = {
        "checkpoint_signature": checkpoint_content_signature(checkpoint),
        "negative_image_signature": png_content_signature(negative_dir),
        "positive_image_signature": png_content_signature(positive_dir),
    }
    return all(entry.get(key) == value for key, value in expected.items())


def create_parallel_run_dir(logs_dir: Path) -> Path:
    """Create a unique directory for one orchestration, including repeated runs."""
    parent = Path(logs_dir) / "parallel_generation"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = time.time_ns()
    while True:
        run_dir = parent / f"run_{stamp}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            stamp += 1


def validate_sd_evaluation_source(
    source_dir: Path,
    checkpoint_path: str,
    class_name: str,
    prompt: str,
    allow_legacy_seed_mix: bool = True,
    *,
    base_seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    resolution: int,
    checkpoint_type: str,
    base_model_dir: Path,
) -> dict:
    """Validate the sampling parameters of an evaluation directory before reuse."""
    source_dir = Path(source_dir)
    manifest_path = source_dir / ".generation_manifest.json"
    if not manifest_path.is_file():
        if not allow_legacy_seed_mix:
            raise RuntimeError(
                f"Evaluation source {source_dir} has no generation manifest; refusing legacy seed reuse."
            )
        return {
            "source_directory": str(source_dir.resolve()),
            "source_manifest": None,
            "seed_strategy": "legacy_unknown",
            "legacy_seed_mix": True,
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Unreadable evaluation manifest: {manifest_path}") from exc
    expected = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_signature": checkpoint_content_signature(Path(checkpoint_path)),
        "class_name": str(class_name),
        "prompt": str(prompt),
        "phase": "evaluation",
        "class_offset": SD_SEED_OFFSETS[f"evaluation:{class_name}"],
        "base_seed": int(base_seed),
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "resolution": int(resolution),
        "checkpoint_type": str(checkpoint_type),
        "base_model": sd_base_model_signature(base_model_dir),
    }
    incompatible = [key for key, value in expected.items() if manifest.get(key) != value]
    if incompatible:
        raise RuntimeError(f"Incompatible evaluation manifest {manifest_path}: {incompatible}")
    if manifest.get("seed_strategy") != SD_SEED_STRATEGY:
        if allow_legacy_seed_mix and manifest.get("legacy_seed_mix"):
            return {
                "source_directory": str(source_dir.resolve()),
                "source_manifest": str(manifest_path.resolve()),
                "seed_strategy": manifest.get("seed_strategy", "legacy_unknown"),
                "base_seed": manifest.get("base_seed"),
                "phase": manifest.get("phase"),
                "class_offset": manifest.get("class_offset"),
                "legacy_seed_mix": True,
            }
        raise RuntimeError(
            f"Evaluation manifest {manifest_path} is not pure {SD_SEED_STRATEGY}"
        )
    return {
        "source_directory": str(source_dir.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "seed_strategy": manifest["seed_strategy"],
        "base_seed": int(manifest["base_seed"]),
        "phase": manifest["phase"],
        "class_offset": int(manifest["class_offset"]),
        "legacy_seed_mix": False,
    }


def copy_validated_sd_evaluation_images(
    source_dir: Path,
    final_dir: Path,
    count: int,
    reused_prefix: str,
    checkpoint_path: str,
    class_name: str,
    prompt: str,
    allow_legacy_seed_mix: bool = True,
    *,
    base_seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    resolution: int,
    checkpoint_type: str,
    base_model_dir: Path,
) -> dict:
    """Validate evaluation metadata, then copy an exact readable indexed subset."""
    source_metadata = validate_sd_evaluation_source(
        source_dir, checkpoint_path, class_name, prompt, allow_legacy_seed_mix,
        base_seed=base_seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        resolution=resolution,
        checkpoint_type=checkpoint_type,
        base_model_dir=base_model_dir,
    )
    scan = scan_named_png_set(Path(source_dir), count, "gen_")
    if scan["missing_indices"] or scan["corrupt_indices"] or scan["extra_files"]:
        raise RuntimeError(
            f"Evaluation source {source_dir} is incomplete: "
            f"missing={scan['missing_indices']}, corrupt={scan['corrupt_indices']}, "
            f"extra={scan['extra_files']}"
        )
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for index in range(int(count)):
        source = Path(source_dir) / f"gen_{index:04d}.png"
        destination = final_dir / f"{reused_prefix}_{index:04d}.png"
        if not destination.is_file() or not _valid_png(destination) or _sha256(source) != _sha256(destination):
            import shutil
            shutil.copy2(source, destination)
        copied.append({"source": source.name, "destination": destination.name})
    payload = {
        **source_metadata,
        "class_name": class_name,
        "prompt": prompt,
        "checkpoint": str(checkpoint_path),
        "checkpoint_signature": checkpoint_content_signature(Path(checkpoint_path)),
        "base_seed": int(base_seed),
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "resolution": int(resolution),
        "checkpoint_type": checkpoint_type,
        "base_model": sd_base_model_signature(base_model_dir),
        "reused_prefix": reused_prefix,
        "files": copied,
    }
    _atomic_json(final_dir / ".evaluation_reuse.json", payload)
    return payload


def prepare_sd_manifest(
    request: dict,
    checkpoint_path: str,
    allow_legacy_seed_mix: bool = True,
    dry_run: bool = False,
) -> None:
    """Validate/write a per-directory seed manifest before scheduling workers."""
    out_dir = Path(request["out_dir"])
    manifest_path = out_dir / ".generation_manifest.json"
    # Requests are persisted for GPU workers as JSON, so carry the same explicit
    # checkpoint identity there as in the directory manifest.
    request["checkpoint_signature"] = checkpoint_content_signature(Path(checkpoint_path))
    expected = {
        "seed_strategy": SD_SEED_STRATEGY,
        "base_seed": int(request["seed"]),
        "phase": request["phase"],
        "class_name": request["class_name"],
        "class_offset": int(request["class_offset"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_signature": request["checkpoint_signature"],
        "prompt": request["prompt"],
        "num_inference_steps": int(request["inference_steps"]),
        "guidance_scale": float(request["guidance_scale"]),
        "resolution": int(request["resolution"]),
    }
    if "checkpoint_type" in request:
        expected["checkpoint_type"] = str(request["checkpoint_type"])
    if "base_model_dir" in request:
        expected["base_model"] = sd_base_model_signature(Path(request["base_model_dir"]))
    current = None
    if manifest_path.exists():
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Unreadable generation manifest: {manifest_path}") from exc
    png_names = [path.name for path in out_dir.glob("*.png") if not path.name.startswith(".tmp_")]
    has_png = bool(png_names)
    if request["phase"] == "final_new":
        reuse_record_path = out_dir / ".evaluation_reuse.json"
        reuse_record = None
        if reuse_record_path.is_file():
            try:
                reuse_record = json.loads(reuse_record_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"Unreadable evaluation reuse record: {reuse_record_path}") from exc
        reused_files = [item["destination"] for item in (reuse_record or {}).get("files", [])]
        for name in reused_files:
            if name not in png_names or not _valid_png(out_dir / name):
                raise RuntimeError(f"Declared reused evaluation image is missing or corrupt: {out_dir / name}")
        legacy_files = sorted(name for name in png_names if name not in reused_files and not name.startswith("gen_"))
        if current and current.get("seed_strategy") == f"mixed_legacy_and_{SD_SEED_STRATEGY}":
            # A mixed manifest is a deliberate, audited exception.  On resume its
            # original legacy set is immutable; existing v2 files are not legacy.
            legacy_files = sorted(current.get("legacy_files", []))
        elif not manifest_path.exists():
            legacy_files.extend(name for name in png_names if name.startswith("gen_"))
        if reuse_record is None and any(name.startswith("eval_") for name in png_names):
            legacy_files.extend(sorted(name for name in png_names if name.startswith("eval_")))
        if reuse_record and reuse_record.get("legacy_seed_mix"):
            legacy_files.extend(reused_files)
        legacy_files = sorted(set(legacy_files))
        if legacy_files and not allow_legacy_seed_mix:
            raise RuntimeError(
                f"Existing PNGs in {out_dir} have unknown legacy sampling parameters: {legacy_files}"
            )
        requested_count = request.get("count")
        if requested_count is None:
            requested_count = max(
                [int(match.group(1)) + 1 for name in png_names if (match := re.fullmatch(r"gen_(\d+)\.png", name))]
                or [0]
            )
        expected_gen = [f"gen_{index:04d}.png" for index in range(int(requested_count))]
        mixed = bool(legacy_files)
        v2_final_files = [name for name in expected_gen if name not in legacy_files]
        expected.update({
            "seed_strategy": (
                f"mixed_legacy_and_{SD_SEED_STRATEGY}" if mixed else SD_SEED_STRATEGY
            ),
            "legacy_seed_mix": mixed,
            "legacy_files": legacy_files,
            "v2_files": sorted(set(v2_final_files + ([] if mixed else reused_files))),
            "groups": {
                "evaluation_reused": reuse_record,
                "final_new": {
                    "files": v2_final_files,
                    "seed_strategy": SD_SEED_STRATEGY,
                    "base_seed": int(request["seed"]),
                    "phase": "final_new",
                    "class_offset": int(request["class_offset"]),
                },
            },
        })
    if request["phase"] != "final_new" and has_png and allow_legacy_seed_mix:
        legacy_files = (
            sorted(current.get("legacy_files", []))
            if current and current.get("seed_strategy") == f"mixed_legacy_and_{SD_SEED_STRATEGY}"
            else sorted(png_names)
        )
        expected.update({
            "seed_strategy": f"mixed_legacy_and_{SD_SEED_STRATEGY}",
            "legacy_seed_mix": True,
            "legacy_files": legacy_files,
            "v2_files": [
                f"gen_{index:04d}.png" for index in range(int(request.get("count", 0)))
                if f"gen_{index:04d}.png" not in legacy_files
            ],
        })
    if current is not None:
        if current.get("seed_strategy") == f"mixed_legacy_and_{SD_SEED_STRATEGY}" and not allow_legacy_seed_mix:
            raise RuntimeError(
                f"Generation manifest {manifest_path} contains legacy seed mixing; "
                "set ALLOW_LEGACY_SEED_MIX=True only after explicit scientific review."
            )
        incompatible = [key for key, value in expected.items() if current.get(key) != value]
        if incompatible:
            raise RuntimeError(f"Incompatible generation manifest {manifest_path}: {incompatible}")
    elif has_png and request["phase"] != "final_new" and not allow_legacy_seed_mix:
        raise RuntimeError(
            f"Existing PNGs in {out_dir} have no seed manifest; refusing to mix legacy stateful output. "
            "Set ALLOW_LEGACY_SEED_MIX=True only after an explicit scientific review."
        )
    elif not dry_run:
        _atomic_json(manifest_path, expected)


def worker_environment(device: str) -> dict[str, str]:
    """Environment for a child that sees one physical GPU as logical cuda:0."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    return env


def _tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "<log unavailable>"


def worker_log_path(logs_dir: Path, label: str, device: str) -> Path:
    """Deterministic per-job log name; callers provide stage/class in ``label``."""
    safe_label = str(label).replace("/", "_").replace(" ", "_")
    return logs_dir / f"{safe_label}_gpu_{device}.log"


def run_dynamic_gpu_jobs(
    jobs: Iterable[dict],
    devices: list[str],
    command_for_job: Callable[[dict, str], list[str]],
    logs_dir: Path,
    dry_run: bool = False,
    cwd: Path | None = None,
    poll_seconds: float = 0.1,
) -> None:
    """Run a dynamic job queue with at most one process per CUDA device.

    Each child has its own log. On failure or Ctrl-C, remaining children are
    terminated while already atomically-written PNGs are left untouched.
    """
    queue = deque(jobs)
    if not dry_run:
        logs_dir.mkdir(parents=True, exist_ok=True)
    active: dict[str, tuple[dict, subprocess.Popen, Path, object]] = {}

    def start_next(device: str) -> bool:
        if not queue:
            return False
        job = queue.popleft()
        command = command_for_job(job, device)
        label = str(job.get("label") or job.get("checkpoint_id") or job.get("worker_id") or "job")
        log_path = worker_log_path(logs_dir, label, device)
        print(f"[GPU {device}] started {label}")
        print("  command:", " ".join(command))
        if dry_run:
            print(f"  log: {log_path}")
            return True
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd is not None else None,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=worker_environment(device),
            start_new_session=True,
        )
        active[device] = (job, process, log_path, handle)
        return True

    if dry_run:
        position = 0
        while queue:
            start_next(devices[position % len(devices)])
            position += 1
        return

    try:
        for device in devices:
            start_next(device)
        while active:
            progressed = False
            for device, (job, process, log_path, handle) in list(active.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                progressed = True
                handle.close()
                del active[device]
                label = str(job.get("label") or job.get("checkpoint_id") or job.get("worker_id") or "job")
                if return_code:
                    print(f"[GPU {device}] failed {label}: return code {return_code}; log: {log_path}")
                    print(_tail(log_path))
                    raise RuntimeError(f"Generation worker failed on GPU {device}: {label}")
                print(f"[GPU {device}] completed {label}")
                start_next(device)
            if active and not progressed:
                time.sleep(poll_seconds)
    except BaseException:
        for _, process, _, handle in active.values():
            if process.poll() is None:
                process.terminate()
            handle.close()
        for _, process, _, _ in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise


def missing_named_png_indices(directory: Path, count: int, prefix: str = "gen_") -> list[int]:
    """Indices whose expected PNG is absent or corrupt; unrelated files do not count."""
    valid = set(valid_named_png_indices(directory, count, prefix))
    return [index for index in range(int(count)) if index not in valid]


def remove_stale_generation_temps(directory: Path) -> None:
    """Remove stale SD atomic-write files before any concurrent worker starts."""
    for path in Path(directory).glob(".tmp_gen_*.png"):
        try:
            path.unlink()
        except OSError:
            pass


ORCHESTRATION_LOCK_NAME = ".parallel_generation.lock"


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_lock_pid(lock_path: Path) -> int | None:
    try:
        return int(json.loads(lock_path.read_text(encoding="utf-8"))["pid"])
    except Exception:
        return None


def acquire_parallel_generation_lock(out_dir: Path) -> Path:
    """Atomically claim ``.parallel_generation.lock`` in ``out_dir``.

    Protects against two independent orchestrations (e.g. two notebook kernels)
    scheduling parallel generation into the same output directory at once.
    Workers within a single orchestration are unaffected and untouched.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ORCHESTRATION_LOCK_NAME
    payload = json.dumps({"pid": os.getpid(), "timestamp": time.time()}).encode("utf-8")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder_pid = _read_lock_pid(lock_path)
        if holder_pid is not None and _pid_is_running(holder_pid):
            raise RuntimeError(
                f"Un'altra orchestrazione (PID {holder_pid}) detiene gia' {lock_path}; "
                f"rifiuto una seconda orchestrazione parallela concorrente su {out_dir}."
            )
        print(f"WARNING: rimuovo lock stale ({lock_path}) del PID {holder_pid}, non piu' attivo.")
        try:
            lock_path.unlink()
        except OSError:
            pass
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
    return lock_path


def release_parallel_generation_lock(lock_path: Path) -> None:
    try:
        Path(lock_path).unlink()
    except OSError:
        pass


def run_sd_generation_jobs(
    jobs: list[dict],
    requested_devices: str | None,
    max_workers: int | None,
    logs_dir: Path,
    project_root: Path,
    dry_run: bool = False,
    generation_scheduler: str = DEFAULT_GENERATION_SCHEDULER,
    reservation_size: int = DEFAULT_GENERATION_RESERVATION_SIZE,
) -> list[str]:
    """Schedule JSON-described Stable Diffusion jobs through the shared worker."""
    devices = resolve_generation_gpu_devices(requested_devices, max_workers)
    print_generation_diagnostics(requested_devices, devices)
    if generation_scheduler not in GENERATION_SCHEDULERS:
        raise ValueError(f"Unsupported generation scheduler: {generation_scheduler}")
    requested_scheduler = generation_scheduler
    if generation_scheduler in {"auto", "checkpoint_queue"}:
        generation_scheduler = "round_robin" if generation_scheduler == "checkpoint_queue" or len(jobs) >= len(devices) else "dynamic_reservations"
    if requested_scheduler == "auto" and generation_scheduler == "dynamic_reservations" and jobs:
        originals = list(jobs)
        jobs = []
        for worker_id in range(len(devices)):
            duplicate = json.loads(json.dumps(originals[worker_id % len(originals)]))
            duplicate["label"] = f"{duplicate.get('label', 'checkpoint')}_image_worker_{worker_id}"
            duplicate["worker_id"] = worker_id
            jobs.append(duplicate)
    print(f"Evaluation/final scheduler: requested={requested_scheduler}, resolved={generation_scheduler}")
    run_dir = logs_dir / ("sd_run_dry_run" if dry_run else f"run_{time.time_ns()}")
    jobs_dir = run_dir / "jobs"
    prepared = []
    out_dirs = {Path(request["out_dir"]).resolve() for job in jobs for request in job["requests"]}
    locks: list[Path] = []
    try:
        if not dry_run:
            # Deliberately parent-only, one lock per out_dir: workers within this
            # orchestration share an output directory freely, but a second
            # concurrent orchestration on the same out_dir is refused.
            for out_dir in sorted(out_dirs):
                locks.append(acquire_parallel_generation_lock(out_dir))
            for out_dir in sorted(out_dirs):
                remove_stale_generation_temps(out_dir)
        if generation_scheduler == "dynamic_reservations":
            queue_by_output: dict[Path, Path] = {}
            for job in jobs:
                for request in job["requests"]:
                    out_dir = Path(request["out_dir"]).resolve()
                    if out_dir not in queue_by_output:
                        count = int(request["count"])
                        state = scan_named_png_set(out_dir, count)
                        indices = sorted(set(request.get("indices", state["missing_indices"])) | set(state["corrupt_indices"]))
                        queue_dir = run_dir / "work_queue" / hashlib.sha256(str(out_dir).encode()).hexdigest()[:12]
                        prepare_dynamic_queue(
                            queue_dir, indices, target_count=count,
                            valid_indices=state["valid_indices"], corrupt_indices=state["corrupt_indices"],
                            reservation_size=reservation_size, output_dir=out_dir,
                            metadata={"seed_strategy": request.get("seed_strategy", SD_SEED_STRATEGY), "parameters": {k: request.get(k) for k in ("phase", "class_name", "seed", "inference_steps", "guidance_scale", "resolution")}},
                            dry_run=dry_run,
                        )
                        queue_by_output[out_dir] = queue_dir
                    request["dynamic_queue_dir"] = str(queue_by_output[out_dir])
                    request["generation_reservation_size"] = reservation_size
        for number, job in enumerate(jobs):
            for request in job["requests"]:
                prepare_sd_manifest(
                    request,
                    checkpoint_path=job["checkpoint_path"],
                    # I PNG gia presenti sono output di una precedente esecuzione
                    # del notebook. Il controllo stretto resta disponibile ai
                    # chiamanti espliciti, ma il workflow notebook li riusa.
                    allow_legacy_seed_mix=bool(request.get("allow_legacy_seed_mix", True)),
                    dry_run=dry_run,
                )
            job_path = jobs_dir / f"job_{number:04d}.json"
            if not dry_run:
                job_path.parent.mkdir(parents=True, exist_ok=True)
                job_path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
            prepared.append({"label": job.get("label", f"sd job {number}"), "job_path": job_path})
        worker = Path(__file__).resolve().with_name("sd_generation_worker.py")
        run_dynamic_gpu_jobs(
            jobs=prepared,
            devices=devices,
            command_for_job=lambda job, gpu: [sys.executable, str(worker), "--job-file", str(job["job_path"])],
            logs_dir=run_dir,
            dry_run=dry_run,
            cwd=project_root,
        )
        return devices
    finally:
        for lock_path in locks:
            release_parallel_generation_lock(lock_path)


def run_sd_final_generation(
    checkpoint_path: Path,
    checkpoint_type: str,
    base_model_dir: Path,
    requests: list[dict],
    requested_devices: str | None,
    max_workers: int | None,
    logs_dir: Path,
    project_root: Path,
    dry_run: bool = False,
    generation_scheduler: str = DEFAULT_GENERATION_SCHEDULER,
    reservation_size: int = DEFAULT_GENERATION_RESERVATION_SIZE,
) -> list[str]:
    """Run persistent workers; dynamic reservations are the default."""
    devices = resolve_generation_gpu_devices(requested_devices, max_workers)
    print_generation_diagnostics(requested_devices, devices)
    if generation_scheduler == "dynamic_reservations":
        jobs = [{
            "label": f"final_dynamic_worker_{worker_id}", "worker_id": worker_id,
            "checkpoint_path": str(checkpoint_path), "checkpoint_type": checkpoint_type,
            "base_model_dir": str(base_model_dir), "requests": [dict(request) for request in requests],
        } for worker_id, _device in enumerate(devices)]
        return run_sd_generation_jobs(
            jobs, ",".join(devices), len(devices), logs_dir, project_root, dry_run=dry_run,
            generation_scheduler=generation_scheduler, reservation_size=reservation_size,
        )
    assignments = [[] for _ in devices]
    position = 0
    for request in requests:
        for index in request["indices"]:
            assignments[position % len(assignments)].append((request, index))
            position += 1
    jobs = []
    for worker_id, items in enumerate(assignments):
        if not items:
            continue
        by_request: dict[int, dict] = {}
        for request, index in items:
            key = id(request)
            if key not in by_request:
                by_request[key] = {key_: value for key_, value in request.items() if key_ != "indices"}
                by_request[key]["indices"] = []
            by_request[key]["indices"].append(index)
        class_label = "_".join(sorted({str(request.get("class_name", request.get("name", "class"))) for request, _ in items}))
        jobs.append({
            "label": f"final_{class_label}_worker_{worker_id}",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_type": checkpoint_type,
            "base_model_dir": str(base_model_dir),
            "requests": list(by_request.values()),
        })
    # ``run_sd_generation_jobs`` resolves the same devices again; it is safe and
    # keeps its public API useful for the checkpoint-based evaluation queue.
    return run_sd_generation_jobs(
        jobs, ",".join(devices), len(devices), logs_dir, project_root, dry_run=dry_run,
        generation_scheduler="round_robin", reservation_size=reservation_size,
    )
