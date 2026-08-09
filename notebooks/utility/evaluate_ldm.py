#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from ldm_project_paths import (
    RESULTS_STAGE_NAME,
    ExperimentPaths,
    get_experiment_paths,
    get_results_paths,
    normalize_processed_path,
)
from parallel_generation_utils import (
    checkpoint_content_signature,
    create_parallel_run_dir,
    png_content_signature,
    print_generation_diagnostics,
    resolve_generation_gpu_devices,
    run_dynamic_gpu_jobs,
    sd_base_model_signature,
)

POSITIVE_CLASS = 1
BEST_SELECTION_METRIC = f"fid_{POSITIVE_CLASS}"
BEST_SELECTION_TIE_BREAKER = f"is_mean_{POSITIVE_CLASS}"
SELECTION_DIRECTION = "minimize"
TIE_BREAKER_DIRECTION = "maximize"
SELECTION_REASON = "Best generation quality for the positive mammography class"
BEST_ROW_METRIC_COLUMNS = [
    "fid_0",
    "fid_1",
    "fid_mean",
    "is_mean_0",
    "is_mean_1",
    "is_mean_avg",
    "precision_0",
    "precision_1",
    "recall_0",
    "recall_1",
    "density_0",
    "density_1",
    "coverage_0",
    "coverage_1",
]


def parse_args() -> argparse.Namespace:
    """Parse checkpoint, sampling, metric, and sustainability sweep arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and evaluate MammoDiffusion LDM checkpoint sweeps."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help="Shared checkpoint directory; evaluation outputs and images stay in the experiment.",
    )
    parser.add_argument("--gpu-visible-devices", default=None)
    parser.add_argument(
        "--generation-gpus",
        default="auto",
        help="GPUs for generation orchestration: auto, off, or a comma-separated list.",
    )
    parser.add_argument("--max-generation-workers", type=int, default=None)
    parser.add_argument("--generation-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Show workers and commands without loading models.")
    parser.add_argument(
        "--cuda-root",
        type=Path,
        default=Path(os.environ.get("MAMMODIFFUSION_CUDA_ROOT", sys.prefix)),
    )
    parser.add_argument("--min-step", type=int, default=14_000)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--checkpoint-id", default=None)
    parser.add_argument(
        "--mode",
        choices=["both", "generate", "metrics", "artifacts"],
        default="both",
        help=(
            "both orchestrates generate+metrics in separate processes; generate creates "
            "images only; metrics computes metrics only; artifacts rebuilds only "
            "JSON/CSV/plot/manifest da checkpoint_metrics.json."
        ),
    )
    parser.add_argument("--n-gen-per-class", type=int, default=50)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument(
        "--mini-batch",
        type=int,
        default=1,
        help="Retained for compatibility; sweep sampling forces batch size 1.",
    )
    parser.add_argument("--classes", default="0,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-on-cpu", action="store_true")
    parser.add_argument(
        "--vae-backend",
        choices=["keras", "sd"],
        default="keras",
        help="Decoder used to transform latents into images.",
    )
    parser.add_argument(
        "--sd-vae-model",
        default=None,
        help="Local Stable Diffusion model path from which to load the VAE.",
    )
    parser.add_argument("--sd-vae-batch-size", type=int, default=1)
    parser.add_argument(
        "--parameterization",
        choices=["eps", "v"],
        default="eps",
        help=(
            "Parameterization of evaluated checkpoints: eps (default, compatible with "
            "04b/04b1/04b2) or v (Salimans & Ho, 2022, used by 04b3). It must "
            "match how the checkpoints were trained."
        ),
    )
    parser.add_argument(
        "--unet-version",
        choices=["v2", "v3"],
        default="v2",
        help="Informational evaluated-checkpoint architecture stored in output JSON.",
    )
    parser.add_argument(
        "--vae-source",
        default="sd_vae_original",
        help="Informational source of the latent VAE stored in output JSON.",
    )
    parser.add_argument(
        "--uses-vae-ft-from-03",
        action="store_true",
        help="Record in output JSON that the VAE is the fine-tuned VAE from notebook 03.",
    )
    parser.add_argument(
        "--notebook-name",
        default=None,
        help="Calling notebook name stored in output JSON.",
    )
    parser.add_argument("--vram-log-every", type=int, default=10)
    parser.add_argument("--inception-batch", type=int, default=8)
    parser.add_argument("--is-splits", type=int, default=10)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument(
        "--results-stage-name",
        default=RESULTS_STAGE_NAME,
        help="Results subdirectory for metrics, plots, and EcoTracker logs.",
    )
    parser.add_argument(
        "--inception-weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Use 'none' only for limited trial runs; metrics are not meaningful.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore checkpoint_metrics.json and recompute the sweep.",
    )
    parser.add_argument(
        "--eco-track",
        action="store_true",
        help=(
            "Measure major stages with EcoTracker and save JSONL under "
            "results/<results-stage-name>/ecotracker."
        ),
    )
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    """Set visible GPUs and XLA's CUDA path before importing TensorFlow."""
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mammodiffusion-matplotlib")
    )
    if args.gpu_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible_devices

    libdevice = args.cuda_root / "nvvm" / "libdevice" / "libdevice.10.bc"
    if libdevice.exists() and "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={args.cuda_root}"


def parse_classes(classes_value: str) -> list[int]:
    """Parse --classes (for example 0,1) into unique valid integer labels."""
    classes = []
    for part in str(classes_value).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in (0, 1):
            raise ValueError(f"Invalid class: {value}")
        if value not in classes:
            classes.append(value)
    if not classes:
        raise ValueError("--classes cannot be empty")
    return classes


def format_cfg_label(guidance_scale: float, sample_steps: int) -> str:
    """Format guidance scale and step count for generated-image directory names."""
    gs_label = f"{guidance_scale:g}".replace(".", "p")
    return f"cfg_gs{gs_label}_st{sample_steps}"


def step_from_checkpoint(path: Path) -> int:
    """Extract the training step from a checkpoint filename such as ldm_step20000.keras."""
    return int(path.stem.replace("ldm_step", ""))


def collect_checkpoint_candidates(paths: ExperimentPaths, args: argparse.Namespace) -> list[dict]:
    """List sweep checkpoints from best training loss and eligible periodic steps."""
    step_ckpt_paths = sorted(
        paths.checkpoints_dir.glob("ldm_step*.keras"),
        key=step_from_checkpoint,
    )
    step_ckpt_paths = [
        path for path in step_ckpt_paths
        if step_from_checkpoint(path) >= args.min_step
    ]
    if args.max_checkpoints is not None:
        step_ckpt_paths = step_ckpt_paths[: args.max_checkpoints]

    checkpoint_candidates = []
    best_train_path = paths.checkpoints_dir / "ldm_unet_best.keras"
    if best_train_path.exists():
        checkpoint_candidates.append(
            {
                "checkpoint_id": "best_train",
                "kind": "best_train_loss",
                "step": None,
                "path": best_train_path,
            }
        )
    for path in step_ckpt_paths:
        step = step_from_checkpoint(path)
        checkpoint_candidates.append(
            {
                "checkpoint_id": f"step_{step}",
                "kind": "step",
                "step": step,
                "path": path,
            }
        )
    for order, candidate in enumerate(checkpoint_candidates):
        candidate["order"] = order

    if args.checkpoint_id is not None:
        checkpoint_candidates = [
            candidate
            for candidate in checkpoint_candidates
            if candidate["checkpoint_id"] == args.checkpoint_id
        ]

    if not checkpoint_candidates:
        raise FileNotFoundError(
            f"No candidate checkpoint found in {paths.checkpoints_dir} "
            f"per checkpoint_id={args.checkpoint_id!r}"
        )
    return checkpoint_candidates


def build_eval_config(args: argparse.Namespace) -> dict:
    """Build the evaluation configuration that uniquely identifies a sweep run."""
    return {
        "metric_backend": "generative_evaluator.py",
        "min_step": args.min_step,
        "max_checkpoints": args.max_checkpoints,
        "n_gen_per_class": args.n_gen_per_class,
        "sample_steps": args.sample_steps,
        "guidance_scale": args.guidance_scale,
        "mini_batch": 1,
        "inception_batch": args.inception_batch,
        "is_splits": args.is_splits,
        "knn_k": args.knn_k,
        "inception_weights": args.inception_weights,
        "seed": args.seed,
        "vae_backend": args.vae_backend,
        "sd_vae_model": args.sd_vae_model,
        "sd_vae_batch_size": args.sd_vae_batch_size,
        "best_selection_metric": BEST_SELECTION_METRIC,
        "best_selection_tie_breaker": BEST_SELECTION_TIE_BREAKER,
        "positive_class": POSITIVE_CLASS,
        "parameterization": args.parameterization,
        "unet_version": args.unet_version,
        "seed_strategy": "stateless_seed_per_image_v1",
        "parallel_generation": str(args.generation_gpus).strip().lower() not in {"off", "none", "false", ""},
        "generation_gpus_requested": args.generation_gpus,
    }


def normalize_eval_config(config: dict | None) -> dict:
    """Fill missing cached optional fields with current defaults before comparison.

    ``parameterization`` and ``unet_version`` default to ``eps`` and ``v2``.
    Caches written by 04b2 before these flags omit both fields and are therefore
    historically equivalent to eps/v2 rather than invalid.
    """
    normalized = dict(config or {})
    normalized.setdefault("seed", 42)
    normalized.setdefault("best_selection_metric", BEST_SELECTION_METRIC)
    normalized.setdefault("best_selection_tie_breaker", BEST_SELECTION_TIE_BREAKER)
    normalized.setdefault("positive_class", POSITIVE_CLASS)
    normalized.setdefault("parameterization", "eps")
    normalized.setdefault("unet_version", "v2")
    return normalized


def selection_policy() -> dict:
    """Describe the best-checkpoint metric, direction, tie-breaker, and rationale."""
    return {
        "selection_metric": BEST_SELECTION_METRIC,
        "selection_direction": SELECTION_DIRECTION,
        "tie_breaker": BEST_SELECTION_TIE_BREAKER,
        "tie_breaker_direction": TIE_BREAKER_DIRECTION,
        "selected_class": POSITIVE_CLASS,
        "selection_reason": SELECTION_REASON,
    }


def selection_policy_is_compatible(selection: dict) -> bool:
    """Require a cached selection policy to match the current policy."""
    if selection.get("selection_metric") != BEST_SELECTION_METRIC:
        return False
    explicit_tie_breaker = selection.get("tie_breaker")
    if explicit_tie_breaker is not None and explicit_tie_breaker != BEST_SELECTION_TIE_BREAKER:
        return False
    explicit_class = selection.get("selected_class")
    if explicit_class is not None and int(explicit_class) != POSITIVE_CLASS:
        return False
    explicit_direction = selection.get("selection_direction")
    if explicit_direction is not None and explicit_direction != SELECTION_DIRECTION:
        return False
    explicit_tie_direction = selection.get("tie_breaker_direction")
    if explicit_tie_direction is not None and explicit_tie_direction != TIE_BREAKER_DIRECTION:
        return False
    return True


def results_stage_dirs(paths: ExperimentPaths, stage_name: str) -> dict[str, Path]:
    """Create and return the stage's plot, metrics, and EcoTracker output directories."""
    results_paths = get_results_paths(paths.project_root, stage_name)
    dirs = {
        "stage": results_paths.stage_dir,
        "plots": results_paths.plots_dir,
        "metrics": results_paths.metrics_dir,
        "ecotracker": results_paths.ecotracker_dir,
        "evaluation_plots": paths.evaluation_dir / "plots",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def append_jsonl(path: Path, payload: dict) -> None:
    """Append an EcoTracker JSONL record and flush it to survive interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


@contextmanager
def maybe_measure(args: argparse.Namespace, paths: ExperimentPaths, label: str):
    """Measure a stage with EcoTracker when enabled; otherwise act as a transparent no-op."""
    if not getattr(args, "eco_track", False):
        yield None
        return

    jsonl_path = (
        get_results_paths(paths.project_root, args.results_stage_name).ecotracker_dir
        / "ldm_evaluation_ecotracker.jsonl"
    )
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.touch(exist_ok=True)

    try:
        from eco_tracker import measure_sustainability
    except Exception as exc:
        print(f"EcoTracker unavailable; continuing without measurement: {exc}")
        yield None
        return

    measure_context = measure_sustainability(label=label, sample_interval=0.5)
    try:
        tracker = measure_context.__enter__()
    except Exception as exc:
        print(f"EcoTracker could not start; continuing without measurement: {exc}")
        yield None
        return

    exc_info = (None, None, None)
    try:
        yield tracker
    except BaseException:
        exc_info = sys.exc_info()
        raise
    finally:
        try:
            measure_context.__exit__(*exc_info)
        except Exception as exc:
            print(f"EcoTracker did not close cleanly; continuing without measurement: {exc}")
        if tracker is not None and tracker.metrics is not None:
            payload = tracker.metrics.to_dict()
            append_jsonl(jsonl_path, payload)
            print(f"EcoTracker record saved: {jsonl_path}")
            print(tracker.metrics)


def build_candidate_signature(checkpoint_candidates: list[dict]) -> list[dict]:
    """Extract cache-relevant candidate fields independently of internal iteration order."""
    return [
        {
            "checkpoint_id": candidate["checkpoint_id"],
            "kind": candidate["kind"],
            "step": candidate["step"],
            "path": str(candidate["path"]),
            "checkpoint_signature": checkpoint_content_signature(Path(candidate["path"])),
        }
        for candidate in checkpoint_candidates
    ]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{path.name}.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sweep_vae_signature(args: argparse.Namespace, paths: ExperimentPaths) -> dict:
    if args.vae_backend == "sd":
        from sd_vae_utils import resolve_sd_vae_model

        return sd_base_model_signature(Path(resolve_sd_vae_model(paths.project_root, args.sd_vae_model)))
    return checkpoint_content_signature(paths.models_dir / "vae_decoder_best.keras")


def sweep_generation_manifest(
    args: argparse.Namespace, paths: ExperimentPaths, candidate: dict, cls: int
) -> dict:
    """Operational compatibility record for one checkpoint/CFG/class sweep directory."""
    return {
        "schema_version": 2,
        "seed_strategy": "stateless_seed_per_image_v1",
        "checkpoint": str(candidate["path"]),
        "checkpoint_signature": checkpoint_content_signature(Path(candidate["path"])),
        "seed": int(args.seed),
        "target_label": int(cls),
        "n_raw": int(args.n_gen_per_class),
        "sample_steps": int(args.sample_steps),
        "guidance_scale": float(args.guidance_scale),
        "parameterization": args.parameterization,
        "unet_version": args.unet_version,
        "vae_backend": args.vae_backend,
        "vae_source": args.vae_source,
        "decode_on_cpu": bool(getattr(args, "decode_on_cpu", False)),
        "sd_vae_batch_size": int(getattr(args, "sd_vae_batch_size", 1)),
        "latent_stats_signature": checkpoint_content_signature(
            getattr(paths, "latents_dir", paths.project_root / "latents") / "latent_stats.npz"
        ),
        "vae_model_signature": sweep_vae_signature(args, paths),
    }


def prepare_sweep_generation_manifest(
    args: argparse.Namespace, paths: ExperimentPaths, candidate: dict, cls: int, dry_run: bool = False
) -> dict:
    """Validate sweep outputs before resuming holes; force recompute clears them."""
    out_dir = fake_output_dir(paths, args, candidate["checkpoint_id"], cls, create=not dry_run)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / ".generation_manifest.json"
    expected = sweep_generation_manifest(args, paths, candidate, cls)
    png_paths = [path for path in out_dir.glob("*.png") if not path.name.startswith(".tmp_")]
    current = None
    if manifest_path.is_file():
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Unreadable sweep generation manifest: {manifest_path}") from exc
    force_generation = bool(args.generation_worker and args.force_recompute)
    if dry_run:
        if current != expected:
            changed = sorted(key for key in set(current or {}) | set(expected) if (current or {}).get(key) != expected.get(key))
            print(f"DRY-RUN: incompatible sweep manifest under {out_dir}: {changed}")
        elif force_generation:
            print(f"DRY-RUN: verrebbero rigenerati {len(png_paths)} PNG in {out_dir}")
        return expected
    if current == expected and force_generation:
        for path in png_paths:
            path.unlink()
    elif current != expected:
        if current is None and png_paths and not force_generation:
            raise RuntimeError(
                f"Sweep directory {out_dir} contains PNGs without a generation manifest; "
                "use a new directory or --force-recompute."
            )
        if current is not None and not force_generation:
            changed = sorted(key for key in set(current) | set(expected) if current.get(key) != expected.get(key))
            raise RuntimeError(
                f"Incompatible sweep generation manifest in {out_dir}: {changed}. "
                "Use a new directory or --force-recompute."
            )
        if force_generation:
            for path in png_paths:
                path.unlink()
        _atomic_json(manifest_path, expected)
    return expected


def sweep_generated_image_signature(
    args: argparse.Namespace, paths: ExperimentPaths, checkpoint_candidates: list[dict]
) -> dict:
    return {
        candidate["checkpoint_id"]: {
            str(cls): png_content_signature(
                fake_output_dir(paths, args, candidate["checkpoint_id"], cls), r"\d+\.png"
            )
            for cls in (0, 1)
        }
        for candidate in checkpoint_candidates
    }


def sweep_metrics_input_signature(args: argparse.Namespace, paths: ExperimentPaths) -> dict:
    """Content-aware identities for every non-generated metrics dependency."""
    val_csv = paths.metadata_dir / "val.csv"
    reference_images = []
    if val_csv.is_file():
        import pandas as pd

        for _, row in pd.read_csv(val_csv).iterrows():
            image_path = normalize_processed_path(row, paths.data_processed_dir)
            reference_images.append({
                "path": str(image_path),
                "signature": checkpoint_content_signature(image_path),
            })
    return {
        "vae_model_signature": sweep_vae_signature(args, paths),
        "validation_csv_signature": checkpoint_content_signature(val_csv),
        "validation_reference_image_signature": reference_images,
        "latent_stats_signature": checkpoint_content_signature(paths.latents_dir / "latent_stats.npz"),
        "decode_on_cpu": bool(args.decode_on_cpu),
        "sd_vae_batch_size": int(args.sd_vae_batch_size),
    }


def ldm_metrics_cache_compatible(
    payload: object,
    args: argparse.Namespace,
    paths: ExperimentPaths,
    checkpoint_candidates: list[dict],
) -> bool:
    """Whether a sweep metrics payload still matches weights and generated PNGs."""
    base_compatible = bool(
        isinstance(payload, dict)
        and payload.get("schema_version") == 2
        and payload.get("candidate_signature") == build_candidate_signature(checkpoint_candidates)
        and payload.get("generation_image_signature")
        == sweep_generated_image_signature(args, paths, checkpoint_candidates)
    )
    if not base_compatible:
        return False
    expected_inputs = sweep_metrics_input_signature(args, paths)
    stored_inputs = payload.get("metrics_input_signature")
    if stored_inputs is None:
        stored_inputs = {key: payload.get(key) for key in expected_inputs}
    return stored_inputs == expected_inputs


def use_cached_metrics_if_valid(
    args: argparse.Namespace,
    paths: ExperimentPaths,
    checkpoint_candidates: list[dict],
) -> bool:
    """Reuse checkpoint_metrics.json when configuration, candidates, and selection policy match."""
    metrics_json_path = paths.evaluation_dir / "checkpoint_metrics.json"
    best_eval_model_path = paths.checkpoints_dir / "ldm_unet_best_eval.keras"
    eval_config = build_eval_config(args)
    candidate_signature = build_candidate_signature(checkpoint_candidates)

    if args.force_recompute or not metrics_json_path.exists():
        return False
    try:
        with open(metrics_json_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception as exc:
        print(f"Unreadable metric cache; recomputing: {exc}")
        return False

    if payload.get("schema_version") != 2:
        print("Metric cache has an invalid schema; recomputing.")
        return False
    if normalize_eval_config(payload.get("config")) != eval_config:
        print("Metric cache has a different configuration; recomputing.")
        return False
    if payload.get("candidate_signature") != candidate_signature:
        print("Metric cache has different checkpoints; recomputing.")
        return False
    if not ldm_metrics_cache_compatible(payload, args, paths, checkpoint_candidates):
        print("Metric cache has different generated images; recomputing.")
        return False

    selection = payload.get("selection", {})
    if not selection_policy_is_compatible(selection):
        print(
            "Metric cache has a different selection policy "
            f"({selection.get('selection_metric')} / {selection.get('tie_breaker')}), "
            "recomputing."
        )
        return False
    best_checkpoint_value = selection.get("best_checkpoint")
    best_model_value = selection.get("best_model") or selection.get("best_eval_model")
    if not best_checkpoint_value or not best_model_value:
        print("Metric cache lacks best_checkpoint/best_model; recomputing.")
        return False
    best_checkpoint = Path(best_checkpoint_value)
    if not best_checkpoint.exists():
        print("Metric cache is valid but best_checkpoint is missing; recomputing.")
        return False
    best_eval_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, best_eval_model_path)
    selection["best_model"] = str(best_eval_model_path)
    selection["best_eval_model"] = str(best_eval_model_path)
    print(f"Best model synchronized from cache: {best_eval_model_path}")

    if not refresh_artifacts_from_cache(args, paths, checkpoint_candidates, payload):
        return False

    print(f"Checkpoint metrics already present: {metrics_json_path}")
    print(
        "Best model da cache: "
        f"{selection.get('best_checkpoint_id')} -> {best_eval_model_path}"
    )
    return True


def evaluation_generated_dir(paths: ExperimentPaths, create: bool = True) -> Path:
    """Return the sweep-generated image root, creating it when requested."""
    out_dir = paths.evaluation_dir / "sweep_generated"
    if create:
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def fake_output_dir(
    paths: ExperimentPaths,
    args: argparse.Namespace,
    checkpoint_id: str,
    cls: int,
    create: bool = True,
) -> Path:
    """Return the synthetic-image directory for one checkpoint, CFG setting, and class."""
    cfg_label = format_cfg_label(args.guidance_scale, args.sample_steps)
    return evaluation_generated_dir(paths, create=create) / checkpoint_id / cfg_label / f"class_{cls}"


def missing_fake_image_indices(
    paths: ExperimentPaths, args: argparse.Namespace, checkpoint_id: str, cls: int
) -> list[int]:
    """Return holes/corrupt output indices; unrelated PNG names are ignored."""
    valid = set()
    for path in fake_output_dir(paths, args, checkpoint_id, cls).glob("*.png"):
        if path.name.startswith(".tmp_"):
            continue
        try:
            index = int(path.stem)
        except ValueError:
            continue
        if not 0 <= index < args.n_gen_per_class:
            continue
        try:
            from PIL import Image
            with Image.open(path) as image:
                image.verify()
            valid.add(index)
        except Exception:
            pass
    return [index for index in range(args.n_gen_per_class) if index not in valid]


def child_generate_command(
    args: argparse.Namespace,
    paths: ExperimentPaths,
    checkpoint_id: str,
    gpu_device: str | None = None,
) -> list[str]:
    """Build a generate-mode child command for one checkpoint."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode", "generate",
        "--checkpoint-id", checkpoint_id,
        "--project-root", str(paths.project_root),
        "--experiment-dir", str(paths.experiment_dir),
        "--cuda-root", str(args.cuda_root),
        "--min-step", str(args.min_step),
        "--n-gen-per-class", str(args.n_gen_per_class),
        "--sample-steps", str(args.sample_steps),
        "--guidance-scale", str(args.guidance_scale),
        "--mini-batch", "1",
        "--classes", args.classes,
        "--seed", str(args.seed),
        "--vram-log-every", str(args.vram_log_every),
        "--results-stage-name", args.results_stage_name,
        "--generation-gpus", "off",
        "--generation-worker",
    ]
    if gpu_device is not None:
        command.extend(["--gpu-visible-devices", str(gpu_device)])
    if args.decode_on_cpu:
        command.append("--decode-on-cpu")
    if args.vae_backend != "keras":
        command.extend(["--vae-backend", args.vae_backend])
    if args.sd_vae_model is not None:
        command.extend(["--sd-vae-model", str(args.sd_vae_model)])
    if args.sd_vae_batch_size != 1:
        command.extend(["--sd-vae-batch-size", str(args.sd_vae_batch_size)])
    if args.force_recompute:
        command.append("--force-recompute")
    command.extend(["--parameterization", args.parameterization])
    command.extend(["--unet-version", args.unet_version])
    command.extend(["--vae-source", args.vae_source])
    if args.uses_vae_ft_from_03:
        command.append("--uses-vae-ft-from-03")
    if args.notebook_name is not None:
        command.extend(["--notebook-name", str(args.notebook_name)])
    return command


def orchestrate_generation(
    args: argparse.Namespace,
    paths: ExperimentPaths,
    checkpoint_candidates: list[dict],
) -> None:
    """Schedule checkpoint jobs dynamically, one CUDA-isolated process per GPU."""
    devices = resolve_generation_gpu_devices(args.generation_gpus, args.max_generation_workers)
    print_generation_diagnostics(args.generation_gpus, devices)
    print("Orchestrated generate mode: one TensorFlow subprocess per checkpoint.")
    print("CFG_LABEL:", format_cfg_label(args.guidance_scale, args.sample_steps))
    for candidate in checkpoint_candidates:
        for cls in parse_classes(args.classes):
            prepare_sweep_generation_manifest(args, paths, candidate, cls, dry_run=args.dry_run)
    jobs = [
        {
            **candidate,
            "label": (
                f"{paths.experiment_dir.name}_classes_0_1_evaluation_"
                f"checkpoint_{candidate['checkpoint_id']}_worker_{worker_id}"
            ),
        }
        for worker_id, candidate in enumerate(checkpoint_candidates)
    ]
    logs_dir = (
        paths.logs_dir / "parallel_generation" / "run_dry_run"
        if args.dry_run else create_parallel_run_dir(paths.logs_dir)
    )
    run_dynamic_gpu_jobs(
        jobs=jobs,
        devices=devices,
        command_for_job=lambda job, gpu: child_generate_command(
            args, paths, job["checkpoint_id"], gpu
        ),
        logs_dir=logs_dir,
        dry_run=args.dry_run,
        cwd=paths.project_root,
    )


def save_single_fake_image(image_np, output_path: Path) -> None:
    """Save a generated PNG, mapping [-1, 1] values to [0, 255] when needed."""
    import numpy as np
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = image_np.copy()
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0
    tmp_path = output_path.with_name(f".tmp_gen_{output_path.stem}_{os.getpid()}.png")
    Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).save(tmp_path)
    with Image.open(tmp_path) as image:
        image.verify()
    os.replace(tmp_path, output_path)


def sampler_trace_count(compiled_sampler) -> int | None:
    """Return sampler tf.function tracing count to detect unintended sweep retracing."""
    getter = getattr(compiled_sampler, "experimental_get_tracing_count", None)
    if getter is None:
        return None
    return int(getter())


def run_generation_worker(
    args: argparse.Namespace,
    paths: ExperimentPaths,
    candidate: dict,
) -> None:
    """Generate missing images for one checkpoint with one sampler and deterministic class/index seeds."""
    checkpoint_id = candidate["checkpoint_id"]
    ckpt_path = Path(candidate["path"])
    classes = parse_classes(args.classes)

    print(f"\n[checkpoint {checkpoint_id}] {ckpt_path.name}")
    print("CFG_LABEL:", format_cfg_label(args.guidance_scale, args.sample_steps))
    print("decode_on_cpu:", args.decode_on_cpu)
    print("effective sampling batch: 1")

    preflight_missing = []
    for cls in classes:
        prepare_sweep_generation_manifest(args, paths, candidate, cls)
        missing = missing_fake_image_indices(paths, args, checkpoint_id, cls)
        preflight_missing.append(len(missing))
        print(f"class {cls}: {args.n_gen_per_class - len(missing)}/{args.n_gen_per_class} already present")
        print(f"class {cls}: generating {len(missing)} missing or corrupt images")

    if sum(preflight_missing) == 0:
        print("No images are missing; TensorFlow, LDM, and VAE will not be loaded.")
        return

    configure_environment(args)

    if args.vae_backend == "sd" and not args.decode_on_cpu:
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    import tensorflow as tf

    from ldm_keras_utils import (
        build_schedule,
        configure_tensorflow,
        load_latent_stats,
        load_ldm_model,
        load_vae_decoder,
        make_compiled_latent_sampler,
        make_compiled_sampler,
        vram_gb,
    )

    configure_tensorflow(
        allow_gpu_memory_growth=args.vae_backend == "sd" and not args.decode_on_cpu
    )
    vram_gb("VRAM before checkpoint")

    schedule = build_schedule()
    latent_mean, latent_std = load_latent_stats(paths.latents_dir / "latent_stats.npz")
    ldm_model = load_ldm_model(ckpt_path)
    vram_gb("VRAM after loading LDM")

    sd_vae = sd_device = sd_dtype = None
    if args.vae_backend == "sd":
        from sd_vae_utils import decode_sd_latents_to_grayscale, load_sd_vae, resolve_sd_vae_model

        sd_model = resolve_sd_vae_model(paths.project_root, args.sd_vae_model)
        sd_device = "cpu" if args.decode_on_cpu else None
        sd_vae, sd_device, sd_dtype, _ = load_sd_vae(sd_model, device=sd_device)
        print("SD-VAE decoder:", sd_model)
        print("SD-VAE device:", sd_device)
        compiled_sampler = make_compiled_latent_sampler(
            ldm_model=ldm_model,
            schedule=schedule,
            latent_mean=latent_mean,
            latent_std=latent_std,
            num_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
            parameterization=args.parameterization,
        )
    else:
        vae_decoder = load_vae_decoder(paths.models_dir / "vae_decoder_best.keras")
        vram_gb("VRAM after loading VAE decoder")
        compiled_sampler = make_compiled_sampler(
            ldm_model=ldm_model,
            vae_decoder=vae_decoder,
            schedule=schedule,
            latent_mean=latent_mean,
            latent_std=latent_std,
            num_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
            decode_on_cpu=args.decode_on_cpu,
            parameterization=args.parameterization,
        )
    print(f"sampler compiled once and reused for the entire checkpoint (parameterization={args.parameterization})")

    for cls in classes:
        # Re-validate in the worker immediately before looking for holes.  This
        # remains cheap and catches a checkpoint replaced while jobs were queued.
        prepare_sweep_generation_manifest(args, paths, candidate, cls)
        missing_indices = missing_fake_image_indices(paths, args, checkpoint_id, cls)
        print(f"class {cls}: generating {len(missing_indices)} missing or corrupt images")

        generated = 0
        for output_index in missing_indices:
    # Use the same seed for every checkpoint at a given (class, image index), so
    # the checkpoint weights are the only sweep variable, not the initial noise.
            sample_seed = tf.constant(
                [
                    int(args.seed) + int(cls) * 1_000,
                    int(output_index),
                ],
                dtype=tf.int32,
            )
            sample_out = compiled_sampler(tf.constant(cls, dtype=tf.int32), sample_seed)
            if args.vae_backend == "sd":
                image_np = decode_sd_latents_to_grayscale(
                    sample_out.numpy(),
                    sd_vae,
                    sd_device,
                    sd_dtype,
                    batch_size=args.sd_vae_batch_size,
                )[0]
            else:
                image_np = sample_out[0].numpy().squeeze()
            output_path = fake_output_dir(paths, args, checkpoint_id, cls) / (
                f"{output_index:04d}.png"
            )
            save_single_fake_image(image_np, output_path)

            del sample_out
            del image_np
            generated += 1

            trace_count = sampler_trace_count(compiled_sampler)
            trace_msg = f" | sampler_traces={trace_count}" if trace_count is not None else ""
            print(
                f"  class {cls}: generated {generated}/{len(missing_indices)} missing "
                f"-> {output_path.name}{trace_msg}",
                flush=True,
            )
            if (
                generated == len(missing_indices)
                or (args.vram_log_every > 0 and generated % args.vram_log_every == 0)
            ):
                gc.collect()
                vram_gb(
                    f"VRAM checkpoint {checkpoint_id} class {cls} after {generated} images"
                )

        remaining = missing_fake_image_indices(paths, args, checkpoint_id, cls)
        if remaining:
            raise RuntimeError(
                f"Class {cls}, checkpoint {checkpoint_id}: missing/corrupt indices {remaining[:20]}"
            )

    print("sampler_traces_finale:", sampler_trace_count(compiled_sampler))
    del compiled_sampler
    del ldm_model
    if args.vae_backend == "keras":
        del vae_decoder
    else:
        del sd_vae
    del schedule
    del latent_mean
    del latent_std
    gc.collect()
    tf.keras.backend.clear_session()
    vram_gb("VRAM at end of checkpoint")


def ensure_metrics_classes(args: argparse.Namespace) -> None:
    """Require --classes 0,1 in metrics mode because checkpoint_metrics.json assumes both classes."""
    classes = sorted(parse_classes(args.classes))
    if classes != [0, 1]:
        raise ValueError(
            "--mode metrics requires --classes 0,1 to preserve the "
            "a compatible checkpoint_metrics.json file."
        )


def json_scalar(value):
    """Convert a NumPy/Pandas scalar to JSON-safe Python, mapping NaN to None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def select_best_row(results_df):
    """Select deterministically by ascending fid_1, descending is_mean_1, then checkpoint order."""
    required = [BEST_SELECTION_METRIC, BEST_SELECTION_TIE_BREAKER, "checkpoint_order"]
    missing = [column for column in required if column not in results_df.columns]
    if missing:
        raise RuntimeError(f"Missing metrics required to select the best checkpoint: {missing}")
    ranked_df = results_df.dropna(subset=[BEST_SELECTION_METRIC]).sort_values(
        by=[BEST_SELECTION_METRIC, BEST_SELECTION_TIE_BREAKER, "checkpoint_order"],
        ascending=[True, False, True],
    )
    if ranked_df.empty:
        raise RuntimeError(
            f"No valid result is available to select the best checkpoint with {BEST_SELECTION_METRIC}"
        )
    return ranked_df.iloc[0]


def build_selection(best_row, best_eval_model_path: Path, args: argparse.Namespace) -> dict:
    """Combine selection policy, metadata, and winning-row metrics into the best record."""
    import pandas as pd

    step_value = best_row.get("step")
    selection = {
        **selection_policy(),
        "best_checkpoint_id": str(best_row["checkpoint_id"]),
        "best_step": None if pd.isna(step_value) else int(step_value),
        "best_checkpoint": str(Path(best_row["checkpoint_path"])),
        "best_model": str(best_eval_model_path),
        "best_eval_model": str(best_eval_model_path),
        "notebook": args.notebook_name,
        "unet_version": args.unet_version,
        "parameterization": args.parameterization,
        "vae_source": args.vae_source,
        "uses_vae_ft_from_03": args.uses_vae_ft_from_03,
    }
    for column in BEST_ROW_METRIC_COLUMNS:
        if column in best_row:
            selection[column] = json_scalar(best_row[column])
    return selection


def copy_artifact(source: Path, destination: Path) -> Path:
    """Copy a CSV, JSON, or plot artifact into results, creating directories as needed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def plot_checkpoint_metric_comparisons(results_df, selection: dict, plots_dir: Path) -> dict:
    """Plot sweep FID and Inception Score and mark the selected checkpoint."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_df = results_df.sort_values("checkpoint_order").reset_index(drop=True)
    x = np.arange(len(plot_df))
    labels = plot_df["checkpoint_id"].astype(str).tolist()
    best_id = selection["best_checkpoint_id"]
    best_positions = plot_df.index[plot_df["checkpoint_id"].astype(str) == best_id].tolist()
    best_x = best_positions[0] if best_positions else None

    saved = {}
    comparisons = [
        (
            "checkpoint_fid_comparison.png",
            [("fid_0", "FID class 0"), ("fid_1", "FID class 1"), ("fid_mean", "Mean FID")],
            "Checkpoint FID comparison - BEST = minimum fid_1",
            "FID",
            "lower",
        ),
        (
            "checkpoint_is_comparison.png",
            [
                ("is_mean_0", "IS class 0"),
                ("is_mean_1", "IS class 1"),
                ("is_mean_avg", "Mean IS"),
            ],
            "Checkpoint Inception Score comparison - is_mean_1 is only a tie-breaker",
            "Inception Score",
            "higher",
        ),
    ]

    for filename, series, title, ylabel, direction in comparisons:
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        for column, label in series:
            ax.plot(x, plot_df[column].astype(float), marker="o", linewidth=1.8, label=label)
        if best_x is not None:
            ax.axvline(best_x, color="crimson", linestyle="--", linewidth=1.4)
            y_values = [plot_df.loc[best_x, column] for column, _ in series]
            y_anchor = min(y_values) if direction == "lower" else max(y_values)
            ax.annotate(
                f"BEST: {best_id}",
                xy=(best_x, y_anchor),
                xytext=(best_x, y_anchor),
                textcoords="data",
                ha="center",
                va="bottom" if direction == "lower" else "top",
                color="crimson",
                fontweight="bold",
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")
        output_path = plots_dir / filename
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        saved[filename] = output_path
    return saved


def plot_best_checkpoint_summary(results_df, selection: dict, plots_dir: Path) -> Path:
    """Compose a text, FID, IS, and PRDC summary figure for the winning checkpoint."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_id = selection["best_checkpoint_id"]
    best_row = results_df.loc[results_df["checkpoint_id"].astype(str) == best_id].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    axes[0, 0].axis("off")
    axes[0, 0].text(
        0.02,
        0.95,
        "\n".join(
            [
                f"Best checkpoint: {best_id}",
                "Selection criterion: minimum positive-class FID",
                "Tie-breaker: maximum positive-class Inception Score",
                f"Positive-class FID: {float(best_row['fid_1']):.4f}",
                f"Mean FID: {float(best_row['fid_mean']):.4f}",
            ]
        ),
        va="top",
        fontsize=12,
    )

    fid_values = [best_row["fid_0"], best_row["fid_1"], best_row["fid_mean"]]
    axes[0, 1].bar(["class 0", "class 1", "mean"], fid_values, color=["#4C78A8", "#E45756", "#72B7B2"])
    axes[0, 1].set_title("Best-checkpoint FID")
    axes[0, 1].set_ylabel("FID")
    axes[0, 1].grid(axis="y", alpha=0.2)

    is_values = [best_row["is_mean_0"], best_row["is_mean_1"], best_row["is_mean_avg"]]
    axes[1, 0].bar(["class 0", "class 1", "mean"], is_values, color=["#4C78A8", "#E45756", "#72B7B2"])
    axes[1, 0].set_title("Best-checkpoint Inception Score")
    axes[1, 0].set_ylabel("IS mean")
    axes[1, 0].grid(axis="y", alpha=0.2)

    grouped_metrics = ["precision", "recall", "density", "coverage"]
    x = range(len(grouped_metrics))
    class0 = [best_row[f"{metric}_0"] for metric in grouped_metrics]
    class1 = [best_row[f"{metric}_1"] for metric in grouped_metrics]
    axes[1, 1].bar([pos - 0.18 for pos in x], class0, width=0.36, label="class 0", color="#4C78A8")
    axes[1, 1].bar([pos + 0.18 for pos in x], class1, width=0.36, label="class 1", color="#E45756")
    axes[1, 1].set_title("PRDC by class")
    axes[1, 1].set_xticks(list(x))
    axes[1, 1].set_xticklabels(grouped_metrics)
    axes[1, 1].grid(axis="y", alpha=0.2)
    axes[1, 1].legend(loc="best")

    output_path = plots_dir / "best_checkpoint_summary.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def write_manifest(
    paths: ExperimentPaths,
    selection: dict,
    artifact_paths: dict[str, Path],
    stage_name: str,
) -> dict:
    """Save the stage selection, policy, and artifact manifest in evaluation and results."""
    dirs = results_stage_dirs(paths, stage_name)
    manifest = {
        "schema_version": 1,
        "stage": stage_name,
        "selection": selection,
        "selection_policy": selection_policy(),
        "artifacts": {name: str(path) for name, path in sorted(artifact_paths.items())},
    }
    manifest_paths = [
        paths.evaluation_dir / "artifacts_manifest.json",
        dirs["stage"] / "artifacts_manifest.json",
    ]
    for manifest_path in manifest_paths:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)
    return manifest


def persist_sweep_artifacts(
    args: argparse.Namespace,
    paths: ExperimentPaths,
    checkpoint_candidates: list[dict],
    results_df,
    copy_best_model: bool,
) -> dict:
    """Select the best checkpoint and persist its model, tables, plots, and manifests."""
    dirs = results_stage_dirs(paths, args.results_stage_name)
    results_path = paths.evaluation_dir / "sweep_results.csv"
    summary_path = paths.evaluation_dir / "sweep_summary.json"
    best_path = paths.evaluation_dir / "best_checkpoint.json"
    metrics_json_path = paths.evaluation_dir / "checkpoint_metrics.json"

    with maybe_measure(args, paths, "ldm_select_best_checkpoint"):
        best_row = select_best_row(results_df)
        best_checkpoint_path = Path(best_row["checkpoint_path"])
        if not best_checkpoint_path.exists():
            raise FileNotFoundError(best_checkpoint_path)

        best_eval_model_path = paths.checkpoints_dir / "ldm_unet_best_eval.keras"
        if copy_best_model:
            shutil.copy2(best_checkpoint_path, best_eval_model_path)

        results_df = results_df.sort_values("checkpoint_order").reset_index(drop=True)
        results_df.to_csv(results_path, index=False)
        canonical_results_path = copy_artifact(results_path, dirs["metrics"] / "sweep_results.csv")

        selection = build_selection(best_row, best_eval_model_path, args)
        checkpoint_records = json.loads(results_df.to_json(orient="records"))
        metrics_input_signature = sweep_metrics_input_signature(args, paths)
        metrics_payload = {
            "schema_version": 2,
            "config": build_eval_config(args),
            "candidate_signature": build_candidate_signature(checkpoint_candidates),
            "generation_image_signature": sweep_generated_image_signature(
                args, paths, checkpoint_candidates
            ),
            "metrics_input_signature": metrics_input_signature,
            **metrics_input_signature,
            "selection_policy": selection_policy(),
            "checkpoints": checkpoint_records,
            "selection": selection,
        }
        with open(metrics_json_path, "w", encoding="utf-8") as file:
            json.dump(metrics_payload, file, indent=2, ensure_ascii=False)
        canonical_metrics_json = copy_artifact(
            metrics_json_path,
            dirs["metrics"] / "checkpoint_metrics.json",
        )

    with maybe_measure(args, paths, "ldm_generate_evaluation_plots"):
        plots = plot_checkpoint_metric_comparisons(results_df, selection, dirs["plots"])
        summary_plot = plot_best_checkpoint_summary(results_df, selection, dirs["plots"])
        plots["best_checkpoint_summary.png"] = summary_plot
        eval_plot_paths = {
            name: copy_artifact(path, dirs["evaluation_plots"] / name)
            for name, path in plots.items()
        }

    summary = {
        "n_checkpoints": len(results_df),
        "n_gen_per_class": args.n_gen_per_class,
        "sample_steps": args.sample_steps,
        "guidance_scale": args.guidance_scale,
        "results_csv": str(results_path),
        "canonical_results_csv": str(canonical_results_path),
        "best_metric": BEST_SELECTION_METRIC,
        "checkpoint_metrics_json": str(metrics_json_path),
        "canonical_checkpoint_metrics_json": str(canonical_metrics_json),
        "plots": {name: str(path) for name, path in sorted(plots.items())},
        **selection_policy(),
        "best_checkpoint_id": selection["best_checkpoint_id"],
        "best_step": selection["best_step"],
        "best_checkpoint": selection["best_checkpoint"],
        "best_eval_model": selection["best_eval_model"],
        "notebook": args.notebook_name,
        "unet_version": args.unet_version,
        "parameterization": args.parameterization,
        "vae_source": args.vae_source,
        "uses_vae_ft_from_03": args.uses_vae_ft_from_03,
    }
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    canonical_summary_path = copy_artifact(summary_path, dirs["metrics"] / "sweep_summary.json")

    with open(best_path, "w", encoding="utf-8") as file:
        json.dump(selection, file, indent=2, ensure_ascii=False)
    canonical_best_path = copy_artifact(best_path, dirs["metrics"] / "best_checkpoint.json")

    artifact_paths = {
        "sweep_results_csv": results_path,
        "canonical_sweep_results_csv": canonical_results_path,
        "checkpoint_metrics_json": metrics_json_path,
        "canonical_checkpoint_metrics_json": canonical_metrics_json,
        "sweep_summary_json": summary_path,
        "canonical_sweep_summary_json": canonical_summary_path,
        "best_checkpoint_json": best_path,
        "canonical_best_checkpoint_json": canonical_best_path,
        "best_eval_model": best_eval_model_path,
        **{f"plot_{name}": path for name, path in plots.items()},
        **{f"evaluation_plot_{name}": path for name, path in eval_plot_paths.items()},
    }
    ecotracker_jsonl = dirs["ecotracker"] / "ldm_evaluation_ecotracker.jsonl"
    if ecotracker_jsonl.exists():
        artifact_paths["evaluation_ecotracker_jsonl"] = ecotracker_jsonl
    manifest = write_manifest(paths, selection, artifact_paths, args.results_stage_name)
    return {
        "selection": selection,
        "results_path": results_path,
        "metrics_json_path": metrics_json_path,
        "summary_path": summary_path,
        "best_path": best_path,
        "manifest": manifest,
        "plots": plots,
    }


def refresh_artifacts_from_cache(
    args: argparse.Namespace,
    paths: ExperimentPaths,
    checkpoint_candidates: list[dict],
    payload: dict,
) -> bool:
    """Regenerate artifacts from cached metrics after verifying the selected checkpoint.

    In `--mode artifacts`, rebuild CSV, JSON, plots, and the manifest only after confirming
    that the recomputed best checkpoint matches the saved selection.
    """
    import pandas as pd

    try:
        results_df = pd.DataFrame(payload["checkpoints"])
        best_row = select_best_row(results_df)
    except Exception as exc:
        print(f"Metric cache lacks a valid checkpoint table; recomputing: {exc}")
        return False

    selection = payload.get("selection", {})
    cached_best_id = selection.get("best_checkpoint_id")
    expected_best_id = str(best_row["checkpoint_id"])
    if cached_best_id is not None and cached_best_id != expected_best_id:
        print(
            "Metric cache is inconsistent: saved best "
            f"{cached_best_id}, fid_1 best {expected_best_id}."
        )
        return False

    persist_sweep_artifacts(
        args=args,
        paths=paths,
        checkpoint_candidates=checkpoint_candidates,
        results_df=results_df,
        copy_best_model=True,
    )
    return True


def run_metrics(
    args: argparse.Namespace,
    paths: ExperimentPaths,
    checkpoint_candidates: list[dict],
) -> None:
    """Compute per-checkpoint FID/IS/PRDC against real validation classes and persist the sweep."""
    ensure_metrics_classes(args)

    import pandas as pd

    from ldm_evaluation_utils import evaluate_generated_paths_against_metadata

    if args.inception_weights == "none":
        print(
            "Note: --inception-weights=none is ignored by generative_evaluator.py; "
            "torchmetrics uses its own Inception model."
        )

    val_csv_path = paths.metadata_dir / "val.csv"
    val_df = pd.read_csv(val_csv_path)
    print("Real metric reference: validation set")
    print(val_df["label"].astype(int).value_counts().sort_index().to_string())

    results = []
    with maybe_measure(args, paths, "ldm_evaluate_checkpoints"):
        for candidate in checkpoint_candidates:
            checkpoint_id = candidate["checkpoint_id"]
            ckpt_path = Path(candidate["path"])
            step = candidate["step"]
            print(f"\n{'=' * 60}\n[METRICS] {checkpoint_id} - {ckpt_path.name}")
            row_result = {
                "checkpoint_id": checkpoint_id,
                "checkpoint_kind": candidate["kind"],
                "checkpoint_path": str(ckpt_path),
                "checkpoint_order": candidate["order"],
                "step": step,
            }

            for cls in [0, 1]:
                prepare_sweep_generation_manifest(args, paths, candidate, cls)
                missing = missing_fake_image_indices(paths, args, checkpoint_id, cls)
                if missing:
                    raise RuntimeError(
                        f"Class {cls}, checkpoint {checkpoint_id}: missing or corrupt indices {missing} in "
                        f"{fake_output_dir(paths, args, checkpoint_id, cls)}"
                    )
                fake_paths = [
                    fake_output_dir(paths, args, checkpoint_id, cls) / f"{index:04d}.png"
                    for index in range(args.n_gen_per_class)
                ]

                row_result[f"existing_before_{cls}"] = len(fake_paths)
                row_result[f"generated_now_{cls}"] = 0
                row_result[f"images_used_{cls}"] = len(fake_paths)
                metrics = evaluate_generated_paths_against_metadata(
                    generated_paths=fake_paths,
                    metadata_df=val_df,
                    data_processed_dir=paths.data_processed_dir,
                    label=cls,
                    batch_size=args.inception_batch,
                    nearest_k=args.knn_k,
                    is_splits=args.is_splits,
                )

                row_result[f"fid_{cls}"] = metrics["FID"]
                row_result[f"is_mean_{cls}"] = metrics["IS_mean"]
                row_result[f"is_std_{cls}"] = metrics["IS_std"]
                row_result[f"is_splits_{cls}"] = args.is_splits
                row_result[f"precision_{cls}"] = metrics["precision"]
                row_result[f"recall_{cls}"] = metrics["recall"]
                row_result[f"density_{cls}"] = metrics["density"]
                row_result[f"coverage_{cls}"] = metrics["coverage"]
                row_result[f"real_reference_{cls}"] = metrics["n_real_reference"]
                print(
                    f"  cls {cls}: FID={metrics['FID']:.2f} "
                    f"IS={metrics['IS_mean']:.3f}+/-{metrics['IS_std']:.3f} "
                    f"Prec={metrics['precision']:.3f} Rec={metrics['recall']:.3f}"
                )
                del fake_paths
                gc.collect()

            row_result["fid_mean"] = (row_result["fid_0"] + row_result["fid_1"]) / 2
            row_result["is_mean_avg"] = (
                row_result["is_mean_0"] + row_result["is_mean_1"]
            ) / 2
            row_result["precision_mean"] = (
                row_result["precision_0"] + row_result["precision_1"]
            ) / 2
            row_result["recall_mean"] = (
                row_result["recall_0"] + row_result["recall_1"]
            ) / 2
            row_result["density_mean"] = (
                row_result["density_0"] + row_result["density_1"]
            ) / 2
            row_result["coverage_mean"] = (
                row_result["coverage_0"] + row_result["coverage_1"]
            ) / 2
            results.append(row_result)

    results_df = (
        pd.DataFrame(results)
        .sort_values("checkpoint_order")
        .reset_index(drop=True)
    )
    artifacts = persist_sweep_artifacts(
        args=args,
        paths=paths,
        checkpoint_candidates=checkpoint_candidates,
        results_df=results_df,
        copy_best_model=True,
    )
    selection = artifacts["selection"]
    print("\n=== SWEEP RESULTS ===")
    print(results_df.to_string(index=False))
    print("Saved:", artifacts["results_path"])
    print("Saved:", artifacts["metrics_json_path"])
    print("Saved:", paths.evaluation_dir / "artifacts_manifest.json")
    print(
        f"Best checkpoint ({BEST_SELECTION_METRIC}): "
        f"{selection['best_checkpoint_id']} -> {selection['best_eval_model']}"
    )


def main() -> None:
    """Prepare the environment and candidates, then dispatch generate/metrics/artifacts/both."""
    args = parse_args()
    configure_environment(args)
    if args.mini_batch != 1:
        print(
            f"Note: --mini-batch={args.mini_batch} is ignored for the sweep; "
            "using fixed batch size 1."
        )
        args.mini_batch = 1

    paths = get_experiment_paths(args.project_root, args.experiment_dir, create=not args.dry_run)
    if args.checkpoints_dir is not None:
        checkpoint_source = args.checkpoints_dir.expanduser()
        if not checkpoint_source.is_absolute():
            checkpoint_source = paths.project_root / checkpoint_source
        checkpoint_source = checkpoint_source.resolve()
        if not checkpoint_source.is_dir():
            raise FileNotFoundError(f"Shared checkpoint directory not found: {checkpoint_source}")
        paths = replace(paths, checkpoints_dir=checkpoint_source)
    checkpoint_candidates = collect_checkpoint_candidates(paths, args)
    print(f"Checkpoint candidates: {len(checkpoint_candidates)}")
    for candidate in checkpoint_candidates:
        print(f"  {candidate['checkpoint_id']} -> {Path(candidate['path']).name}")

    if args.dry_run:
        # A dry run never reaches a child worker or expensive metrics/artifact stages.
        orchestrate_generation(args, paths, checkpoint_candidates)
        return

    if args.mode == "generate" and args.generation_worker:
        run_generation_worker(args, paths, checkpoint_candidates[0])
        return

    if args.mode == "generate":
        orchestrate_generation(args, paths, checkpoint_candidates)
        return

    if use_cached_metrics_if_valid(args, paths, checkpoint_candidates):
        return

    if args.mode == "artifacts":
        raise RuntimeError(
            "Metric cache is invalid or absent; artifacts cannot be rebuilt without "
            "a compatible checkpoint_metrics.json."
        )

    if args.mode == "both":
        orchestrate_generation(args, paths, checkpoint_candidates)
        run_metrics(args, paths, checkpoint_candidates)
        return

    if args.mode == "metrics":
        run_metrics(args, paths, checkpoint_candidates)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
