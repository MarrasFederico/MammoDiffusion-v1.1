#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path


from ldm_project_paths import (
    RESULTS_STAGE_NAME,
    class_name_for_label,
    get_class_evaluation_dir,
    get_class_image_dirs,
    get_class_metrics_dir,
    get_experiment_paths,
    get_results_paths,
    normalize_processed_path,
)
from parallel_generation_utils import (
    checkpoint_content_signature,
    acquire_parallel_generation_lock,
    create_parallel_run_dir,
    file_content_signature,
    claim_index,
    claim_next_chunk,
    complete_claimed_chunk,
    DEFAULT_GENERATION_RESERVATION_SIZE,
    DEFAULT_GENERATION_SCHEDULER,
    partition_indices,
    prepare_dynamic_queue,
    print_generation_diagnostics,
    resolve_generation_gpu_devices,
    run_dynamic_gpu_jobs,
    release_index_claim,
    release_parallel_generation_lock,
    sd_base_model_signature,
)


def parse_args() -> argparse.Namespace:
    """Parse generation, filtering, validation and visualization modes."""
    parser = argparse.ArgumentParser(
        description="Generate, filter and evaluate MammoDiffusion LDM final images."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--gpu-visible-devices", default=None)
    parser.add_argument(
        "--generation-gpus",
        default="auto",
        help="GPU per la sola generazione RAW: auto, off o lista comma-separata.",
    )
    parser.add_argument("--max-generation-workers", type=int, default=None)
    parser.add_argument("--generation-scheduler", choices=["dynamic_reservations", "round_robin"], default=DEFAULT_GENERATION_SCHEDULER)
    parser.add_argument("--generation-reservation-size", type=int, default=DEFAULT_GENERATION_RESERVATION_SIZE)
    parser.add_argument("--generation-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--generation-indices-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--generation-queue-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Mostra GPU, shard e comandi senza caricare modelli.")
    parser.add_argument(
        "--cuda-root",
        type=Path,
        default=Path(os.environ.get("MAMMODIFFUSION_CUDA_ROOT", sys.prefix)),
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["generate", "filter", "validate", "all", "reverse", "both"],
        default="generate",
    )
    parser.add_argument("--target-label", type=int, default=1)
    parser.add_argument("--n-raw", type=int, default=2722)
    parser.add_argument("--n-selected", type=int, default=1361)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Mantenuto per compatibilita'; la generazione LDM finale forza batch 1.",
    )
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--nonblack-threshold", type=int, default=10)
    parser.add_argument("--decode-on-cpu", action="store_true")
    parser.add_argument(
        "--vae-backend",
        choices=["keras", "sd"],
        default="keras",
        help="Decoder da usare per trasformare i latenti in immagini.",
    )
    parser.add_argument(
        "--sd-vae-model",
        default=None,
        help="Path locale del modello Stable Diffusion da cui caricare il VAE.",
    )
    parser.add_argument("--sd-vae-batch-size", type=int, default=1)
    parser.add_argument(
        "--parameterization",
        choices=["eps", "v"],
        default="eps",
        help=(
            "Parameterization del modello LDM caricato: eps (default, retrocompatibile "
            "con 04b/04b1/04b2) oppure v (Salimans & Ho 2022, usata da 04b3). Deve "
            "corrispondere a come il checkpoint e' stato addestrato (vedi training_manifest.json)."
        ),
    )
    parser.add_argument(
        "--unet-version",
        choices=["v2", "v3"],
        default="v2",
        help="Informativo: architettura del checkpoint caricato, salvato nei manifest di output.",
    )
    parser.add_argument(
        "--vae-source",
        default="sd_vae_original",
        help="Informativo: sorgente del VAE usato per i latenti, salvato nei manifest di output.",
    )
    parser.add_argument(
        "--uses-vae-ft-from-03",
        action="store_true",
        help="Informativo: marca nei manifest che il VAE e' quello fine-tuned di 03.",
    )
    parser.add_argument(
        "--notebook-name",
        default=None,
        help="Informativo: nome del notebook chiamante, salvato nei manifest di output.",
    )
    parser.add_argument("--vram-log-every", type=int, default=25)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-seed", type=int, default=42)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--filtered-dir", type=Path, default=None)
    parser.add_argument(
        "--results-stage-name",
        default=RESULTS_STAGE_NAME,
        help="Sottocartella di results dove salvare metriche, plot e log EcoTracker.",
    )
    parser.add_argument("--inception-batch", type=int, default=8)
    parser.add_argument("--is-splits", type=int, default=10)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument(
        "--inception-weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Mantenuto per compatibilita'; generative_evaluator.py usa torchmetrics.",
    )
    parser.add_argument(
        "--max-eval-images",
        type=int,
        default=None,
        help="Limite leggero sul numero di immagini per le metriche; default usa tutti gli input.",
    )
    parser.add_argument("--reverse-labels", default="1,0")
    parser.add_argument("--reverse-steps-to-show", type=int, default=10)
    parser.add_argument("--eco-track", action="store_true")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    """Imposta le variabili d'ambiente necessarie prima di importare TensorFlow (cartella matplotlib, GPU visibili, percorso libdevice per XLA)."""
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mammodiffusion-matplotlib")
    )
    if args.gpu_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible_devices
    libdevice = args.cuda_root / "nvvm" / "libdevice" / "libdevice.10.bc"
    if libdevice.exists() and "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={args.cuda_root}"


def is_valid_keras_file(path: Path) -> bool:
    """Verifica che il file checkpoint esista, non sia vuoto e sia un archivio .keras leggibile (evita di caricare pesi troncati)."""
    return path.exists() and path.stat().st_size > 0 and zipfile.is_zipfile(path)


def resolve_model_path(exp: Path, requested: Path | None = None) -> Path:
    """Seleziona il checkpoint U-Net da usare per la generazione: usa quello passato esplicitamente se valido, altrimenti cerca tra i candidati noti (best_eval, best, ultimo step) nella cartella dell'esperimento."""
    if requested is not None:
        requested = Path(requested).expanduser().resolve()
        if is_valid_keras_file(requested):
            return requested
        raise FileNotFoundError(f"Modello LDM non valido: {requested}")

    ckpt_dir = exp / "checkpoints_ldm"
    models_dir = exp / "models"
    candidates = [
        ckpt_dir / "ldm_unet_best_eval.keras",
        ckpt_dir / "ldm_unet_best.keras",
        models_dir / "ldm_unet_best_eval.keras",
        models_dir / "ldm_unet_best.keras",
    ]
    candidates.extend(sorted(ckpt_dir.glob("ldm_step*.keras"), reverse=True))
    candidates.extend(sorted(ckpt_dir.glob("ldm_unet_final_step*.keras"), reverse=True))
    for candidate in candidates:
        if is_valid_keras_file(candidate):
            return candidate
    raise FileNotFoundError(f"Nessun modello LDM valido trovato in {exp}")


def resolve_image_dirs(paths, args: argparse.Namespace) -> tuple[Path, Path]:
    """Determina raw e filtered, usando percorsi canonici distinti per classe."""
    canonical_raw_dir, canonical_filtered_dir = get_class_image_dirs(paths, args.target_label)
    raw_dir = (
        args.raw_dir.expanduser().resolve()
        if args.raw_dir is not None
        else canonical_raw_dir
    )
    filtered_dir = (
        args.filtered_dir.expanduser().resolve()
        if args.filtered_dir is not None
        else canonical_filtered_dir
    )
    if not getattr(args, "dry_run", False):
        raw_dir.mkdir(parents=True, exist_ok=True)
        filtered_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, filtered_dir


def is_canonical_image_run(paths, args: argparse.Namespace, raw_dir: Path, filtered_dir: Path) -> bool:
    """Riconosce come canonici anche gli override che coincidono con la classe richiesta."""
    canonical_raw_dir, canonical_filtered_dir = get_class_image_dirs(paths, args.target_label)
    return raw_dir == canonical_raw_dir and filtered_dir == canonical_filtered_dir


def mirror_positive_legacy_outputs(source_dir: Path, legacy_dir: Path, target_label: int) -> None:
    """Mantiene leggibili i percorsi positivi storici mentre i nuovi output sono class-scoped."""
    # Con un filtro ripreso da cache i report possono esistere senza grafici: in quel
    # caso source_dir non viene creato e il mirror e' semplicemente non necessario.
    if int(target_label) != 1 or source_dir == legacy_dir or not source_dir.is_dir():
        return
    for source in source_dir.iterdir():
        if source.is_file():
            copy_if_exists(source, legacy_dir / source.name)


def expected_raw_path(raw_dir: Path, index: int) -> Path:
    """Costruisce il path attesso per l'immagine raw di indice dato, con padding a 5 cifre (convenzione synth_NNNNN.png)."""
    return raw_dir / f"synth_{index:05d}.png"


def raw_index_from_name(path: Path) -> int | None:
    """Estrae l'indice numerico dal nome file synth_NNNNN.png, restituendo None se il nome non rispetta la convenzione."""
    stem = path.stem
    if not stem.startswith("synth_"):
        return None
    try:
        return int(stem.replace("synth_", "", 1))
    except ValueError:
        return None


def is_readable_png(path: Path) -> bool:
    """Verifica che il PNG esista e sia decodificabile (verify + conversione in scala di grigi), per scartare file troncati da run interrotte."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("L").load()
        return True
    except Exception:
        return False


def scan_raw_generation_state(raw_dir: Path, n_raw: int, force_recompute: bool = False) -> dict:
    """Classifica gli indici da 0 a n_raw in validi/mancanti/corrotti scandendo la cartella raw, per riprendere la generazione senza ripartire da zero; con force_recompute considera tutto da rigenerare."""
    valid_indices: list[int] = []
    missing_indices: list[int] = []
    corrupt_indices: list[int] = []
    corrupt_files: list[str] = []
    if force_recompute:
        return {
            "valid_indices": [],
            "missing_indices": list(range(n_raw)),
            "corrupt_indices": [],
            "corrupt_files": [],
            "extra_files": [
                str(path)
                for path in sorted(raw_dir.glob("*.png"))
                if not path.name.startswith(".tmp_") and (raw_index_from_name(path) is None or raw_index_from_name(path) >= n_raw)
            ],
            "force_recompute": True,
        }

    for index in range(n_raw):
        path = expected_raw_path(raw_dir, index)
        if not path.exists():
            missing_indices.append(index)
        elif is_readable_png(path):
            valid_indices.append(index)
        else:
            corrupt_indices.append(index)
            corrupt_files.append(str(path))

    extra_files = []
    for path in sorted(raw_dir.glob("*.png")):
        if path.name.startswith(".tmp_"):
            continue
        index = raw_index_from_name(path)
        if index is None or index < 0 or index >= n_raw:
            extra_files.append(str(path))

    return {
        "valid_indices": valid_indices,
        "missing_indices": missing_indices,
        "corrupt_indices": corrupt_indices,
        "corrupt_files": corrupt_files,
        "extra_files": extra_files,
        "force_recompute": False,
    }


def write_json(path: Path, payload: dict) -> None:
    """Scrive un dizionario su file JSON indentato, creando le cartelle intermedie se necessario."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def atomic_write_json(path: Path, payload: dict) -> None:
    """Atomically publish a manifest: readers never observe a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".tmp_{path.name}.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp_path, path)


def ldm_vae_signature(args: argparse.Namespace, paths) -> dict:
    """Return the effective decoder identity without loading a model."""
    if args.vae_backend == "sd":
        from sd_vae_utils import resolve_sd_vae_model

        return sd_base_model_signature(Path(resolve_sd_vae_model(paths.project_root, args.sd_vae_model)))
    return checkpoint_content_signature(paths.models_dir / "vae_decoder_best.keras")


def raw_generation_manifest_payload(args: argparse.Namespace, paths, raw_dir: Path) -> dict:
    """Operational parameters required to resume a RAW output directory safely."""
    model_path = resolve_model_path(paths.experiment_dir, args.model_path)
    stage_name = getattr(args, "results_stage_name", None)
    generator_id = str(stage_name).split("/")[-1] if stage_name else Path(paths.experiment_dir).name
    return {
        "schema_version": 2,
        "generator_id": generator_id,
        "model_family": "ldm",
        "checkpoint_path": str(model_path),
        "seed_strategy": "stateless_seed_per_image_v1",
        "seed": int(args.seed),
        "target_label": int(args.target_label),
        "n_raw": int(args.n_raw),
        "expected_count": int(args.n_raw),
        "image_size": 512,
        "output_directory": str(raw_dir),
        "sample_steps": int(args.sample_steps),
        "guidance_scale": float(args.guidance_scale),
        "parameterization": args.parameterization,
        "unet_version": args.unet_version,
        "vae_backend": args.vae_backend,
        "vae_source": args.vae_source,
        "decode_on_cpu": bool(getattr(args, "decode_on_cpu", False)),
        "sd_vae_batch_size": int(getattr(args, "sd_vae_batch_size", 1)),
        "latent_stats_signature": checkpoint_content_signature(
            getattr(paths, "latents_dir", paths.experiment_dir / "latents") / "latent_stats.npz"
        ),
        "sd_vae_model_signature": ldm_vae_signature(args, paths),
        "model_signature": checkpoint_content_signature(model_path),
    }


def prepare_raw_generation_manifest(
    args: argparse.Namespace, paths, raw_dir: Path, *, parent: bool, dry_run: bool = False
) -> dict:
    """Validate RAW resume compatibility before inspecting missing image indices.

    Only the parent may create/replace the global manifest.  A worker merely
    validates the already-published manifest, preserving the multi-GPU protocol.
    """
    raw_dir = Path(raw_dir)
    manifest_path = raw_dir / ".generation_manifest.json"
    expected = raw_generation_manifest_payload(args, paths, raw_dir)
    png_paths = [path for path in raw_dir.glob("*.png") if not path.name.startswith(".tmp_")]
    current = None
    if manifest_path.is_file():
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"RAW generation manifest unreadable: {manifest_path}") from exc
    if current != expected:
        if dry_run:
            changed = sorted(key for key in set(current or {}) | set(expected) if (current or {}).get(key) != expected.get(key))
            print(f"DRY-RUN: manifest RAW incompatibile ({changed}); verrebbe sostituito senza riusare i PNG.")
            return expected
        if not parent:
            raise RuntimeError(
                f"RAW generation manifest is missing or incompatible in worker: {manifest_path}. "
                "Restart the parent orchestration."
            )
        if current is None and png_paths and not args.force_recompute:
            raise RuntimeError(
                f"RAW directory {raw_dir} contains PNGs without .generation_manifest.json; "
                "refusing resume. Use a new directory or --force-recompute."
            )
        if current is not None and not args.force_recompute:
            changed = sorted(key for key in set(current) | set(expected) if current.get(key) != expected.get(key))
            raise RuntimeError(
                f"Incompatible RAW generation manifest in {raw_dir}: {changed}. "
                "Use a new directory or --force-recompute; existing images will not be reused."
            )
        if args.force_recompute:
            for path in png_paths:
                path.unlink()
        atomic_write_json(manifest_path, expected)
    return expected


def sampler_trace_count(compiled_sampler) -> int | None:
    """Legge quante volte la funzione tf.function del sampler e' stata ritracciata, utile per verificare che la compilazione sia avvenuta una sola volta."""
    getter = getattr(compiled_sampler, "experimental_get_tracing_count", None)
    if getter is None:
        return None
    return int(getter())


def save_single_image(image_np, output_path: Path) -> None:
    """Salva su disco l'immagine decodificata dal VAE come PNG in scala di grigi, scrivendo prima su file temporaneo e poi rinominando per evitare file parziali in caso di interruzione."""
    import numpy as np
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image_np)
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0
    arr = np.squeeze(arr)
    tmp_path = output_path.with_name(f".tmp_gen_{output_path.stem}_{os.getpid()}.png")
    Image.fromarray(
        np.clip(arr * 255.0, 0, 255).astype(np.uint8),
        mode="L",
    ).save(tmp_path)
    with Image.open(tmp_path) as image:
        image.verify()
    os.replace(tmp_path, output_path)


def append_jsonl(path: Path, payload: dict) -> None:
    """Accoda una riga JSON al file di log (jsonl), forzando flush e fsync per non perdere righe se il processo viene interrotto bruscamente."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def manifest_operational_fields(args: argparse.Namespace) -> dict:
    """Sampling/model fields needed to interpret generated outputs."""
    return {
        "notebook": args.notebook_name,
        "unet_version": args.unet_version,
        "parameterization": args.parameterization,
        "vae_source": args.vae_source,
        "uses_vae_ft_from_03": args.uses_vae_ft_from_03,
    }


def write_pipeline_manifest(paths, payload: dict, results_stage_name: str) -> None:
    """Aggiorna il manifest cumulativo della classe senza sovrascrivere l'altra classe."""
    results_paths = get_results_paths(paths.project_root, results_stage_name)
    target_label = int(payload["target_label"])
    class_evaluation_dir = get_class_evaluation_dir(paths, target_label)
    class_metrics_dir = get_class_metrics_dir(results_paths, target_label)
    manifest_paths = [
        class_evaluation_dir / "ldm_pipeline_manifest.json",
        class_metrics_dir / "ldm_pipeline_manifest.json",
    ]
    for manifest_path in manifest_paths:
        existing = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as file:
                    existing = json.load(file)
            except Exception:
                existing = {}
        existing.setdefault("schema_version", 1)
        existing.setdefault("stages", {})
        existing["stages"][payload["stage"]] = payload
        write_json(manifest_path, existing)
    if target_label == 1:
        copy_if_exists(
            class_evaluation_dir / "ldm_pipeline_manifest.json",
            paths.evaluation_dir / "ldm_pipeline_manifest.json",
        )
        copy_if_exists(
            class_metrics_dir / "ldm_pipeline_manifest.json",
            results_paths.metrics_dir / "ldm_pipeline_manifest.json",
        )


@contextmanager
def maybe_measure(args: argparse.Namespace, paths, label: str):
    """Context manager che avvolge una fase della pipeline con EcoTracker se --eco-track e' attivo, degradando silenziosamente a no-op se il tracker non e' disponibile o non si avvia."""
    if not args.eco_track:
        yield None
        return

    results_paths = get_results_paths(paths.project_root, args.results_stage_name)
    jsonl_path = results_paths.ecotracker_dir / "ldm_pipeline_ecotracker.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.touch(exist_ok=True)

    try:
        from eco_tracker import measure_sustainability
    except Exception as exc:
        print(f"EcoTracker non disponibile, proseguo senza misura: {exc}")
        yield None
        return

    measure_context = measure_sustainability(label=label, sample_interval=0.5)
    try:
        tracker = measure_context.__enter__()
    except Exception as exc:
        print(f"EcoTracker non avviabile, proseguo senza misura: {exc}")
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
            print(f"EcoTracker non chiuso correttamente, proseguo senza misura: {exc}")
        if tracker is not None and tracker.metrics is not None:
            payload = tracker.metrics.to_dict()
            append_jsonl(jsonl_path, payload)
            print(f"EcoTracker salvato: {jsonl_path}")
            print(tracker.metrics)


def _worker_indices(args: argparse.Namespace) -> list[int] | None:
    if args.generation_indices_file is None:
        return None
    with open(args.generation_indices_file, "r", encoding="utf-8") as file:
        payload = json.load(file)
    indices = payload.get("indices", payload)
    if not isinstance(indices, list) or any(not isinstance(value, int) for value in indices):
        raise ValueError(f"Indice worker non valido: {args.generation_indices_file}")
    return sorted(set(indices))


def generation_child_command(args: argparse.Namespace, paths, indices_file: Path | None, gpu: str, queue_dir: Path | None = None) -> list[str]:
    """Create a non-recursive, CUDA-isolated RAW-generation child command."""
    command = child_command(args, paths, "generate")
    # The generate child already received the parent's settings. A worker must be
    # non-recursive, so replace that one value while preserving max-workers.
    gpu_option = command.index("--generation-gpus")
    command[gpu_option + 1] = "off"
    command.extend(["--generation-worker", "--gpu-visible-devices", str(gpu)])
    if indices_file is not None: command.extend(["--generation-indices-file", str(indices_file)])
    if queue_dir is not None: command.extend(["--generation-queue-dir", str(queue_dir)])
    # Workers never write aggregate energy records concurrently.
    while "--eco-track" in command:
        command.remove("--eco-track")
    return command


def orchestrate_parallel_raw_generation(args: argparse.Namespace, paths, raw_dir: Path, targets: list[int]) -> dict:
    devices = resolve_generation_gpu_devices(args.generation_gpus, args.max_generation_workers)
    print_generation_diagnostics(args.generation_gpus, devices)
    log_dir = (
        paths.logs_dir / "parallel_generation" / "dry_run"
        if args.dry_run else create_parallel_run_dir(paths.logs_dir)
    )
    shard_dir = log_dir / "raw_index_shards"
    scheduler = args.generation_scheduler
    state = scan_raw_generation_state(raw_dir, args.n_raw, force_recompute=False)
    if scheduler == "dynamic_reservations":
        queue_dir = log_dir / "work_queue"
        prepare_dynamic_queue(queue_dir, targets, target_count=args.n_raw, valid_indices=state["valid_indices"],
            corrupt_indices=state["corrupt_indices"], reservation_size=args.generation_reservation_size,
            output_dir=raw_dir, metadata={"seed_strategy": "stateless_seed_per_image_v1", "parameters": {"seed": args.seed, "target_label": args.target_label}}, dry_run=args.dry_run)
        shards = [[-1] for _ in devices]
    else:
        queue_dir = None
        shards = partition_indices(targets, len(devices))
    jobs = []
    for worker_id, indices in enumerate(shards):
        if not indices:
            continue
        experiment_label = paths.experiment_dir.name
        class_label = class_name_for_label(args.target_label)
        phase = "raw_generation"
        indices_file = None if scheduler == "dynamic_reservations" else shard_dir / f"{experiment_label}_{class_label}_{phase}_worker_{worker_id}.json"
        if not args.dry_run and indices_file is not None:
            write_json(indices_file, {"indices": indices, "raw_dir": str(raw_dir)})
        jobs.append({
            "worker_id": worker_id,
            "indices_file": indices_file, "queue_dir": queue_dir,
            "label": (
                f"{experiment_label}_{class_label}_target_{args.target_label}_"
                f"{phase}_worker_{worker_id}"
            ),
        })
    print(f"RAW generation jobs: {len(targets)} indices across {len(jobs)} workers")
    start = time.perf_counter()
    run_dynamic_gpu_jobs(
        jobs=jobs,
        devices=devices,
        command_for_job=lambda job, gpu: generation_child_command(
            args, paths, job["indices_file"], gpu, job["queue_dir"]
        ),
        logs_dir=log_dir,
        dry_run=args.dry_run,
        cwd=paths.project_root,
    )
    return {
        "devices": devices,
        "worker_count": len(jobs),
        "wall_clock_seconds": time.perf_counter() - start,
        "job_partition_strategy": scheduler,
        "generation_scheduler": scheduler,
        "generation_reservation_size": args.generation_reservation_size,
    }


def write_parallel_generation_parent_artifacts(args: argparse.Namespace, paths, raw_dir: Path, filtered_dir: Path, log_dir: Path, state: dict, parallel_info: dict) -> None:
    """Only the parent writes the global state and manifest after all children finish."""
    state_path = log_dir / "generation_raw_state.json"
    write_json(state_path, {**state, "raw_dir": str(raw_dir), "n_raw": args.n_raw, "target_label": args.target_label, "seed": args.seed})
    canonical_run = is_canonical_image_run(paths, args, raw_dir, filtered_dir)
    model_path = resolve_model_path(paths.experiment_dir, args.model_path)
    payload = {
        "stage": "generate",
        "raw_dir": str(raw_dir),
        "n_raw": args.n_raw,
        "target_label": args.target_label,
        "sample_steps": args.sample_steps,
        "guidance_scale": args.guidance_scale,
        "vae_backend": args.vae_backend,
        "sd_vae_model": args.sd_vae_model,
        "model_path": str(model_path),
        "state_json": str(state_path),
        "canonical_run": canonical_run,
        "parallel_generation": True,
        "generation_gpu_devices_requested": args.generation_gpus,
        "generation_gpu_devices_resolved": parallel_info["devices"],
        "generation_worker_count": parallel_info["worker_count"],
        "job_partition_strategy": parallel_info["job_partition_strategy"],
        "generation_scheduler": parallel_info["generation_scheduler"],
        "generation_reservation_size": parallel_info["generation_reservation_size"],
        "seed_strategy": "stateless_seed_per_image_v1",
        "sampler_randomness": "stateless_seed_per_image_v1",
        "wall_clock_seconds": parallel_info["wall_clock_seconds"],
        "energy_measurement": "not_aggregated_across_parallel_workers",
        **manifest_operational_fields(args),
    }
    if canonical_run:
        write_pipeline_manifest(paths, payload, args.results_stage_name)
    else:
        write_json(log_dir / "generation_manifest_trial.json", payload)


def _run_generate_locked_body(args: argparse.Namespace, paths) -> None:
    """Esegue la generazione RAW: carica VAE decoder e U-Net LDM, compila il sampler una sola volta e genera via reverse diffusion solo le immagini mancanti o corrotte rispetto allo stato gia' su disco, salvando log/manifest a fine corsa."""
    if args.batch_size != 1:
        print(f"Nota: --batch-size={args.batch_size} ignorato; uso batch fisso 1.")
        args.batch_size = 1

    raw_dir, filtered_dir = resolve_image_dirs(paths, args)
    canonical_run = is_canonical_image_run(paths, args, raw_dir, filtered_dir)
    log_dir = (
        paths.logs_dir / class_name_for_label(args.target_label)
        if canonical_run
        else raw_dir.parent / "generation_logs_trial"
    )
    if args.generation_worker:
        # No worker writes aggregate state, manifests or shared JSONL files.
        log_dir = paths.logs_dir / "parallel_generation" / f"final_worker_gpu_{args.gpu_visible_devices or 'unknown'}"
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    # This must precede the no-target fast path: a complete directory is safe to
    # skip only when its model/decoder/sampling parameters are still compatible.
    prepare_raw_generation_manifest(
        args, paths, raw_dir, parent=not args.generation_worker, dry_run=args.dry_run
    )
    state = scan_raw_generation_state(raw_dir, args.n_raw, args.force_recompute)
    targets = sorted(set(state["missing_indices"]) | set(state["corrupt_indices"]))
    state_path = log_dir / "generation_raw_state.json"
    if not args.generation_worker and not args.dry_run:
        write_json(
            state_path,
            {
                **state,
                "raw_dir": str(raw_dir),
                "n_raw": args.n_raw,
                "target_label": args.target_label,
                "seed": args.seed,
            },
        )
    print(
        "RAW resume: "
        f"valid={len(state['valid_indices'])} "
        f"missing={len(state['missing_indices'])} "
        f"corrupt={len(state['corrupt_indices'])} "
        f"extra={len(state['extra_files'])}"
    )
    if state["corrupt_files"]:
        print("File corrotti che verranno rigenerati:")
        for path in state["corrupt_files"][:20]:
            print("  ", path)
    if not targets:
        print("Skip: immagini raw gia' complete e leggibili. Non carico TensorFlow.")
        return

    if not args.generation_worker and str(args.generation_gpus).strip().lower() not in {"off", "none", "false", ""}:
        parallel_info = orchestrate_parallel_raw_generation(args, paths, raw_dir, targets)
        if args.dry_run:
            return
        final_state = scan_raw_generation_state(raw_dir, args.n_raw, force_recompute=False)
        if final_state["missing_indices"] or final_state["corrupt_indices"]:
            raise RuntimeError(
                "Generazione RAW parallela incompleta: "
                f"missing={len(final_state['missing_indices'])}, corrupt={len(final_state['corrupt_indices'])}"
            )
        write_parallel_generation_parent_artifacts(
            args, paths, raw_dir, filtered_dir, log_dir, final_state, parallel_info
        )
        return

    assigned_indices = _worker_indices(args)
    if assigned_indices is not None:
        target_set = set(targets)
        unexpected = set(assigned_indices) - target_set
        if unexpected:
            raise RuntimeError(f"Worker ha ricevuto indici non mancanti: {sorted(unexpected)[:20]}")
        targets = assigned_indices
        if not targets:
            print("Worker senza indici assegnati: esco senza caricare TensorFlow.")
            return

    configure_environment(args)

    if args.vae_backend == "sd" and not args.decode_on_cpu:
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    import numpy as np
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
        seed=args.seed,
        allow_gpu_memory_growth=args.vae_backend == "sd" and not args.decode_on_cpu,
    )
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    model_path = resolve_model_path(paths.experiment_dir, args.model_path)
    print("MODEL_PATH:", model_path)
    print("decode_on_cpu:", args.decode_on_cpu)
    print("batch sampling effettivo: 1")
    print("target_label:", args.target_label)
    print("sample_steps:", args.sample_steps)
    print("guidance_scale:", args.guidance_scale)

    with maybe_measure(args, paths, "ldm_generate_raw"):
        vram_gb("VRAM prima del modello")
        schedule = build_schedule()
        latent_mean, latent_std = load_latent_stats(paths.latents_dir / "latent_stats.npz")
        ldm_model = load_ldm_model(model_path)
        vram_gb("VRAM dopo load LDM")

        sd_vae = sd_device = sd_dtype = None
        if args.vae_backend == "sd":
            from sd_vae_utils import (
                decode_sd_latents_to_grayscale,
                load_sd_vae,
                resolve_sd_vae_model,
            )

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
            vram_gb("VRAM dopo load VAE decoder")
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
        print(f"sampler compilato una sola volta e riusato per tutte le immagini (parameterization={args.parameterization})")

        start_time = time.perf_counter()
        worker_stats = {"worker_id": os.getpid(), "gpu_physical_id": args.gpu_visible_devices,
            "chunks_claimed": 0, "chunks_completed": 0, "indices_reserved": 0,
            "images_generated": 0, "images_already_completed": 0, "images_failed": 0}

        def iter_worker_targets():
            if args.generation_queue_dir is None:
                yield from targets
                return
            while True:
                claimed = claim_next_chunk(args.generation_queue_dir, str(os.getpid()))
                if claimed is None: break
                claimed_path, chunk = claimed; worker_stats["chunks_claimed"] += 1
                for index in chunk["indices"]:
                    output_path = expected_raw_path(raw_dir, index)
                    if is_readable_png(output_path):
                        worker_stats["images_already_completed"] += 1
                        continue
                    reservation = claim_index(args.generation_queue_dir, index, chunk["chunk_id"], str(os.getpid()), str(args.gpu_visible_devices))
                    if reservation is None:
                        if is_readable_png(output_path): worker_stats["images_already_completed"] += 1
                        continue
                    worker_stats["indices_reserved"] += 1
                    try: yield index
                    finally: release_index_claim(reservation)
                complete_claimed_chunk(args.generation_queue_dir, claimed_path, chunk)
                worker_stats["chunks_completed"] += 1

        for generated_count, index in enumerate(iter_worker_targets(), start=1):
            output_path = expected_raw_path(raw_dir, index)
            if output_path.exists() and is_readable_png(output_path) and not args.force_recompute:
                continue

            sample_seed = tf.constant([int(args.seed), int(index)], dtype=tf.int32)
            sample_out = compiled_sampler(
                tf.constant(args.target_label, dtype=tf.int32),
                sample_seed,
            )
            if args.vae_backend == "sd":
                image_np = decode_sd_latents_to_grayscale(
                    sample_out.numpy(),
                    sd_vae,
                    sd_device,
                    sd_dtype,
                    batch_size=args.sd_vae_batch_size,
                )[0]
            else:
                image_np = sample_out[0].numpy()
            save_single_image(image_np, output_path)
            worker_stats["images_generated"] += 1

            del sample_out
            del image_np
            trace_count = sampler_trace_count(compiled_sampler)
            trace_msg = f" | sampler_traces={trace_count}" if trace_count is not None else ""
            print(
                f"  raw {generated_count}/{len(targets)} "
                f"idx={index:05d} -> {output_path.name}{trace_msg}",
                flush=True,
            )
            if (
                generated_count == len(targets)
                or (args.vram_log_every > 0 and generated_count % args.vram_log_every == 0)
            ):
                gc.collect()
                vram_gb(f"VRAM dopo {generated_count} nuove RAW")

        elapsed = time.perf_counter() - start_time
        worker_stats["wall_clock_seconds"] = elapsed
        worker_stats["images_per_second"] = worker_stats["images_generated"] / elapsed if elapsed else 0.0
        if args.generation_queue_dir is not None:
            write_json(args.generation_queue_dir / "worker_stats" / f"worker_{os.getpid()}.json", worker_stats)
        if not args.generation_worker:
            final_state = scan_raw_generation_state(raw_dir, args.n_raw, force_recompute=False)
            write_json(
                state_path,
                {
                    **final_state,
                    "raw_dir": str(raw_dir),
                    "n_raw": args.n_raw,
                    "target_label": args.target_label,
                    "seed": args.seed,
                    "elapsed_seconds": elapsed,
                },
            )
            if final_state["missing_indices"] or final_state["corrupt_indices"]:
                raise RuntimeError(
                    "Generazione RAW incompleta: "
                    f"missing={len(final_state['missing_indices'])}, "
                    f"corrupt={len(final_state['corrupt_indices'])}"
                )
        else:
            print(f"Worker completed {len(targets)} assigned RAW indices in {elapsed:.1f}s")
        print("sampler_traces_finale:", sampler_trace_count(compiled_sampler))
        append_jsonl(
            log_dir / "generation_summary.jsonl",
            {
                "phase": "ldm_generation_raw",
                "raw_dir": str(raw_dir),
                "n_raw": args.n_raw,
                "target_label": args.target_label,
                "sample_steps": args.sample_steps,
                "guidance_scale": args.guidance_scale,
                "decode_on_cpu": args.decode_on_cpu,
                "vae_backend": args.vae_backend,
                "sd_vae_model": args.sd_vae_model,
                "seed": args.seed,
                "sampler_randomness": "stateless_seed_per_image_v1",
                "elapsed_seconds": elapsed,
                **manifest_operational_fields(args),
            },
        )
        manifest_payload = {
            "stage": "generate",
            "raw_dir": str(raw_dir),
            "n_raw": args.n_raw,
            "target_label": args.target_label,
            "sample_steps": args.sample_steps,
            "guidance_scale": args.guidance_scale,
            "vae_backend": args.vae_backend,
            "sd_vae_model": args.sd_vae_model,
            "model_path": str(model_path),
            "state_json": str(state_path),
            "sampler_randomness": "stateless_seed_per_image_v1",
            "canonical_run": canonical_run,
            **manifest_operational_fields(args),
        }
        if canonical_run and not args.generation_worker:
            write_pipeline_manifest(paths, manifest_payload, args.results_stage_name)
        else:
            write_json(log_dir / "generation_manifest_trial.json", manifest_payload)

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
        vram_gb("VRAM alla fine della generazione")


def run_generate(args: argparse.Namespace, paths) -> None:
    """Acquire the output lock before scanning, cleanup or queue construction."""
    if args.dry_run or args.generation_worker:
        return _run_generate_locked_body(args, paths)
    raw_dir, _ = resolve_image_dirs(paths, args)
    lock = acquire_parallel_generation_lock(raw_dir)
    try:
        return _run_generate_locked_body(args, paths)
    finally:
        release_parallel_generation_lock(lock)


def train_reference_paths(paths, target_label: int) -> list[Path]:
    """Recupera dal train.csv i path normalizzati delle immagini reali della classe target, usate come riferimento dal filtro adattivo."""
    import pandas as pd

    train_df = pd.read_csv(paths.metadata_dir / "train.csv")
    label_df = train_df[train_df["label"].astype(int) == int(target_label)]
    if label_df.empty:
        raise RuntimeError(f"Nessuna immagine train per label {target_label}")
    return [
        normalize_processed_path(row, paths.data_processed_dir)
        for _, row in label_df.iterrows()
    ]


def copy_if_exists(source: Path, destination: Path) -> None:
    """Copia il file sorgente verso la destinazione solo se esiste, senza generare errore altrimenti (usato per duplicare gli output nei percorsi canonici)."""
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def run_filter(args: argparse.Namespace, paths) -> None:
    """Applica il filtro adattivo alle immagini RAW per selezionare le n_selected migliori (confrontate con le reali della classe target) e salva il relativo report/manifest. Genera prima le RAW mancanti se la cartella non ne contiene ancora a sufficienza."""
    from adaptive_mammography_filter import filter_generated_directory

    generation_args = copy.copy(args)
    generation_args.force_recompute = False
    run_generate(generation_args, paths)

    raw_dir, filtered_dir = resolve_image_dirs(paths, args)
    results_paths = get_results_paths(paths.project_root, args.results_stage_name)
    canonical_run = is_canonical_image_run(paths, args, raw_dir, filtered_dir)
    class_name = class_name_for_label(args.target_label)
    output_dir = (
        get_class_metrics_dir(results_paths, args.target_label)
        if canonical_run
        else filtered_dir.parent / "filter_outputs_trial"
    )
    plots_dir = (
        results_paths.plots_dir / class_name
        if canonical_run
        else filtered_dir.parent / "filter_plots_trial"
    )
    with maybe_measure(args, paths, "ldm_filter_raw"):
        summary = filter_generated_directory(
            raw_dir=raw_dir,
            filtered_dir=filtered_dir,
            reference_paths=train_reference_paths(paths, args.target_label),
            n_raw=args.n_raw,
            n_selected=args.n_selected,
            target_label=args.target_label,
            nonblack_threshold=args.nonblack_threshold,
            output_dir=output_dir,
            plots_dir=plots_dir,
            force_recompute=args.force_recompute,
            verbose=True,
        )

    report_path = output_dir / "synthetic_filter_report.csv"
    summary_path = output_dir / "synthetic_filter_summary.json"
    manifest_payload = {
        "stage": "filter",
        "raw_dir": str(raw_dir),
        "filtered_dir": str(filtered_dir),
        "n_raw": args.n_raw,
        "n_selected": args.n_selected,
        "target_label": args.target_label,
        "summary_json": str(summary_path),
        "report_csv": str(report_path),
        "canonical_run": canonical_run,
        "cached": bool(summary.get("cached", False)),
        **manifest_operational_fields(args),
    }
    if canonical_run:
        mirror_positive_legacy_outputs(output_dir, results_paths.metrics_dir, args.target_label)
        mirror_positive_legacy_outputs(plots_dir, results_paths.plots_dir, args.target_label)
        write_pipeline_manifest(paths, manifest_payload, args.results_stage_name)
    else:
        write_json(output_dir / "filter_manifest_trial.json", manifest_payload)


def readable_png_paths(directory: Path, limit: int | None = None) -> list[Path]:
    """Elenca in ordine i PNG leggibili in una cartella, scartando quelli corrotti, con limite opzionale."""
    paths = [path for path in sorted(directory.glob("*.png")) if not path.name.startswith(".tmp_") and is_readable_png(path)]
    return paths[:limit] if limit is not None else paths


def select_metric_paths(paths: list[Path], limit: int | None) -> list[Path]:
    """Applica il limite (max_eval_images) alla lista di path da passare al calcolo delle metriche."""
    return paths[:limit] if limit is not None else paths


def file_signature(paths: list[Path]) -> list[dict]:
    """Build a content digest used only to invalidate a stale metrics cache."""
    return [file_content_signature(path) for path in paths]


def use_metrics_cache(json_path: Path, csv_path: Path, config: dict, input_signature: dict) -> bool:
    """Reuse metrics only when configuration, synthetic inputs and validation references match."""
    if not json_path.exists():
        return False
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception as exc:
        print(f"Cache metriche validation non leggibile: {exc}")
        return False
    if payload.get("schema_version") != 2:
        return False
    if payload.get("config") != config or payload.get("input_signature") != input_signature:
        return False
    if not csv_path.exists():
        import pandas as pd

        pd.DataFrame(payload["rows"]).to_csv(csv_path, index=False)
    print(f"Metriche validation gia' presenti: {json_path}")
    return True


def run_validate(args: argparse.Namespace, paths) -> None:
    """Confronta su tre dataset (RAW completo, RAW bilanciato allo stesso numero di immagini, FILTERED) le metriche generative rispetto al validation set reale, per quantificare il guadagno introdotto dal filtro adattivo; usa la cache se input e config non sono cambiati."""
    import pandas as pd

    from ldm_evaluation_utils import evaluate_generated_paths_against_metadata

    raw_dir, filtered_dir = resolve_image_dirs(paths, args)
    results_paths = get_results_paths(paths.project_root, args.results_stage_name)
    canonical_run = is_canonical_image_run(paths, args, raw_dir, filtered_dir)
    output_dir = (
        get_class_evaluation_dir(paths, args.target_label)
        if canonical_run
        else filtered_dir.parent / "validation_outputs_trial"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    from parallel_generation_utils import exact_filtered_png_paths, ldm_raw_png_paths
    raw_paths = ldm_raw_png_paths(raw_dir)
    filtered_paths = exact_filtered_png_paths(filtered_dir, args.n_selected)
    expected_raw = [raw_dir / f"synth_{index:05d}.png" for index in range(args.n_raw)]
    if raw_paths != expected_raw:
        raise RuntimeError(f"RAW complete insufficienti: {len(raw_paths)}/{args.n_raw}")

    raw_complete = raw_paths[:args.n_raw]
    rng = random.Random(args.balanced_seed)
    raw_balanced = rng.sample(raw_complete, min(args.n_selected, len(raw_complete)))
    filtered_selected = filtered_paths[:args.n_selected]

    datasets = {
        "raw_complete": select_metric_paths(raw_complete, args.max_eval_images),
        f"raw_balanced_seed{args.balanced_seed}": select_metric_paths(raw_balanced, args.max_eval_images),
        "filtered": select_metric_paths(filtered_selected, args.max_eval_images),
    }
    config = {
        "metric_backend": "generative_evaluator.py",
        "reference_split": "val",
        "target_label": int(args.target_label),
        "n_raw": int(args.n_raw),
        "n_selected": int(args.n_selected),
        "balanced_seed": int(args.balanced_seed),
        "max_eval_images": args.max_eval_images,
        "inception_batch": int(args.inception_batch),
        "is_splits": int(args.is_splits),
        "knn_k": int(args.knn_k),
        "inception_weights": args.inception_weights,
    }
    val_csv = paths.metadata_dir / "val.csv"
    val_df = pd.read_csv(val_csv)
    val_label_df = val_df[val_df["label"].astype(int) == int(args.target_label)]
    val_reference_paths = [normalize_processed_path(row, paths.data_processed_dir) for _, row in val_label_df.iterrows()]
    input_signature = {
        name: file_signature(dataset_paths)
        for name, dataset_paths in datasets.items()
    }
    input_signature.update({
        "validation_csv_signature": checkpoint_content_signature(val_csv),
        "validation_reference_image_signature": [checkpoint_content_signature(path) for path in val_reference_paths],
        "metric_backend": "generative_evaluator.py",
        "metric_backend_version": "content_signature_v1",
    })
    csv_path = output_dir / "raw_vs_filtered_validation.csv"
    json_path = output_dir / "raw_vs_filtered_validation.json"
    class_metrics_dir = get_class_metrics_dir(results_paths, args.target_label)
    canonical_csv_path = class_metrics_dir / "raw_vs_filtered_validation.csv"
    canonical_json_path = class_metrics_dir / "raw_vs_filtered_validation.json"
    if canonical_run:
        # CSV writes do not create parent directories (unlike write_json/copy_if_exists).
        # A fresh validation run must therefore prepare the class-scoped metrics dir.
        class_metrics_dir.mkdir(parents=True, exist_ok=True)

    if not args.force_recompute and use_metrics_cache(json_path, csv_path, config, input_signature):
        if canonical_run:
            copy_if_exists(csv_path, canonical_csv_path)
            copy_if_exists(json_path, canonical_json_path)
            mirror_positive_legacy_outputs(class_metrics_dir, results_paths.metrics_dir, args.target_label)
            mirror_positive_legacy_outputs(output_dir, paths.evaluation_dir, args.target_label)
        return

    rows = []
    with maybe_measure(args, paths, "ldm_validate_raw_filtered"):
        for dataset_name, generated_paths in datasets.items():
            print(f"[VALIDATE] {dataset_name}: {len(generated_paths)} immagini")
            evaluator_metrics = evaluate_generated_paths_against_metadata(
                generated_paths=generated_paths,
                metadata_df=val_df,
                data_processed_dir=paths.data_processed_dir,
                label=args.target_label,
                batch_size=args.inception_batch,
                nearest_k=args.knn_k,
                is_splits=args.is_splits,
            )
            rows.append({
                "dataset": dataset_name,
                "reference_split": "val",
                "target_label": int(args.target_label),
                "n_real_reference": int(evaluator_metrics["n_real_reference"]),
                "n_generated": int(evaluator_metrics["n_generated"]),
                "FID": float(evaluator_metrics["FID"]),
                "IS_mean": float(evaluator_metrics["IS_mean"]),
                "IS_std": float(evaluator_metrics["IS_std"]),
                "precision": float(evaluator_metrics["precision"]),
                "recall": float(evaluator_metrics["recall"]),
                "density": float(evaluator_metrics["density"]),
                "coverage": float(evaluator_metrics["coverage"]),
            })

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    payload = {
        "schema_version": 2,
        "config": config,
        "input_signature": input_signature,
        "rows": rows,
    }
    write_json(json_path, payload)
    manifest_payload = {
        "stage": "validate",
        "target_label": int(args.target_label),
        "csv": str(csv_path),
        "json": str(json_path),
        "canonical_run": canonical_run,
        **manifest_operational_fields(args),
    }
    if canonical_run:
        pd.DataFrame(rows).to_csv(canonical_csv_path, index=False)
        write_json(canonical_json_path, payload)
        mirror_positive_legacy_outputs(class_metrics_dir, results_paths.metrics_dir, args.target_label)
        mirror_positive_legacy_outputs(output_dir, paths.evaluation_dir, args.target_label)
        manifest_payload.update({
            "canonical_csv": str(canonical_csv_path),
            "canonical_json": str(canonical_json_path),
        })
        write_pipeline_manifest(paths, manifest_payload, args.results_stage_name)
    else:
        write_json(output_dir / "validation_manifest_trial.json", manifest_payload)
    print("Salvato:", csv_path)
    print("Salvato:", json_path)


def child_command(args: argparse.Namespace, paths, mode: str) -> list[str]:
    """Costruisce la riga di comando per rilanciare questo stesso script in un sottoprocesso isolato con una singola modalita', usata dall'orchestrazione 'all'/'both'."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode", mode,
        "--project-root", str(paths.project_root),
        "--experiment-dir", str(paths.experiment_dir),
        "--cuda-root", str(args.cuda_root),
        "--target-label", str(args.target_label),
        "--n-raw", str(args.n_raw),
        "--n-selected", str(args.n_selected),
        "--batch-size", "1",
        "--sample-steps", str(args.sample_steps),
        "--guidance-scale", str(args.guidance_scale),
        "--nonblack-threshold", str(args.nonblack_threshold),
        "--vram-log-every", str(args.vram_log_every),
        "--seed", str(args.seed),
        "--balanced-seed", str(args.balanced_seed),
        "--inception-batch", str(args.inception_batch),
        "--is-splits", str(args.is_splits),
        "--knn-k", str(args.knn_k),
        "--inception-weights", args.inception_weights,
        "--results-stage-name", args.results_stage_name,
    ]
    if mode == "generate":
        command.extend(["--generation-gpus", str(args.generation_gpus),
                        "--generation-scheduler", str(args.generation_scheduler),
                        "--generation-reservation-size", str(args.generation_reservation_size)])
        if args.max_generation_workers is not None:
            command.extend(["--max-generation-workers", str(args.max_generation_workers)])
    if args.model_path is not None:
        command.extend(["--model-path", str(args.model_path)])
    if args.gpu_visible_devices is not None:
        command.extend(["--gpu-visible-devices", args.gpu_visible_devices])
    if args.raw_dir is not None:
        command.extend(["--raw-dir", str(args.raw_dir)])
    if args.filtered_dir is not None:
        command.extend(["--filtered-dir", str(args.filtered_dir)])
    if args.max_eval_images is not None:
        command.extend(["--max-eval-images", str(args.max_eval_images)])
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
    if args.eco_track:
        command.append("--eco-track")
    command.extend(["--parameterization", args.parameterization])
    command.extend(["--unet-version", args.unet_version])
    command.extend(["--vae-source", args.vae_source])
    if args.uses_vae_ft_from_03:
        command.append("--uses-vae-ft-from-03")
    if args.notebook_name is not None:
        command.extend(["--notebook-name", str(args.notebook_name)])
    return command


def orchestrate_modes(args: argparse.Namespace, paths, modes: list[str]) -> None:
    """Esegue in sequenza le modalita' richieste, ognuna in un sottoprocesso separato, cosi' che la memoria GPU venga liberata completamente tra una fase e l'altra; interrompe tutto al primo fallimento."""
    print("Orchestrazione in subprocess separati:", ", ".join(modes))
    for mode in modes:
        command = child_command(args, paths, mode)
        print("\n[orchestrator]", " ".join(command))
        completed = subprocess.run(command, cwd=str(paths.project_root), check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


def run_reverse(args: argparse.Namespace, paths) -> None:
    """Genera per ciascuna label richiesta una figura con gli step intermedi della reverse diffusion (rumore -> immagine), decodificando con il VAE solo i timestep scelti per il plot e salvando il risultato in plots_dir."""
    configure_environment(args)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf

    from ldm_keras_utils import (
        CLASS_NAMES,
        LATENT_CHANNELS,
        LATENT_SIZE,
        NUM_DIFF_STEPS,
        build_schedule,
        configure_tensorflow,
        load_latent_stats,
        load_ldm_model,
        load_vae_decoder,
        p_sample_ldm,
    )

    configure_tensorflow(seed=args.seed)
    tf.random.set_seed(args.seed)
    results_paths = get_results_paths(paths.project_root, args.results_stage_name)
    schedule = build_schedule()
    latent_mean, latent_std = load_latent_stats(paths.latents_dir / "latent_stats.npz")
    sd_vae = sd_device = sd_dtype = None
    if args.vae_backend == "sd":
        from sd_vae_utils import decode_sd_latents_to_grayscale, load_sd_vae, resolve_sd_vae_model

        sd_model = resolve_sd_vae_model(paths.project_root, args.sd_vae_model)
        sd_device = "cpu" if args.decode_on_cpu else None
        sd_vae, sd_device, sd_dtype, _ = load_sd_vae(sd_model, device=sd_device)
        vae_decoder = None
        print("SD-VAE reverse decoder:", sd_model)
        print("SD-VAE reverse device:", sd_device)
    else:
        vae_decoder = load_vae_decoder(paths.models_dir / "vae_decoder_best.keras")
    model_path = resolve_model_path(paths.experiment_dir, args.model_path)
    ldm_model = load_ldm_model(model_path)

    labels = [int(label.strip()) for label in args.reverse_labels.split(",") if label.strip()]
    for label in labels:
        z = tf.random.normal((1, LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS))
        y = tf.fill([1], int(label))
        stride = max(1, NUM_DIFF_STEPS // int(args.sample_steps))
        timesteps = list(range(0, NUM_DIFF_STEPS, stride))[::-1]
        save_at = set(
            np.round(np.linspace(0, len(timesteps) - 1, args.reverse_steps_to_show)).astype(int)
        )
        saved, ts = [], []
        for index, t in enumerate(timesteps):
            t_prev = timesteps[index + 1] if index + 1 < len(timesteps) else 0
            z = p_sample_ldm(
                ldm_model,
                schedule,
                z,
                int(t),
                int(t_prev),
                y,
                guidance_scale=args.guidance_scale,
                parameterization=args.parameterization,
            )
            if index in save_at:
                z_denorm = z * latent_std + latent_mean
                if args.vae_backend == "sd":
                    img = decode_sd_latents_to_grayscale(
                        z_denorm.numpy(),
                        sd_vae,
                        sd_device,
                        sd_dtype,
                        batch_size=args.sd_vae_batch_size,
                    )[0]
                else:
                    img = vae_decoder(z_denorm, training=False)
                    img = np.clip(((img[0] + 1.0) / 2.0).numpy(), 0.0, 1.0)
                saved.append(img)
                ts.append(t)

        fig = plt.figure(figsize=(20, 3))
        for index, (img, t) in enumerate(zip(saved, ts)):
            ax = fig.add_subplot(1, len(saved), index + 1)
            ax.imshow(img.squeeze(), cmap="gray")
            ax.set_title(f"t={t}", fontsize=7)
            ax.axis("off")
        plt.suptitle(
            f"Reverse Diffusion LDM - {CLASS_NAMES[label]} | CFG={args.guidance_scale}",
            fontsize=11,
        )
        plt.tight_layout()
        out_path = results_paths.plots_dir / f"reverse_ldm_{label}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Salvato:", out_path)


def main() -> None:
    """Punto di ingresso CLI: parsa gli argomenti e smista l'esecuzione verso la modalita' richiesta (singola fase o orchestrazione di piu' fasi)."""
    args = parse_args()
    if (
        not args.generation_worker
        and str(args.generation_gpus).strip().lower() in {"off", "none", "false", ""}
        and args.gpu_visible_devices is None
    ):
        # The non-orchestrated compatibility path still uses exactly one GPU.
        args.gpu_visible_devices = resolve_generation_gpu_devices("off", 1)[0]
    configure_environment(args)
    paths = get_experiment_paths(args.project_root, args.experiment_dir, create=not args.dry_run)

    if args.dry_run:
        if args.mode != "generate":
            print("--dry-run mostra soltanto il piano della generazione RAW; filter/validate/reverse non vengono avviati.")
        run_generate(args, paths)
        return

    if args.mode == "all":
        orchestrate_modes(args, paths, ["generate", "filter", "validate"])
        return
    if args.mode == "both":
        orchestrate_modes(args, paths, ["generate", "filter", "reverse"])
        return
    if args.mode == "generate":
        run_generate(args, paths)
        return
    if args.mode == "filter":
        run_filter(args, paths)
        return
    if args.mode == "validate":
        run_validate(args, paths)
        return
    if args.mode == "reverse":
        run_reverse(args, paths)
        return
    raise ValueError(f"Mode non gestito: {args.mode}")


if __name__ == "__main__":
    main()
