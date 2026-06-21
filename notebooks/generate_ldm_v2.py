#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path


from ldm_project_paths import (
    get_experiment_paths,
    get_results_paths,
    normalize_processed_path,
)


def parse_args() -> argparse.Namespace:
    """Definisce e legge gli argomenti CLI condivisi da tutte le modalita' dello script (generate/filter/validate/test/all/reverse/both)."""
    parser = argparse.ArgumentParser(
        description="Generate, filter and evaluate MammoDiffusion LDM final images."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--gpu-visible-devices", default=None)
    parser.add_argument(
        "--cuda-root",
        type=Path,
        default=Path("/home/fede/miniforge3/envs/tf-gpu"),
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["generate", "filter", "validate", "test", "all", "reverse", "both"],
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
    parser.add_argument("--vram-log-every", type=int, default=25)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-seed", type=int, default=42)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--filtered-dir", type=Path, default=None)
    parser.add_argument(
        "--results-stage-name",
        default="04_ldm_keras_v2",
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
        help="Limite leggero per smoke test delle metriche; default usa tutti gli input.",
    )
    parser.add_argument("--reverse-labels", default="1,0")
    parser.add_argument("--reverse-steps-to-show", type=int, default=10)
    parser.add_argument("--eco-track", action="store_true")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    """Imposta le variabili d'ambiente necessarie prima di importare TensorFlow (cartella matplotlib, GPU visibili, percorso libdevice per XLA)."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
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
    """Determina le cartelle raw e filtered da usare (override da CLI per smoke test, altrimenti i percorsi canonici dell'esperimento) e le crea se mancanti."""
    raw_dir = (
        args.raw_dir.expanduser().resolve()
        if args.raw_dir is not None
        else paths.synthetic_raw_dir
    )
    filtered_dir = (
        args.filtered_dir.expanduser().resolve()
        if args.filtered_dir is not None
        else paths.synthetic_filtered_dir
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, filtered_dir


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
                if raw_index_from_name(path) is None or raw_index_from_name(path) >= n_raw
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
    tmp_path = output_path.with_name(f".tmp_{output_path.name}")
    Image.fromarray(
        np.clip(arr * 255.0, 0, 255).astype(np.uint8),
        mode="L",
    ).save(tmp_path)
    tmp_path.replace(output_path)


def append_jsonl(path: Path, payload: dict) -> None:
    """Accoda una riga JSON al file di log (jsonl), forzando flush e fsync per non perdere righe se il processo viene interrotto bruscamente."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def write_pipeline_manifest(paths, payload: dict, results_stage_name: str) -> None:
    """Aggiorna il manifest cumulativo della pipeline (uno per stage: generate/filter/validate/test) sia nella cartella dell'esperimento che in quella dei risultati, senza sovrascrivere gli altri stage gia' registrati."""
    results_paths = get_results_paths(paths.project_root, results_stage_name)
    manifest_paths = [
        paths.evaluation_dir / "ldm_pipeline_manifest.json",
        results_paths.metrics_dir / "ldm_pipeline_manifest.json",
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


def run_generate(args: argparse.Namespace, paths) -> None:
    """Esegue la generazione RAW: carica VAE decoder e U-Net LDM, compila il sampler una sola volta e genera via reverse diffusion solo le immagini mancanti o corrotte rispetto allo stato gia' su disco, salvando log/manifest a fine corsa."""
    if args.batch_size != 1:
        print(f"Nota: --batch-size={args.batch_size} ignorato; uso batch fisso 1.")
        args.batch_size = 1

    raw_dir, _ = resolve_image_dirs(paths, args)
    canonical_run = raw_dir == paths.synthetic_raw_dir
    log_dir = paths.logs_dir if canonical_run else raw_dir.parent / "generation_logs_smoke"
    log_dir.mkdir(parents=True, exist_ok=True)
    state = scan_raw_generation_state(raw_dir, args.n_raw, args.force_recompute)
    targets = sorted(set(state["missing_indices"]) | set(state["corrupt_indices"]))
    state_path = log_dir / "generation_raw_state.json"
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

    configure_environment(args)

    import numpy as np
    import tensorflow as tf

    from ldm_keras_utils import (
        build_schedule,
        configure_tensorflow,
        load_latent_stats,
        load_ldm_model,
        load_vae_decoder,
        make_compiled_sampler,
        vram_gb,
    )

    configure_tensorflow(seed=args.seed)
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
        vae_decoder = load_vae_decoder(paths.models_dir / "vae_decoder_best.keras")
        vram_gb("VRAM dopo load VAE decoder")
        ldm_model = load_ldm_model(model_path)
        vram_gb("VRAM dopo load LDM")

        compiled_sampler = make_compiled_sampler(
            ldm_model=ldm_model,
            vae_decoder=vae_decoder,
            schedule=schedule,
            latent_mean=latent_mean,
            latent_std=latent_std,
            num_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
            decode_on_cpu=args.decode_on_cpu,
        )
        print("sampler compilato una sola volta e riusato per tutte le immagini")

        start_time = time.perf_counter()
        for generated_count, index in enumerate(targets, start=1):
            output_path = expected_raw_path(raw_dir, index)
            if output_path.exists() and is_readable_png(output_path) and not args.force_recompute:
                continue

            sample_seed = tf.constant([int(args.seed), int(index)], dtype=tf.int32)
            image_tensor = compiled_sampler(
                tf.constant(args.target_label, dtype=tf.int32),
                sample_seed,
            )
            image_np = image_tensor[0].numpy()
            save_single_image(image_np, output_path)

            del image_tensor
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
                "seed": args.seed,
                "sampler_randomness": "stateless_seed_per_image_v1",
                "elapsed_seconds": elapsed,
            },
        )
        manifest_payload = {
            "stage": "generate",
            "raw_dir": str(raw_dir),
            "n_raw": args.n_raw,
            "target_label": args.target_label,
            "sample_steps": args.sample_steps,
            "guidance_scale": args.guidance_scale,
            "model_path": str(model_path),
            "state_json": str(state_path),
            "sampler_randomness": "stateless_seed_per_image_v1",
            "canonical_run": canonical_run,
        }
        if canonical_run:
            write_pipeline_manifest(paths, manifest_payload, args.results_stage_name)
        else:
            write_json(log_dir / "generation_manifest_smoke.json", manifest_payload)

        del compiled_sampler
        del ldm_model
        del vae_decoder
        del schedule
        del latent_mean
        del latent_std
        gc.collect()
        tf.keras.backend.clear_session()
        vram_gb("VRAM alla fine della generazione")


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
    """Applica il filtro adattivo alle immagini RAW per selezionare le n_selected migliori (confrontate con le reali della classe target) e salva il relativo report/manifest."""
    from adaptive_mammography_filter import filter_generated_directory

    raw_dir, filtered_dir = resolve_image_dirs(paths, args)
    results_paths = get_results_paths(paths.project_root, args.results_stage_name)
    canonical_run = raw_dir == paths.synthetic_raw_dir and filtered_dir == paths.synthetic_filtered_dir
    output_dir = results_paths.metrics_dir if canonical_run else filtered_dir.parent / "filter_outputs_smoke"
    plots_dir = results_paths.plots_dir if canonical_run else filtered_dir.parent / "filter_plots_smoke"
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
    }
    if canonical_run:
        write_pipeline_manifest(paths, manifest_payload, args.results_stage_name)
    else:
        write_json(output_dir / "filter_manifest_smoke.json", manifest_payload)


def readable_png_paths(directory: Path, limit: int | None = None) -> list[Path]:
    """Elenca in ordine i PNG leggibili in una cartella, scartando quelli corrotti, con limite opzionale."""
    paths = [path for path in sorted(directory.glob("*.png")) if is_readable_png(path)]
    return paths[:limit] if limit is not None else paths


def select_metric_paths(paths: list[Path], limit: int | None) -> list[Path]:
    """Applica il limite di smoke test (max_eval_images) alla lista di path da passare al calcolo delle metriche."""
    return paths[:limit] if limit is not None else paths


def file_signature(paths: list[Path]) -> list[dict]:
    """Costruisce una firma leggera (nome, dimensione, mtime) dei file di input, usata per invalidare la cache delle metriche se i file cambiano."""
    return [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]


def use_metrics_cache(json_path: Path, csv_path: Path, config: dict, input_signature: dict) -> bool:
    """Riusa le metriche già calcolate se config e input (file sintetici + test.csv) non sono cambiati, evitando di ripetere un calcolo costoso (FID/IS su GPU) senza motivo."""
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
    canonical_run = raw_dir == paths.synthetic_raw_dir and filtered_dir == paths.synthetic_filtered_dir
    output_dir = paths.evaluation_dir if canonical_run else filtered_dir.parent / "validation_outputs_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = readable_png_paths(raw_dir)
    filtered_paths = readable_png_paths(filtered_dir)
    if len(raw_paths) < args.n_raw:
        raise RuntimeError(f"RAW complete insufficienti: {len(raw_paths)}/{args.n_raw}")
    if len(filtered_paths) < args.n_selected:
        raise RuntimeError(f"FILTERED insufficienti: {len(filtered_paths)}/{args.n_selected}")

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
    input_signature = {
        name: file_signature(dataset_paths)
        for name, dataset_paths in datasets.items()
    }
    csv_path = output_dir / "raw_vs_filtered_validation.csv"
    json_path = output_dir / "raw_vs_filtered_validation.json"
    canonical_csv_path = results_paths.metrics_dir / "raw_vs_filtered_validation.csv"
    canonical_json_path = results_paths.metrics_dir / "raw_vs_filtered_validation.json"

    if not args.force_recompute and use_metrics_cache(json_path, csv_path, config, input_signature):
        if canonical_run:
            copy_if_exists(csv_path, canonical_csv_path)
            copy_if_exists(json_path, canonical_json_path)
        return

    val_df = pd.read_csv(paths.metadata_dir / "val.csv")
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
        "csv": str(csv_path),
        "json": str(json_path),
        "canonical_run": canonical_run,
    }
    if canonical_run:
        pd.DataFrame(rows).to_csv(canonical_csv_path, index=False)
        write_json(canonical_json_path, payload)
        manifest_payload.update({
            "canonical_csv": str(canonical_csv_path),
            "canonical_json": str(canonical_json_path),
        })
        write_pipeline_manifest(paths, manifest_payload, args.results_stage_name)
    else:
        write_json(output_dir / "validation_manifest_smoke.json", manifest_payload)
    print("Salvato:", csv_path)
    print("Salvato:", json_path)


def evaluate_filtered_command(args: argparse.Namespace, paths, filtered_dir: Path) -> list[str]:
    """Costruisce la riga di comando per lanciare in sottoprocesso evaluate_filtered_ldm_v2.py (confronto FILTERED vs test set), propagando gli argomenti coerenti con questa run."""
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "evaluate_filtered_ldm_v2.py"),
        "--project-root", str(paths.project_root),
        "--experiment-dir", str(paths.experiment_dir),
        "--cuda-root", str(args.cuda_root),
        "--synthetic-dir", str(filtered_dir),
        "--target-label", str(args.target_label),
        "--inception-batch", str(args.inception_batch),
        "--is-splits", str(args.is_splits),
        "--knn-k", str(args.knn_k),
        "--inception-weights", args.inception_weights,
        "--results-stage-name", args.results_stage_name,
    ]
    if args.gpu_visible_devices is not None:
        command.extend(["--gpu-visible-devices", args.gpu_visible_devices])
    if args.max_eval_images is not None:
        command.extend(["--max-synthetic", str(args.max_eval_images)])
    if args.force_recompute:
        command.append("--force-recompute")
    if filtered_dir != paths.synthetic_filtered_dir:
        command.extend(["--output-dir", str(filtered_dir.parent / "test_outputs_smoke")])
        command.append("--no-canonical-results")
    return command


def run_test(args: argparse.Namespace, paths) -> None:
    """Lancia in sottoprocesso la valutazione finale FILTERED vs test set e registra l'esito nel manifest della pipeline, propagando l'eventuale errore del sottoprocesso."""
    _, filtered_dir = resolve_image_dirs(paths, args)
    canonical_run = filtered_dir == paths.synthetic_filtered_dir
    command = evaluate_filtered_command(args, paths, filtered_dir)
    print("Comando test finale:")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=str(paths.project_root), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    manifest_payload = {
        "stage": "test",
        "filtered_dir": str(filtered_dir),
        "csv": str(
            (paths.evaluation_dir if canonical_run else filtered_dir.parent / "test_outputs_smoke")
            / "final_filtered_vs_test.csv"
        ),
        "json": str(
            (paths.evaluation_dir if canonical_run else filtered_dir.parent / "test_outputs_smoke")
            / "final_filtered_vs_test.json"
        ),
        "canonical_run": canonical_run,
    }
    if canonical_run:
        write_pipeline_manifest(paths, manifest_payload, args.results_stage_name)
    else:
        write_json(filtered_dir.parent / "test_manifest_smoke.json", manifest_payload)


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
    if args.force_recompute:
        command.append("--force-recompute")
    if args.eco_track:
        command.append("--eco-track")
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
            )
            if index in save_at:
                z_denorm = z * latent_std + latent_mean
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
    configure_environment(args)
    paths = get_experiment_paths(args.project_root, args.experiment_dir)

    if args.mode == "all":
        orchestrate_modes(args, paths, ["generate", "filter", "validate", "test"])
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
    if args.mode == "test":
        run_test(args, paths)
        return
    if args.mode == "reverse":
        run_reverse(args, paths)
        return
    raise ValueError(f"Mode non gestito: {args.mode}")


if __name__ == "__main__":
    main()
