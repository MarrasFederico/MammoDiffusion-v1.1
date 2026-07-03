#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from ldm_project_paths import (
    DEFAULT_EXPERIMENT_NAME,
    PROJECT_NAME,
    ExperimentPaths,
    get_experiment_paths,
    get_results_paths,
)

POSITIVE_CLASS = 1
BEST_SELECTION_METRIC = f"fid_{POSITIVE_CLASS}"
BEST_SELECTION_TIE_BREAKER = f"is_mean_{POSITIVE_CLASS}"
SELECTION_DIRECTION = "minimize"
TIE_BREAKER_DIRECTION = "maximize"
SELECTION_REASON = "Best generation quality for the positive mammography class"
RESULTS_STAGE_NAME = "04_ldm_keras_v2"
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
    """Definisce e legge gli argomenti CLI dello sweep (checkpoint, sampling, metriche, eco-tracking)."""
    parser = argparse.ArgumentParser(
        description="Generate and evaluate MammoDiffusion LDM checkpoint sweeps."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--gpu-visible-devices", default=None)
    parser.add_argument(
        "--cuda-root",
        type=Path,
        default=Path("/home/fede/miniforge3/envs/tf-gpu"),
    )
    parser.add_argument("--min-step", type=int, default=14_000)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--checkpoint-id", default=None)
    parser.add_argument(
        "--mode",
        choices=["both", "generate", "metrics", "artifacts"],
        default="both",
        help=(
            "both orchestra generate+metrics in processi separati; generate genera "
            "solo immagini; metrics calcola solo metriche; artifacts rigenera solo "
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
        help="Mantenuto per compatibilita'; il sampling sweep forza batch 1.",
    )
    parser.add_argument("--classes", default="0,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-on-cpu", action="store_true")
    parser.add_argument("--vram-log-every", type=int, default=10)
    parser.add_argument("--inception-batch", type=int, default=8)
    parser.add_argument("--is-splits", type=int, default=10)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument(
        "--results-stage-name",
        default=RESULTS_STAGE_NAME,
        help="Sottocartella di results dove salvare metriche, plot e log EcoTracker.",
    )
    parser.add_argument(
        "--inception-weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Use 'none' only for smoke tests; metrics are not meaningful.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignora checkpoint_metrics.json e ricalcola lo sweep.",
    )
    parser.add_argument(
        "--eco-track",
        action="store_true",
        help=(
            "Misura gli stage principali con EcoTracker e salva JSONL in "
            "results/<results-stage-name>/ecotracker."
        ),
    )
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    """Imposta le variabili d'ambiente necessarie prima di importare TensorFlow (GPU visibili, percorso CUDA per XLA)."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    if args.gpu_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible_devices

    libdevice = args.cuda_root / "nvvm" / "libdevice" / "libdevice.10.bc"
    if libdevice.exists() and "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={args.cuda_root}"


def parse_classes(classes_value: str) -> list[int]:
    """Converte la stringa --classes (es. "0,1") in una lista di interi validi (0 o 1), senza duplicati."""
    classes = []
    for part in str(classes_value).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in (0, 1):
            raise ValueError(f"Classe non valida: {value}")
        if value not in classes:
            classes.append(value)
    if not classes:
        raise ValueError("--classes non puo' essere vuoto")
    return classes


def format_cfg_label(guidance_scale: float, sample_steps: int) -> str:
    """Costruisce l'etichetta della configurazione di sampling (guidance scale + step) usata per nominare le cartelle delle immagini generate."""
    gs_label = f"{guidance_scale:g}".replace(".", "p")
    return f"cfg_gs{gs_label}_st{sample_steps}"


def step_from_checkpoint(path: Path) -> int:
    """Estrae il numero di step di training dal nome file del checkpoint (es. ldm_step20000.keras)."""
    return int(path.stem.replace("ldm_step", ""))


def collect_checkpoint_candidates(paths: ExperimentPaths, args: argparse.Namespace) -> list[dict]:
    """Elenca i checkpoint da valutare nello sweep: il best-train-loss (se presente) piu' tutti gli step-checkpoint con step >= min_step, filtrati per max_checkpoints o checkpoint_id se richiesto."""
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
            f"Nessun checkpoint candidato in {paths.checkpoints_dir} "
            f"per checkpoint_id={args.checkpoint_id!r}"
        )
    return checkpoint_candidates


def build_eval_config(args: argparse.Namespace) -> dict:
    """Riassume in un dizionario i parametri di valutazione che identificano univocamente una run dello sweep, usato per decidere se la cache delle metriche e' ancora valida."""
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
        "best_selection_metric": BEST_SELECTION_METRIC,
        "best_selection_tie_breaker": BEST_SELECTION_TIE_BREAKER,
        "positive_class": POSITIVE_CLASS,
    }


def normalize_eval_config(config: dict | None) -> dict:
    """Riempie i campi opzionali assenti in una config salvata in cache con i default correnti, per confrontarla in modo equo con build_eval_config."""
    normalized = dict(config or {})
    normalized.setdefault("seed", 42)
    normalized.setdefault("best_selection_metric", BEST_SELECTION_METRIC)
    normalized.setdefault("best_selection_tie_breaker", BEST_SELECTION_TIE_BREAKER)
    normalized.setdefault("positive_class", POSITIVE_CLASS)
    return normalized


def selection_policy() -> dict:
    """Descrive il criterio di selezione del best checkpoint (metrica, direzione, tie-breaker e motivazione) da salvare insieme ai risultati."""
    return {
        "selection_metric": BEST_SELECTION_METRIC,
        "selection_direction": SELECTION_DIRECTION,
        "tie_breaker": BEST_SELECTION_TIE_BREAKER,
        "tie_breaker_direction": TIE_BREAKER_DIRECTION,
        "selected_class": POSITIVE_CLASS,
        "selection_reason": SELECTION_REASON,
    }


def selection_policy_is_compatible(selection: dict) -> bool:
    """Verifica che il criterio di selezione salvato in una cache precedente coincida con quello attuale, altrimenti la cache va invalidata."""
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
    """Crea (se mancanti) e restituisce le cartelle di output dello stage results (plot, metriche, ecotracker)."""
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
    """Aggiunge una riga JSON al file di log dell'EcoTracker, forzando il flush su disco per non perdere dati se il processo viene interrotto."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


@contextmanager
def maybe_measure(args: argparse.Namespace, paths: ExperimentPaths, label: str):
    """Avvolge uno stage con EcoTracker se --eco-track e' attivo, salvando le metriche di sostenibilita' in JSONL; altrimenti e' un no-op trasparente."""
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


def build_candidate_signature(checkpoint_candidates: list[dict]) -> list[dict]:
    """Estrae dai candidati solo i campi rilevanti per il confronto con la cache, ignorando l'ordine di iterazione interno."""
    return [
        {
            "checkpoint_id": candidate["checkpoint_id"],
            "kind": candidate["kind"],
            "step": candidate["step"],
            "path": str(candidate["path"]),
        }
        for candidate in checkpoint_candidates
    ]


def use_cached_metrics_if_valid(
    args: argparse.Namespace,
    paths: ExperimentPaths,
    checkpoint_candidates: list[dict],
) -> bool:
    """Riusa checkpoint_metrics.json se config, candidati e criterio di selezione non sono cambiati, evitando di rigenerare immagini e ricalcolare FID/IS/PRDC senza motivo; in caso di cache valida sincronizza anche il best model su disco."""
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
        print(f"Cache metriche non leggibile, ricalcolo: {exc}")
        return False

    if payload.get("schema_version") != 2:
        print("Cache metriche con schema non valido, ricalcolo.")
        return False
    if normalize_eval_config(payload.get("config")) != eval_config:
        print("Cache metriche con configurazione diversa, ricalcolo.")
        return False
    if payload.get("candidate_signature") != candidate_signature:
        print("Cache metriche con checkpoint diversi, ricalcolo.")
        return False

    selection = payload.get("selection", {})
    if not selection_policy_is_compatible(selection):
        print(
            "Cache metriche con criterio di selezione diverso "
            f"({selection.get('selection_metric')} / {selection.get('tie_breaker')}), "
            "ricalcolo."
        )
        return False
    best_checkpoint_value = selection.get("best_checkpoint")
    best_model_value = selection.get("best_model") or selection.get("best_eval_model")
    if not best_checkpoint_value or not best_model_value:
        print("Cache metriche senza best_checkpoint/best_model, ricalcolo.")
        return False
    best_checkpoint = Path(best_checkpoint_value)
    best_model = Path(best_model_value)
    if not best_checkpoint.exists():
        print("Cache metriche valida ma best_checkpoint mancante, ricalcolo.")
        return False
    shutil.copy2(best_checkpoint, best_model)
    print(f"Best model sincronizzato da cache: {best_model}")

    if not refresh_artifacts_from_cache(args, paths, checkpoint_candidates, payload):
        return False

    print(f"Metriche checkpoint gia' presenti: {metrics_json_path}")
    print(
        "Best model da cache: "
        f"{selection.get('best_checkpoint_id')} -> {best_model_value}"
    )
    return True


def evaluation_generated_dir(paths: ExperimentPaths) -> Path:
    """Restituisce (creandola se serve) la cartella radice dove vengono salvate le immagini generate durante lo sweep."""
    out_dir = paths.evaluation_dir / "sweep_generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def fake_output_dir(
    paths: ExperimentPaths,
    args: argparse.Namespace,
    checkpoint_id: str,
    cls: int,
) -> Path:
    """Calcola la cartella delle immagini sintetiche per uno specifico checkpoint/configurazione CFG/classe."""
    cfg_label = format_cfg_label(args.guidance_scale, args.sample_steps)
    return evaluation_generated_dir(paths) / checkpoint_id / cfg_label / f"class_{cls}"


def list_fake_image_paths(
    paths: ExperimentPaths,
    args: argparse.Namespace,
    checkpoint_id: str,
    cls: int,
) -> list[Path]:
    """Elenca, in ordine di indice, le immagini sintetiche gia' generate per un checkpoint/classe, ignorando file con nome non numerico."""
    indexed_paths = []
    for path in fake_output_dir(paths, args, checkpoint_id, cls).glob("*.png"):
        try:
            indexed_paths.append((int(path.stem), path))
        except ValueError:
            continue
    return [path for _, path in sorted(indexed_paths)]


def next_fake_image_index(paths_for_class: list[Path]) -> int:
    """Determina il prossimo indice libero per nominare una nuova immagine generata, in modo da riprendere il sampling senza sovrascrivere quelle esistenti."""
    indices = []
    for path in paths_for_class:
        try:
            indices.append(int(path.stem))
        except ValueError:
            continue
    return max(indices) + 1 if indices else 0


def child_generate_command(
    args: argparse.Namespace,
    paths: ExperimentPaths,
    checkpoint_id: str,
) -> list[str]:
    """Costruisce la riga di comando per rilanciare questo stesso script in un sottoprocesso, in modalita' generate, su un singolo checkpoint."""
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
    ]
    if args.gpu_visible_devices is not None:
        command.extend(["--gpu-visible-devices", args.gpu_visible_devices])
    if args.decode_on_cpu:
        command.append("--decode-on-cpu")
    return command


def orchestrate_generation(
    args: argparse.Namespace,
    paths: ExperimentPaths,
    checkpoint_candidates: list[dict],
) -> None:
    """Lancia un sottoprocesso TensorFlow per ciascun checkpoint dello sweep, cosi' la sessione/memoria GPU viene azzerata tra un checkpoint e l'altro; si interrompe e propaga l'errore se un sottoprocesso fallisce, lasciando su disco le immagini gia' generate per poter riprendere."""
    print("Modalita generate orchestrata: un subprocess TensorFlow per checkpoint.")
    print("CFG_LABEL:", format_cfg_label(args.guidance_scale, args.sample_steps))
    for candidate in checkpoint_candidates:
        checkpoint_id = candidate["checkpoint_id"]
        print(f"\n[orchestrator] avvio checkpoint {checkpoint_id}")
        command = child_generate_command(args, paths, checkpoint_id)
        print(" ".join(command))
        completed = subprocess.run(command, cwd=str(paths.project_root), check=False)
        if completed.returncode != 0:
            print(
                f"[orchestrator] checkpoint {checkpoint_id} fallito con "
                f"return code {completed.returncode}."
            )
            print("Le immagini gia' salvate restano su disco: rilancia per riprendere.")
            raise SystemExit(completed.returncode)
        print(f"[orchestrator] checkpoint {checkpoint_id} completato.")


def save_single_fake_image(image_np, output_path: Path) -> None:
    """Salva un'immagine generata su disco come PNG, riportando i valori da [-1, 1] a [0, 255] se necessario."""
    import numpy as np
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = image_np.copy()
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0
    Image.fromarray(
        np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    ).save(output_path)


def sampler_trace_count(compiled_sampler) -> int | None:
    """Legge quante volte la funzione tf.function del sampler e' stata ritracciata, per accorgersi di retrace indesiderati durante lo sweep."""
    getter = getattr(compiled_sampler, "experimental_get_tracing_count", None)
    if getter is None:
        return None
    return int(getter())


def run_generation_worker(
    args: argparse.Namespace,
    paths: ExperimentPaths,
    candidate: dict,
) -> None:
    """Genera le immagini sintetiche mancanti per un singolo checkpoint: carica LDM/VAE solo se serve, compila il sampler una volta e lo riusa per tutte le classi, usando un seed deterministico per (classe, indice) cosi' il confronto tra checkpoint varia solo per i pesi del modello."""
    checkpoint_id = candidate["checkpoint_id"]
    ckpt_path = Path(candidate["path"])
    classes = parse_classes(args.classes)

    print(f"\n[checkpoint {checkpoint_id}] {ckpt_path.name}")
    print("CFG_LABEL:", format_cfg_label(args.guidance_scale, args.sample_steps))
    print("decode_on_cpu:", args.decode_on_cpu)
    print("batch sampling effettivo: 1")

    preflight_missing = []
    for cls in classes:
        existing_paths = list_fake_image_paths(paths, args, checkpoint_id, cls)
        existing_count = len(existing_paths)
        missing = max(0, args.n_gen_per_class - existing_count)
        preflight_missing.append(missing)
        if existing_count > args.n_gen_per_class:
            print(
                f"classe {cls}: {args.n_gen_per_class}/{args.n_gen_per_class} "
                f"gia' presenti ({existing_count} file totali, uso le prime "
                f"{args.n_gen_per_class})"
            )
        else:
            print(f"classe {cls}: {existing_count}/{args.n_gen_per_class} gia' presenti")
        print(f"classe {cls}: genero {missing} immagini mancanti")

    if sum(preflight_missing) == 0:
        print("Nessuna immagine mancante: non carico TensorFlow, LDM o VAE.")
        return

    configure_environment(args)

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

    configure_tensorflow()
    vram_gb("VRAM prima del checkpoint")

    schedule = build_schedule()
    latent_mean, latent_std = load_latent_stats(paths.latents_dir / "latent_stats.npz")
    vae_decoder = load_vae_decoder(paths.models_dir / "vae_decoder_best.keras")
    vram_gb("VRAM dopo load VAE decoder")
    ldm_model = load_ldm_model(ckpt_path)
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
    print("sampler compilato una sola volta e riusato per tutto il checkpoint")

    for cls in classes:
        existing_paths = list_fake_image_paths(paths, args, checkpoint_id, cls)
        existing_count = len(existing_paths)
        missing = max(0, args.n_gen_per_class - existing_count)
        if existing_count > args.n_gen_per_class:
            print(
                f"classe {cls}: {args.n_gen_per_class}/{args.n_gen_per_class} "
                f"gia' presenti ({existing_count} file totali, uso le prime "
                f"{args.n_gen_per_class})"
            )
        else:
            print(f"classe {cls}: {existing_count}/{args.n_gen_per_class} gia' presenti")
        print(f"classe {cls}: genero {missing} immagini mancanti")

        generated = 0
        next_index = next_fake_image_index(existing_paths)
        while generated < missing:
            output_index = next_index + generated
            # Seed identico per ogni checkpoint a parita' di (classe, indice immagine):
            # la variabile confrontata nello sweep deve essere solo il checkpoint, non
            # anche il rumore iniziale da cui si parte.
            sample_seed = tf.constant(
                [
                    int(args.seed) + int(cls) * 1_000,
                    int(output_index),
                ],
                dtype=tf.int32,
            )
            image_tensor = compiled_sampler(tf.constant(cls, dtype=tf.int32), sample_seed)
            image_np = image_tensor[0].numpy().squeeze()
            output_path = fake_output_dir(paths, args, checkpoint_id, cls) / (
                f"{output_index:04d}.png"
            )
            save_single_fake_image(image_np, output_path)

            del image_tensor
            del image_np
            generated += 1

            trace_count = sampler_trace_count(compiled_sampler)
            trace_msg = f" | sampler_traces={trace_count}" if trace_count is not None else ""
            print(
                f"  classe {cls}: generate {generated}/{missing} mancanti "
                f"-> {output_path.name}{trace_msg}",
                flush=True,
            )
            if (
                generated == missing
                or (args.vram_log_every > 0 and generated % args.vram_log_every == 0)
            ):
                gc.collect()
                vram_gb(
                    f"VRAM checkpoint {checkpoint_id} classe {cls} dopo {generated} immagini"
                )

        fake_paths = list_fake_image_paths(paths, args, checkpoint_id, cls)
        if len(fake_paths) < min(args.n_gen_per_class, existing_count + missing):
            raise RuntimeError(
                f"Classe {cls}, checkpoint {checkpoint_id}: trovate {len(fake_paths)} "
                f"immagini in {fake_output_dir(paths, args, checkpoint_id, cls)}"
            )

    print("sampler_traces_finale:", sampler_trace_count(compiled_sampler))
    del compiled_sampler
    del ldm_model
    del vae_decoder
    del schedule
    del latent_mean
    del latent_std
    gc.collect()
    tf.keras.backend.clear_session()
    vram_gb("VRAM alla fine del checkpoint")


def ensure_metrics_classes(args: argparse.Namespace) -> None:
    """Blocca l'esecuzione in --mode metrics se --classes non e' esattamente 0,1, perche' checkpoint_metrics.json assume sempre entrambe le classi."""
    classes = sorted(parse_classes(args.classes))
    if classes != [0, 1]:
        raise ValueError(
            "--mode metrics richiede --classes 0,1 per mantenere il formato "
            "checkpoint_metrics.json compatibile."
        )


def json_scalar(value):
    """Converte uno scalare numpy/pandas in un valore Python serializzabile in JSON, mappando i NaN su None."""
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
    """Seleziona la riga del checkpoint migliore: ordina per fid_1 crescente, poi per is_mean_1 decrescente come tie-breaker, poi per ordine di checkpoint per rendere la scelta deterministica."""
    required = [BEST_SELECTION_METRIC, BEST_SELECTION_TIE_BREAKER, "checkpoint_order"]
    missing = [column for column in required if column not in results_df.columns]
    if missing:
        raise RuntimeError(f"Metriche mancanti per scegliere il best: {missing}")
    ranked_df = results_df.dropna(subset=[BEST_SELECTION_METRIC]).sort_values(
        by=[BEST_SELECTION_METRIC, BEST_SELECTION_TIE_BREAKER, "checkpoint_order"],
        ascending=[True, False, True],
    )
    if ranked_df.empty:
        raise RuntimeError(
            f"Nessun risultato valido per scegliere il best con {BEST_SELECTION_METRIC}"
        )
    return ranked_df.iloc[0]


def build_selection(best_row, best_eval_model_path: Path) -> dict:
    """Costruisce il dizionario di selezione del best checkpoint, combinando la policy di selezione con i metadati e le metriche della riga vincente."""
    import pandas as pd

    step_value = best_row.get("step")
    selection = {
        **selection_policy(),
        "best_checkpoint_id": str(best_row["checkpoint_id"]),
        "best_step": None if pd.isna(step_value) else int(step_value),
        "best_checkpoint": str(Path(best_row["checkpoint_path"])),
        "best_model": str(best_eval_model_path),
        "best_eval_model": str(best_eval_model_path),
    }
    for column in BEST_ROW_METRIC_COLUMNS:
        if column in best_row:
            selection[column] = json_scalar(best_row[column])
    return selection


def copy_artifact(source: Path, destination: Path) -> Path:
    """Copia un artefatto (CSV/JSON/plot) nella cartella results, creando le directory mancanti."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def plot_checkpoint_metric_comparisons(results_df, selection: dict, plots_dir: Path) -> dict:
    """Genera i grafici di confronto FID e Inception Score tra tutti i checkpoint dello sweep, evidenziando con una linea verticale il checkpoint selezionato come best."""
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
            [("fid_0", "FID classe 0"), ("fid_1", "FID classe 1"), ("fid_mean", "FID medio")],
            "Checkpoint FID comparison - BEST = minimo fid_1",
            "FID",
            "lower",
        ),
        (
            "checkpoint_is_comparison.png",
            [
                ("is_mean_0", "IS classe 0"),
                ("is_mean_1", "IS classe 1"),
                ("is_mean_avg", "IS medio"),
            ],
            "Checkpoint Inception Score comparison - is_mean_1 solo tie-breaker",
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
    """Compone un'unica figura di riepilogo (testo, FID, IS, PRDC) sul checkpoint risultato vincitore, da allegare ai risultati finali."""
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
                "Criterio di selezione: minimo FID sulla classe positiva",
                "Tie-breaker: massimo Inception Score sulla classe positiva",
                f"FID classe positiva: {float(best_row['fid_1']):.4f}",
                f"FID medio: {float(best_row['fid_mean']):.4f}",
            ]
        ),
        va="top",
        fontsize=12,
    )

    fid_values = [best_row["fid_0"], best_row["fid_1"], best_row["fid_mean"]]
    axes[0, 1].bar(["classe 0", "classe 1", "medio"], fid_values, color=["#4C78A8", "#E45756", "#72B7B2"])
    axes[0, 1].set_title("FID del best checkpoint")
    axes[0, 1].set_ylabel("FID")
    axes[0, 1].grid(axis="y", alpha=0.2)

    is_values = [best_row["is_mean_0"], best_row["is_mean_1"], best_row["is_mean_avg"]]
    axes[1, 0].bar(["classe 0", "classe 1", "medio"], is_values, color=["#4C78A8", "#E45756", "#72B7B2"])
    axes[1, 0].set_title("Inception Score del best checkpoint")
    axes[1, 0].set_ylabel("IS mean")
    axes[1, 0].grid(axis="y", alpha=0.2)

    grouped_metrics = ["precision", "recall", "density", "coverage"]
    x = range(len(grouped_metrics))
    class0 = [best_row[f"{metric}_0"] for metric in grouped_metrics]
    class1 = [best_row[f"{metric}_1"] for metric in grouped_metrics]
    axes[1, 1].bar([pos - 0.18 for pos in x], class0, width=0.36, label="classe 0", color="#4C78A8")
    axes[1, 1].bar([pos + 0.18 for pos in x], class1, width=0.36, label="classe 1", color="#E45756")
    axes[1, 1].set_title("PRDC per classe")
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
    """Salva il manifest JSON dello stage (selezione + policy + elenco artefatti) sia nella cartella di evaluation che in quella results, per tracciabilita'."""
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
    """Finalizza lo sweep: seleziona il best checkpoint, copia il suo file .keras come modello di valutazione, scrive CSV/JSON/plot/manifest sia nella cartella di evaluation che nella cartella results, e restituisce i percorsi di tutti gli artefatti prodotti."""
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

        selection = build_selection(best_row, best_eval_model_path)
        checkpoint_records = json.loads(results_df.to_json(orient="records"))
        metrics_payload = {
            "schema_version": 2,
            "config": build_eval_config(args),
            "candidate_signature": build_candidate_signature(checkpoint_candidates),
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
    """Rigenera CSV/JSON/plot/manifest a partire dalle metriche gia' presenti in cache (--mode artifacts), verificando prima che il best checkpoint ricalcolato coincida con quello salvato."""
    import pandas as pd

    try:
        results_df = pd.DataFrame(payload["checkpoints"])
        best_row = select_best_row(results_df)
    except Exception as exc:
        print(f"Cache metriche senza tabella checkpoint valida, ricalcolo: {exc}")
        return False

    selection = payload.get("selection", {})
    cached_best_id = selection.get("best_checkpoint_id")
    expected_best_id = str(best_row["checkpoint_id"])
    if cached_best_id is not None and cached_best_id != expected_best_id:
        print(
            "Cache metriche incoerente: best salvato "
            f"{cached_best_id}, best da fid_1 {expected_best_id}."
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
    """Calcola FID/IS/PRDC di ogni checkpoint contro il validation set reale (classi 0 e 1), aggrega i risultati in una tabella e delega a persist_sweep_artifacts la selezione del best e il salvataggio degli artefatti."""
    ensure_metrics_classes(args)

    import pandas as pd

    from ldm_evaluation_utils import evaluate_generated_paths_against_metadata

    if args.inception_weights == "none":
        print(
            "Nota: --inception-weights=none viene ignorato con generative_evaluator.py; "
            "torchmetrics usa il suo modello Inception."
        )

    val_csv_path = paths.metadata_dir / "val.csv"
    val_df = pd.read_csv(val_csv_path)
    print("Riferimento reale per le metriche: validation set")
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
                fake_paths = list_fake_image_paths(paths, args, checkpoint_id, cls)[
                    : args.n_gen_per_class
                ]
                if len(fake_paths) < args.n_gen_per_class:
                    raise RuntimeError(
                        f"Classe {cls}, checkpoint {checkpoint_id}: trovate {len(fake_paths)} "
                        f"immagini su {args.n_gen_per_class} richieste in "
                        f"{fake_output_dir(paths, args, checkpoint_id, cls)}"
                    )

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
    print("\n=== RISULTATI SWEEP ===")
    print(results_df.to_string(index=False))
    print("Salvato:", artifacts["results_path"])
    print("Salvato:", artifacts["metrics_json_path"])
    print("Salvato:", paths.evaluation_dir / "artifacts_manifest.json")
    print(
        f"Best checkpoint ({BEST_SELECTION_METRIC}): "
        f"{selection['best_checkpoint_id']} -> {selection['best_eval_model']}"
    )


def main() -> None:
    """Punto di ingresso CLI: prepara l'ambiente e i checkpoint candidati, poi smista l'esecuzione tra generate/metrics/artifacts/both in base a --mode, riusando la cache delle metriche quando possibile."""
    args = parse_args()
    configure_environment(args)
    if args.mini_batch != 1:
        print(
            f"Nota: --mini-batch={args.mini_batch} ignorato per lo sweep; "
            "uso batch fisso 1."
        )
        args.mini_batch = 1

    paths = get_experiment_paths(args.project_root, args.experiment_dir)
    checkpoint_candidates = collect_checkpoint_candidates(paths, args)
    print(f"Checkpoint candidati: {len(checkpoint_candidates)}")
    for candidate in checkpoint_candidates:
        print(f"  {candidate['checkpoint_id']} -> {Path(candidate['path']).name}")

    if args.mode == "generate" and args.checkpoint_id is not None:
        run_generation_worker(args, paths, checkpoint_candidates[0])
        return

    if args.mode == "generate":
        orchestrate_generation(args, paths, checkpoint_candidates)
        return

    if use_cached_metrics_if_valid(args, paths, checkpoint_candidates):
        return

    if args.mode == "artifacts":
        raise RuntimeError(
            "Cache metriche non valida o assente: non rigenero artefatti senza "
            "checkpoint_metrics.json compatibile."
        )

    if args.mode == "both":
        orchestrate_generation(args, paths, checkpoint_candidates)
        run_metrics(args, paths, checkpoint_candidates)
        return

    if args.mode == "metrics":
        run_metrics(args, paths, checkpoint_candidates)
        return

    raise ValueError(f"Mode non gestito: {args.mode}")


if __name__ == "__main__":
    main()
