#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   MammoDiffusion — train_ldm.py                                             ║
# ║   Script STANDALONE: cella 4 → fine LDM Training (celle 4-23)              ║
# ║                                                                              ║
# ║   Prerequisiti su disco:                                                    ║
# ║     data/processed/{train,val,test}/{0,1}/*.png                              ║
# ║     data/processed/metadata/{train,val,test}.csv                             ║
# ║     experiments/<esperimento>/models/vae_{encoder,decoder}_best.keras        ║
# ║                                                                              ║
# ║   Avvio dal notebook (processo figlio indipendente):                        ║
# ║     import subprocess, sys                                                   ║
# ║     proc = subprocess.Popen(                                                 ║
# ║         [sys.executable, "notebooks/train_ldm.py"],                      ║
# ║         start_new_session=True,   # <-- figlio indipendente, sopravvive     ║
# ║         stdout=open("experiments/.../logs/ldm_train.log","a"),              ║
# ║         stderr=subprocess.STDOUT,                                            ║
# ║     )                                                                        ║
# ║     print(f"LDM training avviato — PID {proc.pid}")                         ║
# ║     print("Puoi terminare il kernel: il processo continua.")                 ║
# ║                                                                              ║
# ║   Output prodotti da questo script:                                          ║
# ║     experiments/<esperimento>/checkpoints_ldm/ldm_unet_best.keras           ║
# ║     experiments/<esperimento>/checkpoints_ldm/ldm_stepXXXXXX.keras          ║
# ║     experiments/<esperimento>/checkpoints_ldm/ldm_stepXXXXXX.keras          ║
# ║     experiments/<esperimento>/checkpoints_ldm/ldm_unet_final_stepXXXXXX.keras║
# ║     experiments/<esperimento>/logs/ldm_history.json                         ║
# ║     results/<stage>/plots/ldm_metrics.png                                  ║
# ║     results/<stage>/ecotracker/sustainability_log.jsonl                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations
import argparse
import gc
import hashlib
import math
import os
import shutil
import subprocess
import sys
import time
import json
import contextlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ldm_project_paths import RESULTS_STAGE_NAME, find_project_root

import numpy as np
import pandas as pd

def configure_tensorflow_runtime_environment() -> None:
    """Imposta i flag TF/XLA prima dell'import di TensorFlow.

    La RTX 5060 Ti (Blackwell, CC 12.0) non e' coperta dai binari CUDA di TF 2.15:
    alcuni kernel vengono compilati al volo. Dare a XLA il path di libdevice evita
    crash su op elementari quando il processo parte fuori dal notebook.
    """
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
    if "XLA_FLAGS" in os.environ:
        return

    candidates = [
        Path(sys.prefix) / "nvvm" / "libdevice" / "libdevice.10.bc",
        Path("/home/fede/miniforge3/envs/tf-gpu/nvvm/libdevice/libdevice.10.bc"),
        Path("/usr/local/cuda/nvvm/libdevice/libdevice.10.bc"),
        Path("/usr/local/cuda-12.4/nvvm/libdevice/libdevice.10.bc"),
        Path("/usr/local/cuda-12.2/nvvm/libdevice/libdevice.10.bc"),
        Path("/usr/lib/nvidia-cuda-toolkit/nvvm/libdevice/libdevice.10.bc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            cuda_data_dir = candidate.parent.parent.parent
            os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={cuda_data_dir}"
            break


configure_tensorflow_runtime_environment()

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")   # non-interactive — nessuna GUI richiesta
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers

from ldm_keras_utils import predict_epsilon_from_model_output
from ldm_v3_unet_keras import build_ldm_unet_v3, make_v_target, min_snr_weight

DEFAULT_EXPERIMENT_NAME = "20260617_ldm_basic"


def parse_args() -> argparse.Namespace:
    """Definisce e legge gli argomenti CLI con cui il notebook lancia questo script in sottoprocesso."""
    parser = argparse.ArgumentParser(
        description="Train MammoDiffusion LDM v2 using shared project data and experiment-scoped artifacts."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Root della repo MammoDiffusion. Se omesso viene rilevata automaticamente.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Cartella esperimento dove salvare latenti, checkpoint, modelli, log e output.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=80_000,
        help="Numero totale di step LDM.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=7_000,
        help="Frequenza di salvataggio checkpoint.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="Frequenza di log training.",
    )
    parser.add_argument(
        "--resume-from-latest",
        action="store_true",
        help=(
            "Se presenti checkpoint ldm_step*.keras, riprende dal checkpoint "
            "con step piu' alto e continua fino a --total-steps."
        ),
    )
    parser.add_argument(
        "--skip-latent-encoding",
        action="store_true",
        help=(
            "Usa latents_train.npz, latents_val.npz e latent_stats.npz gia' presenti "
            "nell'esperimento, senza caricare un VAE Keras. Usato dal ramo SD-VAE."
        ),
    )
    parser.add_argument(
        "--unet-version",
        choices=["v2", "v3"],
        default="v2",
        help=(
            "v2 = build_ldm_unet() (Conv2DTranspose, LeakyReLU). "
            "v3 = ldm_v3_unet_keras.build_ldm_unet_v3() (Upsample+Conv, ResBlock stile SD)."
        ),
    )
    parser.add_argument(
        "--parameterization",
        choices=["eps", "v"],
        default="eps",
        help=(
            "eps (default, retrocompatibile con 04b/04b1/04b2) = la U-Net predice il rumore. "
            "v (Salimans & Ho 2022) = la U-Net predice v = sqrt(ab)*noise - sqrt(1-ab)*x0."
        ),
    )
    parser.add_argument(
        "--use-min-snr",
        action="store_true",
        help="Applica il weighting Min-SNR-gamma (Hang et al. 2023) alla sola loss_simple.",
    )
    parser.add_argument(
        "--min-snr-gamma",
        type=float,
        default=5.0,
        help="Gamma del weighting Min-SNR, usato solo se --use-min-snr e' attivo.",
    )
    parser.add_argument(
        "--vae-source",
        default="sd_vae_original",
        help=(
            "Informativo: identifica il VAE usato per produrre i latenti (es. "
            "'sd_vae_original' o 'sd_vae_finetuned_03'). Salvato in training_manifest.json."
        ),
    )
    parser.add_argument(
        "--uses-vae-ft-from-03",
        action="store_true",
        help="Informativo: marca nel manifest che i latenti vengono dal VAE fine-tuned di 03.",
    )
    parser.add_argument(
        "--notebook-name",
        default=None,
        help="Informativo: nome del notebook che ha lanciato il training, salvato nel manifest.",
    )
    parser.add_argument(
        "--results-stage-name",
        default=RESULTS_STAGE_NAME,
        help="Sottocartella di results dove salvare il log EcoTracker LDM.",
    )
    return parser.parse_args()


ARGS = parse_args()

tf.random.set_seed(42)
np.random.seed(42)
print("TF version:", tf.__version__)
print("GPU disponibili:", tf.config.list_physical_devices("GPU"))

# ══════════════════════════════════════════════════════════════════════════════
# IPERPARAMETRI GLOBALI
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE        = 512
CHANNELS        = 1
LATENT_SIZE     = 64
LATENT_CHANNELS = 4
NUM_CLASSES     = 2

# VAE (solo per caricare il modello — non si riaddestra)
KL_WEIGHT       = 1e-3
SSIM_WEIGHT     = 0.3
VAE_BATCH_SIZE  = 8

# LDM
LDM_BATCH_SIZE  = 16
LDM_LR          = 1e-4
EMBED_DIM       = 128
MODEL_CHANNELS  = 64
NUM_DIFF_STEPS  = 1000
LAMBDA_VLB      = 0.001
CFG_DROPOUT     = 0.15
CFG_SCALE       = 3.0
SAMPLE_STEPS    = 100

# Augmentation
POSITIVE_AUGMENT_COPIES = 3

# Nomi classi
CLASS_NAMES = {0: "Negativo (sano)", 1: "Positivo (cancro)"}

# Training LDM
LDM_TOTAL_STEPS = ARGS.total_steps
LOG_EVERY       = ARGS.log_every
CKPT_EVERY      = ARGS.checkpoint_every

# v3: architettura / parameterization / min-SNR (default retrocompatibili con 04b/04b1/04b2)
UNET_VERSION     = ARGS.unet_version
PARAMETERIZATION = ARGS.parameterization
USE_MIN_SNR      = ARGS.use_min_snr
MIN_SNR_GAMMA    = ARGS.min_snr_gamma

print(f"IMG_SIZE={IMG_SIZE} | LATENT={LATENT_SIZE}×{LATENT_SIZE}×{LATENT_CHANNELS}")
print(f"LDM: batch={LDM_BATCH_SIZE}, CFG={CFG_SCALE}, steps={LDM_TOTAL_STEPS}")
print(
    f"UNET_VERSION={UNET_VERSION} | PARAMETERIZATION={PARAMETERIZATION} | "
    f"USE_MIN_SNR={USE_MIN_SNR} | MIN_SNR_GAMMA={MIN_SNR_GAMMA}"
)

# ══════════════════════════════════════════════════════════════════════════════
# PATH
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT       = find_project_root(override=ARGS.project_root)
EXPERIMENT_DIR     = (
    ARGS.experiment_dir.expanduser().resolve()
    if ARGS.experiment_dir is not None
    else PROJECT_ROOT / "experiments" / DEFAULT_EXPERIMENT_NAME
)

DATA_DIR           = PROJECT_ROOT / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
ARCHIVES_DIR       = DATA_DIR / "archives"

DATA_AUG            = DATA_DIR / "real_augmented"
LATENTS_DIR         = EXPERIMENT_DIR / "latents"
CKPT_DIR            = EXPERIMENT_DIR / "checkpoints_ldm"
MODELS_DIR          = EXPERIMENT_DIR / "models"
LOGS_DIR            = EXPERIMENT_DIR / "logs"
RESULTS_PLOTS_DIR = PROJECT_ROOT / "results" / ARGS.results_stage_name / "plots"
RESULTS_ECOTRACKER_DIR = PROJECT_ROOT / "results" / ARGS.results_stage_name / "ecotracker"

for _d in [
    DATA_PROCESSED_DIR,
    ARCHIVES_DIR,
    DATA_AUG,
    LATENTS_DIR,
    CKPT_DIR,
    MODELS_DIR,
    LOGS_DIR,
    RESULTS_PLOTS_DIR,
    RESULTS_ECOTRACKER_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)


def sync_existing_training_plots_to_results() -> None:
    """Segnala se il plot del training LDM e' gia' presente nei results, senza rigenerarlo (usata quando il training viene skippato)."""
    for plot_name in ["ldm_metrics.png"]:
        destination_path = RESULTS_PLOTS_DIR / plot_name
        if destination_path.exists():
            print(f"Plot training gia' presente nei results: {destination_path}")
            continue
        print(f"Plot training LDM non trovato nei results: {destination_path}")


LATENTS_TRAIN_PATH    = LATENTS_DIR / "latents_train.npz"
LATENTS_VAL_PATH      = LATENTS_DIR / "latents_val.npz"
LATENT_STATS_PATH     = LATENTS_DIR / "latent_stats.npz"
LATENTS_MANIFEST_PATH = LATENTS_DIR / "latents_manifest.json"

METADATA_DIR   = DATA_PROCESSED_DIR / "metadata"
ALL_CSV_PATH   = METADATA_DIR / "all_processed.csv"
TRAIN_CSV_PATH = METADATA_DIR / "train.csv"
VAL_CSV_PATH   = METADATA_DIR / "val.csv"
TEST_CSV_PATH  = METADATA_DIR / "test.csv"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("EXPERIMENT_DIR:", EXPERIMENT_DIR)
print("DATA_PROCESSED_DIR:", DATA_PROCESSED_DIR)
print("CKPT_DIR:", CKPT_DIR)


def write_training_manifest(final_model_path: Optional[Path], total_steps: Optional[int]) -> Path:
    """Salva/aggiorna training_manifest.json nella cartella dell'esperimento con i campi che generazione/valutazione/04c usano per rilevare automaticamente parameterization, unet_version e sorgente del VAE, invece di doverli assumere."""
    manifest = {
        "notebook": ARGS.notebook_name,
        "unet_version": UNET_VERSION,
        "parameterization": PARAMETERIZATION,
        "use_min_snr": USE_MIN_SNR,
        "min_snr_gamma": MIN_SNR_GAMMA,
        "vae_source": ARGS.vae_source,
        "uses_vae_ft_from_03": ARGS.uses_vae_ft_from_03,
        "total_steps": total_steps,
        "final_model": str(final_model_path) if final_model_path is not None else None,
    }
    manifest_path = EXPERIMENT_DIR / "training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    print(f"Training manifest salvato: {manifest_path}")
    return manifest_path


def step_from_model_path(path: Path, prefix: str) -> Optional[int]:
    """Estrae il numero di step dal nome file di un checkpoint, dato il suo prefisso (es. 'ldm_step000100.keras' -> 100)."""
    if not path.stem.startswith(prefix):
        return None
    try:
        return int(path.stem.replace(prefix, ""))
    except ValueError:
        return None


def latest_step_checkpoint_path() -> tuple[Optional[int], Optional[Path]]:
    """Cerca tra i checkpoint periodici in CKPT_DIR e restituisce quello con lo step piu' alto, per capire se/da dove riprendere."""
    candidates = []
    for path in CKPT_DIR.glob("ldm_step*.keras"):
        step = step_from_model_path(path, "ldm_step")
        if step is not None:
            candidates.append((step, path))
    return max(candidates, default=(None, None), key=lambda item: item[0] or -1)


existing_final_models = sorted(
    CKPT_DIR.glob("ldm_unet_final_step*.keras"),
    key=lambda path: step_from_model_path(path, "ldm_unet_final_step") or -1,
)
latest_existing_step, latest_existing_path = latest_step_checkpoint_path()

if ARGS.resume_from_latest and latest_existing_path is not None:
    if latest_existing_step is not None and latest_existing_step >= LDM_TOTAL_STEPS:
        print(
            f"\nCheckpoint gia' al target: {latest_existing_path.name} "
            f"(step {latest_existing_step} >= {LDM_TOTAL_STEPS})."
        )
        print("Training LDM skippato.")
        sync_existing_training_plots_to_results()
        write_training_manifest(latest_existing_path, latest_existing_step)
        sys.exit(0)
elif existing_final_models:
    print(f"\nFinal model gia' presente: {existing_final_models[-1]}")
    print("Training LDM skippato. Usa --resume-from-latest con --total-steps maggiore per continuare.")
    sync_existing_training_plots_to_results()
    write_training_manifest(
        existing_final_models[-1],
        step_from_model_path(existing_final_models[-1], "ldm_unet_final_step"),
    )
    sys.exit(0)


def copy_if_missing(source: Path, destination: Path) -> Path:
    """Copia un file solo se la destinazione non esiste ancora, per non risovrascrivere asset gia' presenti nell'esperimento."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copy2(source, destination)
    return destination


def ensure_model_asset(filename: str) -> Path:
    """Garantisce che un asset del VAE (encoder/decoder gia' addestrato) sia disponibile nella cartella models dell'esperimento, copiandolo dalle posizioni note se manca."""
    destination = MODELS_DIR / filename
    if destination.exists():
        return destination

    candidates = [
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "alex" / filename,
        PROJECT_ROOT / "alex" / "mammodiffusion_backup_estratto" / "workspace" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            print(f"Copio {filename} in esperimento da: {candidate}")
            return copy_if_missing(candidate, destination)

    raise FileNotFoundError(
        f"Non trovo {filename}. Mettilo in {MODELS_DIR} oppure in alex/ prima di lanciare."
    )


if ARGS.skip_latent_encoding:
    VAE_ENCODER_PATH = None
    VAE_DECODER_PATH = None
    print("Skip encoding latenti: uso cache latenti gia' preparata nell'esperimento.")
else:
    VAE_ENCODER_PATH = ensure_model_asset("vae_encoder_best.keras")
    VAE_DECODER_PATH = ensure_model_asset("vae_decoder_best.keras")

# ══════════════════════════════════════════════════════════════════════════════
# ECOTRACKER — usa il modulo condiviso eco_tracker.py (stessa logica di
# evaluate_ldm.py e generate_ldm.py). Fallback minimale solo se
# psutil/codecarbon non sono disponibili/installabili, per non bloccare il training.
# ══════════════════════════════════════════════════════════════════════════════
try:
    from eco_tracker import SustainabilityMetrics, measure_sustainability
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "codecarbon", "psutil"])
        from eco_tracker import SustainabilityMetrics, measure_sustainability
    except Exception:
        @dataclass
        class SustainabilityMetrics:
            """Fallback minimale delle metriche di sostenibilita', usato quando psutil/codecarbon non sono installabili."""
            elapsed_seconds: float = 0.0
            peak_ram_mb: float = 0.0
            energy_kwh: float = 0.0
            co2_kg: float = 0.0
            label: str = "run"

            def __str__(self) -> str:
                """Riassume in una riga il tempo di esecuzione, segnalando che RAM/energia/CO2 non sono disponibili in questo fallback."""
                return (
                    f"[{self.label}] Tempo: {self.elapsed_seconds:.2f}s | "
                    "RAM/energia/CO2 non disponibili (psutil/codecarbon non installabili)"
                )

            def to_dict(self) -> dict:
                """Serializza le metriche in un dict, per poterle scrivere nel log JSONL di sostenibilita'."""
                return {
                    "label": self.label,
                    "elapsed_seconds": self.elapsed_seconds,
                    "peak_ram_mb": self.peak_ram_mb,
                    "energy_kwh": self.energy_kwh,
                    "co2_kg": self.co2_kg,
                }

        @contextlib.contextmanager
        def measure_sustainability(label: str = "run", sample_interval: float = 0.5):
            """Context manager fallback che misura solo il tempo trascorso, senza tracciare RAM/energia/CO2 (nessuna dipendenza esterna richiesta)."""
            t0 = time.perf_counter()
            tracker = type("_NoOpEcoTracker", (), {"metrics": None})()
            try:
                yield tracker
            finally:
                tracker.metrics = SustainabilityMetrics(
                    elapsed_seconds=time.perf_counter() - t0,
                    label=label,
                )


# ══════════════════════════════════════════════════════════════════════════════
# ── HELPER: caricamento metadata ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
IMAGE_EXTENSIONS  = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EXPECTED_SPLITS   = ["train", "val", "test"]
EXPECTED_LABELS   = ["0", "1"]
REQUIRED_METADATA = ["all_processed.csv", "train.csv", "val.csv", "test.csv"]
DATASET_ROOT      = DATA_PROCESSED_DIR.resolve()


def load_metadata(csv_path, dataset_root):
    """Carica un CSV di metadata, ricostruisce il path assoluto di ogni immagine sotto dataset_root e verifica che tutti i file esistano davvero su disco."""
    df = pd.read_csv(csv_path).copy()
    required_cols = ["patient_id", "image_id", "label", "split", "processed_path"]
    missing_cols  = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Mancano colonne obbligatorie in {csv_path}: {missing_cols}")
    df["patient_id"] = df["patient_id"].astype(str)
    df["image_id"]   = df["image_id"].astype(str)
    df["label"]      = df["label"].astype(int)
    df["split"]      = df["split"].astype(str)
    df["filename"]   = df["processed_path"].apply(
        lambda p: str(p).replace("\\", "/").split("/")[-1]
    )
    df["processed_path"] = df.apply(
        lambda row: str(
            dataset_root / row["split"] / str(row["label"]) / row["filename"]
        ),
        axis=1,
    )
    df["file_exists"] = df["processed_path"].apply(lambda p: Path(p).exists())
    missing_files = df[~df["file_exists"]]
    if len(missing_files) > 0:
        print(f"Attenzione: {len(missing_files)} immagini non trovate.")
        print(missing_files[["patient_id", "image_id", "split", "label",
                              "filename", "processed_path"]].head(10).to_string())
        raise FileNotFoundError("Alcune immagini preprocessate non sono state trovate.")
    df = df.drop(columns=["file_exists"])
    return df


print("\n── Caricamento metadata ──")
processed_df = load_metadata(ALL_CSV_PATH, DATASET_ROOT)
train_df     = load_metadata(TRAIN_CSV_PATH, DATASET_ROOT)
val_df       = load_metadata(VAL_CSV_PATH, DATASET_ROOT)
test_df      = load_metadata(TEST_CSV_PATH, DATASET_ROOT)
print(f"Totale: {len(processed_df)} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(pd.crosstab(processed_df["split"], processed_df["label"]))


# ══════════════════════════════════════════════════════════════════════════════
# ── CELLA 5 — Augmentation condivisa (solo positivi train) ───────────────────
# ══════════════════════════════════════════════════════════════════════════════
RESET_DATASET = False
AUGMENTED_METADATA_PATH = DATA_AUG / "metadata.csv"


def resolve_project_path(path_value: str | Path) -> Path:
    """Risolve un path relativo rispetto alla root del progetto, lasciando invariati i path già assoluti."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_existing_augmented_metadata() -> Optional[pd.DataFrame]:
    """Riusa il dataset augmentato (reali + copie positive) già scritto su disco se ha le colonne giuste e i file referenziati esistono ancora, evitando di rigenerarlo ad ogni run."""
    if RESET_DATASET or not AUGMENTED_METADATA_PATH.exists():
        return None
    df = pd.read_csv(AUGMENTED_METADATA_PATH).copy()
    required_cols = ["file_name", "label", "patient_id", "image_id", "source"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Metadata augmented non riusabile, colonne mancanti: {missing_cols}")
        return None
    missing_paths = [
        path
        for path in df["file_name"].head(20).map(resolve_project_path)
        if not path.exists()
    ]
    if missing_paths:
        print("Metadata augmented non riusabile, primi path mancanti:")
        for path in missing_paths[:5]:
            print(" ", path)
        return None
    return df


augmented_df = load_existing_augmented_metadata()
if augmented_df is not None:
    print("\n── Augmentation positivi ──")
    print("Uso dataset aumentato condiviso gia' presente:")
    print("DATA_AUG:", DATA_AUG)
    print("Metadata:", AUGMENTED_METADATA_PATH)
    print(augmented_df["label"].value_counts())
else:
    if RESET_DATASET and DATA_AUG.exists():
        shutil.rmtree(DATA_AUG)
    DATA_AUG.mkdir(parents=True, exist_ok=True)

    source_df = train_df.copy().reset_index(drop=True)

    print("\n── Augmentation positivi ──")
    print("Creo dataset aumentato condiviso in:", DATA_AUG)
    print("Distribuzione train originale:")
    print(source_df["label"].value_counts())


def mild_positive_augmentation(img, aug_idx):
    """Applica una perturbazione leggera (contrasto, luminosità, rumore gaussiano) a un'immagine positiva, per generare copie augmentate riproducibili (seed legato ad aug_idx) e bilanciare le classi nel train set."""
    arr = np.array(img).astype(np.float32)
    rng = np.random.default_rng(42 + aug_idx)
    contrast   = rng.uniform(0.90, 1.10)
    brightness = rng.uniform(-8, 8)
    noise      = rng.normal(loc=0.0, scale=2.0, size=arr.shape)
    mean = arr.mean()
    arr  = (arr - mean) * contrast + mean
    arr  = arr + brightness + noise
    arr  = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


if augmented_df is None:
    metadata_rows = []
    out_idx = 0

    for row_idx, row in source_df.iterrows():
        label    = int(row["label"])
        src_path = Path(row["processed_path"]).resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"Immagine non trovata: {src_path}")

        real_rel_path = src_path.relative_to(PROJECT_ROOT).as_posix()
        metadata_rows.append({
            "file_name":               real_rel_path,
            "label":                   label,
            "patient_id":              str(row["patient_id"]),
            "image_id":                str(row["image_id"]),
            "source":                  "real",
            "original_processed_path": str(src_path),
        })

        if label == 1 and POSITIVE_AUGMENT_COPIES > 0:
            img_l = Image.open(src_path).convert("L")
            for aug_num in range(POSITIVE_AUGMENT_COPIES):
                aug_img_l = mild_positive_augmentation(img_l, aug_idx=(row_idx * 100 + aug_num))
                aug_name  = f"mammo_{out_idx:06d}_label1_aug{aug_num}.png"
                aug_path  = DATA_AUG / aug_name
                aug_img_l.save(aug_path)
                aug_rel_path = aug_path.relative_to(PROJECT_ROOT).as_posix()
                metadata_rows.append({
                    "file_name":               aug_rel_path,
                    "label":                   1,
                    "patient_id":              str(row["patient_id"]),
                    "image_id":                str(row["image_id"]),
                    "source":                  "positive_augmentation",
                    "original_processed_path": str(src_path),
                })
                out_idx += 1

    augmented_df = pd.DataFrame(metadata_rows)
    augmented_df.to_csv(AUGMENTED_METADATA_PATH, index=False)
    print("Dataset augmentato pronto.")
    print(augmented_df["label"].value_counts())


# ══════════════════════════════════════════════════════════════════════════════
# ── CELLA 6 — Caricamento immagini in-memory ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def load_images_from_df(df, path_col, img_size=IMG_SIZE, desc="", base_dir=PROJECT_ROOT):
    """Carica in RAM tutte le immagini elencate nel dataframe, normalizzandole in scala di grigi su [-1, 1] (range atteso dal VAE) e impacchettandole con le rispettive label."""
    images_list, labels_list = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        path = Path(row[path_col])
        if not path.is_absolute():
            path = base_dir / path
        img = Image.open(path).convert("L")
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        images_list.append((arr / 127.5 - 1.0).astype(np.float32))
        labels_list.append(int(row["label"]))
        if (i + 1) % 200 == 0 or i == 0:
            print(f"  {desc}: {i+1}/{len(df)}...")
    images = np.stack(images_list)[:, :, :, np.newaxis]
    labels = np.array(labels_list, dtype=np.int32)
    print(f"  {desc}: Shape={images.shape}, RAM={images.nbytes/1e6:.0f}MB")
    print(f"  {desc}: pos={int((labels==1).sum())}, neg={int((labels==0).sum())}")
    return images, labels


if ARGS.skip_latent_encoding:
    print("\n── Caricamento immagini skippato ──")
    print("I latenti verranno caricati direttamente da", LATENTS_DIR)
    x_train = x_val = None
    y_train = y_val = None
else:
    print("\n── Caricamento immagini ──")
    print("Caricamento train (reali + augmentate positive)...")
    x_train, y_train = load_images_from_df(augmented_df, "file_name", desc="train")

    print("\nCaricamento val...")
    x_val, y_val = load_images_from_df(val_df, "processed_path", desc="val")


# ══════════════════════════════════════════════════════════════════════════════
# ── ARCHITETTURA VAE (per encoding) ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def build_vae_encoder():
    """Ricostruisce l'architettura dell'encoder VAE (stessa di train_vae.py) usata solo per definire la struttura prima di caricare i pesi già addestrati."""
    x = inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS), name="enc_input")
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.GroupNormalization(groups=32)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = layers.GroupNormalization(groups=32)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.GroupNormalization(groups=32)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = layers.GroupNormalization(groups=32)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.GroupNormalization(groups=32)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.GroupNormalization(groups=32)(x)
    x = layers.LeakyReLU(0.2)(x)
    x      = layers.Conv2D(128, 3, padding="same")(x)
    x      = layers.GroupNormalization(groups=32)(x)
    x      = layers.LeakyReLU(0.2)(x)
    params = layers.Conv2D(LATENT_CHANNELS * 2, 1, padding="same", name="enc_params")(x)
    return tf.keras.Model(inp, params, name="vae_encoder")


def build_vae_decoder():
    """Ricostruisce l'architettura del decoder VAE (stessa di train_vae.py), usata in questo script solo per riportare i latenti generati nello spazio immagine durante eventuali controlli visivi."""
    x = inp = layers.Input(shape=(LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS), name="dec_input")
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.GroupNormalization(groups=32)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = layers.GroupNormalization(groups=32)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.GroupNormalization(groups=32)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = layers.GroupNormalization(groups=32)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.GroupNormalization(groups=32)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = layers.GroupNormalization(groups=32)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.GroupNormalization(groups=32)(x)
    x = layers.LeakyReLU(0.2)(x)
    out = layers.Conv2D(CHANNELS, 3, padding="same", activation="tanh", name="dec_output")(x)
    return tf.keras.Model(inp, out, name="vae_decoder")


# ══════════════════════════════════════════════════════════════════════════════
# ── CODIFICA/CARICAMENTO LATENTI + CLEANUP VRAM (Fix 1 & 2) ─────────────────
# ══════════════════════════════════════════════════════════════════════════════
def encode_dataset_to_latents(images, labels, batch_size=32, desc=""):
    """Passa le immagini nel VAE encoder a batch e tiene solo la media mu della distribuzione latente (niente sampling), così la LDM si addestra su latenti deterministici invece che rumorosi."""
    all_latents = []
    n = len(images)
    for i in range(0, n, batch_size):
        batch = tf.constant(images[i:i+batch_size])
        params = vae_encoder(batch, training=False)
        mu, _  = tf.split(params, 2, axis=-1)
        all_latents.append(mu.numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  {desc}: {min(i+batch_size, n)}/{n}...")
    latents = np.concatenate(all_latents, axis=0)
    print(f"  {desc}: shape={latents.shape}, RAM={latents.nbytes/1e6:.0f}MB")
    return latents


def file_sha256(path: Path) -> str:
    """Calcola l'hash SHA-256 di un file leggendolo a blocchi, usato per rilevare se un asset (VAE, metadata) è cambiato rispetto alla cache dei latenti."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_latents_signature() -> dict:
    """Costruisce la firma di invalidazione della cache dei latenti."""
    # Tutto cio' che, se cambia, rende i latenti in cache non piu' validi:
    # pesi del VAE encoder, contenuto del dataset augmentato, dimensione val.
    return {
        "schema_version": 1,
        "vae_encoder_sha256": file_sha256(VAE_ENCODER_PATH),
        "augmented_metadata_sha256": file_sha256(AUGMENTED_METADATA_PATH),
        "val_csv_sha256": file_sha256(VAL_CSV_PATH),
        "n_train": int(len(augmented_df)),
        "n_val": int(len(val_df)),
        "img_size": IMG_SIZE,
        "latent_size": LATENT_SIZE,
        "latent_channels": LATENT_CHANNELS,
        "positive_augment_copies": POSITIVE_AUGMENT_COPIES,
    }


def load_latents_manifest() -> Optional[dict]:
    """Legge la firma salvata insieme ai latenti in cache, restituendo None se il file manca o è corrotto (forza il ricalcolo)."""
    if not LATENTS_MANIFEST_PATH.exists():
        return None
    try:
        with open(LATENTS_MANIFEST_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


print("\n── Latenti LDM ──")
vae_encoder = None
vae_decoder = None

if ARGS.skip_latent_encoding:
    missing_latent_files = [
        path for path in [LATENTS_TRAIN_PATH, LATENTS_VAL_PATH, LATENT_STATS_PATH]
        if not path.exists()
    ]
    if missing_latent_files:
        raise FileNotFoundError(
            "Latenti precomputati mancanti. Esegui prima prepare_sdvae_latents.py: "
            + ", ".join(str(path) for path in missing_latent_files)
        )
    print("Carico latenti precomputati senza VAE Keras.")
    _d = np.load(str(LATENTS_TRAIN_PATH))
    z_train, y_train = _d["latents"], _d["labels"]
    _d = np.load(str(LATENTS_VAL_PATH))
    z_val, y_val = _d["latents"], _d["labels"]
    _s = np.load(str(LATENT_STATS_PATH))
    LATENT_MEAN, LATENT_STD = _s["latent_mean"], _s["latent_std"]
    z_train_norm = (z_train - LATENT_MEAN) / LATENT_STD
    z_val_norm   = (z_val   - LATENT_MEAN) / LATENT_STD
    print(f"z_train_norm: mean={z_train_norm.mean():.4f}, std={z_train_norm.std():.4f}")
else:
    print("\n── Caricamento VAE best ──")
    _vae_enc_path = VAE_ENCODER_PATH
    _vae_dec_path = VAE_DECODER_PATH

    if not _vae_enc_path.exists() or not _vae_dec_path.exists():
        raise FileNotFoundError(
            "VAE best non trovato. Copia vae_encoder_best.keras e vae_decoder_best.keras "
            f"in {MODELS_DIR}."
        )

    vae_encoder = tf.keras.models.load_model(str(_vae_enc_path))
    vae_decoder = tf.keras.models.load_model(str(_vae_dec_path))
    vae_encoder.trainable = False
    vae_decoder.trainable = False
    print(f"VAE encoder: {vae_encoder.count_params():,} params (frozen)")
    print(f"VAE decoder: {vae_decoder.count_params():,} params (frozen)")

    current_latents_signature = build_latents_signature()
    cached_latents_signature = load_latents_manifest()
    latents_cache_valid = (
        LATENTS_TRAIN_PATH.exists()
        and LATENTS_VAL_PATH.exists()
        and LATENT_STATS_PATH.exists()
        and cached_latents_signature == current_latents_signature
    )

    if latents_cache_valid:
        print("Latenti già presenti su disco e coerenti con VAE/dataset attuali — caricamento...")
        _d = np.load(str(LATENTS_TRAIN_PATH))
        z_train, y_train = _d["latents"], _d["labels"]
        _d = np.load(str(LATENTS_VAL_PATH))
        z_val, y_val = _d["latents"], _d["labels"]
        _s = np.load(str(LATENT_STATS_PATH))
        LATENT_MEAN, LATENT_STD = _s["latent_mean"], _s["latent_std"]
        z_train_norm = (z_train - LATENT_MEAN) / LATENT_STD
        z_val_norm   = (z_val   - LATENT_MEAN) / LATENT_STD
        print(f"z_train_norm: mean={z_train_norm.mean():.4f}, std={z_train_norm.std():.4f}")
    else:
        if LATENTS_TRAIN_PATH.exists() and cached_latents_signature != current_latents_signature:
            print("Cache latenti presente ma non coerente con VAE/dataset attuali (hash diverso): ricalcolo.")
        print("Codifica train in latenti...")
        z_train = encode_dataset_to_latents(
            x_train,
            y_train,
            batch_size=VAE_BATCH_SIZE,
            desc="train",
        )
        np.savez_compressed(str(LATENTS_TRAIN_PATH), latents=z_train, labels=y_train)
        print(f"  Salvati in {LATENTS_TRAIN_PATH}")

        print("\nCodifica val in latenti...")
        z_val = encode_dataset_to_latents(
            x_val,
            y_val,
            batch_size=VAE_BATCH_SIZE,
            desc="val",
        )
        np.savez_compressed(str(LATENTS_VAL_PATH), latents=z_val, labels=y_val)
        print(f"  Salvati in {LATENTS_VAL_PATH}")

        LATENT_MEAN = z_train.mean(axis=(0, 1, 2), keepdims=True)
        LATENT_STD  = z_train.std(axis=(0, 1, 2),  keepdims=True)
        z_train_norm = (z_train - LATENT_MEAN) / LATENT_STD
        z_val_norm   = (z_val   - LATENT_MEAN) / LATENT_STD
        np.savez(str(LATENT_STATS_PATH), latent_mean=LATENT_MEAN, latent_std=LATENT_STD)
        print(f"  Stats salvate in {LATENT_STATS_PATH}")

        with open(LATENTS_MANIFEST_PATH, "w", encoding="utf-8") as file:
            json.dump(current_latents_signature, file, indent=2)
        print(f"  Manifest latenti salvato in {LATENTS_MANIFEST_PATH}")

# Libera memoria: x_train/x_val e VAE non servono più
if x_train is not None:
    del x_train
if x_val is not None:
    del x_val
if vae_encoder is not None:
    del vae_encoder
if vae_decoder is not None:
    del vae_decoder
gc.collect()
try:
    info = tf.config.experimental.get_memory_info("GPU:0")
    print(f"VRAM post-encoding cleanup: {info['current']/1e9:.2f} GB")
except Exception:
    pass
print("Cleanup post-encoding OK. Da qui: solo latenti numpy.")


# ══════════════════════════════════════════════════════════════════════════════
# ── COSINE SCHEDULE ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def normal_kl(mean1, logvar1, mean2, logvar2):
    """Calcola la divergenza KL in forma chiusa tra due gaussiane diagonali, usata nel termine VLB per confrontare la posterior vera con quella predetta dalla U-Net."""
    return 0.5 * (
        logvar2 - logvar1
        + tf.exp(logvar1 - logvar2)
        + tf.square(mean1 - mean2) * tf.exp(-logvar2)
        - 1.0
    )


def cosine_schedule(num_steps, s=0.008):
    """Genera i beta con NumPy per evitare kernel TF/XLA inutili all'avvio."""
    t = np.linspace(0.0, float(num_steps), num_steps + 1, dtype=np.float64)
    f = np.cos((t / float(num_steps) + s) / (1.0 + s) * (math.pi / 2.0)) ** 2
    alpha_bars_np = f / f[0]
    betas_np = 1.0 - alpha_bars_np[1:] / alpha_bars_np[:-1]
    betas_np = np.clip(betas_np, 1e-4, 0.9999).astype(np.float32)
    return tf.constant(betas_np, dtype=tf.float32)


betas      = cosine_schedule(NUM_DIFF_STEPS)
alphas     = 1.0 - betas
alpha_bars = tf.math.cumprod(alphas)

sqrt_alpha_bars           = tf.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = tf.sqrt(1.0 - alpha_bars)

alpha_bars_prev        = tf.concat([[1.0], alpha_bars[:-1]], axis=0)
posterior_variance     = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
posterior_log_variance = tf.math.log(tf.maximum(posterior_variance, 1e-20))

print(f"\nSchedule OK. alpha_bar finale: {alpha_bars[-1].numpy():.6f}")


def extract(values, t, x_shape):
    """Seleziona dal vettore di coefficienti dello schedule (es. beta, alpha_bar) il valore corrispondente al timestep t di ciascun elemento del batch, e lo riporta a shape broadcastabile con i latenti."""
    batch_size = tf.shape(t)[0]
    out        = tf.gather(values, t)
    return tf.reshape(out, [batch_size, 1, 1, 1])


def q_sample(x0, t, noise):
    """Forward diffusion: applica direttamente la formula chiusa per ottenere il latente rumoroso al timestep t a partire da x0 e dal rumore campionato, senza dover iterare passo-passo."""
    sqrt_ab   = extract(sqrt_alpha_bars,           t, tf.shape(x0))
    sqrt_omab = extract(sqrt_one_minus_alpha_bars, t, tf.shape(x0))
    return sqrt_ab * x0 + sqrt_omab * noise


# ══════════════════════════════════════════════════════════════════════════════
# ── U-NET LDM ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
@tf.keras.utils.register_keras_serializable()
class SinusoidalTimeEmbedding(layers.Layer):
    """Trasforma lo scalare timestep t in un embedding continuo (encoding sinusoidale + MLP), così la U-Net può condizionare l'output sul livello di rumore corrente."""
    def __init__(self, embed_dim, **kwargs):
        """Istanzia le due Dense che proiettano l'encoding sinusoidale nello spazio di embedding finale."""
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        half = embed_dim // 2
        frequencies = np.exp(
            -math.log(10000.0) * np.arange(half, dtype=np.float32) / float(half)
        )
        self._frequencies = tf.constant(frequencies, dtype=tf.float32)
        self.dense1 = layers.Dense(embed_dim * 4, activation="relu")
        self.dense2 = layers.Dense(embed_dim * 4)

    def call(self, t):
        """Calcola le frequenze sinusoidali (stile positional encoding dei Transformer) per ogni t del batch e le passa nell'MLP a due livelli."""
        t    = tf.cast(t, tf.float32)
        freqs = tf.cast(self._frequencies, tf.float32)
        args = t[:, None] * freqs[None, :]
        emb  = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        return self.dense2(self.dense1(emb))

    def get_config(self):
        """Aggiunge embed_dim alla config, necessario per ricostruire il layer al caricamento del checkpoint .keras."""
        cfg = super().get_config()
        cfg.update({"embed_dim": self.embed_dim})
        return cfg


@tf.keras.utils.register_keras_serializable()
class LabelEmbedding(layers.Layer):
    """Mappa la classe (0=sano, 1=cancro, più una classe extra per il condizionamento nullo della CFG) in un vettore di embedding da sommare al time embedding."""
    def __init__(self, num_classes, embed_dim, **kwargs):
        """Crea la tabella di embedding con num_classes+1 voci: la voce in più rappresenta la label 'nulla' usata dal classifier-free guidance dropout."""
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        self.embedding   = layers.Embedding(num_classes + 1, embed_dim * 4)

    def call(self, y):
        """Restituisce l'embedding corrispondente alla label y."""
        return self.embedding(y)

    def get_config(self):
        """Aggiunge num_classes ed embed_dim alla config per la serializzazione del layer nel checkpoint .keras."""
        cfg = super().get_config()
        cfg.update({"num_classes": self.num_classes, "embed_dim": self.embed_dim})
        return cfg


@tf.keras.utils.register_keras_serializable()
class ResBlock(layers.Layer):
    """Blocco residuo della U-Net: due convoluzioni con GroupNorm che iniettano l'embedding (tempo+label) a metà, più uno skip 1x1 per cambiare il numero di canali senza perdere il segnale originale."""
    def __init__(self, channels, embed_dim, **kwargs):
        """Crea i sotto-layer del blocco (norm/conv per il ramo principale, proiezione dell'embedding, conv 1x1 di skip)."""
        super().__init__(**kwargs)
        self.channels  = channels
        self.embed_dim = embed_dim
        self.norm1    = layers.GroupNormalization(groups=min(32, channels))
        self.conv1    = layers.Conv2D(channels, 3, padding="same")
        self.norm2    = layers.GroupNormalization(groups=min(32, channels))
        self.conv2    = layers.Conv2D(channels, 3, padding="same")
        self.emb_proj = layers.Dense(channels)
        self.skip     = layers.Conv2D(channels, 1)
        self.act      = layers.LeakyReLU(alpha=0.2)

    def call(self, x, emb):
        """Applica le due convoluzioni del ramo principale, somma l'embedding (tempo+label) proiettato sui canali a metà del blocco, poi aggiunge la connessione residua."""
        h       = self.conv1(self.act(self.norm1(x)))
        emb_out = tf.reshape(self.emb_proj(self.act(emb)), [-1, 1, 1, self.channels])
        h       = self.conv2(self.act(self.norm2(h + emb_out)))
        return h + self.skip(x)

    def get_config(self):
        """Aggiunge channels ed embed_dim alla config per la serializzazione del layer nel checkpoint .keras."""
        cfg = super().get_config()
        cfg.update({"channels": self.channels, "embed_dim": self.embed_dim})
        return cfg


@tf.keras.utils.register_keras_serializable()
class SelfAttentionBlock(layers.Layer):
    """Blocco di self-attention sulle feature map a bassa risoluzione della U-Net, per catturare dipendenze spaziali a lungo raggio che le convoluzioni locali non vedono."""
    def __init__(self, channels, num_heads=4, **kwargs):
        """Crea normalizzazione, multi-head attention e proiezione finale del blocco."""
        super().__init__(**kwargs)
        self.channels  = channels
        self.num_heads = num_heads
        self.norm = layers.GroupNormalization(groups=min(32, channels))
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=channels // num_heads,
            value_dim=channels // num_heads,
        )
        self.proj = layers.Dense(channels)

    def call(self, x):
        """Appiattisce la feature map spaziale (H×W) in una sequenza di token, applica self-attention tra tutte le posizioni e riporta il risultato alla shape originale con connessione residua."""
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = self.channels
        h = self.norm(x)
        h = tf.reshape(h, [B, H * W, C])
        h = self.attn(h, h)
        h = self.proj(h)
        h = tf.reshape(h, [B, H, W, C])
        return x + h

    def get_config(self):
        """Aggiunge channels e num_heads alla config per la serializzazione del layer nel checkpoint .keras."""
        cfg = super().get_config()
        cfg.update({"channels": self.channels, "num_heads": self.num_heads})
        return cfg


def build_ldm_unet():
    """Costruisce la U-Net di diffusione: encoder-decoder simmetrico con skip connection in stile U-Net, ResBlock condizionati su tempo+label e self-attention nei livelli più profondi (dove le feature map sono piccole e l'attention costa poco)."""
    C = MODEL_CHANNELS * 2  # 128

    lat_input = layers.Input(shape=(LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS), name="lat_input")
    t_input   = layers.Input(shape=(), dtype=tf.int32, name="t_input")
    y_input   = layers.Input(shape=(), dtype=tf.int32, name="y_input")

    time_emb  = SinusoidalTimeEmbedding(EMBED_DIM)(t_input)
    label_emb = LabelEmbedding(NUM_CLASSES, EMBED_DIM)(y_input)
    emb       = time_emb + label_emb

    x = layers.Conv2D(C, 3, padding="same")(lat_input)

    x1 = ResBlock(C,   EMBED_DIM)(x,  emb)
    x1 = ResBlock(C,   EMBED_DIM)(x1, emb)
    p1 = layers.Conv2D(C, 3, strides=2, padding="same")(x1)      # 32×32

    x2 = ResBlock(C*2, EMBED_DIM)(p1, emb)
    x2 = ResBlock(C*2, EMBED_DIM)(x2, emb)
    p2 = layers.Conv2D(C*2, 3, strides=2, padding="same")(x2)    # 16×16

    x3 = ResBlock(C*4, EMBED_DIM)(p2, emb)
    x3 = SelfAttentionBlock(C*4, num_heads=4)(x3)
    x3 = ResBlock(C*4, EMBED_DIM)(x3, emb)
    p3 = layers.Conv2D(C*4, 3, strides=2, padding="same")(x3)    # 8×8

    b = ResBlock(C*4, EMBED_DIM)(p3, emb)
    b = SelfAttentionBlock(C*4, num_heads=4)(b)
    b = ResBlock(C*4, EMBED_DIM)(b,  emb)
    b = SelfAttentionBlock(C*4, num_heads=4)(b)

    u3 = layers.Conv2DTranspose(C*4, 3, strides=2, padding="same")(b)
    u3 = layers.Concatenate()([u3, x3])
    u3 = ResBlock(C*4, EMBED_DIM)(u3, emb)
    u3 = SelfAttentionBlock(C*4, num_heads=4)(u3)
    u3 = ResBlock(C*4, EMBED_DIM)(u3, emb)

    u2 = layers.Conv2DTranspose(C*2, 3, strides=2, padding="same")(u3)
    u2 = layers.Concatenate()([u2, x2])
    u2 = ResBlock(C*2, EMBED_DIM)(u2, emb)
    u2 = ResBlock(C*2, EMBED_DIM)(u2, emb)

    u1 = layers.Conv2DTranspose(C, 3, strides=2, padding="same")(u2)
    u1 = layers.Concatenate()([u1, x1])
    u1 = ResBlock(C, EMBED_DIM)(u1, emb)
    u1 = ResBlock(C, EMBED_DIM)(u1, emb)

    out = layers.GroupNormalization(groups=min(32, C))(u1)
    out = layers.Conv2D(LATENT_CHANNELS * 2, 3, padding="same")(out)

    return tf.keras.Model([lat_input, t_input, y_input], out, name="ldm_unet_64px")


print("\n── Build LDM U-Net ──")
if UNET_VERSION == "v3":
    ldm_model = build_ldm_unet_v3(
        latent_size=LATENT_SIZE,
        latent_channels=LATENT_CHANNELS,
        model_channels=MODEL_CHANNELS,
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
    )
else:
    ldm_model = build_ldm_unet()
print(f"LDM U-Net params: {ldm_model.count_params():,} (unet_version={UNET_VERSION})")


# ══════════════════════════════════════════════════════════════════════════════
# ── LOSS + TRAINING STEP ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def get_learned_log_variance(v, t):
    """Interpola tra la varianza minima e massima ammesse per il timestep t (log_beta e log_beta_tilde) usando v come peso appreso dalla rete, così la varianza della reverse diffusion non è fissa ma viene predetta dal modello (come in 'Improved DDPM')."""
    shape            = tf.shape(v)
    log_beta_t       = tf.math.log(extract(betas, t, shape))
    log_beta_tilde_t = tf.math.log(extract(posterior_variance + 1e-8, t, shape))
    v_sigmoid        = tf.sigmoid(tf.clip_by_value(v, -8.0, 8.0))
    log_var          = v_sigmoid * log_beta_t + (1.0 - v_sigmoid) * log_beta_tilde_t
    return tf.clip_by_value(log_var, -20.0, 2.0)


def vb_term_ldm(x0, x_t, t, eps_pred, v_pred):
    """Calcola il termine variational lower bound (KL tra posterior vera e posterior predetta dalla rete) che, pesato da LAMBDA_VLB, affianca la loss semplice sul rumore per far apprendere anche la varianza della reverse diffusion."""
    shape     = tf.shape(x0)
    ab        = extract(alpha_bars,       t, shape)
    ab_prev   = extract(alpha_bars_prev,  t, shape)
    b         = extract(betas,            t, shape)
    alpha_t   = extract(alphas,           t, shape)
    sqrt_ab   = extract(sqrt_alpha_bars,  t, shape)
    sqrt_omab = extract(sqrt_one_minus_alpha_bars, t, shape)

    mean_true    = (tf.sqrt(ab_prev) * b / (1.0 - ab)) * x0 + \
                   (tf.sqrt(alpha_t) * (1.0 - ab_prev) / (1.0 - ab)) * x_t
    log_var_true = extract(posterior_log_variance, t, shape) * tf.ones_like(x0)

    x0_pred   = tf.clip_by_value((x_t - sqrt_omab * eps_pred) / sqrt_ab, -3.0, 3.0)
    mean_pred = (tf.sqrt(ab_prev) * b / (1.0 - ab)) * x0_pred + \
                (tf.sqrt(alpha_t) * (1.0 - ab_prev) / (1.0 - ab)) * x_t
    log_var_pred = get_learned_log_variance(v_pred, t)

    kl = tf.clip_by_value(
        normal_kl(mean_true, log_var_true, mean_pred, log_var_pred), 0.0, 50.0
    )
    return tf.reduce_mean(kl) / math.log(2.0)


ldm_optimizer = tf.keras.optimizers.Adam(LDM_LR)


@tf.function(jit_compile=False, reduce_retracing=True)
def ldm_train_step(z0, y):
    """Singolo step di training: campiona un timestep e un rumore casuali, applica la forward diffusion al latente, droppa la label con probabilità CFG_DROPOUT (per insegnare alla rete anche il caso non condizionato), e aggiorna i pesi sulla loss combinata (target primario + VLB).

    Il target della loss_simple dipende da PARAMETERIZATION: rumore puro per "eps"
    (comportamento invariato rispetto a 04b/04b1/04b2), oppure il target v di
    Salimans & Ho per "v". La loss_vlb (che modella la varianza appresa) lavora
    sempre in spazio epsilon: se PARAMETERIZATION="v" l'uscita primaria della rete
    viene prima convertita in una stima di epsilon con predict_epsilon_from_model_output,
    la stessa funzione usata da generazione e valutazione."""
    batch_size = tf.shape(z0)[0]
    t         = tf.random.uniform((batch_size,), 0, NUM_DIFF_STEPS, dtype=tf.int32)
    noise     = tf.random.normal(tf.shape(z0))
    z_t       = q_sample(z0, t, noise)

    cfg_mask  = tf.cast(tf.random.uniform((batch_size,)) > CFG_DROPOUT, tf.int32)
    y_dropped = y * cfg_mask + NUM_CLASSES * (1 - cfg_mask)

    if PARAMETERIZATION == "v":
        target = make_v_target(z0, noise, t, sqrt_alpha_bars, sqrt_one_minus_alpha_bars)
    else:
        target = noise

    with tf.GradientTape() as tape:
        out                   = ldm_model([z_t, t, y_dropped], training=True)
        primary_pred, v_pred  = tf.split(out, 2, axis=-1)
        squared_error         = tf.square(target - primary_pred)
        if USE_MIN_SNR:
            weight       = min_snr_weight(t, alpha_bars, gamma=MIN_SNR_GAMMA, parameterization=PARAMETERIZATION)
            loss_simple  = tf.reduce_mean(weight * squared_error)
        else:
            loss_simple  = tf.reduce_mean(squared_error)

        if PARAMETERIZATION == "v":
            ab_t              = extract(alpha_bars, t, tf.shape(z0))
            eps_pred_for_vlb  = predict_epsilon_from_model_output(primary_pred, z_t, ab_t, PARAMETERIZATION)
        else:
            eps_pred_for_vlb  = primary_pred

        loss_vlb  = vb_term_ldm(z0, z_t, t, eps_pred_for_vlb, v_pred)
        loss      = loss_simple + LAMBDA_VLB * loss_vlb

    grads = tape.gradient(loss, ldm_model.trainable_variables)
    ldm_optimizer.apply_gradients(zip(grads, ldm_model.trainable_variables))
    return loss, loss_simple, loss_vlb


# ══════════════════════════════════════════════════════════════════════════════
# ── DATASET tf.data (latenti) ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
n_total = len(y_train)
print(f"\nLatenti train: {n_total} | pos={int((y_train==1).sum())}, neg={int((y_train==0).sum())}")

ldm_train_ds = (
    tf.data.Dataset.from_tensor_slices(
        (tf.constant(z_train_norm, dtype=tf.float32), tf.constant(y_train))
    )
    .shuffle(n_total, reshuffle_each_iteration=True)
    .batch(LDM_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

for batch_z, batch_y in ldm_train_ds.take(1):
    print(f"LDM batch shape: {batch_z.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# ── LDM TRAINING (step-based) ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
best_ldm_loss = float("inf")
global_step   = 0

ldm_history = {
    "step": [], "loss_total": [], "loss_simple": [], "loss_vlb": [],
}
_history_path = LOGS_DIR / "ldm_history.json"

print("=" * 70)
print("FASE 2 -- TRAINING LDM")
print(f"Step totali     : {LDM_TOTAL_STEPS}")
print(f"Batch size      : {LDM_BATCH_SIZE}")
print(f"Learning rate   : {LDM_LR}")
print(f"Checkpoint ogni : {CKPT_EVERY} step")
print(f"Log ogni        : {LOG_EVERY} step")
print(f"Latenti         : {LATENT_SIZE}×{LATENT_SIZE}×{LATENT_CHANNELS}")
print(f"Checkpoint dir  : {CKPT_DIR}")
print("=" * 70)
sys.stdout.flush()

# Controlla se esiste già un final model o un checkpoint da cui riprendere
def step_from_final_model(path: Path) -> Optional[int]:
    """Estrae il numero di step dal nome file di un modello finale (es. 'ldm_unet_final_step080000.keras' -> 80000)."""
    prefix = "ldm_unet_final_step"
    if not path.stem.startswith(prefix):
        return None
    try:
        return int(path.stem.replace(prefix, ""))
    except ValueError:
        return None


def step_from_checkpoint(path: Path) -> Optional[int]:
    """Estrae il numero di step dal nome file di un checkpoint periodico (es. 'ldm_step007000.keras' -> 7000)."""
    prefix = "ldm_step"
    if not path.stem.startswith(prefix):
        return None
    try:
        return int(path.stem.replace(prefix, ""))
    except ValueError:
        return None


def latest_step_checkpoint() -> tuple[Optional[int], Optional[Path]]:
    """Cerca tra i checkpoint periodici quello con lo step piu' alto, per decidere da dove riprendere il training (duplica latest_step_checkpoint_path definita sopra, qui usata dopo il backfill del layout)."""
    candidates = []
    for path in CKPT_DIR.glob("ldm_step*.keras"):
        step = step_from_checkpoint(path)
        if step is not None:
            candidates.append((step, path))
    return max(candidates, default=(None, None), key=lambda item: item[0] or -1)


def load_existing_history() -> dict:
    """Ricarica la history delle loss da un training precedente (se presente e leggibile), per continuare i grafici e il calcolo della best loss dopo un resume."""
    if not _history_path.exists():
        return {"step": [], "loss_total": [], "loss_simple": [], "loss_vlb": []}
    try:
        with open(_history_path, "r", encoding="utf-8") as file:
            loaded = json.load(file)
    except Exception as exc:
        print(f"History non leggibile, riparto con history vuota: {exc}")
        return {"step": [], "loss_total": [], "loss_simple": [], "loss_vlb": []}

    history = {"step": [], "loss_total": [], "loss_simple": [], "loss_vlb": []}
    for key in history:
        values = loaded.get(key, [])
        history[key] = values if isinstance(values, list) else []
    return history


def backfill_checkpoint_layout_from_existing_models() -> None:
    """Migra i modelli LDM salvati nella vecchia struttura (cartella models/) verso checkpoints_ldm/, ricostruendo anche il checkpoint 'latest' a partire dal final model se manca, per restare compatibili con esperimenti lanciati prima di questa riorganizzazione."""
    legacy_best_path = MODELS_DIR / "ldm_unet_best.keras"
    best_ckpt_path = CKPT_DIR / "ldm_unet_best.keras"
    if legacy_best_path.exists() and not best_ckpt_path.exists():
        shutil.move(str(legacy_best_path), str(best_ckpt_path))
        print(f"Best LDM spostato nei checkpoint: {best_ckpt_path}")
    elif legacy_best_path.exists():
        legacy_best_path.unlink()
        print(f"Duplicato LDM rimosso da models: {legacy_best_path}")

    for legacy_final_path in sorted(MODELS_DIR.glob("ldm_unet_final_step*.keras")):
        final_ckpt_path = CKPT_DIR / legacy_final_path.name
        if not final_ckpt_path.exists():
            shutil.move(str(legacy_final_path), str(final_ckpt_path))
            print(f"Final LDM spostato nei checkpoint: {final_ckpt_path}")
        else:
            legacy_final_path.unlink()
            print(f"Duplicato final LDM rimosso da models: {legacy_final_path}")

    final_models = sorted(CKPT_DIR.glob("ldm_unet_final_step*.keras"))
    if not final_models:
        return
    final_model_path = final_models[-1]
    final_step = step_from_final_model(final_model_path)
    if final_step is None:
        return
    latest_ckpt_path = CKPT_DIR / f"ldm_step{final_step:06d}.keras"
    if not latest_ckpt_path.exists():
        shutil.copy2(final_model_path, latest_ckpt_path)
        print(f"Latest checkpoint creato da final model: {latest_ckpt_path}")


backfill_checkpoint_layout_from_existing_models()
existing_finals = sorted(CKPT_DIR.glob("ldm_unet_final_step*.keras"))
latest_ckpt_step, latest_ckpt_path = latest_step_checkpoint()

if ARGS.resume_from_latest and latest_ckpt_path is not None:
    if latest_ckpt_step is not None and latest_ckpt_step >= LDM_TOTAL_STEPS:
        print(
            f"\nCheckpoint gia' al target: {latest_ckpt_path.name} "
            f"(step {latest_ckpt_step} >= {LDM_TOTAL_STEPS})."
        )
        print("Training LDM skippato.")
        sync_existing_training_plots_to_results()
        write_training_manifest(latest_ckpt_path, latest_ckpt_step)
        sys.exit(0)

    print(f"\nRiprendo training da checkpoint: {latest_ckpt_path}")
    ldm_model = tf.keras.models.load_model(str(latest_ckpt_path), compile=False)
    global_step = int(latest_ckpt_step or 0)
    ldm_history = load_existing_history()
    if ldm_history["loss_total"]:
        best_ldm_loss = float(np.nanmin(ldm_history["loss_total"]))
    print(f"Global step iniziale: {global_step}")
    print(f"Best loss caricata da history: {best_ldm_loss:.4f}")
elif ARGS.resume_from_latest:
    print("\nFlag --resume-from-latest attivo, ma nessun ldm_step*.keras trovato.")
    print("Training LDM avviato da zero.")
elif existing_finals:
    print(f"\nFinal model già presente: {existing_finals[-1]}")
    print("Training LDM skippato. Usa --resume-from-latest con --total-steps maggiore per continuare.")
    sync_existing_training_plots_to_results()
    write_training_manifest(
        existing_finals[-1],
        step_from_model_path(existing_finals[-1], "ldm_unet_final_step"),
    )
    sys.exit(0)


start_global_step = global_step
t_start = time.time()
ds_iter = iter(ldm_train_ds.repeat())

with measure_sustainability(label="ldm_training", sample_interval=0.5) as eco_ldm:
    while global_step < LDM_TOTAL_STEPS:
        batch_z, batch_y = next(ds_iter)
        loss, ls, lv     = ldm_train_step(batch_z, batch_y)
        global_step += 1

        loss_val = float(loss.numpy())
        ls_val   = float(ls.numpy())
        lv_val   = float(lv.numpy())

        if global_step % LOG_EVERY == 0 or global_step == start_global_step + 1:
            elapsed = time.time() - t_start
            steps_this_run = max(1, global_step - start_global_step)
            eta_s = elapsed / steps_this_run * (LDM_TOTAL_STEPS - global_step)
            msg = (
                f"[{global_step:06d}/{LDM_TOTAL_STEPS}] "
                f"loss={loss_val:.4f} simple={ls_val:.4f} vlb={lv_val:.6f} "
                f"ETA={int(eta_s//3600):02d}h{int((eta_s%3600)//60):02d}m"
            )
            print(msg)
            sys.stdout.flush()
            ldm_history["step"].append(global_step)
            ldm_history["loss_total"].append(loss_val)
            ldm_history["loss_simple"].append(ls_val)
            ldm_history["loss_vlb"].append(lv_val)

        # best_ldm_loss resta solo una statistica informativa (loss minima vista su un
        # singolo batch, rumorosa): la selezione del checkpoint migliore si fa dopo, sul
        # validation set, con lo sweep FID/PRDC di evaluate_ldm.py — non qui salvando
        # un modello ogni volta che un batch fortunato abbassa la loss.
        best_ldm_loss = min(best_ldm_loss, loss_val)

        if global_step % CKPT_EVERY == 0:
            ckpt_path = CKPT_DIR / f"ldm_step{global_step:06d}.keras"
            ldm_model.save(str(ckpt_path))
            print(f"\n{'='*70}\n[CHECKPOINT] STEP {global_step} — {ckpt_path.name}\n{'='*70}\n")
            sys.stdout.flush()

final_path = CKPT_DIR / f"ldm_unet_final_step{global_step:06d}.keras"
ldm_model.save(str(final_path))
latest_ckpt_path = CKPT_DIR / f"ldm_step{global_step:06d}.keras"
if not latest_ckpt_path.exists():
    shutil.copy2(final_path, latest_ckpt_path)
print(f"\nTRAINING LDM COMPLETATO — Step: {global_step} | Best loss: {best_ldm_loss:.4f}")
print(f"Final model: {final_path.name}")
print(f"Latest checkpoint: {latest_ckpt_path.name}")
print(f"\n{eco_ldm.metrics}")
sys.stdout.flush()

write_training_manifest(final_path, global_step)

# ── Salva history JSON ────────────────────────────────────────────────────────
_history_path = LOGS_DIR / "ldm_history.json"
with open(_history_path, "w", encoding="utf-8") as _f:
    json.dump(ldm_history, _f, indent=2)
print(f"History salvata: {_history_path}")

# ── Eco log ───────────────────────────────────────────────────────────────────
_ldm_eco = eco_ldm.metrics.to_dict()
_ldm_eco.update({
    "phase": "ldm_training", "total_steps": global_step,
    "best_loss": best_ldm_loss, "batch_size": LDM_BATCH_SIZE, "lr": LDM_LR,
})
_eco_log = RESULTS_ECOTRACKER_DIR / "sustainability_log.jsonl"
with open(_eco_log, "a", encoding="utf-8") as _f:
    _f.write(json.dumps(_ldm_eco, ensure_ascii=False) + "\n")

# ── Plot metriche (salva su file, no GUI) ─────────────────────────────────────
steps = ldm_history["step"]
if steps:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(steps, ldm_history["loss_simple"], label="L_simple", color="blue",  lw=1.2)
    axes[0].plot(steps, ldm_history["loss_total"],  label="L_total",  color="red",   lw=1.2, ls="--")
    axes[0].set_title("LDM — Noise Prediction Loss")
    axes[0].set_xlabel("Global Step")
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    axes[1].plot(steps, ldm_history["loss_vlb"], label="L_vlb", color="orange", lw=1.2)
    axes[1].set_title("LDM — VLB Loss")
    axes[1].set_xlabel("Global Step")
    axes[1].legend(); axes[1].grid(True, alpha=0.4)

    if len(steps) >= 10:
        kernel   = np.ones(10) / 10
        smoothed = np.convolve(ldm_history["loss_total"], kernel, mode="valid")
        axes[2].plot(steps[9:], smoothed, color="purple", lw=1.5, label="L_total smoothed")
        axes[2].set_title("LDM — Loss (smoothed)")
        axes[2].set_xlabel("Global Step")
        axes[2].legend(); axes[2].grid(True, alpha=0.4)
    else:
        axes[2].set_visible(False)

    plt.suptitle("Metriche LDM — training step-based", fontsize=13)
    plt.tight_layout()
    _metrics_path = RESULTS_PLOTS_DIR / "ldm_metrics.png"
    plt.savefig(str(_metrics_path), dpi=150, bbox_inches="tight")
    plt.close()
    best_idx = int(np.argmin(ldm_history["loss_total"]))
    print(f"Best loss: {ldm_history['loss_total'][best_idx]:.4f} @ step {steps[best_idx]}")
    print(f"Plot salvato nei results: {_metrics_path}")

print("\n[train_ldm.py] Script completato con successo.")
