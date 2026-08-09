#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from ldm_project_paths import (
    RESULTS_STAGE_NAME,
    class_name_for_label,
    get_class_image_dirs,
)


IMG_SIZE = 512
CHANNELS = 1
LATENT_SIZE = 64
LATENT_CHANNELS = 4
POSITIVE_AUGMENT_COPIES = 3
VAE_BATCH_SIZE = 8
VAE_EPOCHS = 100
VAE_LR = 1e-4
KL_WEIGHT = 1e-3
SSIM_WEIGHT = 0.3
VAE_ES_PATIENCE = 6
VAE_ES_MIN_DELTA = 1e-4
# Validation MSE from one epoch can be noisy, so this low weight keeps it a
# secondary correction to SSIM when selecting the best model.
MSE_SELECTION_WEIGHT = 0.05
def parse_args() -> argparse.Namespace:
    """Define and parse the command-line arguments used by the notebook subprocess."""
    parser = argparse.ArgumentParser(
        description="Train MammoDiffusion VAE in the selected experiment folder."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--gpu-visible-devices", default=None)
    parser.add_argument(
        "--cuda-root",
        type=Path,
        default=Path(sys.prefix),
    )
    parser.add_argument("--epochs", type=int, default=VAE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=VAE_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=VAE_LR)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--also-reset-downstream", action="store_true")
    parser.add_argument(
        "--results-stage-name",
        default=RESULTS_STAGE_NAME,
        help="Results subdirectory for the VAE sustainability log.",
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


def sustainability_log_path(
    project_root: Path,
    stage_name: str = RESULTS_STAGE_NAME,
) -> Path:
    """Return the JSONL energy-log path, creating parent directories as needed."""
    path = (
        project_root
        / "results"
        / stage_name
        / "ecotracker"
        / "sustainability_log.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def results_plot_path(project_root: Path, stage_name: str, plot_name: str) -> Path:
    """Return a plot path in the stage results directory, creating it as needed."""
    path = project_root / "results" / stage_name / "plots" / plot_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sync_existing_vae_plots_to_results(paths, stage_name: str) -> None:
    """Report whether saved VAE-training plots exist when an existing VAE skips training."""
    for plot_name in ["vae_metrics.png", "vae_reconstruction.png"]:
        destination_path = results_plot_path(paths.project_root, stage_name, plot_name)
        if destination_path.exists():
            print(f"Training plot already present in results: {destination_path}")
            continue
        print(f"VAE training plot not found in results: {destination_path}")


def load_metadata(csv_path: Path, dataset_root: Path) -> pd.DataFrame:
    """Load a split CSV, resolve preprocessed image paths, and require every image to exist."""
    df = pd.read_csv(csv_path).copy()
    required_cols = ["patient_id", "image_id", "label", "split", "processed_path"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns are missing from {csv_path}: {missing_cols}")

    df["patient_id"] = df["patient_id"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    df["label"] = df["label"].astype(int)
    df["split"] = df["split"].astype(str)
    df["filename"] = df["processed_path"].apply(
        lambda p: str(p).replace("\\", "/").split("/")[-1]
    )
    df["processed_path"] = df.apply(
        lambda row: str(dataset_root / row["split"] / str(row["label"]) / row["filename"]),
        axis=1,
    )
    df["file_exists"] = df["processed_path"].apply(lambda p: Path(p).exists())
    missing_files = df[~df["file_exists"]]
    if len(missing_files) > 0:
        print(missing_files[["patient_id", "image_id", "split", "label", "processed_path"]].head(10))
        raise FileNotFoundError(f"{len(missing_files)} preprocessed images not found")
    return df.drop(columns=["file_exists"])


def mild_positive_augmentation(img: Image.Image, aug_idx: int) -> Image.Image:
    """Create reproducible positive augmentations with mild contrast, brightness, and noise changes."""
    arr = np.array(img).astype(np.float32)
    rng = np.random.default_rng(42 + aug_idx)
    contrast = rng.uniform(0.90, 1.10)
    brightness = rng.uniform(-8, 8)
    noise = rng.normal(loc=0.0, scale=2.0, size=arr.shape)
    mean = arr.mean()
    arr = (arr - mean) * contrast + mean
    arr = arr + brightness + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def build_augmented_train_metadata(train_df: pd.DataFrame, project_root: Path, data_aug: Path) -> pd.DataFrame:
    """Build or reuse the shared augmented training set with positive-only copies."""
    metadata_path = data_aug / "metadata.csv"

    def resolve_project_path(path_value: str | Path) -> Path:
        """Resolve a metadata path relative to the project root."""
        path = Path(path_value)
        return path if path.is_absolute() else project_root / path

    if metadata_path.exists():
        try:
            existing_df = pd.read_csv(metadata_path).copy()
        except Exception as exc:
            print(f"Augmented metadata is unreadable; rebuilding without deleting DATA_AUG: {exc}")
        else:
            required_cols = ["file_name", "label", "patient_id", "image_id", "source"]
            missing_cols = [col for col in required_cols if col not in existing_df.columns]
            if missing_cols:
                print(f"Augmented metadata is not reusable; missing columns: {missing_cols}")
            else:
                missing_paths = [
                    path
                    for path in existing_df["file_name"].head(20).map(resolve_project_path)
                    if not path.exists()
                ]
                if missing_paths:
                    print("Augmented metadata is not reusable; first missing paths:")
                    for path in missing_paths[:5]:
                        print(" ", path)
                else:
                    print("Using the existing shared augmented dataset:")
                    print("DATA_AUG:", data_aug)
                    print("Metadata:", metadata_path)
                    print(existing_df["label"].value_counts())
                    return existing_df

    data_aug.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    out_idx = 0
    source_df = train_df.copy().reset_index(drop=True)
    for row_idx, row in source_df.iterrows():
        label = int(row["label"])
        src_path = Path(row["processed_path"]).resolve()
        if not src_path.exists():
            raise FileNotFoundError(src_path)

        metadata_rows.append({
            "file_name": src_path.relative_to(project_root).as_posix(),
            "label": label,
            "patient_id": str(row["patient_id"]),
            "image_id": str(row["image_id"]),
            "source": "real",
            "original_processed_path": str(src_path),
        })

        if label == 1:
            img_l = Image.open(src_path).convert("L")
            for aug_num in range(POSITIVE_AUGMENT_COPIES):
                aug_img_l = mild_positive_augmentation(
                    img_l,
                    aug_idx=(row_idx * 100 + aug_num),
                )
                aug_name = f"mammo_{out_idx:06d}_label1_aug{aug_num}.png"
                aug_path = data_aug / aug_name
                aug_img_l.save(aug_path)
                metadata_rows.append({
                    "file_name": aug_path.relative_to(project_root).as_posix(),
                    "label": 1,
                    "patient_id": str(row["patient_id"]),
                    "image_id": str(row["image_id"]),
                    "source": "positive_augmentation",
                    "original_processed_path": str(src_path),
                })
                out_idx += 1

    augmented_df = pd.DataFrame(metadata_rows)
    augmented_df.to_csv(metadata_path, index=False)
    print("VAE training dataset ready:")
    print(augmented_df["label"].value_counts())
    return augmented_df


def load_images_from_df(
    df: pd.DataFrame,
    path_col: str,
    project_root: Path,
    img_size: int = IMG_SIZE,
    desc: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Load dataframe images into RAM, resize as needed, and normalize to VAE range [-1, 1]."""
    images_list, labels_list = [], []
    for index, (_, row) in enumerate(df.iterrows()):
        path = Path(row[path_col])
        if not path.is_absolute():
            path = project_root / path
        img = Image.open(path).convert("L")
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        images_list.append((arr / 127.5 - 1.0).astype(np.float32))
        labels_list.append(int(row["label"]))
        if (index + 1) % 200 == 0 or index == 0 or index + 1 == len(df):
            print(f"  {desc}: {index + 1}/{len(df)}")
    images = np.stack(images_list)[:, :, :, np.newaxis]
    labels = np.array(labels_list, dtype=np.int32)
    print(f"  {desc}: shape={images.shape}, RAM={images.nbytes / 1e6:.0f}MB")
    return images, labels


def build_vae_encoder() -> tf.keras.Model:
    """Build the convolutional encoder from 512x512 mammograms to 64x64 latent mean/log-variance."""
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
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.GroupNormalization(groups=32)(x)
    x = layers.LeakyReLU(0.2)(x)
    params = layers.Conv2D(LATENT_CHANNELS * 2, 1, padding="same", name="enc_params")(x)
    return tf.keras.Model(inp, params, name="vae_encoder")


def build_vae_decoder() -> tf.keras.Model:
    """Build the convolutional decoder from a 64x64 latent sample to a 512x512 mammogram."""
    x = inp = layers.Input(
        shape=(LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS),
        name="dec_input",
    )
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


def reset_downstream_artifacts(paths) -> None:
    """Drop every artifact a retrained VAE invalidates: latents, LDM checkpoints,
    evaluation records, both synthetic pools of both classes, and the generation logs.

    RAW and filtered images alike descend from the VAE decoder -- filtering selects
    1,361 images out of the RAW pool, it does not regenerate them -- so a filtered
    pool kept across a reset would stay registered in configs/generator_registry.json
    while being output of a decoder that no longer exists, indistinguishable from new
    images. Clearing them is deliberate: the caller reaches this function only through
    the --force-retrain --also-reset-downstream double opt-in.
    """
    for directory in [
        paths.latents_dir,
        paths.checkpoints_dir,
        paths.evaluation_dir,
    ]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    # Pools are recreated empty only where one existed: a positive-only experiment
    # (G05, "classes": ["positive"]) must not gain an empty negative directory here.
    cleared_pools = []
    for target_label in (1, 0):
        for pool_dir in get_class_image_dirs(paths, target_label):
            if pool_dir.exists():
                shutil.rmtree(pool_dir)
                pool_dir.mkdir(parents=True, exist_ok=True)
                cleared_pools.append(pool_dir)

    for pattern in [
        "ldm_unet_best*.keras",
        "ldm_unet_final_step*.keras",
    ]:
        for path in paths.models_dir.glob(pattern):
            path.unlink()

    for log_name in ["ldm_train.log", "ldm_evaluate.log", "ldm_generate.log"]:
        log_path = paths.logs_dir / log_name
        if log_path.exists():
            log_path.unlink()

    # generation_raw_state.json is a snapshot of the RAW pool, not a history: surviving
    # the reset would leave it asserting that just-emptied directories are full.
    # generation_summary.jsonl stays, like the .log files: it records runs that happened.
    class_log_dirs = [paths.logs_dir / class_name_for_label(label) for label in (1, 0)]
    for log_dir in [paths.logs_dir, *class_log_dirs]:
        state_path = log_dir / "generation_raw_state.json"
        if state_path.exists():
            state_path.unlink()
    print("Previous LDM/evaluation/generation artifacts removed for the new VAE.")
    # A silent reset forces the operator to guess what was invalidated. Every pool
    # named here has to be regenerated and refiltered before any benchmark run.
    for pool_dir in cleared_pools:
        print("  cleared pool, regenerate before use:", pool_dir)


def reset_vae_artifacts(paths) -> None:
    """Remove prior VAE checkpoints and logs before training a fresh run."""
    for pattern in [
        "vae_encoder*.keras",
        "vae_decoder*.keras",
    ]:
        for path in paths.models_dir.glob(pattern):
            path.unlink()

    for log_name in ["vae_history.json", "vae_summary.json"]:
        log_path = paths.logs_dir / log_name
        if log_path.exists():
            log_path.unlink()
    print("Removed previous VAE artifacts; starting a fresh VAE training run.")


def plot_history(history: dict, output_path: Path) -> None:
    """Plot per-epoch reconstruction, SSIM, KL, and validation SSIM histories."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(history["loss_recon"], label="train recon", color="blue")
    axes[0].set_title("VAE recon loss")
    axes[1].plot(history["loss_ssim"], label="train SSIM loss", color="purple")
    axes[1].set_title("VAE SSIM loss")
    axes[2].plot(history["loss_kl"], label="train KL", color="orange")
    axes[2].set_title("VAE KL")
    axes[3].plot(history["val_ssim"], label="val SSIM", color="green")
    axes[3].set_title("VAE val SSIM")
    for ax in axes:
        ax.grid(True, alpha=0.4)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_reconstruction_preview(
    vae_encoder: tf.keras.Model,
    vae_decoder: tf.keras.Model,
    x_val: np.ndarray,
    output_path: Path,
    n_images: int = 6,
) -> None:
    """Compare validation originals with VAE reconstructions from the latent mean, without sampling."""
    sample = tf.constant(x_val[:n_images], dtype=tf.float32)
    params = vae_encoder(sample, training=False)
    mu, _ = tf.split(params, 2, axis=-1)
    recons = vae_decoder(mu, training=False).numpy()

    n = min(n_images, len(x_val))
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5))
    for index in range(n):
        original = np.clip((x_val[index].squeeze() + 1.0) / 2.0, 0, 1)
        recon = np.clip((recons[index].squeeze() + 1.0) / 2.0, 0, 1)
        axes[0, index].imshow(original, cmap="gray", vmin=0, vmax=1)
        axes[0, index].axis("off")
        axes[1, index].imshow(recon, cmap="gray", vmin=0, vmax=1)
        axes[1, index].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Load data, train with validation SSIM/MSE early stopping, and save VAE artifacts."""
    global np, pd, tf, Image, layers, plt
    global configure_tensorflow, get_experiment_paths, vram_gb

    args = parse_args()
    configure_environment(args)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from PIL import Image
    from tensorflow.keras import layers

    from ldm_keras_utils import configure_tensorflow, get_experiment_paths, vram_gb
    from eco_tracker import measure_sustainability

    configure_tensorflow()

    paths = get_experiment_paths(args.project_root, args.experiment_dir)
    data_aug = paths.project_root / "data" / "real_augmented"
    best_encoder_path = paths.models_dir / "vae_encoder_best.keras"
    best_decoder_path = paths.models_dir / "vae_decoder_best.keras"
    if best_encoder_path.exists() and best_decoder_path.exists() and not args.force_retrain:
        print(f"VAE already present: {best_encoder_path.name} / {best_decoder_path.name}")
        print("VAE training skipped. Use --force-retrain to run it again.")
        sync_existing_vae_plots_to_results(paths, args.results_stage_name)
        sys.exit(0)

    train_csv_path = paths.metadata_dir / "train.csv"
    val_csv_path = paths.metadata_dir / "val.csv"
    if not train_csv_path.exists() or not val_csv_path.exists():
        raise FileNotFoundError("Train/validation metadata not found in data/processed/metadata")

    print("PROJECT_ROOT:", paths.project_root)
    print("EXPERIMENT_DIR:", paths.experiment_dir)
    print("MODELS_DIR:", paths.models_dir)
    print("DATA_PROCESSED_DIR:", paths.data_processed_dir)
    print(f"VAE: epochs={args.epochs}, batch={args.batch_size}, lr={args.learning_rate}")

    reset_vae_artifacts(paths)
    train_df = load_metadata(train_csv_path, paths.data_processed_dir)
    val_df = load_metadata(val_csv_path, paths.data_processed_dir)
    augmented_df = build_augmented_train_metadata(train_df, paths.project_root, data_aug)

    print("\nLoading VAE images...")
    x_train, y_train = load_images_from_df(augmented_df, "file_name", paths.project_root, desc="train")
    x_val, y_val = load_images_from_df(val_df, "processed_path", paths.project_root, desc="val")
    print(f"Train pos={int((y_train == 1).sum())}, neg={int((y_train == 0).sum())}")
    print(f"Val pos={int((y_val == 1).sum())}, neg={int((y_val == 0).sum())}")

    vae_encoder = build_vae_encoder()
    vae_decoder = build_vae_decoder()
    vae_optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    print(f"VAE encoder params: {vae_encoder.count_params():,}")
    print(f"VAE decoder params: {vae_decoder.count_params():,}")
    vram_gb("after building VAE")

    def reparameterize(mu, log_var):
        """Apply the reparameterization trick for differentiable sampling from mu and log_var."""
        eps = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps

    @tf.function
    def vae_train_step(x):
        """Run one encoder/decoder step and optimize reconstruction, SSIM, and KL loss."""
        with tf.GradientTape() as tape:
            params = vae_encoder(x, training=True)
            mu, log_var = tf.split(params, 2, axis=-1)
            z = reparameterize(mu, log_var)
            x_recon = vae_decoder(z, training=True)

            loss_recon = tf.reduce_mean(tf.square(x - x_recon))
            x_true_01 = (x + 1.0) / 2.0
            x_recon_01 = (x_recon + 1.0) / 2.0
            loss_ssim = 1.0 - tf.reduce_mean(
                tf.image.ssim(x_true_01, x_recon_01, max_val=1.0)
            )
            loss_kl = -0.5 * tf.reduce_mean(
                1.0 + log_var - tf.square(mu) - tf.exp(log_var)
            )
            loss_total = loss_recon + SSIM_WEIGHT * loss_ssim + KL_WEIGHT * loss_kl

        all_vars = vae_encoder.trainable_variables + vae_decoder.trainable_variables
        grads = tape.gradient(loss_total, all_vars)
        vae_optimizer.apply_gradients(zip(grads, all_vars))
        return loss_total, loss_recon, loss_ssim, loss_kl

    vae_train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train), reshuffle_each_iteration=True)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    vae_val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    print(f"VAE train batches: {len(vae_train_ds)}")
    print(f"VAE val batches:   {len(vae_val_ds)}")

    best_vae_ssim = 0.0
    best_vae_mse = float("inf")
    best_vae_score = -float("inf")
    es_counter = 0
    stopped_early = False
    history = {
        "loss_total": [],
        "loss_recon": [],
        "loss_ssim": [],
        "loss_kl": [],
        "val_ssim": [],
        "val_mse": [],
    }

    final_encoder_path = paths.models_dir / "vae_encoder_final.keras"
    final_decoder_path = paths.models_dir / "vae_decoder_final.keras"

    print("=" * 70)
    print("PHASE 1 -- VAE TRAINING")
    print(f"Maximum epochs: {args.epochs} | Patience: {VAE_ES_PATIENCE}")
    print("=" * 70)
    sys.stdout.flush()

    with measure_sustainability(label="vae_training", sample_interval=0.5) as eco_vae:
        for epoch in range(args.epochs):
            losses_t, losses_r, losses_s, losses_k = [], [], [], []
            epoch_t0 = time.perf_counter()
            for step, (batch_x, _) in enumerate(vae_train_ds):
                lt, lr, ls, lk = vae_train_step(batch_x)
                losses_t.append(float(lt.numpy()))
                losses_r.append(float(lr.numpy()))
                losses_s.append(float(ls.numpy()))
                losses_k.append(float(lk.numpy()))
                if step % 20 == 0:
                    print(
                        f"  VAE Ep {epoch + 1:03d} | Step {step:03d} | "
                        f"total={losses_t[-1]:.4f} | recon={losses_r[-1]:.4f} | "
                        f"ssim_l={losses_s[-1]:.4f} | kl={losses_k[-1]:.4f}"
                    )
                    sys.stdout.flush()

            ssim_vals, mse_vals = [], []
            for val_x, _ in vae_val_ds:
                params = vae_encoder(val_x, training=False)
                mu, _ = tf.split(params, 2, axis=-1)
                x_recon = vae_decoder(mu, training=False)
                val_x_01 = (val_x + 1.0) / 2.0
                recon_01 = (x_recon + 1.0) / 2.0
                ssim_vals.append(tf.reduce_mean(tf.image.ssim(val_x_01, recon_01, max_val=1.0)).numpy())
                mse_vals.append(tf.reduce_mean(tf.square(val_x - x_recon)).numpy())

            val_ssim = float(np.mean(ssim_vals))
            val_mse = float(np.mean(mse_vals))
            mean_t = float(np.mean(losses_t))
            mean_r = float(np.mean(losses_r))
            mean_s = float(np.mean(losses_s))
            mean_k = float(np.mean(losses_k))
            elapsed = time.perf_counter() - epoch_t0

            history["loss_total"].append(mean_t)
            history["loss_recon"].append(mean_r)
            history["loss_ssim"].append(mean_s)
            history["loss_kl"].append(mean_k)
            history["val_ssim"].append(val_ssim)
            history["val_mse"].append(val_mse)

            print(
                f"--- VAE EPOCH {epoch + 1:03d}/{args.epochs} | "
                f"total={mean_t:.4f} | recon={mean_r:.4f} | ssim_l={mean_s:.4f} | "
                f"kl={mean_k:.4f} | val_ssim={val_ssim:.4f} | val_mse={val_mse:.4f} | "
                f"time={elapsed:.0f}s | ES={es_counter}/{VAE_ES_PATIENCE} ---"
            )

            # SSIM remains the dominant selection metric; validation MSE contributes
            # only as a low-weight correction (see MSE_SELECTION_WEIGHT).
            val_score = val_ssim - MSE_SELECTION_WEIGHT * val_mse
            if val_score > best_vae_score + VAE_ES_MIN_DELTA:
                best_vae_score = val_score
                best_vae_ssim = val_ssim
                best_vae_mse = val_mse
                es_counter = 0
                vae_encoder.save(str(best_encoder_path))
                vae_decoder.save(str(best_decoder_path))
                print(f"  [BEST] val_ssim={val_ssim:.4f} val_mse={val_mse:.4f} score={best_vae_score:.4f}")
            else:
                es_counter += 1
                print(f"  [ES] no improvement ({es_counter}/{VAE_ES_PATIENCE})")

            if (epoch + 1) % 25 == 0:
                vae_encoder.save(str(paths.models_dir / f"vae_encoder_ep{epoch + 1:03d}.keras"))
                vae_decoder.save(str(paths.models_dir / f"vae_decoder_ep{epoch + 1:03d}.keras"))
                print(f"  [CKPT] ep{epoch + 1}")

            sys.stdout.flush()
            if es_counter >= VAE_ES_PATIENCE:
                stopped_early = True
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    vae_encoder.save(str(final_encoder_path))
    vae_decoder.save(str(final_decoder_path))
    if not best_encoder_path.exists() or not best_decoder_path.exists():
        shutil.copy2(final_encoder_path, best_encoder_path)
        shutil.copy2(final_decoder_path, best_decoder_path)

    vae_encoder = tf.keras.models.load_model(str(best_encoder_path), compile=False)
    vae_decoder = tf.keras.models.load_model(str(best_decoder_path), compile=False)
    vae_encoder.trainable = False
    vae_decoder.trainable = False

    if args.force_retrain and args.also_reset_downstream:
        reset_downstream_artifacts(paths)
    elif args.force_retrain:
        print(
            "WARNING: VAE retrained without resetting downstream artifacts. "
            "LDM, evaluation, synthetic, and output artifacts may no longer match "
            "the new VAE and require manual reevaluation."
        )
    vae_metrics_path = results_plot_path(
        paths.project_root,
        args.results_stage_name,
        "vae_metrics.png",
    )
    vae_reconstruction_path = results_plot_path(
        paths.project_root,
        args.results_stage_name,
        "vae_reconstruction.png",
    )
    plot_history(history, vae_metrics_path)
    save_reconstruction_preview(
        vae_encoder,
        vae_decoder,
        x_val,
        vae_reconstruction_path,
    )
    print(f"Plot saved to results: {vae_metrics_path}")
    print(f"Plot saved to results: {vae_reconstruction_path}")

    summary = {
        "phase": "vae_training",
        "epochs_run": len(history["val_ssim"]),
        "epochs_max": args.epochs,
        "stopped_early": stopped_early,
        "best_val_ssim": best_vae_ssim,
        "best_val_mse": best_vae_mse,
        "best_selection_score": best_vae_score,
        "mse_selection_weight": MSE_SELECTION_WEIGHT,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "kl_weight": KL_WEIGHT,
        "ssim_weight": SSIM_WEIGHT,
        "encoder_best": str(best_encoder_path),
        "decoder_best": str(best_decoder_path),
        **eco_vae.metrics.to_dict(),
    }
    with open(paths.logs_dir / "vae_history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    with open(paths.logs_dir / "vae_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with open(
        sustainability_log_path(paths.project_root, args.results_stage_name),
        "a",
        encoding="utf-8",
    ) as file:
        file.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"VAE completed. Best val_ssim={best_vae_ssim:.4f} val_mse={best_vae_mse:.4f}")
    print("Best encoder:", best_encoder_path)
    print("Best decoder:", best_decoder_path)
    print(eco_vae.metrics)

    del x_train, y_train, x_val, y_val, vae_train_ds, vae_val_ds
    gc.collect()
    vram_gb("end of VAE training")


if __name__ == "__main__":
    main()
