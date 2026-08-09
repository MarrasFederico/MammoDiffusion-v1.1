#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


IMG_SIZE = 512
POSITIVE_AUGMENT_COPIES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Stable Diffusion VAE latents for the MammoDiffusion LDM."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--sd-vae-model", default=None)
    parser.add_argument("--gpu-visible-devices", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mammodiffusion-matplotlib")
    )
    if args.gpu_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible_devices


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(csv_path: Path, dataset_root: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    required_cols = ["patient_id", "image_id", "label", "split", "processed_path"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_cols}")

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
    missing = [path for path in df["processed_path"].map(Path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} preprocessed images are missing; example: {missing[0]}")
    return df


def mild_positive_augmentation(img: Image.Image, aug_idx: int) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    rng = np.random.default_rng(42 + aug_idx)
    contrast = rng.uniform(0.90, 1.10)
    brightness = rng.uniform(-8, 8)
    noise = rng.normal(loc=0.0, scale=2.0, size=arr.shape)
    mean = arr.mean()
    arr = (arr - mean) * contrast + mean + brightness + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def build_augmented_train_metadata(train_df: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    data_aug = project_root / "data" / "real_augmented"
    metadata_path = data_aug / "metadata.csv"

    def resolve_project_path(path_value: str | Path) -> Path:
        path = Path(path_value)
        return path if path.is_absolute() else project_root / path

    if metadata_path.exists():
        existing_df = pd.read_csv(metadata_path).copy()
        required_cols = ["file_name", "label", "patient_id", "image_id", "source"]
        if all(col in existing_df.columns for col in required_cols):
            missing_paths = [
                path for path in existing_df["file_name"].head(20).map(resolve_project_path)
                if not path.exists()
            ]
            if not missing_paths:
                print("Using existing augmentation metadata:", metadata_path)
                print(existing_df["label"].value_counts())
                return existing_df
        print("Existing augmentation metadata is not reusable; rebuilding it.")

    data_aug.mkdir(parents=True, exist_ok=True)
    rows = []
    out_idx = 0
    for row_idx, row in train_df.reset_index(drop=True).iterrows():
        label = int(row["label"])
        src_path = Path(row["processed_path"]).resolve()
        rows.append({
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
                aug_img = mild_positive_augmentation(img_l, row_idx * 100 + aug_num)
                aug_path = data_aug / f"mammo_{out_idx:06d}_label1_aug{aug_num}.png"
                aug_img.save(aug_path)
                rows.append({
                    "file_name": aug_path.relative_to(project_root).as_posix(),
                    "label": 1,
                    "patient_id": str(row["patient_id"]),
                    "image_id": str(row["image_id"]),
                    "source": "positive_augmentation",
                    "original_processed_path": str(src_path),
                })
                out_idx += 1

    augmented_df = pd.DataFrame(rows)
    augmented_df.to_csv(metadata_path, index=False)
    print("SD-VAE training dataset ready:")
    print(augmented_df["label"].value_counts())
    return augmented_df


def load_images_from_df(df: pd.DataFrame, path_col: str, project_root: Path, desc: str) -> tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    for index, (_, row) in enumerate(df.iterrows()):
        path = Path(row[path_col])
        if not path.is_absolute():
            path = project_root / path
        img = Image.open(path).convert("L")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        images.append((arr / 127.5 - 1.0).astype(np.float32))
        labels.append(int(row["label"]))
        if (index + 1) % 200 == 0 or index == 0 or index + 1 == len(df):
            print(f"  {desc}: {index + 1}/{len(df)}")
    x = np.stack(images)[:, :, :, None]
    y = np.array(labels, dtype=np.int32)
    print(f"  {desc}: shape={x.shape}, pos={int((y == 1).sum())}, neg={int((y == 0).sum())}")
    return x, y


def main() -> None:
    args = parse_args()
    configure_environment(args)

    from ldm_project_paths import get_experiment_paths
    from sd_vae_utils import (
        encode_images_array_to_sd_latents,
        load_sd_vae,
        resolve_sd_vae_model,
        write_sd_vae_metadata,
    )

    paths = get_experiment_paths(args.project_root, args.experiment_dir)
    train_csv = paths.metadata_dir / "train.csv"
    val_csv = paths.metadata_dir / "val.csv"
    augmented_metadata_path = paths.project_root / "data" / "real_augmented" / "metadata.csv"
    latents_manifest_path = paths.latents_dir / "latents_manifest.json"
    train_latents_path = paths.latents_dir / "latents_train.npz"
    val_latents_path = paths.latents_dir / "latents_val.npz"
    latent_stats_path = paths.latents_dir / "latent_stats.npz"

    train_df = load_metadata(train_csv, paths.data_processed_dir)
    val_df = load_metadata(val_csv, paths.data_processed_dir)
    augmented_df = build_augmented_train_metadata(train_df, paths.project_root)

    sd_vae_model = resolve_sd_vae_model(paths.project_root, args.sd_vae_model)
    manifest = {
        "schema_version": 2,
        "vae_backend": "stable_diffusion",
        "sd_vae_model": str(sd_vae_model),
        "train_csv_sha256": file_sha256(train_csv),
        "val_csv_sha256": file_sha256(val_csv),
        "augmented_metadata_sha256": file_sha256(augmented_metadata_path),
        "n_train": int(len(augmented_df)),
        "n_val": int(len(val_df)),
        "img_size": IMG_SIZE,
        "latent_size": 64,
        "latent_channels": 4,
        "positive_augment_copies": POSITIVE_AUGMENT_COPIES,
    }

    cache_ready = (
        train_latents_path.exists()
        and val_latents_path.exists()
        and latent_stats_path.exists()
        and latents_manifest_path.exists()
        and not args.force_recompute
    )
    if cache_ready:
        try:
            with open(latents_manifest_path, "r", encoding="utf-8") as file:
                old_manifest = json.load(file)
            cache_ready = old_manifest == manifest
        except Exception:
            cache_ready = False
    if cache_ready:
        print("SD-VAE latents already present and consistent:", paths.latents_dir)
        return

    vae, device, dtype, scaling_factor = load_sd_vae(sd_vae_model)
    print("SD-VAE:", sd_vae_model)
    print("device:", device)
    print("dtype:", dtype)
    print("scaling_factor:", scaling_factor)

    x_train, y_train = load_images_from_df(augmented_df, "file_name", paths.project_root, "train")
    z_train = encode_images_array_to_sd_latents(x_train, vae, device, dtype, args.batch_size)
    np.savez_compressed(train_latents_path, latents=z_train, labels=y_train)
    del x_train

    x_val, y_val = load_images_from_df(val_df, "processed_path", paths.project_root, "val")
    z_val = encode_images_array_to_sd_latents(x_val, vae, device, dtype, args.batch_size)
    np.savez_compressed(val_latents_path, latents=z_val, labels=y_val)
    del x_val

    latent_mean = z_train.mean(axis=(0, 1, 2), keepdims=True)
    latent_std = z_train.std(axis=(0, 1, 2), keepdims=True)
    latent_std = np.maximum(latent_std, 1e-6)
    np.savez(latent_stats_path, latent_mean=latent_mean, latent_std=latent_std)

    with open(latents_manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    write_sd_vae_metadata(
        paths.models_dir / "sd_vae_config.json",
        sd_vae_model,
        scaling_factor,
        {"latents_manifest": str(latents_manifest_path)},
    )
    print("SD-VAE latents saved to:", paths.latents_dir)
    print("z_train:", z_train.shape, "z_val:", z_val.shape)
    print("latent_mean:", latent_mean.reshape(-1).tolist())
    print("latent_std:", latent_std.reshape(-1).tolist())


if __name__ == "__main__":
    main()
