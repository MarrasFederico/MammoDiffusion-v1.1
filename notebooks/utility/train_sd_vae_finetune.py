#!/usr/bin/env python3
"""Fine-tune Stable Diffusion 2.1's VAE on project mammograms.

Adapt SD2.1 ``AutoencoderKL`` to grayscale-as-RGB mammography, normally by
unfreezing only the decoder. That default preserves the latent distribution
expected by the U-Net. Notebook ``03_SD21_VAE_FineTuned.ipynb`` launches this
module in a subprocess.

Main outputs under ``--output-dir``:

* ``vae_finetuned/``: Diffusers directory with updated config and safetensors;
* ``vae_training_history.csv``: per-step loss;
* ``vae_best_metrics.json``: best epoch, validation loss/PSNR, and parameters;
* ``vae_reconstruction_before_after.png``: qualitative before/after grid.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_RESOLUTION = 512
DEFAULT_BATCH_SIZE = 2
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-5
DEFAULT_KL_WEIGHT = 0.0
DEFAULT_L1_WEIGHT = 1.0
DEFAULT_LPIPS_WEIGHT = 0.1
DEFAULT_SEED = 42
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    """Parse paths, hyperparameters, and the VAE blocks to unfreeze."""
    parser = argparse.ArgumentParser(description="Fine-tune SD2.1 VAE on mammograms")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pretrained-model-dir", type=Path, required=True,
                        help="Diffusers directory for the base SD2.1 model (contains the 'vae' subfolder).")
    parser.add_argument("--train-metadata-csv", type=Path, required=True,
                        help="Training CSV (real + augmented) with file_name, label, and source columns.")
    parser.add_argument("--val-metadata-csv", type=Path, required=True,
                        help="Real-validation CSV with a processed_path column.")
    parser.add_argument("--data-processed-dir", type=Path, required=True)
    parser.add_argument("--data-augmented-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for fine-tuned VAE weights, logs, plots, and metrics.")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--l1-weight", type=float, default=DEFAULT_L1_WEIGHT)
    parser.add_argument("--lpips-weight", type=float, default=DEFAULT_LPIPS_WEIGHT)
    parser.add_argument("--kl-weight", type=float, default=DEFAULT_KL_WEIGHT,
                        help="Posterior KL weight; zero disables it for decoder-only training.")
    parser.add_argument("--decoder-only", action="store_true", default=True,
                        help="When set (the default), unfreeze only the VAE decoder.")
    parser.add_argument("--full-vae", dest="decoder_only", action="store_false",
                        help="Unfreeze encoder and decoder. Requires kl-weight > 0 for stability.")
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1,
                        help="Save an intermediate checkpoint every N epochs; zero saves only the best.")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--max-train-images", type=int, default=None,
                        help="Optionally limit the training set for trial runs.")
    parser.add_argument("--early-stopping-patience", type=int, default=4,
                        help="Epochs without val_loss improvement before stopping; zero disables it.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4,
                        help="Minimum val_loss improvement required to reset patience.")
    parser.add_argument("--min-epochs", type=int, default=5,
                        help="Minimum number of epochs before early stopping can apply.")
    parser.add_argument("--allow-lpips-fallback", action="store_true",
                        help=(
                            "Continue with L1 only if LPIPS cannot load. The default fails explicitly, "
                            "preventing a run from being mistaken for one that used LPIPS."
                        ))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_train_image_path(row, data_processed_dir: Path, data_augmented_dir: Path) -> Path:
    """Resolve one training sample path, preserving 03b real/augmented compatibility."""
    import pandas as pd

    file_name = Path(str(row["file_name"])).name
    label = str(int(row["label"]))
    source = str(row.get("source", "")).strip().lower() if not pd.isna(row.get("source", "")) else ""
    real_candidate = data_processed_dir / "train" / label / file_name
    augmented_candidate = data_augmented_dir / file_name
    if source == "real":
        candidates = [real_candidate]
    elif source in {"positive_augmentation", "augmentation", "augmented"}:
        candidates = [augmented_candidate]
    else:
        candidates = [augmented_candidate, real_candidate]
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate.resolve()
    raise FileNotFoundError(
        f"Image not found for file_name={file_name} source={source}. "
        f"Checked paths: {', '.join(str(p) for p in candidates)}"
    )


def resolve_val_image_path(processed_path: str, data_processed_dir: Path, project_root: Path) -> Path:
    """Resolve a val.csv row whose processed_path is relative to the repository."""
    raw = Path(str(processed_path).replace("\\", "/"))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.append(project_root / raw)
    if "data" in raw.parts:
        candidates.append(project_root / Path(*raw.parts[raw.parts.index("data"):]))
    candidates.append(data_processed_dir / raw.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Validation image not found for {processed_path}. Checked paths: {checked}")


class MammoVAEDataset:
    """PyTorch dataset: grayscale mammograms repeated to three channels and normalized to [-1, 1]."""

    def __init__(self, image_paths, resolution: int, augment: bool = False):
        from PIL import Image
        import torch
        self._Image = Image
        self._torch = torch
        self.image_paths = [Path(path) for path in image_paths]
        self.resolution = int(resolution)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        Image = self._Image
        torch = self._torch
        path = self.image_paths[index]
        with Image.open(path) as image:
            gray = image.convert("L").resize((self.resolution, self.resolution), Image.BILINEAR)
        arr = np.asarray(gray, dtype=np.float32) / 255.0
        if self.augment:
            arr = np.clip(arr + np.random.uniform(-0.03, 0.03), 0.0, 1.0)
        arr = arr * 2.0 - 1.0  # [-1, 1]
        arr_rgb = np.repeat(arr[None, ...], 3, axis=0)  # (3, H, W)
        return torch.from_numpy(arr_rgb.copy())


def build_dataloader(paths, resolution, batch_size, num_workers, shuffle, augment, seed):
    import torch
    dataset = MammoVAEDataset(paths, resolution=resolution, augment=augment)
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        generator=generator,
    )


def maybe_load_lpips(device, allow_fallback: bool = False):
    """Load LPIPS in fp32.

    LPIPS is genuinely used when ``lpips_weight > 0``. By default, a load
    failure stops training with a clear error so a long L1-only run cannot be
    mistaken for an LPIPS run. Use ``--allow-lpips-fallback`` only for debugging
    or quick trials.
    """
    try:
        import lpips  # type: ignore
        model = lpips.LPIPS(net="alex").to(device)
        model.float()
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        print("[03-VAE] LPIPS loaded successfully (net=alex, fp32).", flush=True)
        return model
    except Exception as exc:  # noqa: BLE001
        message = (
            "LPIPS_WEIGHT > 0 but LPIPS cannot be loaded. "
            "Install or verify the 'lpips' package and torchvision, or pass "
            "--allow-lpips-fallback to continue with L1 only. "
            f"Original error: {repr(exc)}"
        )
        if allow_fallback:
            print("[WARN] " + message, flush=True)
            return None
        raise RuntimeError(message) from exc


def freeze_encoder(vae) -> int:
    """Freeze VAE encoder and quant_conv; return the remaining trainable parameter count."""
    for name, param in vae.named_parameters():
        if name.startswith("encoder") or name.startswith("quant_conv"):
            param.requires_grad_(False)
    trainable = sum(param.numel() for param in vae.parameters() if param.requires_grad)
    return int(trainable)


def compute_recon_loss(recon, target, lpips_model, l1_weight, lpips_weight):
    """Return L1(reconstruction, target) + lpips_weight * LPIPS.

    LPIPS is forced to float32 even when the VAE forward pass uses fp16
    autocast. This is the key correction over the first script version because
    LPIPS is often unstable in half precision.
    """
    import torch

    recon_f32 = recon.float().clamp(-1, 1)
    target_f32 = target.float().clamp(-1, 1)
    loss = l1_weight * torch.nn.functional.l1_loss(recon_f32, target_f32)

    if lpips_model is not None and lpips_weight > 0:
        device_type = "cuda" if recon_f32.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            lpips_loss = lpips_model(recon_f32, target_f32).mean()
        loss = loss + lpips_weight * lpips_loss
    return loss


def gather_val_paths(val_metadata_csv, data_processed_dir, project_root, max_val_images=None):
    import pandas as pd
    metadata = pd.read_csv(val_metadata_csv)
    if "processed_path" not in metadata.columns:
        raise ValueError(f"Missing 'processed_path' column in {val_metadata_csv}")
    paths = [resolve_val_image_path(row, data_processed_dir, project_root) for row in metadata["processed_path"]]
    if max_val_images is not None:
        paths = paths[: int(max_val_images)]
    return paths


def gather_train_paths(train_metadata_csv, data_processed_dir, data_augmented_dir, max_images=None):
    import pandas as pd
    metadata = pd.read_csv(train_metadata_csv)
    metadata["file_name"] = metadata["file_name"].astype(str).str.replace("\\", "/", regex=False)
    if "source" not in metadata.columns:
        metadata["source"] = ""
    paths = [
        resolve_train_image_path(row, data_processed_dir, data_augmented_dir)
        for _, row in metadata.iterrows()
    ]
    if max_images is not None:
        paths = paths[: int(max_images)]
    return paths


def evaluate_vae(vae, dataloader, device, dtype, lpips_model, l1_weight, lpips_weight):
    """Return mean validation loss and PSNR with encoder and decoder in inference mode."""
    import torch
    vae.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device=device, dtype=dtype)
            posterior = vae.encode(batch).latent_dist
            latents = posterior.sample()
            recon = vae.decode(latents).sample
            loss = compute_recon_loss(recon, batch, lpips_model, l1_weight, lpips_weight)
            mse = torch.nn.functional.mse_loss(recon.clamp(-1, 1), batch).item()
            psnr = 10.0 * math.log10(4.0 / max(mse, 1e-12))  # range [-1, 1] -> max^2 = 4
            batch_size = batch.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_psnr += float(psnr) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        return float("nan"), float("nan")
    return total_loss / total_samples, total_psnr / total_samples


def save_reconstruction_grid(vae, val_paths, resolution, device, dtype, output_path, n_samples=6, seed=42):
    """Save an original/reconstruction grid for ``n_samples`` validation images."""
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(val_paths), size=min(n_samples, len(val_paths)), replace=False)
    fig, axes = plt.subplots(2, len(indices), figsize=(2.5 * len(indices), 5.0), squeeze=False)
    vae.eval()
    with torch.no_grad():
        for column, idx in enumerate(indices):
            with Image.open(val_paths[int(idx)]) as image:
                gray = image.convert("L").resize((resolution, resolution), Image.BILINEAR)
            arr = np.asarray(gray, dtype=np.float32) / 255.0
            arr = arr * 2.0 - 1.0
            arr_rgb = np.repeat(arr[None, ...], 3, axis=0)
            tensor = torch.from_numpy(arr_rgb.copy()).unsqueeze(0).to(device=device, dtype=dtype)
            recon = vae.decode(vae.encode(tensor).latent_dist.mean).sample
            recon_gray = recon.mean(dim=1).squeeze().detach().float().cpu().numpy()
            recon_gray = np.clip((recon_gray + 1.0) / 2.0, 0.0, 1.0)
            axes[0, column].imshow(np.asarray(gray, dtype=np.float32) / 255.0, cmap="gray")
            axes[0, column].set_title("original", fontsize=8)
            axes[0, column].axis("off")
            axes[1, column].imshow(recon_gray, cmap="gray")
            axes[1, column].set_title("VAE reconstruction", fontsize=8)
            axes[1, column].axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)


def save_vae_diffusers(vae, output_dir: Path) -> Path:
    """Save the VAE in Diffusers format (config.json + safetensors)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vae.save_pretrained(str(output_dir), safe_serialization=True)
    return output_dir


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_csv = output_dir / "vae_training_history.csv"
    best_metrics_path = output_dir / "vae_best_metrics.json"
    finetuned_vae_dir = output_dir / "vae_finetuned"
    reconstruction_plot = output_dir / "vae_reconstruction_before_after.png"

    print("[03-VAE] output dir:", output_dir)
    print("[03-VAE] pretrained SD2.1:", args.pretrained_model_dir)

    train_paths = gather_train_paths(
        args.train_metadata_csv,
        args.data_processed_dir,
        args.data_augmented_dir,
        max_images=args.max_train_images,
    )
    val_paths = gather_val_paths(args.val_metadata_csv, args.data_processed_dir, args.project_root)
    print(f"[03-VAE] train images: {len(train_paths)} | val images: {len(val_paths)}")

    import torch
    from diffusers import AutoencoderKL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Keep the VAE in fp32 for training stability; autocast applies only to the forward pass.
    vae = AutoencoderKL.from_pretrained(str(args.pretrained_model_dir), subfolder="vae")
    vae.to(device)

    # VRAM reduction matters most at 512x512 with LPIPS enabled. These calls are
    # harmless no-ops when the installed Diffusers version does not support them.
    if hasattr(vae, "enable_gradient_checkpointing"):
        try:
            vae.enable_gradient_checkpointing()
            print("[03-VAE] VAE gradient checkpointing enabled.", flush=True)
        except Exception as exc:
            print(f"[03-VAE][WARN] VAE gradient checkpointing not enabled: {repr(exc)}", flush=True)
    if hasattr(vae, "enable_slicing"):
        try:
            vae.enable_slicing()
            print("[03-VAE] VAE slicing enabled.", flush=True)
        except Exception as exc:
            print(f"[03-VAE][WARN] VAE slicing not enabled: {repr(exc)}", flush=True)

    vae.train()

    if args.decoder_only:
        trainable_params = freeze_encoder(vae)
        print(f"[03-VAE] decoder-only mode: trainable parameters = {trainable_params:,}")
    else:
        trainable_params = sum(param.numel() for param in vae.parameters() if param.requires_grad)
        print(f"[03-VAE] full-VAE mode: trainable parameters = {trainable_params:,}")

    train_loader = build_dataloader(
        train_paths, args.resolution, args.batch_size, args.num_workers,
        shuffle=True, augment=True, seed=args.seed,
    )
    val_loader = build_dataloader(
        val_paths, args.resolution, args.batch_size, args.num_workers,
        shuffle=False, augment=False, seed=args.seed,
    )

    lpips_model = maybe_load_lpips(device, allow_fallback=args.allow_lpips_fallback) if args.lpips_weight > 0 else None

    optimizer = torch.optim.AdamW(
        [param for param in vae.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(args.mixed_precision == "fp16" and device.type == "cuda"))
    except TypeError:  # Compatibility with older PyTorch releases.
        scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16" and device.type == "cuda"))
    autocast_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else None

    save_reconstruction_grid(vae, val_paths, args.resolution, device, torch.float32,
                             output_dir / "vae_reconstruction_before_training.png")

    with log_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "step", "train_loss", "val_loss", "val_psnr", "wall_time_s"])

    best_val = math.inf
    best_epoch = -1
    best_val_psnr = float("nan")
    epochs_without_improvement = 0
    stopped_early = False
    start_time = time.time()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        vae.train()
        epoch_loss = 0.0
        epoch_samples = 0
        optimizer.zero_grad(set_to_none=True)
        accumulation_counter = 0

        for batch_index, batch in enumerate(train_loader):
            batch = batch.to(device=device, dtype=torch.float32)
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    posterior = vae.encode(batch).latent_dist
                    latents = posterior.sample()
                    recon = vae.decode(latents).sample
                    recon_loss = compute_recon_loss(recon, batch, lpips_model, args.l1_weight, args.lpips_weight)
                    if args.kl_weight > 0 and not args.decoder_only:
                        kl = posterior.kl().mean()
                        loss = recon_loss + args.kl_weight * kl
                    else:
                        loss = recon_loss
                loss_for_backward = loss / max(1, args.gradient_accumulation)
                scaler.scale(loss_for_backward).backward()
            else:
                posterior = vae.encode(batch).latent_dist
                latents = posterior.sample()
                recon = vae.decode(latents).sample
                recon_loss = compute_recon_loss(recon, batch, lpips_model, args.l1_weight, args.lpips_weight)
                if args.kl_weight > 0 and not args.decoder_only:
                    kl = posterior.kl().mean()
                    loss = recon_loss + args.kl_weight * kl
                else:
                    loss = recon_loss
                (loss / max(1, args.gradient_accumulation)).backward()

            accumulation_counter += 1
            if accumulation_counter % args.gradient_accumulation == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [param for param in vae.parameters() if param.requires_grad], max_norm=1.0
                )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            batch_size = batch.shape[0]
            epoch_loss += float(loss.item()) * batch_size
            epoch_samples += batch_size

            if batch_index % 20 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[03-VAE] epoch {epoch:03d} batch {batch_index:04d}/{len(train_loader):04d} "
                    f"loss={loss.item():.4f} | step={global_step} | elapsed={elapsed:.0f}s",
                    flush=True,
                )

        # If the batch count is not divisible by gradient_accumulation, apply the
        # final accumulated step instead of dropping it.
        if accumulation_counter % args.gradient_accumulation != 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [param for param in vae.parameters() if param.requires_grad], max_norm=1.0
            )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        train_loss = epoch_loss / max(1, epoch_samples)
        val_loss, val_psnr = evaluate_vae(
            vae, val_loader, device, torch.float32, lpips_model, args.l1_weight, args.lpips_weight
        )
        wall_time = time.time() - start_time
        print(
            f"[03-VAE] EPOCH {epoch:03d} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} val_psnr={val_psnr:.2f}dB | elapsed={wall_time:.0f}s",
            flush=True,
        )

        with log_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([epoch, global_step, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{val_psnr:.4f}", f"{wall_time:.2f}"])

        improved = val_loss < best_val - args.early_stopping_min_delta
        if improved:
            best_val = val_loss
            best_val_psnr = val_psnr
            best_epoch = epoch
            epochs_without_improvement = 0
            save_vae_diffusers(vae, finetuned_vae_dir)
            with best_metrics_path.open("w", encoding="utf-8") as handle:
                json.dump({
                    "best_epoch": best_epoch,
                    "best_val_loss": round(float(best_val), 6),
                    "best_val_psnr": round(float(best_val_psnr), 4),
                    "global_step": int(global_step),
                    "wall_time_s": round(float(wall_time), 2),
                    "stopped_early": False,
                    "args": {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "l1_weight": args.l1_weight,
                        "lpips_weight": args.lpips_weight,
                        "kl_weight": args.kl_weight,
                        "decoder_only": bool(args.decoder_only),
                        "resolution": args.resolution,
                        "seed": args.seed,
                        "mixed_precision": args.mixed_precision,
                        "early_stopping_patience": args.early_stopping_patience,
                        "early_stopping_min_delta": args.early_stopping_min_delta,
                        "min_epochs": args.min_epochs,
                    },
                    "pretrained_model_dir": str(args.pretrained_model_dir),
                    "output_dir": str(output_dir),
                }, handle, indent=2, ensure_ascii=False)
            print(f"[03-VAE]   new best -> saved to {finetuned_vae_dir}", flush=True)
        else:
            epochs_without_improvement += 1
            print(
                f"[03-VAE]   no meaningful improvement "
                f"(minimum delta={args.early_stopping_min_delta:g}); "
                f"patience {epochs_without_improvement}/{args.early_stopping_patience}",
                flush=True,
            )
            if args.checkpoint_every_epochs > 0 and epoch % args.checkpoint_every_epochs == 0:
                intermediate_dir = output_dir / "checkpoints" / f"epoch_{epoch:03d}"
                save_vae_diffusers(vae, intermediate_dir)
                print(f"[03-VAE]   intermediate checkpoint -> {intermediate_dir}", flush=True)

        if (
            args.early_stopping_patience > 0
            and epoch >= args.min_epochs
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"[03-VAE] Early stopping: val_loss has not improved for "
                f"{epochs_without_improvement} epochs. "
                f"Best epoch={best_epoch}, best_val_loss={best_val:.6f}, "
                f"best_val_psnr={best_val_psnr:.2f}dB.",
                flush=True,
            )
            break

    if not finetuned_vae_dir.exists():
        # No epoch improved validation loss; save the final state instead of leaving an empty directory.
        save_vae_diffusers(vae, finetuned_vae_dir)
        with best_metrics_path.open("w", encoding="utf-8") as handle:
            json.dump({
                "best_epoch": args.epochs,
                "best_val_loss": None,
                "best_val_psnr": None,
                "note": "No epoch improved validation loss; saved the final state.",
                "output_dir": str(output_dir),
            }, handle, indent=2, ensure_ascii=False)

    # Reload the best state for final reconstruction instead of using an intermediate state.
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    final_vae = AutoencoderKL.from_pretrained(str(finetuned_vae_dir))
    final_vae.to(device).eval()
    save_reconstruction_grid(final_vae, val_paths, args.resolution, device, torch.float32,
                             reconstruction_plot)

    print(f"[03-VAE] Fine-tuned VAE saved to {finetuned_vae_dir}")
    # Add early-stopping information to the final JSON when applicable.
    if best_metrics_path.is_file():
        try:
            with best_metrics_path.open(encoding="utf-8") as handle:
                best_payload = json.load(handle)
            best_payload["stopped_early"] = bool(stopped_early)
            best_payload["completed_epochs"] = int(epoch)
            with best_metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(best_payload, handle, indent=2, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] could not update {best_metrics_path}: {exc}", flush=True)

    print(f"[03-VAE] Best epoch: {best_epoch} | best val loss: {best_val:.4f} | val PSNR: {best_val_psnr:.2f}dB")
    print(f"[03-VAE] History log: {log_csv}")
    print(f"[03-VAE] Final reconstruction grid: {reconstruction_plot}")


if __name__ == "__main__":
    sys.exit(main())
