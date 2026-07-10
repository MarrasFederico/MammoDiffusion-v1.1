from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def resolve_sd_vae_model(project_root: Path, requested: str | Path | None = None) -> str:
    """Resolve a local Stable Diffusion model path without relying on Hugging Face Hub."""
    if requested is not None and str(requested).strip():
        candidate = Path(str(requested)).expanduser()
        if candidate.exists():
            return str(candidate.absolute())
        raise FileNotFoundError(
            "Modello Stable Diffusion 2.1 locale non trovato: "
            f"{candidate}. Scaricalo/preparalo da Drive come nel notebook 03b."
        )

    env_value = os.environ.get("MAMMODIFFUSION_SD21_BASE")
    if env_value:
        candidate = Path(env_value).expanduser()
        if candidate.exists():
            return str(candidate.absolute())
        raise FileNotFoundError(
            "MAMMODIFFUSION_SD21_BASE punta a un modello locale non trovato: "
            f"{candidate}"
        )

    candidates = [
        project_root / "notebooks" / "pretrained_model" / "stable-diffusion-2-1-base",
        project_root / "pretrained_model" / "stable-diffusion-2-1-base",
        project_root / "models" / "stable-diffusion-2-1-base",
        project_root / "experiments" / "diffusers" / "07_ldm_sdvae_extra1361" / "pretrained_model" / "stable-diffusion-2-1-base",
        project_root / "experiments" / "diffusers" / "03_sd21_vae_finetuned" / "pretrained_model" / "stable-diffusion-2-1-base",
        # Fallback per checkout creati prima della migrazione della struttura.
        project_root / "experiments" / "20260703_ldm_sdvae_extra1361" / "pretrained_model" / "stable-diffusion-2-1-base",
        project_root / "experiments" / "20260706_sd21_vaeft_extra1361" / "pretrained_model" / "stable-diffusion-2-1-base",
        project_root / "experiments" / "20260611_sd21_rsna_mlo_512_inference_100_steps" / "pretrained_model" / "stable-diffusion-2-1-base",
        project_root / "experiments" / "20260607_sd21_rsna_mlo_512" / "pretrained_model" / "stable-diffusion-2-1-base",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.absolute())
    checked = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Modello Stable Diffusion 2.1 locale non trovato. "
        "Non uso Hugging Face Hub perché il modello pubblico non è più disponibile.\n"
        f"Percorsi controllati:\n{checked}"
    )


def _has_vae_config(path: Path) -> bool:
    return (path / "config.json").is_file() and (
        (path / "diffusion_pytorch_model.safetensors").is_file()
        or (path / "diffusion_pytorch_model.bin").is_file()
    )


def load_sd_vae(
    model_name_or_path: str | Path,
    device: str | None = None,
    dtype: str | None = None,
):
    """Load the SD AutoencoderKL from a local full pipeline folder or VAE folder."""
    import torch
    from diffusers import AutoencoderKL

    resolved = str(model_name_or_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)

    path = Path(resolved)
    if path.exists() and _has_vae_config(path):
        vae = AutoencoderKL.from_pretrained(str(path), torch_dtype=torch_dtype, local_files_only=True)
    elif path.exists():
        vae = AutoencoderKL.from_pretrained(
            str(path),
            subfolder="vae",
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
    else:
        raise FileNotFoundError(
            "Modello Stable Diffusion 2.1 locale non trovato: "
            f"{resolved}. Passa una cartella locale con la subfolder vae."
        )
    vae = vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
    return vae, device, torch_dtype, scaling_factor


def image_batch_to_sd_tensor(images: np.ndarray, device: str, dtype) -> "torch.Tensor":
    """Convert grayscale NHWC images in [-1, 1] to SD RGB NCHW tensors."""
    import torch

    arr = np.asarray(images, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., None]
    if arr.shape[-1] != 1:
        raise ValueError(f"Expected grayscale NHWC images, got shape {arr.shape}")
    arr = np.repeat(arr, 3, axis=-1)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    tensor = tensor.to(device=device, dtype=dtype)
    return tensor


def encode_images_array_to_sd_latents(
    images: np.ndarray,
    vae,
    device: str,
    dtype,
    batch_size: int = 4,
) -> np.ndarray:
    """Encode grayscale images to scaled SD latents in TensorFlow-friendly NHWC layout."""
    import torch

    latents: list[np.ndarray] = []
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = image_batch_to_sd_tensor(images[start : start + batch_size], device, dtype)
            encoded = vae.encode(batch).latent_dist.mean * scaling_factor
            latents.append(encoded.detach().float().cpu().permute(0, 2, 3, 1).numpy())
            del batch, encoded
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return np.concatenate(latents, axis=0).astype(np.float32)


def decode_sd_latents_to_grayscale(
    latents_nhwc: np.ndarray,
    vae,
    device: str,
    dtype,
    batch_size: int = 1,
) -> np.ndarray:
    """Decode scaled SD latents in NHWC layout to grayscale images in [0, 1]."""
    import torch

    arr = np.asarray(latents_nhwc, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.shape[-1] != 4:
        raise ValueError(f"Expected latent NHWC with 4 channels, got shape {arr.shape}")

    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(arr), batch_size):
            batch_np = arr[start : start + batch_size]
            latents = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous()
            latents = latents.to(device=device, dtype=dtype) / scaling_factor
            decoded = vae.decode(latents).sample
            decoded = decoded.detach().float().cpu().clamp(-1.0, 1.0)
            gray = decoded.mean(dim=1, keepdim=False)
            gray = ((gray + 1.0) / 2.0).clamp(0.0, 1.0).numpy()
            outputs.append(gray)
            del latents, decoded
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return np.concatenate(outputs, axis=0).astype(np.float32)


def write_sd_vae_metadata(
    output_path: Path,
    model_name_or_path: str,
    scaling_factor: float,
    extra: dict | None = None,
) -> None:
    payload = {
        "vae_backend": "stable_diffusion",
        "model_name_or_path": str(model_name_or_path),
        "scaling_factor": float(scaling_factor),
    }
    if extra:
        payload.update(extra)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def file_signature(paths: Iterable[Path]) -> list[dict]:
    rows = []
    for path in paths:
        rows.append({
            "path": str(path),
            "size": path.stat().st_size if path.exists() else None,
            "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
        })
    return rows
