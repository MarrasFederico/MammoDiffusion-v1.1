from __future__ import annotations

import json
import importlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


def _infer_project_root(path: Path) -> Path | None:
    """Find the project root from a local SD/VAE model path."""
    for candidate in [path.resolve(), *path.resolve().parents]:
        if (candidate / "notebooks").is_dir() and (candidate / "experiments").is_dir():
            return candidate
    return None


def _local_diffusers_src_candidates(project_root: Path | None) -> list[Path]:
    """Return only the configured/canonical checkout, never experiment fallbacks."""
    candidates: list[Path] = []
    env_value = os.environ.get("MAMMODIFFUSION_DIFFUSERS_SRC")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    if project_root is not None:
        try:
            from shared_diffusers_assets import resolve_shared_diffusers_repo
            candidates.append(resolve_shared_diffusers_repo() / "src")
        except ImportError as exc:
            raise RuntimeError(
                "shared_diffusers_assets is required to resolve the canonical Diffusers checkout"
            ) from exc

    unique: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in unique and (candidate / "diffusers" / "__init__.py").is_file():
            unique.append(candidate)
    return unique


def ensure_diffusers_available(project_root: Path | None = None):
    """Import Diffusers, repairing an editable install broken by a directory rename."""
    try:
        return importlib.import_module("diffusers")
    except ModuleNotFoundError as original_error:
        for source_dir in _local_diffusers_src_candidates(project_root):
            source_text = str(source_dir)
            if source_text not in sys.path:
                sys.path.insert(0, source_text)
            importlib.invalidate_caches()
            try:
                return importlib.import_module("diffusers")
            except ModuleNotFoundError:
                sys.modules.pop("diffusers", None)

        checked = _local_diffusers_src_candidates(project_root)
        checked_text = "\n".join(f"- {path}" for path in checked) or "- no local checkout found"
        raise ModuleNotFoundError(
            "The 'diffusers' package cannot be imported. "
            "Install requirements.txt or set MAMMODIFFUSION_DIFFUSERS_SRC.\n"
            f"Checked local checkouts:\n{checked_text}"
        ) from original_error


def resolve_sd_vae_model(project_root: Path, requested: str | Path | None = None) -> str:
    """Resolve a local Stable Diffusion model path without relying on Hugging Face Hub."""
    if requested is not None and str(requested).strip():
        candidate = Path(str(requested)).expanduser()
        if candidate.exists():
            return str(candidate.absolute())
        raise FileNotFoundError(
            "Local Stable Diffusion 2.1 model not found: "
            f"{candidate}. Download or prepare it from Drive as described in "
            "notebooks/2_diffusers/03_SD21_VAE_FineTuned.ipynb."
        )

    env_value = os.environ.get("MAMMODIFFUSION_SD21_BASE")
    if env_value:
        candidate = Path(env_value).expanduser()
        if candidate.exists():
            return str(candidate.absolute())
        raise FileNotFoundError(
            "MAMMODIFFUSION_SD21_BASE points to a missing local model: "
            f"{candidate}"
        )

    from shared_diffusers_assets import resolve_shared_sd21_base, verify_shared_sd21_base

    candidate = resolve_shared_sd21_base()
    try:
        verify_shared_sd21_base(candidate)
        return str(candidate.absolute())
    except FileNotFoundError:
        checked = str(candidate)
    raise FileNotFoundError(
        "Local Stable Diffusion 2.1 model not found. "
        "Hugging Face Hub is not used because the public model is no longer available.\n"
        f"Checked canonical path:\n- {checked}"
    )


def _has_vae_config(path: Path) -> bool:
    """True if ``path`` is a standalone Diffusers VAE folder.

    Accepts full-precision, FP16 (``diffusion_pytorch_model.fp16.safetensors``)
    and sharded (``diffusion_pytorch_model*-of-*.safetensors`` plus a
    ``*.index.json``) weight layouts instead of only the two exact legacy names.
    """
    if not (path / "config.json").is_file():
        return False
    return any(
        any(path.glob(pattern))
        for pattern in ("diffusion_pytorch_model*.safetensors", "diffusion_pytorch_model*.bin", "*.index.json")
    )


def load_sd_vae(
    model_name_or_path: str | Path,
    device: str | None = None,
    dtype: str | None = None,
):
    """Load the SD AutoencoderKL from a local full pipeline folder or VAE folder."""
    project_root = _infer_project_root(Path(model_name_or_path).expanduser())
    ensure_diffusers_available(project_root)

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
            "Local Stable Diffusion 2.1 model not found: "
            f"{resolved}. Pass a local directory containing the vae subfolder."
        )
    vae = vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
    return vae, device, torch_dtype, scaling_factor


def image_batch_to_sd_tensor(images: np.ndarray, device: str, dtype) -> Tensor:
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
