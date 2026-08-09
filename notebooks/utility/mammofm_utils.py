"""Shared utilities for fine-tuned Mammo-FM classifiers (EfficientNet-B5, PyTorch).

This replaces `mammodino_utils.py` (now archived under `_deprecated_mammodino/`): public
MammoDINO checkpoints (GE HealthCare, arXiv:2510.11883) were not available, whereas
**Mammo-FM** (batmanlab, arXiv:2512.00198) is a mammography foundation model with public
checkpoints on Hugging Face (`batmanLab/Mammo-FM`). The module follows the style of the
project's other helpers (`maxvit_utils.py`) and reuses its Keras-style callbacks
(EarlyStopping, ModelCheckpoint, and CSVLogger).

The architecture was verified directly against the real checkpoint
(`Mammo-FM_BatmanlabTrained_CLIP.tar`, downloaded and inspected while developing this helper):

- The checkpoint is a torch dictionary (`torch.load`) with `config` and `model` keys, using
  the same format as `batmanlab/Mammo-CLIP`, from which Mammo-FM evolved. The
  `config["model"]["image_encoder"]` field declares `{"source": "cnn", "name":
  "tf_efficientnet_b5_ns-detect", "model_type": "cnn"}`.
- The image encoder is a **custom** EfficientNet-B5, nearly identical to lukemelas's
  `efficientnet_pytorch` package, with `_conv_stem`, `_blocks`, `_conv_head`, and related
  attributes. It expects **three-channel input** (`_conv_stem.weight` has shape
  (48, 3, 3, 3)), so grayscale mammograms must be replicated to RGB rather than supplied
  as one-channel images.
- Image-encoder weights use the `"image_encoder."` prefix in the complete CLIP state dict,
  which also contains `text_encoder.*`, `image_projection.*`, `text_projection.*`, and
  `logit_scale` entries that are irrelevant to classification. The final `_fc` layer is not
  stored because Mammo-CLIP/Mammo-FM does not train it; this helper extracts only the globally
  pooled features (`out_dim=2048` for B5).
- Deserializing the checkpoint requires `omegaconf`, because `config` is stored as a
  Hydra/OmegaConf object. Building the encoder requires `efficientnet_pytorch`. Both imports
  fail with an explicit diagnostic when the package is unavailable.
- The official normalization from Mammo-CLIP's `configs/pre_train_b5_clip.yaml` uses
  mean=0.3089279 and std=0.25053555408335154 (scalars, not ImageNet statistics) after per-image
  min-max normalization to [0, 1] (`img -= img.min(); img /= img.max()`), matching the upstream
  `ImageClassificationDataset`.
- Native pre-training resolution is 1520x912. This project uses square 512x512 inputs for
  consistency with MaxViT-512, RAD-DINO, and the other classifiers. EfficientNet is fully
  convolutional and ends in global average pooling, so this resolution is valid, although
  absolute performance may differ from results reported at native resolution. This is an
  explicit protocol choice.
- Official pre-training preprocessing also includes CLAHE (see
  `configs/transform/clahe.yaml`). It is disabled here by default
  (`USE_CLAHE_PREPROCESSING=False`) to remain consistent with the other project classifiers,
  but can be enabled in a dedicated experiment.

**Mammo-FM weight license**: the weights use a Custom Academic License for Model Weights
(non-commercial academic research only, **no clinical or diagnostic use**, and no weight
redistribution). This helper downloads the checkpoint to the local Hugging Face cache
(`~/.cache/huggingface/hub`, outside the repository) and never copies it elsewhere. Do not
store or redistribute the weights in this project repository.

There is no silent fallback. If the Mammo-FM checkpoint is unavailable, unconfigured, or
incompatible with the expected architecture, this module raises `MammoFMConfigError` and
never substitutes a generic DINOv2, RAD-DINO, or ImageNet backbone.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from maxvit_utils import (  # noqa: F401  (re-exported for notebook convenience)
    BinaryFocalLoss,
    CSVLogger,
    EarlyStopping,
    History,
    ModelCheckpoint,
    bootstrap_balanced,
    compute_pos_weight,
    count_trainable_params,
    optimal_threshold_youden,
    refreeze_batchnorm,
)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_HF_REPO = "batmanLab/Mammo-FM"
DEFAULT_CHECKPOINT_NAME = "Mammo-FM_BatmanlabTrained_CLIP.tar"
DEFAULT_IMG_SIZE = 512  # Project protocol: 512x512. Native Mammo-FM resolution: 1520x912.
DEFAULT_MAMMOFM_MEAN = 0.3089279  # Official scalar pre-training statistic, not ImageNet.
DEFAULT_MAMMOFM_STD = 0.25053555408335154

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_LABEL_RE = re.compile(r"_label([01])")

# CNN image encoders supported by official Mammo-FM/Mammo-CLIP checkpoints. Each configured
# name maps to the equivalent `efficientnet_pytorch` architecture and pooled-feature width;
# both were verified against the real `Mammo-FM_BatmanlabTrained_CLIP.tar` checkpoint.
_SUPPORTED_CNN_ENCODERS = {
    "tf_efficientnet_b5_ns-detect": ("efficientnet-b5", 2048),
    "tf_efficientnetv2-detect": ("efficientnet-b2", 1408),
}

# The published classifier protocol fixes Mammo-FM to the B5 image encoder.  A
# complete project checkpoint contains every parameter of this architecture, so
# inference can reconstruct the module and then load that state dict without
# reopening the separately licensed foundation archive.
PROJECT_CHECKPOINT_ENCODER_NAME = "tf_efficientnet_b5_ns-detect"

# Common prefixes used to nest image-encoder tensors in complete CLIP state dicts. Remove them
# iteratively because prefixes may be nested in any order, for example
# `module.image_encoder._conv_stem...` becomes `_conv_stem...`.
_CHECKPOINT_PREFIX_CANDIDATES = (
    "module.", "model.", "backbone.", "encoder.", "visual.",
    "image_encoder.", "clip.", "student.", "teacher.", "net.",
)


def _strip_known_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _CHECKPOINT_PREFIX_CANDIDATES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


class MammoFMConfigError(RuntimeError):
    """Raised when Mammo-FM is unconfigured or its weights are unavailable or incompatible.

    This exception deliberately never triggers a silent fallback to a generic
    DINOv2, RAD-DINO, or ImageNet backbone. The notebook must stop until the official
    Mammo-FM checkpoint is available in the local Hugging Face cache.
    """


# ---------------------------------------------------------------------------
# Model: EfficientNet-B5 image encoder (Mammo-FM/Mammo-CLIP) plus linear head
# ---------------------------------------------------------------------------

class MammoFMImageEncoder(nn.Module):
    """Extract globally pooled features from `efficientnet_pytorch.EfficientNet`.

    This matches how the official Mammo-FM CLIP checkpoint uses its image encoder; see
    `breastclip/model/modules/efficientnet_custom.py` in batmanlab/Mammo-CLIP. The architecture
    and parameter names (`_conv_stem`, `_blocks`, `_conv_head`, and so on) are the same. The
    final `_fc` layer is never used or included in the official checkpoint, so this wrapper
    returns the pooled features immediately before that classifier.
    """

    def __init__(self, arch_name: str, out_dim: int):
        super().__init__()
        try:
            from efficientnet_pytorch import EfficientNet
        except ImportError as exc:
            raise MammoFMConfigError(
                "The 'efficientnet_pytorch' package is required to construct the EfficientNet "
                "image encoder used by official Mammo-FM checkpoints. Install it with: "
                "pip install efficientnet_pytorch"
            ) from exc
        self.model = EfficientNet.from_name(arch_name, num_classes=1)  # `_fc` is not used.
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_features = self.model.extract_features(x)
        return F.adaptive_avg_pool2d(raw_features, 1).flatten(1)


class MammoFMClassifier(nn.Module):
    """Mammo-FM image encoder (EfficientNet-B5) plus a binary-classification head."""

    def __init__(self, image_encoder: MammoFMImageEncoder, hidden_size: int,
                 num_classes: int = 1, dropout: float = 0.1):
        super().__init__()
        self.image_encoder = image_encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(x)
        return self.classifier(self.dropout(features))


def _resolve_checkpoint_path(hf_repo: Optional[str], checkpoint_name: Optional[str],
                              use_local_checkpoint: bool, local_checkpoint_path: Optional[str],
                              local_files_only: bool = False) -> str:
    if use_local_checkpoint:
        if not local_checkpoint_path or not Path(local_checkpoint_path).is_file():
            raise MammoFMConfigError(
                f"USE_LOCAL_CHECKPOINT=True, but checkpoint '{local_checkpoint_path}' does "
                "not exist. Provide a valid local path to authorized Mammo-FM weights, for "
                "example after downloading them manually from "
                "https://huggingface.co/batmanLab/Mammo-FM)."
            )
        return str(local_checkpoint_path)

    if not hf_repo or not checkpoint_name:
        raise MammoFMConfigError(
            "Mammo-FM is not configured. Specify the official batmanLab/Mammo-FM Hugging Face "
            "repository and checkpoint name, or provide a local checkpoint. This notebook "
            "never silently substitutes a generic backbone."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise MammoFMConfigError(
            "The 'huggingface_hub' package is required to download the Mammo-FM checkpoint. "
            "Install it with: pip install huggingface_hub"
        ) from exc
    try:
        return hf_hub_download(
            repo_id=hf_repo,
            filename=checkpoint_name,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        availability = (
            "The notebook uses only the local cache. First download the authorized file to "
            "the standard Hugging Face cache."
            if local_files_only
            else "Check the network connection and repository/file name."
        )
        raise MammoFMConfigError(
            f"Could not download Mammo-FM checkpoint '{checkpoint_name}' from Hugging Face "
            f"repository '{hf_repo}': {exc}\n"
            f"{availability} There is no automatic fallback to a generic model."
        ) from exc


def _load_raw_checkpoint(ckpt_path: str) -> dict:
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        raise MammoFMConfigError(
            f"Could not deserialize Mammo-FM checkpoint '{ckpt_path}': missing module ({exc}). "
            "The checkpoint configuration uses OmegaConf/Hydra; install it with "
            "'pip install omegaconf' and retry."
        ) from exc
    if not isinstance(raw, dict) or "model" not in raw or "config" not in raw:
        raise MammoFMConfigError(
            f"Unrecognized checkpoint format in '{ckpt_path}': expected 'model' and 'config' "
            "keys (the official Mammo-CLIP/Mammo-FM format is a torch-saved dictionary with "
            "{'model': ..., 'config': ..., ...}). The notebook does not attempt a generic "
            "alternative loader."
        )
    return raw


def _resolve_encoder_arch(enc_config: dict) -> tuple:
    name = str(enc_config.get("name", "")).strip()
    source = str(enc_config.get("source", "")).strip().lower()
    if source != "cnn" or name not in _SUPPORTED_CNN_ENCODERS:
        raise MammoFMConfigError(
            f"This helper does not support the Mammo-FM checkpoint image encoder "
            f"(source={enc_config.get('source')!r}, name={name!r}). It supports only the "
            f"EfficientNet CNN encoders used by official checkpoints: "
            f"{sorted(_SUPPORTED_CNN_ENCODERS)}. There is no automatic fallback to a different "
            "architecture such as DINOv2, RAD-DINO, or a generic ViT."
        )
    return _SUPPORTED_CNN_ENCODERS[name]


def build_mammofm_checkpoint_architecture(
    num_classes: int = 1,
    dropout: float = 0.1,
):
    """Build the exact architecture used by complete project checkpoints.

    This constructor intentionally does not load or claim to provide pretrained
    Mammo-FM weights.  It is only the architecture half of a strict full-state
    restore performed by :class:`ArchitectureAdapter`; training initialization
    continues to require the authorized upstream Mammo-FM archive.
    """
    arch_name, out_dim = _resolve_encoder_arch(
        {"source": "cnn", "name": PROJECT_CHECKPOINT_ENCODER_NAME}
    )
    image_encoder = MammoFMImageEncoder(arch_name=arch_name, out_dim=out_dim)
    return MammoFMClassifier(
        image_encoder,
        hidden_size=out_dim,
        num_classes=num_classes,
        dropout=dropout,
    )


def build_mammofm_model(
    hf_repo: Optional[str] = DEFAULT_HF_REPO,
    checkpoint_name: Optional[str] = DEFAULT_CHECKPOINT_NAME,
    use_local_checkpoint: bool = False,
    local_checkpoint_path: Optional[str] = None,
    num_classes: int = 1,
    dropout: float = 0.1,
    local_files_only: bool = False,
):
    """Build a trainable Mammo-FM classifier (EfficientNet-B5 plus a linear head).

    Return ``(model, mean, std, img_size, hidden_size, backend, source_desc)``. Raise
    ``MammoFMConfigError`` if the checkpoint is unconfigured, unreachable, or incompatible
    with the expected architecture; never silently substitute a generic backbone.
    """
    ckpt_path = _resolve_checkpoint_path(
        hf_repo,
        checkpoint_name,
        use_local_checkpoint,
        local_checkpoint_path,
        local_files_only,
    )
    raw = _load_raw_checkpoint(ckpt_path)

    try:
        enc_config = raw["config"]["model"]["image_encoder"]
    except (KeyError, TypeError) as exc:
        raise MammoFMConfigError(
            f"Checkpoint '{ckpt_path}' has no config['model']['image_encoder']; the Mammo-FM "
            "image-encoder architecture cannot be determined."
        ) from exc

    arch_name, out_dim = _resolve_encoder_arch(enc_config)
    image_encoder = MammoFMImageEncoder(arch_name=arch_name, out_dim=out_dim)

    full_state_dict = raw["model"]
    cleaned = {_strip_known_prefixes(k): v for k, v in full_state_dict.items()}
    backbone_keys = set(image_encoder.model.state_dict().keys())
    matched_keys = backbone_keys & set(cleaned.keys())
    match_ratio = len(matched_keys) / max(len(backbone_keys), 1)
    if match_ratio < 0.5:
        raise MammoFMConfigError(
            f"Checkpoint '{ckpt_path}' does not appear compatible with architecture "
            f"'{arch_name}': only {len(matched_keys)}/{len(backbone_keys)} tensors match by "
            "name. Check the official checkpoint file; the notebook will not present a "
            "partial or invalid load as Mammo-FM."
        )
    missing, _ = image_encoder.model.load_state_dict(cleaned, strict=False)
    if missing:
        warnings.warn(
            f"Mammo-FM checkpoint: {len(missing)} backbone tensors were not found and retain "
            f"their random initialization, for example {list(missing)[:3]}. The final '_fc' "
            "layer is expected to be missing because it is absent from the official checkpoint "
            "and this helper extracts only pooled features."
        )

    model = MammoFMClassifier(image_encoder, hidden_size=out_dim, num_classes=num_classes, dropout=dropout)
    source_desc = (
        f"local_checkpoint:{ckpt_path}" if use_local_checkpoint
        else f"huggingface:{hf_repo}/{checkpoint_name}"
    )
    source_desc += f" (encoder={arch_name}, config_name={enc_config.get('name')!r})"
    return model, DEFAULT_MAMMOFM_MEAN, DEFAULT_MAMMOFM_STD, DEFAULT_IMG_SIZE, out_dim, "efficientnet_pytorch", source_desc


# ---------------------------------------------------------------------------
# Freeze/unfreeze controls for real fine-tuning (no adapter or LoRA)
# ---------------------------------------------------------------------------

def freeze_backbone_all(model: MammoFMClassifier) -> None:
    for p in model.image_encoder.parameters():
        p.requires_grad_(False)


def unfreeze_head(model: MammoFMClassifier) -> None:
    for p in model.classifier.parameters():
        p.requires_grad_(True)
    for p in model.dropout.parameters():
        p.requires_grad_(True)


def unfreeze_last_n_blocks(model: MammoFMClassifier, n: int = 2) -> None:
    """Unfreeze the last ``n`` EfficientNet-B5 MBConv stages and final convolutional head.

    This exposes `_conv_head` and `_bn1` as well, mirroring the partial fine-tuning of the last
    Transformer blocks in ViT backbones. It is full parameter fine-tuning, not an adapter.
    """
    backbone = model.image_encoder.model  # efficientnet_pytorch.EfficientNet
    blocks = list(backbone._blocks)
    if not blocks:
        raise MammoFMConfigError("Could not locate the Mammo-FM backbone's MBConv blocks.")
    for block in blocks[-int(n):]:
        for p in block.parameters():
            p.requires_grad_(True)
    for p in backbone._conv_head.parameters():
        p.requires_grad_(True)
    for p in backbone._bn1.parameters():
        p.requires_grad_(True)


def unfreeze_all(model: MammoFMClassifier) -> None:
    for p in model.parameters():
        p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Preprocessing/dataset (grayscale to RGB with official Mammo-FM normalization)
# ---------------------------------------------------------------------------

def _apply_clahe(arr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE as in official Mammo-CLIP/Mammo-FM pre-training preprocessing.

    See `configs/transform/clahe.yaml` upstream. The notebook uses this only when
    `USE_CLAHE_PREPROCESSING=True`; it is disabled by default for protocol consistency.
    """
    try:
        import cv2
    except ImportError as exc:
        raise MammoFMConfigError(
            "USE_CLAHE_PREPROCESSING=True requires the OpenCV (cv2) package. Install it with: "
            "pip install opencv-python-headless"
        ) from exc
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(np.clip(arr, 0, 255).astype(np.uint8)).astype(np.float32)


class MammoFMDataset(Dataset):
    """Load grayscale mammograms and apply official Mammo-CLIP/Mammo-FM normalization.

    Replicate each image to three channels because the official EfficientNet-B5 checkpoint's
    `_conv_stem` expects RGB input. Normalize each image to [0, 1], then standardize it with
    the scalar `mean` and `std` values rather than ImageNet statistics.
    """

    def __init__(self, paths, labels, mean: float, std: float, img_size: int,
                 augment: bool = False, use_clahe: bool = False, metadata=None):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.mean = float(mean)
        self.std = float(std)
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.use_clahe = bool(use_clahe)
        self.metadata = list(metadata) if metadata is not None else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        with Image.open(path) as image:
            gray = image.convert("L").resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(gray, dtype=np.float32)

        if self.use_clahe:
            arr = _apply_clahe(arr)

        if self.augment:
            # Preprocessing already orients the tissue. Avoiding flips preserves the anatomical
            # convention used by the project's other classifiers.
            arr = np.clip(arr + np.random.uniform(-0.05 * 255, 0.05 * 255), 0.0, 255.0)

        arr_min, arr_max = float(arr.min()), float(arr.max())
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)
        arr = (arr - self.mean) / self.std

        tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1).float()  # One channel to RGB.
        result = (tensor, torch.tensor(label, dtype=torch.float32))
        return (*result, self.metadata[idx]) if self.metadata is not None else result


def make_mammofm_dataloader(df: pd.DataFrame, path_col: str, label_col: str,
                             mean: float, std: float, img_size: int,
                             batch_size: int = 8, shuffle: bool = False, augment: bool = False,
                             use_clahe: bool = False, seed: int = 42, num_workers: int = 2,
                             drop_last: Optional[bool] = None, metadata=None) -> DataLoader:
    """Build a DataLoader consistent with `maxvit_utils.make_dataloader`.

    The default `drop_last=None` is equivalent to `drop_last=shuffle`. Mammo-FM notebooks set
    it explicitly to `False` on training loaders so that a partial final batch is retained for
    the small, imbalanced real dataset.
    """
    dataset = MammoFMDataset(
        paths=df[path_col].values, labels=df[label_col].values,
        mean=mean, std=std, img_size=img_size, augment=augment, use_clahe=use_clahe, metadata=metadata,
    )
    generator = torch.Generator().manual_seed(seed)
    if drop_last is None:
        drop_last = shuffle
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=drop_last,
        generator=generator if shuffle else None,
    )


# ---------------------------------------------------------------------------
# Dataset loaders with source tracking (real/synthetic/augmented)
# ---------------------------------------------------------------------------

def image_paths(directory) -> list:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    )


def load_real_split(base_path, split: str) -> pd.DataFrame:
    """Load preprocessed real images for one train, validation, or test split."""
    rows = []
    split_dir = Path(base_path) / "data" / "processed" / split
    for label in (0, 1):
        for path in image_paths(split_dir / str(label)):
            rows.append({
                "processed_path": str(path), "cancer": label,
                "source": "real", "source_detail": f"real_{split}", "split": split,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No real images found in {split_dir}")
    return df


def load_synthetic_both(root, source_name: str, split_label: str = "train") -> pd.DataFrame:
    """Load filtered positive and negative synthetic images from `root`."""
    root = Path(root)
    rows = []
    for folder, label in (("negative", 0), ("positive", 1)):
        folder_path = root / folder
        paths = image_paths(folder_path)
        if not paths:
            raise FileNotFoundError(
                f"No '{folder}' synthetic images found in {folder_path}. Run the corresponding "
                "generation/filtering notebook first or check the configured path."
            )
        for path in paths:
            rows.append({
                "processed_path": str(path), "cancer": label,
                "source": "synthetic", "source_detail": source_name, "split": split_label,
            })
    return pd.DataFrame(rows)


def load_augmented_positive(base_path, split_label: str = "train") -> pd.DataFrame:
    """Load the traditional-augmentation dataset (`data/real_augmented`)."""
    aug_dir = Path(base_path) / "data" / "real_augmented"
    rows = []
    if not aug_dir.is_dir():
        raise FileNotFoundError(
            f"Directory {aug_dir} is missing. Run "
            "notebooks/1_preprocessing/02_Data_Augmentation_Trad.ipynb first."
        )
    for path in image_paths(aug_dir):
        match = _LABEL_RE.search(path.name)
        if match is None:
            continue
        rows.append({
            "processed_path": str(path), "cancer": int(match.group(1)),
            "source": "augmented", "source_detail": "traditional_positive_augmentation", "split": split_label,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No recognized augmented images found in {aug_dir}")
    return df


def print_counts(name: str, df: pd.DataFrame) -> None:
    labels = df["cancer"].astype(int)
    print(f"{name:24s} total={len(df):5d} | healthy(0)={(labels == 0).sum():5d} | cancer(1)={(labels == 1).sum():5d}")


def source_table(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(df["source_detail"], df["cancer"])
    table = table.rename(columns={0: "healthy_0", 1: "cancer_1"})
    table["total"] = table.sum(axis=1)
    return table


def check_duplicate_paths(df: pd.DataFrame, path_col: str = "processed_path") -> int:
    """Warn about duplicate paths in the combined training dataframe."""
    dup_mask = df.duplicated(subset=[path_col], keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        examples = df.loc[dup_mask, path_col].unique()[:5].tolist()
        warnings.warn(f"Found {n_dup} duplicate paths in the combined dataset: {examples} ...")
    return n_dup


def check_no_split_overlap(splits: dict) -> None:
    """Verify that no real path appears in more than one split.

    ``splits`` maps split names to dataframes containing a ``processed_path`` column.
    """
    seen = {}
    for name, df in splits.items():
        for p in df["processed_path"]:
            if p in seen and seen[p] != name:
                raise AssertionError(
                    f"Data leakage detected: '{p}' appears in both '{seen[p]}' and '{name}'."
                )
            seen[p] = name


# ---------------------------------------------------------------------------
# Training with mixed precision, gradient clipping, and gradient accumulation
# ---------------------------------------------------------------------------

def train_one_epoch_amp(model, loader, optimizer, criterion, device, scaler=None,
                         grad_clip_norm: Optional[float] = None, accumulation_steps: int = 1,
                         start_batch: int = 0, global_step: int = 0, max_optimizer_updates: int | None = None,
                         on_optimizer_step=None, on_before_optimizer_step=None, on_batch_processed=None) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    model.train()
    refreeze_batchnorm(model)  # Frozen EfficientNet BatchNorm2d layers remain in eval mode.
    total_loss, n_seen = 0.0, 0
    y_true, y_prob = [], []
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if step < start_batch:
            continue
        imgs, labels, metadata = (*batch, None) if len(batch) == 2 else batch
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, enabled=(scaler is not None and device.type == "cuda")):
            logits = model(imgs).squeeze(-1)
            loss = criterion(logits, labels) / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if on_batch_processed is not None:
            on_batch_processed(metadata)

        is_last_batch = (step + 1) == len(loader)
        if (step + 1) % accumulation_steps == 0 or is_last_batch:
            next_step = global_step + 1
            if on_before_optimizer_step is not None:
                on_before_optimizer_step(next_step, step)
            if grad_clip_norm is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if on_optimizer_step is not None:
                on_optimizer_step(global_step, step)
            if max_optimizer_updates is not None and global_step >= max_optimizer_updates:
                break

        total_loss += loss.item() * accumulation_steps * imgs.size(0)
        n_seen += imgs.size(0)
        y_true.extend(labels.detach().cpu().numpy())
        y_prob.extend(torch.sigmoid(logits).detach().cpu().numpy())

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float("nan")
    return {"loss": total_loss / max(n_seen, 1), "auc": auc, "pr_auc": pr_auc,
            "global_step": global_step, "last_batch": step}


@torch.no_grad()
def evaluate_amp(model, loader, criterion, device) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    model.eval()
    total_loss, n_seen = 0.0, 0
    y_true, y_prob = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs).squeeze(-1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        n_seen += imgs.size(0)
        y_true.extend(labels.cpu().numpy())
        y_prob.extend(torch.sigmoid(logits).cpu().numpy())

    y_true_arr, y_prob_arr = np.array(y_true), np.array(y_prob)
    try:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    except ValueError:
        pr_auc = float("nan")
    return {"loss": total_loss / max(n_seen, 1), "auc": auc, "pr_auc": pr_auc,
            "y_true": y_true_arr, "y_prob": y_prob_arr}


def fit_mammofm(model, train_loader, val_loader, optimizer, criterion, epochs: int, device,
                early_stopping=None, checkpoint=None, csv_logger=None, lr_scheduler=None,
                use_amp: bool = True, grad_clip_norm: Optional[float] = 1.0,
                accumulation_steps: int = 1, start_epoch: int = 1, start_batch: int = 0,
                global_step: int = 0, max_optimizer_updates: int | None = None,
                scaler=None, on_optimizer_step=None, on_before_optimizer_step=None,
                on_epoch_begin=None, on_epoch_end=None, on_batch_processed=None,
                resume_history: dict | None = None) -> History:
    """Run a Keras-style fit loop with AMP, clipping, and gradient accumulation.

    This is distinct from `maxvit_utils.fit` because it explicitly handles the EfficientNet
    encoder's BatchNorm2d layers, which are absent from the ViT backbones, in addition to AMP
    and gradient accumulation.

    When present, `resume_history` seeds the history with epochs from an earlier segment so a
    periodic or final checkpoint never loses metrics recorded before interruption.
    """
    history = History()
    if resume_history:
        for key, values in resume_history.items():
            history.history[key] = list(values)
    scaler = scaler or (torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None)

    for epoch in range(start_epoch, epochs + 1):
        if on_epoch_begin is not None:
            on_epoch_begin(epoch)
        train_metrics = train_one_epoch_amp(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, grad_clip_norm=grad_clip_norm, accumulation_steps=accumulation_steps,
                start_batch=start_batch if epoch == start_epoch else 0, global_step=global_step,
            max_optimizer_updates=max_optimizer_updates, on_optimizer_step=on_optimizer_step,
            on_before_optimizer_step=on_before_optimizer_step, on_batch_processed=on_batch_processed,
        )
        global_step = train_metrics.pop("global_step")
        val_metrics = evaluate_amp(model, val_loader, criterion, device)
        history.append(train_metrics, val_metrics)

        print(f"Epoch {epoch}/{epochs} - loss: {train_metrics['loss']:.4f} - ROC-AUC: {train_metrics['auc']:.4f} "
              f"- PR-AUC: {train_metrics['pr_auc']:.4f} - val_loss: {val_metrics['loss']:.4f} "
              f"- val_ROC-AUC: {val_metrics['auc']:.4f} - val_PR-AUC: {val_metrics['pr_auc']:.4f}")

        if csv_logger is not None:
            csv_logger.log({
                "epoch": epoch, "loss": train_metrics["loss"], "auc": train_metrics["auc"],
                "pr_auc": train_metrics["pr_auc"], "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc"], "val_pr_auc": val_metrics["pr_auc"],
            })
        if checkpoint is not None:
            checkpoint.step(val_metrics["pr_auc"], model, val_metrics["loss"])
        if lr_scheduler is not None:
            lr_scheduler.step(val_metrics["pr_auc"])
        improved = False
        if early_stopping is not None:
            before = early_stopping.best
            early_stopping.step(val_metrics["pr_auc"], model, val_metrics["loss"])
            improved = early_stopping.best != before
        if on_epoch_end is not None:
            on_epoch_end(epoch, global_step, scaler, history, val_metrics, improved)
        if early_stopping is not None and early_stopping.stop:
            print(f"Early stopping at epoch {epoch} (best val_pr_auc={early_stopping.best:.4f})")
            break
        if max_optimizer_updates is not None and global_step >= max_optimizer_updates:
            break

    if early_stopping is not None:
        early_stopping.restore(model)
    history.global_step = global_step
    history.scaler = scaler
    return history


def predict_with_probs(model, df: pd.DataFrame, path_col: str, label_col: str,
                        mean: float, std: float, img_size: int, batch_size: int, device) -> tuple:
    """Predict with shuffle disabled so `y_true` and `y_prob` preserve dataframe order."""
    loader = make_mammofm_dataloader(
        df, path_col, label_col, mean, std, img_size, batch_size=batch_size, shuffle=False,
    )
    model.eval()
    y_prob = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(-1)
            y_prob.extend(torch.sigmoid(logits).cpu().numpy())
    y_prob = np.array(y_prob)
    y_true = df[label_col].values.astype(int)
    if len(y_prob) != len(df):
        raise RuntimeError("Prediction count does not match dataframe rows; was shuffling enabled?")
    return y_true, y_prob


# ---------------------------------------------------------------------------
# Extended metrics
# ---------------------------------------------------------------------------

def compute_full_metrics(y_true, y_prob, threshold: float, split: str,
                          experiment_name: str, config_name: str, extra: Optional[dict] = None):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, _, _ = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    metrics = {
        "experiment_name": experiment_name,
        "config": config_name,
        "split": split,
        "n_samples": int(len(y_true)),
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc_average_precision": round(float(average_precision_score(y_true, y_prob)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall_sensitivity": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "specificity": round(specificity, 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Healthy", "Cancer"], zero_division=0
        ),
    }
    if extra:
        metrics.update(extra)
    return metrics, y_pred


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_history(history: History, title_suffix: str, save_path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax1.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set(title=f"Loss {title_suffix}", xlabel="Epoch", ylabel="Loss")
    ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(history.history["auc"], label="Train AUC", linewidth=2)
    ax2.plot(history.history["val_auc"], label="Val AUC", linewidth=2)
    ax2.set(title=f"AUC {title_suffix}", xlabel="Epoch", ylabel="AUC")
    ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_pr_confusion(y_true, y_prob, y_pred, title: str, save_path) -> None:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Healthy", "Cancer"]); axes[0].set_yticklabels(["Healthy", "Cancer"])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual"); axes[0].set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            axes[0].text(j, i, cm[i, j], ha="center", va="center",
                         fontsize=13, fontweight="bold", color=color)

    axes[1].plot(fpr, tpr, lw=2, label=f"AUC={auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR"); axes[1].set_title("ROC curve")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(rec, prec, lw=2, label=f"AP={ap:.4f}")
    axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision"); axes[2].set_title("Precision-Recall curve")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_calibration_curve(y_true, y_prob, title: str, save_path, n_bins: int = 10) -> None:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    import matplotlib.pyplot as plt

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, "o-", label=f"Brier={brier:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def save_training_history_csv(history: History, path) -> None:
    path = Path(path)
    df = pd.DataFrame(history.history)
    df.insert(0, "epoch", np.arange(1, len(df) + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_training_history_combined_csv(history_phase1: History, history_phase2: History, path) -> None:
    """Save both training phases in one CSV with an explicit `phase` column.

    This does not replace the compatibility files `training_history_fase1.csv` and
    `training_history_fase2.csv`, which remain separate artifacts.
    """
    df1 = pd.DataFrame(history_phase1.history)
    df1.insert(0, "epoch", np.arange(1, len(df1) + 1))
    df1.insert(1, "phase", "head_training")

    df2 = pd.DataFrame(history_phase2.history)
    df2.insert(0, "epoch", np.arange(1, len(df2) + 1))
    df2.insert(1, "phase", "fine_tuning")

    combined = pd.concat([df1, df2], ignore_index=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


def save_test_predictions(df_test: pd.DataFrame, y_true, y_prob, threshold: float,
                           path_col: str, source_col: str, save_path, split_name: str = "test") -> pd.DataFrame:
    out = pd.DataFrame({
        "path": df_test[path_col].values,
        "label": np.asarray(y_true).astype(int),
        "probability": np.asarray(y_prob).astype(float),
        "prediction": (np.asarray(y_prob) >= threshold).astype(int),
        "split": split_name,
        "source": df_test[source_col].values if source_col in df_test.columns else "real",
    })
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_path, index=False)
    return out


def auto_interpret(test_metrics: dict, val_metrics: dict, config_name: str) -> str:
    """Generate a short automatic interpretation of test-set results."""
    auc = test_metrics["roc_auc"]
    if auc >= 0.9:
        quality = "excellent"
    elif auc >= 0.8:
        quality = "good"
    elif auc >= 0.7:
        quality = "fair"
    elif auc >= 0.6:
        quality = "weak"
    else:
        quality = "poor (close to chance)"

    delta_auc = test_metrics["roc_auc"] - val_metrics["roc_auc"]
    overfit_note = ""
    if abs(delta_auc) > 0.05:
        overfit_note = (
            f"\n- **Warning**: validation-to-test ROC-AUC gap of {delta_auc:+.4f}; this may "
            "indicate validation overfitting or a distribution shift between splits."
        )

    lines = [
        f"### Automatic interpretation — {config_name}",
        "",
        f"- Test-set ROC-AUC: **{test_metrics['roc_auc']:.4f}** -> {quality} discrimination.",
        f"- PR-AUC (average precision): **{test_metrics['pr_auc_average_precision']:.4f}**.",
        f"- Balanced Accuracy: **{test_metrics['balanced_accuracy']:.4f}**, F1: **{test_metrics['f1']:.4f}**.",
        f"- Sensitivity/Recall: **{test_metrics['recall_sensitivity']:.4f}**, "
        f"Specificity: **{test_metrics['specificity']:.4f}**.",
        f"- Brier score: **{test_metrics['brier_score']:.4f}** (lower means better-calibrated probabilities)."
        + overfit_note,
    ]
    return "\n".join(lines)


__all__ = [
    "BinaryFocalLoss", "CSVLogger", "EarlyStopping", "History", "ModelCheckpoint",
    "bootstrap_balanced", "compute_pos_weight", "count_trainable_params", "optimal_threshold_youden",
    "refreeze_batchnorm",
    "MammoFMConfigError", "MammoFMImageEncoder", "MammoFMClassifier",
    "build_mammofm_model",
    "freeze_backbone_all", "unfreeze_head", "unfreeze_last_n_blocks", "unfreeze_all",
    "MammoFMDataset", "make_mammofm_dataloader",
    "image_paths", "load_real_split", "load_synthetic_both", "load_augmented_positive",
    "print_counts", "source_table", "check_duplicate_paths", "check_no_split_overlap",
    "train_one_epoch_amp", "evaluate_amp", "fit_mammofm", "predict_with_probs",
    "compute_full_metrics",
    "plot_training_history", "plot_roc_pr_confusion", "plot_calibration_curve",
    "seed_everything", "save_json", "save_training_history_csv", "save_training_history_combined_csv",
    "save_test_predictions",
    "auto_interpret",
    "DEFAULT_HF_REPO", "DEFAULT_CHECKPOINT_NAME", "DEFAULT_IMG_SIZE",
    "DEFAULT_MAMMOFM_MEAN", "DEFAULT_MAMMOFM_STD",
]
