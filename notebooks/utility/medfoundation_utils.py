"""Utility condivise per i classificatori basati su foundation model medici.

Fornisce l'equivalente di `maxvit_utils.py` per backbone pre-addestrati su dati
medici (di default `microsoft/rad-dino`, ViT-B/14 pre-addestrato su radiografie
toraciche/mammografie), riusando le callback in stile Keras (EarlyStopping,
ModelCheckpoint, CSVLogger), la focal loss binaria e il training a due fasi
gia' presenti in `maxvit_utils`.

L'obiettivo e' rispondere a una D2c: *un backbone pre-addestrato su dati medici
batte MaxViT-Tiny-512 (pre-addestrato ImageNet-1k) a parita' di configurazione
Baseline / Real+Synth?*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# I moduli condivisi con MaxViT vengono importati direttamente per evitare duplicazione.
from maxvit_utils import (
    BinaryFocalLoss,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    compute_pos_weight,
    count_trainable_params,
    fit,
    make_gradcam_heatmap,
    optimal_threshold_youden,
    predict_probs,
    show_gradcam,
)

DEFAULT_MODEL_NAME = "microsoft/rad-dino"
DEFAULT_IMG_SIZE = 518  # ViT-B/14 patch => 14*37 = 518
DEFAULT_IMAGENET_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_normalization_medfoundation(image_processor=None):
    """Restituisce mean/std/input_size letti dal `image_processor` HF quando disponibile."""
    if image_processor is None:
        return DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD, DEFAULT_IMG_SIZE
    mean = tuple(image_processor.image_mean) if hasattr(image_processor, "image_mean") else DEFAULT_IMAGENET_MEAN
    std = tuple(image_processor.image_std) if hasattr(image_processor, "image_std") else DEFAULT_IMAGENET_STD
    size_dict = getattr(image_processor, "size", None) or {}
    img_size = int(size_dict.get("height") or size_dict.get("shortest_edge") or DEFAULT_IMG_SIZE)
    return mean, std, img_size


class MedFoundationClassifier(nn.Module):
    """Backbone HF (RAD-DINO / DINOv2 medico) + testa lineare per classificazione binaria."""

    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int = 1, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        # Foundation ViT-based: prendere il CLS token, cioe' last_hidden_state[:, 0]
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            features = outputs.last_hidden_state[:, 0]
        else:
            raise RuntimeError("Backbone HF: output senza pooler_output / last_hidden_state.")
        return self.classifier(self.dropout(features))


def build_medfoundation_model(
    model_name: str = DEFAULT_MODEL_NAME,
    num_classes: int = 1,
    dropout: float = 0.1,
):
    """Costruisce (backbone, model, image_processor) per il fine-tuning del foundation medico."""
    from transformers import AutoImageProcessor, AutoModel

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    hidden_size = int(getattr(backbone.config, "hidden_size", 768))
    model = MedFoundationClassifier(backbone, hidden_size=hidden_size,
                                    num_classes=num_classes, dropout=dropout)
    return model, image_processor, hidden_size


# ---------------------------------------------------------------------------
# Dataset e dataloader
# ---------------------------------------------------------------------------

class MammoMedFoundationDataset(Dataset):
    """Carica mammografie grayscale, le converte a 3 canali e le normalizza per il foundation model."""

    def __init__(self, paths, labels, mean, std, img_size: int, augment: bool = False):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.img_size = int(img_size)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        with Image.open(path) as image:
            gray = image.convert("L").resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(gray, dtype=np.float32) / 255.0
        if self.augment:
            arr = np.clip(arr + np.random.uniform(-0.05, 0.05), 0.0, 1.0)
        tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)  # (3,H,W)
        tensor = (tensor - self.mean) / self.std
        return tensor, torch.tensor(label, dtype=torch.float32)


def make_medfoundation_dataloader(df, path_column, label_column,
                                  mean, std, img_size,
                                  batch_size=8, shuffle=False, augment=False,
                                  num_workers=0, seed=42, drop_last=None):
    """Costruisce un DataLoader coerente con lo stile di `maxvit_utils.make_dataloader`.

    `drop_last` di default resta `None`, che preserva il comportamento storico
    (`drop_last=shuffle`, usato dai notebook 24/25). Passare esplicitamente
    `drop_last=False` per non perdere immagini nell'ultimo batch parziale del
    training (sicuro qui perche' il backbone e' un ViT/ Transformer con
    LayerNorm, non BatchNorm: un batch di dimensione 1 non e' un problema).
    """
    dataset = MammoMedFoundationDataset(
        paths=df[path_column].tolist(),
        labels=df[label_column].values,
        mean=mean, std=std, img_size=img_size,
        augment=augment,
    )
    generator = torch.Generator().manual_seed(seed)
    if drop_last is None:
        drop_last = shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Politiche di freeze / unfreeze specifiche del backbone HF ViT-based
# ---------------------------------------------------------------------------

def freeze_backbone_all(model: MedFoundationClassifier) -> None:
    for param in model.backbone.parameters():
        param.requires_grad_(False)


def unfreeze_head(model: MedFoundationClassifier) -> None:
    for param in model.classifier.parameters():
        param.requires_grad_(True)
    for param in model.dropout.parameters():
        param.requires_grad_(True)


def unfreeze_last_n_blocks(model: MedFoundationClassifier, n: int = 2) -> None:
    """Scongela gli ultimi n block Transformer + la LayerNorm finale del backbone."""
    encoder = getattr(model.backbone, "encoder", None) or getattr(model.backbone, "layer", None)
    layers = None
    if encoder is not None:
        layers = getattr(encoder, "layer", None) or getattr(encoder, "layers", None)
    if layers is None:
        raise RuntimeError(
            "Impossibile individuare i block del backbone HF per lo sblocco parziale."
        )
    for block in list(layers)[-int(n):]:
        for param in block.parameters():
            param.requires_grad_(True)
    layernorm = getattr(model.backbone, "layernorm", None) or getattr(model.backbone, "norm", None)
    if layernorm is not None:
        for param in layernorm.parameters():
            param.requires_grad_(True)


def unfreeze_all(model: MedFoundationClassifier) -> None:
    for param in model.parameters():
        param.requires_grad_(True)


__all__ = [
    "BinaryFocalLoss", "CSVLogger", "EarlyStopping", "ModelCheckpoint",
    "compute_pos_weight", "count_trainable_params", "fit",
    "make_gradcam_heatmap", "optimal_threshold_youden", "predict_probs", "show_gradcam",
    "MedFoundationClassifier", "MammoMedFoundationDataset",
    "build_medfoundation_model", "make_medfoundation_dataloader",
    "resolve_normalization_medfoundation",
    "freeze_backbone_all", "unfreeze_head", "unfreeze_last_n_blocks", "unfreeze_all",
    "DEFAULT_MODEL_NAME", "DEFAULT_IMG_SIZE",
    "DEFAULT_IMAGENET_MEAN", "DEFAULT_IMAGENET_STD",
]
