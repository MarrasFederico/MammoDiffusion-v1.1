"""Shared utilities for the MaxViT-Tiny-512 classifiers (PyTorch/timm).

Provides the dataset and preprocessing pipeline, two-stage training with
Keras-style callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, and
CSVLogger), and binary focal loss.
"""
from __future__ import annotations

import copy
import csv
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

MODEL_NAME = "maxvit_tiny_tf_512.in1k"


# ---------------------------------------------------------------------------
# Model normalization and configuration
# ---------------------------------------------------------------------------

def build_maxvit_model(num_classes: int = 1, pretrained: bool = True):
    import timm

    model = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MammoDataset(Dataset):
    """Load grayscale mammograms, convert them to three channels, and normalize for MaxViT."""

    def __init__(self, paths, labels, mean, std, img_size: int, augment: bool = False, metadata=None):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.img_size = img_size
        self.augment = augment
        self.metadata = list(metadata) if metadata is not None else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(path).convert("L").resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [0, 1]

        if self.augment:
            # Preprocessing already orients tissue to the left; avoiding flips
            # preserves the anatomical convention used by the classifier notebooks.
            delta = np.random.uniform(-0.05, 0.05)
            arr = np.clip(arr + delta, 0.0, 1.0)

        tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)  # One channel -> three.
        tensor = (tensor - self.mean) / self.std
        result = (tensor, torch.tensor(label, dtype=torch.float32))
        return (*result, self.metadata[idx]) if self.metadata is not None else result


def make_dataloader(df: pd.DataFrame, path_col: str, label_col: str, mean, std, img_size: int,
                     batch_size: int, shuffle: bool = False, augment: bool = False,
                     seed: int = 42, num_workers: int = 2, metadata=None) -> DataLoader:
    ds = MammoDataset(df[path_col].values, df[label_col].values, mean, std, img_size, augment=augment,
                      metadata=metadata)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), generator=generator if shuffle else None,
    )


# ---------------------------------------------------------------------------
# Freeze and unfreeze the backbone (stem + stages), analogous to partial ResNet unfreezing
# ---------------------------------------------------------------------------

def freeze_all(model) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_head(model) -> None:
    for p in model.head.parameters():
        p.requires_grad = True


def unfreeze_stages_from(model, start_stage: int) -> None:
    """Unfreeze stages from the zero-indexed ``start_stage`` onward."""
    for i, stage in enumerate(model.stages):
        if i >= start_stage:
            for p in stage.parameters():
                p.requires_grad = True


def unfreeze_all(model) -> None:
    for p in model.parameters():
        p.requires_grad = True


def refreeze_batchnorm(model) -> None:
    """Return frozen BatchNorm2d layers to eval mode after ``model.train()``.

    This preserves ImageNet statistics for layers that remain frozen, matching
    the behavior of ``layer.trainable=False`` for Keras BatchNorm layers.
    """
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            frozen = not any(p.requires_grad for p in mod.parameters(recurse=False))
            if frozen:
                mod.eval()


def count_trainable_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class BinaryFocalLoss(nn.Module):
    """PyTorch equivalent of keras.losses.BinaryFocalCrossentropy with class balancing."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


def compute_pos_weight(labels) -> torch.Tensor:
    """Return a BCEWithLogitsLoss ``pos_weight`` equivalent to Keras balanced class weights."""
    labels = np.asarray(labels)
    n_pos = max(int((labels == 1).sum()), 1)
    n_neg = max(int((labels == 0).sum()), 1)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Keras-style callbacks
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max", min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best = -np.inf if mode == "max" else np.inf
        self.best_secondary = np.inf
        self.wait = 0
        self.best_state: Optional[dict] = None
        self.stop = False

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float, model, secondary: float | None = None) -> None:
        primary_improved = self._is_improvement(value)
        tied_with_better_secondary = (
            secondary is not None and np.isclose(value, self.best, rtol=1e-12, atol=1e-12)
            and float(secondary) < self.best_secondary
        )
        if primary_improved or tied_with_better_secondary:
            self.best = value
            if secondary is not None:
                self.best_secondary = float(secondary)
            self.wait = 0
            if self.restore_best_weights:
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop = True

    def restore(self, model) -> None:
        if self.restore_best_weights and self.best_state is not None:
            model.load_state_dict(self.best_state)


class ModelCheckpoint:
    def __init__(self, filepath: str, mode: str = "max"):
        self.filepath = filepath
        self.mode = mode
        self.best = -np.inf if mode == "max" else np.inf
        self.best_secondary = np.inf

    def step(self, value: float, model, secondary: float | None = None) -> None:
        improved = value > self.best if self.mode == "max" else value < self.best
        improved = improved or (secondary is not None and np.isclose(value, self.best, rtol=1e-12, atol=1e-12)
                                and float(secondary) < self.best_secondary)
        if improved:
            self.best = value
            if secondary is not None:
                self.best_secondary = float(secondary)
            torch.save(model.state_dict(), self.filepath)


class CSVLogger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._header_written = False

    def log(self, row: dict) -> None:
        write_header = not self._header_written and not os.path.isfile(self.filepath)
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._header_written = True


@dataclass
class History:
    history: dict = field(default_factory=lambda: {
        "loss": [], "auc": [], "pr_auc": [], "val_loss": [], "val_auc": [], "val_pr_auc": [],
    })

    def append(self, train_metrics: dict, val_metrics: dict) -> None:
        self.history["loss"].append(train_metrics["loss"])
        self.history["auc"].append(train_metrics["auc"])
        self.history["pr_auc"].append(train_metrics["pr_auc"])
        self.history["val_loss"].append(val_metrics["loss"])
        self.history["val_auc"].append(val_metrics["auc"])
        self.history["val_pr_auc"].append(val_metrics["pr_auc"])


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")  # Only one class is present in the batch or epoch.
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")
    return {"auc": float(auc), "pr_auc": float(pr_auc)}


def train_one_epoch(model, loader: DataLoader, optimizer, criterion, device) -> dict:
    model.train()
    refreeze_batchnorm(model)
    total_loss, n_seen = 0.0, 0
    y_true, y_prob = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        n_seen += imgs.size(0)
        y_true.extend(labels.detach().cpu().numpy())
        y_prob.extend(torch.sigmoid(logits).detach().cpu().numpy())

    metrics = _binary_metrics(np.array(y_true), np.array(y_prob))
    metrics["loss"] = total_loss / max(n_seen, 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader: DataLoader, criterion, device) -> dict:
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

    y_true, y_prob = np.array(y_true), np.array(y_prob)
    metrics = _binary_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / max(n_seen, 1)
    metrics["y_true"] = y_true
    metrics["y_prob"] = y_prob
    return metrics


def fit(model, train_loader, val_loader, optimizer, criterion, epochs: int, device,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint: Optional[ModelCheckpoint] = None,
        csv_logger: Optional[CSVLogger] = None,
        lr_scheduler=None) -> History:
    history = History()

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append(train_metrics, val_metrics)

        print(f"Epoch {epoch}/{epochs} - loss: {train_metrics['loss']:.4f} - auc: {train_metrics['auc']:.4f} "
              f"- val_loss: {val_metrics['loss']:.4f} - val_auc: {val_metrics['auc']:.4f}")

        if csv_logger is not None:
            csv_logger.log({
                "epoch": epoch, "loss": train_metrics["loss"], "auc": train_metrics["auc"],
                "val_loss": val_metrics["loss"], "val_auc": val_metrics["auc"],
            })
        if checkpoint is not None:
            checkpoint.step(val_metrics["pr_auc"], model, val_metrics["loss"])
        if lr_scheduler is not None:
            lr_scheduler.step(val_metrics["pr_auc"])
        if early_stopping is not None:
            early_stopping.step(val_metrics["pr_auc"], model, val_metrics["loss"])
            if early_stopping.stop:
                print(f"Early stopping at epoch {epoch} (best val_pr_auc={early_stopping.best:.4f})")
                break

    if early_stopping is not None:
        early_stopping.restore(model)
    return history


def predict_probs(model, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(-1)
            y_prob.extend(torch.sigmoid(logits).cpu().numpy())
            y_true.extend(labels.numpy())
    return np.array(y_true), np.array(y_prob)


def optimal_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def bootstrap_balanced(y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
                        n_rounds: int = 1000, seed: int = 42) -> dict:
    """Balanced bootstrap: resample positive and negative cases in equal numbers."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n = min(len(pos_idx), len(neg_idx))
    out = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "ROC_AUC": []}

    for _ in range(n_rounds):
        sel_pos = rng.choice(pos_idx, size=n, replace=True)
        sel_neg = rng.choice(neg_idx, size=n, replace=True)
        idx = np.concatenate([sel_pos, sel_neg])
        yt, yp = y_true[idx], y_prob[idx]
        pred = (yp >= threshold).astype(int)

        out["Accuracy"].append(accuracy_score(yt, pred))
        out["Precision"].append(precision_score(yt, pred, pos_label=1, zero_division=0))
        out["Recall"].append(recall_score(yt, pred, pos_label=1, zero_division=0))
        out["F1"].append(f1_score(yt, pred, pos_label=1, zero_division=0))
        out["ROC_AUC"].append(roc_auc_score(yt, yp))

    return out
