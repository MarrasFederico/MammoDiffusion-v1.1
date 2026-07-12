"""Canonical train/validation adapters for the four classifier architectures.

Imports of TensorFlow, torch, timm and transformers are deliberately lazy.  The tiny adapter
is a dependency-free executable contract used by integration tests and notebook dry-runs; it
is selected only with ``--tiny``/``MAMMO_CLASSIFIER_TINY=1`` and is never an implicit fallback.
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path


ARCHITECTURES = ("resnet50", "maxvit512", "mammofm", "raddino")


def _dataframes(train_rows, validation_rows):
    import pandas as pd
    return pd.DataFrame(train_rows), pd.DataFrame(validation_rows)


def _torch_payload(raw):
    """Normalize common checkpoint wrappers and a uniform DataParallel prefix."""
    if not isinstance(raw, dict):
        raise ValueError("checkpoint must be a mapping")
    state = raw
    for key in ("state_dict", "model_state_dict", "model"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not state or not all(isinstance(key, str) for key in state):
        raise ValueError("checkpoint does not contain a usable state dict")
    has_module = [key.startswith("module.") for key in state]
    if any(has_module) and not all(has_module):
        raise ValueError("checkpoint has a non-uniform module. prefix")
    if all(has_module):
        state = {key[len("module."):]: value for key, value in state.items()}
    return state


class TinyAdapter:
    """Small deterministic logistic learner proving the orchestration end to end."""

    def __init__(self, architecture, policy, root):
        self.architecture, self.policy, self.root = architecture, policy, Path(root)

    def build_model(self, seed=42, **_):
        return {"bias": (int(seed) % 13 - 6) / 100.0}

    def build_train_dataloaders(self, train_rows, validation_rows, **_):
        return list(train_rows), list(validation_rows)

    def build_validation_dataloader(self, validation_rows, **_):
        return list(validation_rows)

    def save_checkpoint(self, model, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"architecture": self.architecture, "state_dict": model}, sort_keys=True) + "\n")
        return path

    def load_checkpoint(self, path, **_):
        payload = json.loads(Path(path).read_text())
        if payload.get("architecture") != self.architecture:
            raise ValueError("tiny checkpoint architecture mismatch")
        return payload["state_dict"]

    def validate_checkpoint_compatibility(self, path):
        try:
            self.load_checkpoint(path)
        except Exception as exc:
            return False, str(exc)
        return True, "compatible"

    def train(self, train_rows, validation_rows, checkpoint_path, seed=42, **_):
        labels = [int(row["label"]) for row in train_rows]
        prevalence = sum(labels) / max(len(labels), 1)
        model = {"bias": prevalence + (int(seed) % 7) * 1e-4}
        self.save_checkpoint(model, checkpoint_path)
        return {"checkpoint": str(checkpoint_path), "history": {"loss": [0.0]}, "optimizer_updates": 1}

    def predict_validation(self, checkpoint_path, validation_rows, **_):
        model = self.load_checkpoint(checkpoint_path)
        bias = float(model["bias"])
        labels, probs = [], []
        for index, row in enumerate(validation_rows):
            label = int(row["label"])
            labels.append(label)
            # Deterministic, non-perfect probabilities; orchestration tests need both classes.
            probs.append(max(0.001, min(0.999, 0.25 + 0.5 * label + bias * 0.01 + index * 1e-7)))
        return {"labels": labels, "probabilities": probs,
                "sample_ids": [row.get("image_id", str(i)) for i, row in enumerate(validation_rows)]}

    def predict_locked_test(self, *_args, **_kwargs):
        raise PermissionError("locked test is not available through classifier adapters")

    def estimate_memory_profile(self, **_):
        return {"mode": "tiny", "estimated_peak_mb": 32}


class ArchitectureAdapter:
    def __init__(self, architecture, policy, root):
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unsupported architecture: {architecture}")
        self.architecture, self.policy, self.root = architecture, policy, Path(root)

    def build_model(self, pretrained=True, seed=42):
        random.seed(seed)
        if self.architecture == "resnet50":
            from resnet50_utils import build_resnet50_model
            return build_resnet50_model(tuple(self.policy["input_size"]), pretrained=pretrained)[0]
        if self.architecture == "maxvit512":
            import maxvit_utils as utils
            return utils.build_maxvit_model(num_classes=1, pretrained=pretrained)
        if self.architecture == "mammofm":
            import mammofm_utils as utils
            local = os.environ.get("MAMMOFM_LOCAL_CHECKPOINT_PATH")
            return utils.build_mammofm_model(
                hf_repo=os.environ.get("MAMMOFM_HF_REPO", utils.DEFAULT_HF_REPO),
                checkpoint_name=os.environ.get("MAMMOFM_CHECKPOINT_NAME", utils.DEFAULT_CHECKPOINT_NAME),
                use_local_checkpoint=bool(local), local_checkpoint_path=local,
            )[0]
        import medfoundation_utils as utils
        return utils.build_medfoundation_model(os.environ.get("RADDINO_MODEL_PATH", utils.DEFAULT_MODEL_NAME))[0]

    def build_train_dataloaders(self, train_rows, validation_rows, seed=42):
        train_df, val_df = _dataframes(train_rows, validation_rows)
        batch = int(self.policy["physical_batch_size"])
        workers = int(self.policy.get("dataloader_workers", 0))
        if self.architecture == "resnet50":
            from resnet50_utils import make_dataset
            size = tuple(self.policy["input_size"])
            return (make_dataset(train_rows, size, batch, True, seed),
                    make_dataset(validation_rows, size, batch, False, seed))
        if self.architecture == "maxvit512":
            import maxvit_utils as utils
            mean, std = self.policy["normalization"]["mean"], self.policy["normalization"]["std"]
            size = int(self.policy["input_size"][0])
            return (utils.make_dataloader(train_df, "processed_path", "label", mean, std, size, batch, True, True, seed, workers),
                    utils.make_dataloader(val_df, "processed_path", "label", mean, std, size, batch, False, False, seed, workers))
        if self.architecture == "mammofm":
            import mammofm_utils as utils
            return (utils.make_mammofm_dataloader(train_df, "processed_path", "label", utils.DEFAULT_MAMMOFM_MEAN,
                    utils.DEFAULT_MAMMOFM_STD, utils.DEFAULT_IMG_SIZE, batch_size=batch, shuffle=True,
                    augment=True, seed=seed, num_workers=workers, drop_last=False),
                    utils.make_mammofm_dataloader(val_df, "processed_path", "label", utils.DEFAULT_MAMMOFM_MEAN,
                    utils.DEFAULT_MAMMOFM_STD, utils.DEFAULT_IMG_SIZE, batch_size=batch, shuffle=False,
                    augment=False, seed=seed, num_workers=workers, drop_last=False))
        import medfoundation_utils as utils
        mean, std, size = utils.resolve_normalization_medfoundation()
        return (utils.make_medfoundation_dataloader(train_df, "processed_path", "label", mean, std, size, batch, True, True, workers, seed, False),
                utils.make_medfoundation_dataloader(val_df, "processed_path", "label", mean, std, size, batch, False, False, workers, seed, False))

    def build_validation_dataloader(self, validation_rows, seed=42):
        return self.build_train_dataloaders(validation_rows, validation_rows, seed)[1]

    def save_checkpoint(self, model, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.architecture == "resnet50":
            model.save(path)
        else:
            import torch
            torch.save({"schema_version": 1, "architecture": self.architecture,
                        "model_state_dict": model.state_dict()}, path)
        return path

    def load_checkpoint(self, path, model=None, strict=True):
        if self.architecture == "resnet50":
            from resnet50_utils import load_keras_checkpoint
            return load_keras_checkpoint(Path(path))
        import torch
        model = model or self.build_model(pretrained=False)
        state = _torch_payload(torch.load(path, map_location="cpu", weights_only=False))
        result = model.load_state_dict(state, strict=strict)
        if strict and (result.missing_keys or result.unexpected_keys):
            raise ValueError(f"checkpoint mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
        return model

    def validate_checkpoint_compatibility(self, path):
        try:
            self.load_checkpoint(path, strict=True)
        except Exception as exc:
            return False, str(exc)
        return True, "compatible"

    def _epochs(self, train_size):
        batches = max(1, math.ceil(train_size / int(self.policy["physical_batch_size"])))
        accumulation = int(self.policy.get("gradient_accumulation_steps", 1))
        return min(int(self.policy.get("max_epochs_secondary_limit", 60)),
                   max(1, math.ceil(int(self.policy["max_optimizer_updates"]) * accumulation / batches)))

    def train(self, train_rows, validation_rows, checkpoint_path, seed=42, **_):
        model = self.build_model(pretrained=True, seed=seed)
        train_loader, val_loader = self.build_train_dataloaders(train_rows, validation_rows, seed)
        epochs = self._epochs(len(train_rows))
        if self.architecture == "resnet50":
            import tensorflow as tf
            # Two protocol phases: head first, then conv4+ with BatchNorm frozen.
            backbone = next(layer for layer in model.layers if layer.name == "resnet50")
            from resnet50_utils import set_fine_tuning, set_head_training
            set_head_training(backbone)
            model.compile(tf.keras.optimizers.Adam(1e-3), "binary_crossentropy",
                          metrics=[tf.keras.metrics.AUC(name="auc")])
            head_epochs = max(1, epochs // 5)
            h1 = model.fit(train_loader, validation_data=val_loader, epochs=head_epochs, verbose=2)
            set_fine_tuning(backbone)
            model.compile(tf.keras.optimizers.Adam(1e-5),
                          tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.75),
                          metrics=[tf.keras.metrics.AUC(name="auc")])
            h2 = model.fit(train_loader, validation_data=val_loader, epochs=max(1, epochs-head_epochs), verbose=2,
                           callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10,
                                                                      restore_best_weights=True)])
            history = {key: list(h1.history.get(key, [])) + list(h2.history.get(key, []))
                       for key in set(h1.history) | set(h2.history)}
        else:
            import torch
            import maxvit_utils as common
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            if self.architecture == "maxvit512":
                common.freeze_all(model); common.unfreeze_head(model)
                if hasattr(model, "stages"): common.unfreeze_stages_from(model, max(0, len(model.stages)-2))
            elif self.architecture == "mammofm":
                import mammofm_utils as u
                u.freeze_backbone_all(model); u.unfreeze_head(model); u.unfreeze_last_n_blocks(model, 2)
            else:
                import medfoundation_utils as u
                u.freeze_backbone_all(model); u.unfreeze_head(model); u.unfreeze_last_n_blocks(model, 2)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=float(self.policy["training_phases"][0]["learning_rate"]),
                                          weight_decay=float(self.policy.get("weight_decay", 0.0)))
            criterion = common.BinaryFocalLoss()
            early = common.EarlyStopping(patience=int(self.policy["early_stopping"]["patience"]), mode="max")
            # The Mammo-FM loop is architecture-agnostic and adds AMP, clipping and gradient
            # accumulation missing from the older MaxViT loop; using it keeps the registered
            # effective batch size unchanged for all three torch families.
            import mammofm_utils as amp_utils
            history_obj = amp_utils.fit_mammofm(
                model, train_loader, val_loader, optimizer, criterion, epochs, device,
                early_stopping=early, use_amp=bool(self.policy.get("amp", False)),
                grad_clip_norm=self.policy.get("gradient_clipping"),
                accumulation_steps=int(self.policy.get("gradient_accumulation_steps", 1)),
            )
            history = history_obj.history if hasattr(history_obj, "history") else vars(history_obj)
        self.save_checkpoint(model, Path(checkpoint_path))
        return {"checkpoint": str(checkpoint_path), "history": history,
                "optimizer_updates_limit": int(self.policy["max_optimizer_updates"]), "epochs": epochs}

    def predict_validation(self, checkpoint_path, validation_rows, seed=42, **_):
        loader = self.build_validation_dataloader(validation_rows, seed)
        model = self.load_checkpoint(checkpoint_path)
        if self.architecture == "resnet50":
            from resnet50_utils import predict_validation
            labels, probabilities = predict_validation(model, loader)
        else:
            import torch
            import maxvit_utils as utils
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            labels, probabilities = utils.predict_probs(model, loader, device)
            labels, probabilities = labels.astype(int).tolist(), probabilities.astype(float).tolist()
        return {"labels": labels, "probabilities": probabilities,
                "sample_ids": [row.get("image_id", str(i)) for i, row in enumerate(validation_rows)]}

    def predict_locked_test(self, *_args, **_kwargs):
        raise PermissionError("locked test is not available through the normal classifier runner")

    def estimate_memory_profile(self, **_):
        return {"resource_profile": self.policy.get("expected_vram_profile"),
                "physical_batch_size": self.policy.get("physical_batch_size"),
                "effective_batch_size": self.policy.get("effective_batch_size")}


def get_adapter(architecture, policy=None, root=None, tiny=False):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture: {architecture}")
    if policy is None:
        raise ValueError("training policy is required")
    use_tiny = bool(tiny) or os.environ.get("MAMMO_CLASSIFIER_TINY") == "1"
    return (TinyAdapter if use_tiny else ArchitectureAdapter)(architecture, policy, root or Path.cwd())


__all__ = ["ARCHITECTURES", "ArchitectureAdapter", "TinyAdapter", "get_adapter", "_torch_payload"]
