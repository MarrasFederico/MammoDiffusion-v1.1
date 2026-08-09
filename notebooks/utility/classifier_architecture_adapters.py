"""Training and validation adapters for MaxViT-512 and Mammo-FM."""
from __future__ import annotations

import os
import random
from pathlib import Path

ARCHITECTURES = ("maxvit512", "mammofm")
ACCOUNTING_FIELDS = (
    "real_negative_seen",
    "real_positive_seen",
    "traditional_augmented_seen",
    "finetuned_synthetic_seen",
    "fromscratch_synthetic_seen",
)


def _pytorch_resume_position(payload: dict) -> tuple[int, int]:
    if int(payload.get("batch_index", -1)) != -1:
        raise RuntimeError("classifier resume checkpoints must be saved at a validated block boundary")
    return int(payload["epoch"]), 0


def _dataframes(train_rows, validation_rows):
    import pandas as pd

    return pd.DataFrame(train_rows), pd.DataFrame(validation_rows)


def _accounting_metadata(rows):
    output = []
    for index, row in enumerate(rows):
        source = str(row.get("source", "")).lower()
        family = str(row.get("synthetic_family", "")).lower()
        if source == "augmented":
            field = "traditional_augmented_seen"
        elif source == "synthetic":
            if family == "finetuned":
                field = "finetuned_synthetic_seen"
            elif family == "from_scratch":
                field = "fromscratch_synthetic_seen"
            else:
                raise ValueError(f"synthetic row has invalid synthetic_family: {row.get('synthetic_family')!r}")
        else:
            field = "real_positive_seen" if int(row.get("label", 0)) == 1 else "real_negative_seen"
        output.append(
            {
                "sample_id": str(row.get("image_id") or row.get("sample_id") or index),
                "source": str(row.get("source", "unknown")),
                "accounting_field": field,
            }
        )
    return output


def _torch_payload(raw):
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
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


class ArchitectureAdapter:
    def __init__(self, architecture, policy, root):
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unsupported architecture: {architecture}")
        self.architecture = architecture
        self.policy = policy
        self.root = Path(root)

    def build_model(self, pretrained=True, seed=42):
        random.seed(seed)
        if self.architecture == "maxvit512":
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            import maxvit_utils as utils

            return utils.build_maxvit_model(num_classes=1, pretrained=pretrained)
        if self.architecture == "mammofm":
            import mammofm_utils as utils

            if not pretrained:
                return utils.build_mammofm_checkpoint_architecture()
            return utils.build_mammofm_model(
                hf_repo=utils.DEFAULT_HF_REPO,
                checkpoint_name=utils.DEFAULT_CHECKPOINT_NAME,
                use_local_checkpoint=False,
                local_files_only=True,
            )[0]
        raise ValueError(f"unsupported architecture: {self.architecture}")

    def build_train_dataloaders(self, train_rows, validation_rows, seed=42):
        train_df, val_df = _dataframes(train_rows, validation_rows)
        batch = int(self.policy["physical_batch_size"])
        workers = int(self.policy.get("dataloader_workers", 0))
        if self.architecture == "maxvit512":
            import maxvit_utils as utils

            mean = self.policy["normalization"]["mean"]
            std = self.policy["normalization"]["std"]
            size = int(self.policy["input_size"][0])
            return (
                utils.make_dataloader(
                    train_df, "processed_path", "label", mean, std, size, batch,
                    True, True, seed, workers, metadata=_accounting_metadata(train_rows)
                ),
                utils.make_dataloader(
                    val_df, "processed_path", "label", mean, std, size, batch,
                    False, False, seed, workers
                ),
            )
        if self.architecture == "mammofm":
            import mammofm_utils as utils

            return (
                utils.make_mammofm_dataloader(
                    train_df, "processed_path", "label", utils.DEFAULT_MAMMOFM_MEAN,
                    utils.DEFAULT_MAMMOFM_STD, utils.DEFAULT_IMG_SIZE, batch_size=batch,
                    shuffle=True, augment=True, seed=seed, num_workers=workers,
                    drop_last=False, metadata=_accounting_metadata(train_rows)
                ),
                utils.make_mammofm_dataloader(
                    val_df, "processed_path", "label", utils.DEFAULT_MAMMOFM_MEAN,
                    utils.DEFAULT_MAMMOFM_STD, utils.DEFAULT_IMG_SIZE, batch_size=batch,
                    shuffle=False, augment=False, seed=seed, num_workers=workers,
                    drop_last=False
                ),
            )
        raise ValueError(f"unsupported architecture: {self.architecture}")

    def build_validation_dataloader(self, validation_rows, seed=42):
        return self.build_train_dataloaders(validation_rows, validation_rows, seed)[1]

    def save_checkpoint(self, model, path):
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        torch.save(
            {
                "schema_version": 1,
                "architecture": self.architecture,
                "model_state_dict": model.state_dict(),
            },
            temporary,
        )
        os.replace(temporary, path)
        return path

    def load_checkpoint(self, path, model=None, strict=True):
        import torch

        model = model or self.build_model(pretrained=False)
        state = _torch_payload(torch.load(path, map_location="cpu", weights_only=False))
        result = model.load_state_dict(state, strict=strict)
        if strict and (result.missing_keys or result.unexpected_keys):
            raise ValueError(
                f"checkpoint mismatch: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
        return model

    def validate_checkpoint_compatibility(self, path):
        try:
            self.load_checkpoint(path, strict=True)
        except Exception as exc:
            return False, str(exc)
        return True, "compatible"

    def predict_validation(self, checkpoint_path, validation_rows, seed=42, **_):
        import torch
        import maxvit_utils as utils

        loader = self.build_validation_dataloader(validation_rows, seed)
        model = self.load_checkpoint(checkpoint_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        labels, probabilities = utils.predict_probs(model, loader, device)
        return {
            "labels": labels.astype(int).tolist(),
            "probabilities": probabilities.astype(float).tolist(),
            "sample_ids": [
                row.get("image_id", str(index))
                for index, row in enumerate(validation_rows)
            ],
        }

    def estimate_memory_profile(self, **_):
        return {
            "resource_profile": self.policy.get("expected_vram_profile"),
            "physical_batch_size": self.policy.get("physical_batch_size"),
            "effective_batch_size": self.policy.get("effective_batch_size"),
        }


def get_adapter(architecture, policy=None, root=None):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture: {architecture}")
    if policy is None:
        raise ValueError("training policy is required")
    return ArchitectureAdapter(architecture, policy, root or Path.cwd())


__all__ = ["ARCHITECTURES", "ArchitectureAdapter", "get_adapter", "_torch_payload"]
