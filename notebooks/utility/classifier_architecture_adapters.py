"""Training and validation adapters for MaxViT-512 and Mammo-FM."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np


ARCHITECTURES = ("maxvit512", "mammofm")
ACCOUNTING_FIELDS = (
    "real_negative_seen",
    "real_positive_seen",
    "traditional_augmented_seen",
    "finetuned_synthetic_seen",
    "fromscratch_synthetic_seen",
)


class _FixedBatchLoader:
    def __init__(self, loader, batch_count):
        self.loader = loader
        self.batch_count = int(batch_count)

    def __len__(self):
        return self.batch_count

    def __iter__(self):
        emitted = 0
        while emitted < self.batch_count:
            cycle_count = 0
            for batch in self.loader:
                yield batch
                emitted += 1
                cycle_count += 1
                if emitted >= self.batch_count:
                    return
            if cycle_count == 0:
                raise RuntimeError("training loader is empty")


def _pytorch_resume_position(payload: dict) -> tuple[int, int]:
    return int(payload["epoch"]), 0


def _dataframes(train_rows, validation_rows):
    import pandas as pd

    return pd.DataFrame(train_rows), pd.DataFrame(validation_rows)


def _accounting_metadata(rows):
    output = []
    for index, row in enumerate(rows):
        source = str(row.get("source", "")).lower()
        field = (
            "traditional_augmented_seen"
            if "augment" in source
            else "finetuned_synthetic_seen"
            if "finetuned" in source
            else "fromscratch_synthetic_seen"
            if "from_scratch" in source or "fromscratch" in source
            else "real_positive_seen"
            if int(row.get("label", 0)) == 1
            else "real_negative_seen"
        )
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


def _normalized_gpu_uuid(value) -> str:
    return str(value or "").strip().lower().removeprefix("gpu-")


def _gpu_resume_provenance(payload, runtime_gpu_uuid):
    checkpoint_gpu_uuid = payload.get("gpu_uuid") if payload is not None else None
    gpu_changed = bool(
        payload is not None
        and checkpoint_gpu_uuid
        and runtime_gpu_uuid
        and _normalized_gpu_uuid(checkpoint_gpu_uuid)
        != _normalized_gpu_uuid(runtime_gpu_uuid)
    )
    return {
        "checkpoint_gpu_uuid": checkpoint_gpu_uuid,
        "runtime_gpu_uuid": runtime_gpu_uuid,
        "gpu_changed": gpu_changed,
    }


def _validate_resume_payload(payload: dict) -> None:
    required = (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "global_step",
        "rng_states",
        "source_accounting",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"resume checkpoint is incomplete: {missing}")
    accounting = payload["source_accounting"]
    if accounting.get("accounting_mode") != "actual":
        raise RuntimeError("resume checkpoint lacks actual source accounting")
    missing_accounting = [field for field in ACCOUNTING_FIELDS if field not in accounting]
    if missing_accounting:
        raise RuntimeError(f"resume checkpoint source accounting is incomplete: {missing_accounting}")


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

            local = os.environ.get("MAMMOFM_LOCAL_CHECKPOINT_PATH")
            if not local or not Path(local).is_file():
                raise RuntimeError(
                    "MAMMO-FM requires MAMMOFM_LOCAL_CHECKPOINT_PATH to reference a local checkpoint"
                )
            return utils.build_mammofm_model(
                hf_repo=os.environ.get("MAMMOFM_HF_REPO", utils.DEFAULT_HF_REPO),
                checkpoint_name=os.environ.get(
                    "MAMMOFM_CHECKPOINT_NAME", utils.DEFAULT_CHECKPOINT_NAME
                ),
                use_local_checkpoint=True,
                local_checkpoint_path=local,
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

    def _epochs(self):
        return min(
            int(self.policy.get("max_epochs_secondary_limit", 60)),
            max(
                1,
                math.ceil(
                    int(self.policy["max_optimizer_updates"])
                    / int(self.policy["validation_interval_updates"])
                ),
            ),
        )

    def train(self, train_rows, validation_rows, checkpoint_path, seed=42, **context):
        import classifier_checkpoint_io as ckio

        run_dir = Path(context["run_dir"])
        expected = {
            key: context[key]
            for key in (
                "architecture",
                "experiment_id",
                "dataset_variant_id",
                "training_policy",
                "config_signature",
                "dataset_signature",
            )
        }
        expected["seed"] = int(seed)
        resume, resume_source = ckio.load_resume_checkpoint(run_dir, expected)
        checkpoint_files_exist = any(run_dir.glob("checkpoint_*"))
        if resume is None and resume_source == "no resume checkpoint" and checkpoint_files_exist:
            raise RuntimeError(
                f"checkpoint files exist in {run_dir}, but no resumable checkpoint is available. "
                "Move or remove this run directory to start from zero."
            )
        if resume is None and resume_source != "no resume checkpoint":
            raise RuntimeError(
                f"resume checkpoint(s) exist in {run_dir} but are corrupt or incompatible: "
                f"{resume_source}. Move or remove this run directory to start from zero."
            )
        if resume is not None:
            _validate_resume_payload(resume)

        runtime_gpu_uuid = context.get("gpu_uuid")
        gpu_provenance = _gpu_resume_provenance(resume, runtime_gpu_uuid)

        model = self.build_model(pretrained=True, seed=seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        train_loader, val_loader = self.build_train_dataloaders(
            train_rows, validation_rows, seed
        )
        accumulation = int(self.policy.get("gradient_accumulation_steps", 1))
        train_loader = _FixedBatchLoader(
            train_loader,
            int(self.policy["validation_interval_updates"]) * accumulation,
        )
        epochs = self._epochs()

        import torch
        import maxvit_utils as common
        import mammofm_utils as amp_utils

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if self.architecture == "maxvit512":
            common.freeze_all(model)
            common.unfreeze_head(model)
            if hasattr(model, "stages"):
                common.unfreeze_stages_from(model, max(0, len(model.stages) - 2))
        else:
            amp_utils.freeze_backbone_all(model)
            amp_utils.unfreeze_head(model)
            amp_utils.unfreeze_last_n_blocks(model, 2)

        parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        (run_dir / "model_summary.json").write_text(
            json.dumps(
                {
                    "architecture": self.architecture,
                    "parameters": parameters,
                    "trainable_parameters": trainable_parameters,
                    "input_size": self.policy["input_size"],
                },
                indent=2,
            )
            + "\n"
        )

        parameters_to_optimize = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            parameters_to_optimize,
            lr=float(self.policy["training_phases"][0]["learning_rate"]),
            weight_decay=float(self.policy.get("weight_decay", 0.0)),
        )
        criterion = common.BinaryFocalLoss()
        early = common.EarlyStopping(
            patience=int(self.policy["early_stopping"]["patience"]), mode="max"
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(self.policy["scheduler_params"]["factor"]),
            patience=int(self.policy["scheduler_params"]["patience"]),
            min_lr=float(self.policy["scheduler_params"]["min_lr"]),
        )
        scaler = (
            torch.amp.GradScaler("cuda")
            if bool(self.policy.get("amp")) and device.type == "cuda"
            else None
        )
        start_epoch, start_batch, global_step = 1, 0, 0
        best_epoch = None
        prior_history = {}
        actual_source_counts = {field: 0 for field in ACCOUNTING_FIELDS}

        if resume is not None:
            model.load_state_dict(resume["model_state_dict"], strict=True)
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            scheduler.load_state_dict(resume["scheduler_state_dict"])
            if scaler is not None and resume.get("scaler_state_dict"):
                scaler.load_state_dict(resume["scaler_state_dict"])
            start_epoch, start_batch = _pytorch_resume_position(resume)
            global_step = int(resume["global_step"])
            early.wait = int(resume.get("early_stopping_counter", 0))
            if resume.get("best_metric") is not None:
                early.best = float(resume["best_metric"])
            early.best_secondary = float(
                resume.get("best_validation_loss", float("inf"))
            )
            best_epoch = resume.get("best_epoch")
            prior_history = dict(resume.get("history", {}))
            actual_source_counts.update(
                {
                    field: int(resume["source_accounting"].get(field, 0))
                    for field in ACCOUNTING_FIELDS
                }
            )
            states = resume["rng_states"]
            if states.get("python"):
                random.setstate(states["python"])
            if states.get("numpy"):
                np.random.set_state(states["numpy"])
            if states.get("torch") is not None:
                torch.set_rng_state(states["torch"])
            if torch.cuda.is_available() and states.get("torch_cuda"):
                torch.cuda.set_rng_state_all(states["torch_cuda"])

        def actual_accounting():
            return {
                "schema_version": 1,
                "accounting_mode": "actual",
                **actual_source_counts,
                "total_samples_seen": sum(actual_source_counts.values()),
            }

        def record_processed_batch(metadata):
            if not isinstance(metadata, dict) or "accounting_field" not in metadata:
                raise RuntimeError(
                    "training batch is missing sample_id/source accounting metadata"
                )
            for field in metadata["accounting_field"]:
                if field not in actual_source_counts:
                    raise RuntimeError(f"unknown source accounting field: {field}")
                actual_source_counts[field] += 1

        segment = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
        current = {
            "epoch": start_epoch,
            "history": prior_history,
            "best_epoch": best_epoch,
        }

        def save_torch(step, batch, epoch=None, history=None, best=False, best_epoch=None):
            epoch = current["epoch"] if epoch is None else epoch
            history = current["history"] if history is None else history
            payload = {
                **expected,
                **gpu_provenance,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "epoch": epoch,
                "batch_index": batch,
                "global_step": step,
                "checkpoint_metric": "val_pr_auc",
                "best_metric": getattr(early, "best", None),
                "best_validation_loss": getattr(early, "best_secondary", None),
                "best_epoch": current["best_epoch"] if best_epoch is None else best_epoch,
                "early_stopping_counter": getattr(early, "wait", 0),
                "history": history or {},
                "source_accounting": actual_accounting(),
                "gpu_uuid": runtime_gpu_uuid,
                "rng_states": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "torch_cuda": (
                        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                    ),
                },
                "resume_segment_id": segment,
            }
            ckio.save_resume_checkpoint(run_dir, payload, best=best)

        interval = int(self.policy.get("checkpoint_interval_updates", 250))
        warmup = int(self.policy.get("warmup_updates", 0))
        target_lr = float(self.policy["training_phases"][0]["learning_rate"])

        def before_optimizer_step(step, _batch):
            if warmup and step <= warmup:
                for group in optimizer.param_groups:
                    group["lr"] = target_lr * step / warmup

        def epoch_begin(epoch):
            current["epoch"] = epoch

        def periodic(step, batch):
            if step % interval == 0:
                save_torch(step, batch)

        def epoch_end(epoch, step, _scaler, history_object, values, improved):
            current["epoch"] = epoch
            history_object.history.setdefault("learning_rate", []).append(
                float(optimizer.param_groups[0]["lr"])
            )
            history_object.history.setdefault("optimizer_steps", []).append(int(step))
            current["history"] = history_object.history
            if improved:
                current["best_epoch"] = epoch
            scheduler.step(values["pr_auc"])
            save_torch(step, -1, epoch + 1, history_object.history, best=improved)

        history_object = amp_utils.fit_mammofm(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            epochs,
            device,
            early_stopping=early,
            use_amp=bool(self.policy.get("amp", False)),
            grad_clip_norm=self.policy.get("gradient_clipping"),
            accumulation_steps=accumulation,
            lr_scheduler=None,
            start_epoch=start_epoch,
            start_batch=start_batch,
            global_step=global_step,
            max_optimizer_updates=int(self.policy["max_optimizer_updates"]),
            scaler=scaler,
            on_optimizer_step=periodic,
            on_before_optimizer_step=before_optimizer_step,
            on_epoch_begin=epoch_begin,
            on_epoch_end=epoch_end,
            resume_history=prior_history,
            on_batch_processed=record_processed_batch,
        )
        history = (
            history_object.history
            if hasattr(history_object, "history")
            else vars(history_object)
        )

        best_path = ckio.resume_checkpoint_path(run_dir, "checkpoint_best")
        if best_path.is_file():
            best_payload = ckio.read_resume_checkpoint(best_path)
            if best_payload.get("best_metric") is not None and (
                getattr(early, "best", float("-inf")) is None
                or best_payload["best_metric"] >= getattr(early, "best", float("-inf"))
            ):
                model.load_state_dict(best_payload["model_state_dict"], strict=True)
                best_epoch = best_payload.get("best_epoch")

        self.save_checkpoint(model, Path(checkpoint_path))
        return {
            "checkpoint": str(checkpoint_path),
            "history": history,
            "resumed_from": resume_source if resume is not None else None,
            "optimizer_updates_limit": int(self.policy["max_optimizer_updates"]),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "source_accounting": actual_accounting(),
            **gpu_provenance,
        }

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

    def predict_locked_test(self, checkpoint_path, test_rows, *, lock_verified=False, **kwargs):
        if not lock_verified:
            raise PermissionError("a verified downstream test lock is required")
        return self.predict_validation(checkpoint_path, test_rows, **kwargs)

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
