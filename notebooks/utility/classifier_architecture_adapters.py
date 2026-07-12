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
import hashlib
import numpy as np
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

    def train(self, train_rows, validation_rows, checkpoint_path, seed=42, **context):
        import classifier_checkpoint_io as ckio
        labels = [int(row["label"]) for row in train_rows]
        prevalence = sum(labels) / max(len(labels), 1)
        model = {"bias": prevalence + (int(seed) % 7) * 1e-4}
        if context.get("run_dir"):
            expected = {key: context[key] for key in ("architecture", "experiment_id", "dataset_variant_id",
                        "training_policy", "config_signature", "dataset_signature")}; expected["seed"] = int(seed)
            prior, source = ckio.load_resume_checkpoint(Path(context["run_dir"]), expected)
            global_step = int((prior or {}).get("global_step", 0)) + 1
            ckio.save_resume_checkpoint(Path(context["run_dir"]), {**expected, "model_state_dict": model,
                "optimizer_state_dict": {"step": global_step}, "scheduler_state_dict": {"step": global_step},
                "scaler_state_dict": None, "epoch": global_step, "batch_index": -1, "global_step": global_step,
                "best_metric": prevalence, "best_epoch": global_step, "early_stopping_counter": 0,
                "history": {"loss": [0.0]}, "rng_states": {"python": random.getstate()},
                "resume_segment_id": f"tiny-{global_step}"}, best=True)
        self.save_checkpoint(model, checkpoint_path)
        return {"checkpoint": str(checkpoint_path), "history": {"loss": [0.0]},
                "optimizer_updates": global_step if context.get("run_dir") else 1,
                "resumed_from": source if context.get("run_dir") and prior else None}

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
            weights = Path.home() / ".keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
            if pretrained and not weights.is_file():
                raise RuntimeError("ResNet50 ImageNet weights are not cached locally; downloads in workers are disabled")
            from resnet50_utils import build_resnet50_model
            return build_resnet50_model(tuple(self.policy["input_size"]), pretrained=pretrained)[0]
        if self.architecture == "maxvit512":
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            import maxvit_utils as utils
            return utils.build_maxvit_model(num_classes=1, pretrained=pretrained)
        if self.architecture == "mammofm":
            import mammofm_utils as utils
            local = os.environ.get("MAMMOFM_LOCAL_CHECKPOINT_PATH")
            if not local or not Path(local).is_file():
                raise RuntimeError("MAMMO-FM matrix training requires MAMMOFM_LOCAL_CHECKPOINT_PATH; downloads in workers are disabled")
            return utils.build_mammofm_model(
                hf_repo=os.environ.get("MAMMOFM_HF_REPO", utils.DEFAULT_HF_REPO),
                checkpoint_name=os.environ.get("MAMMOFM_CHECKPOINT_NAME", utils.DEFAULT_CHECKPOINT_NAME),
                use_local_checkpoint=bool(local), local_checkpoint_path=local,
            )[0]
        import medfoundation_utils as utils
        local = os.environ.get("RADDINO_MODEL_PATH")
        if not local or not Path(local).is_dir():
            raise RuntimeError("RAD-DINO matrix training requires a local RADDINO_MODEL_PATH; downloads in workers are disabled")
        os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return utils.build_medfoundation_model(local)[0]

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

    def train(self, train_rows, validation_rows, checkpoint_path, seed=42, **context):
        import classifier_checkpoint_io as ckio
        run_dir = Path(context["run_dir"])
        expected = {key: context[key] for key in (
            "architecture", "experiment_id", "dataset_variant_id", "training_policy",
            "config_signature", "dataset_signature")}
        expected["seed"] = int(seed)
        resume, resume_source = ckio.load_resume_checkpoint(run_dir, expected)
        if resume is None and resume_source != "no resume checkpoint":
            # A resume file existed (checkpoint_latest and/or checkpoint_previous) but every
            # one was corrupted or scientifically incompatible - silently falling through to
            # start_epoch=1 here would discard however many hours of training already ran
            # without anyone deciding that on purpose.
            allow_discard = os.environ.get("ALLOW_DISCARD_INVALID_RESUME") == "True" or bool(context.get("allow_discard_invalid_resume"))
            if not allow_discard:
                raise RuntimeError(
                    f"resume checkpoint(s) present for {run_dir} but all invalid/incompatible "
                    f"({resume_source}); refusing to silently restart from scratch. Set "
                    "ALLOW_DISCARD_INVALID_RESUME=True (env var) to explicitly discard and start over."
                )
        model = self.build_model(pretrained=True, seed=seed)
        results_dir = Path(context.get("run_dir", run_dir))
        if context.get("run_dir"):
            # Mirror the run layout under results while keeping checkpoints operational-only.
            project_root = self.root
            results_dir = (project_root / "results/classifiers_matrix" / self.architecture /
                           context["dataset_variant_id"] / context["training_policy"] / f"seed_{seed}")
            results_dir.mkdir(parents=True, exist_ok=True)
            if self.architecture == "resnet50":
                params = int(model.count_params()); trainable = sum(int(v.shape.num_elements()) for v in model.trainable_weights)
            else:
                params = sum(p.numel() for p in model.parameters()); trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            (results_dir / "model_summary.txt").write_text(f"architecture: {self.architecture}\nparameters: {params}\ntrainable_before_policy: {trainable}\n")
            (results_dir / "model_architecture.json").write_text(json.dumps({"architecture":self.architecture,"parameters":params,
                "trainable_before_policy":trainable,"input_size":self.policy["input_size"]},indent=2)+"\n")
        train_loader, val_loader = self.build_train_dataloaders(train_rows, validation_rows, seed)
        epochs = self._epochs(len(train_rows))
        best_epoch = None
        if self.architecture == "resnet50":
            import tensorflow as tf
            # Two protocol phases: head first, then conv4+ with BatchNorm frozen.
            backbone = next(layer for layer in model.layers if layer.name == "resnet50")
            from resnet50_utils import set_fine_tuning, set_head_training
            head_epochs = max(1, epochs // 5)
            phase = (resume or {}).get("phase", "head")
            prior_history = dict((resume or {}).get("history", {}))
            labels = [int(row["label"]) for row in train_rows]
            positives, negatives = sum(labels), len(labels) - sum(labels)
            class_weight = {0: len(labels) / max(2 * negatives, 1), 1: len(labels) / max(2 * positives, 1)}
            global_step = int((resume or {}).get("global_step", 0))

            def configure(current_phase):
                if current_phase == "head":
                    set_head_training(backbone); optimizer = tf.keras.optimizers.Adam(1e-3); loss = "binary_crossentropy"
                else:
                    set_fine_tuning(backbone); optimizer = tf.keras.optimizers.Adam(1e-5)
                    loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.75)
                model.compile(optimizer, loss, metrics=[tf.keras.metrics.AUC(name="auc")])
                return optimizer

            def restore(optimizer):
                if not resume: return
                model.set_weights(resume["model_state"])
                if hasattr(optimizer, "build"): optimizer.build(model.trainable_variables)
                for variable, value in zip(optimizer.variables, resume.get("optimizer_state", [])): variable.assign(value)
                states = resume.get("rng_states", {})
                if states.get("python"): random.setstate(states["python"])
                if states.get("numpy"): np.random.set_state(states["numpy"])
                try:
                    if states.get("tensorflow") is not None: tf.random.get_global_generator().state.assign(states["tensorflow"])
                except Exception: pass

            def restore_callback_state(reduce_cb, early_cb, resume_phase, current_phase):
                # Gated on the *checkpoint's own* recorded phase, never the outer mutable
                # `phase` variable: a head->finetune transition in the same run must never
                # apply head-phase scheduler/early-stopping state to the new finetune objects.
                if not resume or resume_phase != current_phase:
                    return
                for key, value in (resume.get("scheduler_state", {}) or {}).items():
                    if hasattr(reduce_cb, key): setattr(reduce_cb, key, value)
                if resume.get("best_metric") is not None:
                    early_cb.best = float(resume["best_metric"])
                early_cb.wait = int(resume.get("early_stopping_counter", 0))

            class DurableResume(tf.keras.callbacks.Callback):
                def __init__(self, optimizer, current_phase, initial_epoch):
                    super().__init__(); self.optimizer=optimizer; self.phase=current_phase; self.epoch=initial_epoch
                    self.global_step=int((resume or {}).get("global_step", global_step)); self.best=float((resume or {}).get("best_metric", "-inf")); self.best_epoch=(resume or {}).get("best_epoch")
                    self.wait=int((resume or {}).get("early_stopping_counter", 0)); self.history=prior_history
                def payload(self, batch):
                    try: tf_rng=tf.random.get_global_generator().state.numpy()
                    except Exception: tf_rng=None
                    scheduler_state={key:getattr(self.reduce,key) for key in ("best","wait","cooldown_counter","factor","patience","min_lr","mode","monitor") if hasattr(self.reduce,key)}
                    return {**expected, "model_state": model.get_weights(), "optimizer_state": [v.numpy() for v in self.optimizer.variables],
                            "scheduler_state": scheduler_state, "scaler_state": None, "phase": self.phase,
                            "epoch": int(self.epoch), "batch_index": int(batch), "global_step": self.global_step,
                            "best_metric": self.best, "best_epoch": self.best_epoch, "early_stopping_counter": self.wait,
                            "history": self.history, "rng_states": {"python": random.getstate(), "numpy": np.random.get_state(), "tensorflow": tf_rng},
                            "resume_segment_id": hashlib.sha256(os.urandom(16)).hexdigest()[:16]}
                def on_train_batch_end(self, batch, logs=None):
                    self.global_step += 1
                    if self.global_step % self.interval == 0:
                        ckio.save_resume_checkpoint(run_dir, self.payload(batch))
                    if self.global_step >= int(self.max_updates): self.model.stop_training = True
                def on_epoch_end(self, epoch, logs=None):
                    self.epoch=epoch+1; logs=logs or {}; metric=float(logs.get("val_auc", float("-inf")))
                    if metric > self.best: self.best=metric; self.best_epoch=self.epoch; self.wait=0; best=True
                    else: self.wait += 1; best=False
                    for key,value in logs.items(): self.history.setdefault(key,[]).append(float(value))
                    ckio.save_resume_checkpoint(run_dir, self.payload(-1), best=best)
            def fit_phase(current_phase, start, stop):
                optimizer=configure(current_phase)
                resume_phase = (resume or {}).get("phase")
                if resume and resume_phase == current_phase: restore(optimizer)
                reduce=tf.keras.callbacks.ReduceLROnPlateau(**self.policy["scheduler_params"])
                durable=DurableResume(optimizer,current_phase,start); durable.reduce=reduce
                durable.interval=int(self.policy.get("checkpoint_interval_updates",250)); durable.max_updates=int(self.policy["max_optimizer_updates"])
                early=tf.keras.callbacks.EarlyStopping(monitor="val_auc",mode="max",patience=int(self.policy["early_stopping"]["patience"]),restore_best_weights=True)
                # Keras resets EarlyStopping's own best/wait to their defaults inside its own
                # on_train_begin, which model.fit() triggers - restoring them here, before
                # fit() is called, would just be overwritten. Restoring from durable's
                # on_train_begin instead (registered after `early` in the callback list) runs
                # after Keras's own reset, so it actually sticks.
                durable.early = early
                original_on_train_begin = durable.on_train_begin
                def on_train_begin(logs=None, _orig=original_on_train_begin):
                    _orig(logs)
                    restore_callback_state(reduce, early, resume_phase, current_phase)
                durable.on_train_begin = on_train_begin
                return model.fit(train_loader, validation_data=val_loader, initial_epoch=start, epochs=stop, class_weight=class_weight, verbose=2, callbacks=[reduce,early,durable])
            h1 = fit_phase("head", int((resume or {}).get("epoch",0)) if phase=="head" else 0, head_epochs) if phase == "head" else None
            if h1 is not None:
                resume, _ = ckio.load_resume_checkpoint(run_dir, expected)
                # Preserve trained head weights in memory; a new fine-tuning optimizer must not
                # receive incompatible head-phase slot variables. Persisted to disk (not just
                # the local variable) so a crash between head completion and the first
                # fine-tuning batch can never re-train an already-complete head on resume.
                resume = {**(resume or {}), "phase": "transition", "epoch": 0}
                ckio.save_resume_checkpoint(run_dir, resume)
            phase = "finetune"
            h2 = fit_phase("finetune", int((resume or {}).get("epoch",0)) if (resume or {}).get("phase")=="finetune" else 0, epochs)
            history = prior_history
            for part in (h1,h2):
                if part:
                    for key,values in part.history.items(): history.setdefault(key,[]).extend(list(values))
            # Same disk-backed global-best guarantee as the PyTorch branch: checkpoint_best.pkl
            # was kept current by DurableResume.on_epoch_end(best=...) throughout both phases.
            best_path = ckio.resume_checkpoint_path(run_dir, "checkpoint_best")
            if best_path.is_file():
                try:
                    best_payload = ckio.read_resume_checkpoint(best_path)
                except Exception:
                    best_payload = None
                if best_payload is not None and best_payload.get("model_state") is not None:
                    model.set_weights(best_payload["model_state"])
                    best_epoch = best_payload.get("best_epoch")
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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=float(self.policy["scheduler_params"]["factor"]),
                patience=int(self.policy["scheduler_params"]["patience"]),
                min_lr=float(self.policy["scheduler_params"]["min_lr"]))
            scaler = torch.amp.GradScaler("cuda") if bool(self.policy.get("amp")) and device.type == "cuda" else None
            start_epoch, start_batch, global_step = 1, 0, 0
            prior_history: dict = {}
            if resume:
                model.load_state_dict(resume["model_state_dict"], strict=True)
                optimizer.load_state_dict(resume["optimizer_state_dict"])
                scheduler.load_state_dict(resume["scheduler_state_dict"])
                if scaler is not None and resume.get("scaler_state_dict"): scaler.load_state_dict(resume["scaler_state_dict"])
                start_epoch, start_batch = int(resume["epoch"]), int(resume.get("batch_index", -1)) + 1
                global_step = int(resume["global_step"])
                # EarlyStopping's real patience-counter attribute is `wait`, not `counter` -
                # restoring the wrong name silently restored nothing on every resume.
                early.wait = int(resume.get("early_stopping_counter", 0))
                if resume.get("best_metric") is not None: early.best = float(resume["best_metric"])
                best_epoch = resume.get("best_epoch")
                prior_history = dict(resume.get("history", {}))
                if resume.get("rng_states", {}).get("python"): random.setstate(resume["rng_states"]["python"])
                if resume.get("rng_states", {}).get("numpy"): np.random.set_state(resume["rng_states"]["numpy"])
                if resume.get("rng_states", {}).get("torch") is not None: torch.set_rng_state(resume["rng_states"]["torch"])
                if torch.cuda.is_available() and resume.get("rng_states", {}).get("torch_cuda"):
                    torch.cuda.set_rng_state_all(resume["rng_states"]["torch_cuda"])
            segment = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
            # `current` tracks the *real* in-progress epoch/history so a periodic (intra-epoch)
            # checkpoint never reports the previous epoch's number or wipes history to {}.
            current = {"epoch": start_epoch, "history": prior_history, "best_epoch": best_epoch}
            def save_torch(step, batch, epoch=None, history=None, best=False, best_epoch=None):
                epoch = current["epoch"] if epoch is None else epoch
                history = current["history"] if history is None else history
                payload = {**expected, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict() if scaler else None,
                    "epoch": epoch, "batch_index": batch, "global_step": step,
                    "best_metric": getattr(early, "best", None),
                    "best_epoch": current["best_epoch"] if best_epoch is None else best_epoch,
                    "early_stopping_counter": getattr(early, "wait", 0), "history": history or {},
                    "rng_states": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                                   "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, "resume_segment_id": segment}
                ckio.save_resume_checkpoint(run_dir, payload, best=best)
            interval = int(self.policy.get("checkpoint_interval_updates", 250))
            warmup = int(self.policy.get("warmup_updates", 0))
            target_lr = float(self.policy["training_phases"][0]["learning_rate"])
            def before_optimizer_step(step, _batch):
                if warmup and step <= warmup:
                    for group in optimizer.param_groups: group["lr"] = target_lr * step / warmup
            def epoch_begin(epoch):
                current["epoch"] = epoch
            def periodic(step, batch):
                if step % interval == 0: save_torch(step, batch)
            def epoch_end(epoch, step, _scaler, hist, metrics, improved):
                current["epoch"] = epoch
                current["history"] = hist.history
                if improved: current["best_epoch"] = epoch
                scheduler.step(metrics["auc"])
                save_torch(step, -1, epoch + 1, hist.history, best=improved)
            history_obj = amp_utils.fit_mammofm(
                model, train_loader, val_loader, optimizer, criterion, epochs, device,
                early_stopping=early, use_amp=bool(self.policy.get("amp", False)),
                grad_clip_norm=self.policy.get("gradient_clipping"),
                accumulation_steps=int(self.policy.get("gradient_accumulation_steps", 1)),
                lr_scheduler=None, start_epoch=start_epoch, start_batch=start_batch, global_step=global_step,
                max_optimizer_updates=int(self.policy["max_optimizer_updates"]), scaler=scaler,
                on_optimizer_step=periodic, on_before_optimizer_step=before_optimizer_step,
                on_epoch_begin=epoch_begin, on_epoch_end=epoch_end, resume_history=prior_history,
            )
            history = history_obj.history if hasattr(history_obj, "history") else vars(history_obj)
            # The in-memory early.best_state does not survive a process restart; the disk-backed
            # checkpoint_best.pkl (written by save_torch(..., best=True) above) does, so the
            # final model is always the true global best, even if it predates an interruption.
            best_path = ckio.resume_checkpoint_path(run_dir, "checkpoint_best")
            if best_path.is_file():
                try:
                    best_payload = ckio.read_resume_checkpoint(best_path)
                except Exception:
                    best_payload = None
                if best_payload is not None and best_payload.get("best_metric") is not None:
                    if getattr(early, "best", float("-inf")) is None or best_payload["best_metric"] >= getattr(early, "best", float("-inf")):
                        model.load_state_dict(best_payload["model_state_dict"], strict=True)
                        best_epoch = best_payload.get("best_epoch")
        self.save_checkpoint(model, Path(checkpoint_path))
        return {"checkpoint": str(checkpoint_path), "history": history, "resumed_from": resume_source if resume else None,
                "optimizer_updates_limit": int(self.policy["max_optimizer_updates"]), "epochs": epochs,
                "best_epoch": best_epoch}

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
