"""Gradient-based attribution adapters used by matrix notebooks (validation/train only)."""
from __future__ import annotations

import json
from pathlib import Path


def normalize_heatmap(value):
    import numpy as np
    array = np.nan_to_num(np.asarray(value, dtype="float32"), nan=0.0, posinf=0.0, neginf=0.0)
    array = np.maximum(array, 0); maximum = float(array.max()) if array.size else 0.0
    return array / maximum if maximum > 0 else np.zeros_like(array)


def ensemble_heatmaps(seed_maps):
    import numpy as np
    if not seed_maps: raise ValueError("at least one seed heatmap is required")
    shape = np.asarray(seed_maps[0]).shape
    if any(np.asarray(item).shape != shape for item in seed_maps): raise ValueError("seed heatmaps are not aligned")
    return normalize_heatmap(np.mean([normalize_heatmap(item) for item in seed_maps], axis=0))


def tensorflow_gradcam(model, image_batch, target_layer=None):
    import tensorflow as tf
    if target_layer is None:
        target_layer = next(layer for layer in reversed(model.layers) if len(getattr(layer.output, "shape", ())) == 4)
    grad_model = tf.keras.Model(model.inputs, [target_layer.output, model.output])
    with tf.GradientTape() as tape:
        features, prediction = grad_model(image_batch, training=False)
        score = tf.reshape(prediction, (-1,))[0]
    gradients = tape.gradient(score, features); weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    return normalize_heatmap(tf.reduce_sum(weights * features, axis=-1)[0].numpy())


def torch_spatial_gradcam(model, image_batch, target_module):
    activations, gradients = [], []
    hook_a = target_module.register_forward_hook(lambda _m, _i, output: activations.append(output))
    hook_g = target_module.register_full_backward_hook(lambda _m, _gi, output: gradients.append(output[0]))
    try:
        model.zero_grad(set_to_none=True); output = model(image_batch).reshape(-1)[0]; output.backward()
        feature, gradient = activations[-1], gradients[-1]
        weights = gradient.mean(dim=(-2, -1), keepdim=True)
        return normalize_heatmap((weights * feature).sum(dim=1)[0].detach().cpu().numpy())
    finally:
        hook_a.remove(); hook_g.remove()


def mammofm_gradcam(model, image_batch):
    """Mammo-FM attribution uses EfficientNet spatial features before global pooling."""
    encoder = getattr(model, "image_encoder", None) or getattr(model, "backbone", None)
    # The production MammoFMImageEncoder wraps EfficientNet in ``.model``.  Keep direct
    # EfficientNet support for compact fixtures, but never assume the wrapper itself exposes
    # ``extract_features``.
    backbone = getattr(encoder, "model", encoder)
    if backbone is None or not hasattr(backbone, "extract_features"):
        raise TypeError("Mammo-FM image encoder must wrap an EfficientNet with extract_features()")
    # The final convolution is the feature-producing module used by extract_features.
    target = getattr(backbone, "_conv_head", None)
    if target is None: raise TypeError("Mammo-FM EfficientNet has no _conv_head")
    return torch_spatial_gradcam(model, image_batch, target)


def raddino_token_attribution(model, image_batch, token_module=None):
    """Gradient-weighted patch-token attribution; deliberately uses no MaxViT APIs."""
    import math
    backbone = getattr(model, "backbone", None) or getattr(model, "encoder", None)
    if token_module is None:
        layers = getattr(getattr(backbone, "encoder", None), "layer", None)
        if layers is None: layers = getattr(backbone, "layer", None)
        if not layers: raise TypeError("RAD-DINO backbone has no transformer token layer")
        token_module = layers[-1]
    activations, gradients = [], []
    def unpack(output): return output[0] if isinstance(output, (tuple, list)) else getattr(output, "last_hidden_state", output)
    ha = token_module.register_forward_hook(lambda _m, _i, output: activations.append(unpack(output)))
    hg = token_module.register_full_backward_hook(lambda _m, _gi, output: gradients.append(unpack(output)))
    try:
        model.zero_grad(set_to_none=True); model(image_batch).reshape(-1)[0].backward()
        tokens, grads = activations[-1][0], gradients[-1][0]
        patch = (tokens[1:] * grads[1:]).sum(dim=-1); side = math.isqrt(int(patch.numel()))
        if side * side != patch.numel(): raise ValueError("RAD-DINO patch token count is not square")
        return normalize_heatmap(patch.reshape(side, side).detach().cpu().numpy())
    finally:
        ha.remove(); hg.remove()


def save_overlay(image, heatmap, path: Path, title="Gradient-based attribution") -> Path:
    import matplotlib.pyplot as plt
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True); array=np.asarray(image); heat=normalize_heatmap(heatmap)
    fig,axes=plt.subplots(1,3,figsize=(12,4)); axes[0].imshow(array,cmap="gray"); axes[0].set_title("Original")
    axes[1].imshow(heat,cmap="jet"); axes[1].set_title("Heatmap")
    axes[2].imshow(array,cmap="gray"); axes[2].imshow(heat,cmap="jet",alpha=.45,extent=(0,array.shape[1],array.shape[0],0)); axes[2].set_title(title)
    for ax in axes: ax.axis("off")
    fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig); return path


def write_manifest(path: Path, *, architecture: str, samples: list[dict], method: str) -> Path:
    path.parent.mkdir(parents=True,exist_ok=True); payload={"schema_version":1,"architecture":architecture,"method":method,
        "gradient_based":True,"aggregation":"mean normalized seed heatmaps","samples":samples,"test_access":False}
    path.write_text(json.dumps(payload,indent=2)+"\n"); return path


def _last_spatial_torch_module(model):
    import torch
    modules=[module for module in model.modules() if isinstance(module,torch.nn.Conv2d)]
    if not modules: raise TypeError("model has no spatial convolution for Grad-CAM")
    return modules[-1]


def display_new_attributions(paths: list[str], root: Path) -> None:
    """Show just-created overlay PNGs inline, right where they were generated - independent
    of whatever order a notebook happens to call this vs. reporting.render_complete_report()
    (which only globs+displays files that already existed at the time *it* ran).
    """
    try:
        from IPython.display import Image as IPImage, display
    except Exception:
        return
    for rel in paths:
        candidate = Path(root) / rel
        if candidate.is_file():
            display(IPImage(filename=str(candidate)))


def _persist_fallback_selection(shared_path: Path, selected: list[dict]) -> None:
    """The deterministic lexicographic fallback must be written once, not silently
    re-derived on every call with nothing durable to inspect or reuse.
    """
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "policy": "deterministic_lexicographic_fallback_persisted_once",
               "samples": [{"patient_id": row["patient_id"], "image_id": row["image_id"]} for row in selected]}
    shared_path.write_text(json.dumps(payload, indent=2) + "\n")


def generate_configuration_attributions(root: Path, architecture: str, dataset_variant_id: str,
                                        policy: dict, seeds=(17,42,73), real_limit: int=8,
                                        synthetic_limit: int=8, display: bool=True) -> dict:
    """Generate per-seed and ensemble validation maps from real checkpoints.

    `real_limit` (shared/cross-model validation panel) and `synthetic_limit` (this model's own
    synthetic/augmented error-specific panel) are independent - GRADCAM_NUM_SYNTHETIC_SAMPLES
    must actually reach this function, not be silently dropped in favor of one shared limit.
    """
    import csv
    import numpy as np
    from PIL import Image
    import classifier_checkpoint_io as ckio
    from classifier_architecture_adapters import TinyAdapter, get_adapter
    base=Path(root)/"results/classifiers_matrix"/architecture/dataset_variant_id/f"{architecture}_standard"
    pred=base/"ensemble/predictions/ensemble_validation_predictions.csv"
    if not pred.is_file(): return {"status":"waiting_for_ensemble"}
    with pred.open(newline="",encoding="utf-8") as stream: predictions=list(csv.DictReader(stream))
    threshold=json.loads((base/"ensemble/metrics/locked_validation_threshold.json").read_text())["threshold"]
    for row in predictions:
        truth=int(row["label"]); predicted=float(row["probability"])>=float(threshold)
        row["category"]="TP" if truth and predicted else "FN" if truth else "FP" if predicted else "TN"
    shared_path=Path(root)/"configs/interpretability_validation_samples.json"
    shared=json.loads(shared_path.read_text()) if shared_path.is_file() else {"samples": []}
    requested=[(str(item.get("patient_id")),str(item.get("image_id"))) for item in shared.get("samples",[]) if item.get("patient_id") and item.get("image_id")]
    by_key={(str(row["patient_id"]),str(row["image_id"])):row for row in predictions}
    selected=[by_key[key] for key in requested if key in by_key][:real_limit]
    if not selected:
        # The same lexicographic fallback is independent of each model's errors and therefore
        # remains comparable until the tracked shared manifest is populated deliberately - and
        # is now persisted so it becomes *the* recorded choice rather than being re-derived
        # (and only implicitly reproducible) on every future call.
        selected=sorted(predictions,key=lambda row:(row["patient_id"],row["image_id"]))[:real_limit]
        if not shared_path.is_file() or not shared.get("samples"):
            _persist_fallback_selection(shared_path, selected)
    # Resolve canonical validation paths without touching test.
    import classifier_dataset_builder as datasets
    validation={(str(r["patient_id"]),str(r["image_id"])):r for r in datasets.validation_rows(Path(root))}
    seed_maps={}
    new_pngs: list[str] = []
    for seed in seeds:
        adapter=get_adapter(architecture,policy,root)
        if isinstance(adapter,TinyAdapter): continue
        checkpoint=ckio.checkpoint_path(ckio.run_dir(Path(root),architecture,dataset_variant_id,f"{architecture}_standard",seed),policy["framework"])
        model=adapter.load_checkpoint(checkpoint); maps=[]
        for sample in selected:
            row=validation[(sample["patient_id"],sample["image_id"])]
            loader=adapter.build_validation_dataloader([row],seed=seed); image_batch,_label=next(iter(loader))
            if architecture=="resnet50": heat=tensorflow_gradcam(model,image_batch)
            else:
                import torch
                device=next(model.parameters()).device; image_batch=image_batch.to(device)
                if architecture=="mammofm": heat=mammofm_gradcam(model,image_batch)
                elif architecture=="raddino": heat=raddino_token_attribution(model,image_batch)
                else: heat=torch_spatial_gradcam(model,image_batch,_last_spatial_torch_module(model))
            maps.append(heat)
            target=base/f"seed_{seed}/interpretability/real/{sample['patient_id']}_{sample['image_id']}.npy"
            target.parent.mkdir(parents=True,exist_ok=True); np.save(target,heat)
            # Per-seed overlay PNG too, not just the .npy, so a user can inspect a single
            # seed's attribution from the notebook without recomputing the ensemble.
            seed_png = target.with_suffix(".png")
            seed_image=np.asarray(Image.open(row["processed_path"]).convert("L"))
            save_overlay(seed_image,heat,seed_png,title=f"seed {seed}  {sample['category']}")
        seed_maps[seed]=maps
    if len(seed_maps)!=len(tuple(seeds)): raise RuntimeError("attribution requires all three real seed models")
    manifest=[]
    for index,sample in enumerate(selected):
        heat=ensemble_heatmaps([seed_maps[seed][index] for seed in seeds]); row=validation[(sample["patient_id"],sample["image_id"])]
        image=np.asarray(Image.open(row["processed_path"]).convert("L")); out=base/"ensemble/interpretability/real"/f"{sample['patient_id']}_{sample['image_id']}.png"
        save_overlay(image,heat,out,title=f"{sample['category']} p={float(sample['probability']):.3f}")
        np.save(out.with_suffix(".npy"),heat); rel=str(out.relative_to(root)); manifest.append({**sample,"path":rel}); new_pngs.append(rel)
    write_manifest(base/"ensemble/interpretability/real/manifest.json",architecture=architecture,samples=manifest,
                   method={"resnet50":"Grad-CAM final convolution","maxvit512":"Grad-CAM final spatial convolution",
                           "mammofm":"EfficientNet extract_features Grad-CAM","raddino":"gradient-weighted patch tokens"}[architecture])
    extra={}
    registry=json.loads((Path(root)/"configs/dataset_variant_registry.json").read_text())
    variant=next(v for v in registry["variants"] if v["dataset_variant_id"]==dataset_variant_id)
    train_rows,_,_=datasets.build_training_and_validation_rows(Path(root),variant)
    for kind in ("synthetic","augmented"):
        picked=sorted((r for r in train_rows if r.get("source")==kind),key=lambda r:r["processed_path"])[:synthetic_limit]
        if not picked: continue
        kind_maps={}
        for seed in seeds:
            adapter=get_adapter(architecture,policy,root); checkpoint=ckio.checkpoint_path(ckio.run_dir(Path(root),architecture,dataset_variant_id,f"{architecture}_standard",seed),policy["framework"])
            model=adapter.load_checkpoint(checkpoint); maps=[]
            for row in picked:
                image_batch,_=next(iter(adapter.build_validation_dataloader([row],seed=seed)))
                if architecture=="resnet50": heat=tensorflow_gradcam(model,image_batch)
                else:
                    device=next(model.parameters()).device; image_batch=image_batch.to(device)
                    heat=(mammofm_gradcam(model,image_batch) if architecture=="mammofm" else
                          raddino_token_attribution(model,image_batch) if architecture=="raddino" else
                          torch_spatial_gradcam(model,image_batch,_last_spatial_torch_module(model)))
                maps.append(heat); target=base/f"seed_{seed}/interpretability/{kind}/{Path(row['processed_path']).stem}.npy"; target.parent.mkdir(parents=True,exist_ok=True); np.save(target,heat)
                seed_image=np.asarray(Image.open(row["processed_path"]).convert("L"))
                save_overlay(seed_image,heat,target.with_suffix(".png"),title=f"seed {seed}  {kind}")
            kind_maps[seed]=maps
        kind_manifest=[]
        for index,row in enumerate(picked):
            heat=ensemble_heatmaps([kind_maps[seed][index] for seed in seeds]); image=np.asarray(Image.open(row["processed_path"]).convert("L")); out=base/f"ensemble/interpretability/{kind}/{Path(row['processed_path']).stem}.png"
            save_overlay(image,heat,out,title=f"{kind} y={row['label']}"); np.save(out.with_suffix(".npy"),heat)
            rel=str(out.relative_to(root)); kind_manifest.append({"image_id":row.get("image_id"),"label":row["label"],"source":kind,"path":rel}); new_pngs.append(rel)
        write_manifest(base/f"ensemble/interpretability/{kind}/manifest.json",architecture=architecture,samples=kind_manifest,method="same gradient-based architecture adapter as validation")
        extra[kind]=len(kind_manifest)
    if display:
        display_new_attributions(new_pngs, root)
    return {"status":"complete","real_samples":len(manifest),**extra}
