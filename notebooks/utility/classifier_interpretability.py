"""Validation-only attribution helpers for MaxViT-512 and Mammo-FM."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


def normalize_heatmap(value):
    import numpy as np
    array = np.nan_to_num(np.asarray(value, dtype="float32"), nan=0.0, posinf=0.0, neginf=0.0)
    array = np.maximum(array, 0)
    maximum = float(array.max()) if array.size else 0.0
    return array / maximum if maximum > 0 else np.zeros_like(array)


def ensemble_heatmaps(seed_maps):
    import numpy as np
    if not seed_maps:
        raise ValueError("at least one seed heatmap is required")
    shape = np.asarray(seed_maps[0]).shape
    if any(np.asarray(item).shape != shape for item in seed_maps):
        raise ValueError("seed heatmaps are not aligned")
    return normalize_heatmap(np.mean([normalize_heatmap(item) for item in seed_maps], axis=0))


def torch_spatial_attribution(model, image_batch, target_module):
    """Gradient-weighted spatial attribution suitable for MaxViT's final spatial stage."""
    activations, gradients = [], []
    hook_a = target_module.register_forward_hook(lambda _module, _inputs, output: activations.append(output))
    hook_g = target_module.register_full_backward_hook(lambda _module, _inputs, output: gradients.append(output[0]))
    try:
        model.zero_grad(set_to_none=True)
        model(image_batch).reshape(-1)[0].backward()
        feature, gradient = activations[-1], gradients[-1]
        weights = gradient.mean(dim=(-2, -1), keepdim=True)
        return normalize_heatmap((weights * feature).sum(dim=1)[0].detach().cpu().numpy())
    finally:
        hook_a.remove(); hook_g.remove()


def mammofm_attribution(model, image_batch):
    """Use Mammo-FM EfficientNet spatial features rather than forcing a generic layer name."""
    encoder = getattr(model, "image_encoder", None) or getattr(model, "backbone", None)
    backbone = getattr(encoder, "model", encoder)
    target = getattr(backbone, "_conv_head", None)
    if target is None or not hasattr(backbone, "extract_features"):
        raise TypeError("Mammo-FM requires its architecture-compatible EfficientNet feature stage")
    return torch_spatial_attribution(model, image_batch, target)


def preregistered_cases(rows: Sequence[Mapping], threshold: float, limit_per_category: int = 1) -> list[dict]:
    """Deterministically select TP/TN/FP/FN by confidence, never by visual appearance."""
    categories = {name: [] for name in ("TP", "TN", "FP", "FN")}
    for source in rows:
        row = dict(source)
        truth, prediction = int(row["label"]), float(row["probability"]) >= threshold
        category = "TP" if truth and prediction else "FN" if truth else "FP" if prediction else "TN"
        categories[category].append(row)
    selected = []
    for category in ("TP", "TN", "FP", "FN"):
        ordered = sorted(categories[category], key=lambda row: (-abs(float(row["probability"]) - threshold),
                                                                 str(row.get("patient_id")), str(row.get("image_id"))))
        selected.extend([{**row, "category": category} for row in ordered[:limit_per_category]])
    return selected


def largest_ft_fs_disagreements(finetuned_rows: Sequence[Mapping], fromscratch_rows: Sequence[Mapping],
                                limit: int = 4) -> list[dict]:
    left = {(str(row["patient_id"]), str(row["image_id"])): row for row in finetuned_rows}
    right = {(str(row["patient_id"]), str(row["image_id"])): row for row in fromscratch_rows}
    if set(left) != set(right):
        raise ValueError("fine-tuned/from-scratch attribution rows are not aligned")
    rows = [{"patient_id": key[0], "image_id": key[1],
             "finetuned_probability": float(left[key]["probability"]),
             "fromscratch_probability": float(right[key]["probability"]),
             "absolute_disagreement": abs(float(left[key]["probability"]) - float(right[key]["probability"]))}
            for key in left]
    return sorted(rows, key=lambda row: (-row["absolute_disagreement"], row["patient_id"], row["image_id"]))[:limit]


def render_attribution_overlays(cases: Sequence[Mapping], attribution_dir: Path):
    """Overlay each preregistered case's saved heatmap on its source image.

    Top row: original mammogram with the Grad-CAM heatmap over it. Bottom row: the
    isolated heatmap. Reads the ``{category}_{index}.npy`` files written next to the run.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    directory = Path(attribution_dir)
    cases = list(cases)
    columns = max(len(cases), 1)
    figure, axes = plt.subplots(2, columns, figsize=(4 * columns, 8), squeeze=False)
    for index, case in enumerate(cases):
        heatmap = normalize_heatmap(np.load(directory / f"{case['category']}_{index}.npy"))
        image = Image.open(case["processed_path"]).convert("L")
        base = np.asarray(image, dtype="float32") / 255.0
        upscaled = np.asarray(
            Image.fromarray((heatmap * 255).astype("uint8")).resize(image.size, Image.BILINEAR),
            dtype="float32",
        ) / 255.0
        overlay = axes[0][index]
        overlay.imshow(base, cmap="gray")
        overlay.imshow(upscaled, cmap="jet", alpha=0.4)
        overlay.set_title(
            f"{case['category']} · p={float(case['probability']):.2f} · y={int(case['label'])}",
            fontsize=11,
        )
        overlay.axis("off")
        isolated = axes[1][index]
        isolated.imshow(heatmap, cmap="jet")
        isolated.set_title("Grad-CAM", fontsize=11)
        isolated.axis("off")
    return figure


def write_manifest(path: Path, *, architecture: str, method: str, samples: Sequence[Mapping]) -> Path:
    if architecture not in {"maxvit512", "mammofm"}:
        raise ValueError("interpretability is defined only for the two downstream architectures")
    payload = {"schema_version": 1, "architecture": architecture, "method": method,
               "selection": "preregistered_tp_tn_fp_fn_and_largest_ft_fs_disagreement",
               "samples": list(samples), "test_access": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


__all__ = ["ensemble_heatmaps", "largest_ft_fs_disagreements", "mammofm_attribution",
           "normalize_heatmap", "preregistered_cases", "render_attribution_overlays",
           "torch_spatial_attribution", "write_manifest"]
