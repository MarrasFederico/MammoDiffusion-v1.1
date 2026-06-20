from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi


QUALITY_FEATURES = [
    "nonblack_ratio",
    "mean_nonblack",
    "std_nonblack",
    "entropy",
    "lap_var",
    "largest_component_frac",
]
FILTER_SCHEMA_VERSION = 1
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def estimate_foreground_mask(arr, min_threshold: int = 10):
    arr = np.asarray(arr, dtype=np.uint8)
    height, width = arr.shape
    border_width = max(2, int(round(min(height, width) * 0.02)))
    corner_height = max(8, int(round(height * 0.12)))
    corner_width = max(8, int(round(width * 0.12)))

    background_candidates = np.concatenate([
        arr[:border_width, :].ravel(),
        arr[-border_width:, :].ravel(),
        arr[:, :border_width].ravel(),
        arr[:, -border_width:].ravel(),
        arr[:corner_height, :corner_width].ravel(),
        arr[:corner_height, -corner_width:].ravel(),
        arr[-corner_height:, :corner_width].ravel(),
        arr[-corner_height:, -corner_width:].ravel(),
    ]).astype(np.float64)

    low_background = background_candidates[
        background_candidates <= np.quantile(background_candidates, 0.60)
    ]
    background_median = float(np.median(low_background))
    background_mad = float(np.median(np.abs(low_background - background_median)))
    adaptive_threshold = max(
        float(min_threshold),
        background_median + 6.0 * 1.4826 * background_mad + 5.0,
    )
    adaptive_threshold = max(
        float(min_threshold),
        min(adaptive_threshold, 220.0, float(np.quantile(arr, 0.95))),
    )

    raw_mask = arr > adaptive_threshold
    raw_labels, component_count = ndi.label(raw_mask)
    component_sizes = (
        np.bincount(raw_labels.ravel())[1:]
        if component_count
        else np.array([], dtype=np.int64)
    )
    largest_component_frac = (
        float(component_sizes.max() / max(int(raw_mask.sum()), 1))
        if component_sizes.size
        else 0.0
    )

    structure = ndi.generate_binary_structure(2, 1)
    clean_mask = ndi.binary_opening(raw_mask, structure=structure, iterations=1)
    clean_mask = ndi.binary_closing(clean_mask, structure=structure, iterations=2)
    clean_mask = ndi.binary_fill_holes(clean_mask)
    clean_labels, clean_count = ndi.label(clean_mask)
    clean_sizes = (
        np.bincount(clean_labels.ravel())[1:]
        if clean_count
        else np.array([], dtype=np.int64)
    )
    if clean_sizes.size:
        clean_mask = clean_labels == int(clean_sizes.argmax() + 1)
    else:
        clean_mask = np.zeros_like(raw_mask, dtype=bool)

    diagnostics = {
        "adaptive_threshold": float(adaptive_threshold),
        "background_median": background_median,
        "background_mad": background_mad,
        "raw_nonblack_ratio": float(raw_mask.mean()),
        "largest_component_frac": largest_component_frac,
    }
    return clean_mask, diagnostics


def image_quality_metrics(path, nonblack_threshold: int = 10):
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)

    mask, diagnostics = estimate_foreground_mask(gray, nonblack_threshold)
    foreground_values = gray[mask].astype(np.float64)
    foreground_count = int(mask.sum())
    pixel_count = int(mask.size)

    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / pixel_count
    entropy = float(-(probabilities * np.log2(probabilities)).sum())

    return {
        **diagnostics,
        "nonblack_ratio": float(foreground_count / pixel_count),
        "mean_nonblack": float(foreground_values.mean()) if foreground_count else 0.0,
        "std_nonblack": float(foreground_values.std()) if foreground_count else 0.0,
        "entropy": entropy,
        "lap_var": float(ndi.laplace(gray.astype(np.float32)).var()),
        "top_border": float(mask[0, :].mean()),
        "bottom_border": float(mask[-1, :].mean()),
    }


def compute_reference_statistics(
    reference_paths: Iterable[Path],
    nonblack_threshold: int = 10,
    verbose: bool = False,
) -> dict:
    paths = [Path(path) for path in reference_paths]
    if not paths:
        raise ValueError("Nessuna immagine reale per calibrare il filtro.")

    rows = []
    for index, path in enumerate(paths, start=1):
        rows.append({"path": str(path), **image_quality_metrics(path, nonblack_threshold)})
        if verbose and (index % 100 == 0 or index == len(paths)):
            print(f"  Calibrazione filtro: {index}/{len(paths)}")

    train_metrics = pd.DataFrame(rows)
    thresholds = {
        "min_nonblack_ratio": float(max(0.02, train_metrics["nonblack_ratio"].quantile(0.01) * 0.70)),
        "max_nonblack_ratio": float(min(0.95, train_metrics["nonblack_ratio"].quantile(0.99) * 1.30)),
        "min_std_nonblack": float(max(3.0, train_metrics["std_nonblack"].quantile(0.01) * 0.50)),
        "min_largest_component_frac": 0.75,
        "max_top_border": 0.65,
        "max_bottom_border": 0.65,
    }
    medians = {
        feature: float(train_metrics[feature].median())
        for feature in QUALITY_FEATURES
    }
    iqrs = {}
    for feature in QUALITY_FEATURES:
        iqr = float(
            train_metrics[feature].quantile(0.75)
            - train_metrics[feature].quantile(0.25)
        )
        iqrs[feature] = iqr if iqr > 0 else 1e-6

    return {
        "thresholds": thresholds,
        "medians": medians,
        "iqrs": iqrs,
        "n_train": len(paths),
        "train_metrics": train_metrics,
    }


def _score_metrics(metrics: dict, reference: dict) -> tuple[bool, str, float]:
    thresholds = reference["thresholds"]
    rejection_rules = [
        (metrics["nonblack_ratio"] < thresholds["min_nonblack_ratio"], "too_empty"),
        (metrics["nonblack_ratio"] > thresholds["max_nonblack_ratio"], "too_full"),
        (metrics["std_nonblack"] < thresholds["min_std_nonblack"], "low_contrast"),
        (
            metrics["largest_component_frac"] < thresholds["min_largest_component_frac"],
            "fragmented_mask",
        ),
        (metrics["top_border"] > thresholds["max_top_border"], "touches_top_border"),
        (metrics["bottom_border"] > thresholds["max_bottom_border"], "touches_bottom_border"),
    ]
    for rejected, reason in rejection_rules:
        if rejected:
            return True, reason, float("nan")

    selection_score = -sum(
        abs(float(metrics[feature]) - reference["medians"][feature])
        / reference["iqrs"][feature]
        for feature in QUALITY_FEATURES
    )
    return False, "", float(selection_score)


def evaluate_candidate(
    path,
    reference: dict,
    nonblack_threshold: int = 10,
) -> dict:
    path = Path(path)
    metrics = image_quality_metrics(path, nonblack_threshold)
    rejected, reason, score = _score_metrics(metrics, reference)
    return {
        "raw_path": str(path),
        "raw_filename": path.name,
        **metrics,
        "rejected": bool(rejected),
        "reason": reason,
        "score": score,
    }


def _image_signature(paths: list[Path]) -> list[dict]:
    return [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]


def _existing_filtered_complete(filtered_dir: Path, n_selected: int) -> bool:
    for index in range(n_selected):
        path = filtered_dir / f"synth_filtered_{index:04d}.png"
        if not path.exists() or path.stat().st_size == 0:
            return False
    return True


def _signature_matches(summary_path: Path, signature: dict) -> bool:
    if not summary_path.exists():
        return False
    try:
        with open(summary_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return False
    return (
        payload.get("schema_version") == FILTER_SCHEMA_VERSION
        and payload.get("input_signature") == signature
    )


def _clear_filtered_outputs(filtered_dir: Path) -> None:
    filtered_dir.mkdir(parents=True, exist_ok=True)
    for path in filtered_dir.glob("synth_filtered_*.png"):
        path.unlink()


def _make_filter_plots(
    candidates_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    train_metrics_df: pd.DataFrame,
    plots_dir: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    sample_path = plots_dir / "synthetic_filter_sample.png"
    distribution_path = plots_dir / "synthetic_filter_distribution.png"

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for index, (_, row) in enumerate(selected_df.head(6).iterrows()):
        axes[0, index].imshow(Image.open(row["raw_path"]).convert("L"), cmap="gray")
        axes[0, index].set_title(f"OK | {row['score']:.2f}", fontsize=7)
        axes[0, index].axis("off")
    rejected_head = candidates_df[candidates_df["rejected"]].head(6)
    for index, (_, row) in enumerate(rejected_head.iterrows()):
        axes[1, index].imshow(Image.open(row["raw_path"]).convert("L"), cmap="gray")
        axes[1, index].set_title(f"NO: {row['reason']}", fontsize=7)
        axes[1, index].axis("off")
    for ax in axes.ravel():
        if not ax.has_data():
            ax.axis("off")
    plt.suptitle("Filtro adattivo | selezionate vs rifiutate", fontsize=11)
    plt.tight_layout()
    plt.savefig(sample_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    accepted_df = candidates_df[~candidates_df["rejected"]]
    axes[0].hist(accepted_df["score"], bins=40, color="steelblue", edgecolor="white")
    if not selected_df.empty:
        axes[0].axvline(selected_df["score"].min(), color="red", ls="--")
    axes[0].set_title("Score distribuzione")
    axes[0].grid(True, alpha=0.4)
    for ax, feature, title in zip(
        axes[1:],
        ["nonblack_ratio", "std_nonblack"],
        ["nonblack_ratio", "std_nonblack"],
    ):
        ax.hist(candidates_df[feature], bins=40, color="orange", alpha=0.7, label="sintetiche")
        ax.hist(train_metrics_df[feature], bins=40, color="green", alpha=0.7, label="reali train")
        ax.set_title(f"Confronto {title}")
        ax.legend()
        ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(distribution_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "sample_plot": str(sample_path),
        "distribution_plot": str(distribution_path),
    }


def filter_generated_directory(
    raw_dir,
    filtered_dir,
    reference_paths: Iterable[Path],
    n_raw: int,
    n_selected: int,
    target_label: int = 1,
    nonblack_threshold: int = 10,
    output_dir: Path | None = None,
    plots_dir: Path | None = None,
    force_recompute: bool = False,
    verbose: bool = True,
) -> dict:
    raw_dir = Path(raw_dir)
    filtered_dir = Path(filtered_dir)
    output_dir = Path(output_dir) if output_dir is not None else filtered_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "synthetic_filter_report.csv"
    summary_path = output_dir / "synthetic_filter_summary.json"

    raw_paths = sorted(raw_dir.glob("*.png"))
    if len(raw_paths) < n_raw:
        raise RuntimeError(f"Raw insufficienti: {len(raw_paths)} disponibili, {n_raw} richieste.")
    raw_paths = raw_paths[:n_raw]
    reference_paths = [Path(path) for path in reference_paths]
    input_signature = {
        "raw_dir": str(raw_dir),
        "filtered_dir": str(filtered_dir),
        "n_raw": int(n_raw),
        "n_selected": int(n_selected),
        "target_label": int(target_label),
        "nonblack_threshold": int(nonblack_threshold),
        "raw_files": _image_signature(raw_paths),
        "reference_files": _image_signature(reference_paths),
    }

    if (
        not force_recompute
        and _signature_matches(summary_path, input_signature)
        and report_path.exists()
        and _existing_filtered_complete(filtered_dir, n_selected)
    ):
        if verbose:
            print(f"Filtro gia' completo: {summary_path}")
        return {
            "schema_version": FILTER_SCHEMA_VERSION,
            "cached": True,
            "report_path": str(report_path),
            "summary_path": str(summary_path),
            "filtered_dir": str(filtered_dir),
        }

    existing_filtered = sorted(filtered_dir.glob("*.png"))
    if existing_filtered and not force_recompute:
        raise RuntimeError(
            f"{filtered_dir} contiene gia' {len(existing_filtered)} PNG ma la cache "
            "non e' compatibile. Rilancia con --force-recompute per rigenerare "
            "solo la cartella filtrata."
        )

    if force_recompute:
        _clear_filtered_outputs(filtered_dir)
    else:
        filtered_dir.mkdir(parents=True, exist_ok=True)

    reference = compute_reference_statistics(
        reference_paths,
        nonblack_threshold=nonblack_threshold,
        verbose=verbose,
    )
    if verbose:
        print(f"Immagini reali positive per calibrazione: {reference['n_train']}")

    candidate_rows = []
    for index, path in enumerate(raw_paths, start=1):
        candidate_rows.append(evaluate_candidate(path, reference, nonblack_threshold))
        if verbose and (index % 250 == 0 or index == len(raw_paths)):
            print(f"  Analizzate {index}/{len(raw_paths)}")

    candidates_df = pd.DataFrame(candidate_rows)
    n_rejected = int(candidates_df["rejected"].sum())
    n_accepted = len(candidates_df) - n_rejected
    reject_counts = (
        candidates_df[candidates_df["rejected"]]["reason"].value_counts().to_dict()
    )
    if verbose:
        print(
            f"Filtro: {len(candidates_df)} totali | {n_accepted} accettate | "
            f"{n_rejected} rifiutate"
        )
    if n_accepted < n_selected:
        raise RuntimeError(
            f"Solo {n_accepted} accettate, servono {n_selected}. Aumenta --n-raw."
        )

    accepted_df = (
        candidates_df[~candidates_df["rejected"]]
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    selected_df = accepted_df.head(n_selected).copy()
    selected_df["rank"] = range(n_selected)
    candidates_df["selected"] = False
    candidates_df["rank"] = np.nan
    candidates_df["filtered_path"] = ""

    for rank, row in selected_df.iterrows():
        output_path = filtered_dir / f"synth_filtered_{rank:04d}.png"
        shutil.copy2(row["raw_path"], output_path)
        mask = candidates_df["raw_path"] == row["raw_path"]
        candidates_df.loc[mask, "selected"] = True
        candidates_df.loc[mask, "rank"] = int(rank)
        candidates_df.loc[mask, "filtered_path"] = str(output_path)
        selected_df.loc[rank, "filtered_path"] = str(output_path)

    plot_paths = (
        _make_filter_plots(
            candidates_df,
            selected_df,
            reference["train_metrics"],
            Path(plots_dir),
        )
        if plots_dir is not None
        else {}
    )

    candidates_df.to_csv(report_path, index=False)
    summary = {
        "schema_version": FILTER_SCHEMA_VERSION,
        "input_signature": input_signature,
        "cached": False,
        "n_raw": int(n_raw),
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "n_selected": int(n_selected),
        "target_label": int(target_label),
        "reject_counts": reject_counts,
        "thresholds": reference["thresholds"],
        "reference_medians": reference["medians"],
        "reference_iqrs": reference["iqrs"],
        "raw_dir": str(raw_dir),
        "filtered_dir": str(filtered_dir),
        "report_path": str(report_path),
        "plots": plot_paths,
    }
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    if verbose:
        print(f"Completato: {n_raw} raw -> {n_selected} filtrate")
        print("Filtered dir:", filtered_dir)
    return summary


__all__ = [
    "QUALITY_FEATURES",
    "compute_reference_statistics",
    "estimate_foreground_mask",
    "evaluate_candidate",
    "filter_generated_directory",
    "image_quality_metrics",
]
