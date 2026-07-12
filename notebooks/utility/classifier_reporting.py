"""Persist notebook-visible classifier diagnostics under the canonical results tree.

This module is deliberately independent from locked-test code.  Plotting is lazy so plan mode
still works in minimal environments.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def atomic_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)
    return path


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    return path


def canonical_results_dirs(base: Path) -> dict[str, Path]:
    dirs = {name: base / name for name in ("config", "dataset", "figures", "interpretability", "logs", "notebook")}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def persist_dataset_summary(base: Path, train_rows: list[dict], validation_rows: list[dict], manifest: dict) -> dict:
    dirs = canonical_results_dirs(base)
    counts = {}
    for row in train_rows:
        key = (str(row.get("source", "unknown")), int(row["label"]))
        counts[key] = counts.get(key, 0) + 1
    rows = [{"source": source, "label": label, "count": count} for (source, label), count in sorted(counts.items())]
    payload = {"train_samples": len(train_rows), "validation_samples": len(validation_rows),
               "validation_real_only": all(str(r.get("source", "")).startswith("real") for r in validation_rows),
               "source_by_class": rows, "manifest_signature": manifest.get("signature")}
    write_csv(dirs["dataset"] / "dataset_summary.csv", rows)
    atomic_json(dirs["dataset"] / "dataset_summary.json", payload)
    import matplotlib.pyplot as plt
    import numpy as np
    for filename, field, title in (("dataset_class_distribution.png", "label", "Training class distribution"),
                                   ("dataset_source_distribution.png", "source", "Training source distribution")):
        values = {}
        for row in train_rows: values[str(row.get(field))] = values.get(str(row.get(field)), 0) + 1
        fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(values, values.values()); ax.set_title(title); ax.set_ylabel("samples")
        fig.tight_layout(); fig.savefig(dirs["dataset"] / filename, dpi=140); plt.close(fig)
    sources = sorted({row["source"] for row in rows}); labels = (0, 1)
    fig, ax = plt.subplots(figsize=(7, 4)); bottom = [0] * len(sources)
    for label in labels:
        values = [next((r["count"] for r in rows if r["source"] == source and r["label"] == label), 0) for source in sources]
        ax.bar(sources, values, bottom=bottom, label=str(label)); bottom = [a + b for a, b in zip(bottom, values)]
    ax.legend(title="label"); ax.set_title("Source by class"); fig.tight_layout()
    fig.savefig(dirs["dataset"] / "dataset_source_by_class.png", dpi=140); plt.close(fig)
    from PIL import Image
    chosen=[]
    for key in sorted({(str(r.get("source")),int(r["label"])) for r in train_rows}):
        candidates=sorted((r for r in train_rows if (str(r.get("source")),int(r["label"]))==key),key=lambda r:str(r["processed_path"]))
        chosen.extend(candidates[:2])
    if chosen:
        columns=min(4,len(chosen)); rows_n=(len(chosen)+columns-1)//columns; fig,axes=plt.subplots(rows_n,columns,figsize=(3*columns,3*rows_n)); axes=np.asarray(axes).reshape(-1)
        for ax,row in zip(axes,chosen): ax.imshow(Image.open(row["processed_path"]).convert("L"),cmap="gray"); ax.set_title(f"{row['source']} · y={row['label']}"); ax.axis("off")
        for ax in axes[len(chosen):]: ax.axis("off")
        fig.tight_layout(); fig.savefig(dirs["dataset"] / "training_samples_grid.png",dpi=140); plt.close(fig)
    return payload


def save_placeholder_figure(path: Path, title: str, message: str) -> Path:
    """Save an honest placeholder when results do not exist yet; never fabricate metrics."""
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.axis("off")
    ax.set_title(title); ax.text(.5, .5, message, ha="center", va="center", wrap=True)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


def persist_plan_figures(base: Path, has_synthetic: bool, has_augmented: bool) -> list[str]:
    # Plan mode is read-only: absence is reported in the notebook and no fictitious PNG is made.
    return []


def render_training_curves(base: Path, seeds=(17, 42, 73)) -> list[Path]:
    import matplotlib.pyplot as plt
    made = []
    for seed in seeds:
        path = base / f"seed_{seed}" / f"training_history_seed_{seed}.csv"
        if not path.is_file(): continue
        with path.open(newline="", encoding="utf-8") as stream: rows = list(csv.DictReader(stream))
        if not rows: continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for key in ("loss", "val_loss"):
            if key in rows[0]: axes[0].plot([float(r[key]) for r in rows if r.get(key) not in (None, "")], label=key)
        for key in ("auc", "val_auc", "pr_auc", "val_pr_auc"):
            if key in rows[0]: axes[1].plot([float(r[key]) for r in rows if r.get(key) not in (None, "")], label=key)
        for ax in axes:
            if ax.lines: ax.legend()
            ax.grid(alpha=.25)
        out = base / "ensemble/figures" / f"seed_{seed}_training_curves.png"; out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig); made.append(out)
    return made


def render_validation_figures(base: Path) -> list[Path]:
    """Build real ROC/PR/calibration/confusion/probability/error plots from ensemble CSV."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve, roc_curve
    pred = base / "ensemble/predictions/ensemble_validation_predictions.csv"
    metrics_path = base / "ensemble/metrics/ensemble_validation_metrics.json"
    if not pred.is_file() or not metrics_path.is_file(): return []
    with pred.open(newline="", encoding="utf-8") as stream: rows = list(csv.DictReader(stream))
    y = np.array([int(r["label"]) for r in rows]); p = np.array([float(r["probability"]) for r in rows])
    threshold = float(json.loads(metrics_path.read_text())["threshold"]); yp = (p >= threshold).astype(int)
    figures = base / "ensemble/figures"; figures.mkdir(parents=True, exist_ok=True); made=[]
    fpr,tpr,_=roc_curve(y,p); precision,recall,_=precision_recall_curve(y,p)
    for name,x,z,xlabel,ylabel in (("roc_curve_all_seeds_and_ensemble.png",fpr,tpr,"FPR","TPR"),
                                   ("pr_curve_all_seeds_and_ensemble.png",recall,precision,"Recall","Precision")):
        fig,ax=plt.subplots(); ax.plot(x,z); ax.set(xlabel=xlabel,ylabel=ylabel); ax.grid(alpha=.25); fig.tight_layout(); out=figures/name; fig.savefig(out,dpi=140); plt.close(fig); made.append(out)
    fig,ax=plt.subplots(); ConfusionMatrixDisplay(confusion_matrix(y,yp)).plot(ax=ax); out=figures/"confusion_matrix_ensemble.png"; fig.savefig(out,dpi=140); plt.close(fig); made.append(out)
    true,prob=calibration_curve(y,p,n_bins=10); fig,ax=plt.subplots(); ax.plot(prob,true,marker="o"); ax.plot([0,1],[0,1],"--"); out=figures/"calibration_curve_ensemble.png"; fig.savefig(out,dpi=140); plt.close(fig); made.append(out)
    fig,ax=plt.subplots(); ax.hist(p[y==0],alpha=.6,label="negative"); ax.hist(p[y==1],alpha=.6,label="positive"); ax.legend(); out=figures/"probability_distribution_ensemble.png"; fig.savefig(out,dpi=140); plt.close(fig); made.append(out)
    cases=[]
    for row,predicted in zip(rows,yp): cases.append({**row,"prediction":int(predicted),"case":("TP" if int(row["label"]) and predicted else "FN" if int(row["label"]) else "FP" if predicted else "TN")})
    cases=sorted(cases,key=lambda r:(r["case"],-abs(float(r["probability"])-threshold)))
    write_csv(base/"ensemble/validation_error_cases.csv",cases)
    return made


def render_complete_report(base: Path) -> dict:
    return {"training_curves": [str(p) for p in render_training_curves(base)],
            "validation_figures": [str(p) for p in render_validation_figures(base)]}


ATTRIBUTION_METHODS = {
    "resnet50": "Grad-CAM on final convolutional feature map",
    "maxvit512": "gradient-weighted spatial attribution on final MaxViT stage",
    "mammofm": "gradient-weighted token/spatial attribution selected from the resolved backbone",
    "raddino": "gradient-weighted patch-token attribution (not a plain attention map)",
}


def attribution_manifest(base: Path, architecture: str, sample_kind: str, samples: list[dict]) -> Path:
    if sample_kind not in {"real", "synthetic", "augmented"}:
        raise ValueError(sample_kind)
    return atomic_json(base / "interpretability" / sample_kind / "manifest.json", {
        "schema_version": 1, "architecture": architecture, "method": ATTRIBUTION_METHODS[architecture],
        "gradient_based": True, "ensemble": "mean of normalized seed heatmaps", "samples": samples,
        "test_access": False,
    })
