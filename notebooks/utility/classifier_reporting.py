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


def _read_seed_history(base: Path, seed: int) -> list[dict] | None:
    path = base / f"seed_{seed}" / f"training_history_seed_{seed}.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows or None


def render_training_curves(base: Path, seeds=(17, 42, 73)) -> list[Path]:
    import matplotlib.pyplot as plt
    made = []
    for seed in seeds:
        rows = _read_seed_history(base, seed)
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


def render_training_curves_all_seeds(base: Path, seeds=(17, 42, 73)) -> Path | None:
    """Real overlaid/aggregated per-seed curves - never a text listing of filenames."""
    import matplotlib.pyplot as plt
    by_seed = {seed: rows for seed in seeds if (rows := _read_seed_history(base, seed))}
    if not by_seed:
        return None
    panels = (
        ("loss", "train loss"), ("val_loss", "val loss"), ("val_auc", "val AUC"),
        ("val_pr_auc", "val PR-AUC"), ("pr_auc", "PR-AUC (train)"), ("lr", "learning rate"),
    )
    present = [(key, title) for key, title in panels if any(key in rows[0] for rows in by_seed.values())]
    if not present:
        return None
    fig, axes = plt.subplots(1, len(present), figsize=(4.5 * len(present), 4))
    axes = [axes] if len(present) == 1 else list(axes)
    colors = {17: "tab:blue", 42: "tab:orange", 73: "tab:green"}
    for ax, (key, title) in zip(axes, present):
        for seed, rows in by_seed.items():
            if key not in rows[0]:
                continue
            values = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
            if values:
                ax.plot(values, label=f"seed {seed}", color=colors.get(seed))
        ax.set_title(title); ax.grid(alpha=.25)
        if ax.lines: ax.legend(fontsize=8)
    fig.suptitle(f"Training curves — all seeds ({', '.join(str(s) for s in by_seed)})")
    fig.tight_layout()
    out = base / "ensemble/figures/training_curves_all_seeds.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


def _read_seed_predictions(base: Path, seed: int) -> list[dict] | None:
    path = base / f"seed_{seed}" / f"validation_predictions_seed_{seed}.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows or None


def render_validation_figures(base: Path, seeds=(17, 42, 73)) -> list[Path]:
    """Build real ROC/PR/calibration/confusion/probability/error plots.

    ROC/PR overlay every seed's own predictions plus the ensemble, all labelled, on one
    figure - not the ensemble alone. Error examples show the real validation image per case,
    not just a text listing.
    """
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

    series = [("ensemble", y, p, "black", 2.0)]
    colors = {17: "tab:blue", 42: "tab:orange", 73: "tab:green"}
    for seed in seeds:
        seed_rows = _read_seed_predictions(base, seed)
        if not seed_rows: continue
        sy = np.array([int(r["label"]) for r in seed_rows]); sp = np.array([float(r["probability"]) for r in seed_rows])
        series.append((f"seed {seed}", sy, sp, colors.get(seed), 1.0))

    fpr_tpr = [(label, *roc_curve(sy, sp)[:2], color, lw) for label, sy, sp, color, lw in series]
    pr = [(label, *precision_recall_curve(sy, sp)[1::-1], color, lw) for label, sy, sp, color, lw in series]
    for name, curves, xlabel, ylabel in (("roc_curve_all_seeds_and_ensemble.png", fpr_tpr, "FPR", "TPR"),
                                          ("pr_curve_all_seeds_and_ensemble.png", pr, "Recall", "Precision")):
        fig, ax = plt.subplots()
        for label, x, z, color, lw in curves:
            ax.plot(x, z, label=label, color=color, linewidth=lw)
        ax.set(xlabel=xlabel, ylabel=ylabel); ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.tight_layout(); out = figures / name; fig.savefig(out, dpi=140); plt.close(fig); made.append(out)

    fig,ax=plt.subplots(); ConfusionMatrixDisplay(confusion_matrix(y,yp)).plot(ax=ax); out=figures/"confusion_matrix_ensemble.png"; fig.savefig(out,dpi=140); plt.close(fig); made.append(out)
    true,prob=calibration_curve(y,p,n_bins=10); fig,ax=plt.subplots(); ax.plot(prob,true,marker="o"); ax.plot([0,1],[0,1],"--"); out=figures/"calibration_curve_ensemble.png"; fig.savefig(out,dpi=140); plt.close(fig); made.append(out)
    fig,ax=plt.subplots(); ax.hist(p[y==0],alpha=.6,label="negative"); ax.hist(p[y==1],alpha=.6,label="positive"); ax.legend(); out=figures/"probability_distribution_ensemble.png"; fig.savefig(out,dpi=140); plt.close(fig); made.append(out)

    cases=[]
    for row,predicted in zip(rows,yp): cases.append({**row,"prediction":int(predicted),"case":("TP" if int(row["label"]) and predicted else "FN" if int(row["label"]) else "FP" if predicted else "TN")})
    cases=sorted(cases,key=lambda r:(r["case"],-abs(float(r["probability"])-threshold)))
    write_csv(base/"ensemble/validation_error_cases.csv",cases)  # kept regardless of image availability

    from PIL import Image
    per_case = {}
    for case in cases:
        per_case.setdefault(case["case"], []).append(case)
    kinds = [k for k in ("TP", "TN", "FP", "FN") if per_case.get(k)]
    if kinds:
        columns = 4
        fig, axes = plt.subplots(len(kinds), columns, figsize=(3 * columns, 3.2 * len(kinds)), squeeze=False)
        for row_idx, kind in enumerate(kinds):
            examples = per_case[kind][:columns]
            for col_idx in range(columns):
                ax = axes[row_idx][col_idx]; ax.axis("off")
                if col_idx >= len(examples):
                    continue
                case = examples[col_idx]
                image_path = case.get("processed_path")
                caption = (f"{kind}  label={case['label']} pred={case['prediction']}\n"
                           f"p={float(case['probability']):.3f}  {case['patient_id']}/{case['image_id']}")
                if image_path and Path(image_path).is_file():
                    ax.imshow(Image.open(image_path).convert("L"), cmap="gray")
                else:
                    ax.text(.5, .5, "image path\nunavailable", ha="center", va="center", fontsize=8)
                ax.set_title(caption, fontsize=7)
            axes[row_idx][0].set_ylabel(kind, fontsize=10)
        fig.tight_layout()
        out = figures / "validation_error_examples.png"
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); made.append(out)
    return made


def persist_summary(base: Path) -> dict:
    rows=[]
    for seed in (17,42,73):
        path=base/f"seed_{seed}"/f"validation_metrics_seed_{seed}.json"
        if path.is_file(): rows.append({"kind":f"seed_{seed}",**json.loads(path.read_text())})
    ensemble=base/"ensemble/metrics/ensemble_validation_metrics.json"
    if ensemble.is_file(): rows.append({"kind":"ensemble",**json.loads(ensemble.read_text())})
    if rows:
        write_csv(base/"ensemble/metrics_summary.csv",rows); atomic_json(base/"ensemble/metrics_summary.json",{"rows":rows})
    curves=render_training_curves(base)
    render_training_curves_all_seeds(base)
    payload={"metrics_rows":len(rows),"results_dir":str(base),"complete":bool(rows)}
    atomic_json(base/"notebook/execution_summary.json",payload)
    return payload


def render_complete_report(base: Path) -> dict:
    result = {"training_curves": [str(p) for p in render_training_curves(base)],
              "validation_figures": [str(p) for p in render_validation_figures(base)],
              "summary": persist_summary(base)}
    # Notebook callers invoke this one function: show the saved scientific artefacts directly,
    # while retaining a no-op fallback for CLI/test environments without IPython.
    try:
        from IPython.display import Image, display
        images = [Path(path) for key in ("training_curves", "validation_figures") for path in result[key]]
        images += [base / "dataset" / name for name in ("dataset_class_distribution.png", "dataset_source_distribution.png", "dataset_source_by_class.png", "training_samples_grid.png")]
        images += sorted((base / "ensemble" / "interpretability" / "real").glob("*.png"))
        for image in images:
            if image.is_file(): display(Image(filename=str(image)))
    except Exception:
        pass
    return result


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
