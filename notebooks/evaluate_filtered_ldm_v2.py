#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Definisce gli argomenti CLI dello script: percorsi dell'esperimento, cartella delle sintetiche filtrate e parametri delle metriche generative."""
    parser = argparse.ArgumentParser(
        description="Evaluate selected filtered LDM images against the held-out test set."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--gpu-visible-devices", default=None)
    parser.add_argument(
        "--cuda-root",
        type=Path,
        default=Path("/home/fede/miniforge3/envs/tf-gpu"),
    )
    parser.add_argument("--synthetic-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-label", type=int, default=1)
    parser.add_argument("--max-synthetic", type=int, default=None)
    parser.add_argument("--inception-batch", type=int, default=8)
    parser.add_argument("--is-splits", type=int, default=10)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument(
        "--results-stage-name",
        default="04_ldm_keras_v2",
        help="Sottocartella di results dove salvare le metriche finali.",
    )
    parser.add_argument(
        "--inception-weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Mantenuto per compatibilita'; generative_evaluator.py usa torchmetrics.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignora final_filtered_vs_test.json e ricalcola le metriche finali.",
    )
    parser.add_argument(
        "--no-canonical-results",
        action="store_true",
        help="Non copiare CSV/JSON nello stage results canonico (utile per smoke test).",
    )
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    """Imposta le variabili d'ambiente per GPU e XLA prima di importare TensorFlow/PyTorch, così la configurazione hardware è quella richiesta da CLI e non quella di default del sistema."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    if args.gpu_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible_devices

    libdevice = args.cuda_root / "nvvm" / "libdevice" / "libdevice.10.bc"
    if libdevice.exists() and "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={args.cuda_root}"


def main() -> None:
    """Valuta le sintetiche filtrate selezionate per l'augmentation contro il test set reale (FID, IS, PRDC): è la valutazione finale, congelata, del modello LDM."""
    args = parse_args()
    configure_environment(args)

    import pandas as pd

    from ldm_project_paths import get_experiment_paths, get_results_paths

    paths = get_experiment_paths(args.project_root, args.experiment_dir)
    results_paths = get_results_paths(paths.project_root, args.results_stage_name)
    synthetic_dir = (
        args.synthetic_dir.expanduser().resolve()
        if args.synthetic_dir is not None
        else paths.synthetic_filtered_dir
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else paths.evaluation_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_paths = sorted(synthetic_dir.glob("*.png"))
    if args.max_synthetic is not None:
        synthetic_paths = synthetic_paths[: args.max_synthetic]
    if not synthetic_paths:
        raise FileNotFoundError(f"Nessuna immagine sintetica filtrata in {synthetic_dir}")

    test_df = pd.read_csv(paths.metadata_dir / "test.csv")
    test_label_df = test_df[test_df["label"].astype(int) == int(args.target_label)].copy()
    if test_label_df.empty:
        raise RuntimeError(f"Nessuna immagine test per label {args.target_label}")

    if args.inception_weights == "none":
        print(
            "Nota: --inception-weights=none viene ignorato con generative_evaluator.py; "
            "torchmetrics usa il suo modello Inception."
        )

    print(f"Reali test label {args.target_label}: {len(test_label_df)}")
    print(f"Sintetiche filtrate selezionate: {len(synthetic_paths)}")

    csv_path = output_dir / "final_filtered_vs_test.csv"
    json_path = output_dir / "final_filtered_vs_test.json"
    canonical_csv_path = results_paths.metrics_dir / "final_filtered_vs_test.csv"
    canonical_json_path = results_paths.metrics_dir / "final_filtered_vs_test.json"
    eval_config = {
        "metric_backend": "generative_evaluator.py",
        "reference_split": "test",
        "target_label": int(args.target_label),
        "synthetic_dir": str(synthetic_dir),
        "max_synthetic": args.max_synthetic,
        "inception_batch": int(args.inception_batch),
        "is_splits": int(args.is_splits),
        "knn_k": int(args.knn_k),
        "inception_weights": args.inception_weights,
    }
    input_signature = {
        "synthetic_files": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in synthetic_paths
        ],
        "test_csv": {
            "path": str(paths.metadata_dir / "test.csv"),
            "size": (paths.metadata_dir / "test.csv").stat().st_size,
            "mtime_ns": (paths.metadata_dir / "test.csv").stat().st_mtime_ns,
        },
    }

    def use_cached_metrics_if_valid() -> bool:
        """Riusa le metriche già calcolate se config e input (file sintetici + test.csv) non sono cambiati, evitando di ripetere un calcolo costoso (FID/IS su GPU) senza motivo."""
        if args.force_recompute or not json_path.exists():
            return False
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception as exc:
            print(f"Cache metriche finali non leggibile, ricalcolo: {exc}")
            return False

        if payload.get("schema_version") != 2:
            print("Cache metriche finali con schema non valido, ricalcolo.")
            return False
        if payload.get("config") != eval_config:
            print("Cache metriche finali con configurazione diversa, ricalcolo.")
            return False
        if payload.get("input_signature") != input_signature:
            print("Cache metriche finali con input diversi, ricalcolo.")
            return False

        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            print("Cache metriche finali senza payload metrics, ricalcolo.")
            return False
        if not csv_path.exists():
            pd.DataFrame([metrics]).to_csv(csv_path, index=False)
            print(f"CSV metriche finali ricreato da cache: {csv_path}")
        if not args.no_canonical_results:
            pd.DataFrame([metrics]).to_csv(canonical_csv_path, index=False)
            with open(canonical_json_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
        print(f"Metriche finali filtrate gia' presenti: {json_path}")
        print(pd.DataFrame([metrics]).to_string(index=False))
        return True

    if use_cached_metrics_if_valid():
        return

    from ldm_evaluation_utils import evaluate_generated_paths_against_metadata

    evaluator_metrics = evaluate_generated_paths_against_metadata(
        generated_paths=synthetic_paths,
        metadata_df=test_df,
        data_processed_dir=paths.data_processed_dir,
        label=args.target_label,
        batch_size=args.inception_batch,
        nearest_k=args.knn_k,
        is_splits=args.is_splits,
    )

    metrics = {
        "metric_backend": "generative_evaluator.py",
        "reference_split": "test",
        "target_label": int(args.target_label),
        "n_real_test": int(evaluator_metrics["n_real_reference"]),
        "n_synthetic_filtered": int(evaluator_metrics["n_generated"]),
        "synthetic_dir": str(synthetic_dir),
        "inception_batch": int(args.inception_batch),
        "is_splits": int(args.is_splits),
        "knn_k": int(args.knn_k),
        "FID": float(evaluator_metrics["FID"]),
        "IS_mean": float(evaluator_metrics["IS_mean"]),
        "IS_std": float(evaluator_metrics["IS_std"]),
        "precision": float(evaluator_metrics["precision"]),
        "recall": float(evaluator_metrics["recall"]),
        "density": float(evaluator_metrics["density"]),
        "coverage": float(evaluator_metrics["coverage"]),
    }

    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    if not args.no_canonical_results:
        pd.DataFrame([metrics]).to_csv(canonical_csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "schema_version": 2,
                "config": eval_config,
                "input_signature": input_signature,
                "metrics": metrics,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    if not args.no_canonical_results:
        with open(canonical_json_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "schema_version": 2,
                    "config": eval_config,
                    "input_signature": input_signature,
                    "metrics": metrics,
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

    print("\n=== METRICHE FINALI FILTRATE VS TEST ===")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print("Salvato:", csv_path)
    print("Salvato:", json_path)
    if not args.no_canonical_results:
        print("Salvato:", canonical_csv_path)
        print("Salvato:", canonical_json_path)


if __name__ == "__main__":
    main()
