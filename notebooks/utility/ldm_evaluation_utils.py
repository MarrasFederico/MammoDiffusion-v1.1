from __future__ import annotations

import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mammodiffusion-matplotlib")
)

# Garantisce che la cartella notebooks/ sia in sys.path anche se questo modulo
# viene importato in un contesto dove lo script invocato direttamente non e'
# nella stessa cartella (es. mount di rete, simlink, kernel diversi): senza
# questo, l'import del modulo fratello sotto puo' fallire con
# "ModuleNotFoundError: No module named 'generative_evaluator'" anche quando
# il file esiste, perche' sys.path[0] non e' quello che ci si aspetta.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from generative_evaluator import GenerativeEvaluator
from ldm_project_paths import normalize_processed_path
from prdc import compute_prdc


def _link_or_copy(source: Path, destination: Path) -> None:
    """Crea un symlink verso l'immagine originale (evita di duplicare i dati); se il filesystem non lo permette, ricade su una copia."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, destination)
    except OSError:
        shutil.copy2(source, destination)


@contextmanager
def temporary_image_dir(image_paths: Iterable[Path], prefix: str):
    """Raccoglie un set di immagini sparse in un'unica cartella temporanea con nomi numerati, formato richiesto da GenerativeEvaluator; la cartella viene rimossa automaticamente all'uscita dal context manager."""
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError(f"Nessuna immagine per {prefix}")

    with tempfile.TemporaryDirectory(prefix=prefix) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for index, source in enumerate(paths):
            if not source.exists():
                raise FileNotFoundError(source)
            suffix = source.suffix.lower() or ".png"
            _link_or_copy(source, tmp_dir / f"{index:05d}{suffix}")
        yield tmp_dir


def real_paths_from_metadata(metadata_df, data_processed_dir: Path, label: int) -> list[Path]:
    """Filtra il CSV di metadata su una singola classe e ne ricostruisce i path reali, da usare come riferimento per FID/PRDC."""
    label_df = metadata_df[metadata_df["label"].astype(int) == int(label)].copy()
    if label_df.empty:
        raise RuntimeError(f"Nessuna immagine reale per label {label}")
    return [
        normalize_processed_path(row, data_processed_dir)
        for _, row in label_df.iterrows()
    ]


def compute_prdc_metrics(real_features, fake_features, nearest_k: int = 3) -> dict[str, float]:
    """Calcola Precision/Recall/Density/Coverage tra due insiemi di feature, riducendo k se il batch è troppo piccolo per supportare quello richiesto."""
    # prdc internally queries k + 1 neighbours, so tiny trial runs need at least
    # k + 2 samples per side.
    k = min(int(nearest_k), len(real_features) - 2, len(fake_features) - 2)
    if k < 1:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "density": float("nan"),
            "coverage": float("nan"),
        }

    metrics = compute_prdc(
        real_features=np.asarray(real_features),
        fake_features=np.asarray(fake_features),
        nearest_k=k,
    )
    return {
        name: round(float(metrics[name]), 4)
        for name in ("precision", "recall", "density", "coverage")
    }


def evaluate_generated_paths_against_real_paths(
    real_paths: Iterable[Path],
    generated_paths: Iterable[Path],
    batch_size: int = 8,
    image_size: int = 299,
    num_workers: int = 0,
    device: str | None = None,
    nearest_k: int = 3,
    is_splits: int = 10,
) -> dict[str, float]:
    """Calcola FID/IS/PRDC tra due liste di immagini già note, materializzandole in cartelle temporanee per GenerativeEvaluator."""
    real_paths = [Path(path) for path in real_paths]
    generated_paths = [Path(path) for path in generated_paths]

    with temporary_image_dir(real_paths, "ldm_real_") as real_dir:
        with temporary_image_dir(generated_paths, "ldm_generated_") as generated_dir:
            evaluator = GenerativeEvaluator(
                real_dir=real_dir,
                generated_dir=generated_dir,
                batch_size=batch_size,
                image_size=image_size,
                num_workers=num_workers,
                device=device,
                is_splits=is_splits,
            )
            fid_is_metrics, real_features, fake_features = evaluator.compute_with_features()

    return {
        **fid_is_metrics,
        **compute_prdc_metrics(real_features, fake_features, nearest_k=nearest_k),
        "n_real_reference": len(real_paths),
        "n_generated": len(generated_paths),
    }


def evaluate_generated_paths_against_metadata(
    generated_paths: Iterable[Path],
    metadata_df,
    data_processed_dir: Path,
    label: int,
    batch_size: int = 8,
    image_size: int = 299,
    num_workers: int = 0,
    device: str | None = None,
    nearest_k: int = 3,
    is_splits: int = 10,
) -> dict[str, float]:
    """Stessa valutazione di evaluate_generated_paths_against_real_paths, ma usando come riferimento reale una classe del CSV di metadata invece di un elenco di path già pronto."""
    real_paths = real_paths_from_metadata(metadata_df, data_processed_dir, label)
    return evaluate_generated_paths_against_real_paths(
        real_paths=real_paths,
        generated_paths=generated_paths,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        device=device,
        nearest_k=nearest_k,
        is_splits=is_splits,
    )
