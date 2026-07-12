from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import re
from datetime import datetime
from pathlib import Path


FILTER_SUMMARY_SCHEMA_VERSION = 2
FILTER_SCHEMA_VERSION = "adaptive_mask_v2"
from parallel_generation_utils import GENERATED_PNG_PATTERN
_CANDIDATE_NAME = re.compile(GENERATED_PNG_PATTERN, re.IGNORECASE)


def count_pngs(directory: Path) -> int:
    """Conta solo PNG leggibili e non temporanei."""
    return len(candidate_png_paths(directory))


def candidate_png_paths(directory: Path) -> list[Path]:
    """Dataset effettivo del filtro: immagini leggibili, mai file temporanei."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    from PIL import Image
    result = []
    for path in sorted(directory.glob("*.png")):
        if path.name.startswith(".tmp_"):
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            result.append(path)
        except Exception:
            continue
    return (
        [path for path in result if _CANDIDATE_NAME.fullmatch(path.name)]
        if any(_CANDIDATE_NAME.fullmatch(path.name) for path in result)
        else result
    )


def file_sha256(path: Path) -> str:
    """Calcola l'hash SHA256 di un file per individuare duplicati esatti."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duplicate_png_groups(directory: Path) -> dict[str, list[str]]:
    """Raggruppa i PNG per hash del contenuto e restituisce solo i gruppi con più di un file."""
    by_hash: dict[str, list[str]] = {}
    for path in candidate_png_paths(directory):
        by_hash.setdefault(file_sha256(path), []).append(path.name)
    return {digest: names for digest, names in by_hash.items() if len(names) > 1}


def real_train_paths(
    train_metadata: Path,
    data_processed_dir: Path,
    label: int,
    seed: int,
) -> list[Path]:
    """Recupera, in ordine casuale riproducibile, i path delle immagini reali di training con la label richiesta, da usare come riferimento per il filtro adattivo."""
    import pandas as pd

    from ldm_project_paths import normalize_processed_path

    metadata = pd.read_csv(train_metadata)
    required = {"label", "processed_path"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Colonne mancanti in {train_metadata}: {sorted(missing)}")

    subset = metadata[metadata["label"].astype(int) == int(label)]
    if subset.empty:
        raise ValueError(f"Nessuna immagine label={label} in {train_metadata}")
    subset = subset.sample(n=len(subset), random_state=seed)

    paths: list[Path] = []
    for _, row in subset.iterrows():
        if "split" in metadata.columns:
            candidate = normalize_processed_path(row, data_processed_dir)
        else:
            raw_path = Path(str(row["processed_path"]).replace("\\", "/"))
            candidate = data_processed_dir / train_metadata.stem.lower() / str(int(label)) / raw_path.name
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        paths.append(candidate.resolve())
    return paths


def selected_output_name(class_name: str, rank: int) -> str:
    """Costruisce il nome file dell'immagine accettata in base alla classe e alla posizione in classifica."""
    # Mantiene la convenzione storica gia' presente nel 03b.
    prefix = "selected_pos" if class_name == "positive" else f"selected_{class_name}"
    return f"{prefix}_{rank:04d}.png"


def filter_ranking_config(args: argparse.Namespace) -> dict[str, object]:
    """Raccoglie i parametri che determinano la classifica e la selezione, usati anche per verificare che un summary salvato sia ancora compatibile con la configurazione corrente."""
    return {
        "schema_version": FILTER_SUMMARY_SCHEMA_VERSION,
        "filter_schema_version": FILTER_SCHEMA_VERSION,
        "selection_score": "robust_distance_from_train_reference",
        "sort_order": "descending_selection_score",
        "n_selected": int(args.n_selected),
        "eval_seed": int(args.eval_seed),
        "reference_seed": int(args.eval_seed) + int(args.label),
        "nonblack_threshold": int(args.nonblack_threshold),
        "output_naming": "selected_output_name_v1",
    }


def paths_match(left: object, right: Path) -> bool:
    """Confronta due path risolvendoli a percorso assoluto, per non farsi ingannare da differenze di formattazione (relativo vs assoluto, slash finali, ecc.)."""
    if left is None:
        return False
    left_path = Path(str(left)).expanduser()
    right_path = Path(right).expanduser()
    try:
        return left_path.resolve() == right_path.resolve()
    except OSError:
        return left_path == right_path


def summary_is_complete_and_compatible(args: argparse.Namespace) -> bool:
    """Verifica se il summary_json salvato in precedenza è ancora valido per la run corrente (stesso schema, stessi parametri, directory e conteggi coerenti), per evitare di ripetere il filtro se non serve."""
    summary_path = Path(args.summary_json)
    raw_dir = Path(args.raw_dir)
    filtered_dir = Path(args.filtered_dir)
    n_selected = int(args.n_selected)

    if not summary_path.is_file():
        return False
    try:
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False

    required_non_null = [
        "schema_version",
        "filter_schema_version",
        "status",
        "experiment_name",
        "class_name",
        "label",
        "raw_dir",
        "filtered_dir",
        "n_raw",
        "n_selected",
        "thresholds",
        "reference_medians",
        "reference_iqrs",
        "n_real_train_reference",
        "reject_reason_counts",
        "ranking_config",
    ]
    if any(summary.get(key) is None for key in required_non_null):
        return False
    if summary.get("status") != "completed":
        return False
    if summary.get("schema_version") != FILTER_SUMMARY_SCHEMA_VERSION:
        return False
    if summary.get("filter_schema_version") != FILTER_SCHEMA_VERSION:
        return False
    if summary.get("experiment_name") != args.experiment_name:
        return False
    if summary.get("class_name") != args.class_name:
        return False
    if int(summary.get("label")) != int(args.label):
        return False
    if int(summary.get("n_raw")) != count_pngs(raw_dir):
        return False
    if int(summary.get("n_selected")) != n_selected:
        return False
    if not paths_match(summary.get("raw_dir"), raw_dir):
        return False
    if not paths_match(summary.get("filtered_dir"), filtered_dir):
        return False
    if summary.get("ranking_config") != filter_ranking_config(args):
        return False
    if count_pngs(filtered_dir) != n_selected:
        return False
    if duplicate_png_groups(filtered_dir):
        return False
    return True


def run_filter(args: argparse.Namespace) -> None:
    """Esegue l'intera pipeline del filtro adattivo per una classe: valuta ogni immagine RAW contro le statistiche di riferimento, seleziona le n_selected migliori, copia gli accettati in filtered_dir e scrive report CSV e summary JSON. Salta il lavoro se trova già un risultato completo e compatibile."""
    raw_dir = Path(args.raw_dir)
    filtered_dir = Path(args.filtered_dir)
    report_csv = Path(args.report_csv)
    summary_json = Path(args.summary_json)
    n_selected = int(args.n_selected)

    if count_pngs(filtered_dir) == n_selected and summary_is_complete_and_compatible(args):
        print(
            f"{args.class_name}: {n_selected} immagini filtrate già presenti "
            "con summary compatibile, skip"
        )
        return
    if count_pngs(filtered_dir) == n_selected:
        print(
            f"{args.class_name}: directory filtrata completa ma summary assente "
            "o incompatibile; rieseguo il filtro."
        )

    raw_paths = candidate_png_paths(raw_dir)
    if not raw_paths:
        raise RuntimeError(f"Nessuna immagine RAW in {raw_dir}")

    import pandas as pd

    from adaptive_mammography_filter import (
        compute_reference_statistics,
        evaluate_candidate,
        image_quality_metrics,
        estimate_foreground_mask,
    )

    _ = image_quality_metrics, estimate_foreground_mask
    reference_paths = real_train_paths(
        train_metadata=Path(args.train_metadata),
        data_processed_dir=Path(args.data_processed_dir),
        label=int(args.label),
        seed=int(args.eval_seed) + int(args.label),
    )
    reference = compute_reference_statistics(
        reference_paths,
        nonblack_threshold=int(args.nonblack_threshold),
        verbose=True,
    )

    candidate_rows = []
    for index, path in enumerate(raw_paths, start=1):
        candidate = evaluate_candidate(
            path,
            reference,
            nonblack_threshold=int(args.nonblack_threshold),
        )
        candidate_rows.append({
            "class": args.class_name,
            "source_path": candidate["raw_path"],
            "source_name": candidate["raw_filename"],
            **{
                key: value
                for key, value in candidate.items()
                if key not in {"raw_path", "raw_filename", "score", "reason"}
            },
            "reject_reason": candidate["reason"],
            "selection_score": candidate["score"],
        })
        if index % 250 == 0 or index == len(raw_paths):
            print(f"  Analisi {args.class_name}: {index}/{len(raw_paths)}")

    candidates = pd.DataFrame(candidate_rows)
    accepted = (
        candidates.loc[~candidates["rejected"]]
        .sort_values("selection_score", ascending=False)
        .copy()
    )
    reject_counts = {
        str(reason): int(count)
        for reason, count in candidates.loc[
            candidates["rejected"], "reject_reason"
        ].value_counts().items()
    }

    report_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    if len(accepted) < n_selected:
        candidates.to_csv(report_csv, index=False)
        raise RuntimeError(
            f"Dopo il filtro {args.class_name} restano {len(accepted)} immagini; "
            f"ne servono {n_selected}."
        )

    selected = accepted.head(n_selected).copy()
    selected["selection_rank"] = range(n_selected)
    rank_by_path = dict(zip(selected["source_path"], selected["selection_rank"]))
    candidates["selected"] = candidates["source_path"].isin(rank_by_path)
    candidates["selection_rank"] = candidates["source_path"].map(rank_by_path)

    filtered_dir.mkdir(parents=True, exist_ok=True)
    for path in filtered_dir.glob("*.png"):
        path.unlink()
    for rank, source_path in enumerate(selected["source_path"]):
        shutil.copy2(source_path, filtered_dir / selected_output_name(args.class_name, rank))

    if count_pngs(filtered_dir) != n_selected:
        raise RuntimeError(f"Numero immagini filtrate non valido per {args.class_name}.")
    duplicates = duplicate_png_groups(filtered_dir)
    if duplicates:
        raise RuntimeError(f"Il dataset filtrato {args.class_name} contiene duplicati.")

    candidates.to_csv(report_csv, index=False)
    summary = {
        "schema_version": FILTER_SUMMARY_SCHEMA_VERSION,
        "filter_schema_version": FILTER_SCHEMA_VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": args.experiment_name,
        "class_name": args.class_name,
        "experiment": args.experiment_name,
        "class": args.class_name,
        "label": int(args.label),
        "raw_dir": str(raw_dir),
        "filtered_dir": str(filtered_dir),
        "n_raw": len(candidates),
        "n_accepted": len(accepted),
        "n_rejected": int(candidates["rejected"].sum()),
        "n_selected": n_selected,
        "thresholds": reference["thresholds"],
        "reference_medians": reference["medians"],
        "reference_iqrs": reference["iqrs"],
        "n_real_train_reference": reference["n_train"],
        "reject_reason_counts": reject_counts,
        "ranking_config": filter_ranking_config(args),
        "status": "completed",
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"{args.class_name}: selezionate {n_selected} immagini")
    print("Report:", report_csv)
    print("Summary:", summary_json)


def parse_args() -> argparse.Namespace:
    """Definisce e legge gli argomenti da riga di comando dello script di filtro."""
    parser = argparse.ArgumentParser(description="Filtro adattivo mammografico per il notebook 03b.")
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--label", required=True, type=int)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--filtered-dir", required=True, type=Path)
    parser.add_argument("--n-selected", required=True, type=int)
    parser.add_argument("--train-metadata", required=True, type=Path)
    parser.add_argument("--data-processed-dir", required=True, type=Path)
    parser.add_argument("--report-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--eval-seed", required=True, type=int)
    parser.add_argument("--nonblack-threshold", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    """Entry point dello script: lancia il filtro e stampa eventuali errori su stderr prima di propagarli."""
    try:
        run_filter(parse_args())
    except Exception as exc:
        print(f"Errore filtro adattivo: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
