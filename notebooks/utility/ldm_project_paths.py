from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_NAME = "MammoDiffusion"
DEFAULT_EXPERIMENT_NAME = "diffusers/05_ldm_basic_fromscratch"
RESULTS_STAGE_NAME = "diffusers/05_ldm_basic_fromscratch"


@dataclass(frozen=True)
class ExperimentPaths:
    """Raggruppa tutti i path di un esperimento LDM (checkpoint, modelli, latenti, sintetiche), così gli script non li ricostruiscono ognuno per conto proprio."""

    project_root: Path
    experiment_dir: Path
    data_processed_dir: Path
    metadata_dir: Path
    checkpoints_dir: Path
    models_dir: Path
    latents_dir: Path
    logs_dir: Path
    evaluation_dir: Path
    synthetic_raw_dir: Path
    synthetic_filtered_dir: Path


@dataclass(frozen=True)
class ResultsPaths:
    """Raggruppa i path di una stage di results (plot, metriche, ecotracker) condivisi tra notebook e script di valutazione."""

    stage_dir: Path
    plots_dir: Path
    metrics_dir: Path
    ecotracker_dir: Path


CLASS_NAME_BY_LABEL = {0: "negative", 1: "positive"}


def class_name_for_label(target_label: int) -> str:
    """Restituisce il nome stabile della classe usato nei percorsi di output."""
    try:
        return CLASS_NAME_BY_LABEL[int(target_label)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"target_label deve essere 0 o 1, ricevuto: {target_label!r}") from exc


def get_class_image_dirs(paths: ExperimentPaths, target_label: int) -> tuple[Path, Path]:
    """Restituisce raw e filtered canonici senza rompere gli output LDM positivi storici."""
    class_name = class_name_for_label(target_label)
    if class_name == "positive":
        return paths.synthetic_raw_dir, paths.synthetic_filtered_dir
    return (
        paths.experiment_dir / "synthetic_raw_negative",
        paths.synthetic_filtered_dir.parent / "negative",
    )


def get_class_evaluation_dir(paths: ExperimentPaths, target_label: int) -> Path:
    """Separa validation e test delle due classi nella cartella dell'esperimento."""
    return paths.evaluation_dir / class_name_for_label(target_label)


def get_class_metrics_dir(paths: ResultsPaths, target_label: int) -> Path:
    """Separa le metriche canoniche positive e negative nello stage results."""
    return paths.metrics_dir / class_name_for_label(target_label)


def find_project_root(
    project_name: str = PROJECT_NAME,
    override: Path | None = None,
) -> Path:
    """Risale dalla cwd cercando la cartella del progetto, con fallback per ambienti Colab/Drive."""
    if override is not None:
        root = Path(override).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"PROJECT_ROOT non esiste: {root}")
        return root

    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        has_notebook_dir = (candidate / "notebooks").exists()
        if candidate.name == project_name:
            return candidate
        if (candidate / "data").exists() and has_notebook_dir:
            return candidate

    for candidate in [
        cwd / project_name,
        Path("/content") / project_name,
        Path("/content/drive/MyDrive") / project_name,
        Path.home() / project_name,
    ]:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Non riesco a trovare la root MammoDiffusion. "
        "Passa --project-root oppure esegui dalla repo."
    )


def get_experiment_paths(
    project_root: Path | None = None,
    experiment_dir: Path | None = None,
    create: bool = True,
) -> ExperimentPaths:
    """Centralizza i path dell'esperimento LDM (checkpoint, latents, sintetiche) usati da tutti gli script helper."""
    root = find_project_root(override=project_root)
    exp = (
        Path(experiment_dir).expanduser().resolve()
        if experiment_dir is not None
        else root / "experiments" / DEFAULT_EXPERIMENT_NAME
    )
    # Ogni esperimento scrive le immagini filtrate finali in una sottocartella
    # esplicitamente legata al suo ID, evitando collisioni e nomi non interpretabili.
    FILTERED_DIR_NAME_BY_EXPERIMENT = {
        "05_ldm_basic_fromscratch": "05_ldm_basic_fromscratch",
        "06_ldm_extra1361_fromscratch": "06_ldm_extra1361_fromscratch",
        "20260703_ldm_sdvae_extra1361": "07_ldm_sdvae_extra1361",
        "20260707_ldm_v3_sdvae_extra1361": "08_ldm_v3_sdvae_fromscratch",
        "07_ldm_sdvae_extra1361": "07_ldm_sdvae_extra1361",
        "08_ldm_v3_sdvae_fromscratch": "08_ldm_v3_sdvae_fromscratch",
    }
    filtered_dir_name = FILTERED_DIR_NAME_BY_EXPERIMENT.get(exp.name, exp.name)

    paths = ExperimentPaths(
        project_root=root,
        experiment_dir=exp,
        data_processed_dir=root / "data" / "processed",
        metadata_dir=root / "data" / "processed" / "metadata",
        checkpoints_dir=exp / "checkpoints_ldm",
        models_dir=exp / "models",
        latents_dir=exp / "latents",
        logs_dir=exp / "logs",
        evaluation_dir=exp / "evaluation",
        synthetic_raw_dir=exp / "synthetic_raw",
        synthetic_filtered_dir=root / "data" / "synthetic" / filtered_dir_name / "positive",
    )
    if create:
        for directory in [
            paths.checkpoints_dir,
            paths.models_dir,
            paths.latents_dir,
            paths.logs_dir,
            paths.evaluation_dir,
            paths.synthetic_raw_dir,
            paths.synthetic_filtered_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    return paths


def get_results_paths(
    project_root: Path | None = None,
    stage_name: str = RESULTS_STAGE_NAME,
    create: bool = True,
) -> ResultsPaths:
    """Centralizza i path dei risultati (plot, metriche, ecotracker) usati dagli script di valutazione e dai notebook."""
    root = find_project_root(override=project_root)
    stage_dir = root / "results" / stage_name
    paths = ResultsPaths(
        stage_dir=stage_dir,
        plots_dir=stage_dir / "plots",
        metrics_dir=stage_dir / "metrics",
        ecotracker_dir=stage_dir / "ecotracker",
    )
    if create:
        for directory in [
            paths.stage_dir,
            paths.plots_dir,
            paths.metrics_dir,
            paths.ecotracker_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    return paths


def normalize_processed_path(row, dataset_root: Path) -> Path:
    """Ricalcola il path dell'immagine a partire da split/label/filename, per non dipendere dal path assoluto salvato nel CSV (cambia da macchina a macchina)."""
    processed_path = Path(str(row["processed_path"]).replace("\\", "/"))
    filename = processed_path.name
    split = str(row["split"])
    label = str(int(row["label"]))
    return Path(dataset_root) / split / label / filename
