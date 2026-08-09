from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_NAME = "MammoDiffusion"
DEFAULT_EXPERIMENT_NAME = "diffusers/05_ldm_basic_fromscratch"
RESULTS_STAGE_NAME = "2_diffusers/05_ldm_basic_fromscratch"


@dataclass(frozen=True)
class ExperimentPaths:
    """Every path of one LDM experiment, so no script rebuilds them on its own.

    Each synthetic pool is a field here rather than a string assembled at the call
    site: a pool that only exists as an inline literal is a pool a maintenance step
    can forget, which is how a VAE reset once cleared the positive images and left
    the negative ones behind.
    """

    project_root: Path
    experiment_dir: Path
    data_processed_dir: Path
    metadata_dir: Path
    checkpoints_dir: Path
    models_dir: Path
    latents_dir: Path
    logs_dir: Path
    evaluation_dir: Path
    synthetic_raw_positive_dir: Path
    synthetic_raw_negative_dir: Path
    synthetic_filtered_positive_dir: Path


@dataclass(frozen=True)
class ResultsPaths:
    """Paths of one results stage (plots, metrics, ecotracker), shared by the
    notebooks and the evaluation scripts."""

    stage_dir: Path
    plots_dir: Path
    metrics_dir: Path
    ecotracker_dir: Path


CLASS_NAME_BY_LABEL = {0: "negative", 1: "positive"}

# G05 and G06 predate the shared ``data/synthetic`` layout. Their registered
# operational manifests and benchmark inputs bind the positive pool to
# the experiment-local directory. Keep that identity explicit instead of
# silently substituting a similarly named pool from another experiment.
EXPERIMENT_LOCAL_POSITIVE_FILTERED = {
    "05_ldm_basic_fromscratch",
    "06_ldm_extra1361_fromscratch",
}

FILTERED_DIR_NAME_BY_EXPERIMENT = {
    "05_ldm_basic_fromscratch": "05_ldm_basic_fromscratch",
    "06_ldm_extra1361_fromscratch": "06_ldm_extra1361_fromscratch",
    "20260703_ldm_sdvae_extra1361": "07_ldm_sdvae_extra1361",
    "20260707_ldm_v3_sdvae_extra1361": "08_ldm_v3_sdvae_fromscratch",
    "07_ldm_sdvae_extra1361": "07_ldm_sdvae_extra1361",
    "08_ldm_v3_sdvae_fromscratch": "08_ldm_v3_sdvae_fromscratch",
}


def class_name_for_label(target_label: int) -> str:
    """The stable class name used in every output path."""
    try:
        return CLASS_NAME_BY_LABEL[int(target_label)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"target_label must be 0 or 1; received {target_label!r}") from exc


def get_class_image_dirs(paths: ExperimentPaths, target_label: int) -> tuple[Path, Path]:
    """The registered (raw, filtered) pool pair for one class.

    The positive filtered pool is experiment-local for the generators that predate
    the shared ``data/synthetic`` layout and shared for the others; the negative one
    is always shared. Callers must not reconstruct either path themselves.
    """
    class_name = class_name_for_label(target_label)
    if class_name == "positive":
        return paths.synthetic_raw_positive_dir, paths.synthetic_filtered_positive_dir
    filtered_dir_name = FILTERED_DIR_NAME_BY_EXPERIMENT.get(
        paths.experiment_dir.name,
        paths.experiment_dir.name,
    )
    return (
        paths.synthetic_raw_negative_dir,
        paths.project_root / "data" / "synthetic" / filtered_dir_name / "negative",
    )


def get_class_evaluation_dir(paths: ExperimentPaths, target_label: int) -> Path:
    """Keep validation and test of the two classes apart inside the experiment."""
    return paths.evaluation_dir / class_name_for_label(target_label)


def get_class_metrics_dir(paths: ResultsPaths, target_label: int) -> Path:
    """Keep the canonical positive and negative metrics apart in the results stage."""
    return paths.metrics_dir / class_name_for_label(target_label)


def find_project_root(
    project_name: str = PROJECT_NAME,
    override: Path | None = None,
) -> Path:
    """Walk up from the working directory to the project root, with Colab/Drive
    fallbacks, so notebooks never hard-code a workstation-specific path."""
    if override is not None:
        root = Path(override).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"PROJECT_ROOT does not exist: {root}")
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
        "Could not locate the MammoDiffusion repository root. "
        "Pass --project-root or run from the repository."
    )


def get_experiment_paths(
    project_root: Path | None = None,
    experiment_dir: Path | None = None,
    create: bool = True,
) -> ExperimentPaths:
    """Resolve every path of one LDM experiment; ``create`` builds the skeleton."""
    root = find_project_root(override=project_root)
    exp = (
        Path(experiment_dir).expanduser().resolve()
        if experiment_dir is not None
        else root / "experiments" / DEFAULT_EXPERIMENT_NAME
    )
    # Each experiment writes its final filtered images under a subdirectory tied to
    # its own ID, which avoids collisions and directory names nobody can interpret.
    filtered_dir_name = FILTERED_DIR_NAME_BY_EXPERIMENT.get(exp.name, exp.name)
    if exp.name in EXPERIMENT_LOCAL_POSITIVE_FILTERED:
        positive_filtered_dir = exp / "synthetic_filtered_positive"
    else:
        positive_filtered_dir = root / "data" / "synthetic" / filtered_dir_name / "positive"

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
        synthetic_raw_positive_dir=exp / "synthetic_raw_positive",
        synthetic_raw_negative_dir=exp / "synthetic_raw_negative",
        synthetic_filtered_positive_dir=positive_filtered_dir,
    )
    if create:
        # synthetic_raw_negative_dir stays out of the eagerly created skeleton:
        # positive-only experiments (G05, "classes": ["positive"] in the registry)
        # must not end up with an empty negative pool suggesting one that was never
        # generated. Generation creates it if and when it is actually needed.
        for directory in [
            paths.checkpoints_dir,
            paths.models_dir,
            paths.latents_dir,
            paths.logs_dir,
            paths.evaluation_dir,
            paths.synthetic_raw_positive_dir,
            paths.synthetic_filtered_positive_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    return paths


def get_results_paths(
    project_root: Path | None = None,
    stage_name: str = RESULTS_STAGE_NAME,
    create: bool = True,
) -> ResultsPaths:
    """Resolve the results-stage paths used by the evaluation scripts and notebooks."""
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
    """Rebuild an image path from split/label/filename.

    The absolute path stored in the CSV is machine-specific; recomputing it keeps
    the metadata portable across the workstations the project has run on.
    """
    processed_path = Path(str(row["processed_path"]).replace("\\", "/"))
    filename = processed_path.name
    split = str(row["split"])
    label = str(int(row["label"]))
    return Path(dataset_root) / split / label / filename
