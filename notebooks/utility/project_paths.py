"""The one authoritative way to interpret a path recorded by an earlier run.

Frozen manifests, caches, and result records store absolute paths from the
workstation that produced them, typically under a project root that no longer
exists at that location. Those strings are provenance and are never rewritten.
Every *consumer*, however, has to answer two questions safely:

* is this recorded location the same place as the one I am looking at now?
* where does it live in this checkout?

Answering either by string comparison is wrong twice over. It makes a valid
frozen artifact look incompatible as soon as the project is renamed or moved,
and on the original workstation the historical prefix now resolves, through a
symlink, into an entirely different repository -- so following it verbatim would
read another project's files.

The rule implemented here is: identity is the repository-relative suffix, and
resolution always lands inside the current checkout or fails.
"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

__all__ = [
    "PROJECT_MARKERS",
    "project_relative_suffix",
    "paths_equivalent",
    "resolve_project_path",
]

# Top-level directories that make a path recognisably part of this project.
PROJECT_MARKERS = ("data", "experiments", "results", "configs", "notebooks", "assets", "tests")


def _normalise(path: Path) -> Path:
    """Absolute, ``..``-free, and *without* following symlinks.

    ``Path.resolve()`` would be simpler but resolves symlinks, which breaks
    containment checks for the common setup where ``data/`` or ``experiments/``
    is a symlink to a larger volume: a file genuinely inside the project would
    then appear to live outside it. Only ``..`` needs collapsing, and that can be
    done logically.
    """
    absolute = path if path.is_absolute() else Path.cwd() / path
    return Path(os.path.normpath(str(absolute)))


def project_relative_suffix(value: object) -> PurePosixPath | None:
    """The repository-relative part of a recorded path, or None if unrecognised.

    ``/old/root/MammoDiffusion/experiments/diffusers/x`` and
    ``/new/MammoDiffusion-v1.1/experiments/diffusers/x`` both reduce to
    ``experiments/diffusers/x``. The *last* marker occurrence wins, so a project
    that itself lives under a directory called ``data`` is handled correctly.
    """
    if value is None:
        return None
    text = str(value).replace("\\", "/").strip()
    if not text:
        return None
    parts = PurePosixPath(text).parts
    parts = tuple(part for part in parts if part not in ("/", ""))
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] in PROJECT_MARKERS:
            return PurePosixPath(*parts[index:])
    return None


def paths_equivalent(first: object, second: object) -> bool:
    """True when two recorded paths denote the same location in this project.

    Falls back to exact comparison when neither side is recognisably
    project-relative, so unrelated absolute paths are still distinguished.
    """
    left, right = project_relative_suffix(first), project_relative_suffix(second)
    if left is not None and right is not None:
        return left == right
    if left is None and right is None:
        return str(first) == str(second)
    return False


def resolve_project_path(root: Path | str, value: object, *,
                         must_stay_inside: bool = True) -> Path:
    """Map a recorded path onto ``root``, refusing to leave the checkout.

    A relative path is joined to ``root``. An absolute path is reduced to its
    repository-relative suffix and re-anchored, which is what keeps a stale
    prefix from being followed. An absolute path with no recognisable marker is
    an external asset: returned unchanged when ``must_stay_inside`` is False,
    and rejected otherwise.
    """
    root = _normalise(Path(root))
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        resolved = _normalise(root / path)
    else:
        normalised = _normalise(path)
        if normalised == root or root in normalised.parents:
            resolved = normalised
        else:
            suffix = project_relative_suffix(normalised)
            if suffix is None:
                if must_stay_inside:
                    raise ValueError(
                        f"path cannot be safely rerooted under the project: {value}")
                return normalised
            resolved = _normalise(root / Path(*suffix.parts))
    if must_stay_inside and not (resolved == root or root in resolved.parents):
        raise ValueError(f"resolved path escapes the project root: {value} -> {resolved}")
    return resolved
