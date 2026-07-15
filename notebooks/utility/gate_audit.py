"""Post-benchmark gate calibration audit utilities.

These helpers support a *methodological review* of the generator-benchmark eligibility gates
after the benchmark has run.  They never load feature encoders, never re-extract embeddings, and
never touch the test split.  They only consume already-computed CSVs, cached embeddings, and the
image files needed for perceptual-hash / SSIM diagnostics.

Scope of this module (pure, testable functions):

* order-independent perceptual-hash cluster diagnostics (``perceptual_hash_cluster_diagnostics``);
* confirmed-duplicate detection using the preregistered ``exact | (pHash<=2 AND SSIM>=0.98)`` rule;
* real-vs-real RAD-DINO PRDC baselines from cached embeddings;
* a purely descriptive generator ranking that ignores the eligibility gates;
* a strict efficiency parser that refuses to derive seconds-per-image from ambiguous durations.

Nothing here changes the active gate policy or the protocol configuration.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:  # allow both "import gate_audit" (utility on sys.path) and package-style import
    import generator_benchmark as gb
except ImportError:  # pragma: no cover - fallback for package execution
    from notebooks.utility import generator_benchmark as gb  # type: ignore


# ---------------------------------------------------------------------------
# Order-independent perceptual-hash diagnostics
# ---------------------------------------------------------------------------

def _popcount64(values: np.ndarray) -> np.ndarray:
    """Vectorised 64-bit population count."""
    x = values.astype(np.uint64)
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    h01 = np.uint64(0x0101010101010101)
    one, two, four, fifty_six = (np.uint64(1), np.uint64(2), np.uint64(4), np.uint64(56))
    x = x - ((x >> one) & m1)
    x = (x & m2) + ((x >> two) & m2)
    x = (x + (x >> four)) & m4
    return np.asarray((x * h01) >> fifty_six, dtype=np.int64)


def perceptual_hash_ints(paths: Sequence[str | Path]) -> list[int]:
    """Compute the 64-bit perceptual hash of each image as an integer (order preserved)."""
    return [int(gb.perceptual_hash(Path(path)), 16) for path in paths]


def phash_neighbour_pairs(hash_ints: Sequence[int], max_hamming_distance: int = 2) -> list[tuple[int, int]]:
    """Return all unordered index pairs (i<j) whose perceptual-hash Hamming distance is <= threshold.

    Order-independent: the returned pair set depends only on the multiset of hashes, not on the
    order in which the paths were supplied (aside from index labelling, which sorting removes).
    """
    hashes = np.asarray(list(hash_ints), dtype=np.uint64)
    pairs: list[tuple[int, int]] = []
    for index in range(len(hashes) - 1):
        distances = _popcount64(hashes[index + 1:] ^ hashes[index])
        for offset in np.nonzero(distances <= int(max_hamming_distance))[0]:
            pairs.append((index, index + 1 + int(offset)))
    return pairs


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        self.parent[self.find(left)] = self.find(right)


def connected_components(n_nodes: int, pairs: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Connected components over ``n_nodes`` given undirected edges; singletons are components."""
    union = _UnionFind(n_nodes)
    for left, right in pairs:
        union.union(left, right)
    groups: dict[int, list[int]] = {}
    for node in range(n_nodes):
        groups.setdefault(union.find(node), []).append(node)
    return sorted((sorted(group) for group in groups.values()), key=lambda g: (-len(g), g[0]))


def perceptual_hash_cluster_diagnostics(paths: Sequence[str | Path], max_hamming_distance: int = 2,
                                        *, hash_ints: Sequence[int] | None = None) -> dict[str, Any]:
    """Order-independent perceptual-hash cluster diagnostics.

    ``phash_any_neighbour_rate`` is the fraction of images that have at least one *other* image at
    perceptual-hash distance <= ``max_hamming_distance``.  ``phash_component_excess_rate`` is
    ``sum(component_size - 1) / n_images`` over connected components with at least two images.
    Both are invariant to the order of ``paths``.
    """
    resolved = [Path(path) for path in paths]
    n_images = len(resolved)
    if hash_ints is None:
        hash_ints = perceptual_hash_ints(resolved)
    pairs = phash_neighbour_pairs(hash_ints, max_hamming_distance)
    nodes_with_neighbour = {node for pair in pairs for node in pair}
    components = connected_components(n_images, pairs)
    nontrivial = [group for group in components if len(group) >= 2]
    excess = sum(len(group) - 1 for group in nontrivial)
    largest = max((len(group) for group in components), default=0)
    return {
        "n_images": n_images,
        "n_phash_pairs": len(pairs),
        "n_images_with_any_phash_neighbour": len(nodes_with_neighbour),
        "phash_any_neighbour_rate": len(nodes_with_neighbour) / n_images if n_images else 0.0,
        "n_phash_connected_components": len(components),
        "n_nontrivial_phash_components": len(nontrivial),
        "largest_phash_component_size": largest,
        "phash_component_excess_count": excess,
        "phash_component_excess_rate": excess / n_images if n_images else 0.0,
        "max_hamming_distance": int(max_hamming_distance),
    }


def legacy_order_dependent_phash_rate(hash_ints: Sequence[int], max_hamming_distance: int = 2) -> float:
    """Deprecated: reproduce the original order-dependent rate (fraction with an *earlier* neighbour).

    Retained only for backward comparison; do not use for new decisions.
    """
    hashes = list(hash_ints)
    flagged = sum(
        any((int(current) ^ int(earlier)).bit_count() <= int(max_hamming_distance)
            for earlier in hashes[:index])
        for index, current in enumerate(hashes)
    )
    return flagged / len(hashes) if hashes else 0.0


# ---------------------------------------------------------------------------
# Confirmed duplicates (exact hash OR pHash<=2 AND SSIM>=0.98)
# ---------------------------------------------------------------------------

def confirmed_duplicate_analysis(paths: Sequence[str | Path], *, max_hamming_distance: int = 2,
                                 ssim_threshold: float = 0.98, hash_ints: Sequence[int] | None = None,
                                 exact_hashes: Sequence[str] | None = None,
                                 ssim_fn: Callable[[Path, Path], float] | None = None) -> dict[str, Any]:
    """Detect confirmed duplicates using the preregistered rule and return per-pair evidence.

    An unordered pair is a confirmed duplicate when the two files are byte-identical, or when their
    perceptual-hash distance is <= ``max_hamming_distance`` *and* their SSIM is >= ``ssim_threshold``.
    ``confirmed_duplicate_rate`` is the fraction of images that participate in at least one confirmed
    duplicate pair (order-independent).
    """
    resolved = [Path(path) for path in paths]
    n_images = len(resolved)
    if hash_ints is None:
        hash_ints = perceptual_hash_ints(resolved)
    if exact_hashes is None:
        exact_hashes = [gb.file_sha256(path) for path in resolved]
    if ssim_fn is None:
        ssim_fn = _default_ssim

    # Exact duplicate pairs from identical byte hashes.
    by_hash: dict[str, list[int]] = {}
    for index, digest in enumerate(exact_hashes):
        by_hash.setdefault(digest, []).append(index)
    exact_pairs = {(a, b) for indices in by_hash.values() if len(indices) > 1
                   for i, a in enumerate(indices) for b in indices[i + 1:]}

    phash_pairs = phash_neighbour_pairs(hash_ints, max_hamming_distance)
    records: list[dict[str, Any]] = []
    confirmed_nodes: set[int] = set()
    for left, right in sorted(set(phash_pairs) | exact_pairs):
        exact = (left, right) in exact_pairs
        distance = (int(hash_ints[left]) ^ int(hash_ints[right])).bit_count()
        ssim = 1.0 if exact else float(ssim_fn(resolved[left], resolved[right]))
        confirmed = bool(exact or (distance <= max_hamming_distance and ssim >= ssim_threshold))
        if confirmed:
            confirmed_nodes.update((left, right))
        records.append({"left_index": left, "right_index": right, "phash_distance": distance,
                        "ssim": ssim, "exact_hash_match": exact, "confirmed_duplicate": confirmed})
    exact_nodes = {node for pair in exact_pairs for node in pair}
    return {
        "n_images": n_images,
        "confirmed_duplicate_rate": len(confirmed_nodes) / n_images if n_images else 0.0,
        "exact_duplicate_rate": len(exact_nodes) / n_images if n_images else 0.0,
        "n_confirmed_pairs": sum(1 for row in records if row["confirmed_duplicate"]),
        "n_candidate_pairs": len(records),
        "pairs": records,
    }


def _default_ssim(left: Path, right: Path) -> float:
    from PIL import Image
    with Image.open(left) as image:
        left_array = np.asarray(image.convert("L").resize((512, 512)), dtype=np.uint8)
    with Image.open(right) as image:
        right_array = np.asarray(image.convert("L").resize((512, 512)), dtype=np.uint8)
    return gb._ssim(left_array, right_array)


# ---------------------------------------------------------------------------
# Real-vs-real RAD-DINO PRDC baselines (from cached embeddings only)
# ---------------------------------------------------------------------------

def repeated_prdc_baseline(reference: np.ndarray, candidate: np.ndarray, *, subset_reference: int,
                           subset_candidate: int, repetitions: int, seed: int, nearest_k: int = 5
                           ) -> list[dict[str, Any]]:
    """Repeated-subsampling PRDC between two embedding pools (``prdc(reference, candidate)``).

    Sampling is without replacement, deterministic under ``seed`` and index-order independent.
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for repetition in range(int(repetitions)):
        rng = np.random.default_rng(int(seed) + repetition)
        ref_idx = rng.choice(len(reference), int(subset_reference), replace=False)
        cand_idx = rng.choice(len(candidate), int(subset_candidate), replace=False)
        measured = gb.prdc(reference[ref_idx], candidate[cand_idx], nearest_k=int(nearest_k))
        rows.append({"repetition": repetition, "seed": int(seed) + repetition,
                     "reference_subset": int(subset_reference), "candidate_subset": int(subset_candidate),
                     "nearest_k": int(nearest_k), **measured})
    return rows


def split_half_prdc_baseline(embeddings: np.ndarray, *, repetitions: int, seed: int,
                             nearest_k: int = 5) -> list[dict[str, Any]]:
    """Repeated deterministic split-half PRDC on a single real pool (lower numerosity by design)."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    n = len(embeddings)
    half = n // 2
    rows: list[dict[str, Any]] = []
    for repetition in range(int(repetitions)):
        rng = np.random.default_rng(int(seed) + repetition)
        order = rng.permutation(n)
        left, right = order[:half], order[half:]
        measured = gb.prdc(embeddings[left], embeddings[right], nearest_k=int(nearest_k))
        rows.append({"repetition": repetition, "seed": int(seed) + repetition,
                     "reference_subset": len(left), "candidate_subset": len(right),
                     "nearest_k": int(nearest_k), **measured})
    return rows


def summarize_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    """Mean / median / stability interval for one PRDC metric across repetitions."""
    summary = gb.summarize([float(row[metric]) for row in rows if row.get(metric) is not None])
    return {f"{metric}_{field}": value for field, value in summary.items()}


# ---------------------------------------------------------------------------
# Descriptive, gate-independent generator ranking
# ---------------------------------------------------------------------------

DESCRIPTIVE_RANK_KEY_ORDER = (
    "raddino_kid", "raddino_coverage", "raddino_precision", "raddino_fid",
    "inception_kid", "raddino_kid_std", "generator_id",
)


def descriptive_generator_ranking(rows: Sequence[Mapping[str, Any]], family: str) -> list[dict[str, Any]]:
    """Rank official candidates within a family by the preregistered metric hierarchy, ignoring gates.

    Lower is better for KID / FID / kid_std; higher is better for coverage / precision.
    Every candidate receives a ``descriptive_family_rank`` regardless of eligibility.
    """
    candidates = [dict(row) for row in rows if str(row.get("family")) == family]

    def key(row: Mapping[str, Any]):
        return (
            gb._metric_value(row, "raddino_kid"),
            -gb._metric_value(row, "raddino_coverage", default=-math.inf),
            -gb._metric_value(row, "raddino_precision", default=-math.inf),
            gb._metric_value(row, "raddino_fid"),
            gb._metric_value(row, "inception_kid"),
            gb._metric_value(row, "raddino_kid_std"),
            str(row.get("generator_id", "")),
        )

    ordered = sorted(candidates, key=key)
    return [{**row, "descriptive_family_rank": index + 1} for index, row in enumerate(ordered)]


# ---------------------------------------------------------------------------
# Strict efficiency parser
# ---------------------------------------------------------------------------

INVALID_DURATION_STATUS = "unavailable_invalid_duration_semantics"


def efficiency_from_manifest_strict(root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Import runtime efficiency without ever inferring seconds-per-image from ambiguous durations.

    ``generation_seconds_per_image`` is derived from ``elapsed_seconds`` only when the manifest
    explicitly declares ``duration_semantics == 'wall_clock_full_generation'``,
    ``duration_unit == 'seconds'`` and ``measurement_complete == true``.  Otherwise it is ``None`` and
    ``efficiency_status`` is ``unavailable_invalid_duration_semantics``.  ``energy_kwh`` and
    ``peak_vram_mb`` are imported only when their own semantics are declared verified.
    """
    base = {"generation_seconds_per_image": None, "peak_vram_mb": None, "energy_kwh": None,
            "checkpoint_size_bytes": None, "efficiency_source": None,
            "efficiency_status": "unavailable", "generation_efficiency_status": "unavailable"}
    value = entry.get("efficiency_manifest")
    path = Path(root) / str(value) if value else None
    checkpoint = Path(root) / str(entry.get("checkpoint", ""))
    checkpoint_size = checkpoint.stat().st_size if checkpoint.is_file() else None
    if not path or not path.is_file():
        return {**base, "checkpoint_size_bytes": checkpoint_size,
                "efficiency_source": str(value) if value else None,
                "efficiency_status": "unavailable" if checkpoint_size is None else "checkpoint_size_only",
                "generation_efficiency_status": "unavailable"}
    try:
        payload = gb._read_manifest(path)
    except Exception:
        return {**base, "checkpoint_size_bytes": checkpoint_size,
                "efficiency_status": "unavailable_unreadable_manifest",
                "generation_efficiency_status": "unavailable_unreadable_manifest"}
    if not isinstance(payload, Mapping):
        return {**base, "checkpoint_size_bytes": checkpoint_size,
                "efficiency_status": "unavailable_unreadable_manifest",
                "generation_efficiency_status": "unavailable_unreadable_manifest"}

    count = payload.get("images_generated") or payload.get("n_generated")
    if not count and payload.get("n_per_class") and payload.get("generated_classes"):
        count = int(payload["n_per_class"]) * len(payload["generated_classes"])

    seconds_per_image = payload.get("seconds_per_image")
    generation_status = "available" if seconds_per_image is not None else "unavailable"
    duration_ok = (
        str(payload.get("duration_semantics")) == "wall_clock_full_generation"
        and str(payload.get("duration_unit")) == "seconds"
        and bool(payload.get("measurement_complete")) is True
    )
    if seconds_per_image is None and payload.get("elapsed_seconds") is not None and count:
        if duration_ok:
            seconds_per_image = float(payload["elapsed_seconds"]) / int(count)
            generation_status = "available"
        else:
            generation_status = INVALID_DURATION_STATUS

    energy_verified = bool(payload.get("energy_semantics_verified")) or bool(payload.get("energy_provenance_verified"))
    vram_verified = payload.get("peak_vram_mb") is not None and bool(payload.get("vram_semantics_verified"))
    values = {
        "generation_seconds_per_image": seconds_per_image,
        "peak_vram_mb": payload.get("peak_vram_mb") if vram_verified else None,
        "energy_kwh": payload.get("energy_kwh") if energy_verified else None,
        "checkpoint_size_bytes": checkpoint_size,
        "efficiency_source": str(value),
    }
    available = any(values[name] is not None for name in
                    ("generation_seconds_per_image", "peak_vram_mb", "energy_kwh", "checkpoint_size_bytes"))
    efficiency_status = generation_status if generation_status == INVALID_DURATION_STATUS else (
        "available" if available else "unavailable")
    return {**values, "efficiency_status": efficiency_status, "generation_efficiency_status": generation_status}
