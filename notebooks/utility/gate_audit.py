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

import json
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
# Content-aware perceptual-hash cache (compute once per image, reuse everywhere)
# ---------------------------------------------------------------------------

PHASH_IMPLEMENTATION_VERSION = "dct8x8-median-v1"  # matches generator_benchmark.perceptual_hash


class PerceptualHashCache:
    """Persistent, content-aware perceptual-hash + SHA-256 cache.

    Each entry is keyed by absolute path and validated against file size and the perceptual-hash
    implementation version; the SHA-256 is stored as part of the content identity.  The cache lets the
    size-matched audit compute a pool's hashes once and reuse them across all 200 repetitions instead
    of re-reading and re-hashing the same images.
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "phash_index.json"
        self.index: dict[str, dict[str, Any]] = {}
        if self.index_path.exists():
            try:
                self.index = json.loads(self.index_path.read_text())
            except Exception:
                self.index = {}
        self._dirty = False

    def _entry(self, path: str | Path) -> dict[str, Any]:
        resolved = Path(path)
        key = str(resolved.resolve())
        size = resolved.stat().st_size
        cached = self.index.get(key)
        if cached and cached.get("file_size") == size \
                and cached.get("phash_version") == PHASH_IMPLEMENTATION_VERSION:
            return cached
        entry = {"path": key, "file_size": size, "sha256": gb.file_sha256(resolved),
                 "phash_hex": gb.perceptual_hash(resolved), "phash_version": PHASH_IMPLEMENTATION_VERSION}
        self.index[key] = entry
        self._dirty = True
        return entry

    def phash_int(self, path: str | Path) -> int:
        return int(self._entry(path)["phash_hex"], 16)

    def sha256(self, path: str | Path) -> str:
        return self._entry(path)["sha256"]

    def hash_ints(self, paths: Sequence[str | Path]) -> list[int]:
        return [self.phash_int(path) for path in paths]

    def exact_hashes(self, paths: Sequence[str | Path]) -> list[str]:
        return [self.sha256(path) for path in paths]

    def save(self) -> None:
        if self._dirty:
            gb.atomic_json(self.index_path, self.index)
            self._dirty = False


# ---------------------------------------------------------------------------
# Size-matched perceptual-hash diagnostics (compare like-sized pools only)
# ---------------------------------------------------------------------------

def phash_pool_evidence(paths: Sequence[str | Path], *, cache: PerceptualHashCache | None = None,
                        max_hamming_distance: int = 2, ssim_threshold: float = 0.98,
                        ssim_fn: Callable[[Path, Path], float] | None = None) -> dict[str, Any]:
    """Precompute, once for a pool, everything needed to derive size-matched subset diagnostics.

    Returns the pool size and the global neighbour / confirmed / exact pair sets.  SSIM is evaluated
    only on the pHash-near pairs, once.  All downstream repetition metrics are obtained by restricting
    these pair sets to a sampled subset, so no image is read or hashed more than once.
    """
    resolved = [Path(path) for path in paths]
    if cache is not None:
        hash_ints = cache.hash_ints(resolved)
        exact = cache.exact_hashes(resolved)
    else:
        hash_ints = perceptual_hash_ints(resolved)
        exact = [gb.file_sha256(path) for path in resolved]
    if ssim_fn is None:
        ssim_fn = _default_ssim

    neighbour_pairs = set(phash_neighbour_pairs(hash_ints, max_hamming_distance))
    by_hash: dict[str, list[int]] = {}
    for index, digest in enumerate(exact):
        by_hash.setdefault(digest, []).append(index)
    exact_pairs = {(a, b) for indices in by_hash.values() if len(indices) > 1
                   for i, a in enumerate(indices) for b in indices[i + 1:]}
    confirmed_pairs = set(exact_pairs)
    for left, right in neighbour_pairs:
        if (left, right) in exact_pairs:
            continue
        if float(ssim_fn(resolved[left], resolved[right])) >= ssim_threshold:
            confirmed_pairs.add((left, right))
    return {"n": len(resolved), "hash_ints": hash_ints, "neighbour_pairs": neighbour_pairs,
            "confirmed_pairs": confirmed_pairs, "exact_pairs": exact_pairs}


def _subset_phash_metrics(evidence: Mapping[str, Any], subset: Sequence[int]) -> dict[str, Any]:
    members = set(int(index) for index in subset)
    position = {index: order for order, index in enumerate(subset)}
    n = len(subset)
    neighbour = [(a, b) for a, b in evidence["neighbour_pairs"] if a in members and b in members]
    confirmed = [(a, b) for a, b in evidence["confirmed_pairs"] if a in members and b in members]
    exact = [(a, b) for a, b in evidence["exact_pairs"] if a in members and b in members]
    neighbour_nodes = {node for pair in neighbour for node in pair}
    confirmed_nodes = {node for pair in confirmed for node in pair}
    exact_nodes = {node for pair in exact for node in pair}
    components = connected_components(n, [(position[a], position[b]) for a, b in neighbour])
    excess = sum(len(group) - 1 for group in components if len(group) >= 2)
    largest = max((len(group) for group in components), default=0)
    return {"sample_size": n,
            "phash_any_neighbour_rate": len(neighbour_nodes) / n if n else 0.0,
            "phash_component_excess_rate": excess / n if n else 0.0,
            "confirmed_duplicate_rate": len(confirmed_nodes) / n if n else 0.0,
            "exact_duplicate_rate": len(exact_nodes) / n if n else 0.0,
            "largest_component_size": largest}


def repeated_size_matched_phash(evidence: Mapping[str, Any], *, subset_size: int, repetitions: int,
                                seed: int) -> list[dict[str, Any]]:
    """200-style repeated size-matched perceptual-hash diagnostics via without-replacement subsampling.

    Deterministic under ``seed`` and independent of the pool's storage order.
    """
    pool = int(evidence["n"])
    if subset_size > pool:
        raise ValueError(f"subset_size {subset_size} exceeds pool size {pool}")
    rows = []
    for repetition in range(int(repetitions)):
        rng = np.random.default_rng(int(seed) + repetition)
        subset = [int(index) for index in rng.choice(pool, int(subset_size), replace=False)]
        rows.append({"repetition": repetition, "seed": int(seed) + repetition,
                     **_subset_phash_metrics(evidence, subset)})
    return rows


SIZE_MATCHED_METRICS = ("phash_any_neighbour_rate", "phash_component_excess_rate",
                        "confirmed_duplicate_rate", "exact_duplicate_rate", "largest_component_size")


def summarize_size_matched(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    values = gb.summarize([float(row[metric]) for row in rows if row.get(metric) is not None])
    return {"metric_name": metric, "repetitions": len(rows),
            "metric_mean": values["mean"], "metric_median": values["median"],
            "metric_std": values["standard_deviation"],
            "stability_interval_low": values["percentile_2_5"],
            "stability_interval_high": values["percentile_97_5"]}


# ---------------------------------------------------------------------------
# Amended (Option B) safety-gate eligibility
# ---------------------------------------------------------------------------

def amended_safety_gate_failures(row: Mapping[str, Any], gates: Mapping[str, Any], *,
                                 confirmed_duplicate_rate: float | None = None) -> list[str]:
    """Option B eligibility: block only on safety gates, never on pHash-only or coverage.

    ``perceptual_duplicate_rate`` (pHash-only) and ``rad_dino_coverage`` are deliberately *not* gates
    here; they are descriptive / ranking metrics under the approved amendment.  ``confirmed_duplicate_rate``
    (exact | pHash<=2 AND SSIM>=0.98) is the duplicate safety gate and must be supplied from the audit
    when it is not already on the row.
    """
    confirmed = confirmed_duplicate_rate
    if confirmed is None:
        confirmed = gb._metric_value(row, "confirmed_duplicate_rate", default=math.inf)
    checks = [
        (int(gb._metric_value(row, "valid_positive_images", default=0)) >= int(gates["minimum_valid_positive_images"]),
         "valid_positive_images"),
        (gb._metric_value(row, "synthetic_exact_duplicate_rate", default=math.inf) <= gates["maximum_exact_duplicate_rate"],
         "exact_duplicate_rate"),
        (float(confirmed) <= gates["maximum_confirmed_duplicate_rate"], "confirmed_duplicate_rate"),
        (gb._metric_value(row, "train_memorization_rate", default=math.inf) <= gates["maximum_train_memorization_rate"],
         "train_memorization_rate"),
        (gb._metric_value(row, "corrupted_rate", "n_corrupt", default=0.0) <= gates["maximum_corrupted_file_rate"],
         "corrupt_files"),
        (gb._as_bool(row.get("filter_manifest_valid")), "filter_manifest"),
        (gb._as_bool(row.get("filter_provenance_complete")), "filter_mapping"),
        (gb._as_bool(row.get("metrics_complete")), "metrics_complete"),
        (not gb._as_bool(row.get("test_access")), "test_access"),
    ]
    if gates.get("require_lineage", True):
        checks.append((gb._as_bool(row.get("lineage_complete")), "lineage"))
    if gates.get("require_provenance", True):
        checks.append((gb._as_bool(row.get("provenance_manifest_valid")), "provenance"))
    checks.append((gb._as_bool(row.get("training_corpus_manifest_valid")), "training_corpus"))
    if not gb._as_bool(row.get("eligible_for_selection", row.get("eligible_for_downstream_selection", False))):
        checks.append((False, "registry_role"))
    return [name for passed, name in checks if not passed]


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

INVALID_DURATION_STATUS = gb.INVALID_DURATION_STATUS  # single source of truth in generator_benchmark


def efficiency_from_manifest_strict(root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Deprecated alias: the strict semantics now live in the canonical ``efficiency_from_manifest``.

    Kept as a thin delegator so there is a single implementation that cannot drift.
    """
    return gb.efficiency_from_manifest(root, entry)


# ---------------------------------------------------------------------------
# Amendment-aware selection (Option B) for notebook 06
# ---------------------------------------------------------------------------

def load_active_amendment(root: Path) -> dict[str, Any] | None:
    """Return the active protocol amendment payload, or ``None`` when the protocol declares none."""
    protocol = gb.load_protocol(root)
    reference = protocol.get("active_amendment")
    if not reference:
        return None
    import json
    return json.loads((Path(root) / str(reference)).read_text())


def confirmed_duplicate_rates(root: Path) -> dict[str, float]:
    """Map full generator_id -> confirmed_duplicate_rate from the pHash audit CSV (safety input)."""
    import csv
    path = Path(root) / gb.BENCHMARK_ROOT / "gate_audit/perceptual_hash_diagnostics.csv"
    if not path.is_file():
        return {}
    rates: dict[str, float] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row.get("full_generator_id") or row.get("generator_id")
            rates[str(key)] = float(row["confirmed_duplicate_rate"])
    return rates


def amended_family_ranking(rows: Sequence[Mapping[str, Any]], family: str, amendment: Mapping[str, Any],
                           confirmed_by_gen: Mapping[str, float]) -> list[dict[str, Any]]:
    """Rank a family by the preregistered hierarchy, gating on Option B safety gates only."""
    gates = amendment["new_blocking_gates"]
    candidates = []
    for source in rows:
        if str(source.get("family", source.get("scientific_family"))) != family:
            continue
        row = dict(source)
        confirmed = confirmed_by_gen.get(str(row.get("generator_id")))
        failures = amended_safety_gate_failures(row, gates, confirmed_duplicate_rate=confirmed)
        row["amended_exclusion_reasons"] = sorted(set(failures))
        row["eligible"] = not row["amended_exclusion_reasons"]
        row["confirmed_duplicate_rate"] = confirmed
        candidates.append(row)

    def key(row: Mapping[str, Any]):
        return (not row["eligible"],
                gb._metric_value(row, "raddino_kid", "raddino_kid_mean"),
                -gb._metric_value(row, "raddino_coverage", "coverage", default=-math.inf),
                -gb._metric_value(row, "raddino_precision", "precision", default=-math.inf),
                gb._metric_value(row, "raddino_fid", "fid_descriptive"),
                gb._metric_value(row, "inception_kid", "inception_kid_mean"),
                gb._metric_value(row, "raddino_kid_std", "kid_standard_deviation"),
                str(row.get("generator_id", "")))
    ordered = sorted(candidates, key=key)
    return [{**row, "family_rank": index + 1 if row["eligible"] else None}
            for index, row in enumerate(ordered)]


def validate_amended_selection(root: Path, finetuned: str, from_scratch: str, registry: Mapping[str, Any],
                               benchmark_rows: Sequence[Mapping[str, Any]], amendment: Mapping[str, Any],
                               confirmed_by_gen: Mapping[str, float],
                               synthetic_pool_target: int = 1361) -> dict[str, str]:
    """Validate a selection against family, provenance and the Option B safety gates (never coverage/pHash)."""
    by_id = {entry["id"]: entry for entry in registry["generators"]}
    results = {str(row["generator_id"]): row for row in benchmark_rows if row.get("condition") == "FILTERED"}
    gates = amendment["new_blocking_gates"]
    for family, generator_id in (("finetuned", finetuned), ("from_scratch", from_scratch)):
        entry = by_id.get(generator_id)
        if not entry:
            raise ValueError(f"unknown generator ID: {generator_id}")
        if entry.get("scientific_family") != family:
            raise ValueError(f"{generator_id} has wrong family")
        if not entry.get("eligible_for_downstream_selection", False):
            raise ValueError(f"{generator_id} is not selection-eligible")
        row = results.get(generator_id)
        if not row:
            raise ValueError(f"benchmark incomplete for {generator_id}")
        if not gb._as_bool(row.get("metrics_complete")):
            raise ValueError(f"benchmark metrics incomplete for {generator_id}")
        if int(gb._metric_value(row, "valid_positive_images", default=0)) < int(synthetic_pool_target):
            raise ValueError(f"insufficient images for {generator_id}")
        if gb._as_bool(row.get("test_access")):
            raise ValueError("generator selection must not use test data")
        failures = amended_safety_gate_failures(row, gates,
                                                confirmed_duplicate_rate=confirmed_by_gen.get(generator_id))
        if failures:
            raise ValueError(f"amended safety gates failed for {generator_id}: {failures}")
    return {"finetuned": finetuned, "from_scratch": from_scratch}


SELECTION_SCHEMA_VERSION = 2
CANONICAL_SUMMARY_RELATIVE = str(Path(gb.BENCHMARK_ROOT) / "generator_summary_corrected.csv")
AMENDED_GATE_RESULTS_RELATIVE = str(Path(gb.BENCHMARK_ROOT) / "gate_audit/amended_gate_results.csv")


def _amended_rank_lookup(root: Path) -> dict[str, dict[str, Any]]:
    import csv
    path = Path(root) / AMENDED_GATE_RESULTS_RELATIVE
    if not path.is_file():
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            lookup[str(row["full_generator_id"])] = row
    return lookup


def selection_identity_entry(root: Path, generator_id: str, family: str,
                             rank_row: Mapping[str, Any] | None, primary_metric: str) -> dict[str, Any]:
    """Derive the content identity of one selected generator from its verified provenance manifest."""
    import csv
    registry = gb.load_registry(root)
    entry = next(item for item in registry["generators"] if item["id"] == generator_id)
    provenance = json.loads((Path(root) / entry["provenance_manifest"]).read_text())
    filtered_manifest = str(provenance["filtered_sample_manifest"])
    manifest_path = Path(root) / filtered_manifest
    with manifest_path.open(newline="") as stream:
        filtered_count = sum(1 for _ in csv.DictReader(stream))
    metric_value = None
    rank = None
    if rank_row is not None:
        rank = int(rank_row["descriptive_family_rank"]) if rank_row.get("descriptive_family_rank") not in (None, "") else None
        metric_value = float(rank_row["selection_metric_value"]) if rank_row.get("selection_metric_value") not in (None, "") else None
    return {
        "generator_id": generator_id,
        "family": family,
        "descriptive_family_rank": rank,
        "primary_metric": primary_metric,
        "primary_metric_value": metric_value,
        "model_identity_sha256": provenance["model_identity_sha256"],
        "generation_identity_sha256": provenance["generation_identity_sha256"],
        "filtered_manifest_path": filtered_manifest,
        "filtered_manifest_sha256": provenance["manifest_sha256"]["filtered_samples"],
        "filtered_image_count": filtered_count,
    }


def _relative_sha(root: Path, relative: str) -> str | None:
    path = Path(root) / relative
    return gb.file_sha256(path) if path.is_file() else None


def save_amended_selection(root: Path, finetuned: str, from_scratch: str,
                           benchmark_rows: Sequence[Mapping[str, Any]], *, notes: str = "") -> Path:
    """Validate and write a content-aware configs/selected_generators.json (schema_version 2).

    Backward-compatible ``finetuned`` / ``from_scratch`` fields are retained.  The selection is bound to
    the benchmark summary, the amended gate results, the active amendment, and each generator's model /
    generation identity and FILTERED manifest by SHA-256 so later silent edits can be detected.
    """
    registry = gb.load_registry(root)
    protocol = gb.load_protocol(root)
    amendment = load_active_amendment(root)
    if amendment is None:
        raise ValueError("no active amendment declared in the protocol")
    confirmed = confirmed_duplicate_rates(root)
    selected = validate_amended_selection(root, finetuned, from_scratch, registry, benchmark_rows,
                                          amendment, confirmed, protocol["synthetic_pool_target"])
    ranks = _amended_rank_lookup(root)
    primary_metric = protocol["selection"]["primary_metric"]
    payload = {
        # Backward-compatible minimal selection.
        "finetuned": selected["finetuned"], "from_scratch": selected["from_scratch"],
        "schema_version": SELECTION_SCHEMA_VERSION,
        "primary_metric": primary_metric,
        "benchmark_HEAD": amendment["benchmark_HEAD"],
        "benchmark_run_id": amendment["benchmark_run_id"],
        "benchmark_summary_path": CANONICAL_SUMMARY_RELATIVE,
        "benchmark_summary_sha256": _relative_sha(root, CANONICAL_SUMMARY_RELATIVE),
        "amended_gate_results_path": AMENDED_GATE_RESULTS_RELATIVE,
        "amended_gate_results_sha256": _relative_sha(root, AMENDED_GATE_RESULTS_RELATIVE),
        "active_amendment": str(protocol.get("active_amendment")),
        "active_amendment_sha256": _relative_sha(root, str(protocol.get("active_amendment"))),
        "original_protocol_result": amendment["original_outcome"],
        "post_benchmark_amendment": True,
        "test_access": False,
        "selection_notes": notes,
        "selection_identity": {
            "finetuned": selection_identity_entry(root, selected["finetuned"], "finetuned",
                                                  ranks.get(selected["finetuned"]), primary_metric),
            "from_scratch": selection_identity_entry(root, selected["from_scratch"], "from_scratch",
                                                    ranks.get(selected["from_scratch"]), primary_metric)},
    }
    return gb.atomic_json(Path(root) / "configs/selected_generators.json", payload)
