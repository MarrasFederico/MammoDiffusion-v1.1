# Generator benchmark protocol — amendment v1 (Option B), approved post-benchmark

- **Status:** approved (human), post-benchmark amendment.
- **Approval date:** 2026-07-15.
- **Benchmark run:** `generator_benchmark_20260714T221055Z_cd05886c` (HEAD `cd05886c`).
- **Machine-readable record:** [`configs/generator_benchmark_protocol_amendment_v1.json`](../configs/generator_benchmark_protocol_amendment_v1.json).

This amendment is a **transparent post-benchmark change**. It is not presented as if it had been
preregistered. The original protocol and the original zero-eligible outcome are preserved verbatim and
are not rewritten.

## 1. Original gates

The original active gates (frozen in
`results/publication_v2/generator_benchmark/gate_audit/original_protocol.json`, SHA-256
`c5f189d8…af3084`) included, among the safety checks, two thresholds that this amendment revisits:

- `maximum_perceptual_duplicate_rate = 0.02` — a **pHash-only**, order-dependent duplicate gate.
- `minimum_rad_dino_coverage = 0.5` — a **binary coverage** gate from a single 73-image balanced point.

## 2. Original outcome — zero eligible

Five official FILTERED candidates were measured (G02, G03, G04, G07, G08). **Zero** were eligible:
every candidate failed both the pHash-only gate (rates 0.03–0.11) and the coverage gate (best G07 ≈
0.42 < 0.5). This outcome is unchanged and remains the record of what the original protocol produced.

## 3. Why pHash-only does not measure confirmed duplication

The original gate flags an image when any other image is within perceptual-hash distance 2. That is:

1. **order-dependent** (it counts only *earlier* neighbours), and
2. **pHash-only** — a 64-bit DCT hash collision does not establish that two images are duplicates.

Under the preregistered confirmed-duplicate rule — exact byte match, or (pHash ≤ 2 **and** SSIM ≥
0.98) — the `confirmed_duplicate_rate` is **0 for all five candidates**, and `exact_duplicate_rate`
and `train_memorization_rate` are also 0. The pHash-near pairs are **pHash-near pairs not confirmed as
duplicates**; they remain a descriptive signal of structural similarity and possible reduced diversity,
not confirmed duplication. Perceptual-hash any-neighbour and component-excess rates additionally depend
on pool size and are only interpretable **size-matched** (see the size-matched audit,
`gate_audit/phash_size_matched_summary.csv`), not by comparing a 1361-image synthetic pool to a
340-image or 73-image real pool.

## 4. Why coverage point is no longer a binary gate

Balanced-point coverage at 73 images is unstable (wide 200-repetition stability intervals) and the
0.5 value was not derived from data. Real-vs-real coverage baselines (real train positives vs
validation ≈ 0.96; validation split-half ≈ 0.98) are an **upper/reference benchmark, not a directly
transferable synthetic eligibility threshold**. Coverage is therefore retained as a
**descriptive/ranking metric**, not a hard binary gate. **No new coverage threshold is introduced.**

## 5. Approved policy — Option B

**Blocking safety gates** (a candidate is ineligible if any fails):

| gate | threshold |
|---|---|
| minimum valid positive images | 1361 |
| maximum exact duplicate rate | 0.01 |
| maximum confirmed duplicate rate | 0.02 |
| maximum train memorization rate | 0.01 |
| maximum corrupted file rate | 0.0 |
| valid filter manifest | required |
| complete RAW/FILTERED mapping | required |
| complete metrics | required |
| lineage | required |
| provenance | required |
| test access | forbidden |

**Descriptive / ranking metrics** (never binary gates): pHash-only similarity,
`phash_any_neighbour_rate`, `phash_component_excess_rate`, RAD-DINO coverage / precision / recall /
density, RAD-DINO FID, Inception KID, diversity, validation similarity.

**Selection hierarchy** (unchanged; KID stays primary):
`raddino_kid` → `raddino_coverage` → `raddino_precision` → `raddino_fid` → `inception_kid` →
`raddino_kid_std` → `generator_id`.

Under Option B all five official candidates pass the safety gates
(`amended_safety_gate_eligible = true`), and the descriptive ranking is fine-tuned G02 → G03 → G04 and
from-scratch G07 → G08.

## 6. Amendment is post-benchmark; test was not accessed

This decision was taken **after** the benchmark had already produced its results. The internal
historical test split was **not** opened at any point in the audit or the amendment. The active
protocol records `protocol_version` and `active_amendment`; the original values are retained in the
historical record and in the frozen `original_protocol.json`.

## 7. Consequence for downstream interpretation

Because eligibility was relaxed from the original binary coverage/pHash gates to the Option B safety
gates through a transparent post-benchmark amendment, any downstream result obtained with the selected
generators must be **interpreted with this limitation stated**: the generators were not selected under
the original preregistered eligibility thresholds, but under the approved amendment.
