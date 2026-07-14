<div align="center">

<img src="assets/logo_MammoDiffusion.png" alt="MammoDiffusion Logo" width="180"/>

# MammoDiffusion v2

**Publication-oriented benchmark of synthetic mammogram generators with a compact downstream check.**

[Versione italiana](README.md)

</div>

## Research objective

MammoDiffusion v2 compares fine-tuned and from-scratch generators on RSNA Breast Cancer Detection. Scientific depth is centered on generated-image quality; downstream classifiers provide a small robustness check.

- **RQ1 — Generator quality:** which fine-tuned generator and which from-scratch generator best balance fidelity, diversity, coverage, and lack of memorization?
- **RQ2 — Downstream utility:** do positive synthetic mammograms improve classification over real-only training and traditional augmentation?
- **RQ3 — Classifier robustness:** is the effect consistent across a modern general-purpose model and a mammography-specific foundation model?

The test set is never used for generator, checkpoint, or threshold selection.

## Compact design

The unified benchmark uses 1,361 positive images per valid registry candidate, separate RAW and FILTERED analyses, deterministic bootstrap, InceptionV3, and frozen RAD-DINO. RAD-DINO KID is primary; FID, PRDC, LPIPS, MS-SSIM, duplicates, technical validity, and train/validation nearest-neighbour memorization checks complete RQ1. One winner per generator family is proposed and must be approved explicitly.

Downstream validation is exactly:

```text
2 architectures: MaxViT-512 and Mammo-FM
4 data conditions
3 seeds: 17, 42, 73
= 24 primary jobs and 8 ensembles
```

MaxViT-512 is the general-purpose convolution/transformer backbone. Mammo-FM supplies mammography-specific domain pretraining. ResNet-50 is a historical V1 baseline; RAD-DINO is not a downstream classifier.

## Canonical flow

```text
preprocessing → traditional augmentation → generator development
→ generator_benchmark → generator_selection → explicit approval
→ downstream_validation → three-seed ensembles → protocol freeze
→ one-shot locked_test → patient-level statistics → final_report
```

Jobs are manually launched one at a time. There is no cluster scheduler and no mandatory GPU certificate or canary chain. Checkpoint/resume, dataset validation, prediction alignment, ensembles, statistics, and the strict locked test are retained.

See [`docs/publication_experimental_design.md`](docs/publication_experimental_design.md) for the full design and [`docs/execution_guide.md`](docs/execution_guide.md) for commands. `configs/approved_generators.json` is created only after a real benchmark and explicit approval; no winner is hardcoded.

The primary downstream endpoint is patient-level PR-AUC from the mean-probability three-seed ensemble. Validation thresholds are frozen before test inference. Mammo-FM weights and derivatives must not be committed or redistributed; see [`docs/mammo_fm_license_note.md`](docs/mammo_fm_license_note.md).

Historical V1 results remain separate in [`docs/legacy_v1_classifier_results.md`](docs/legacy_v1_classifier_results.md). The complete pre-simplification implementation is recoverable from the Git tag `classifier-matrix-v2-full`.
