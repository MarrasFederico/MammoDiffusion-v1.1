# Generator status for MammoDiffusion v1.1

`configs/generator_registry.json` describes the eight generator experiments.
`configs/selected_generators.json` is the single authoritative selection record and resolves G02
and G07 directly. It contains no dependency on an auxiliary evidence file.

| ID | Classes | Role | Selection eligibility |
|---|---|---|---|
| G01 | negative, positive | 50-step sampling ablation of G02 | no |
| G02 | negative, positive | fine-tuned primary candidate; selected | yes |
| G03 | negative, positive | fine-tuned VAE primary candidate | yes |
| G04 | negative, positive | LoRA primary candidate | yes |
| G05 | positive | from-scratch descriptive baseline | no |
| G06 | negative, positive | larger generation-pool ablation of G05 | no |
| G07 | negative, positive | SD-VAE from-scratch candidate; selected | yes |
| G08 | negative, positive | v3 SD-VAE candidate and step-count ablation | yes |

The active benchmark is validation-only. It checks concrete RAW/FILTERED pools, image counts,
readability, uniqueness, technical validity, metric completeness, train memorization, and registry
role. G02 ranks first among eligible fine-tuned candidates; G07 ranks first among eligible
from-scratch candidates under the declared RAD-DINO KID-first hierarchy.

Eligibility is specific to this technical ranking protocol. It does not establish that a requested
positive contains a localized cancer finding, that a non-cancer image is lesion-free, or that an
image is radiologically or clinically valid. Perceptual-hash duplicate rate and RAD-DINO coverage
remain descriptive reference values; efficiency is reported separately and is not a ranking field.

Generator training-corpus lookup uses `data/processed/metadata/train.csv` directly and includes
traditional augmentation rows from `data/real_augmented/metadata.csv` when available. The list is
built in memory. Non-train paths, invalid or inconsistent labels, duplicates, missing files, and
patient overlap with validation are rejected; test metadata is not opened.

Canonical small outputs are in `results/2_diffusers/benchmark/`. Heavy embedding caches and
per-image diagnostics remain runtime-only.

In the current pipeline, generator benchmarking and selection use only training and validation
data. The test split is used for final classifier evaluation.

Hashes are used only for duplicate detection, cache validation, and resume safety.
