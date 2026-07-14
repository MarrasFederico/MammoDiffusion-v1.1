# Publication-oriented experimental design

## Study objective

MammoDiffusion v2 is organized around three preregistered questions:

- **RQ1 — Generator quality:** which fine-tuned generator and which from-scratch generator provide the best balance of fidelity, diversity, coverage, and absence of memorization?
- **RQ2 — Downstream utility:** does adding cancer-positive synthetic mammograms improve classification over real-only training and traditional augmentation?
- **RQ3 — Classifier robustness:** is the synthetic-data effect consistent between a modern general-purpose classifier and a mammography-specific foundation model?

The scientific depth is concentrated in the generator benchmark. Downstream classification is a compact robustness check with 24 primary jobs.

## Data

The study uses the RSNA Breast Cancer Detection data, restricted to MLO views and transformed into canonical 512×512 grayscale images. Splits are patient-disjoint. Training metadata supplies real examples; validation metadata is the only reference for generator selection, checkpoint selection, thresholds, and downstream comparisons. The test manifest, images, labels, embeddings, and nearest neighbours are unavailable until the final lock.

The exact class counts are recorded in signed runtime manifests rather than copied into this design document. This avoids stale numbers after a legitimate preprocessing rerun while preserving auditable patient and image identities.

## Generative models

`configs/generator_registry.json` assigns every candidate to `finetuned` or `from_scratch`, plus a model subtype. Runtime inclusion requires a verifiable checkpoint, lineage, provenance, canonical positive output, at least 1,361 non-duplicated candidates, and no scientific blocker. The audit must not infer missing lineage or duplicate images to meet sample size.

Cancer-positive images form the primary benchmark because they are consistently available and directly support the augmentation question. Negative images may be reported separately without affecting the primary ranking. RAW and FILTERED outputs are evaluated and reported separately; no RAW-to-FILTERED cross-generator comparison is valid.

## Generative benchmark

Every executable candidate contributes the same deterministic sample of 1,361 positive images. Distribution metrics use real validation positives. Memorization searches use real train positives, real validation positives, and same-candidate synthetic images—never test data.

Two independent feature spaces are required:

- InceptionV3 for standard FID/KID continuity and comparison with general generative literature;
- frozen RAD-DINO as an independent medical-image encoder. RAD-DINO is radiology-specific rather than mammography-specific and is not a downstream classifier.

KID is primary. FID, precision, recall, density, and coverage are also bootstrapped. Diversity includes synthetic-to-synthetic nearest distance, LPIPS with grayscale images coherently replicated to three channels and mapped to the required range, MS-SSIM, exact duplicates, and perceptual-hash duplicates. Technical validity includes corruption, near-black output, dynamic range, dimensions, filename/content duplication, and filtering acceptance.

For each synthetic image, deterministic nearest train and validation neighbours record embedding distance, SSIM, perceptual-hash distance, and both identifiers. Visual panels are selected by the registered distance rule, not appearance. Training time, generation time, peak VRAM, energy, and checkpoint size are descriptive efficiency measures when available.

The default bootstrap uses 200 deterministic repeated samples and reports mean, median, standard deviation, 2.5th/97.5th percentiles, and valid/failed repetitions. Eligibility gates are fixed in `configs/generator_benchmark_protocol.json` before execution.

One eligible winner is selected per family without a weighted composite score: RAD-DINO FILTERED KID; coverage; precision; FID; Inception FILTERED KID; bootstrap stability; then generator ID as a deterministic technical fallback. Substantial interval overlap must be described as statistical or practical similarity. Automatic output is only a proposal; explicit signed approval is required.

## Downstream classification

MaxViT-512 represents a modern general-purpose convolution/transformer backbone with local and global modeling and no mammography-specific pretraining. Mammo-FM represents a mammography-specific foundation model with strong domain representations.

The four primary conditions are real-only, real plus traditional positive augmentation, real plus 1,361 approved fine-tuned positive synthetics, and real plus 1,361 approved from-scratch positive synthetics. Seeds are 17, 42, and 73: 2 architectures × 4 conditions × 3 seeds = 24 jobs.

Within each architecture, preprocessing, optimizer family, schedule, fixed maximum optimizer-update budget, early stopping, validation set, loss, real-image augmentation, sampler/class weighting, checkpoint criterion, precision mode, and metrics are invariant. Architecture-specific fine-tuning recipes may differ. A single manual runner owns configuration resolution, dataset validation, one-job locking, checkpoint/resume, training, validation predictions, metrics, and terminal state.

Interpretability cases follow preregistered TP, TN, FP, FN, and largest fine-tuned/from-scratch disagreement rules. MaxViT uses valid spatial gradient attribution; Mammo-FM uses architecture-compatible gradient or feature attribution.

## Statistics

The primary downstream endpoint is PR-AUC. Secondary endpoints are ROC-AUC, sensitivity, specificity, sensitivity at fixed specificity, F1, balanced accuracy, Brier score, ECE, and confusion matrices. The primary result is the mean-probability ensemble of seeds 17, 42, and 73; individual-seed mean ± standard deviation describes stability.

Confidence intervals and paired comparisons use patient-level bootstrap. For each architecture, four comparisons are preregistered: real-only versus traditional augmentation, fine-tuned synthetics, and from-scratch synthetics; plus fine-tuned versus from-scratch synthetics. Holm correction covers the eight primary comparisons. Cross-architecture comparisons are mainly descriptive.

Validation chooses and freezes decision thresholds. The locked test requires 24 completed jobs, eight ensembles, finalized validation, verified checkpoints, approved-generator signature, exact protocol, test-manifest hash, code revision, and artifact hashes. Test inference is one-shot, patient-level, and cannot trigger model selection or threshold tuning.

## Reproducibility

Deterministic sampling, evaluation and training seeds are stored with manifests. Protocol, registry, approval, dataset, checkpoint, prediction, ensemble, validation-finalization, and test-lock artifacts are content-signed. Runtime metadata records the code revision and environment. Checkpoints are atomically saved and resumable but are excluded from Git.

## Limitations

- one dataset and no external validation;
- a small positive reference sample;
- filtering can improve fidelity while reducing diversity and acceptance;
- RAD-DINO has a radiology-to-mammography domain mismatch;
- Mammo-FM weights and derivatives have restrictive academic licensing;
- residual acquisition/site domain shift is possible;
- optional ablations and some interpretability/efficiency analyses are exploratory.
