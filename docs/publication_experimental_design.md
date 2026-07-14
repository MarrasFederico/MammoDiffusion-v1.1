# Publication experimental design

## Research questions

- RQ1 compares eligible generators on validation-only fidelity, diversity, coverage, efficiency, duplication and train memorization.
- RQ2 compares real-only training, traditional augmentation, and positive synthetic augmentation.
- RQ3 tests whether the downstream effect is consistent across MaxViT-512 and Mammo-FM.

## Generator study

Each candidate provides a uniform synthetic pool target of 1,361 positive images. All available positive validation images form the real reference pool. A metric repetition uses the same number of real and synthetic features, sampled without replacement with registered deterministic seeds. The balanced size is `min(real_reference_count, synthetic_pool_count)`; real images are never duplicated to reach 1,361.

KID is primary (200 repeated balanced subsets by default). PRDC uses 100 repeated balanced subsets, `replace=False`, and requires `subset_size > nearest_neighbour_k`. FID is secondary and descriptive (one repetition by default) because it is unstable with few real positives. Embeddings are cached once per generator × representation × extractor.

Train memorization compares synthetic images with real training positives and may gate selection. Validation similarity is descriptive and cannot trigger a memorization gate. Synthetic-to-synthetic duplication is reported separately.

The 50-step Stable Diffusion configuration is a sampling ablation and is not automatically selection-eligible. The first LDM is a descriptive baseline while its lineage is unproven. Both remain visible in appropriate benchmark tables.

## Downstream design

The primary design is fixed:

- architectures: MaxViT-512, Mammo-FM;
- conditions: `real_only`, `real_augmented`, `real_plus_best_finetuned_positive`, `real_plus_best_fromscratch_positive`;
- seeds: 17, 42, 73;
- total: 2 × 4 × 3 = 24 experiments and 8 three-seed validation ensembles.

Within an architecture, all conditions share maximum optimizer updates, effective batch size, loss, class-weighting policy, online real augmentation, validation frequency, scheduler, early stopping and checkpoint criterion. The primary checkpoint metric is validation PR-AUC. Exact PR-AUC ties use lower validation loss and then the earlier epoch.

## Inference and statistics

The validation notebook reports patient-level PR-AUC, ROC-AUC, Brier score, ECE, sensitivity, specificity, balanced accuracy, bootstrap intervals, seed mean ± standard deviation, and mean-probability ensembles. The eight declared PR-AUC comparisons form one Holm family. Any additional comparison is exploratory.

Final evaluation uses a visible opt-in Boolean, readiness checklist and plain JSON protocol snapshot. It does not use a cryptographic lock, Git revision gate or technical one-shot enforcement. Scientific discipline requires that no model or threshold selection occur after final evaluation begins.
