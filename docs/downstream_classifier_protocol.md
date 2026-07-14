# Compact downstream classifier protocol

This protocol addresses RQ2 and RQ3 with MaxViT-512 and Mammo-FM only. RAD-DINO is reserved for generator evaluation; ResNet-50 is a historical V1 baseline.

Four conditions—real-only, real plus traditional positive augmentation, real plus approved fine-tuned positive synthetics, and real plus approved from-scratch positive synthetics—are crossed with two architectures and seeds 17, 42, and 73. The exact primary inventory is 24 unique jobs in `configs/downstream_classifier_jobs.json`.

Within an architecture, conditions differ only in added training data. The fixed maximum optimizer-update policy prevents a larger dataset from receiving a larger training budget. The real validation set, loss, sampler/weighting policy, preprocessing, schedule, early stopping, checkpoint criterion, and evaluation remain fixed.

PR-AUC is primary. Three-seed mean probability is the primary model output. Validation thresholds are frozen before patient-level, one-shot test inference. Eight within-architecture comparisons use paired patient bootstrap and Holm correction.
