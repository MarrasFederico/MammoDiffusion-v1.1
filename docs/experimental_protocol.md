# Canonical experimental protocol

The canonical design is [`publication_experimental_design.md`](publication_experimental_design.md). It addresses RQ1 (generator quality), RQ2 (downstream utility), and RQ3 (robustness across MaxViT-512 and Mammo-FM).

The executable sequence is:

1. dataset preprocessing;
2. traditional positive augmentation;
3. generator training and generation;
4. unified generator benchmark;
5. one proposed winner per generator family;
6. explicit generator approval;
7. 24 downstream validation jobs;
8. eight three-seed ensembles;
9. validation finalization and protocol freeze;
10. one-shot locked test;
11. patient-level statistics;
12. publication report.

Generator selection is validation-only. The test is not a generator reference, nearest-neighbour pool, checkpoint-selection set, or threshold-selection set.
