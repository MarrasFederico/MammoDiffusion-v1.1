# Unified generator benchmark protocol

This protocol implements RQ1. `configs/generator_registry.json` is the only candidate registry and assigns `finetuned` or `from_scratch`. `configs/generator_benchmark_protocol.json` fixes sample count, seeds, feature spaces, metrics, bootstrap, eligibility gates, and ranking.

The primary class is cancer-positive. Every candidate uses 1,361 images sampled without replacement. RAW and FILTERED outputs are evaluated in separate homogeneous tables. The reference set is real validation positives; nearest-neighbour pools are real train positives, real validation positives, and same-candidate synthetics. Test access fails closed.

InceptionV3 and frozen RAD-DINO supply independent spaces. KID is primary; FID and PRDC are required. LPIPS, MS-SSIM, nearest synthetic distance, exact/perceptual duplicates, technical validity, filtering acceptance, and memorization records complete the benchmark. Bootstrap failures are counted explicitly.

The selection proposal chooses one eligible generator per family using the preregistered lexicographic metric rule, not a weighted score. Run:

```bash
python scripts/run_generator_benchmark.py --dry-run
python scripts/run_generator_benchmark.py --execute --confirm
python scripts/approve_generator_selection.py \
  --proposal results/generator_benchmark/generator_selection_proposal.json \
  --confirm
```

Only the last command creates `configs/approved_generators.json`.
