# Minimal execution guide

Commands that evaluate images or train models are shown for later use; none are part of repository refactoring.

```bash
# 1–3. Benchmark, inspect, approve
python scripts/run_generator_benchmark.py --execute --confirm
less results/generator_benchmark/generator_selection_report.md
python scripts/approve_generator_selection.py --proposal results/generator_benchmark/generator_selection_proposal.json --confirm

# 4. List exactly 24 manual jobs
python scripts/list_downstream_jobs.py

# 5. Optional smoke, no certificate chain
python scripts/smoke_downstream_classifier.py --architecture maxvit512 --gpu 0
python scripts/smoke_downstream_classifier.py --architecture mammofm --gpu 0

# 6. Run jobs individually (repeat only for the listed architecture/condition/seed)
python scripts/run_downstream_classifier.py --architecture maxvit512 --condition real_only --seed 17 --gpu 0

# 7–11. Ensemble, finalize validation, freeze, one-shot test, report
python scripts/build_downstream_ensembles.py
python scripts/finalize_downstream_validation.py
python scripts/lock_downstream_test.py --confirm
python scripts/run_downstream_locked_test.py --confirm
python scripts/finalize_publication_report.py
```

Use either `--gpu` or `CUDA_VISIBLE_DEVICES`, never both. Synthetic conditions fail until approval exists. Checkpoint/resume is automatic per experiment; a live process claim only prevents duplicate concurrent execution of the same experiment ID.
