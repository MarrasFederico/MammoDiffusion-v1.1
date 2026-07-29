# Test suite

The `tests/` suite is model-free. Fixtures use tiny temporary CSVs, JSON files, and images; tests do
not train, generate, run classifier inference, open the scientific image cohort, or require a GPU.

Run:

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -p 'test_*.py' -v
python -m pytest -q
```

`pytest.ini` restricts discovery to `tests/` and excludes the vendored Diffusers repository,
`data/`, `experiments/`, `results/`, Git metadata, and cache directories.

The main protected behaviors are:

- validation-frozen seed and ensemble thresholds on test;
- a hard error when test thresholds are missing;
- fixed thresholds in every bootstrap replica;
- invariant ROC-AUC, PR-AUC, Brier, and ECE;
- deterministic CSV-only report regeneration;
- final-evaluation opt-in and overwrite protection;
- generator benchmark review mode without data, experiments, GPU, or mutation;
- no test paths in active generator notebooks or selection utilities;
- train-only memorization reference and patient-level split isolation;
- exact 1,361-image selected pools and unique file lists;
- RAW/FILTERED representation policy and declared eligibility/ranking fields;
- operational generation resume checks and cache invalidation;
- GPU auto/index/UUID selection as an execution feature only;
- notebook syntax, required repository files, and absence of tracked model/bytecode artifacts.

Tests of a real benchmark or real cohort remain integration checks and must be run manually with
the required local assets. Unit tests do not silently skip into those paths.

## Source archive

Build the delivery ZIP from the existing tracked files plus non-ignored working-tree files. Keep
the root and nested `.gitignore` files, `assets/`, and the seven canonical lightweight generator
benchmark outputs. Exclude `data/`, `experiments/`, vendored/base diffusion repositories, Git
metadata, checkpoints and weights, caches and bytecode, heavy benchmark runtime artifacts, and
large interpretability figures. Validate the ZIP with `unzip -t`, extract it into a temporary
directory, and rerun both test commands above from the extracted repository before delivery.
