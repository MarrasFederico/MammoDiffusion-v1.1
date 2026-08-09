# Test suite

The `tests/` suite is model-free. Fixtures use tiny temporary CSVs, JSON files, and images; tests do
not train, generate, run classifier inference, open the scientific image cohort, or require a GPU.

Run:

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -p 'test_*.py' -v
python -m pytest -q
```

`requirements-dev.txt` constrains the runner plus the light scientific packages the utilities import
(NumPy, pandas, SciPy, Matplotlib, Pillow, IPython). It deliberately omits TensorFlow, PyTorch,
Diffusers, Gradio, and scikit-learn, which the suite never reaches; `requirements.txt` covers those
for real runs.

`pytest.ini` restricts discovery to `tests/` and excludes the vendored Diffusers repository,
`data/`, `experiments/`, `results/`, Git metadata, and cache directories.

The main protected behaviors are:

- validation-frozen seed and ensemble thresholds on test;
- a hard error when test thresholds are missing;
- fixed thresholds in every bootstrap replica;
- average-precision invariance to input order when prediction scores are tied;
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

## Source-release check

The Git/source release and the full scientific archive have different boundaries. A source archive
should be built from the exact `v1.1.0` tracked tree, including versioned attribution figures and
canonical lightweight generator outputs. It excludes ignored `data/`, `experiments/`, checkpoints,
weights, runtime caches, and separately archived shared-model assets. Validate a ZIP with
`unzip -t`, extract it into a temporary directory, and rerun the lightweight tests there.

Full generator/training reproduction additionally requires the assets listed in
[ARCHIVE_AND_RELEASE.md](ARCHIVE_AND_RELEASE.md) and [SHARED_ASSETS.md](SHARED_ASSETS.md). The
requirements files use version bounds rather than an exact scientific lock; the current
release-verification environment is recorded in the archive document and must not be mistaken for
the exact environment of every historical training run.
