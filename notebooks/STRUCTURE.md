# Notebook structure

- `1_preprocessing/`: split-safe preprocessing and traditional augmentation.
- `2_diffusers/`: generator training, generation and filtering notebooks.
- `3_generator_benchmark/01_Unified_Generator_Benchmark.ipynb`: validation-only benchmark.
- `3_generator_benchmark/02_Generator_Selection.ipynb`: transparent manual family selection.
- `04_classifiers/01_MaxViT512.ipynb`: all four MaxViT conditions per execution, automatically
  cycling seeds 17/42/73 with independent resume state for 12 jobs; the complete model setup and PyTorch
  training/validation loop remains visible in notebook cells.
- `04_classifiers/02_MammoFM.ipynb`: all four Mammo-FM conditions per execution, automatically
  cycling seeds 17/42/73 with independent resume state for 12 jobs; the complete model setup and PyTorch
  training/validation loop remains visible in notebook cells.
- `04_classifiers/03_Validation_Comparison.ipynb`: eight seed ensembles and validation comparisons.
- `04_classifiers/04_Final_Evaluation_and_Report.ipynb`: optional final evaluation and factual report.
- `utility/`: reusable architecture, dataset, metric and atomic-I/O primitives imported by notebooks.

No Python file launches notebooks, and no scientific notebook invokes a subprocess wrapper.
