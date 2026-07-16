# Notebook structure

- `1_preprocessing/`: split-safe preprocessing and traditional augmentation.
- `2_diffusers/`: generator training, generation and filtering notebooks.
- `3_generator_benchmark/05_Unified_Generator_Benchmark.ipynb`: validation-only benchmark.
- `3_generator_benchmark/06_Generator_Selection.ipynb`: transparent manual family selection.
- `04_classifiers/07_MaxViT512_Downstream.ipynb`: one of 12 MaxViT jobs per execution.
- `04_classifiers/08_MammoFM_Downstream.ipynb`: one of 12 Mammo-FM jobs per execution.
- `04_classifiers/09_Downstream_Validation_Comparison.ipynb`: eight seed ensembles and validation comparisons.
- `04_classifiers/10_Final_Evaluation_and_Report.ipynb`: optional final evaluation and factual report.
- `utility/`: reusable logic imported directly by notebooks.

No Python file launches notebooks, and no scientific notebook invokes a subprocess wrapper.
