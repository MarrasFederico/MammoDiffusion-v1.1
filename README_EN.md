<p align="center"><img src="assets/logo_MammoDiffusion.png" alt="MammoDiffusion Logo" width="180"/></p>

# MammoDiffusion v2

MammoDiffusion is a notebook-first study of synthetic mammography and downstream utility.

- RQ1 compares eligible fine-tuned and from-scratch generators.
- RQ2 compares real-only, traditional augmentation, and positive synthetic augmentation.
- RQ3 evaluates consistency across MaxViT-512 and Mammo-FM.

Run the generator notebooks if outputs are missing, then notebooks 05 through 10 in order. Notebook 06 saves the transparent manual selection to `configs/selected_generators.json`. Notebooks 07 and 08 each cover 4 conditions × 3 seeds, preserving exactly 24 downstream experiments. Notebook 09 is validation-only. Notebook 10 has a visible final-evaluation opt-in and plain protocol snapshot.

The new workflow reads only `results/publication_v2/`. Historical V1 and retired matrix outputs stay in their existing paths but are ignored. Existing evidence shows that the old internal test was historically reused; it is not an untouched independent confirmation. See [the experimental design](docs/publication_experimental_design.md), [manual execution guide](docs/execution_guide.md), and [final-evaluation dataset status](docs/final_evaluation_dataset_status.md).

Dataset files, synthetic images, embeddings and model weights remain local. Mammo-FM weights retain their academic-license restrictions.
