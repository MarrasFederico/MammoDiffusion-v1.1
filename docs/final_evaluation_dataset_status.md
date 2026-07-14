# Final evaluation dataset status

## Finding

The previous internal test is a **historically reused internal evaluation set**, not an untouched internal holdout.

The repository contains V1 test metric files, historical MaxViT and ResNet test outputs, final-test prediction paths and prior final-evaluation coverage tables. This is sufficient evidence that the split has informed previous project analyses. This audit did not open new test images, run inference, or modify any split.

## Honest terminology

If reused, this dataset must be described as a historical internal test, development holdout, or reused internal evaluation set. Results are internal, exploratory, and not an independent external confirmation. It must not be called unopened, untouched, or pristine.

For a future publication, prefer an external dataset, a new untouched holdout created through an explicit scientific decision, or cross-dataset external validation. This refactoring does not create a new split, move patients, or alter existing manifests.
