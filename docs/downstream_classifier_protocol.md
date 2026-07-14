# Downstream classifier protocol

## Fixed design

The publication protocol contains only MaxViT-512 and Mammo-FM, four data conditions, and seeds 17, 42 and 73: exactly 24 logical experiments. RAD-DINO is not a downstream classifier and ResNet-50 remains historical V1 material.

## Notebook execution

`07_MaxViT512_Downstream.ipynb` and `08_MammoFM_Downstream.ipynb` expose configuration, environment, selected generators, dataset construction and audit, model loading, trainable parameters, training policy, resume state, training, curves, best checkpoint, validation inference, metrics, calibration, error analysis, interpretability and saved artifacts.

Each run writes beneath `results/publication_v2/downstream/<architecture>/<condition>/seed_<seed>/`:

- `configuration.json`;
- `dataset_summary.json`;
- checkpoint and resume files;
- `training_history.csv`;
- `validation_predictions.csv`;
- `validation_metrics.json`;
- `interpretability/` when produced.

## Selection and fairness

The best checkpoint maximises validation PR-AUC. Early stopping and ReduceLROnPlateau monitor the same value. An equal PR-AUC uses lower validation loss; a further tie uses the earlier epoch. ROC-AUC is reported but never selects the primary checkpoint.

Within an architecture, every condition uses at most 6,400 optimizer updates, the same effective batch size, loss, class weighting, real-image online augmentation, validation manifest and validation frequency. The notebook reports samples seen by source, optimizer updates and effective epochs over each source. No hidden oversampling is permitted.

The validation comparison requires all three seeds, identical patient/image keys and labels, no duplicates or missing values, finite probabilities, and the same validation manifest. Ensemble probabilities are averaged, then metrics and bootstrap are computed patient-level.
