# Classifier Stage 1 checkpoint and resume policy

The final TensorFlow ResNet checkpoint is a single HDF5 model stored at the canonical
`model.keras` path with optimizer state excluded. TensorFlow/Keras 2.15 cannot serialize this
ResNet application with the native Keras writer, while the HDF5 representation is loadable in a
fresh process and preserves predictions. Optimizer state remains in the rotating resume
checkpoints.

TensorFlow resume phases are interpreted as follows:

- `head`: restore model weights and the head optimizer;
- `transition`: restore the best head model weights, then create a fresh fine-tuning optimizer;
- `finetune`: restore model weights and the fine-tuning optimizer;
- `complete`: load the saved model state without repeating training.

PyTorch DataLoaders use a locally seeded generator whose sampler state is not serialized. An
intra-epoch restart therefore retains completed epochs, model/optimizer/scheduler state and the
global update count, but restarts the incomplete epoch at batch zero. At most one incomplete
epoch is repeated; the runner does not claim an exact same-batch continuation.

Resume compatibility uses one dataset signature combining content-aware training and validation
signatures. Each record includes its project-relative path, byte size, SHA-256, label, source,
patient/image identity and augmentation provenance when applicable. Changing either split
invalidates the scientific resume configuration.
