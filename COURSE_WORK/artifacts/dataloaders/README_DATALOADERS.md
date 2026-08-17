# DATALOADERS-v1

SequenceWindowDataset is a map-style Dataset that slices registered WINDOWS-v1 records lazily from a read-only float32 feature timeline.

TRAIN and VALIDATION return x, y_model, y_raw_wh and sample_idx. TEST_LOCKED returns only x and sample_idx until the explicit Phase 47 evaluation gate is authorized.

Batches use the batch-first layout B by L by F. Batch sizes 32 and 64 are supported, with 64 as the baseline.

TRAIN shuffles window order with a split-specific torch.Generator. VALIDATION and TEST preserve chronological order. Every loader uses drop_last false.

The correctness baseline uses num_workers 0. The top-level worker initializer seeds NumPy and Python from torch.initial_seed for controlled optional multi-worker execution.

CUDA enables pin_memory. CPU and MPS keep pin_memory disabled. Dataset outputs remain CPU tensors and device transfer belongs to the training engine.

Audit loaders are disposable. Every training run must create fresh loaders so audit iteration never consumes the production generator state.
