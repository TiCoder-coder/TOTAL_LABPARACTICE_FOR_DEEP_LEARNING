# Phase 3 — Data Loading and Split Construction

## Notebook location

- Notebook: [demo_practice_2.ipynb](../../notebooks/demo_practice_2.ipynb)
- Main cells: **Cells 5–6**
- Source: [data.py](../../processing_own_phase/data.py)
- Full mapping: [Cell–Output Map](../description_result/README.md)

## 1. Dataset source

The dataset comes from **TorchVision**, not from a manually downloaded Kaggle dataset. `torchvision.datasets.CIFAR10` manages download, checksum validation, image structure, and labels.

Official CIFAR-10 contains:

- 50,000 images in the official training pool;
- 10,000 images in the official Test set;
- 10 balanced classes, with 1,000 Test images per class.

The archive stored in `data/` is downloaded and managed by TorchVision.

## 2. Split procedure

The implementation follows this order:

1. Load the 50,000-image training pool with `transform=None`.
2. Use seed 42 to create a deterministic permutation of sample indices.
3. Compute `n_val = round(n_total × val_ratio)`.
4. Compute `n_train = n_total - n_val`.
5. Assign non-overlapping Train and Validation index lists.
6. Create separate Train and Validation dataset objects with their respective transforms.
7. Wrap each dataset object with its precomputed indices.
8. Load the official Test dataset with the deterministic evaluation transform.

With `train_split_ratio = 0.9`, the required sizes are:

| Split | Samples | Role |
|---|---:|---|
| Train | 45,000 | Fit model parameters |
| Validation | 5,000 | Early Stopping, checkpoint selection, and experiment selection |
| Test | 10,000 | Final evaluation after model decisions are locked |

## 3. Why indices must be fixed before preprocessing

If a learned preprocessing operation is fit before splitting, statistics from Validation or Test can influence the Train representation. If random augmentation is materialized before splitting, related variants can also cross split boundaries.

In the current implementation, indices define sample identity before transformed views are created. TorchVision transforms are lazy: they execute only when `__getitem__` accesses an image. The split therefore does not depend on a preprocessing result.

## 4. Separate dataset objects

Train and Validation use two independent `datasets.CIFAR10` instances before being wrapped as subsets. This design matters because `transform` is an attribute of the dataset object. If both subsets shared the same dataset instance, changing the Validation transform could overwrite Train augmentation or vice versa.

The current design guarantees:

- Train has stochastic augmentation.
- Validation has a deterministic transform.
- Test uses the same deterministic transform policy as Validation.
- Train and Validation index sets are disjoint.
- Train and Validation transforms cannot overwrite one another through shared object state.

## 5. Assertions and evidence

Cell 6 displays the split table and enforces:

- `len(train) == 45000`;
- `len(validation) == 5000`;
- `len(test) == 10000`;
- Train plus Validation equals the 50,000-image training pool;
- Train and Validation indices are disjoint;
- Train and Validation are backed by different dataset objects.

Automated tests are available in [test_data.py](../../tests/test_data.py).

## 6. Permitted use of each split

- **Train:** forward pass, loss, backward pass, optimizer updates, and scheduler-supported training.
- **Validation:** per-epoch evaluation, Early Stopping, best epoch, hyperparameter comparison, and experiment selection.
- **Test:** final evaluation after the checkpoint has been selected and verified.

Displaying Test sample count is metadata reporting, not model selection. No Test metric is written to the controlled experiment artifacts.

## 7. Related outputs

This phase primarily displays notebook tables. Dataset-oriented figures used in the following EDA phase include:

- [class_distribution.png](../../reports/class_distribution.png)
- [data_samples.png](../../reports/data_samples.png)

## 8. Suggested presentation script

> CIFAR-10 is loaded through TorchVision. From the official 50,000-image training pool, I determine indices with seed 42 before attaching transforms, producing 45,000 Train and 5,000 Validation samples. The official 10,000-image Test set remains separate. Train and Validation use different dataset objects, so their transforms cannot be shared accidentally.

## 9. Transition to the next phase

[Phase 4](phase_04_eda.md) examines class distribution, representative images, and the visual difficulty of CIFAR-10 before modeling.
