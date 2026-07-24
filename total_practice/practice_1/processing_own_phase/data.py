"""Dataset, transforms, and DataLoader utilities for FashionMNIST."""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from .config import CLASS_NAMES, DATA_DIR, NUM_CLASSES


def get_transforms(normalize: bool = False, mean: float = 0.0, std: float = 1.0):
    """Return a tuple of (train_transform, test_transform).

    Both transforms convert PIL images to float32 tensors in [0, 1].
    If normalize=True, also apply Normalize(mean, std).
    """
    def make_transform():
        ops = [
            transforms.ToTensor(),
        ]
        if normalize:
            ops.append(transforms.Normalize(mean=[mean], std=[std]))
        return transforms.Compose(ops)

    return make_transform(), make_transform()


def compute_mean_std(dataset) -> Tuple[float, float]:
    """Compute mean and std of the dataset in [0, 1] range."""
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    n_samples = 0
    channel_sum = 0.0
    channel_sq_sum = 0.0
    for images, _ in loader:
        n_samples += images.size(0)
        channel_sum += images.sum().item()
        channel_sq_sum += (images ** 2).sum().item()
    mean = channel_sum / (n_samples * 28 * 28)
    var = (channel_sq_sum / (n_samples * 28 * 28)) - mean ** 2
    std = float(np.sqrt(max(var, 1e-12)))
    return float(mean), float(std)


def load_datasets(data_dir: str = str(DATA_DIR), normalize: bool = False):
    """Load FashionMNIST train and test datasets.

    Returns:
        (train_dataset, test_dataset) where each dataset returns
        (image_tensor, label_int).
    """
    train_tfm, test_tfm = get_transforms(normalize=normalize)
    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=train_tfm
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=test_tfm
    )
    return train_dataset, test_dataset


def split_train_val(train_dataset, val_ratio: float = 0.1, seed: int = 42):
    """Split train_dataset into train/val subsets using a fixed seed."""
    n_total = len(train_dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset, [n_train, n_val], generator=generator
    )
    return train_subset, val_subset


def get_class_distribution(dataset, class_names=CLASS_NAMES) -> dict:
    """Compute the count of each class in a Subset or Dataset."""
    if hasattr(dataset, "dataset"):
        labels = [dataset.dataset.targets[i].item() for i in dataset.indices]
    else:
        labels = [t.item() for t in dataset.targets]
    counts = {name: 0 for name in class_names}
    for label in labels:
        counts[class_names[label]] += 1
    return counts


def make_dataloaders(
    train_subset,
    val_subset,
    test_dataset,
    batch_size: int = 64,
    num_workers: int = 0,
):
    """Create DataLoaders for train, val, and test."""
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def sanity_check_sample(image, label) -> dict:
    """Verify the shape/dtype/range of a single sample."""
    return {
        "shape": tuple(image.shape),
        "dtype": str(image.dtype),
        "min": float(image.min()),
        "max": float(image.max()),
        "label": int(label),
    }
