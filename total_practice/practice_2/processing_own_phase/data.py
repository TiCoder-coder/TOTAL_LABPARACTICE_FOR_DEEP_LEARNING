"""Dataset, transforms, and DataLoader utilities for CIFAR-10."""

from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from configs import CLASS_NAMES, DATA_DIR, CONFIG


def get_transforms(image_size: int = CONFIG["image_size"]) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return a tuple of (train_transform, test_transform) for pre-trained models.
    
    Standard ImageNet normalization is used.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Resize slightly larger, then crop, flip, jitter, normalize
    train_transform = transforms.Compose([
        transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Validation/Test should NOT have random augmentations
    test_transform = transforms.Compose([
        transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transform, test_transform


def load_datasets(data_dir: str = str(DATA_DIR), val_ratio: float = CONFIG["train_split_ratio"], seed: int = CONFIG["seed"]):
    """Load CIFAR-10 and split into train, val, and test without data leakage.

    Creates two separate instances of the training dataset to apply different
    transforms (with and without augmentation) to the train and val splits.

    Returns:
        (train_subset, val_subset, test_dataset)
    """
    train_tfm, test_tfm = get_transforms()
    
    # Create two dataset instances for the training data to prevent augmentation leakage
    full_train_dataset_augmented = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tfm
    )
    full_train_dataset_clean = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=test_tfm
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tfm
    )

    # Calculate split sizes
    # Note: val_ratio is actually passed as 1.0 - train_split_ratio in main.py, so it's a proportion of val.
    n_total = len(full_train_dataset_augmented)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    # Generate fixed random indices for splitting
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Assert no overlap to prevent data leakage
    assert len(set(train_indices).intersection(set(val_indices))) == 0, "Data Leakage: Train and Val sets overlap!"

    # Create subsets pointing to the respective dataset instances
    train_subset = Subset(full_train_dataset_augmented, train_indices)
    val_subset = Subset(full_train_dataset_clean, val_indices)

    return train_subset, val_subset, test_dataset


def get_class_distribution(dataset, class_names=CLASS_NAMES) -> Dict[str, int]:
    """Compute the count of each class in a Subset or Dataset."""
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        # For Subset
        labels = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        # For standard Dataset
        labels = dataset.targets
        
    counts = {name: 0 for name in class_names}
    for label in labels:
        counts[class_names[label]] += 1
    return counts


def validate_dataset(dataset: Any, expected_classes: int = len(CLASS_NAMES)) -> bool:
    """Validate dataset integrity (empty check, classes count, corruption)."""
    if len(dataset) == 0:
        raise ValueError("Dataset is empty!")
        
    # Test loading a sample to ensure no corruption
    try:
        sample, label = dataset[0]
    except Exception as e:
        raise ValueError(f"Corrupted sample found at index 0: {e}")
        
    if label >= expected_classes or label < 0:
        raise ValueError(f"Label out of bounds: {label} (Expected max {expected_classes-1})")
        
    return True


def make_dataloaders(
    train_subset,
    val_subset,
    test_dataset,
    batch_size: int = CONFIG["batch_size"],
    num_workers: int = CONFIG["num_workers"],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, val, and test."""
    
    persistent = True if num_workers > 0 else False
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader
