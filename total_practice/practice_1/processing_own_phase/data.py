from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import CLASS_NAMES, DATA_DIR


def load_raw_datasets(
    data_dir: str = str(DATA_DIR),
    download: bool = True,
) -> Tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    raw_transform = transforms.ToTensor()
    training_pool = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=raw_transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=raw_transform,
    )
    return training_pool, test_dataset


def stratified_split_indices(
    targets: torch.Tensor,
    validation_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in the interval (0, 1)")

    generator = torch.Generator().manual_seed(seed)
    train_parts = []
    validation_parts = []

    for class_index in torch.unique(targets, sorted=True):
        class_indices = torch.where(targets == class_index)[0]
        class_order = torch.randperm(
            len(class_indices),
            generator=generator,
        )
        shuffled_indices = class_indices[class_order]
        validation_count = int(
            len(class_indices) * validation_ratio
        )
        validation_parts.append(
            shuffled_indices[:validation_count]
        )
        train_parts.append(shuffled_indices[validation_count:])

    train_indices = torch.cat(train_parts)
    validation_indices = torch.cat(validation_parts)
    train_indices = train_indices[
        torch.randperm(len(train_indices), generator=generator)
    ]
    validation_indices = validation_indices[
        torch.randperm(
            len(validation_indices),
            generator=generator,
        )
    ]
    return train_indices, validation_indices


def compute_mean_std_from_indices(
    image_data: torch.Tensor,
    indices: torch.Tensor,
) -> Tuple[float, float]:
    pixels = image_data[indices].to(torch.float32).div(255.0)
    mean = pixels.mean().item()
    std = pixels.std().item()
    if std <= 0.0:
        raise ValueError("training standard deviation must be positive")
    return mean, std


def build_transforms(
    mean: float,
    std: float,
):
    baseline_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])
    augmented_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])
    evaluation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])
    return (
        baseline_transform,
        augmented_transform,
        evaluation_transform,
    )


def prepare_datasets(
    data_dir: str = str(DATA_DIR),
    validation_ratio: float = 0.10,
    seed: int = 42,
    download: bool = True,
) -> Dict:
    raw_training_pool, _ = load_raw_datasets(
        data_dir=data_dir,
        download=download,
    )
    train_indices, validation_indices = stratified_split_indices(
        raw_training_pool.targets,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    mean, std = compute_mean_std_from_indices(
        raw_training_pool.data,
        train_indices,
    )
    (
        baseline_transform,
        augmented_transform,
        evaluation_transform,
    ) = build_transforms(mean, std)

    baseline_training_pool = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=baseline_transform,
    )
    augmented_training_pool = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=augmented_transform,
    )
    validation_pool = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=evaluation_transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=evaluation_transform,
    )

    train_index_list = train_indices.tolist()
    validation_index_list = validation_indices.tolist()

    return {
        "train_subset": Subset(
            baseline_training_pool,
            train_index_list,
        ),
        "augmented_train_subset": Subset(
            augmented_training_pool,
            train_index_list,
        ),
        "validation_subset": Subset(
            validation_pool,
            validation_index_list,
        ),
        "test_dataset": test_dataset,
        "baseline_training_pool": baseline_training_pool,
        "augmented_training_pool": augmented_training_pool,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "mean": mean,
        "std": std,
        "class_names": list(raw_training_pool.classes),
        "targets": raw_training_pool.targets,
    }


def make_train_loader(
    dataset,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
    )


def make_evaluation_loader(
    dataset,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def get_class_distribution(
    dataset,
    class_names=CLASS_NAMES,
) -> Dict[str, int]:
    if isinstance(dataset, Subset):
        targets = dataset.dataset.targets[dataset.indices]
    else:
        targets = dataset.targets
    counts = torch.bincount(
        torch.as_tensor(targets),
        minlength=len(class_names),
    )
    return {
        class_name: int(counts[index])
        for index, class_name in enumerate(class_names)
    }
