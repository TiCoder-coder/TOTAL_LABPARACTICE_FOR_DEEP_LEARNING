from typing import Dict, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import CLASS_NAMES, DATA_DIR


def _histogram_quantile(
    histogram: torch.Tensor,
    quantile: float,
) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in the interval [0, 1]")
    cumulative = histogram.cumsum(dim=0)
    rank = int(quantile * (int(cumulative[-1]) - 1)) + 1
    index = torch.searchsorted(
        cumulative,
        torch.tensor(rank, dtype=cumulative.dtype),
    ).item()
    return float(index / 255.0)


def _duplicate_report(
    image_data: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, int]:
    records = {}
    for index in range(len(image_data)):
        key = image_data[index].numpy().tobytes()
        label = int(targets[index])
        if key not in records:
            records[key] = [1, label, False]
            continue
        records[key][0] += 1
        records[key][2] = (
            records[key][2]
            or label != records[key][1]
        )

    duplicate_records = [
        record
        for record in records.values()
        if record[0] > 1
    ]
    return {
        "duplicate_group_count": len(duplicate_records),
        "duplicate_sample_count": sum(
            record[0]
            for record in duplicate_records
        ),
        "additional_duplicate_count": sum(
            record[0] - 1
            for record in duplicate_records
        ),
        "conflicting_label_group_count": sum(
            int(record[2])
            for record in duplicate_records
        ),
    }


def analyze_training_pool(
    image_data: torch.Tensor,
    targets: torch.Tensor,
    class_names: Sequence[str] = CLASS_NAMES,
    seed: int = 42,
    samples_per_class: int = 2,
    outliers_per_group: int = 4,
    chunk_size: int = 2_048,
) -> Dict:
    if image_data.ndim != 3:
        raise ValueError("image_data must have shape [N, H, W]")
    if image_data.dtype != torch.uint8:
        raise ValueError("image_data must use torch.uint8")
    if targets.ndim != 1 or len(targets) != len(image_data):
        raise ValueError("targets must have shape [N]")
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    if outliers_per_group <= 0:
        raise ValueError("outliers_per_group must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    image_count, height, width = image_data.shape
    class_count = len(class_names)
    class_counts = torch.bincount(
        targets,
        minlength=class_count,
    )
    invalid_label_mask = (
        (targets < 0)
        | (targets >= class_count)
    )
    pixel_histogram = torch.zeros(256, dtype=torch.int64)
    image_means = torch.empty(image_count, dtype=torch.float32)
    image_stds = torch.empty(image_count, dtype=torch.float32)
    class_image_sums = torch.zeros(
        (class_count, height, width),
        dtype=torch.float64,
    )
    pixel_sum = 0.0
    pixel_square_sum = 0.0
    finite_image_count = 0
    black_image_count = 0
    white_image_count = 0
    constant_image_count = 0

    for start in range(0, image_count, chunk_size):
        end = min(start + chunk_size, image_count)
        raw_chunk = image_data[start:end]
        label_chunk = targets[start:end]
        pixel_chunk = raw_chunk.to(torch.float64).div(255.0)
        flat_chunk = pixel_chunk.flatten(start_dim=1)
        chunk_means = flat_chunk.mean(dim=1)
        chunk_stds = flat_chunk.std(dim=1, correction=0)
        chunk_minimums = raw_chunk.flatten(start_dim=1).amin(dim=1)
        chunk_maximums = raw_chunk.flatten(start_dim=1).amax(dim=1)

        pixel_histogram += torch.bincount(
            raw_chunk.flatten().to(torch.int64),
            minlength=256,
        )
        pixel_sum += pixel_chunk.sum().item()
        pixel_square_sum += pixel_chunk.square().sum().item()
        image_means[start:end] = chunk_means.to(torch.float32)
        image_stds[start:end] = chunk_stds.to(torch.float32)
        finite_image_count += int(
            torch.isfinite(flat_chunk).all(dim=1).sum()
        )
        black_image_count += int((chunk_maximums == 0).sum())
        white_image_count += int((chunk_minimums == 255).sum())
        constant_image_count += int(
            (chunk_minimums == chunk_maximums).sum()
        )

        for class_index in range(class_count):
            class_mask = label_chunk == class_index
            if class_mask.any():
                class_image_sums[class_index] += (
                    pixel_chunk[class_mask].sum(dim=0)
                )

    pixel_count = int(image_data.numel())
    pixel_mean = pixel_sum / pixel_count
    pixel_variance = (
        pixel_square_sum
        - pixel_sum * pixel_sum / pixel_count
    ) / (pixel_count - 1)
    pixel_std = max(pixel_variance, 0.0) ** 0.5
    class_mean_images = (
        class_image_sums
        / class_counts.to(torch.float64).view(-1, 1, 1)
    ).to(torch.float32)

    generator = torch.Generator().manual_seed(seed)
    sample_indices = []
    for class_index in range(class_count):
        class_indices = torch.where(targets == class_index)[0]
        if len(class_indices) < samples_per_class:
            raise ValueError(
                "each class must contain enough samples"
            )
        selected = class_indices[
            torch.randperm(
                len(class_indices),
                generator=generator,
            )[:samples_per_class]
        ]
        sample_indices.extend(selected.tolist())

    outlier_indices = {
        "darkest": torch.argsort(image_means)[
            :outliers_per_group
        ].tolist(),
        "brightest": torch.argsort(
            image_means,
            descending=True,
        )[:outliers_per_group].tolist(),
        "lowest_contrast": torch.argsort(image_stds)[
            :outliers_per_group
        ].tolist(),
        "highest_contrast": torch.argsort(
            image_stds,
            descending=True,
        )[:outliers_per_group].tolist(),
    }

    per_class_statistics = {}
    class_distribution = {}
    for class_index, class_name in enumerate(class_names):
        class_mask = targets == class_index
        class_distribution[class_name] = {
            "count": int(class_counts[class_index]),
            "percentage": float(
                class_counts[class_index] / image_count * 100.0
            ),
        }
        per_class_statistics[class_name] = {
            "mean_brightness": float(
                image_means[class_mask].mean()
            ),
            "median_brightness": float(
                image_means[class_mask].median()
            ),
            "mean_contrast": float(
                image_stds[class_mask].mean()
            ),
            "median_contrast": float(
                image_stds[class_mask].median()
            ),
        }

    duplicate_report = _duplicate_report(image_data, targets)
    quality_report = {
        "non_finite_image_count": (
            image_count - finite_image_count
        ),
        "black_image_count": black_image_count,
        "white_image_count": white_image_count,
        "constant_image_count": constant_image_count,
        "invalid_label_count": int(invalid_label_mask.sum()),
        "missing_class_count": int((class_counts == 0).sum()),
        **duplicate_report,
    }
    report = {
        "scope": "official_training_pool_only",
        "schema": {
            "sample_count": image_count,
            "height": height,
            "width": width,
            "channels": 1,
            "raw_dtype": str(image_data.dtype),
            "raw_min": int(image_data.min()),
            "raw_max": int(image_data.max()),
            "label_min": int(targets.min()),
            "label_max": int(targets.max()),
            "class_count": class_count,
        },
        "class_distribution": class_distribution,
        "pixel_statistics": {
            "pixel_count": pixel_count,
            "mean": pixel_mean,
            "std": pixel_std,
            "minimum": float(image_data.min() / 255.0),
            "maximum": float(image_data.max() / 255.0),
            "q01": _histogram_quantile(
                pixel_histogram,
                0.01,
            ),
            "q05": _histogram_quantile(
                pixel_histogram,
                0.05,
            ),
            "q25": _histogram_quantile(
                pixel_histogram,
                0.25,
            ),
            "median": _histogram_quantile(
                pixel_histogram,
                0.50,
            ),
            "q75": _histogram_quantile(
                pixel_histogram,
                0.75,
            ),
            "q95": _histogram_quantile(
                pixel_histogram,
                0.95,
            ),
            "q99": _histogram_quantile(
                pixel_histogram,
                0.99,
            ),
            "zero_fraction": float(
                pixel_histogram[0] / pixel_count
            ),
            "maximum_fraction": float(
                pixel_histogram[255] / pixel_count
            ),
        },
        "per_class_statistics": per_class_statistics,
        "quality": quality_report,
        "sample_seed": seed,
        "samples_per_class": samples_per_class,
    }
    return {
        "report": report,
        "pixel_histogram": pixel_histogram,
        "image_means": image_means,
        "image_stds": image_stds,
        "targets": targets.clone(),
        "class_mean_images": class_mean_images,
        "sample_indices": sample_indices,
        "outlier_indices": outlier_indices,
    }


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
    eda = analyze_training_pool(
        raw_training_pool.data,
        raw_training_pool.targets,
        class_names=raw_training_pool.classes,
        seed=seed,
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
        "eda": eda,
        "raw_image_data": raw_training_pool.data,
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
