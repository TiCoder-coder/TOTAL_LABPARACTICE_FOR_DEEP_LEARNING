import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Subset

from .config import (
    CLASS_NAMES,
    CONFIG,
    EXPERIMENTS,
    MODEL_SAVE_PATH,
    OUTPUT_DIR,
    RUNS_DIR,
)
from .data import (
    get_class_distribution,
    make_evaluation_loader,
    make_train_loader,
    prepare_datasets,
)
from .evaluate import (
    evaluate_classifier,
    per_class_accuracy,
)
from .experiment import run_experiments
from .model import build_model, sanity_check_model
from .save_load import (
    load_model_from_checkpoint,
    save_checkpoint,
    verify_loaded_model,
)
from .train import train_final_model
from .utils import (
    count_parameters,
    format_time,
    get_device,
    print_environment_summary,
    setup_reproducibility,
)
from .visualize import (
    plot_accuracy_curves,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_data_samples,
    plot_experiment_comparison,
    plot_loss_curves,
    plot_predictions_grid,
)


def limit_dataset(dataset, sample_count: int):
    return Subset(
        dataset,
        range(min(sample_count, len(dataset))),
    )


def run_full_pipeline(quick_test: bool = False):
    started_at = time.perf_counter()
    setup_reproducibility(CONFIG["seed"])
    device = get_device()
    print_environment_summary(device)

    data = prepare_datasets(
        validation_ratio=CONFIG["validation_ratio"],
        seed=CONFIG["seed"],
        download=True,
    )

    assert len(data["train_subset"]) == 54_000
    assert len(data["validation_subset"]) == 6_000
    assert len(data["test_dataset"]) == 10_000
    assert len(
        set(data["train_indices"].tolist()).intersection(
            data["validation_indices"].tolist()
        )
    ) == 0

    train_subset = data["train_subset"]
    augmented_train_subset = data[
        "augmented_train_subset"
    ]
    validation_dataset = data["validation_subset"]
    test_dataset = data["test_dataset"]
    baseline_training_pool = data[
        "baseline_training_pool"
    ]
    augmented_training_pool = data[
        "augmented_training_pool"
    ]
    experiment_configs = [
        dict(config)
        for config in EXPERIMENTS
    ]

    if quick_test:
        train_subset = limit_dataset(train_subset, 1_024)
        augmented_train_subset = limit_dataset(
            augmented_train_subset,
            1_024,
        )
        validation_dataset = limit_dataset(
            validation_dataset,
            512,
        )
        test_dataset = limit_dataset(test_dataset, 512)
        baseline_training_pool = limit_dataset(
            baseline_training_pool,
            1_024,
        )
        augmented_training_pool = limit_dataset(
            augmented_training_pool,
            1_024,
        )
        for config in experiment_configs:
            config["epochs"] = 1

    sample_loader = make_train_loader(
        train_subset,
        batch_size=CONFIG["batch_size"],
        seed=CONFIG["seed"],
        num_workers=CONFIG["num_workers"],
    )
    validation_loader = make_evaluation_loader(
        validation_dataset,
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )
    test_loader = make_evaluation_loader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )

    sample_images, sample_labels = next(iter(sample_loader))
    baseline_model = build_model(experiment_configs[0])
    model_sanity = sanity_check_model(
        baseline_model,
        sample_images,
        sample_labels,
    )
    assert count_parameters(baseline_model) == 101_770

    experiment_results, selected_result, run_root = (
        run_experiments(
            experiment_configs,
            train_subset,
            augmented_train_subset,
            validation_loader,
            device,
            RUNS_DIR,
            OUTPUT_DIR,
        )
    )

    final_config = dict(selected_result["config"])
    final_config["experiment_id"] = (
        f"{selected_result['experiment_id']}_final"
    )
    final_config["epochs"] = selected_result["best_epoch"]
    final_training_dataset = (
        augmented_training_pool
        if final_config["use_augmentation"]
        else baseline_training_pool
    )
    final_training_result = train_final_model(
        final_config,
        final_training_dataset,
        device,
        log_directory=(
            run_root / final_config["experiment_id"]
        ),
    )
    final_model = final_training_result["model"]

    test_result = evaluate_classifier(
        final_model,
        test_loader,
        nn.CrossEntropyLoss(),
        device,
    )
    assert test_result["sample_count"] == len(test_dataset)

    model_config = {
        "hidden_dims": tuple(final_config["hidden_dims"]),
        "dropout": final_config["dropout"],
        "num_classes": final_config["num_classes"],
        "input_dim": final_config["input_dim"],
    }
    metadata = {
        "selected_experiment": (
            selected_result["experiment_id"]
        ),
        "best_epoch": selected_result["best_epoch"],
        "best_validation_accuracy": (
            selected_result["best_validation_accuracy"]
        ),
        "best_validation_loss": (
            selected_result["best_validation_loss"]
        ),
        "test_accuracy": test_result["accuracy"],
        "test_loss": test_result["loss"],
        "train_mean": data["mean"],
        "train_std": data["std"],
        "class_names": list(CLASS_NAMES),
    }
    save_checkpoint(
        final_model,
        model_config,
        final_config,
        metadata,
        str(MODEL_SAVE_PATH),
    )
    loaded_model = load_model_from_checkpoint(
        str(MODEL_SAVE_PATH),
        device,
    )
    verification_images, verification_labels = next(
        iter(test_loader)
    )
    verification = verify_loaded_model(
        final_model,
        loaded_model,
        verification_images.to(device),
    )
    assert verification["predictions_match"]
    assert verification["logits_match"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    display_training_images = (
        sample_images * data["std"] + data["mean"]
    ).clamp(0.0, 1.0)
    plot_data_samples(
        display_training_images[:20],
        sample_labels[:20].numpy(),
    )
    plot_class_distribution({
        "Train": get_class_distribution(
            data["train_subset"]
        ),
        "Validation": get_class_distribution(
            data["validation_subset"]
        ),
        "Test": get_class_distribution(
            data["test_dataset"]
        ),
    })
    plot_loss_curves(selected_result["history"])
    plot_accuracy_curves(selected_result["history"])
    plot_experiment_comparison(experiment_results)
    plot_confusion_matrix(
        test_result["predictions"],
        test_result["targets"],
    )

    verification_images_device = (
        verification_images.to(device)
    )
    with torch.inference_mode():
        verification_logits = loaded_model(
            verification_images_device
        )
        verification_probabilities = torch.softmax(
            verification_logits,
            dim=1,
        ).cpu().numpy()
        verification_predictions = (
            verification_logits.argmax(dim=1).cpu().numpy()
        )
    display_test_images = (
        verification_images * data["std"] + data["mean"]
    ).clamp(0.0, 1.0)
    plot_predictions_grid(
        display_test_images,
        verification_predictions,
        verification_labels.numpy(),
        verification_probabilities,
    )

    experiment_summary = [
        {
            "experiment_id": result["experiment_id"],
            "config": result["config"],
            "best_epoch": result["best_epoch"],
            "best_validation_accuracy": (
                result["best_validation_accuracy"]
            ),
            "best_validation_loss": (
                result["best_validation_loss"]
            ),
            "parameter_count": result["parameter_count"],
            "total_seconds": result["total_seconds"],
        }
        for result in experiment_results
    ]
    per_class = per_class_accuracy(
        test_result["predictions"],
        test_result["targets"],
    )
    summary = {
        "quick_test": quick_test,
        "device": str(device),
        "data": {
            "train_samples": len(train_subset),
            "validation_samples": len(
                validation_dataset
            ),
            "test_samples": len(test_dataset),
            "train_mean": data["mean"],
            "train_std": data["std"],
        },
        "model_sanity": model_sanity,
        "experiments": experiment_summary,
        "selected_experiment": (
            selected_result["experiment_id"]
        ),
        "final_config": final_config,
        "test_accuracy": test_result["accuracy"],
        "test_loss": test_result["loss"],
        "per_class_accuracy": per_class,
        "checkpoint_verification": verification,
        "total_seconds": (
            time.perf_counter() - started_at
        ),
    }
    summary_path = OUTPUT_DIR / (
        "summary_quick.json"
        if quick_test
        else "summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print(
        f"Selected experiment: "
        f"{selected_result['experiment_id']}"
    )
    print(
        f"Best validation accuracy: "
        f"{selected_result['best_validation_accuracy']:.2%}"
    )
    print(f"Test accuracy: {test_result['accuracy']:.2%}")
    print(
        f"Total time: "
        f"{format_time(summary['total_seconds'])}"
    )
    print(f"Summary: {summary_path}")
    return summary


if __name__ == "__main__":
    import sys

    run_full_pipeline(quick_test="--quick" in sys.argv)
