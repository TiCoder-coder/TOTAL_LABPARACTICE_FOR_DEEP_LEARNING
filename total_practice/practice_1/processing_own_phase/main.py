"""Main entry point for FashionMNIST classification pipeline.

This script orchestrates the full pipeline:
    1. Setup environment and reproducibility
    2. Load data and split train/val/test
    3. Build model and run sanity checks
    4. Train baseline and run controlled experiments
    5. Evaluate best model on the test set (once)
    6. Save/load model and verify
    7. Generate visualizations and a summary report
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Force matplotlib to use a writable cache directory
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs" / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / "outputs" / ".cache"))
(Path(__file__).resolve().parent / "outputs" / ".mpl_cache").mkdir(parents=True, exist_ok=True)
(Path(__file__).resolve().parent / "outputs" / ".cache").mkdir(parents=True, exist_ok=True)

from .config import (
    CLASS_NAMES,
    CONFIG,
    EXPERIMENTS,
    NUM_CLASSES,
    OUTPUT_DIR,
)
from .data import (
    get_class_distribution,
    load_datasets,
    make_dataloaders,
    sanity_check_sample,
    split_train_val,
)
from .evaluate import evaluate, per_class_accuracy
from .experiment import ExperimentRunner
from .model import FashionMLP, build_model, sanity_check_model
from .save_load import load_model_from_checkpoint, save_checkpoint, verify_loaded_model
from .train import fit
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


def run_full_pipeline(quick_test: bool = False):
    """Run the full FashionMNIST training pipeline.

    Args:
        quick_test: if True, run only 2 epochs per experiment for fast verification.
    """
    pipeline_start = time.time()
    print("\n" + "=" * 70)
    print("FashionMNIST Classification Pipeline")
    print("=" * 70)

    # --- Phase 1: Setup ---
    setup_reproducibility(CONFIG["seed"])
    device = get_device()
    print_environment_summary(device)

    # --- Phase 2: Data loading ---
    print("\n[Phase 2] Loading FashionMNIST dataset...")
    train_dataset, test_dataset = load_datasets(data_dir=CONFIG["data_dir"])
    train_subset, val_subset = split_train_val(
        train_dataset, val_ratio=1.0 - CONFIG["train_split_ratio"], seed=CONFIG["seed"]
    )
    train_loader, val_loader, test_loader = make_dataloaders(
        train_subset, val_subset, test_dataset,
        batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"],
    )
    print(f"  Train size: {len(train_subset)}")
    print(f"  Val size:   {len(val_subset)}")
    print(f"  Test size:  {len(test_dataset)}")

    # Sanity check one sample
    sample_img, sample_label = train_dataset[0]
    sample_info = sanity_check_sample(sample_img, sample_label)
    print(f"  Sample shape: {sample_info['shape']} dtype: {sample_info['dtype']} "
          f"min: {sample_info['min']:.3f} max: {sample_info['max']:.3f} "
          f"label: {CLASS_NAMES[sample_info['label']]}")

    # EDA: data samples and class distribution
    print("\n[EDA] Saving data samples and class distribution...")
    sample_images, sample_labels = next(iter(train_loader))
    plot_data_samples(sample_images[:20], sample_labels[:20].numpy())
    distributions = {
        "Train": get_class_distribution(train_subset),
        "Val":   get_class_distribution(val_subset),
        "Test":  get_class_distribution(test_dataset),
    }
    plot_class_distribution(distributions)

    # --- Phase 3: Model sanity checks ---
    print("\n[Phase 3] Building model and running sanity checks...")
    model = build_model(CONFIG["hidden_dims"], CONFIG["dropout"])
    print(f"  Architecture: {CONFIG['hidden_dims']} | Dropout: {CONFIG['dropout']}")
    print(f"  Parameters: {count_parameters(model):,}")
    check = sanity_check_model(model, device)
    print(f"  Forward shape: {check['forward_shape']}")
    print(f"  Loss is finite: {check['loss_is_finite']} (loss={check['loss_value']:.4f})")
    print(f"  Gradients populated: {check['gradients_populated']}")
    print(f"  Params changed after step: {check['params_changed']}")

    # --- Phase 4 & 5: Baseline + experiments ---
    print("\n[Phase 4-5] Running controlled experiments...")
    runner = ExperimentRunner(device=device, output_dir=OUTPUT_DIR)
    epochs_per_exp = 2 if quick_test else CONFIG["epochs"]
    all_results = []
    for exp_id, exp_cfg in EXPERIMENTS.items():
        result = runner.run(
            exp_id=exp_id,
            description=exp_cfg["description"],
            train_loader=train_loader,
            val_loader=val_loader,
            hidden_dims=exp_cfg["hidden_dims"],
            dropout=exp_cfg["dropout"],
            optimizer_name=exp_cfg["optimizer"],
            learning_rate=exp_cfg["learning_rate"],
            batch_size=exp_cfg["batch_size"],
            epochs=epochs_per_exp,
        )
        all_results.append(result)
    runner.save_all()

    # --- Phase 6: Select best experiment ---
    print("\n[Phase 6] Selecting best experiment by validation accuracy...")
    best_result = max(all_results, key=lambda r: r["best_val_acc"])
    best_exp_id = best_result["exp_id"]
    print(f"  Best experiment: {best_exp_id}")
    print(f"  Best Val Accuracy: {best_result['best_val_acc']:.4f}")
    print(f"  Best Epoch: {best_result['best_epoch']}")
    print(f"  Description: {best_result['description']}")

    # Reload best model from its checkpoint
    best_ckpt_path = OUTPUT_DIR / f"{best_exp_id}.pt"
    best_model = load_model_from_checkpoint(str(best_ckpt_path), device=device)
    print(f"  Loaded best checkpoint: {best_ckpt_path}")

    # --- Phase 7: Final test (ONCE) ---
    print("\n[Phase 7] Final evaluation on test set (ONCE)...")
    criterion = nn.CrossEntropyLoss()
    test_result = evaluate(best_model, test_loader, criterion, device)
    print(f"  Test Loss:     {test_result['loss']:.4f}")
    print(f"  Test Accuracy: {test_result['accuracy']:.4f}")
    per_class = per_class_accuracy(test_result["predictions"], test_result["labels"])
    for c, name in enumerate(CLASS_NAMES):
        print(f"    {name:>14s}: {per_class[c]:.4f}")

    # --- Phase 8: Save best model + verify loading ---
    print("\n[Phase 8] Saving best model and verifying loading...")
    save_checkpoint(
        best_model,
        path=str(OUTPUT_DIR / "best_model.pth"),
        config=CONFIG,
        best_epoch=best_result["best_epoch"],
        best_val_acc=best_result["best_val_acc"],
        best_val_loss=best_result["best_val_loss"],
        extra={"test_accuracy": test_result["accuracy"], "test_loss": test_result["loss"]},
    )
    print(f"  Saved: {OUTPUT_DIR / 'best_model.pth'}")

    # Reload into a fresh model on the same device and check predictions match
    fresh_model = load_model_from_checkpoint(str(OUTPUT_DIR / "best_model.pth"), device=device)
    sample_for_verify, _ = next(iter(test_loader))
    sample_for_verify = sample_for_verify.to(device)
    verify = verify_loaded_model(best_model, fresh_model, sample_for_verify)
    print(f"  Same predictions after reload: {verify['same_predictions']}")
    print(f"  Max logit difference: {verify['max_logit_diff']:.2e}")

    # --- Visualizations ---
    print("\n[Visualization] Generating plots...")
    best_history = best_result["history"]
    plot_loss_curves(best_history)
    plot_accuracy_curves(best_history)
    plot_experiment_comparison(all_results)
    plot_confusion_matrix(test_result["predictions"], test_result["labels"])

    # Sample predictions grid
    test_iter = iter(test_loader)
    grid_images, grid_labels = next(test_iter)
    grid_images = grid_images.to(device)
    with torch.inference_mode():
        logits = best_model(grid_images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
    plot_predictions_grid(
        grid_images.cpu(), preds, grid_labels.numpy(), probs,
    )

    # --- Save summary report ---
    summary = {
        "device": str(device),
        "config": CONFIG,
        "best_experiment": best_exp_id,
        "best_val_acc": best_result["best_val_acc"],
        "best_val_loss": best_result["best_val_loss"],
        "best_epoch": best_result["best_epoch"],
        "test_accuracy": test_result["accuracy"],
        "test_loss": test_result["loss"],
        "per_class_accuracy": per_class,
        "verify": verify,
        "total_pipeline_time": time.time() - pipeline_start,
    }
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved: {summary_path}")

    total_time = time.time() - pipeline_start
    print(f"\nPipeline completed in {format_time(total_time)}")
    print(f"Final Test Accuracy: {test_result['accuracy']:.4f}")
    return summary


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv
    run_full_pipeline(quick_test=quick)
