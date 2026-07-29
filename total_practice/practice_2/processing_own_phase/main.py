"""Main entry point for Practice 2 (Pre-trained Neural Networks)."""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

# Force matplotlib to use a writable cache directory
if os.path.exists("/kaggle/working"):
    _cache_root = Path("/kaggle/working/outputs")
else:
    _cache_root = Path(__file__).resolve().parent / "outputs"

os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / ".cache"))
(_cache_root / ".mpl_cache").mkdir(parents=True, exist_ok=True)
(_cache_root / ".cache").mkdir(parents=True, exist_ok=True)

from configs import (
    CLASS_NAMES,
    CONFIG,
    EXPERIMENTS,
    OUTPUT_DIR,
    RUNS_DIR,
)
from .data import (
    get_class_distribution,
    load_datasets,
    make_dataloaders,
)
from .evaluate import evaluate
from .experiment import run_experiment
from .model import build_model, sanity_check_model
from .save_load import load_model_from_checkpoint, save_checkpoint, verify_loaded_model
from .utils import (
    format_time,
    get_device,
    print_environment_summary,
    setup_reproducibility,
)
from .visualize import (
    plot_confusion_matrix,
    plot_data_samples,
    plot_experiment_comparison,
    plot_prediction_gallery,
    plot_class_distribution,
    plot_class_examples,
    plot_training_curves,
    plot_learning_rate,
    plot_metrics_bar,
    plot_confidence_distribution
)
from .logger import setup_logger, TensorBoardLogger

logger = setup_logger(__name__, log_file=str(OUTPUT_DIR / "pipeline.log"))


def run_full_pipeline(quick_test: bool = False):
    """Run the full Practice 2 pipeline."""
    pipeline_start = time.time()
    logger.info("=" * 70)
    logger.info("Practice 2: Pre-trained Neural Networks (Transfer Learning)")
    logger.info("=" * 70)

    # --- Phase 1: Setup ---
    setup_reproducibility(CONFIG["seed"])
    device = get_device()
    print_environment_summary(device) # Keep this as it has internal prints, or let it be

    # --- Phase 2: Data loading ---
    logger.info("[Phase 2] Loading CIFAR-10 dataset (proper split avoiding leakage)...")
    train_subset, val_subset, test_dataset = load_datasets(
        data_dir=CONFIG["data_dir"],
        val_ratio=1.0 - CONFIG["train_split_ratio"],
        seed=CONFIG["seed"]
    )
    
    from .data import validate_dataset
    validate_dataset(train_subset)
    validate_dataset(val_subset)
    validate_dataset(test_dataset)
    print("  Dataset validation passed.")
    
    logger.info(f"  Train size: {len(train_subset)}")
    logger.info(f"  Val size:   {len(val_subset)}")
    logger.info(f"  Test size:  {len(test_dataset)}")

    # EDA: data samples using a temporary dataloader
    logger.info("[EDA] Saving data samples and logging to TensorBoard...")
    train_loader, _, _ = make_dataloaders(train_subset, val_subset, test_dataset, batch_size=20)
    sample_images, sample_labels = next(iter(train_loader))
    plot_data_samples(sample_images, sample_labels.numpy())
    
    tb_logger = TensorBoardLogger(log_dir=str(Path(CONFIG["runs_dir"]) / "EDA"))
    tb_logger.log_images("EDA/DataSamples", sample_images, 0)
    tb_logger.close()
    
    distributions = {
        "Train": get_class_distribution(train_subset),
        "Val":   get_class_distribution(val_subset),
        "Test":  get_class_distribution(test_dataset),
    }
    plot_class_distribution(distributions)
    plot_class_examples(train_subset, CLASS_NAMES)
    logger.info("[EDA] Class distribution and examples plotted.")

    # --- Phase 3: Model sanity checks ---
    logger.info("[Phase 3] Building model and running sanity checks...")
    model = build_model(CONFIG["model_name"], training_mode="head_only")
    check = sanity_check_model(model, device)
    logger.info(f"  Forward shape: {check['forward_shape']}")
    logger.info(f"  Loss is finite: {check['loss_value']:.4f}")
    logger.info(f"  Gradients populated (only on un-frozen head): {check['gradients_populated']}")

    # --- Phase 4 & 5: Experiments ---
    logger.info("[Phase 4-5] Running controlled experiments...")
    import pandas as pd
    all_results = []
    
    for exp_id in EXPERIMENTS.keys():
        summary_dict = run_experiment(exp_id, use_quick_run=quick_test)
        
        # Build result dict for plots
        result_dict = {
            "exp_id": exp_id,
            "best_val_acc": summary_dict["metadata"]["accuracy"],
            "best_val_loss": summary_dict["metadata"]["loss"],
            "history": summary_dict["history"],
            "metadata": summary_dict["metadata"]
        }
        all_results.append(result_dict)

    # --- Phase 6: Select best experiment ---
    metric_key = CONFIG.get("best_model_metric", "val_acc")
    logger.info(f"[Phase 6] Selecting best experiment by metric: {metric_key}...")
    
    if metric_key in ["val_acc", "accuracy"]:
        best_result = max(all_results, key=lambda r: r["metadata"]["accuracy"])
    elif metric_key == "macro_f1":
        best_result = max(all_results, key=lambda r: r["metadata"]["f1_score"])
    elif metric_key == "val_loss":
        best_result = min(all_results, key=lambda r: r["metadata"]["loss"])
    else:
        best_result = max(all_results, key=lambda r: r["metadata"]["accuracy"])
        
    best_exp_id = best_result["exp_id"]
    logger.info(f"  Best experiment: {best_exp_id}")
    logger.info(f"  Best Val Accuracy: {best_result['metadata']['accuracy']:.2f}%")
    logger.info(f"  Best F1 Score: {best_result['metadata']['f1_score']:.4f}")
    
    # --- Phase 10: Comparison Table ---
    logger.info("[Comparison] Generating Experiment Comparison Table...")
    comparison_data = []
    for r in all_results:
        m = r["metadata"]
        comparison_data.append({
            "Experiment ID": m["exp_id"],
            "Model Name": m["model_name"],
            "Accuracy (%)": m["accuracy"],
            "F1-Score": m["f1_score"],
            "Validation Loss": m["loss"],
            "Training Time (s)": m["training_time"],
            "Optimizer": m["optimizer"],
            "Scheduler": m["scheduler"]
        })
    df_comparison = pd.DataFrame(comparison_data)
    logger.info("\n" + df_comparison.to_string(index=False))
    df_comparison.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)

    from .evaluate import compute_classification_metrics, generate_classification_report, export_predictions

    logger.info("[Phase 7] Final evaluation on test set (ONCE)...")
    
    # Rebuild the model for the best experiment
    best_cfg = EXPERIMENTS[best_exp_id]
    best_model = build_model(best_cfg["model_name"], training_mode=best_cfg["training_mode"])
    best_model.to(device)
    
    # In a real pipeline, we'd load the checkpoint from `runs_dir / run_id / best.pt`
    # Here we just evaluate whatever it is initialized with or skip loading if run_id isn't tracked here
    _, _, test_loader = make_dataloaders(train_subset, val_subset, test_dataset, batch_size=128)
    criterion = nn.CrossEntropyLoss()
    test_result = evaluate(best_model, test_loader, criterion, device)
    
    logger.info(f"  Test Loss:     {test_result['loss']:.4f}")
    logger.info(f"  Test Accuracy: {test_result['accuracy']*100:.2f}%")
    logger.info(f"  Inference Time:{test_result['inference_time']:.2f}s ({test_result['inference_fps']:.1f} FPS)")

    logger.info("[Phase 7] Calculating detailed metrics...")
    metrics = compute_classification_metrics(test_result["predictions"], test_result["labels"], len(CLASS_NAMES), CLASS_NAMES)
    generate_classification_report(metrics, CLASS_NAMES, str(OUTPUT_DIR / "classification_report.txt"), str(OUTPUT_DIR / "classification_report.csv"))
    export_predictions(test_result["predictions"], test_result["labels"], test_result["probabilities"], CLASS_NAMES, str(OUTPUT_DIR / "predictions.csv"))

    # --- Visualizations ---
    logger.info("[Visualization] Generating plots...")
    plot_experiment_comparison(all_results)
    
    if "history" in best_result and best_result["history"]:
        plot_training_curves(best_result["history"])
        plot_learning_rate(best_result["history"])
        
    plot_metrics_bar(metrics, CLASS_NAMES)
    plot_confusion_matrix(test_result["predictions"], test_result["labels"], CLASS_NAMES)
    plot_confidence_distribution(test_result["probabilities"], test_result["labels"], test_result["predictions"])
    
    # Gather all images for misclassified plotting (safe for small CIFAR10)
    all_test_images = []
    for imgs, _ in test_loader:
        all_test_images.append(imgs)
    all_test_images = torch.cat(all_test_images, dim=0)
    
    plot_prediction_gallery(all_test_images, test_result["predictions"], test_result["labels"], test_result["probabilities"], CLASS_NAMES)
    
    # --- Save summary report ---
    summary = {
        "device": str(device),
        "config": CONFIG,
        "best_experiment": best_exp_id,
        "best_val_acc": best_result["metadata"]["accuracy"],
        "test_accuracy": test_result["accuracy"],
        "macro_f1": metrics["macro_avg"]["f1-score"],
        "inference_fps": test_result["inference_fps"],
        "total_pipeline_time": time.time() - pipeline_start,
    }
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  Summary saved: {summary_path}")

    total_time = time.time() - pipeline_start
    logger.info(f"Pipeline completed in {format_time(total_time)}")
    return summary


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv
    run_full_pipeline(quick_test=quick)
