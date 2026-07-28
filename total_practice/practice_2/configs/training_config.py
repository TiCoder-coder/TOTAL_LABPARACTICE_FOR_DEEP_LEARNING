"""Training and hyperparameter configuration."""

from configs.core_config import DATA_DIR, OUTPUT_DIR, RUNS_DIR, REPORTS_DIR, MODEL_SAVE_PATH, EXPERIMENT_RESULTS_CSV

CONFIG = {
    "seed": 42,
    "batch_size": 64,
    "num_workers": 0,
    "train_split_ratio": 0.9,
    "image_size": 224,
    "epochs": 10,
    
    # Optimizer & Scheduler
    "optimizer": "Adam",  # Adam, SGD, AdamW
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "scheduler": "ReduceLROnPlateau", # ReduceLROnPlateau, StepLR, CosineAnnealingLR
    "grad_clip": 1.0,  # Max norm for gradient clipping
    
    # Early Stopping
    "early_stopping_patience": 3,
    "best_model_metric": "val_acc",  # or macro_f1, val_loss
    
    # Paths mapped from core config for easy access
    "data_dir": str(DATA_DIR),
    "output_dir": str(OUTPUT_DIR),
    "runs_dir": str(RUNS_DIR),
    "reports_dir": str(REPORTS_DIR),
    "model_save_path": str(MODEL_SAVE_PATH),
    "experiment_results_csv": str(EXPERIMENT_RESULTS_CSV),
    "model_name": "resnet18",
}
