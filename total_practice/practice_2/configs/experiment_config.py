"""Canonical controlled configuration for the two approved experiments."""

# E1 and E2 must share every value below. The single controlled variable is
# ``training_mode``, declared separately in EXPERIMENTS.
CONTROLLED_BASE_CONFIG = {
    "model_name": "resnet18",
    "optimizer": "AdamW",
    "scheduler": "WarmupCosine",
    "learning_rate": 0.001,
    "head_learning_rate": 0.001,
    "backbone_learning_rate": 0.0001,
    "batch_size": 32,
    "epochs": 15,
    "weight_decay": 2e-4,
    "image_size": 224,
    "seed": 42,
    "early_stopping_patience": 3,
    "best_model_metric": "val_acc",
    "early_stopping_metric": "val_loss",
    "scheduler_metric": "val_loss",
    "label_smoothing": 0.05,
    "dropout": 0.20,
    "random_erasing_probability": 0.25,
    "grad_clip": 1.0,
    "num_classes": 10,
    "selection_source": "validation_only",
    "test_data_used": False,

    # ===== Anti-overfitting improvements =====
    # Stronger random augmentation is applied to Train only; Val/Test are
    # deterministic and unchanged.
    "augment_strength": "strong",

    # Mixup/CutMix on the batch.
    "mixup_alpha": 0.20,
    "cutmix_alpha": 1.00,
    "cutmix_prob": 0.50,

    # Exponential Moving Average of model weights. EMA model is used for
    # validation-based selection and final reporting.
    "use_ema": True,
    "ema_decay": 0.999,

    # WarmupCosine scheduler: linear warmup, then cosine decay to
    # min_lr_ratio * base_lr.
    "warmup_epochs": 2,
    "min_lr_ratio": 0.01,

    # ``report_metric`` lets us pick what we monitor and report.
    # ``val_acc`` keeps the current contract; ``val_macro_f1`` is a more
    # balanced alternative when the dataset is mildly imbalanced.
    "report_metric": "val_acc",
}

EXPERIMENTS = {
    "E1_resnet18_head": {
        **CONTROLLED_BASE_CONFIG,
        "training_mode": "head_only",
    },
    "E2_resnet18_partial": {
        **CONTROLLED_BASE_CONFIG,
        "training_mode": "partial_finetune",
    },
}
