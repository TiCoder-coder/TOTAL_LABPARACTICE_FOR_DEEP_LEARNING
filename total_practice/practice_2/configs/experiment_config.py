"""Experiment configurations for transfer learning."""

# E1 and E2 form the controlled comparison used for model selection. Five
# epochs are used because the available environment is CPU-only and historical
# timings are approximately 3-6 minutes per epoch. This is the minimum
# multi-epoch budget approved for the assignment and is never reduced to a
# one-epoch quick run.
CONTROLLED_BASE_CONFIG = {
    "model_name": "resnet18",
    "optimizer": "Adam",
    "scheduler": "ReduceLROnPlateau",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 5,
    "weight_decay": 1e-4,
    "image_size": 224,
    "seed": 42,
    "early_stopping_patience": 3,
    "best_model_metric": "val_acc",
}

EXPERIMENTS = {
    "E1_resnet18_head": {
        **CONTROLLED_BASE_CONFIG,
        "training_mode": "head_only",
        "description": "ResNet18 Transfer Learning: Freeze backbone, train only the classification head.",
    },
    "E2_resnet18_partial": {
        **CONTROLLED_BASE_CONFIG,
        "training_mode": "partial_finetune",
        "description": "ResNet18 Partial Fine Tuning: Unfreeze last block and classifier.",
    },
    "E3_vgg16_head": {
        "model_name": "vgg16",
        "training_mode": "head_only",
        "optimizer": "Adam",
        "scheduler": "StepLR",
        "learning_rate": 0.001,
        "batch_size": 32, # Smaller batch size for VGG
        "epochs": 5,
        "weight_decay": 1e-4,
        "image_size": 224,
        "seed": 42,
        "description": "VGG16 Transfer Learning.",
    },
    "E4_mobilenet_partial": {
        "model_name": "mobilenet_v3_small",
        "training_mode": "partial_finetune",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "learning_rate": 5e-4,
        "batch_size": 64,
        "epochs": 5,
        "weight_decay": 1e-4,
        "image_size": 224,
        "seed": 42,
        "description": "MobileNetV3 Small Partial Fine Tuning.",
    }
}
