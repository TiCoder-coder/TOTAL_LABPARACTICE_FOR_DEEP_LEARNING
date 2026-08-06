from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from practice_2_2.accuracy_pipeline import (
    AccuracyConfig,
    select_validation_inference_strategy,
    train_repeated_seeds,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/new_work/v3_product_visual_group_s42_v1/candidate_split_manifest.csv",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data/final/data_clean_balanced",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/new_work/accuracy_refactor_validation_only_v1",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["resnet18", "efficientnet_b0"],
    )
    parser.add_argument(
        "--label-smoothing",
        nargs="+",
        type=float,
        default=[0.0, 0.05],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--finetune-epochs", type=int, default=18)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--head-learning-rate", type=float, default=5e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--maximum-ensemble-members", type=int, default=3)
    parser.add_argument("--tta-scales", nargs="+", type=float, default=[1.0, 0.95, 0.9])
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    configurations = [
        AccuracyConfig(
            architecture=architecture,
            dropout=arguments.dropout,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
            warmup_epochs=arguments.warmup_epochs,
            finetune_epochs=arguments.finetune_epochs,
            head_learning_rate=arguments.head_learning_rate,
            backbone_learning_rate=arguments.backbone_learning_rate,
            weight_decay=arguments.weight_decay,
            label_smoothing=label_smoothing,
            patience=arguments.patience,
        )
        for architecture in arguments.architectures
        for label_smoothing in arguments.label_smoothing
    ]
    training_summary = train_repeated_seeds(
        arguments.manifest,
        arguments.dataset_root,
        arguments.output_root,
        configurations,
        seeds=arguments.seeds,
    )
    checkpoint_paths = [
        Path(result["checkpoint_path"]) for result in training_summary["results"]
    ]
    inference_selection = select_validation_inference_strategy(
        checkpoint_paths,
        arguments.manifest,
        arguments.dataset_root,
        output_path=arguments.output_root / "validation_inference_selection.json",
        tta_scales=arguments.tta_scales,
        maximum_ensemble_members=arguments.maximum_ensemble_members,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
    )
    result = {
        "selected_configuration_id": training_summary["selected_configuration_id"],
        "selected_inference_strategy": inference_selection["selected"],
        "test_loader_constructed": False,
        "test_evaluated": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
