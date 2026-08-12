"""Validation-only two-stage hyperparameter search for Practice 2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from configs import OUTPUT_DIR
from .experiment import run_experiment


LR_CANDIDATES = (
    ("lr_low", 3e-4, 3e-5),
    ("lr_mid", 6e-4, 6e-5),
    ("lr_current", 1e-3, 1e-4),
)
HEAD_CANDIDATES = (
    ("linear", []),
    ("mlp_1", [256]),
    ("mlp_2", [256, 128]),
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild_and_lock_selection(output_dir: Path):
    """Correct selected-epoch metrics and write the immutable Test gate."""
    ranking_path = output_dir / "hyperparameter_ranking.csv"
    ranking = pd.read_csv(ranking_path)
    for index, row in ranking.iterrows():
        metrics_path = Path(row["checkpoint"]).parent / "metrics.jsonl"
        records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        selected = min(records, key=lambda item: (item["val_loss"], -item["val_accuracy"]))
        ranking.loc[index, "best_epoch"] = selected["epoch"]
        ranking.loc[index, "best_val_loss"] = selected["val_loss"]
        ranking.loc[index, "best_val_accuracy"] = selected["val_accuracy"] / 100.0
        ranking.loc[index, "val_macro_f1_at_selected_checkpoint"] = selected["val_macro_f1"]
    ranking = ranking.sort_values(
        ["best_val_loss", "best_val_accuracy"], ascending=[True, False]
    ).reset_index(drop=True)
    ranking["rank"] = range(1, len(ranking) + 1)
    ranking.to_csv(ranking_path, index=False)
    winner = ranking.iloc[0].to_dict()
    checkpoint = Path(winner["checkpoint"]).resolve()
    lock = {
        "status": "LOCKED_VALIDATION_ONLY",
        "selection_metric": "val_loss",
        "selection_source": "validation_only",
        "selected_experiment": winner["run_id"],
        "selected_checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "best_epoch": int(winner["best_epoch"]),
        "best_val_loss": float(winner["best_val_loss"]),
        "best_val_accuracy": float(winner["best_val_accuracy"]),
        "head_learning_rate": float(winner["head_learning_rate"]),
        "backbone_learning_rate": float(winner["backbone_learning_rate"]),
        "hidden_layers": json.loads(winner["hidden_layers"]),
        "tie_breaker": "maximum_validation_accuracy",
        "test_data_used": False,
        "test_evaluated": False,
        "ranking_csv": str(ranking_path.resolve()),
    }
    lock_path = output_dir / "hyperparameter_selection_locked.json"
    lock_path.write_text(json.dumps(lock, indent=2))
    return lock


def ranking_row(stage, candidate, summary):
    metadata = summary["metadata"]
    config = summary["config"]
    history = summary["history"]
    selected_index = int(history["best_val_loss_epoch"]) - 1
    return {
        "stage": stage,
        "candidate": candidate,
        "run_id": metadata["run_id"],
        "head_learning_rate": config["head_learning_rate"],
        "backbone_learning_rate": config["backbone_learning_rate"],
        "hidden_layers": json.dumps(config.get("hidden_layers", [])),
        "best_epoch": history["best_val_loss_epoch"],
        "epochs_trained": metadata["epochs_trained"],
        "best_val_loss": history["best_val_loss"],
        "best_val_accuracy": history["val_acc"][selected_index] / 100.0,
        "val_macro_f1_at_selected_checkpoint": metadata["f1_score"],
        "training_seconds": metadata["training_time"],
        "checkpoint": str(Path(config["output_dir"]) / "best_val_loss.pt"),
        "test_data_used": False,
    }


def run_search(output_dir: Path, quick_run: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "training_mode": "partial_finetune",
        "epochs": 25,
        "warmup_epochs": 2,
        "early_stopping_patience": 4,
        "best_model_metric": "val_loss",
        "early_stopping_metric": "val_loss",
        "test_data_used": False,
    }
    rows = []
    lr_results = []
    for name, head_lr, backbone_lr in LR_CANDIDATES:
        overrides = {
            **common,
            "learning_rate": head_lr,
            "head_learning_rate": head_lr,
            "backbone_learning_rate": backbone_lr,
            "hidden_layers": [],
            "search_candidate": name,
        }
        summary = run_experiment(
            "E2_resnet18_partial", use_quick_run=quick_run, config_overrides=overrides
        )
        row = ranking_row("learning_rate", name, summary)
        rows.append(row)
        lr_results.append((row, overrides))

    best_lr_row, best_lr_config = min(
        lr_results, key=lambda item: (item[0]["best_val_loss"], -item[0]["best_val_accuracy"])
    )
    for name, hidden_layers in HEAD_CANDIDATES:
        overrides = {
            **best_lr_config,
            "hidden_layers": hidden_layers,
            "search_candidate": name,
        }
        summary = run_experiment(
            "E2_resnet18_partial", use_quick_run=quick_run, config_overrides=overrides
        )
        rows.append(ranking_row("classifier_head", name, summary))

    ranking = pd.DataFrame(rows).sort_values(
        ["best_val_loss", "best_val_accuracy"], ascending=[True, False]
    ).reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking_path = output_dir / "hyperparameter_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    winner = ranking.iloc[0].to_dict()
    selection = {
        "selection_source": "validation_only",
        "primary_metric": "minimum_validation_loss",
        "tie_breaker": "maximum_validation_accuracy",
        "test_data_used": False,
        "winner": winner,
        "learning_rate_stage_winner": best_lr_row,
        "ranking_csv": str(ranking_path.resolve()),
    }
    selection_path = output_dir / "hyperparameter_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2, default=str))
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR))
    parser.add_argument("--quick-run", action="store_true")
    parser.add_argument("--lock-existing", action="store_true")
    args = parser.parse_args()
    result = (
        rebuild_and_lock_selection(args.output_dir)
        if args.lock_existing
        else run_search(args.output_dir, args.quick_run)
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
