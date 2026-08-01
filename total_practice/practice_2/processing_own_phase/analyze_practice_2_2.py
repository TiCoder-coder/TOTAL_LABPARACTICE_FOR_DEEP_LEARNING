"""Validation-only overfitting analysis for Practice 2.2 controlled runs."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-practice2-2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def classify_generalization_gap(gap_percentage_points):
    if gap_percentage_points >= 12.0:
        return "high_overfitting_risk"
    if gap_percentage_points >= 6.0:
        return "moderate_overfitting_risk"
    if gap_percentage_points <= -6.0:
        return "likely_underfitting_or_augmented_train_is_harder"
    return "controlled_gap"


def build_overfitting_assessment(history_path, comparison_path, selection_path):
    histories = json.loads(Path(history_path).read_text())
    comparison = pd.read_csv(comparison_path)
    selection = json.loads(Path(selection_path).read_text())
    if selection.get("test_access_count") != 0:
        raise RuntimeError("Test access is forbidden in overfitting analysis")

    rows = []
    for row in comparison.to_dict(orient="records"):
        experiment = row["experiment"]
        history = histories[experiment]
        best_index = int(row["best_epoch"]) - 1
        recent_val_accuracy = history["val_acc"][-3:]
        recent_val_loss = history["val_loss"][-3:]
        gap = float(row["generalization_gap_at_best"])
        rows.append(
            {
                "experiment": experiment,
                "strategy": row["strategy"],
                "best_epoch": int(row["best_epoch"]),
                "epochs_trained": int(row["epochs_trained"]),
                "train_accuracy_at_best": float(row["train_accuracy_at_best"]),
                "best_val_accuracy": float(row["best_val_accuracy"]),
                "best_val_loss": float(row["best_val_loss"]),
                "best_val_macro_f1": float(row["best_val_macro_f1"]),
                "generalization_gap_at_best": gap,
                "gap_status": classify_generalization_gap(gap),
                "last_3_val_accuracy_std": float(np.std(recent_val_accuracy)),
                "last_3_val_loss_mean": float(np.mean(recent_val_loss)),
                "val_loss_at_selected_epoch": float(history["val_loss"][best_index]),
                "minimum_val_loss": float(min(history["val_loss"])),
                "minimum_val_loss_epoch": int(
                    np.argmin(history["val_loss"]) + 1
                ),
                "early_stopping_triggered": bool(history["stopped_early"]),
            }
        )

    selected_name = selection["selected_experiment"]
    selected = next(row for row in rows if row["experiment"] == selected_name)
    selected_has_high_gap = selected["gap_status"] == "high_overfitting_risk"
    assessment = {
        "selection_source": "validation_only",
        "selected_experiment": selected_name,
        "selected_checkpoint_status": "best_validation_accuracy_checkpoint",
        "overfitting_status": (
            "significant_overfitting_remains"
            if selected_has_high_gap
            else "overfitting_is_controlled"
        ),
        "final_test_authorized": False,
        "test_loader_constructed": False,
        "test_evaluated": False,
        "test_access_count": 0,
        "interpretation_note": (
            "Training accuracy is measured on online-augmented images, so the gap "
            "is conservative. A large positive gap remains meaningful; a negative "
            "E1 gap can occur because augmented Train samples are harder than the "
            "deterministic Validation samples."
        ),
        "experiments": rows,
    }
    return assessment


def plot_controlled_histories(history_path, output_path):
    histories = json.loads(Path(history_path).read_text())
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"E1_head_only": "#2563eb", "E2_partial_finetune": "#dc2626"}
    for experiment, history in histories.items():
        epochs = history["epoch"]
        color = colors.get(experiment)
        axes[0].plot(
            epochs,
            history["train_loss"],
            color=color,
            alpha=0.65,
            label=f"{experiment} Train",
        )
        axes[0].plot(
            epochs,
            history["val_loss"],
            color=color,
            linestyle="--",
            marker="o",
            label=f"{experiment} Validation",
        )
        axes[1].plot(
            epochs,
            history["train_acc"],
            color=color,
            alpha=0.65,
            label=f"{experiment} Train",
        )
        axes[1].plot(
            epochs,
            history["val_acc"],
            color=color,
            linestyle="--",
            marker="o",
            label=f"{experiment} Validation",
        )
        axes[2].plot(
            epochs,
            history["generalization_gap"],
            color=color,
            marker="o",
            label=experiment,
        )
    axes[0].set(title="Loss Curves", xlabel="Epoch", ylabel="Cross-Entropy Loss")
    axes[1].set(title="Accuracy Curves", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[2].set(
        title="Generalization Gap",
        xlabel="Epoch",
        ylabel="Train − Validation (pp)",
    )
    axes[2].axhline(12, color="#991b1b", linestyle=":", label="High-risk threshold")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_assessment(staging_root):
    root = Path(staging_root)
    output = root / "outputs"
    assessment = build_overfitting_assessment(
        output / "controlled_training_history.json",
        output / "controlled_experiment_comparison.csv",
        output / "controlled_experiment_selection.json",
    )
    (output / "overfitting_assessment.json").write_text(
        json.dumps(assessment, indent=2)
    )
    pd.DataFrame(assessment["experiments"]).to_csv(
        output / "overfitting_assessment.csv",
        index=False,
    )
    plot_controlled_histories(
        output / "controlled_training_history.json",
        root / "reports" / "controlled_training_curves.png",
    )
    return assessment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("staging_root", type=Path)
    args = parser.parse_args()
    assessment = write_assessment(args.staging_root)
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
