"""Live Matplotlib dashboard for one or more Practice 2 training runs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt


def read_metrics(path: Path):
    """Read all complete JSONL rows and ignore a partial final write."""
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def newest_metrics_file(runs_dir: Path) -> Path | None:
    candidates = list(runs_dir.glob("**/metrics.jsonl"))
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def draw_dashboard(axes, rows, title):
    for axis in axes.flat:
        axis.clear()
        axis.grid(alpha=0.25)
    epochs = [row["epoch"] for row in rows]
    if not epochs:
        axes[0, 0].set_title("Waiting for the first epoch...")
        return

    best_loss = min(rows, key=lambda row: (row["val_loss"], -row["val_accuracy"]))
    best_acc = max(rows, key=lambda row: (row["val_accuracy"], -row["val_loss"]))

    axes[0, 0].plot(epochs, [row["train_loss"] for row in rows], label="Train")
    axes[0, 0].plot(epochs, [row["val_loss"] for row in rows], label="Validation")
    axes[0, 0].scatter(best_loss["epoch"], best_loss["val_loss"], c="red", zorder=5,
                       label=f"Min val loss: e{best_loss['epoch']}")
    axes[0, 0].set_title("Loss")

    axes[0, 1].plot(epochs, [row["train_accuracy"] for row in rows], label="Train")
    axes[0, 1].plot(epochs, [row["val_accuracy"] for row in rows], label="Validation")
    axes[0, 1].scatter(best_acc["epoch"], best_acc["val_accuracy"], c="blue", zorder=5,
                       label=f"Max val acc: e{best_acc['epoch']}")
    axes[0, 1].set_title("Accuracy (%)")

    axes[1, 0].plot(epochs, [100 * row["val_macro_f1"] for row in rows], label="Val macro F1 (%)")
    axes[1, 0].plot(epochs, [row["generalization_gap"] for row in rows], label="Generalization gap")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Validation quality / overfitting")

    lr_names = sorted({name for row in rows for name in row["learning_rates"]})
    for name in lr_names:
        axes[1, 1].plot(
            epochs,
            [row["learning_rates"].get(name, float("nan")) for row in rows],
            label=name,
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Learning rate (log scale)")

    for axis in axes.flat:
        axis.axvline(epochs[-1], color="gray", linestyle="--", alpha=0.6)
        axis.set_xlabel("Epoch")
        axis.legend(fontsize=8)
    status = (
        f"current=e{epochs[-1]} | min val loss={best_loss['val_loss']:.4f} "
        f"at e{best_loss['epoch']} | max val acc={best_acc['val_accuracy']:.2f}% "
        f"at e{best_acc['epoch']}"
    )
    axes[0, 0].figure.suptitle(f"{title}\n{status}")


def monitor(runs_dir: Path, metrics_file: Path | None, refresh_seconds: float):
    plt.ion()
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    try:
        while plt.fignum_exists(figure.number):
            active = metrics_file or newest_metrics_file(runs_dir)
            rows = read_metrics(active) if active else []
            draw_dashboard(axes, rows, str(active or runs_dir))
            figure.canvas.draw_idle()
            plt.pause(refresh_seconds)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--metrics-file", type=Path)
    parser.add_argument("--refresh-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.refresh_seconds <= 0:
        parser.error("--refresh-seconds must be positive")
    monitor(args.runs_dir, args.metrics_file, args.refresh_seconds)


if __name__ == "__main__":
    main()
