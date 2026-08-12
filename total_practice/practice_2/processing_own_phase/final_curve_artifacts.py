"""Generate locked per-class ROC and PR artifacts from exported probabilities."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from configs import CLASS_NAMES, OUTPUT_DIR, REPORTS_DIR


def generate_curve_artifacts(output_dir=OUTPUT_DIR, reports_dir=REPORTS_DIR):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    predictions = pd.read_csv(output_dir / "predictions.csv")
    labels = predictions["true_label_id"].to_numpy(dtype=int)
    scores = predictions[
        [f"probability_{name}" for name in CLASS_NAMES]
    ].to_numpy(dtype=float)
    binary = label_binarize(labels, classes=np.arange(len(CLASS_NAMES)))

    roc_rows, pr_rows, summary = [], [], {"per_class": {}}
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for index, name in enumerate(CLASS_NAMES):
        fpr, tpr, roc_threshold = roc_curve(binary[:, index], scores[:, index])
        precision, recall, pr_threshold = precision_recall_curve(
            binary[:, index], scores[:, index]
        )
        roc_auc = auc(fpr, tpr)
        average_precision = average_precision_score(binary[:, index], scores[:, index])
        summary["per_class"][name] = {
            "roc_auc": float(roc_auc),
            "average_precision": float(average_precision),
        }
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc:.3f})")
        axes[1].plot(recall, precision, label=f"{name} ({average_precision:.3f})")
        roc_rows.extend(
            {"class": name, "fpr": x, "tpr": y, "threshold": threshold}
            for x, y, threshold in zip(fpr, tpr, roc_threshold)
        )
        padded_threshold = np.r_[pr_threshold, np.nan]
        pr_rows.extend(
            {"class": name, "recall": x, "precision": y, "threshold": threshold}
            for x, y, threshold in zip(recall, precision, padded_threshold)
        )

    micro_fpr, micro_tpr, _ = roc_curve(binary.ravel(), scores.ravel())
    micro_precision, micro_recall, _ = precision_recall_curve(binary.ravel(), scores.ravel())
    summary["micro_roc_auc"] = float(auc(micro_fpr, micro_tpr))
    summary["micro_average_precision"] = float(
        average_precision_score(binary, scores, average="micro")
    )
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="One-vs-Rest ROC curves")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="One-vs-Rest Precision–Recall curves")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    for directory in (output_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(directory / "roc_pr_curves_per_class.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(roc_rows).to_csv(output_dir / "roc_curve_per_class.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(output_dir / "pr_curve_per_class.csv", index=False)
    (output_dir / "roc_pr_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(generate_curve_artifacts(), indent=2))
