"""Build the read-only canonical Practice 2.2 presentation notebook."""

import json
from pathlib import Path

_cell_number = 0


def next_id():
    global _cell_number
    _cell_number += 1
    return f"canonical-{_cell_number:02d}"


def markdown(text):
    return {"cell_type": "markdown", "id": next_id(), "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "id": next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


cells = [
    markdown("""# Practice 2.2 — Canonical Cosmetic Product Classification

**Official artifact-only presentation.** This notebook never trains, creates a Test DataLoader, or repeats Final Test. E2 was frozen before the one-time Test evaluation. All displayed results are loaded read-only from the canonical lineage."""),
    markdown("""## Phase 1 - Problem Definition

Classify cosmetic product images into 10 classes using ImageNet-pretrained ResNet18. Model selection is Validation-only; Final Test is a one-time, post-freeze estimate."""),
    markdown("""## Phase 2 - Environment Setup

This setup resolves the repository paths and loads persisted canonical artifacts. It performs no training and does not construct a Test DataLoader."""),
    code("""import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, Markdown, display

cwd = Path.cwd().resolve()
candidates = [cwd, *cwd.parents]
repo_root = next((p for p in candidates if (p / "total_practice" / "practice_2_2").exists()), None)
if repo_root is None:
    raise FileNotFoundError("Open this notebook from the repository containing total_practice/practice_2_2")

data_root = repo_root / "total_practice" / "practice_2_2"
canonical_root = data_root / "outputs" / "practice_2_2" / "canonical_26dc4625_52aaf974_s42_v1"
split_root = data_root / "outputs" / "practice_2_2" / "canonical_split"
e3_root = data_root / "outputs" / "practice_2_2" / "canonical_26dc4625_52aaf974_s42_phase25_e3_layer4_1_v1"
e4_root = data_root / "outputs" / "practice_2_2" / "canonical_26dc4625_52aaf974_s42_phase26_e4_augmentation_v1"
final_root = canonical_root / "final_test"

required = [
    split_root / "split_manifest.csv", split_root / "split_summary.json",
    canonical_root / "config_snapshot.json", canonical_root / "training_histories.json",
    canonical_root / "validation_comparison.csv", canonical_root / "lineage_manifest.json",
    e3_root / "phase_2_5_report.json", e4_root / "phase_2_6_report.json",
    e3_root / "training_history_e3.json", e4_root / "training_history_e4.json",
    final_root / "final_selection.json", final_root / "final_test_summary.json",
    final_root / "final_test_metrics_by_class.csv", final_root / "final_test_confusion_matrix.csv",
    final_root / "final_test_lineage.json", final_root / "pre_test_verification.json",
    final_root / "FINAL_TEST_COMPLETED.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Canonical artifacts are missing; do not regenerate Final Test. Missing:\\n" + "\\n".join(missing))

manifest = pd.read_csv(split_root / "split_manifest.csv")
split_summary = json.loads((split_root / "split_summary.json").read_text())
config = json.loads((canonical_root / "config_snapshot.json").read_text())
histories = json.loads((canonical_root / "training_histories.json").read_text())
lineage = json.loads((canonical_root / "lineage_manifest.json").read_text())
e3_report = json.loads((e3_root / "phase_2_5_report.json").read_text())
e4_report = json.loads((e4_root / "phase_2_6_report.json").read_text())
e3_history = json.loads((e3_root / "training_history_e3.json").read_text())
e4_history = json.loads((e4_root / "training_history_e4.json").read_text())
selection = json.loads((final_root / "final_selection.json").read_text())
final_summary = json.loads((final_root / "final_test_summary.json").read_text())
final_lineage = json.loads((final_root / "final_test_lineage.json").read_text())
pre_test_gate = json.loads((final_root / "pre_test_verification.json").read_text())
guard = json.loads((final_root / "FINAL_TEST_COMPLETED.json").read_text())
assert guard["FINAL_TEST_COMPLETED"] and guard["final_test_evaluation_count"] == 1
assert selection["test_used_for_selection"] is False
display(Markdown("**Artifact gate: PASS — canonical files exist and Final Test completion guard is locked.**"))"""),
    markdown("""## Phase 3 - Data Loading

### Dataset provenance

- Source: Tiki.vn API.
- Raw images: approximately 2,922.
- Final balanced files: 3,202.
- Per-image source URLs were not retained; complete raw reconstruction is a limitation."""),
    markdown("""### Cleaning pipeline

Historical processing removed corrupt/small images, exact duplicates and low-quality content, then resized to 224×224. Balancing produced 306 generated derivatives. Canonical training excludes all generated files and quarantines 2 suspicious cross-class images."""),
    code("""provenance_table = pd.DataFrame([
    ["Source", "Tiki.vn API"],
    ["Raw images", "2,922"],
    ["Final balanced files", "3,202"],
    ["Model-use originals", "2,894"],
    ["Offline-generated derivatives excluded", "306"],
    ["Cross-class suspicious images quarantined", "2"],
    ["Exact duplicate groups remaining", "0"],
    ["Per-image source URL retained", "No — documented limitation"],
], columns=["Data item", "Canonical value"])
display(provenance_table.style.hide(axis="index").set_caption("Dataset provenance and cleaning summary"))"""),
    markdown("""### Canonical duplicate-aware split"""),
    code("""model_use = manifest.loc[manifest["use_for_model"]]
split_counts = model_use["split"].value_counts().reindex(["Train", "Validation", "Test"])
display(split_counts.rename("model_use_originals").to_frame())
display(pd.DataFrame({
    "dataset_fingerprint": [split_summary["dataset_fingerprint_sha256"]],
    "split_fingerprint": [split_summary["split_fingerprint_sha256"]],
    "generated_excluded": [int(manifest["is_generated"].sum())],
    "quarantine": [int(manifest["quarantined"].sum())],
}))
assert split_counts.to_dict() == {"Train": 2016, "Validation": 438, "Test": 440}"""),
    code("""leakage_table = pd.DataFrame([
    ["Train/Validation/Test disjoint", "PASS"],
    ["Duplicate clusters do not cross splits", "PASS"],
    ["Generated families do not cross splits", "PASS"],
    ["Exact hashes do not cross splits", "PASS"],
    ["Generated derivatives used by model", "0"],
    ["Quarantined images used by model", "0"],
], columns=["Leakage control", "Status"])
display(leakage_table.style.hide(axis="index").set_caption("Canonical leakage-safety assertions"))"""),
    markdown("""## Phase 4 - Exploratory Data Analysis"""),
    code("""class_split = pd.crosstab(model_use["class_name"], model_use["split"]).reindex(columns=["Train", "Validation", "Test"])
display(class_split)
class_split.plot(kind="bar", figsize=(12, 5), title="Canonical model-use originals by class and split")
plt.ylabel("Images")
plt.tight_layout()
plt.show()"""),
    markdown("""## Phase 5 - Data Preprocessing

Only Train uses random augmentation. Validation and the one-time Test use deterministic Resize(256), CenterCrop(224), ToTensor and ImageNet normalization."""),
    code("""display(Markdown("### Canonical E2 Train transform"))
print(config["train_transform"])
display(Markdown("### Deterministic Validation transform"))
print(config["validation_transform"])"""),
    code("""transform_audit = pd.DataFrame([
    ["RandomResizedCrop", "Train only", "scale 0.70–1.00"],
    ["HorizontalFlip", "Train only", "p=0.50"],
    ["ColorJitter", "Train only", "0.20/0.20/0.20/0.05"],
    ["RandomErasing", "Train only", "p=0.10"],
    ["Resize + CenterCrop", "Validation/Test", "deterministic 256→224"],
    ["ImageNet Normalize", "All", "mean/std from pretrained weights"],
], columns=["Transform", "Applied to", "Configuration"])
display(transform_audit.style.hide(axis="index").set_caption("Preprocessing isolation"))"""),
    markdown("""## Phase 6 - Model Building and Baseline

ResNet18 uses ImageNet pretrained initialization. E2 trains the full `layer4` and dropout-regularized classifier head: 8,398,858 trainable parameters out of 11,181,642."""),
    code("""model_table = pd.DataFrame([
    ["Backbone", "ResNet18"],
    ["Initialization", "ImageNet ResNet18_Weights.DEFAULT"],
    ["Classifier", "Dropout(0.20) + Linear(512, 10)"],
    ["E1 trainable", "Classifier only — 5,130 params"],
    ["E2 trainable", "Full layer4 + classifier — 8,398,858 params"],
    ["Total parameters", "11,181,642"],
    ["Frozen BatchNorm", "eval mode; running statistics locked"],
], columns=["Architecture item", "Canonical configuration"])
display(model_table.style.hide(axis="index").set_caption("Model architecture and fine-tuning depth"))"""),
    markdown("""### E1 — head-only baseline

E1 reached 59.87% Train and 59.82% Validation Accuracy. The near-zero gap reflects underfitting, not superior generalization."""),
    markdown("""## Phase 7 - Model Training

### E2 — full layer4 + head

E2 reached 98.31% Train Accuracy, 78.31% Validation Accuracy and 0.7821 Validation Macro F1. It had the strongest Validation performance and was selected."""),
    code("""training_config = pd.DataFrame([
    ["Seed", 42], ["Batch size", 32], ["Maximum epochs", 15],
    ["Optimizer", "AdamW"], ["Head LR", 0.001], ["Backbone LR", 0.0001],
    ["Weight decay", 0.0002], ["Scheduler", "ReduceLROnPlateau"],
    ["Early stopping", "val_loss, patience=3"],
    ["Best checkpoint", "Validation Accuracy"],
    ["Loss", "Class-weighted CrossEntropy + label smoothing 0.05"],
    ["Gradient clipping", 1.0],
], columns=["Training item", "Value"])
display(training_config.style.hide(axis="index").set_caption("Canonical controlled training configuration"))"""),
    markdown("""### Training dynamics and overfitting analysis"""),
    code("""e2_history = pd.DataFrame(histories["E2_partial_finetune"])
display(e2_history[["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "generalization_gap", "lr"]])
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(e2_history["epoch"], e2_history["train_loss"], label="Train")
axes[0].plot(e2_history["epoch"], e2_history["val_loss"], label="Validation")
axes[0].set_title("E2 loss"); axes[0].legend()
axes[1].plot(e2_history["epoch"], e2_history["train_acc"], label="Train")
axes[1].plot(e2_history["epoch"], e2_history["val_acc"], label="Validation")
axes[1].set_title("E2 accuracy (%)"); axes[1].legend()
plt.tight_layout(); plt.show()"""),
    markdown("""## Phase 8 - Controlled Experiments

### E3 — reduced-capacity ablation

E3 trained only `layer4.1 + head`. Validation Accuracy fell to 70.32%, showing insufficient domain adaptation. It was rejected before Test."""),
    markdown("""### E4 — augmentation ablation

E4 returned to full layer4 and used stronger online augmentation. Gap fell to 13.39 points, but Validation Accuracy fell to 72.37%. It was rejected before Test."""),
    markdown("""### Validation-only experiment comparison"""),
    code("""comparison = pd.DataFrame([
    ["E1", "Head only", 59.87, 59.82, 1.3976, 0.5975, 0.05, "Underfit"],
    ["E2", "Full layer4 + head", 98.31, 78.31, 1.0218, 0.7821, 20.00, "Selected"],
    ["E3", "layer4.1 + head", 90.08, 70.32, 1.2799, 0.7017, 19.76, "Rejected"],
    ["E4", "Stronger online augmentation", 85.76, 72.37, 1.2491, 0.7229, 13.39, "Rejected"],
], columns=["Experiment", "Strategy", "Train Acc", "Val Acc", "Val Loss", "Val Macro F1", "Gap", "Decision"])
display(comparison)
assert selection["selected_experiment_id"] == "E2_partial_finetune"
assert abs(selection["validation_accuracy"] - 78.31050228310502) < 1e-12"""),
    code("""all_histories = {
    "E1": pd.DataFrame(histories["E1_head_only"]),
    "E2": pd.DataFrame(histories["E2_partial_finetune"]),
    "E3": pd.DataFrame(e3_history),
    "E4": pd.DataFrame(e4_history),
}
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for name, history in all_histories.items():
    axes[0].plot(history["epoch"], history["val_acc"], marker="o", label=name)
    axes[1].plot(history["epoch"], history["generalization_gap"], marker="o", label=name)
axes[0].set(title="Validation Accuracy by experiment", xlabel="Epoch", ylabel="Accuracy (%)")
axes[1].set(title="Generalization gap", xlabel="Epoch", ylabel="Train − Validation (points)")
for ax in axes: ax.grid(alpha=0.25); ax.legend()
plt.tight_layout(); plt.show()
display(Markdown("**Evidence:** E2 has the highest Validation peak; E3 loses domain adaptation and E4 lowers memorization but also lowers Validation performance."))"""),
    markdown("""## Phase 9 - Model Selection and Checkpoint Verification

### Freeze canonical winner

E2 was frozen before Test using Validation Accuracy as the selection metric. E3/E4 decisions could not be revisited after Test."""),
    code("""display(pd.Series(selection, name="frozen_selection").to_frame())"""),
    code("""checkpoint_table = pd.DataFrame([
    ["Selected experiment", selection["selected_experiment_id"]],
    ["Best epoch", selection["best_epoch"]],
    ["Selection metric", selection["selection_metric"]],
    ["Validation Accuracy", f'{selection["validation_accuracy"]:.4f}%'],
    ["Validation Macro F1", f'{selection["validation_macro_f1"]:.4f}'],
    ["Checkpoint SHA-256", selection["checkpoint_sha256"]],
    ["Config SHA-256", selection["config_fingerprint_sha256"]],
    ["Pre-Test verification gate", pre_test_gate["status"]],
    ["Test used for selection", selection["test_used_for_selection"]],
], columns=["Verification item", "Value"])
display(checkpoint_table.style.hide(axis="index").set_caption("Frozen winner and checkpoint identity"))"""),
    markdown("""## Phase 10 - Final Test Evaluation

Final Test ran exactly once. No retraining or model reselection occurred afterward."""),
    code("""final_table = pd.DataFrame([
    ["Test Loss", final_summary["test_loss"]],
    ["Test Accuracy (%)", final_summary["test_accuracy"]],
    ["Macro Precision", final_summary["macro_precision"]],
    ["Macro Recall", final_summary["macro_recall"]],
    ["Macro F1", final_summary["macro_f1"]],
    ["Weighted F1", final_summary["weighted_f1"]],
    ["Correct", final_summary["correct_predictions"]],
    ["Incorrect", final_summary["incorrect_predictions"]],
    ["Samples", final_summary["total_predictions"]],
], columns=["Metric", "Result"])
display(final_table)
display(Markdown(f"Validation Accuracy 78.31% vs Test Accuracy {final_summary['test_accuracy']:.2f}%: difference **{78.31050228310502-final_summary['test_accuracy']:.2f} percentage points**."))"""),
    markdown("""## Phase 11 - Error Analysis and Visualization

### Per-class metrics and confusion matrix"""),
    code("""per_class = pd.read_csv(final_root / "final_test_metrics_by_class.csv")
confusion = pd.read_csv(final_root / "final_test_confusion_matrix.csv", index_col=0)
display(per_class)
display(confusion)
display(Image(filename=str(final_root / "final_test_confusion_matrix.png")))"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4))
per_class.sort_values("recall").plot.barh(x="class_name", y="recall", ax=axes[0], legend=False, color="#4C78A8")
per_class.sort_values("f1").plot.barh(x="class_name", y="f1", ax=axes[1], legend=False, color="#F58518")
axes[0].set(title="Final Test recall by class", xlim=(0, 1))
axes[1].set(title="Final Test F1 by class", xlim=(0, 1))
plt.tight_layout(); plt.show()"""),
    markdown("""### Confusion pairs and high-confidence errors

Error analysis is reporting-only and was not used for retraining or tuning."""),
    code("""display(Markdown("### Five lowest-recall classes"))
display(pd.DataFrame(final_summary["five_lowest_recall_classes"]))
display(Markdown("### Top confusion pairs"))
display(pd.DataFrame(final_summary["top_confusion_pairs"]))
display(Markdown("### Highest-confidence errors"))
display(pd.DataFrame(final_summary["high_confidence_errors"]).head(10))"""),
    markdown("""## Phase 12 - Save/Load Verification, Limitations and Conclusion

### Reproducibility and save/load verification

- Canonical E2 checkpoint identity is linked by SHA-256.
- Final selection was frozen before Test.
- Model state did not change during inference.
- Final Test completion guard prevents repeated evaluation.

### Limitations

- Per-image Tiki URLs/product IDs were not persisted.
- Raw reconstruction and historical offline balancing are not fully reproducible.
- Visually overlapping packaging and some weakly informative images remain.
- Results use one canonical split and seed rather than cross-validation.
- This learning project is not production-ready."""),
    code("""artifact_hashes = pd.DataFrame(
    list(final_lineage["output_artifact_sha256"].items()),
    columns=["Artifact", "SHA-256"],
)
display(artifact_hashes)
final_audit = pd.DataFrame([
    ["Dataset provenance", "PARTIAL", "Per-image URLs unavailable"],
    ["Cleaning pipeline", "PARTIAL", "Historical balance scripts unavailable"],
    ["Leakage protection", "PASS", "Canonical duplicate-aware split"],
    ["Training pipeline", "PASS", "Frozen config and histories"],
    ["Validation-only selection", "PASS", "E2 frozen before Test"],
    ["Final Test integrity", "PASS", "Exactly one evaluation; guard locked"],
    ["Artifact lineage", "PASS", "Hashes and canonical paths linked"],
    ["Notebook presentation", "PASS", "Artifact-only Run All"],
], columns=["Area", "Status", "Evidence"])
display(final_audit.style.hide(axis="index").set_caption("Final submission audit"))"""),
    markdown("""### Conclusion

E2 was the best Validation-only experiment and achieved **76.36% Final Test Accuracy** and **0.7617 Macro F1** on 440 samples. Leakage controls, checkpoint identity, artifact hashes and the one-time Test guard passed. The project is internally consistent, with provenance/reconstruction limitations disclosed."""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "practice_2_2": {
            "mode": "canonical_report_only",
            "training_allowed": False,
            "final_test_evaluation_allowed": False,
            "canonical_run_id": "canonical_26dc4625_52aaf974_s42_v1",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = Path(__file__).resolve().parents[1] / "notebooks" / "practice_2_2.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(target)
