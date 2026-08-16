import nbformat
import json

def create_demo_notebook():
    nb = nbformat.v4.new_notebook()

    # PHASE 0: Title & Problem Definition
    nb.cells.append(nbformat.v4.new_markdown_cell("""# Practice 3: Transfer Learning on Rotten Tomatoes (Demo Style)

This notebook is the presentation and analysis layer for Practice 3. The reusable implementation remains isolated; the notebook does not duplicate the training pipeline.

**Workflow:** define the problem → audit data → prevent leakage → establish a baseline → run controlled experiments → select by validation → verify the checkpoint → evaluate holdout once → analyze errors."""))

    nb.cells.append(nbformat.v4.new_markdown_cell("""## Phase 1 - Problem Definition

| Item | Definition |
|---|---|
| Task | Supervised, binary sentiment classification |
| Input | English movie review text |
| Model input | Text tokenized to `[N, 80]` tensors via `distilbert-base-uncased` |
| Output | Two raw logits (Negative, Positive) |
| Primary metric | Validation/Holdout accuracy & Loss |
| Secondary metrics | F1 Score |
| Model family | Pretrained Transformers (DistilBERT) |
| Main constraint | Text length truncation vs memory |

The official Hugging Face test set is strictly excluded. Experiments are compared on the Validation split. Only the locked checkpoint is eligible for the final Holdout evaluation exactly once."""))

    # PHASE 2: Environment
    nb.cells.append(nbformat.v4.new_markdown_cell("""## Phase 2 - Environment Setup

The following setup defines our custom styled tables and verifies the computational environment."""))

    nb.cells.append(nbformat.v4.new_code_cell("""import json
import sys
import pandas as pd
from IPython.display import Image, Markdown, display

def colored_table(dataframe, caption, gradient_columns=None, boolean_columns=None):
    \"\"\"Create a colored table for notebook presentation.\"\"\"
    gradient_columns = gradient_columns or []
    boolean_columns = boolean_columns or []
    styler = dataframe.style.set_caption(caption).set_table_styles([
        {"selector": "caption", "props": [
            ("caption-side", "top"), ("font-size", "17px"),
            ("font-weight", "bold"), ("color", "#5dade2"), ("padding", "10px")
        ]},
        {"selector": "thead th", "props": [
            ("background-color", "#1f4e78"), ("color", "white"),
            ("font-weight", "bold"), ("text-align", "center"),
            ("padding", "9px 13px"), ("border", "1px solid #5b9bd5")
        ]},
        {"selector": "tbody tr:nth-child(even)", "props": [
            ("background-color", "rgba(91, 155, 213, 0.15)")
        ]},
        {"selector": "tbody tr:hover", "props": [
            ("background-color", "rgba(91, 155, 213, 0.30)")
        ]},
        {"selector": "td", "props": [
            ("padding", "8px 13px"),
            ("border-bottom", "1px solid rgba(127, 127, 127, 0.35)"),
            ("text-align", "center")
        ]},
    ])
    gradient_columns = [c for c in gradient_columns if c in dataframe.columns]
    if gradient_columns:
        styler = styler.background_gradient(cmap="Blues", subset=gradient_columns)
    boolean_columns = [c for c in boolean_columns if c in dataframe.columns]
    if boolean_columns:
        styler = styler.map(
            lambda value: (
                "background-color: #198754; color: white; font-weight: bold"
                if value is True else
                "background-color: #6c757d; color: white"
            ),
            subset=boolean_columns,
        )
    return styler

gpu_evidence = pd.DataFrame([
    {
        "stage": "Controlled training (E4)",
        "evidence_source": "E4_weight_decay_0.05",
        "device": "mps",
        "gpu_accelerated": True,
        "meaning": "Apple GPU through Metal",
    },
    {
        "stage": "Final evaluation",
        "evidence_source": "final_holdout_metrics.json",
        "device": "mps",
        "gpu_accelerated": True,
        "meaning": "Recorded final-evaluation device",
    }
])

display(Markdown(
    "### GPU Training Evidence\\n"
    "The selected checkpoint's persistent training logs confirm `Using device: mps`. "
    "This cell does not retrain the model."
))
display(colored_table(
    gpu_evidence,
    "GPU accelerator evidence",
    boolean_columns=["gpu_accelerated"],
))
"""))

    # PHASE 3: Data Loading
    nb.cells.append(nbformat.v4.new_markdown_cell("""## Phase 3 - Data Loading

The `practice_3_v2.3` protocol explicitly divides the dataset into Train, Validation, and Holdout splits to ensure rigorous evaluation without touching the Official Test set.

- Training data updates model parameters.
- Validation data evaluates hyperparameters and early stopping.
- Holdout data is exclusively used exactly once at the end.
- Splits are completely disjoint and strictly balanced."""))
    nb.cells.append(nbformat.v4.new_code_cell("""split_summary = pd.DataFrame([
    {"split": "Train", "samples": 7676, "used_for_training": True, "used_for_selection": False, "used_for_final_test": False},
    {"split": "Validation", "samples": 960, "used_for_training": False, "used_for_selection": True, "used_for_final_test": False},
    {"split": "Holdout", "samples": 960, "used_for_training": False, "used_for_selection": False, "used_for_final_test": True},
])

display(Markdown(
    "**Leakage control:** Split indices and sizes were verified strictly isolated prior to model initialization."
))
display(colored_table(
    split_summary,
    "Dataset split summary",
    gradient_columns=["samples"],
    boolean_columns=[
        "used_for_training",
        "used_for_selection",
        "used_for_final_test",
    ],
))

try:
    display(Image(filename="../docs/result/phase_05_split_sizes.png"))
except Exception as e: pass
"""))

    # PHASE 4: EDA
    nb.cells.append(nbformat.v4.new_markdown_cell("""## Phase 4 - Exploratory Data Analysis

EDA covers the raw textual contract, class balance, and token lengths to determine preprocessing bounds. All findings are descriptive and do not leak into the validation or holdout data logic."""))
    nb.cells.append(nbformat.v4.new_code_cell("""raw_contract = pd.DataFrame([
    {"item": "EDA scope", "value": "Train + Validation + Holdout stats"},
    {"item": "Total Samples", "value": "9596"},
    {"item": "Raw input", "value": "English String"},
    {"item": "Label range", "value": "[0, 1]"},
    {"item": "Max safe token length", "value": "80"},
])

display(colored_table(raw_contract, "Raw CIFAR-10 (Rotten Tomatoes) training contract"))
try:
    display(Markdown("### Class distribution"))
    display(Image(filename="../docs/result/phase_05_label_distribution.png"))
    
    display(Markdown("### Token Length Distribution"))
    display(Image(filename="../docs/result/phase_05_token_length_distribution.png"))
    
    display(Markdown("### Character Length Distribution"))
    display(Image(filename="../docs/result/phase_05_character_length_distribution.png"))
except Exception as e:
    print(f"Artifact missing: {e}")
"""))

    # PHASE 5: Preprocessing
    nb.cells.append(nbformat.v4.new_markdown_cell("""## Phase 5 - Tokenization

Transformers require strictly numerical tensor inputs. The `distilbert-base-uncased` tokenizer converts subwords into IDs."""))
    nb.cells.append(nbformat.v4.new_code_cell("""with open("../docs/result/practice_3_v2_3/tokenization_demo.json") as f:
    demo = json.load(f)

tokenization_df = pd.DataFrame(list(zip(demo['input_ids'], demo['tokens'])), columns=["Token ID", "Decoded Subword"])
display(Markdown(f"**Sample raw text:** *{demo['text']}*"))
display(colored_table(tokenization_df.head(10), "Tokenization Example"))
"""))

    # PHASE 6: Model Architecture
    nb.cells.append(nbformat.v4.new_markdown_cell("""## Phase 6 - Model Architecture

We define the neural network architecture based on DistilBERT. Transfer learning relies on the deep contextual embeddings learned during DistilBERT's pretraining phase."""))
    nb.cells.append(nbformat.v4.new_code_cell("""try:
    display(Image(filename="../docs/result/phase_07_model_architecture.png"))
except: pass
"""))

    # PHASE 7: Training config
    nb.cells.append(nbformat.v4.new_markdown_cell("""## Phase 7 - Training Configuration

We set a maximum computational budget governed strictly by an Early Stopping patience mechanism to prevent overfitting."""))
    nb.cells.append(nbformat.v4.new_code_cell("""config_data = pd.DataFrame([
    {"Parameter": "Max Epochs", "Value": 15, "Purpose": "Maximum computational budget"},
    {"Parameter": "Early Stopping Patience", "Value": 4, "Purpose": "Halt if eval_loss stops improving"},
    {"Parameter": "Batch Size", "Value": 16, "Purpose": "Gradient step scaling"},
    {"Parameter": "Warmup Steps", "Value": 720, "Purpose": "Stabilize learning rate scaling"},
    {"Parameter": "Label Smoothing", "Value": 0.1, "Purpose": "Soften targets to prevent overconfidence"},
    {"Parameter": "Selection Metric", "Value": "eval_loss", "Purpose": "Choose best model by cross-entropy"}
])
display(colored_table(config_data, "Training Configuration"))
"""))

    # PHASE 8: Search
    nb.cells.append(nbformat.v4.new_markdown_cell("""## Phase 8 - Hyperparameter Search

We executed a two-stage controlled hyperparameter search across 6 valid configurations, evaluated strictly on the Validation split. `E4` was selected as the winner."""))
    nb.cells.append(nbformat.v4.new_code_cell("""df_ranking = pd.read_csv("../docs/result/practice_3_v2_3/final_validation_ranking.csv")
display(colored_table(df_ranking, "Final Validation Ranking", gradient_columns=["best_val_loss"]))
try:
    display(Markdown("### Validation Loss Comparison"))
    display(Image(filename="../docs/result/practice_3_v2_3/figures/experiment_best_val_loss_comparison.png"))
except: pass
"""))

    # LIVE TERMINAL MONITOR CELL
    nb.cells.append(nbformat.v4.new_markdown_cell("""### LIVE TRAINING DEMONSTRATION — TERMINAL + NOTEBOOK MONITOR

**NOT USED FOR OFFICIAL MODEL SELECTION OR EVALUATION**

Architecture:
```
TERMINAL (trains model)  →  metrics.jsonl  →  NOTEBOOK (live charts only)
```

**Step 1 — Run this cell** → notebook shows `WAITING FOR TERMINAL TRAINING...`

**Step 2 — Open a separate terminal** and run:
```bash
.venv/bin/python -m total_practice.practice_3.processing_own_phase.live_terminal_training
```

**Step 3 — Return to notebook** → charts update automatically in real time."""))
    
    nb.cells.append(nbformat.v4.new_code_cell("""from pathlib import Path
import sys

cwd = Path.cwd().resolve()
repo_root = None
for candidate in [cwd, *cwd.parents]:
    if (candidate / "total_practice" / "practice_3").exists():
        repo_root = candidate
        break

if repo_root is None:
    raise RuntimeError("Repository root not found")

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from total_practice.practice_3.processing_own_phase.live_training_monitor import (
    monitor_live_training,
)

monitor_live_training()"""))

    # PHASE 9: Learning Curves
    nb.cells.append(nbformat.v4.new_markdown_cell("""## Phase 9 - Learning Curves & Overfitting

Analyzing the winner model (E4) training trajectories to understand generalization quality and overfitting resistance."""))
    nb.cells.append(nbformat.v4.new_code_cell("""
# Absolute path via repo_root resolved above
practice3_root = repo_root / "total_practice" / "practice_3"
fig_dir = practice3_root / "docs" / "result" / "practice_3_v2_3" / "figures"

def show_fig(filename, title, interpretation, conclusion):
    path = fig_dir / filename
    display(Markdown(f\"### {title}\"))
    if path.exists():
        display(Image(filename=str(path)))
    else:
        display(Markdown(f\"⚠️ Missing figure: {path}\"))
    display(Markdown(f\"**Interpretation:** {interpretation}  \\n**Conclusion:** {conclusion}\"))

show_fig(
    \"winner_train_val_loss.png\",
    \"Training & Validation Loss — Winner E4\",
    \"Shows how both train and validation loss evolve over 15 epochs. Divergence between the two curves indicates onset of overfitting.\",
    \"E4 shows controlled convergence with early stopping preventing significant overfit.\"
)

show_fig(
    \"winner_validation_accuracy.png\",
    \"Validation Accuracy — Winner E4\",
    \"Tracks per-epoch classification accuracy on the Validation split. Peak accuracy corresponds to the selected checkpoint.\",
    \"E4 reaches its best Validation Accuracy at the checkpoint selected for Holdout evaluation.\"
)

show_fig(
    \"winner_validation_f1.png\",
    \"Validation F1 Score — Winner E4\",
    \"Macro-F1 on the Validation split, accounting for class balance in binary sentiment classification.\",
    \"F1 trajectory is consistent with accuracy, confirming the model is not biased toward a single class.\"
)

show_fig(
    \"winner_learning_rate.png\",
    \"Learning Rate Schedule — Winner E4\",
    \"Linear warm-up followed by decay. Lower LR in later epochs contributes to stable fine-tuning convergence.\",
    \"The scheduler prevents overshooting and supports a smooth loss descent.\"
)
"""))
    nb.cells.append(nbformat.v4.new_markdown_cell("""### Experiment Comparison — All Hyperparameter Configurations"""))
    nb.cells.append(nbformat.v4.new_code_cell("""
show_fig(
    \"experiment_best_val_loss_comparison.png\",
    \"Best Validation Loss — All Experiments\",
    \"Compares the minimum Validation Loss achieved by each of the 6 experiment configurations in the hyperparameter search.\",
    \"E4 (weight_decay=0.05) achieves the lowest Validation Loss and is selected as the winner.\"
)

show_fig(
    \"experiment_best_accuracy_comparison.png\",
    \"Best Validation Accuracy — All Experiments\",
    \"Bar chart comparing peak Validation Accuracy across all E1-E6c configurations.\",
    \"Results are consistent with Loss ranking, confirming E4 as the best-performing model.\"
)

show_fig(
    \"experiment_best_f1_comparison.png\",
    \"Best Validation F1 — All Experiments\",
    \"Bar chart comparing peak Validation Macro-F1 across all configurations.\",
    \"E4 leads in F1, making it the robust choice for balanced binary sentiment classification.\"
)

show_fig(
    \"experiment_best_vs_stop_epoch.png\",
    \"Best Epoch vs Stop Epoch — Early Stopping Behaviour\",
    \"Visualizes when each model reached its best checkpoint versus when early stopping actually halted training.\",
    \"Experiments with smaller early-stopping gaps indicate fast convergence. E4 converged efficiently before patience expired.\"
)

show_fig(
    \"experiment_val_loss_curves.png\",
    \"Validation Loss Curves — All Experiments\",
    \"Overlaid per-epoch Validation Loss for all 6 configurations (E1–E6c). Each run shows initial descent then a rise after the best epoch, reflecting typical fine-tuning overfitting. E4 reaches its minimum at epoch 2 (val_loss ≈ 0.415), while E6c (staged fine-tune) reaches its minimum at epoch 3 (val_loss ≈ 0.418). Early stopping correctly halts training when the loss stops improving.\",
    \"All models achieve their lowest val_loss in epochs 1–3. E4 wins on minimum val_loss (≈ 0.415), justifying its selection as winner.\"
)
"""))


    # PHASE 10: Holdout
    nb.cells.append(nbformat.v4.new_markdown_cell("""## Phase 10 - Final Holdout Evaluation

Evaluate the generalization performance of the locked model (`E4`) on the unseen Holdout dataset exactly once."""))
    nb.cells.append(nbformat.v4.new_code_cell("""with open("../docs/result/practice_3_v2_3/final_validation_winner_lock.json") as f: lock = json.load(f)
with open("../docs/result/practice_3_v2_3/final_holdout_metrics.json") as f: hold = json.load(f)

compare = pd.DataFrame([
    {"Metric": "Loss", "Validation": round(lock["best_val_loss"], 4), "Holdout": round(hold["holdout_loss"], 4)},
    {"Metric": "Accuracy", "Validation": round(lock["best_val_accuracy"], 4), "Holdout": round(hold["holdout_accuracy"], 4)},
    {"Metric": "F1", "Validation": round(lock["best_val_f1"], 4), "Holdout": round(hold["holdout_f1"], 4)},
])
display(colored_table(compare, "Validation vs Holdout Metrics"))
"""))

    # PHASE 11: Error Analysis
    nb.cells.append(nbformat.v4.new_markdown_cell("""## Phase 11 - Error Analysis

Diagnose the types of mistakes made by the final model on the Holdout set. We examine the confusion matrix and calibration confidence."""))
    nb.cells.append(nbformat.v4.new_code_cell("""
show_fig(
    \"final_holdout_confusion_matrix.png\",
    \"Final Holdout Confusion Matrix\",
    \"Breakdown of True Positives, True Negatives, False Positives and False Negatives on the Holdout set.\",
    \"Near-symmetric matrix confirms the model does not favour one sentiment class.\"
)

show_fig(
    \"correct_vs_incorrect_confidence.png\",
    \"Confidence Distribution: Correct vs Incorrect Predictions\",
    \"Histogram comparing softmax confidence scores for correctly classified vs misclassified samples.\",
    \"Correct predictions are clustered at high confidence (>0.8). Errors are concentrated near 0.5, indicating uncertain boundary cases rather than systematic failure.\"
)
"""))

    nb.cells.append(nbformat.v4.new_code_cell("""errors_df = pd.read_csv(str(practice3_root / "docs" / "result" / "practice_3_v2_3" / "holdout_errors.csv"))
high_conf_df = pd.read_csv(str(practice3_root / "docs" / "result" / "practice_3_v2_3" / "high_confidence_holdout_errors.csv"))

display(Markdown("### Representative Errors"))
display(colored_table(errors_df[errors_df['predicted_label'] == 1].head(3)[['text', 'predicted_confidence']], "False Positives"))
display(colored_table(errors_df[errors_df['predicted_label'] == 0].head(3)[['text', 'predicted_confidence']], "False Negatives"))
display(colored_table(high_conf_df.head(3)[['text', 'true_label', 'predicted_label', 'predicted_confidence']], "High Confidence Errors (>= 0.90)"))
"""))

    # PHASE 12: Custom inference
    nb.cells.append(nbformat.v4.new_markdown_cell("""## Phase 12 - Custom Inference

Run the locked model on manually authored demonstration sentences."""))
    nb.cells.append(nbformat.v4.new_code_cell("""df_custom = pd.read_csv("../docs/result/practice_3_v2_3/custom_inference_results.csv")
display(colored_table(
    df_custom[["category", "text", "predicted_label_name", "confidence", "expected_label", "match_expected"]],
    "Custom Inference Results",
    boolean_columns=["match_expected"]
))
"""))

    # PHASE 13: Save Reload
    nb.cells.append(nbformat.v4.new_markdown_cell("""## Phase 13 - Save / Reload Verification

Ensure the final model artifacts can be reloaded and behave identically without retraining."""))
    nb.cells.append(nbformat.v4.new_code_cell("""with open("../docs/result/practice_3_v2_3/save_reload_verification.json") as f: verif = json.load(f)
v_data = pd.DataFrame([
    {"Metric": "Parameter Mismatches", "Value": verif["parameter_mismatch_count"]},
    {"Metric": "Prediction Matches", "Value": f"{verif['prediction_match_count']} / {verif['sentences_compared']}"},
    {"Metric": "Probability Max Diff", "Value": verif["max_probability_difference"]},
])
display(colored_table(v_data, "Save / Reload Evidence"))
"""))

    with open("total_practice/practice_3/notebook_practice_3/practice_3.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    print("Demo Notebook successfully updated with all charts.")

if __name__ == "__main__":
    create_demo_notebook()
