import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RUNS_ROOT = Path("total_practice/practice_3/runs/practice_3_v2_3")
FIG_DIR = Path("total_practice/practice_3/docs/result/practice_3_v2_3/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

VALID_RUNS = [
    "E1_lr_1e-5",
    "E2_lr_2e-5",
    "E3_lr_3e-5",
    "E4_weight_decay_0.05",
    "E5b_classifier_dropout_0.40",
    "E6c_staged_finetune",
]
WINNER_RUN = "E4_weight_decay_0.05"

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

def load_trainer_state(run_id: str):
    run_dir = RUNS_ROOT / run_id / "checkpoints"
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found for {run_id}")
    last_ckpt = checkpoints[-1]
    state_file = last_ckpt / "trainer_state.json"
    with open(state_file) as f:
        state = json.load(f)
    return state.get("log_history", [])

def extract_metric(log_history, metric_key):
    epochs = []
    values = []
    for entry in log_history:
        if metric_key in entry and "epoch" in entry:
            epochs.append(entry["epoch"])
            values.append(entry[metric_key])
    return epochs, values

def validate_and_print(name, x, y):
    if len(x) == 0 or len(y) == 0:
        raise ValueError(f"Curve '{name}' has zero points!")
    print(f"--- Validation Report for {name} ---")
    print(f"X points: {len(x)}")
    print(f"Y points: {len(y)}")
    print(f"First 3 X values: {x[:3]}")
    print(f"First 3 Y values: {y[:3]}")
    print(f"Last 3 X values: {x[-3:]}")
    print(f"Last 3 Y values: {y[-3:]}")
    print("-" * 40)

def generate_winner_figures():
    print("Generating winner figures for", WINNER_RUN)
    log_history = load_trainer_state(WINNER_RUN)
    
    # 1. winner_train_val_loss.png
    train_epochs, train_losses = extract_metric(log_history, "loss")
    val_epochs, val_losses = extract_metric(log_history, "eval_loss")
    
    validate_and_print("winner_train_loss", train_epochs, train_losses)
    validate_and_print("winner_val_loss", val_epochs, val_losses)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_epochs, train_losses, label="Train Loss", marker='o', color="#2196F3", linewidth=2)
    ax.plot(val_epochs, val_losses, label="Validation Loss", marker='o', color="#F44336", linewidth=2)
    ax.set_title("Training & Validation Loss - E4", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    fig.savefig(FIG_DIR / "winner_train_val_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 2. winner_validation_accuracy.png
    val_epochs, val_acc = extract_metric(log_history, "eval_accuracy")
    validate_and_print("winner_val_acc", val_epochs, val_acc)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(val_epochs, val_acc, label="Validation Accuracy", marker='o', color="#4CAF50", linewidth=2)
    ax.set_title("Validation Accuracy - E4", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    fig.savefig(FIG_DIR / "winner_validation_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 3. winner_validation_f1.png
    val_epochs, val_f1 = extract_metric(log_history, "eval_f1")
    validate_and_print("winner_val_f1", val_epochs, val_f1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(val_epochs, val_f1, label="Validation F1", marker='o', color="#9C27B0", linewidth=2)
    ax.set_title("Validation F1 Score - E4", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    fig.savefig(FIG_DIR / "winner_validation_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 4. winner_learning_rate.png
    lr_epochs, lr_values = extract_metric(log_history, "learning_rate")
    validate_and_print("winner_learning_rate", lr_epochs, lr_values)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lr_epochs, lr_values, label="Learning Rate", marker='o', color="#FF9800", linewidth=2)
    ax.set_title("Learning Rate Schedule - E4", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    fig.savefig(FIG_DIR / "winner_learning_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def generate_experiment_val_loss_curves():
    print("Generating experiment validation loss comparison")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, run_id in enumerate(VALID_RUNS):
        log_history = load_trainer_state(run_id)
        val_epochs, val_losses = extract_metric(log_history, "eval_loss")
        validate_and_print(f"{run_id}_val_loss", val_epochs, val_losses)
        
        ax.plot(val_epochs, val_losses,
                label=run_id,
                color=COLORS[i],
                marker=MARKERS[i],
                linewidth=2,
                markersize=6)
                
    ax.set_title("Validation Loss Curves — All Experiments", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    fig.savefig(FIG_DIR / "experiment_val_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    generate_winner_figures()
    generate_experiment_val_loss_curves()
    print("ALL FIGURES REGENERATED SUCCESSFULLY")
