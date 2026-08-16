"""
Regenerate experiment_val_loss_curves.png from real trainer_state.json artifacts.
Does NOT retrain. Does NOT access Holdout or Official Test.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RUNS_ROOT = Path("total_practice/practice_3/runs/practice_3_v2_3")
OUT_PATH = Path("total_practice/practice_3/docs/result/practice_3_v2_3/figures/experiment_val_loss_curves.png")

VALID_RUNS = [
    "E1_lr_1e-5",
    "E2_lr_2e-5",
    "E3_lr_3e-5",
    "E4_weight_decay_0.05",
    "E5b_classifier_dropout_0.40",
    "E6c_staged_finetune",
]

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

def load_val_loss_history(run_id: str):
    """Load per-epoch eval_loss from the most complete checkpoint trainer_state."""
    run_dir = RUNS_ROOT / run_id / "checkpoints"
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found for {run_id}")
    last_ckpt = checkpoints[-1]
    state_file = last_ckpt / "trainer_state.json"
    with open(state_file) as f:
        state = json.load(f)
    log_history = state.get("log_history", [])
    eval_entries = [e for e in log_history if "eval_loss" in e]
    epochs = [e["epoch"] for e in eval_entries]
    val_losses = [e["eval_loss"] for e in eval_entries]
    return epochs, val_losses

def main():
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, run_id in enumerate(VALID_RUNS):
        epochs, val_losses = load_val_loss_history(run_id)
        
        # Strict validation
        assert len(epochs) > 0, f"{run_id}: zero eval points"
        assert len(val_losses) > 0, f"{run_id}: zero val_loss values"
        assert len(epochs) == len(val_losses), f"{run_id}: epoch/val_loss length mismatch"
        
        print(f"{run_id}: {len(epochs)} eval points | "
              f"epoch range {epochs[0]:.1f}–{epochs[-1]:.1f} | "
              f"val_loss {val_losses[0]:.4f}→{min(val_losses):.4f}(best)→{val_losses[-1]:.4f}")
        
        ax.plot(epochs, val_losses,
                label=run_id,
                color=COLORS[i],
                marker=MARKERS[i],
                linewidth=2,
                markersize=6)

    # Verify axes actually have lines with data
    assert len(ax.lines) == len(VALID_RUNS), f"Expected {len(VALID_RUNS)} lines, got {len(ax.lines)}"
    for line in ax.lines:
        assert len(line.get_xdata()) > 0, "Line has empty x-data"
        assert len(line.get_ydata()) > 0, "Line has empty y-data"
    print(f"\nAxes validation: {len(ax.lines)} lines, all non-empty — PASS")

    ax.set_title("Validation Loss Curves — All Experiments (Practice 3 v2.3)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    file_size = OUT_PATH.stat().st_size
    assert file_size > 10_000, f"Output PNG suspiciously small: {file_size} bytes"
    print(f"\nSaved: {OUT_PATH}")
    print(f"File size: {file_size:,} bytes — PASS")
    print("FIGURE REGENERATION COMPLETE.")

if __name__ == "__main__":
    main()
