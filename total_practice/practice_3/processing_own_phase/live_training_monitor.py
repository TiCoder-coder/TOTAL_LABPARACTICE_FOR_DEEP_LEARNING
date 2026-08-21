import json
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# Absolute path derived from THIS file — never depends on CWD
PRACTICE3_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PRACTICE3_ROOT / "runs" / "live_terminal_demo"
METRICS_FILE = DEMO_DIR / "metrics.jsonl"
STATUS_FILE = DEMO_DIR / "status.json"
SUMMARY_FILE = DEMO_DIR / "final_summary.json"

def _read_status():
    if not STATUS_FILE.exists():
        return {"status": "WAITING"}
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except Exception:
        return {"status": "WAITING"}


def _read_metrics():
    train_steps, train_loss, lr_steps, lr_vals = [], [], [], []
    val_steps, val_loss, val_acc, val_f1 = [], [], [], []
    if not METRICS_FILE.exists():
        return train_steps, train_loss, lr_steps, lr_vals, val_steps, val_loss, val_acc, val_f1
    try:
        with open(METRICS_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "train":
                    train_steps.append(rec["step"])
                    train_loss.append(rec["train_loss"])
                    lr_steps.append(rec["step"])
                    lr_vals.append(rec["learning_rate"])
                elif rec.get("type") == "eval":
                    val_steps.append(rec["step"])
                    val_loss.append(rec["val_loss"])
                    val_acc.append(rec["accuracy"])
                    val_f1.append(rec["f1"])
    except Exception:
        pass
    return train_steps, train_loss, lr_steps, lr_vals, val_steps, val_loss, val_acc, val_f1


def _build_status_md(st, train_loss, val_loss, val_acc, val_f1, lr_vals):
    status_str = st.get("status", "WAITING")
    epoch = st.get("current_epoch", "–")
    step = st.get("current_step", "–")
    total = st.get("total_steps", "–")
    last_upd = st.get("last_update", "–")
    color_map = {"WAITING": "🟡", "RUNNING": "🟢", "COMPLETED": "✅", "FAILED": "🔴"}
    icon = color_map.get(status_str, "⚪")
    t_loss = f"{train_loss[-1]:.4f}" if train_loss else "–"
    v_loss = f"{val_loss[-1]:.4f}" if val_loss else "–"
    v_acc  = f"{val_acc[-1]:.4f}"  if val_acc  else "–"
    v_f1   = f"{val_f1[-1]:.4f}"   if val_f1   else "–"
    lr     = f"{lr_vals[-1]:.2e}"  if lr_vals  else "–"
    return (
        f"{icon} **Training Process:** `{status_str}` &nbsp;|&nbsp; "
        f"**Epoch:** {epoch} &nbsp;|&nbsp; **Step:** {step} / {total}  \n"
        f"**Train Loss:** {t_loss} &nbsp;|&nbsp; **Val Loss:** {v_loss} &nbsp;|&nbsp; "
        f"**Accuracy:** {v_acc} &nbsp;|&nbsp; **F1:** {v_f1} &nbsp;|&nbsp; **LR:** {lr}  \n"
        f"*Last update: {last_upd}*"
    )


def _draw_dashboard(axs, train_steps, train_loss, lr_steps, lr_vals,
                    val_steps, val_loss, val_acc, val_f1):
    for ax in axs.flat:
        ax.clear()

    axs[0, 0].set_title("Loss", fontsize=11, fontweight="bold")
    if train_steps:
        axs[0, 0].plot(train_steps, train_loss, label="Train Loss (window avg)",
                       color="#2196F3", linewidth=2)
    if val_steps:
        axs[0, 0].plot(val_steps, val_loss, label="Demo Val Loss",
                       color="#F44336", linewidth=2, marker="o", markersize=5)
    axs[0, 0].set_xlabel("Step")
    axs[0, 0].set_ylabel("Cross Entropy")
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].grid(True, alpha=0.4)

    axs[0, 1].set_title("Demo Validation Accuracy", fontsize=11, fontweight="bold")
    if val_steps:
        axs[0, 1].plot(val_steps, val_acc, color="#4CAF50", linewidth=2, marker="o", markersize=5)
    axs[0, 1].set_xlabel("Step")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].grid(True, alpha=0.4)

    axs[1, 0].set_title("Demo Validation F1 Score", fontsize=11, fontweight="bold")
    if val_steps:
        axs[1, 0].plot(val_steps, val_f1, color="#9C27B0", linewidth=2, marker="o", markersize=5)
    axs[1, 0].set_xlabel("Step")
    axs[1, 0].set_ylabel("Macro F1")
    axs[1, 0].grid(True, alpha=0.4)

    axs[1, 1].set_title("Learning Rate", fontsize=11, fontweight="bold")
    if lr_steps:
        axs[1, 1].plot(lr_steps, lr_vals, color="#FF9800", linewidth=2)
    axs[1, 1].set_xlabel("Step")
    axs[1, 1].set_ylabel("LR")
    axs[1, 1].grid(True, alpha=0.4)

    plt.tight_layout()


def monitor_live_training():
    """Render one saved-state snapshot and return immediately."""
    st = _read_status()
    values = _read_metrics()
    train_steps, train_loss, lr_steps, lr_vals, val_steps, val_loss, val_acc, val_f1 = values
    if st.get("status") != "RUNNING":
        display(Markdown("**No active training session.**"))
    else:
        display(Markdown(_build_status_md(st, train_loss, val_loss, val_acc, val_f1, lr_vals)))
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("SAVED TRAINING SNAPSHOT — NOT USED FOR OFFICIAL EVALUATION",
                 fontsize=9, color="gray")
    _draw_dashboard(axs, *values)
    display(fig)
    plt.close(fig)
    return {"status": st, "metric_records": len(train_steps) + len(val_steps)}
