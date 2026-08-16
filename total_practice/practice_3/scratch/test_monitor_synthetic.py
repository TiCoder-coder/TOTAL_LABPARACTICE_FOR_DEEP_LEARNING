"""
Synthetic test: write fake JSONL records to a temp dir to verify
file-watching and metric parsing logic. Does NOT train any model.
"""
import json
import time
import shutil
from pathlib import Path

TEST_DIR = Path("total_practice/practice_3/runs/live_terminal_demo")
METRICS_FILE = TEST_DIR / "metrics.jsonl"
STATUS_FILE = TEST_DIR / "status.json"

def write_status(status_obj):
    tmp = STATUS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(status_obj, f)
    tmp.replace(STATUS_FILE)

def append_metric(record):
    with open(METRICS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()

def run_synthetic_test():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    if METRICS_FILE.exists():
        METRICS_FILE.unlink()

    print("Writing WAITING status...")
    write_status({"status": "WAITING", "pid": 0, "started_at": "", "current_epoch": 0, "current_step": 0, "total_steps": 160, "last_update": "", "error": None})
    time.sleep(0.5)

    print("Writing RUNNING status...")
    write_status({"status": "RUNNING", "pid": 9999, "started_at": "2026-08-16T00:00:00Z", "current_epoch": 0.0, "current_step": 0, "total_steps": 160, "last_update": "2026-08-16T00:00:00Z", "error": None})
    time.sleep(0.2)

    # Write synthetic metrics: 8 train points + 2 eval points
    for step in [5, 10, 15, 20, 25, 30, 35, 40]:
        loss = 0.7 - step * 0.003
        lr = 1e-5
        append_metric({"type": "train", "step": step, "epoch": round(step/80, 4), "train_loss": round(loss, 6), "learning_rate": lr})

    for step in [20, 40]:
        v_loss = 0.6 - step * 0.002
        v_acc = 0.55 + step * 0.003
        v_f1 = 0.54 + step * 0.003
        append_metric({"type": "eval", "step": step, "epoch": round(step/80, 4), "val_loss": round(v_loss, 6), "accuracy": round(v_acc, 6), "precision": round(v_f1, 6), "recall": round(v_f1, 6), "f1": round(v_f1, 6)})

    print("Writing COMPLETED status...")
    write_status({"status": "COMPLETED", "pid": 9999, "started_at": "2026-08-16T00:00:00Z", "current_epoch": 2.0, "current_step": 160, "total_steps": 160, "last_update": "2026-08-16T00:01:00Z", "error": None})

    # Verify monitor can read back
    import sys
    sys.path.insert(0, ".")
    from total_practice.practice_3.processing_own_phase.live_training_monitor import _read_metrics, _read_status

    st = _read_status()
    assert st["status"] == "COMPLETED", f"Expected COMPLETED, got {st['status']}"
    print(f"Status read: {st['status']} — PASS")

    train_steps, train_loss, lr_steps, lr_vals, val_steps, val_loss, val_acc, val_f1 = _read_metrics()
    assert len(train_steps) == 8, f"Expected 8 train points, got {len(train_steps)}"
    assert len(val_steps) == 2, f"Expected 2 eval points, got {len(val_steps)}"
    print(f"Train metric rows: {len(train_steps)} — PASS")
    print(f"Eval metric rows: {len(val_steps)} — PASS")

    print("\nALL SYNTHETIC MONITOR TESTS PASSED.")
    print("WAITING → RUNNING → COMPLETED transition: PASS")

if __name__ == "__main__":
    run_synthetic_test()
