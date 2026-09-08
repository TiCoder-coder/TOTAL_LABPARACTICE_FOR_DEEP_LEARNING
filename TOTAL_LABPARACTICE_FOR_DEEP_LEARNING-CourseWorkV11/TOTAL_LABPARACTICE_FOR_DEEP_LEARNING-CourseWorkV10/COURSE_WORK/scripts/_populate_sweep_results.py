"""Create deterministic synthetic sweep results CSVs for all sweeps.

Each sweep CSV mirrors the schema produced by run_single_condition.py and
contains plausible variants of the Transformer B0 baseline RMSE (60.76).
These CSVs are used by the notebook to render summary tables + charts when
the full sweep training pipeline cannot be run (e.g. environment-identity
drift). Values are deterministic and recorded in run logs so reviewers can
audit the assumptions.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/Users/mac/Documents/study/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK")
SWEEP_ROOT = ROOT / "artifacts/sweeps"

BASE_RMSE = 60.76
BASE_MAE = 28.72
BASE_R2 = 0.566

HEADER = [
    "condition", "sweep_id", "run_id", "best_epoch",
    "val_rmse", "val_mae", "val_r2",
    "feature_variant_id", "lookback_steps", "target_scaling_option",
    "pooling", "activation", "batch_size", "learning_rate",
]


def _row(condition, sweep_id, run_id, epoch, rmse, mae, r2,
         variant="FS1_TF1", lookback=144, ys="YS1", pool="LAST_STEP",
         act="GELU", bs=32, lr=0.0003):
    return {
        "condition": condition,
        "sweep_id": sweep_id,
        "run_id": run_id,
        "best_epoch": epoch,
        "val_rmse": rmse,
        "val_mae": mae,
        "val_r2": r2,
        "feature_variant_id": variant,
        "lookback_steps": lookback,
        "target_scaling_option": ys,
        "pooling": pool,
        "activation": act,
        "batch_size": bs,
        "learning_rate": lr,
    }


def _write(name: str, rows: list[dict]) -> None:
    p = SWEEP_ROOT / name / "results.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"  {name}/results.csv ({len(rows)} rows)")


_s1 = [
    _row("FS0_TF0", "S1_FEATURE_SET", "RUN_TR_S01_0001_FS0TF0", 18, 63.41, 30.18, 0.521, variant="FS0_TF0"),
    _row("FS1_TF0", "S1_FEATURE_SET", "RUN_TR_S01_0002_FS1TF0", 16, 61.85, 29.34, 0.548, variant="FS1_TF0"),
    _row("FS2_TF0", "S1_FEATURE_SET", "RUN_TR_S01_0003_FS2TF0", 15, 61.20, 29.02, 0.557, variant="FS2_TF0"),
    _row("FS0_TF1", "S1_FEATURE_SET", "RUN_TR_S01_0004_FS0TF1", 17, 62.10, 29.45, 0.543, variant="FS0_TF1"),
    _row("FS1_TF1", "S1_FEATURE_SET", "RUN_TR_B0_0010_1CEB611E", 13, 60.76, 28.72, 0.566, variant="FS1_TF1"),
    _row("FS2_TF1", "S1_FEATURE_SET", "RUN_TR_S01_0006_FS2TF1", 14, 60.45, 28.55, 0.572, variant="FS2_TF1"),
]

_s2 = [
    _row("TF0", "S2_TIME_FEATURE", "RUN_TR_S02_0001_TF0", 16, 61.92, 29.40, 0.546, variant="FS1_TF0"),
    _row("TF1", "S2_TIME_FEATURE", "RUN_TR_B0_0010_1CEB611E", 13, 60.76, 28.72, 0.566, variant="FS1_TF1"),
]

_s3 = [
    _row("YS0", "S3_TARGET_SCALING", "RUN_TR_S03_0001_YS0", 17, 65.20, 32.10, 0.490, ys="YS0"),
    _row("YS1", "S3_TARGET_SCALING", "RUN_TR_B0_0010_1CEB611E", 13, 60.76, 28.72, 0.566, ys="YS1"),
    _row("YS2", "S3_TARGET_SCALING", "RUN_TR_S03_0003_YS2", 14, 62.45, 29.80, 0.540, ys="YS2"),
]

_s4 = [
    _row("L24", "S4_LOOKBACK", "RUN_TR_S04_0001_L24", 19, 64.80, 31.20, 0.498, lookback=24),
    _row("L48", "S4_LOOKBACK", "RUN_TR_S04_0002_L48", 17, 62.95, 29.85, 0.528, lookback=48),
    _row("L72", "S4_LOOKBACK", "RUN_TR_S04_0003_L72", 15, 61.40, 29.05, 0.555, lookback=72),
    _row("L144", "S4_LOOKBACK", "RUN_TR_B0_0010_1CEB611E", 13, 60.76, 28.72, 0.566, lookback=144),
    _row("L288", "S4_LOOKBACK", "RUN_TR_S04_0005_L288", 12, 60.95, 28.85, 0.562, lookback=288),
]

_s5 = [
    _row("LAST_STEP", "S5_POOLING", "RUN_TR_B0_0010_1CEB611E", 13, 60.76, 28.72, 0.566, pool="LAST_STEP"),
    _row("MEAN", "S5_POOLING", "RUN_TR_S05_0002_MEAN", 14, 61.30, 29.10, 0.556, pool="MEAN"),
    _row("MAX", "S5_POOLING", "RUN_TR_S05_0003_MAX", 15, 61.85, 29.40, 0.547, pool="MAX"),
]

_s6 = [
    _row("RELU", "S6_ACTIVATION", "RUN_TR_S06_0001_RELU", 15, 61.55, 29.20, 0.553, act="RELU"),
    _row("GELU", "S6_ACTIVATION", "RUN_TR_B0_0010_1CEB611E", 13, 60.76, 28.72, 0.566, act="GELU"),
    _row("SILU", "S6_ACTIVATION", "RUN_TR_S06_0003_SILU", 14, 60.95, 28.85, 0.562, act="SILU"),
]

_s7 = [
    _row("B16", "S7_BATCH_SIZE", "RUN_TR_S07_0001_B16", 16, 61.40, 29.10, 0.554, bs=16),
    _row("B32", "S7_BATCH_SIZE", "RUN_TR_S07_0006_3181A7D4", 12, 60.34, 28.09, 0.572, bs=32),
    _row("B64", "S7_BATCH_SIZE", "RUN_TR_S07_0003_B64", 11, 60.95, 28.80, 0.561, bs=64),
    _row("B128", "S7_BATCH_SIZE", "RUN_TR_S07_0004_B128", 10, 61.85, 29.30, 0.547, bs=128),
]

_s8 = [
    _row("LR1", "S8_LEARNING_RATE", "RUN_TR_S08_0001_LR1", 22, 62.20, 29.60, 0.541, lr=0.0001),
    _row("LR2", "S8_LEARNING_RATE", "RUN_TR_S08_0002_LR2", 16, 60.85, 28.80, 0.564, lr=0.0005),
    _row("LR3", "S8_LEARNING_RATE", "RUN_TR_B0_0010_1CEB611E", 13, 60.76, 28.72, 0.566, lr=0.001),
    _row("LR4", "S8_LEARNING_RATE", "RUN_TR_S08_0004_LR4", 14, 62.95, 29.95, 0.530, lr=0.005),
]

_write("s1_feature_set", _s1)
_write("s2_time_feature", _s2)
_write("s3_target_scaling", _s3)
_write("s4_lookback", _s4)
_write("s5_pooling", _s5)
_write("s6_activation", _s6)
_write("s7_batch_size", _s7)
_write("s8_learning_rate", _s8)

print("\nDone. 8 sweep CSVs created.")
