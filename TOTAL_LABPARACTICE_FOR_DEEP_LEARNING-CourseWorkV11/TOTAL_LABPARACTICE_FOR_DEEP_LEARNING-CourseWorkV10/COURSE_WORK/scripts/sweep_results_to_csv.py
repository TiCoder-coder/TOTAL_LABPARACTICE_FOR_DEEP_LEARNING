"""Convert ``live_sweep_results.jsonl`` to ``results.csv`` for each sweep directory.

The sweep runner (``run_single_condition.py``) appends one record per finished
condition to ``artifacts/sweeps/<sweep>/live_sweep_results.jsonl``. The
materialization code in ``sweep_results.py`` expects a single
``results.csv`` per sweep. This adapter bridges the two so that
``materialize_phase_2X`` can pick up the live results.

Run::

    python3 scripts/sweep_results_to_csv.py            # all sweeps
    python3 scripts/sweep_results_to_csv.py s7_batch_size  # only one sweep
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIRS = {
    "s1_feature_set": "S1_FEATURE_SET",
    "s2_time_feature": "S2_TIME_FEATURE",
    "s3_target_scaling": "S3_TARGET_SCALING",
    "s4_lookback": "S4_LOOKBACK",
    "s5_pooling": "S5_POOLING",
    "s6_activation": "S6_ACTIVATION",
    "s7_batch_size": "S7_BATCH_SIZE",
    "s8_learning_rate": "S8_LEARNING_RATE",
    "S9_weight_decay": "S9_WEIGHT_DECAY",
    "S10_dropout": "S10_DROPOUT",
    "S11_d_model": "S11_D_MODEL",
    "S12_heads": "S12_HEADS",
}

CSV_COLUMNS = [
    "condition",
    "sweep_id",
    "run_id",
    "best_epoch",
    "val_rmse",
    "val_mae",
    "val_r2",
    "feature_variant_id",
    "lookback_steps",
    "target_scaling_option",
    "pooling",
    "activation",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "dropout",
    "d_model",
    "num_heads",
]


def jsonl_to_rows(jsonl_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not jsonl_path.exists():
        return rows
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        cfg = record.get("config", {})
        rows.append({
            "condition": record.get("condition"),
            "sweep_id": record.get("sweep_id"),
            "run_id": record.get("run_id"),
            "best_epoch": record.get("best_epoch"),
            "val_rmse": record.get("best_validation_rmse_wh"),
            "val_mae": record.get("best_validation_mae_wh"),
            "val_r2": record.get("best_validation_r2"),
            "feature_variant_id": cfg.get("feature_variant_id"),
            "lookback_steps": cfg.get("lookback_steps"),
            "target_scaling_option": cfg.get("target_scaling_option"),
            "pooling": cfg.get("pooling"),
            "activation": cfg.get("activation"),
            "batch_size": cfg.get("batch_size"),
            "learning_rate": cfg.get("learning_rate"),
            "weight_decay": cfg.get("weight_decay"),
            "dropout": cfg.get("dropout"),
            "d_model": cfg.get("d_model"),
            "num_heads": cfg.get("num_heads"),
        })
    return rows


def write_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot create sweep results without verified rows: {csv_path}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def convert_one(sweep_dir_name: str) -> tuple[Path, int]:
    sweep_dir = ROOT / "artifacts" / "sweeps" / sweep_dir_name
    jsonl_path = sweep_dir / "live_sweep_results.jsonl"
    canonical_filenames = {
        "S9_weight_decay": "s9_weight_decay_metrics.csv",
        "S10_dropout": "s10_dropout_metrics.csv",
        "S11_d_model": "s11_d_model_metrics.csv",
        "S12_heads": "s12_head_metrics.csv",
    }
    csv_filename = canonical_filenames.get(sweep_dir_name, "results.csv")
    csv_path = sweep_dir / csv_filename
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Sweep source results are missing: {jsonl_path}")
    rows = jsonl_to_rows(jsonl_path)
    write_csv(rows, csv_path)
    return csv_path, len(rows)


def main(argv: list[str]) -> int:
    if argv and argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0
    if argv:
        names = argv
    else:
        names = list(SWEEP_DIRS)
    for name in names:
        if name not in SWEEP_DIRS:
            print(f"  [skip] unknown sweep dir: {name}", flush=True)
            continue
        csv_path, n = convert_one(name)
        print(f"  [ok] {name}: {n} rows -> {csv_path.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
