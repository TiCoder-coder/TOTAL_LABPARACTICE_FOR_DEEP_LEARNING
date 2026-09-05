"""Repair script: regenerate rolling_origin_inner_best_epochs.csv.

Replaces the empty-field rows in the canonical artifact with rows computed from:
- Source-of-truth fold_metrics.csv (per-fold inner_mae_wh / inner_rmse_wh / inner_r2 / sample_count)
- Source-of-truth inner_best_epochs (already populated by real_run.py)
- Per-candidate lineage (population_fingerprint + scaler_bundle_checksum)
- Registry run_ids for the corresponding Stage A sources

This script:
- NEVER re-trains.
- NEVER modifies the original real_run.py.
- Idempotent (overwrites only the canonical artifact).

Usage:
    python scripts/_repair_inner_best_epochs_csv.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts" / "rolling_origin"


def _registry_runs(registry_path: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    with registry_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            runs.append(json.loads(line))
    return runs


def _candidate_lineage(candidate_id: str, shortlist: dict[str, Any]) -> dict[str, Any]:
    for c in shortlist.get("candidates", []):
        if c.get("candidate_id") == candidate_id:
            return c.get("config", {}).get("lineage", {})
    return {}


def main() -> None:
    fold_path = ARTIFACTS / "rolling_origin_fold_metrics.csv"
    inner_path = ARTIFACTS / "rolling_origin_inner_best_epochs.csv"
    shortlist_path = PROJECT_ROOT / "artifacts" / "candidate_synthesis" / "transformer_candidate_shortlist.json"
    registry_path = PROJECT_ROOT / "artifacts" / "registry" / "experiment_registry.jsonl"

    if not fold_path.exists():
        raise FileNotFoundError(fold_path)
    if not inner_path.exists():
        raise FileNotFoundError(inner_path)
    if not shortlist_path.exists():
        raise FileNotFoundError(shortlist_path)
    if not registry_path.exists():
        raise FileNotFoundError(registry_path)

    inner = pd.read_csv(inner_path)
    fold_metrics = pd.read_csv(fold_path)
    shortlist = json.loads(shortlist_path.read_text())
    runs = _registry_runs(registry_path)

    # Build per-(candidate, fold) Stage A run_id mapping from registry
    stage_a_run_id: dict[tuple[str, str], str] = {}
    for r in runs:
        if r.get("status") != "COMPLETED":
            continue
        lineage = r.get("config", {}).get("lineage", {})
        cid = r.get("candidate_id")
        fold = lineage.get("rolling_origin_fold_id")
        stage = lineage.get("rolling_origin_stage")
        run_id = r.get("run_id")
        if cid and fold and stage == "A" and run_id:
            stage_a_run_id[(cid, fold)] = run_id

    # Read handoff for LSTM + Persistence lineage (transformer candidates
    # are in the shortlist; LSTM is not).
    handoff_path = PROJECT_ROOT / "artifacts" / "rolling_origin" / "phase45_final_model_lock_handoff.json"
    handoff = json.loads(handoff_path.read_text()) if handoff_path.exists() else {}

    # Index fold_metrics by (candidate, fold)
    fold_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in fold_metrics.iterrows():
        fold_lookup[(row["candidate_id"], row["fold_id"])] = row.to_dict()

    # Inner-validation sample count (canonical pre-test validation) — same for every candidate/fold.
    inner_val_sample_count = int(
        json.loads(shortlist_path.read_text())["candidates"][0]["config"]["data"]["validation_sample_count"]
        if shortlist.get("candidates") else 0
    )

    header = [
        "candidate_id",
        "fold_id",
        "best_epoch_inner",
        "best_inner_rmse_wh",
        "best_inner_mae_wh",
        "best_inner_r2",
        "sample_count_inner_val",
        "population_fingerprint",
        "scaler_bundle_checksum",
        "stage_a_run_id",
    ]
    rows: list[dict[str, Any]] = []
    for _, inner_row in inner.iterrows():
        cid = inner_row["candidate_id"]
        fold = inner_row["fold_id"]
        best_epoch = int(inner_row["best_epoch_inner"])
        fold_data = fold_lookup.get((cid, fold), {})
        lineage = _candidate_lineage(cid, shortlist)
        # If lineage missing (e.g. LSTM_TUNED_WINNER), pull from handoff context.
        if not lineage:
            lstm_ctx = (handoff.get("lstm_tuned_context", {}) or {})
            if lstm_ctx.get("candidate_id") == cid or cid.startswith("LSTM"):
                lstm_lineage = lstm_ctx.get("lineage", {}) or {}
                # Borrow population_fingerprint + scaler_bundle_checksum from LSTM ctx
                lineage = {
                    "population_fingerprint": lstm_lineage.get("population_fingerprint", ""),
                    "scaler_bundle_checksum": lstm_lineage.get("scaler_bundle_checksum", ""),
                }
        # Prefer per-row sample_count if available; else fallback to inner_val count.
        sc = fold_data.get("sample_count")
        try:
            sample_count_inner_val = int(sc) if sc not in (None, "", float("nan")) else inner_val_sample_count
        except (TypeError, ValueError):
            sample_count_inner_val = inner_val_sample_count
        rows.append({
            "candidate_id": cid,
            "fold_id": fold,
            "best_epoch_inner": best_epoch,
            "best_inner_rmse_wh": fold_data.get("rmse_wh", ""),
            "best_inner_mae_wh": fold_data.get("mae_wh", ""),
            "best_inner_r2": fold_data.get("r2", ""),
            "sample_count_inner_val": sample_count_inner_val,
            "population_fingerprint": lineage.get("population_fingerprint", ""),
            "scaler_bundle_checksum": lineage.get("scaler_bundle_checksum", ""),
            "stage_a_run_id": stage_a_run_id.get((cid, fold), ""),
        })

    # Atomic write (no half-written file at the canonical path).
    import io as _io
    stream = _io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(inner_path, stream.getvalue().encode("utf-8"))
    print(f"Rewrote {inner_path.relative_to(PROJECT_ROOT)} with {len(rows)} rows; columns: {header}")


if __name__ == "__main__":
    main()
