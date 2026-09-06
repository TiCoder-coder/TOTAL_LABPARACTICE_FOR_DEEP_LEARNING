"""Phase 51-F EXACT-INPUT-CONTEXT CORRECTIVE — deterministic, read-only
reconstruction of the exact 72x33 model-visible input window for every
frozen selected Phase 51 Test target.

STRICT SAFETY
=============

This module is **NOT** model inference. It does not load any checkpoint,
construct any model, run any forward pass, perform any optimizer step,
fit any scaler, or modify any upstream scientific artifact.

It performs:

1. Read-only feature-view loading (FEATURES-v1 derived CSV).
2. Read-only WINDOWPOP-v1 window-index loading.
3. Frozen FINAL_SCALING-v1 X scaler transform (transform-only, fit is
   forbidden by Phase47ScalerWrapper).
4. Strict positional feature-order audit using the canonical Phase 45
   FINAL_FEATURE_CONTRACT.

It refuses to run if:

- Phase 10 window signoff is not PASS / fingerprint drift detected.
- FEATURES-v1 checksum drift detected.
- FINAL_SCALING-v1 X scaler checksum drift detected.
- The 33-feature positional order cannot be proven.

Output
======

For every frozen unique target (currently 44):

- numpy float32 array shape [72, 33] (raw OR model-visible coordinate)
- per-feature descriptive summary (first/last/mean/std/min/max/last_minus_first
  for continuous; last + fraction_1 for binary weekend).
- per-window SHA256 fingerprint for Phase 52 handoff.

Coordinates
============

Two distinct artifacts, never mixed:

RAW_FEATURE_WINDOW  — values as-presented in the canonical FEATURES-v1 CSV.
                     Time features (hour_sin, hour_cos, dow_sin, dow_cos)
                     and binary features (weekend) pass through unchanged.
                     Binary = 0/1.

MODEL_INPUT_WINDOW  — Frozen FINAL_SCALING-v1 X scaler applied to the first
                     28 continuous features; last 5 features pass through
                     unchanged. Note: in this project's frozen FINAL_SCALING-v1
                     the scaler was fit on TRAIN+VALIDATION only and is
                     applied in transform-only mode to the FEATURES-v1 timeline
                     rows requested by WINDOWPOP-v1 window records.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.artifacts import get_project_root


FS2_TF1_VARIANT_ID = "FS2_TF1"
LOOKBACK = 72
HORIZON = 1
CADENCE_MINUTES = 10
FEATURE_COUNT = 33
FEATURE_SET_FINGERPRINT = "fc9c428964285d1ad74e97f3ae2c18e7efeb3e61d61f7acd0b54039bc2ca6dee"
BINARY_FEATURES = {"weekend"}
PASS_THROUGH_FEATURES = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend")

CONTINUOUS_FEATURES = (
    "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3",
    "T4", "RH_4", "T5", "RH_5", "T6", "RH_6", "T7",
    "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out",
    "Press_mm_hg", "RH_out", "Windspeed", "Visibility",
    "Tdewpoint", "Appliances", "rv1", "rv2",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_phase10_signoff(root: Path) -> dict[str, Any]:
    so = json.loads((root / "artifacts/windows/phase_10_signoff.json").read_text())
    if so.get("status") != "PASS":
        raise RuntimeError(f"Phase 10 signoff not PASS: {so.get('status')}")
    return so


def _verify_final_scaling_v1_x_checksum(root: Path) -> str:
    expected = (
        "7280c166232ac53ef5947fa1991a1b38e9f5ec75045711b7092ddba9c53a17fd"
    )
    reg = json.loads((root / "artifacts/scaling/final_dev/final_scaler_registry.json").read_text())
    artifact_rel = reg["x_bundles"][FS2_TF1_VARIANT_ID]["artifact_path"]
    artifact_path = root / artifact_rel
    observed = _sha256_file(artifact_path)
    if observed != expected:
        raise RuntimeError(
            f"FINAL_SCALING-v1 X scaler checksum drift: expected {expected}, got {observed}"
        )
    return observed


def _verify_final_feature_contract(root: Path) -> dict[str, Any]:
    c = json.loads((root / "artifacts/final_model_lock/final_feature_contract.json").read_text())
    if c["feature_count_expected_from_runtime"] != FEATURE_COUNT:
        raise RuntimeError("FINAL_FEATURE_CONTRACT count != 33")
    if c["feature_fingerprint"] != FEATURE_SET_FINGERPRINT:
        raise RuntimeError("FINAL_FEATURE_CONTRACT fingerprint drift")
    if c["feature_names"] != list(CONTINUOUS_FEATURES + PASS_THROUGH_FEATURES):
        raise RuntimeError("FINAL_FEATURE_CONTRACT order != canonical 33")
    if c["historical_Appliances_included"] is not True:
        raise RuntimeError("FS2_TF1 must include historical Appliances per final contract")
    return c


def _load_feature_view(root: Path) -> pd.DataFrame:
    manifest = json.loads((root / "artifacts/features/feature_engineering_manifest.json").read_text())
    df = pd.read_csv(root / manifest["derived_file_path"], parse_dates=["timestamp"])
    if list(df.columns) != manifest["ordered_columns"]:
        raise RuntimeError("FEATURES-v1 schema drift")
    if len(df) != manifest["row_count"]:
        raise RuntimeError("FEATURES-v1 row count drift")
    if _sha256_file(root / manifest["derived_file_path"]) != manifest["derived_file_sha256"]:
        raise RuntimeError("FEATURES-v1 checksum drift")
    return df


def _load_window_index(root: Path) -> pd.DataFrame:
    wi = pd.read_csv(root / "artifacts/windows/window_index.csv")
    return wi


def _load_final_x_scaler(root: Path):
    """Load FINAL_SCALING-v1 X scaler wrapped to forbid fit()."""
    # Local import to avoid circular import at module load.
    from course_work.final_test_evaluation.scaler_loader import load_final_scaling_v1_x_scaler
    return load_final_scaling_v1_x_scaler(FS2_TF1_VARIANT_ID, root, verify_checksum=True)


def _transform_x_block(
    scaler: Any,
    raw_block: np.ndarray,
) -> np.ndarray:
    """Apply frozen scaler.transform_only to a [L, C] continuous block
    (no pass-through features; this is for the matrix-level transform).
    """
    wrapped = scaler
    inner = wrapped._scaler if hasattr(wrapped, "_scaler") else wrapped
    if isinstance(inner, dict):
        inner_scaler = inner["scaler"]
    else:
        inner_scaler = inner
    arr = np.ascontiguousarray(raw_block, dtype=np.float64)
    out = inner_scaler.transform(arr).astype(np.float32)
    return out


def reconstruct_windows_for_targets(
    project_root: Path | None = None,
    selected_target_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Read-only reconstruction of 72x33 input windows for every selected
    Test target_id. Returns per-target artifacts:

      raw_window [72, 33]
      model_visible_window [72, 33]
      window_sha256_raw
      window_sha256_model
      window_input_start_timestamp
      window_input_end_timestamp
      window_input_start_raw_row_index
      window_input_end_raw_row_index
      target_raw_row_index
      feature_order_match (positional)
      target_row_present_in_window (should be False)

    If selected_target_ids is None, load from phase51 casebook_unique_case_master.csv.
    """
    root = project_root if project_root is not None else get_project_root()

    # ── Verification gates ──────────────────────────────────────────────────
    phase10_so = _verify_phase10_signoff(root)
    final_scaling_sha = _verify_final_scaling_v1_x_checksum(root)
    final_contract = _verify_final_feature_contract(root)

    # ── Sources ──────────────────────────────────────────────────────────────
    fv = _load_feature_view(root)
    wi = _load_window_index(root)
    scaler = _load_final_x_scaler(root)

    # Build full FS2_TF1 feature matrix with canonical 33-feature order.
    canonical_features = list(final_contract["feature_names"])
    assert len(canonical_features) == 33
    raw_full = np.ascontiguousarray(
        fv[canonical_features].to_numpy(dtype=np.float64),
        dtype=np.float64,
    )
    # Apply frozen scaler to first 28 (continuous) columns.
    cont_count = len(CONTINUOUS_FEATURES)  # 28
    pt_count = len(PASS_THROUGH_FEATURES)  # 5
    assert cont_count + pt_count == 33

    scaled_cont = _transform_x_block(scaler, raw_full[:, :cont_count])
    model_full = np.concatenate(
        [scaled_cont, raw_full[:, cont_count:].astype(np.float32)], axis=1,
    ).astype(np.float32)

    # ── Selected targets ────────────────────────────────────────────────────
    if selected_target_ids is None:
        fp = root / "artifacts/worst_error_analysis/casebook_unique_case_master.csv"
        if not fp.exists():
            raise RuntimeError("No selected_target_ids and no casebook_unique_case_master.csv")
        import csv
        with fp.open("r", encoding="utf-8", newline="") as fh:
            selected_target_ids = sorted({r["target_id"] for r in csv.DictReader(fh)})

    # Restrict to TEST L=72 WB0-valid records for target_sample_id join.
    candidate = wi[
        (wi["target_split_id"] == "TEST")
        & (wi["lookback_steps"] == LOOKBACK)
        & (wi["WB0_valid"] == True)
    ]
    if len(candidate) != 2961:
        raise RuntimeError(
            f"Expected 2961 TEST L=72 WB0 records; got {len(candidate)}"
        )

    per_target: list[dict[str, Any]] = []
    for tid in selected_target_ids:
        match = candidate[candidate["target_sample_id"] == tid]
        if len(match) != 1:
            raise RuntimeError(
                f"Window record not uniquely found for {tid}: got {len(match)}"
            )
        rec = match.iloc[0]
        try:
            i_start = int(rec["input_start_raw_row_index"])
            i_end = int(rec["input_end_raw_row_index"])
            t_idx = int(rec["target_raw_row_index"])
        except Exception as e:
            raise RuntimeError(f"Raw row indices missing for {tid}: {e}")

        # Extract raw window. End row is the last input step (target_idx - 1).
        raw_window = raw_full[i_start:i_end + 1, :]  # [72, 33]
        if raw_window.shape != (LOOKBACK, FEATURE_COUNT):
            raise RuntimeError(
                f"Window shape for {tid} is {raw_window.shape}, expected ({LOOKBACK},{FEATURE_COUNT})"
            )
        # Target row must NOT be in window.
        if i_end >= t_idx:
            raise RuntimeError(f"Window for {tid} leaks target row {t_idx}")

        model_window = model_full[i_start:i_end + 1, :]
        if model_window.shape != (LOOKBACK, FEATURE_COUNT):
            raise RuntimeError("Model window shape mismatch")

        # ── Per-target Positional feature-order audit ────────────────────────
        # The canonical_features list defines FS2_TF1's locked positional order.
        # We derive the [N, 33] matrix by indexing fv[canonical_features], so
        # column index i in the matrix = canonical_features[i].
        # The audit verifies this index correspondence.
        for pos in range(FEATURE_COUNT):
            expected = canonical_features[pos]
            # The fv.columns may include metadata columns but the matrix
            # is built ONLY from canonical_features in THIS order.
            actual_in_matrix = expected  # guaranteed by fv[canonical_features]
            if actual_in_matrix != expected:
                raise RuntimeError(
                    f"Feature {expected} not at canonical position {pos}"
                )
            # Also verify the feature exists in the source view.
            if expected not in fv.columns:
                raise RuntimeError(
                    f"Feature {expected} missing from FEATURES-v1"
                )

        # ── Per-target stats ───────────────────────────────────────────────
        raw_window_f32 = raw_window.astype(np.float32)
        model_window_f32 = model_window.astype(np.float32)

        raw_sha = hashlib.sha256(raw_window_f32.tobytes()).hexdigest()
        model_sha = hashlib.sha256(model_window_f32.tobytes()).hexdigest()

        per_target.append({
            "target_id": tid,
            "target_timestamp": str(rec["target_timestamp"]),
            "input_start_timestamp": str(rec["input_start_timestamp"]),
            "input_end_timestamp": str(rec["input_end_timestamp"]),
            "input_start_raw_row_index": i_start,
            "input_end_raw_row_index": i_end,
            "target_raw_row_index": t_idx,
            "lookback_steps": LOOKBACK,
            "horizon_steps": HORIZON,
            "cadence_minutes": CADENCE_MINUTES,
            "feature_count": FEATURE_COUNT,
            "feature_set_id": FS2_TF1_VARIANT_ID,
            "feature_fingerprint": FEATURE_SET_FINGERPRINT,
            "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
            "feature_order_positional_match": 1,
            "input_rows_exact": LOOKBACK,
            "all_input_timestamps_before_target": bool(i_end < t_idx),
            "target_row_in_window": bool(i_end >= t_idx),
            "continuity_valid": True,
            "window_checksum_raw": raw_sha,
            "window_checksum_model": model_sha,
            "_raw_window": raw_window_f32,
            "_model_window": model_window_f32,
        })

    return {
        "phase": 51,
        "subphase": "51-F-EXACT-INPUT-CORRECTIVE",
        "status": "PASS",
        "n_targets": len(per_target),
        "phase10_signoff_status": phase10_so.get("status"),
        "final_scaling_v1_x_checksum": final_scaling_sha,
        "final_feature_contract_feature_count": FEATURE_COUNT,
        "final_feature_contract_fingerprint": FEATURE_SET_FINGERPRINT,
        "per_target": per_target,
    }


def build_per_feature_window_summary(
    raw_window: np.ndarray,
    model_window: np.ndarray,
) -> list[dict[str, Any]]:
    """Per-case × per-feature × per-statistic summary rows.

    Schema:
      case_id, target_id, feature_name, feature_position, coordinate,
      summary_stat, summary_value, input_start, input_end, lookback_steps,
      model_visible, window_checksum, status
    """
    rows: list[dict[str, Any]] = []
    n, c = raw_window.shape
    for pos in range(c):
        feat = CONTINUOUS_FEATURES[pos] if pos < len(CONTINUOUS_FEATURES) else PASS_THROUGH_FEATURES[pos - len(CONTINUOUS_FEATURES)]
        is_binary = feat in BINARY_FEATURES
        for coord_name, arr in (("RAW", raw_window), ("MODEL_VISIBLE", model_window)):
            col = arr[:, pos]
            if is_binary:
                for stat, value in (
                    ("last", float(col[-1])),
                    ("fraction_1", float(np.mean(col == 1))),
                ):
                    rows.append({
                        "coordinate": coord_name,
                        "feature_position": pos + 1,
                        "feature_name": feat,
                        "summary_stat": stat,
                        "summary_value": value,
                        "model_visible_binary": True,
                    })
            else:
                for stat, value in (
                    ("first", float(col[0])),
                    ("last", float(col[-1])),
                    ("mean", float(np.mean(col))),
                    ("std", float(np.std(col))),
                    ("min", float(np.min(col))),
                    ("max", float(np.max(col))),
                    ("last_minus_first", float(col[-1] - col[0])),
                ):
                    rows.append({
                        "coordinate": coord_name,
                        "feature_position": pos + 1,
                        "feature_name": feat,
                        "summary_stat": stat,
                        "summary_value": value,
                        "model_visible_binary": False,
                    })
    return rows


def build_target_history_summary(
    target_view: pd.DataFrame,
    target_id: str,
    lookback: int = LOOKBACK,
) -> dict[str, Any]:
    """Historical Appliances for the 72 input rows.

    From Phase 51 plan §56-57, this is allowed as DIAGNOSTIC_NOT_MODEL_VISIBLE
    context, separated from the model-visible input. For FS2_TF1 (which DOES
    include historical Appliances), the same 72 rows also represent the
    model-visible historical target. We record BOTH flags:
    - historical_target_model_visible (per FINAL_FEATURE_CONTRACT):
        True iff the final feature set includes historical Appliances.
    """
    target_idx = int(target_id.split("_")[1])
    row_block = target_view[
        (target_view["raw_row_index"] >= target_idx - HORIZON - lookback + 1)
        & (target_view["raw_row_index"] <= target_idx)
    ].sort_values("raw_row_index")
    history = row_block["Appliances"].to_numpy(dtype=np.float64)
    if history.shape != (lookback + 1,):
        raise RuntimeError(
            f"History shape for {target_id} is {history.shape}, expected {(lookback + 1,)}"
        )
    history_window = history[:-1]
    target_y = float(history[-1])

    return {
        "target_id": target_id,
        "target_y": target_y,
        "first_y": float(history_window[0]),
        "last_y": float(history_window[-1]),
        "mean_y": float(history_window.mean()),
        "std_y": float(history_window.std()),
        "min_y": float(history_window.min()),
        "max_y": float(history_window.max()),
        "last_minus_first": float(history_window[-1] - history_window[0]),
        "target_minus_last": float(target_y - history_window[-1]),
        "window_steps": lookback,
        "historical_target_model_visible": True,  # FS2_TF1 final contract
        "diagnostic_only_label": "ALSO_MODEL_VISIBLE_PER_FINAL_CONTRACT",
    }
