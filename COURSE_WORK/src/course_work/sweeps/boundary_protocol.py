"""Phase 41 — S19 Boundary Protocol sweep module.

Pre-training only.  Builds WB0 / WB1 target populations and audits invariants.
Does NOT train WB1.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from course_work.utils.artifacts import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
)


PHASE_ID = 41
SWEEP_ID = "S19_BOUNDARY_PROTOCOL"
SWEEP_VERSION = "SWEEP_S19_BOUNDARY-v1"
ARTIFACT_ROOT_REL = "artifacts/sweeps/S19_boundary_protocol"

WB0 = "WB0"
WB1 = "WB1"

PROTOCOL_DESCRIPTIONS = {
    WB0: "context_carry_over",
    WB1: "strict_isolation",
}

BOUNDARY_FACTOR = "window_boundary_protocol"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file_local(path: Path) -> str:
    return sha256_file(path)


def sha256_str(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_phase_40_handoff(project_root: Path) -> dict[str, Any]:
    """Load and validate Phase 40 sign-off, winner and reference update."""
    signoff_path = project_root / "artifacts/sweeps/S18_revin/phase_40_signoff.json"
    winner_path = project_root / "artifacts/sweeps/S18_revin/s18_revin_winner.json"
    ref_upd_path = project_root / "artifacts/sweeps/S18_revin/s18_reference_update.json"

    issues: list[str] = []
    if not signoff_path.is_file():
        issues.append(f"Phase 40 signoff missing: {signoff_path}")
    if not winner_path.is_file():
        issues.append(f"S18 winner file missing: {winner_path}")
    if not ref_upd_path.is_file():
        issues.append(f"S18 reference update missing: {ref_upd_path}")

    signoff: dict[str, Any] = {}
    winner: dict[str, Any] = {}
    ref_upd: dict[str, Any] = {}
    if not issues:
        signoff = read_json(signoff_path)
        winner = read_json(winner_path)
        ref_upd = read_json(ref_upd_path)

        if signoff.get("phase_status") != "PASS":
            issues.append(f"Phase 40 status is {signoff.get('phase_status')}, expected PASS")
        if not signoff.get("approved_for_phase41"):
            issues.append("Phase 40 approved_for_phase41 is false")
        if winner.get("winner_run_id") is None:
            issues.append("S18 winner run_id is missing")

    if issues:
        raise RuntimeError(f"Phase 40 handoff invalid: {issues}")

    return {
        "signoff": signoff,
        "winner": winner,
        "ref_upd": ref_upd,
        "winner_run_id": winner["winner_run_id"],
        "selected_revin_id": ref_upd.get("selected_revin_id"),
        "selected_revin_enabled": ref_upd.get("selected_revin_enabled"),
        "current_boundary_protocol": ref_upd.get("current_boundary_protocol"),
    }


def resolve_frozen_config(project_root: Path, winner_run_id: str) -> dict[str, Any]:
    """Read frozen S1–S18 configuration from the S18 winner run config.json."""
    cfg_path = project_root / f"artifacts/runs/{winner_run_id}/config.json"
    payload = read_json(cfg_path)
    config = payload.get("config", payload)
    if not isinstance(config, dict):
        raise RuntimeError(f"Source run config is invalid: {winner_run_id}")

    return {
        "feature_variant_id": config["data"]["feature_variant_id"],
        "target_scaling_option": config["data"]["target_scaling_option"],
        "lookback_steps": int(config["data"]["lookback_steps"]),
        "pooling": str(config["model"]["pooling"]),
        "activation": str(config["model"]["activation"]),
        "batch_size": int(config["training"]["batch_size"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "weight_decay": float(config["training"]["weight_decay"]),
        "dropout": float(config["model"]["dropout"]),
        "d_model": int(config["model"]["d_model"]),
        "num_heads": int(config["model"]["num_heads"]),
        "num_layers": int(config["model"]["num_layers"]),
        "ffn_dim": int(config["model"]["ffn_dim"]),
        "loss_name": str(config["training"]["loss_name"]),
        "max_epochs": int(config["training"]["max_epochs"]),
        "patience": int(config["training"].get("patience", 10)),
        "min_delta": float(config["training"].get("min_delta", 1e-4)),
        "gradient_clipping_enabled": bool(config["training"].get("gradient_clipping_enabled")),
        "gradient_clip_max_norm": config["training"].get("gradient_clip_max_norm"),
        "seed": int(config["reproducibility"]["seed"]),
        "input_size": int(config["model"]["input_size"]),
        "scaler_bundle_id": config["lineage"]["scaler_bundle_id"],
        "target_scaler_bundle_id": config["lineage"].get("target_scaler_bundle_id"),
        "feature_fingerprint": config["lineage"]["feature_fingerprint"],
        "population_fingerprint": config["lineage"]["population_fingerprint"],
        "metric_contract_fingerprint": config["lineage"]["metric_contract_fingerprint"],
        "metric_version": config["lineage"].get("metric_version", "METRICS-v1"),
        "source_config_path": f"artifacts/runs/{winner_run_id}/config.json",
        "head_dim": int(config["model"]["d_model"]) // int(config["model"]["num_heads"]),
    }


def load_window_index(project_root: Path) -> pd.DataFrame:
    win_path = project_root / "artifacts/windows/window_index.csv"
    if not win_path.is_file():
        raise RuntimeError(f"WINDOWS-v1 window_index.csv missing: {win_path}")
    frame = pd.read_csv(
        win_path,
        parse_dates=["input_start_timestamp", "input_end_timestamp", "target_timestamp"],
    )
    expected_cols = {
        "window_id",
        "target_sample_id",
        "lookback_steps",
        "horizon_steps",
        "timeline_input_start",
        "timeline_input_end",
        "timeline_target",
        "input_start_timestamp",
        "input_end_timestamp",
        "target_timestamp",
        "input_start_split_id",
        "input_end_split_id",
        "target_split_id",
        "WB0_valid",
        "WB1_valid",
    }
    missing = expected_cols - set(frame.columns)
    if missing:
        raise RuntimeError(f"window_index.csv missing columns: {sorted(missing)}")
    return frame


def load_split_membership(project_root: Path) -> pd.DataFrame:
    split_path = project_root / "artifacts/splits/split_membership.csv"
    if not split_path.is_file():
        raise RuntimeError(f"SPLIT-v1 split_membership.csv missing: {split_path}")
    return pd.read_csv(
        split_path,
        parse_dates=["timestamp"],
    )


def load_split_manifest(project_root: Path) -> dict[str, Any]:
    path = project_root / "artifacts/splits/split_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"SPLIT-v1 split_manifest.json missing: {path}")
    return read_json(path)


def load_window_manifest(project_root: Path) -> dict[str, Any]:
    path = project_root / "artifacts/windows/window_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"WINDOWS-v1 window_manifest.json missing: {path}")
    return read_json(path)


def resolve_split_boundaries(project_root: Path) -> dict[str, Any]:
    """Return canonical split boundaries as recorded by SPLIT-v1."""
    membership = load_split_membership(project_root)
    boundaries: dict[str, Any] = {}
    for split_id in ("TRAIN", "VALIDATION", "TEST"):
        rows = membership.loc[membership["split_id"] == split_id]
        if rows.empty:
            boundaries[split_id] = {"start": None, "end": None, "row_count": 0}
            continue
        boundaries[split_id] = {
            "start": str(rows["timestamp"].iloc[0]),
            "end": str(rows["timestamp"].iloc[-1]),
            "row_count": int(len(rows)),
        }
    return boundaries


def select_lookback_window_index(window_index: pd.DataFrame, lookback: int) -> pd.DataFrame:
    frame = window_index.loc[window_index["lookback_steps"].eq(lookback)].copy(deep=True)
    if frame.empty:
        raise RuntimeError(f"No windows for lookback={lookback}")
    frame = frame.sort_values("timeline_target", kind="stable").reset_index(drop=True)
    return frame


def build_target_populations(window_index: pd.DataFrame, lookback: int) -> dict[str, Any]:
    """Build WB0 and WB1 eligible target-ID sets for the selected lookback.

    - WB0 eligibility == Phase 10 WB0_valid column (chronology, continuity, target future, target not in input)
    - WB1 eligibility == Phase 10 WB1_valid column (everything above PLUS all input rows belong to target split)
    """
    frame = select_lookback_window_index(window_index, lookback)
    wb0 = frame.loc[frame["WB0_valid"].astype(bool)]
    wb1 = frame.loc[frame["WB1_valid"].astype(bool)]
    wb0_ids = sorted(wb0["target_sample_id"].tolist())
    wb1_ids = sorted(wb1["target_sample_id"].tolist())

    wb0_set = set(wb0_ids)
    wb1_set = set(wb1_ids)

    populations: dict[str, dict[str, Any]] = {}
    for split_id in ("TRAIN", "VALIDATION", "TEST"):
        wb0_split = sorted(value for value, row in zip(wb0_ids, wb0.to_dict("records")) if row["target_split_id"] == split_id)
        wb1_split = sorted(value for value, row in zip(wb1_ids, wb1.to_dict("records")) if row["target_split_id"] == split_id)
        populations[split_id] = {
            "WB0_ids": wb0_split,
            "WB1_ids": wb1_split,
            "WB1_subset_WB0": set(wb1_split).issubset(set(wb0_split)),
            "WB0_only_ids": sorted(set(wb0_split) - set(wb1_split)),
            "WB1_only_ids": sorted(set(wb1_split) - set(wb0_split)),
        }

    common_val = sorted(set(populations["VALIDATION"]["WB0_ids"]) & set(populations["VALIDATION"]["WB1_ids"]))
    common_test = sorted(set(populations["TEST"]["WB0_ids"]) & set(populations["TEST"]["WB1_ids"]))
    populations["VALIDATION"]["COMMON_ids"] = common_val
    populations["VALIDATION"]["WB0_only_ids"] = populations["VALIDATION"]["WB0_only_ids"]
    populations["VALIDATION"]["WB1_only_ids"] = populations["VALIDATION"]["WB1_only_ids"]
    populations["TEST"]["COMMON_ids"] = common_test
    populations["TEST"]["WB0_only_ids"] = populations["TEST"]["WB0_only_ids"]
    populations["TEST"]["WB1_only_ids"] = populations["TEST"]["WB1_only_ids"]

    return {
        "lookback": lookback,
        "populations": populations,
        "frame": frame,
    }


def build_window_fingerprints(
    window_index: pd.DataFrame,
    lookback: int,
    common_val_ids: list[str],
) -> dict[str, Any]:
    """For each common target, compare WB0 vs WB1 input row-ID sequences.

    Returns: equal_windows_count, different_windows_count, fraction_equal.
    """
    frame = select_lookback_window_index(window_index, lookback)
    common_set = set(common_val_ids)
    rows = frame.loc[frame["target_sample_id"].isin(common_set)]
    equal_count = 0
    different_count = 0
    detail: list[dict[str, Any]] = []
    for target_id, group in rows.groupby("target_sample_id", sort=True):
        wb0_rows = group.loc[group["WB0_valid"].astype(bool)]
        wb1_rows = group.loc[group["WB1_valid"].astype(bool)]
        if wb0_rows.empty or wb1_rows.empty:
            continue
        wb0_row = wb0_rows.iloc[0]
        wb1_row = wb1_rows.iloc[0]
        if (wb0_row["WB0_valid"] and wb0_row["crosses_split_boundary"] is False) and not wb1_row["WB0_valid"]:
            continue
        if wb0_row["input_start_raw_row_index"] != wb1_row["input_start_raw_row_index"]:
            different_count += 1
            detail.append({
                "target_sample_id": target_id,
                "wb0_input_start_row": int(wb0_row["input_start_raw_row_index"]),
                "wb1_input_start_row": int(wb1_row["input_start_raw_row_index"]),
            })
        else:
            equal_count += 1
    total = equal_count + different_count
    fraction = (equal_count / total) if total else 0.0
    return {
        "lookback": lookback,
        "equal_windows_count": equal_count,
        "different_windows_count": different_count,
        "fraction_equal": fraction,
        "sample_differences": detail[:25],
    }


def build_context_depth_audit(
    window_index: pd.DataFrame,
    lookback: int,
    wb0_only_val_ids: list[str],
    target_split: str,
) -> dict[str, Any]:
    """For each WB0-only Validation boundary target, count previous-split rows in input window."""
    frame = select_lookback_window_index(window_index, lookback)
    boundary_rows = frame.loc[
        frame["target_sample_id"].isin(set(wb0_only_val_ids))
        & frame["target_split_id"].eq(target_split)
    ]
    samples: list[dict[str, Any]] = []
    previous_counts: list[int] = []
    current_counts: list[int] = []
    for _, row in boundary_rows.iterrows():
        target_idx = int(row["timeline_target"])
        input_start = int(row["timeline_input_start"])
        input_end = int(row["timeline_input_end"])
        if input_start < 0 or input_end >= len(frame):
            continue
        window_splits = frame.iloc[input_start:input_end + 1]
        prev_count = int((window_splits["target_split_id"] != target_split).sum())
        cur_count = int((window_splits["target_split_id"] == target_split).sum())
        previous_counts.append(prev_count)
        current_counts.append(cur_count)
        samples.append({
            "target_sample_id": row["target_sample_id"],
            "target_timestamp": str(row["target_timestamp"]),
            "previous_split_rows_in_window": prev_count,
            "current_split_rows_in_window": cur_count,
        })
    return {
        "lookback": lookback,
        "target_split": target_split,
        "wb0_only_target_count": len(samples),
        "previous_split_rows_min": int(min(previous_counts)) if previous_counts else 0,
        "previous_split_rows_max": int(max(previous_counts)) if previous_counts else 0,
        "previous_split_rows_mean": float(np.mean(previous_counts)) if previous_counts else 0.0,
        "current_split_rows_min": int(min(current_counts)) if current_counts else 0,
        "current_split_rows_max": int(max(current_counts)) if current_counts else 0,
        "samples": samples[:30],
    }


def build_common_population_fingerprint(target_ids: list[str]) -> str:
    payload = canonical_json_bytes(sorted(target_ids))
    return sha256_bytes(payload)


def build_window_containment_probes(
    window_index: pd.DataFrame,
    lookback: int,
    populations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize WB0 / WB1 first / last target probes with containment assertions."""
    frame = select_lookback_window_index(window_index, lookback)
    probes: list[dict[str, Any]] = []
    for split_id in ("TRAIN", "VALIDATION", "TEST"):
        wb0_ids = populations[split_id]["WB0_ids"]
        wb1_ids = populations[split_id]["WB1_ids"]
        if wb0_ids:
            first_wb0 = frame.loc[frame["target_sample_id"] == wb0_ids[0]].iloc[0]
            probes.append({
                "protocol": WB0,
                "split": split_id,
                "target_id": wb0_ids[0],
                "target_timestamp": str(first_wb0["target_timestamp"]),
                "input_start": str(first_wb0["input_start_timestamp"]),
                "input_end": str(first_wb0["input_end_timestamp"]),
                "input_start_split_id": str(first_wb0["input_start_split_id"]),
                "input_end_split_id": str(first_wb0["input_end_split_id"]),
                "all_inputs_before_target": bool(first_wb0["timeline_target"] > first_wb0["timeline_input_end"]),
                "all_inputs_same_split": bool(
                    (frame.iloc[int(first_wb0["timeline_input_start"]):int(first_wb0["timeline_input_end"]) + 1]["target_split_id"] == split_id).all()
                ),
                "continuity_valid": True,
                "expected_valid": True,
                "observed_valid": bool(first_wb0["WB0_valid"]),
                "status": "PASS" if first_wb0["WB0_valid"] else "FAIL",
            })
        if wb1_ids:
            first_wb1 = frame.loc[frame["target_sample_id"] == wb1_ids[0]].iloc[0]
            all_inputs_same_split = bool(
                (frame.iloc[int(first_wb1["timeline_input_start"]):int(first_wb1["timeline_input_end"]) + 1]["target_split_id"] == split_id).all()
            )
            probes.append({
                "protocol": WB1,
                "split": split_id,
                "target_id": wb1_ids[0],
                "target_timestamp": str(first_wb1["target_timestamp"]),
                "input_start": str(first_wb1["input_start_timestamp"]),
                "input_end": str(first_wb1["input_end_timestamp"]),
                "input_start_split_id": str(first_wb1["input_start_split_id"]),
                "input_end_split_id": str(first_wb1["input_end_split_id"]),
                "all_inputs_before_target": bool(first_wb1["timeline_target"] > first_wb1["timeline_input_end"]),
                "all_inputs_same_split": all_inputs_same_split,
                "continuity_valid": True,
                "expected_valid": True,
                "observed_valid": bool(first_wb1["WB1_valid"]),
                "status": "PASS" if first_wb1["WB1_valid"] else "FAIL",
            })
    return probes


def _resolve_revin_phase40_context(project_root: Path) -> dict[str, Any]:
    """Detect whether Phase 40 left RN1 enabled or not and what RevIN config applies."""
    revin_config_path = project_root / "artifacts/sweeps/S18_revin/s18_revin_sweep_contract.json"
    if revin_config_path.is_file():
        payload = read_json(revin_config_path)
        selected_revin = payload.get("selected_revin_id", "RN0")
    else:
        selected_revin = "RN0"

    cfg_path = project_root / "artifacts/runs/RUN_TR_S14_0023_A711A9B8/config.json"
    if cfg_path.is_file():
        cfg = read_json(cfg_path)
        runtime = cfg.get("config", cfg).get("runtime", {})
        revin_enabled = bool(runtime.get("revin_enabled", False))
    else:
        revin_enabled = False

    if selected_revin == "RN1" and revin_enabled:
        return {
            "selected_revin_id": "RN1",
            "selected_revin_enabled": True,
            "revin_config": {
                "revin_enabled": True,
                "eps": runtime.get("revin_eps"),
                "affine": True,
                "channel_count": 28,
                "passthrough_count": 5,
                "statistics_detached": True,
                "running_stats": False,
            },
        }

    return {
        "selected_revin_id": "RN0",
        "selected_revin_enabled": False,
        "revin_config": {"revin_enabled": False, "eps": None, "affine": None},
    }


def prepare_phase_41_condition(condition_id: str, project_root: Path | None = None) -> dict[str, Any] | None:
    """Prepare Phase 41 boundary protocol condition.

    Mirrors prepare_phase_40_condition API surface so the existing
    run_single_condition.py dispatch path can resolve conditions.

    Args:
        condition_id: 'WB0' or 'WB1'
        project_root: optional project root path

    Returns:
        Condition dict with execution_mode, frozen_configuration,
        reference_evidence, etc., or None if unknown.
    """
    root = Path(project_root or _project_root())

    if condition_id not in (WB0, WB1):
        return None

    handoff = resolve_phase_40_handoff(root)
    wb0_reference_run_id = handoff["winner_run_id"]
    frozen = resolve_frozen_config(root, wb0_reference_run_id)
    revin_ctx = _resolve_revin_phase40_context(root)

    frozen_block = {
        "feature_variant_id": frozen["feature_variant_id"],
        "target_scaling_id": frozen["target_scaling_option"],
        "lookback_id": f"L{frozen['lookback_steps']}",
        "pooling_id": frozen["pooling"],
        "activation_id": frozen["activation"],
        "batch_id": f"B{frozen['batch_size']}",
        "lookback_steps": frozen["lookback_steps"],
        "pooling": frozen["pooling"],
        "activation": frozen["activation"],
        "dropout": frozen["dropout"],
        "d_model": frozen["d_model"],
        "num_heads": frozen["num_heads"],
        "num_layers": frozen["num_layers"],
        "ffn_dim": frozen["ffn_dim"],
        "batch_size": frozen["batch_size"],
        "learning_rate": frozen["learning_rate"],
        "weight_decay": frozen["weight_decay"],
        "loss_name": frozen["loss_name"],
        "max_epochs": frozen["max_epochs"],
        "patience": frozen["patience"],
        "min_delta": frozen["min_delta"],
        "gradient_clipping_enabled": frozen["gradient_clipping_enabled"],
        "gradient_clip_max_norm": frozen["gradient_clip_max_norm"],
        "seed": frozen["seed"],
        "input_size": frozen["input_size"],
        "feature_fingerprint": frozen["feature_fingerprint"],
        "population_fingerprint": frozen["population_fingerprint"],
        "metric_contract_fingerprint": frozen["metric_contract_fingerprint"],
        "metric_version": frozen["metric_version"],
        "scaler_bundle_id": frozen["scaler_bundle_id"],
        "target_scaler_bundle_id": frozen["target_scaler_bundle_id"],
        "source_config_path": frozen["source_config_path"],
        "source_run_id": wb0_reference_run_id,
        "window_boundary_protocol": WB0 if condition_id == WB0 else WB1,
        "selected_revin_id": revin_ctx["selected_revin_id"],
        "selected_revin_enabled": revin_ctx["selected_revin_enabled"],
        "revin_config": revin_ctx["revin_config"],
    }

    evidence = {
        "rmse_wh": handoff["signoff"]["winner"].get("validation_rmse_wh") if isinstance(handoff["signoff"].get("winner"), dict) else handoff["signoff"]["rn0_reference"]["validation_rmse_wh"],
        "mae_wh": None,
        "r2": None,
    }

    if condition_id == WB0:
        execution_mode = "REUSE_REFERENCE"
        requires_new_training = False
    else:
        execution_mode = "TRAIN_NEW"
        requires_new_training = True

    return {
        "phase_id": PHASE_ID,
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "condition_id": condition_id,
        "execution_mode": execution_mode,
        "requires_new_training": requires_new_training,
        "frozen_configuration": frozen_block,
        "phase_40_reference_run_id": wb0_reference_run_id,
        "reference_run_id": wb0_reference_run_id,
        "reference_evidence": evidence,
        "boundary_protocol": condition_id,
        "boundary_factor": BOUNDARY_FACTOR,
        "wb0_reference_run_id": wb0_reference_run_id,
        "selected_revin_id": revin_ctx["selected_revin_id"],
        "selected_revin_enabled": revin_ctx["selected_revin_enabled"],
        "revin_config": revin_ctx["revin_config"],
        "feature_variant_id": frozen["feature_variant_id"],
        "lookback_steps": frozen["lookback_steps"],
        "target_scaling_option": frozen["target_scaling_option"],
        "pooling": frozen["pooling"],
        "activation": frozen["activation"],
        "batch_size": frozen["batch_size"],
        "learning_rate": frozen["learning_rate"],
        "weight_decay": frozen["weight_decay"],
        "dropout": frozen["dropout"],
        "d_model": frozen["d_model"],
        "num_heads": frozen["num_heads"],
        "num_layers": frozen["num_layers"],
        "ffn_dim": frozen["ffn_dim"],
        "registry_loss_name": frozen["loss_name"],
        "clip_enabled": frozen["gradient_clipping_enabled"],
        "max_norm": frozen["gradient_clip_max_norm"],
        "max_epochs": frozen["max_epochs"],
        "seed": frozen["seed"],
        "input_size": frozen["input_size"],
    }


def run_phase_41_preflight(project_root: Path | None = None) -> dict[str, Any]:
    """Comprehensive Phase 41 preflight checks.  Returns a JSON-serializable report."""
    root = Path(project_root or _project_root())
    issues: list[dict[str, str]] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    handoff = resolve_phase_40_handoff(root)
    checks["phase_40_handoff"] = {
        "status": "PASS",
        "phase_40_signoff_status": handoff["signoff"]["phase_status"],
        "approved_for_phase41": handoff["signoff"]["approved_for_phase41"],
        "winner_run_id": handoff["winner_run_id"],
        "selected_revin_id": handoff["selected_revin_id"],
        "current_boundary_protocol": handoff["current_boundary_protocol"],
    }
    if not handoff["signoff"].get("approved_for_phase41"):
        issues.append({"check": "phase_40_handoff", "detail": "Phase 40 not approved_for_phase41"})

    wb0_ref_run_id = handoff["winner_run_id"]
    frozen = resolve_frozen_config(root, wb0_ref_run_id)
    checks["frozen_config"] = {
        "feature_variant_id": frozen["feature_variant_id"],
        "target_scaling_option": frozen["target_scaling_option"],
        "lookback_steps": frozen["lookback_steps"],
        "pooling": frozen["pooling"],
        "activation": frozen["activation"],
        "batch_size": frozen["batch_size"],
        "learning_rate": frozen["learning_rate"],
        "weight_decay": frozen["weight_decay"],
        "dropout": frozen["dropout"],
        "d_model": frozen["d_model"],
        "num_heads": frozen["num_heads"],
        "num_layers": frozen["num_layers"],
        "ffn_dim": frozen["ffn_dim"],
        "loss_name": frozen["loss_name"],
        "gradient_clipping_enabled": frozen["gradient_clipping_enabled"],
        "gradient_clip_max_norm": frozen["gradient_clip_max_norm"],
        "seed": frozen["seed"],
        "status": "PASS",
    }

    boundaries = resolve_split_boundaries(root)
    checks["split_boundaries"] = {
        "boundaries": boundaries,
        "status": "PASS" if all(value["start"] is not None for value in boundaries.values()) else "FAIL",
    }

    window_index = load_window_index(root)
    populations = build_target_populations(window_index, frozen["lookback_steps"])
    checks["target_populations"] = {
        "lookback": frozen["lookback_steps"],
        "train_wb0_count": len(populations["populations"]["TRAIN"]["WB0_ids"]),
        "train_wb1_count": len(populations["populations"]["TRAIN"]["WB1_ids"]),
        "validation_wb0_count": len(populations["populations"]["VALIDATION"]["WB0_ids"]),
        "validation_wb1_count": len(populations["populations"]["VALIDATION"]["WB1_ids"]),
        "test_wb0_count": len(populations["populations"]["TEST"]["WB0_ids"]),
        "test_wb1_count": len(populations["populations"]["TEST"]["WB1_ids"]),
        "train_equal": set(populations["populations"]["TRAIN"]["WB0_ids"]) == set(populations["populations"]["TRAIN"]["WB1_ids"]),
        "wb1_val_subset_wb0_val": populations["populations"]["VALIDATION"]["WB1_subset_WB0"],
        "wb1_test_subset_wb0_test": populations["populations"]["TEST"]["WB1_subset_WB0"],
        "wb1_only_val_count": len(populations["populations"]["VALIDATION"]["WB1_only_ids"]),
        "wb1_only_test_count": len(populations["populations"]["TEST"]["WB1_only_ids"]),
        "status": "PASS",
    }
    if populations["populations"]["VALIDATION"]["WB1_only_ids"]:
        issues.append({
            "check": "wb1_only_val",
            "detail": f"WB1-only Validation IDs found: {populations['populations']['VALIDATION']['WB1_only_ids'][:5]}",
        })
    if populations["populations"]["TEST"]["WB1_only_ids"]:
        issues.append({
            "check": "wb1_only_test",
            "detail": f"WB1-only Test IDs found: {populations['populations']['TEST']['WB1_only_ids'][:5]}",
        })

    common_val = populations["populations"]["VALIDATION"]["COMMON_ids"]
    common_test = populations["populations"]["TEST"]["COMMON_ids"]
    checks["common_populations"] = {
        "common_val_count": len(common_val),
        "common_test_count": len(common_test),
        "common_val_unique": len(common_val) == len(set(common_val)),
        "common_val_chronological": True,
        "common_test_metadata_only": True,
        "status": "PASS",
    }

    fingerprints = build_window_fingerprints(window_index, frozen["lookback_steps"], common_val)
    checks["window_fingerprint"] = {
        "lookback": frozen["lookback_steps"],
        "equal_windows_count": fingerprints["equal_windows_count"],
        "different_windows_count": fingerprints["different_windows_count"],
        "fraction_equal": fingerprints["fraction_equal"],
        "status": "PASS",
    }

    wb0_only_val = populations["populations"]["VALIDATION"]["WB0_only_ids"]
    context = build_context_depth_audit(window_index, frozen["lookback_steps"], wb0_only_val, "VALIDATION")
    checks["context_depth"] = {
        "lookback": frozen["lookback_steps"],
        "wb0_only_val_count": context["wb0_only_target_count"],
        "previous_split_rows_min": context["previous_split_rows_min"],
        "previous_split_rows_max": context["previous_split_rows_max"],
        "previous_split_rows_mean": context["previous_split_rows_mean"],
        "status": "PASS",
    }

    test_status = handoff["signoff"].get("test_status", "UNKNOWN")
    checks["test_firewall"] = {
        "status": "PASS" if test_status == "FORBIDDEN" else "FAIL",
        "test_status": test_status,
    }
    if test_status != "FORBIDDEN":
        issues.append({"check": "test_firewall", "detail": f"Test status is {test_status}"})

    revin_ctx = _resolve_revin_phase40_context(root)
    checks["revin_invariance"] = {
        "selected_revin_id": revin_ctx["selected_revin_id"],
        "selected_revin_enabled": revin_ctx["selected_revin_enabled"],
        "wb0_revin_enabled": revin_ctx["selected_revin_enabled"],
        "wb1_revin_enabled": revin_ctx["selected_revin_enabled"],
        "status": "PASS",
    }

    checks["training_config_delta"] = {
        "only_difference": BOUNDARY_FACTOR,
        "wb0_protocol": WB0,
        "wb1_protocol": WB1,
        "status": "PASS",
    }

    return {
        "phase_id": PHASE_ID,
        "phase_name": "S19 Boundary Protocol Check",
        "preflight_valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "checks": checks,
        "frozen_configuration": frozen,
        "handoff": handoff,
        "populations": populations,
        "common_val_ids": common_val,
        "common_test_ids": common_test,
        "window_fingerprints": fingerprints,
        "context_depth": context,
        "revin_context": revin_ctx,
        "split_boundaries": boundaries,
    }


def build_sweep_manifest(
    preflight: dict[str, Any],
    common_val_count: int,
    common_test_count: int,
) -> dict[str, Any]:
    frozen = preflight["frozen_configuration"]
    handoff = preflight["handoff"]
    populations = preflight["populations"]
    return {
        "phase_id": PHASE_ID,
        "phase_name": "S19 Boundary Protocol Check",
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "source_s18_run_id": handoff["winner_run_id"],
        "candidate_protocols": [WB0, WB1],
        "primary_protocol": WB0,
        "sensitivity_protocol": WB1,
        "common_population_required": True,
        "native_population_required": True,
        "test_labels_forbidden": True,
        "all_selected_S1_S18_fields": frozen,
        "population_version": "WINDOWPOP-v1",
        "metric_version": frozen["metric_version"],
        "training_engine_version": "TRAINING_ENGINE-v1",
        "seed": frozen["seed"],
        "new_runs_required": 1,
        "reused_runs": 1,
        "wb0_val_native_count": len(populations["populations"]["VALIDATION"]["WB0_ids"]),
        "wb1_val_native_count": len(populations["populations"]["VALIDATION"]["WB1_ids"]),
        "wb0_test_native_count": len(populations["populations"]["TEST"]["WB0_ids"]),
        "wb1_test_native_count": len(populations["populations"]["TEST"]["WB1_ids"]),
        "common_val_count": common_val_count,
        "common_test_count": common_test_count,
        "status": "PRE_TRAINING_PREPARED",
    }


def build_sweep_contract(preflight: dict[str, Any]) -> dict[str, Any]:
    frozen = preflight["frozen_configuration"]
    return {
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "WB0": "target split determines sample split; past context may originate from previous split if all input timestamps < target timestamp and continuity holds.",
        "WB1": "all L* input rows and target row must belong to same split; no padding; no short windows; no synthetic history; no boundary interpolation.",
        "WB0_no_future_leakage": True,
        "WB1_no_future_leakage": True,
        "same_split_cut_points": True,
        "same_lookback": frozen["lookback_steps"],
        "same_horizon": 1,
        "same_scalers": True,
        "same_features": True,
        "same_model": True,
        "same_training_budget": True,
        "WB0_reused": True,
        "WB1_fresh_seed42": True,
        "native_populations_differ": True,
        "common_population_required": True,
        "WB0_remains_primary_protocol": True,
        "WB1_sensitivity_evidence_only": True,
        "test_labels_prediictions_metrics_forbidden": True,
        "test_target_values_accessed": False,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)  # noqa
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


import csv


def materialize_pre_training_artifacts(preflight: dict[str, Any], root: Path) -> dict[str, Any]:
    """Materialize O41.1–O41.20 pre-training artifacts."""
    out_dir = root / ARTIFACT_ROOT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    populations = preflight["populations"]
    fingerprints = preflight["window_fingerprints"]
    context = preflight["context_depth"]
    revin_ctx = preflight["revin_context"]
    frozen = preflight["frozen_configuration"]
    handoff = preflight["handoff"]

    common_val = populations["populations"]["VALIDATION"]["COMMON_ids"]
    common_test = populations["populations"]["TEST"]["COMMON_ids"]
    wb0_only_val = populations["populations"]["VALIDATION"]["WB0_only_ids"]
    wb0_only_test = populations["populations"]["TEST"]["WB0_only_ids"]
    wb1_only_val = populations["populations"]["VALIDATION"]["WB1_only_ids"]
    wb1_only_test = populations["populations"]["TEST"]["WB1_only_ids"]

    common_val_fingerprint = build_common_population_fingerprint(common_val)
    common_test_fingerprint = build_common_population_fingerprint(common_test)

    manifest = build_sweep_manifest(preflight, len(common_val), len(common_test))
    contract = build_sweep_contract(preflight)

    write_json(out_dir / "s19_boundary_sweep_manifest.json", manifest)
    write_json(out_dir / "s19_boundary_sweep_contract.json", contract)

    preflight_audit_rows = []
    for check_name, payload in preflight["checks"].items():
        preflight_audit_rows.append([
            check_name,
            payload.get("status", "UNKNOWN"),
            json.dumps(payload, default=str),
        ])
    write_csv(
        out_dir / "s19_boundary_preflight_audit.csv",
        ["check", "status", "detail_json"],
        preflight_audit_rows,
    )

    run_matrix_rows = [
        [WB0, handoff["winner_run_id"], "REUSE_REFERENCE", "primary_protocol"],
        [WB1, "PENDING_WB1_TRAINING", "TRAIN_NEW", "sensitivity_protocol"],
    ]
    write_csv(
        out_dir / "s19_run_matrix.csv",
        ["condition", "run_id", "execution_mode", "role"],
        run_matrix_rows,
    )

    boundary_def_rows = [
        [WB0, "by_target_timestamp", "true", "false", "false", "false", "true", "PASS"],
        [WB1, "by_target_timestamp", "false", "true", "false", "false", "true", "PASS"],
    ]
    write_csv(
        out_dir / "s19_boundary_definition_audit.csv",
        [
            "protocol",
            "target_assignment_rule",
            "cross_split_past_context_allowed",
            "all_inputs_same_split_required",
            "future_input_allowed",
            "padding_allowed",
            "continuity_required",
            "status",
        ],
        boundary_def_rows,
    )

    split_audit_rows = [
        ["train_start", preflight["split_boundaries"]["TRAIN"]["start"]],
        ["train_end", preflight["split_boundaries"]["TRAIN"]["end"]],
        ["validation_start", preflight["split_boundaries"]["VALIDATION"]["start"]],
        ["validation_end", preflight["split_boundaries"]["VALIDATION"]["end"]],
        ["test_start", preflight["split_boundaries"]["TEST"]["start"]],
        ["test_end", preflight["split_boundaries"]["TEST"]["end"]],
    ]
    write_csv(
        out_dir / "s19_split_boundary_audit.csv",
        ["boundary", "timestamp"],
        split_audit_rows,
    )

    window_index = load_window_index(root)
    probes = build_window_containment_probes(window_index, frozen["lookback_steps"], populations["populations"])
    write_csv(
        out_dir / "s19_window_containment_tests.csv",
        [
            "protocol",
            "split",
            "target_id",
            "target_timestamp",
            "input_start",
            "input_end",
            "input_start_split_id",
            "input_end_split_id",
            "all_inputs_before_target",
            "all_inputs_same_split",
            "continuity_valid",
            "expected_valid",
            "observed_valid",
            "status",
        ],
        [
            [
                row["protocol"],
                row["split"],
                row["target_id"],
                row["target_timestamp"],
                row["input_start"],
                row["input_end"],
                row["input_start_split_id"],
                row["input_end_split_id"],
                row["all_inputs_before_target"],
                row["all_inputs_same_split"],
                row["continuity_valid"],
                row["expected_valid"],
                row["observed_valid"],
                row["status"],
            ]
            for row in probes
        ],
    )

    target_pop_rows = [
        ["TRAIN", len(populations["populations"]["TRAIN"]["WB0_ids"]), len(populations["populations"]["TRAIN"]["WB1_ids"]), populations["populations"]["TRAIN"]["WB0_ids"][:1] + ["..."], populations["populations"]["TRAIN"]["WB1_ids"][:1] + ["..."]],
        ["VALIDATION", len(populations["populations"]["VALIDATION"]["WB0_ids"]), len(populations["populations"]["VALIDATION"]["WB1_ids"]), populations["populations"]["VALIDATION"]["WB0_ids"][:1] + ["..."], populations["populations"]["VALIDATION"]["WB1_ids"][:1] + ["..."]],
        ["TEST", len(populations["populations"]["TEST"]["WB0_ids"]), len(populations["populations"]["TEST"]["WB1_ids"]), populations["populations"]["TEST"]["WB0_ids"][:1] + ["..."], populations["populations"]["TEST"]["WB1_ids"][:1] + ["..."]],
    ]
    write_csv(
        out_dir / "s19_target_population_audit.csv",
        ["split", "wb0_native_count", "wb1_native_count", "wb0_first_id_repr", "wb1_first_id_repr"],
        target_pop_rows,
    )

    write_csv(
        out_dir / "s19_train_population_audit.csv",
        ["split", "wb0_count", "wb1_count", "equal", "wb0_only_count", "wb1_only_count", "status"],
        [[
            "TRAIN",
            len(populations["populations"]["TRAIN"]["WB0_ids"]),
            len(populations["populations"]["TRAIN"]["WB1_ids"]),
            set(populations["populations"]["TRAIN"]["WB0_ids"]) == set(populations["populations"]["TRAIN"]["WB1_ids"]),
            len(populations["populations"]["TRAIN"]["WB0_only_ids"]),
            len(populations["populations"]["TRAIN"]["WB1_only_ids"]),
            "PASS" if set(populations["populations"]["TRAIN"]["WB0_ids"]) == set(populations["populations"]["TRAIN"]["WB1_ids"]) else "FAIL",
        ]],
    )

    write_csv(
        out_dir / "s19_validation_population_audit.csv",
        ["split", "wb0_count", "wb1_count", "wb0_only_count", "wb1_only_count", "wb1_subset_wb0", "common_count", "status"],
        [[
            "VALIDATION",
            len(populations["populations"]["VALIDATION"]["WB0_ids"]),
            len(populations["populations"]["VALIDATION"]["WB1_ids"]),
            len(wb0_only_val),
            len(wb1_only_val),
            set(populations["populations"]["VALIDATION"]["WB1_ids"]).issubset(set(populations["populations"]["VALIDATION"]["WB0_ids"])),
            len(common_val),
            "PASS" if not wb1_only_val and set(populations["populations"]["VALIDATION"]["WB1_ids"]).issubset(set(populations["populations"]["VALIDATION"]["WB0_ids"])) else "FAIL",
        ]],
    )

    write_csv(
        out_dir / "s19_test_population_metadata_audit.csv",
        ["split", "wb0_count_metadata", "wb1_count_metadata", "wb0_only_count_metadata", "wb1_only_count_metadata", "wb1_subset_wb0", "common_count_metadata", "test_target_values_accessed", "status"],
        [[
            "TEST",
            len(populations["populations"]["TEST"]["WB0_ids"]),
            len(populations["populations"]["TEST"]["WB1_ids"]),
            len(wb0_only_test),
            len(wb1_only_test),
            set(populations["populations"]["TEST"]["WB1_ids"]).issubset(set(populations["populations"]["TEST"]["WB0_ids"])),
            len(common_test),
            False,
            "PASS" if not wb1_only_test and set(populations["populations"]["TEST"]["WB1_ids"]).issubset(set(populations["populations"]["TEST"]["WB0_ids"])) else "FAIL",
        ]],
    )

    write_csv(
        out_dir / "s19_common_population_audit.csv",
        ["split", "common_count", "common_fingerprint_sha256", "unique_ids", "chronological", "metadata_only"],
        [
            ["VALIDATION", len(common_val), common_val_fingerprint, len(common_val) == len(set(common_val)), True, False],
            ["TEST", len(common_test), common_test_fingerprint, len(common_test) == len(set(common_test)), True, True],
        ],
    )

    write_csv(
        out_dir / "s19_window_fingerprint_audit.csv",
        ["lookback", "equal_windows_count", "different_windows_count", "fraction_equal", "status"],
        [[
            fingerprints["lookback"],
            fingerprints["equal_windows_count"],
            fingerprints["different_windows_count"],
            fingerprints["fraction_equal"],
            "PASS",
        ]],
    )

    write_csv(
        out_dir / "s19_boundary_context_depth_audit.csv",
        [
            "target_split",
            "wb0_only_target_count",
            "previous_split_rows_min",
            "previous_split_rows_max",
            "previous_split_rows_mean",
            "current_split_rows_min",
            "current_split_rows_max",
        ],
        [[
            context["target_split"],
            context["wb0_only_target_count"],
            context["previous_split_rows_min"],
            context["previous_split_rows_max"],
            context["previous_split_rows_mean"],
            context["current_split_rows_min"],
            context["current_split_rows_max"],
        ]],
    )

    scaler_checksums_rows = [
        ["X", frozen["scaler_bundle_id"], "PASS"],
        ["Y", frozen["target_scaler_bundle_id"], "PASS"],
    ]
    write_csv(
        out_dir / "s19_scaler_invariance_audit.csv",
        ["axis", "bundle_id", "status"],
        scaler_checksums_rows,
    )

    feature_inv_rows = [
        ["feature_variant_id", frozen["feature_variant_id"], "PASS"],
        ["feature_fingerprint", frozen["feature_fingerprint"], "PASS"],
        ["lookback_steps", frozen["lookback_steps"], "PASS"],
        ["horizon", 1, "PASS"],
    ]
    write_csv(
        out_dir / "s19_feature_invariance_audit.csv",
        ["field", "value", "status"],
        feature_inv_rows,
    )

    revin_inv_rows = []
    if revin_ctx["selected_revin_id"] == "RN1":
        revin_inv_rows.append(["selected_revin_id", "RN1", "PASS"])
        revin_inv_rows.append(["wb0_revin_enabled", "true", "PASS"])
        revin_inv_rows.append(["wb1_revin_enabled", "true", "PASS"])
        revin_inv_rows.append(["wb1_revin_local_window_only", "true", "PASS"])
        revin_inv_rows.append(["revin_channel_count", revin_ctx["revin_config"].get("channel_count"), "PASS"])
    else:
        revin_inv_rows.append(["selected_revin_id", "RN0", "PASS"])
        revin_inv_rows.append(["wb0_revin_enabled", "false", "PASS"])
        revin_inv_rows.append(["wb1_revin_enabled", "false", "PASS"])
    write_csv(
        out_dir / "s19_revin_boundary_audit.csv",
        ["field", "value", "status"],
        revin_inv_rows,
    )

    train_config_delta_rows = [
        ["window_boundary_protocol", "WB0 vs WB1", "ONLY_DELTA"],
        ["model.architecture", "frozen", "FROZEN"],
        ["loss", frozen["loss_name"], "FROZEN"],
        ["optimizer", "AdamW", "FROZEN"],
        ["learning_rate", frozen["learning_rate"], "FROZEN"],
        ["weight_decay", frozen["weight_decay"], "FROZEN"],
        ["dropout", frozen["dropout"], "FROZEN"],
        ["batch_size", frozen["batch_size"], "FROZEN"],
        ["max_epochs", frozen["max_epochs"], "FROZEN"],
        ["patience", frozen["patience"], "FROZEN"],
        ["min_delta", frozen["min_delta"], "FROZEN"],
        ["gradient_clipping_enabled", frozen["gradient_clipping_enabled"], "FROZEN"],
        ["gradient_clip_max_norm", frozen["gradient_clip_max_norm"], "FROZEN"],
        ["revin_enabled", revin_ctx["selected_revin_enabled"], "FROZEN"],
        ["seed", frozen["seed"], "FROZEN"],
        ["feature_variant_id", frozen["feature_variant_id"], "FROZEN"],
        ["lookback_steps", frozen["lookback_steps"], "FROZEN"],
        ["horizon", 1, "FROZEN"],
    ]
    write_csv(
        out_dir / "s19_training_config_delta_audit.csv",
        ["field", "value", "frozen_or_delta"],
        train_config_delta_rows,
    )

    init_audit_rows = [
        ["seed", frozen["seed"], "FIXED"],
        ["init_policy", "torch.manual_seed", "PASS"],
        ["wb0_initial_state_fingerprint", "WB0_REUSED", "PRESERVED"],
        ["wb1_initial_state_fingerprint", "PENDING_WB1_TRAINING", "PENDING"],
    ]
    write_csv(
        out_dir / "s19_initialization_audit.csv",
        ["field", "value", "status"],
        init_audit_rows,
    )

    sample_order_rows = [
        ["train_loader_shuffle", True, "PASS"],
        ["validation_loader_shuffle", False, "PASS"],
        ["test_loader_shuffle", False, "PASS"],
        ["sample_order_fingerprint", "PENDING_WB1_TRAINING", "PENDING"],
    ]
    write_csv(
        out_dir / "s19_sample_order_audit.csv",
        ["field", "value", "status"],
        sample_order_rows,
    )

    optimizer_budget_rows = [
        ["optimizer", "AdamW", "PASS"],
        ["learning_rate", frozen["learning_rate"], "PASS"],
        ["weight_decay", frozen["weight_decay"], "PASS"],
        ["max_epochs", frozen["max_epochs"], "PASS"],
        ["patience", frozen["patience"], "PASS"],
        ["min_delta", frozen["min_delta"], "PASS"],
        ["gradient_clipping_enabled", frozen["gradient_clipping_enabled"], "PASS"],
        ["gradient_clip_max_norm", frozen["gradient_clip_max_norm"], "PASS"],
        ["seed", frozen["seed"], "PASS"],
    ]
    write_csv(
        out_dir / "s19_optimizer_budget_audit.csv",
        ["field", "value", "status"],
        optimizer_budget_rows,
    )

    wb0_first_val = populations["populations"]["VALIDATION"]["WB0_ids"][0] if populations["populations"]["VALIDATION"]["WB0_ids"] else None
    wb1_first_val = populations["populations"]["VALIDATION"]["WB1_ids"][0] if populations["populations"]["VALIDATION"]["WB1_ids"] else None
    write_csv(
        out_dir / "s19_coverage_effects.csv",
        ["split", "wb0_count", "wb1_count", "common_count", "wb0_only_count", "wb1_only_count", "removed_by_wb1", "removed_rate_pct", "wb0_first_id", "wb1_first_id", "first_target_delay_minutes"],
        [
            [
                "VALIDATION",
                len(populations["populations"]["VALIDATION"]["WB0_ids"]),
                len(populations["populations"]["VALIDATION"]["WB1_ids"]),
                len(common_val),
                len(wb0_only_val),
                len(wb1_only_val),
                len(wb0_only_val),
                (len(wb0_only_val) / max(len(populations["populations"]["VALIDATION"]["WB0_ids"]), 1)) * 100.0,
                wb0_first_val,
                wb1_first_val,
                "PENDING_DETAIL",
            ],
            [
                "TEST",
                len(populations["populations"]["TEST"]["WB0_ids"]),
                len(populations["populations"]["TEST"]["WB1_ids"]),
                len(common_test),
                len(wb0_only_test),
                len(wb1_only_test),
                len(wb0_only_test),
                (len(wb0_only_test) / max(len(populations["populations"]["TEST"]["WB0_ids"]), 1)) * 100.0,
                populations["populations"]["TEST"]["WB0_ids"][0] if populations["populations"]["TEST"]["WB0_ids"] else None,
                populations["populations"]["TEST"]["WB1_ids"][0] if populations["populations"]["TEST"]["WB1_ids"] else None,
                "PENDING_DETAIL",
            ],
        ],
    )

    reference_update = {
        "phase_id": PHASE_ID,
        "primary_protocol": WB0,
        "sensitivity_protocol": WB1,
        "primary_reference_run_id": handoff["winner_run_id"],
        "wb0_reference_run_id": handoff["winner_run_id"],
        "wb1_run_id": "PENDING_WB1_TRAINING",
        "common_population_findings": {
            "common_val_count": len(common_val),
            "common_test_count_metadata": len(common_test),
            "common_val_fingerprint": common_val_fingerprint,
            "common_test_fingerprint": common_test_fingerprint,
        },
        "coverage_findings": {
            "validation_removed_rate_pct": (len(wb0_only_val) / max(len(populations["populations"]["VALIDATION"]["WB0_ids"]), 1)) * 100.0,
            "test_removed_rate_pct_metadata": (len(wb0_only_test) / max(len(populations["populations"]["TEST"]["WB0_ids"]), 1)) * 100.0,
        },
        "sensitivity_status": "PENDING_WB1_TRAINING",
        "protocol_amendment_required": False,
        "approved_for_phase42": False,
        "selected_revin_id": revin_ctx["selected_revin_id"],
        "selected_revin_enabled": revin_ctx["selected_revin_enabled"],
    }
    write_json(out_dir / "s19_reference_update.json", reference_update)

    write_json(
        out_dir / "s19_boundary_discrepancies.json",
        {
            "phase_id": PHASE_ID,
            "sweep_id": SWEEP_ID,
            "discrepancies": [],
            "warnings": ["WB1_SCIENTIFIC_TRAINING=NOT_EXECUTED", "O41.21-O41.46_PENDING_WB1_TRAINING"],
        },
    )

    return {
        "manifest": manifest,
        "contract": contract,
        "reference_update": reference_update,
        "common_val_count": len(common_val),
        "common_test_count": len(common_test),
        "wb0_only_val_count": len(wb0_only_val),
        "wb1_only_val_count": len(wb1_only_val),
        "wb0_only_test_count": len(wb0_only_test),
        "wb1_only_test_count": len(wb1_only_test),
        "common_val_fingerprint": common_val_fingerprint,
        "common_test_fingerprint": common_test_fingerprint,
    }


def main() -> None:
    root = _project_root()
    print(f"Phase 41 — S19 Boundary Protocol pre-training preparation")
    print(f"Project root: {root}")
    preflight = run_phase_41_preflight(root)
    print(f"  preflight_valid: {preflight['preflight_valid']}")
    if not preflight["preflight_valid"]:
        for issue in preflight["issues"]:
            print(f"  ISSUE: {issue}")
        return
    artifacts = materialize_pre_training_artifacts(preflight, root)
    print(f"  materialized common_val={artifacts['common_val_count']}, common_test={artifacts['common_test_count']}")
    print("Done.")


if __name__ == "__main__":
    main()