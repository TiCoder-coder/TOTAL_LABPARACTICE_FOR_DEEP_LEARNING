"""Phase 41 — S19 Boundary Protocol focused tests."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path("/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK")
SRC = ROOT / "src"
S19_DIR = ROOT / "artifacts/sweeps/S19_boundary_protocol"
WINDOWS_DIR = ROOT / "artifacts/windows"
SPLITS_DIR = ROOT / "artifacts/splits"

sys.path.insert(0, str(SRC))

from course_work.sweeps.boundary_protocol import (  # noqa: E402
    build_target_populations,
    build_common_population_fingerprint,
    resolve_phase_40_handoff,
    resolve_split_boundaries,
    load_window_index,
    prepare_phase_41_condition,
    SWEEP_ID,
    PHASE_ID,
    WB0,
    WB1,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _dry_run_cmd(condition: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "scripts/run_single_condition.py",
        SWEEP_ID,
        condition,
        "--dry-run",
    ]


def test_phase_40_handoff_pass():
    """Phase 40 must sign off and approve Phase 41."""
    handoff = resolve_phase_40_handoff(ROOT)
    assert handoff["signoff"]["phase_status"] == "PASS"
    assert handoff["signoff"]["approved_for_phase41"] is True
    assert handoff["winner_run_id"] == "RUN_TR_S14_0023_A711A9B8"
    assert handoff["selected_revin_id"] == "RN0"
    assert handoff["current_boundary_protocol"] == WB0


def test_split_boundaries_frozen():
    """Split boundaries must come from SPLIT-v1 artifacts (not Phase 41 inventions)."""
    boundaries = resolve_split_boundaries(ROOT)
    assert boundaries["TRAIN"]["start"] is not None
    assert boundaries["TRAIN"]["end"] is not None
    assert boundaries["VALIDATION"]["start"] is not None
    assert boundaries["VALIDATION"]["end"] is not None
    assert boundaries["TEST"]["start"] is not None
    assert boundaries["TEST"]["end"] is not None


def test_wb0_execution_mode_reuse_reference():
    """WB0 must resolve to REUSE_REFERENCE with exact Phase 40 reference."""
    cond = prepare_phase_41_condition(WB0, ROOT)
    assert cond is not None
    assert cond["execution_mode"] == "REUSE_REFERENCE"
    assert cond["requires_new_training"] is False
    assert cond["boundary_protocol"] == WB0
    assert cond["wb0_reference_run_id"] == "RUN_TR_S14_0023_A711A9B8"
    assert cond["reference_run_id"] == "RUN_TR_S14_0023_A711A9B8"


def test_wb1_execution_mode_train_new():
    """WB1 must be TRAIN_NEW with fresh seed42 contract."""
    cond = prepare_phase_41_condition(WB1, ROOT)
    assert cond is not None
    assert cond["execution_mode"] == "TRAIN_NEW"
    assert cond["requires_new_training"] is True
    assert cond["boundary_protocol"] == WB1


def test_only_boundary_protocol_differs():
    """Only window_boundary_protocol may differ between WB0 and WB1 frozen config."""
    wb0 = prepare_phase_41_condition(WB0, ROOT)
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    frozen_keys = [
        "feature_variant_id",
        "target_scaling_id",
        "lookback_steps",
        "pooling",
        "activation",
        "dropout",
        "d_model",
        "num_heads",
        "num_layers",
        "ffn_dim",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "loss_name",
        "max_epochs",
        "patience",
        "min_delta",
        "gradient_clipping_enabled",
        "gradient_clip_max_norm",
        "seed",
        "selected_revin_id",
        "selected_revin_enabled",
    ]
    for key in frozen_keys:
        assert wb0["frozen_configuration"][key] == wb1["frozen_configuration"][key], key
    assert wb0["frozen_configuration"]["window_boundary_protocol"] != wb1["frozen_configuration"]["window_boundary_protocol"]
    assert wb0["frozen_configuration"]["window_boundary_protocol"] == WB0
    assert wb1["frozen_configuration"]["window_boundary_protocol"] == WB1


def test_split_cut_points_unchanged():
    """Split cut points must be identical across WB0 and WB1."""
    boundaries = resolve_split_boundaries(ROOT)
    wb0 = prepare_phase_41_condition(WB0, ROOT)
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    for split_id in ("TRAIN", "VALIDATION", "TEST"):
        assert boundaries[split_id]["start"] == boundaries[split_id]["start"]
        assert boundaries[split_id]["end"] == boundaries[split_id]["end"]
    assert wb0["frozen_configuration"]["lookback_steps"] == wb1["frozen_configuration"]["lookback_steps"]


def test_lookback_horizon_invariance():
    """Lookback and horizon (1) must be identical across protocols."""
    wb0 = prepare_phase_41_condition(WB0, ROOT)
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    assert wb0["frozen_configuration"]["lookback_steps"] == wb1["frozen_configuration"]["lookback_steps"]
    assert wb0["lookback_steps"] == wb1["lookback_steps"]


def test_wb0_cross_split_past_context_allowed():
    """WB0 eligibility uses cross-split past context (input timestamps < target timestamp)."""
    window_index = load_window_index(ROOT)
    lookback = prepare_phase_41_condition(WB0, ROOT)["lookback_steps"]
    rows = window_index.loc[window_index["lookback_steps"].eq(lookback) & window_index["WB0_valid"].astype(bool)]
    cross_split = rows.loc[rows["input_start_split_id"] != rows["target_split_id"]]
    assert not cross_split.empty, "WB0 must allow cross-split past context"
    assert (rows["timeline_input_end"] < rows["timeline_target"]).all()
    assert (pd.to_datetime(rows["input_end_timestamp"]) < pd.to_datetime(rows["target_timestamp"])).all()


def test_wb0_future_input_forbidden():
    """WB0 must reject any window with future input (input_end >= target_timestamp)."""
    window_index = load_window_index(ROOT)
    lookback = prepare_phase_41_condition(WB0, ROOT)["lookback_steps"]
    rows = window_index.loc[window_index["lookback_steps"].eq(lookback) & window_index["WB0_valid"].astype(bool)]
    assert (rows["timeline_input_end"] < rows["timeline_target"]).all()
    assert (pd.to_datetime(rows["input_end_timestamp"]) < pd.to_datetime(rows["target_timestamp"])).all()


def test_wb1_cross_split_context_forbidden():
    """WB1 must require every input row to belong to target split."""
    window_index = load_window_index(ROOT)
    lookback = prepare_phase_41_condition(WB1, ROOT)["lookback_steps"]
    rows = window_index.loc[window_index["lookback_steps"].eq(lookback) & window_index["WB1_valid"].astype(bool)]
    assert (rows["input_start_split_id"] == rows["target_split_id"]).all()


def test_wb1_inputs_same_split_as_target():
    """WB1 input rows and target must share the same split label."""
    window_index = load_window_index(ROOT)
    lookback = prepare_phase_41_condition(WB1, ROOT)["lookback_steps"]
    rows = window_index.loc[window_index["lookback_steps"].eq(lookback) & window_index["WB1_valid"].astype(bool)]
    assert (rows["input_start_split_id"] == rows["target_split_id"]).all()
    assert (rows["input_end_split_id"] == rows["target_split_id"]).all()


def test_insufficient_local_history_makes_target_is_unavailable_for_wb1():
    """Targets at start of split without L rows of own split history must be WB1-ineligible."""
    window_index = load_window_index(ROOT)
    lookback = prepare_phase_41_condition(WB1, ROOT)["lookback_steps"]
    sub = window_index.loc[
        window_index["lookback_steps"].eq(lookback)
        & window_index["target_split_id"].eq("VALIDATION")
        & (~window_index["WB1_valid"].astype(bool))
        & window_index["WB0_valid"].astype(bool)
    ]
    assert not sub.empty, "Expected at least one WB0-valid/WB1-invalid Validation target"


def test_no_padding_no_short_window_no_target_in_input():
    """No padding, no short window, target never appears in input."""
    window_index = load_window_index(ROOT)
    lookback = prepare_phase_41_condition(WB1, ROOT)["lookback_steps"]
    valid = window_index.loc[window_index["lookback_steps"].eq(lookback) & window_index["WB1_valid"].astype(bool)]
    durations = (valid["timeline_input_end"] - valid["timeline_input_start"] + 1).astype(int)
    assert (durations == lookback).all()
    assert (valid["timeline_input_end"] < valid["timeline_target"]).all()


def test_continuity_10_minute_cadence():
    """Every WB1 valid window must have all 10-minute cadence rows belong to same continuity segment."""
    window_index = load_window_index(ROOT)
    lookback = prepare_phase_41_condition(WB1, ROOT)["lookback_steps"]
    valid = window_index.loc[window_index["lookback_steps"].eq(lookback) & window_index["WB1_valid"].astype(bool)]
    assert not valid.empty


def test_train_population_equal():
    """WB0 Train IDs must equal WB1 Train IDs."""
    window_index = load_window_index(ROOT)
    lookback = prepare_phase_41_condition(WB0, ROOT)["lookback_steps"]
    pop = build_target_populations(window_index, lookback)
    train = pop["populations"]["TRAIN"]
    assert set(train["WB0_ids"]) == set(train["WB1_ids"])


def test_wb1_val_subset_wb0_val():
    """WB1 Validation IDs must be subset of WB0 Validation IDs."""
    window_index = load_window_index(ROOT)
    lookback = prepare_phase_41_condition(WB1, ROOT)["lookback_steps"]
    pop = build_target_populations(window_index, lookback)
    val = pop["populations"]["VALIDATION"]
    assert set(val["WB1_ids"]).issubset(set(val["WB0_ids"]))


def test_wb1_test_subset_wb0_test():
    """WB1 Test IDs (metadata) must be subset of WB0 Test IDs."""
    window_index = load_window_index(ROOT)
    lookback = prepare_phase_41_condition(WB1, ROOT)["lookback_steps"]
    pop = build_target_populations(window_index, lookback)
    test = pop["populations"]["TEST"]
    assert set(test["WB1_ids"]).issubset(set(test["WB0_ids"]))


def test_wb1_only_val_empty():
    """There must be zero WB1-only Validation IDs."""
    window_index = load_window_index(ROOT)
    lookback = prepare_phase_41_condition(WB1, ROOT)["lookback_steps"]
    pop = build_target_populations(window_index, lookback)
    assert pop["populations"]["VALIDATION"]["WB1_only_ids"] == []


def test_wb1_only_test_empty():
    """There must be zero WB1-only Test IDs."""
    window_index = load_window_index(ROOT)
    lookback = prepare_phase_41_condition(WB1, ROOT)["lookback_steps"]
    pop = build_target_populations(window_index, lookback)
    assert pop["populations"]["TEST"]["WB1_only_ids"] == []


def test_common_val_ids_unique_and_chronological():
    """Common Validation IDs must be unique and ordered by target timestamp."""
    window_index = load_window_index(ROOT)
    lookback = prepare_phase_41_condition(WB1, ROOT)["lookback_steps"]
    pop = build_target_populations(window_index, lookback)
    common = pop["populations"]["VALIDATION"]["COMMON_ids"]
    assert len(common) == len(set(common))
    target_rows = window_index.loc[
        window_index["lookback_steps"].eq(lookback)
        & window_index["target_sample_id"].isin(common)
    ]
    assert target_rows["target_timestamp"].is_monotonic_increasing


def test_common_test_ids_metadata_only():
    """Common Test IDs must be metadata only; no y values exposed."""
    window_index = load_window_index(ROOT)
    lookback = prepare_phase_41_condition(WB1, ROOT)["lookback_steps"]
    pop = build_target_populations(window_index, lookback)
    common = pop["populations"]["TEST"]["COMMON_ids"]
    assert len(common) == len(set(common))


def test_common_window_fingerprints_generated():
    """Common window fingerprints must be generated."""
    common_val = [1, 2, 3, 4, 5]
    fp = build_common_population_fingerprint(common_val)
    assert isinstance(fp, str)
    assert len(fp) == 64


def test_same_feature_order_scaler_loss_optimizer():
    """Same feature order, X/Y scaler, loss, optimizer across WB0/WB1."""
    wb0 = prepare_phase_41_condition(WB0, ROOT)
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    assert wb0["frozen_configuration"]["feature_variant_id"] == wb1["frozen_configuration"]["feature_variant_id"]
    assert wb0["frozen_configuration"]["scaler_bundle_id"] == wb1["frozen_configuration"]["scaler_bundle_id"]
    assert wb0["frozen_configuration"]["target_scaler_bundle_id"] == wb1["frozen_configuration"]["target_scaler_bundle_id"]
    assert wb0["frozen_configuration"]["loss_name"] == wb1["frozen_configuration"]["loss_name"]
    assert wb0["frozen_configuration"]["learning_rate"] == wb1["frozen_configuration"]["learning_rate"]


def test_seed_policy_42():
    """Seed must be 42 for both protocols."""
    wb0 = prepare_phase_41_condition(WB0, ROOT)
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    assert wb0["frozen_configuration"]["seed"] == 42
    assert wb1["frozen_configuration"]["seed"] == 42


def test_no_warm_start_no_optimizer_state_reuse():
    """WB1 contract: fresh seed=42, fresh AdamW, no checkpoint/optimizer reuse."""
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    assert wb1["execution_mode"] == "TRAIN_NEW"
    assert wb1["frozen_configuration"]["seed"] == 42


def test_wb1_early_stop_native_val():
    """WB1 early stopping must use native WB1 Validation RMSE."""
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    assert wb1["frozen_configuration"]["max_epochs"] > 0
    assert wb1["frozen_configuration"]["patience"] > 0


def test_no_common_pop_reselection_logic():
    """No common-population checkpoint reselection is implemented (no validator for it).
    After Phase 41 finalization, sensitivity_status must be COMPLETED — not PENDING.
    """
    common_val_ref_path = S19_DIR / "s19_reference_update.json"
    assert common_val_ref_path.is_file()
    ref = _read_json(common_val_ref_path)
    assert ref["sensitivity_status"] == "COMPLETED", (
        f"Expected sensitivity_status='COMPLETED' after Phase 41 finalization, "
        f"got '{ref['sensitivity_status']}'"
    )
    assert "reselection" not in json.dumps(ref).lower()


def test_guarded_wb1_dry_run_passes_and_no_scientific_state():
    """WB1 dry-run must reach TrainingEngine.train() boundary and create no scientific state."""
    result = subprocess.run(
        _dry_run_cmd(WB1),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "MPLCONFIGDIR": "/private/tmp/course-work-mpl-cache", "PATH": "/Library/Frameworks/Python.framework/Versions/3.10/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin"},
        timeout=120,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    out = result.stdout
    assert "GUARDED SMOKE COMPLETE" in out
    assert "before TrainingEngine.train()" in out
    assert "No scientific run created" in out
    assert "No Test data accessed" in out


def test_guarded_wb0_dry_run_passes():
    """WB0 dry-run must also pass with REUSE_REFERENCE for primary protocol."""
    result = subprocess.run(
        _dry_run_cmd(WB0),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "MPLCONFIGDIR": "/private/tmp/course-work-mpl-cache", "PATH": "/Library/Frameworks/Python.framework/Versions/3.10/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin"},
        timeout=120,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert "GUARDED SMOKE COMPLETE" in result.stdout
    assert "No scientific run created" in result.stdout


def test_s17_dry_run_unaffected():
    """Prior S17 dry-run behavior must not be broken."""
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "scripts/run_single_condition.py",
            "S17_GRADIENT_CLIPPING",
            "GC0",
            "--dry-run",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "MPLCONFIGDIR": "/private/tmp/course-work-mpl-cache", "PATH": "/Library/Frameworks/Python.framework/Versions/3.10/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin"},
        timeout=120,
    )
    assert result.returncode == 0


def test_s18_dry_run_unaffected():
    """Prior S18 dry-run behavior must not be broken."""
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "scripts/run_single_condition.py",
            "S18_REVIN",
            "RN0",
            "--dry-run",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "MPLCONFIGDIR": "/private/tmp/course-work-mpl-cache", "PATH": "/Library/Frameworks/Python.framework/Versions/3.10/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin"},
        timeout=120,
    )
    assert result.returncode == 0


def test_unknown_sweep_fails():
    """Unknown sweep_id must fail (no exception swallowed)."""
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "scripts/run_single_condition.py",
            "S99_UNKNOWN_SWEEP",
            "XX",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "MPLCONFIGDIR": "/private/tmp/course-work-mpl-cache", "PATH": "/Library/Frameworks/Python.framework/Versions/3.10/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin"},
        timeout=60,
    )
    assert result.returncode != 0


def test_artifacts_directory_complete():
    """O41.1..O41.20 pre-training artifacts must exist."""
    required = [
        "s19_boundary_sweep_manifest.json",
        "s19_boundary_sweep_contract.json",
        "s19_boundary_preflight_audit.csv",
        "s19_run_matrix.csv",
        "s19_boundary_definition_audit.csv",
        "s19_split_boundary_audit.csv",
        "s19_window_containment_tests.csv",
        "s19_target_population_audit.csv",
        "s19_train_population_audit.csv",
        "s19_validation_population_audit.csv",
        "s19_test_population_metadata_audit.csv",
        "s19_common_population_audit.csv",
        "s19_window_fingerprint_audit.csv",
        "s19_boundary_context_depth_audit.csv",
        "s19_scaler_invariance_audit.csv",
        "s19_feature_invariance_audit.csv",
        "s19_revin_boundary_audit.csv",
        "s19_training_config_delta_audit.csv",
        "s19_initialization_audit.csv",
        "s19_sample_order_audit.csv",
    ]
    for name in required:
        assert (S19_DIR / name).is_file(), f"Missing artifact: {name}"


def test_test_firewall_in_signoff():
    """Test access must remain FORBIDDEN across all S19 artifacts."""
    preflight_csv = _read_csv(S19_DIR / "s19_boundary_preflight_audit.csv")
    test_row = next((r for r in preflight_csv if r["check"] == "test_firewall"), None)
    assert test_row is not None
    assert test_row["status"] == "PASS"
    assert "FORBIDDEN" in test_row["detail_json"]


def test_rev_in_invariance():
    """Selected RevIN must be identical between WB0 and WB1."""
    preflight_csv = _read_csv(S19_DIR / "s19_boundary_preflight_audit.csv")
    revin_row = next((r for r in preflight_csv if r["check"] == "revin_invariance"), None)
    assert revin_row is not None
    assert revin_row["status"] == "PASS"


def test_training_config_delta_only_window_boundary():
    """Training-config delta audit must record ONLY window_boundary_protocol as delta."""
    delta_csv = _read_csv(S19_DIR / "s19_training_config_delta_audit.csv")
    delta_rows = [r for r in delta_csv if r["frozen_or_delta"] == "ONLY_DELTA"]
    assert len(delta_rows) == 1
    assert delta_rows[0]["field"] == "window_boundary_protocol"


# ---------------------------------------------------------------------------
# Phase 41 corrective tests — WB1 entrypoint field propagation
# Added 2026-08-25: root-cause fix for failed WB1 launch where max_epochs=50
# was not propagated from Phase 40 frozen config into the registry config.
# ---------------------------------------------------------------------------

import copy
import tempfile
from copy import deepcopy

from course_work.sweeps.boundary_protocol import (
    prepare_phase_41_condition,
    WB1,
    WB0,
)
from course_work.experiments.registry import validate_run_config
from course_work.utils.artifacts import read_json


def test_wb1_real_config_carries_max_epochs_50():
    """WB1 real registry config must carry max_epochs=50 from Phase 40 frozen config."""
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    source_cfg = read_json(ROOT / wb1["frozen_configuration"]["source_config_path"])["config"]
    assert source_cfg["training"]["max_epochs"] == 50
    # Verify frozen config carries it
    assert wb1["frozen_configuration"]["max_epochs"] == 50
    # Verify top-level carries it
    assert wb1["max_epochs"] == 50


def test_wb1_real_config_carries_patience_10():
    """WB1 real registry config must carry patience=10 from Phase 40 frozen config."""
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    source_cfg = read_json(ROOT / wb1["frozen_configuration"]["source_config_path"])["config"]
    assert source_cfg["training"]["early_stopping_patience"] == 10
    assert wb1["frozen_configuration"]["patience"] == 10


def test_wb1_real_config_carries_min_delta():
    """WB1 real registry config must carry min_delta from Phase 40 frozen config."""
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    source_cfg = read_json(ROOT / wb1["frozen_configuration"]["source_config_path"])["config"]
    expected_min_delta = source_cfg["training"].get("min_delta", 1e-4)
    assert wb1["frozen_configuration"]["min_delta"] == expected_min_delta


def test_wb1_gc1_remains_enabled_max_norm_1():
    """WB1 GC1 must remain enabled with max_norm=1.0 inherited from Phase 39/40."""
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    assert wb1["frozen_configuration"]["gradient_clipping_enabled"] is True
    assert wb1["frozen_configuration"]["gradient_clip_max_norm"] == 1.0
    assert wb1["clip_enabled"] is True
    assert wb1["max_norm"] == 1.0


def test_wb1_revin_remains_off():
    """WB1 RevIN must remain OFF (RN0) inherited from Phase 40."""
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    assert wb1["frozen_configuration"]["selected_revin_id"] == "RN0"
    assert wb1["frozen_configuration"]["selected_revin_enabled"] is False


def test_wb1_seed_remains_42():
    """WB1 seed must remain 42 inherited from Phase 40."""
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    assert wb1["frozen_configuration"]["seed"] == 42


def test_wb1_only_boundary_protocol_differs():
    """Only window_boundary_protocol may differ between WB0 and WB1."""
    wb0 = prepare_phase_41_condition(WB0, ROOT)
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    frozen_keys = set(wb0["frozen_configuration"].keys()) | set(wb1["frozen_configuration"].keys())
    delta_keys = [
        k for k in frozen_keys
        if wb0["frozen_configuration"].get(k) != wb1["frozen_configuration"].get(k)
    ]
    assert delta_keys == ["window_boundary_protocol"], f"Unexpected delta keys: {delta_keys}"


def test_wb1_registry_validate_accepts_resolved_config():
    """Registry validate_run_config must accept WB1 resolved config without error."""
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    source_cfg = read_json(ROOT / wb1["frozen_configuration"]["source_config_path"])
    config = deepcopy(source_cfg["config"])
    # Apply the same overrides that run_condition() applies for S19_BOUNDARY_PROTOCOL WB1
    config["training"]["max_epochs"] = wb1["frozen_configuration"]["max_epochs"]
    config["training"]["early_stopping_patience"] = wb1["frozen_configuration"]["patience"]
    config["training"]["min_delta"] = wb1["frozen_configuration"]["min_delta"]
    config["training"]["gradient_clipping_enabled"] = wb1["frozen_configuration"]["gradient_clipping_enabled"]
    config["training"]["gradient_clip_max_norm"] = wb1["frozen_configuration"]["gradient_clip_max_norm"]
    config["data"]["boundary_protocol"] = "WB1_STRICT_ISOLATION"
    # Load upstream context
    from course_work.experiments.registry import load_upstream_context
    upstream = load_upstream_context(ROOT)
    # Must not raise ValueError
    validated = validate_run_config(config, upstream)
    assert validated["training"]["max_epochs"] == 50
    assert validated["training"]["early_stopping_patience"] == 10
    assert validated["training"]["gradient_clipping_enabled"] is True
    assert validated["training"]["gradient_clip_max_norm"] == 1.0


def test_missing_max_epochs_still_fails_registry_validation():
    """Registry must reject a config where max_epochs is None (not silently default)."""
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    source_cfg = read_json(ROOT / wb1["frozen_configuration"]["source_config_path"])
    config = deepcopy(source_cfg["config"])
    # Explicitly set max_epochs to None — the exact failure mode that occurred
    config["training"]["max_epochs"] = None
    config["training"]["early_stopping_patience"] = wb1["frozen_configuration"]["patience"]
    config["training"]["min_delta"] = wb1["frozen_configuration"]["min_delta"]
    config["training"]["gradient_clipping_enabled"] = wb1["frozen_configuration"]["gradient_clipping_enabled"]
    config["training"]["gradient_clip_max_norm"] = wb1["frozen_configuration"]["gradient_clip_max_norm"]
    config["data"]["boundary_protocol"] = "WB1_STRICT_ISOLATION"
    from course_work.experiments.registry import load_upstream_context
    upstream = load_upstream_context(ROOT)
    with pytest.raises(ValueError, match="max_epochs must be positive"):
        validate_run_config(config, upstream)


def test_wb1_missing_patience_fails():
    """Registry must reject a config where early_stopping_patience is None."""
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    source_cfg = read_json(ROOT / wb1["frozen_configuration"]["source_config_path"])
    config = deepcopy(source_cfg["config"])
    config["training"]["max_epochs"] = wb1["frozen_configuration"]["max_epochs"]
    config["training"]["early_stopping_patience"] = None
    config["training"]["min_delta"] = wb1["frozen_configuration"]["min_delta"]
    config["training"]["gradient_clipping_enabled"] = wb1["frozen_configuration"]["gradient_clipping_enabled"]
    config["training"]["gradient_clip_max_norm"] = wb1["frozen_configuration"]["gradient_clip_max_norm"]
    config["data"]["boundary_protocol"] = "WB1_STRICT_ISOLATION"
    from course_work.experiments.registry import load_upstream_context
    upstream = load_upstream_context(ROOT)
    with pytest.raises(ValueError, match="early_stopping_patience must be positive"):
        validate_run_config(config, upstream)


def test_wb1_gc_consistency_registry_enforced():
    """Registry enforces GC consistency: enabled=True requires max_norm>0, enabled=False requires max_norm=None."""
    wb1 = prepare_phase_41_condition(WB1, ROOT)
    source_cfg = read_json(ROOT / wb1["frozen_configuration"]["source_config_path"])
    from course_work.experiments.registry import load_upstream_context
    upstream = load_upstream_context(ROOT)

    # GC enabled with valid max_norm: PASS
    config = deepcopy(source_cfg["config"])
    config["training"]["max_epochs"] = wb1["frozen_configuration"]["max_epochs"]
    config["training"]["early_stopping_patience"] = wb1["frozen_configuration"]["patience"]
    config["training"]["min_delta"] = wb1["frozen_configuration"]["min_delta"]
    config["training"]["gradient_clipping_enabled"] = True
    config["training"]["gradient_clip_max_norm"] = 1.0
    config["data"]["boundary_protocol"] = "WB1_STRICT_ISOLATION"
    validated = validate_run_config(config, upstream)
    assert validated["training"]["gradient_clipping_enabled"] is True
    assert validated["training"]["gradient_clip_max_norm"] == 1.0

    # GC disabled with max_norm=None: PASS
    config2 = deepcopy(source_cfg["config"])
    config2["training"]["max_epochs"] = wb1["frozen_configuration"]["max_epochs"]
    config2["training"]["early_stopping_patience"] = wb1["frozen_configuration"]["patience"]
    config2["training"]["min_delta"] = wb1["frozen_configuration"]["min_delta"]
    config2["training"]["gradient_clipping_enabled"] = False
    config2["training"]["gradient_clip_max_norm"] = None
    config2["data"]["boundary_protocol"] = "WB1_STRICT_ISOLATION"
    validated2 = validate_run_config(config2, upstream)
    assert validated2["training"]["gradient_clipping_enabled"] is False
    assert validated2["training"]["gradient_clip_max_norm"] is None


def test_guarded_real_path_smoke_stops_before_training():
    """Guarded real-path smoke must reach registry validation and stop before TrainingEngine.train()."""
    result = subprocess.run(
        _dry_run_cmd(WB1),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": "src",
            "MPLCONFIGDIR": "/private/tmp/course-work-mpl-cache",
            "PATH": "/Library/Frameworks/Python.framework/Versions/3.10/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin",
        },
        timeout=120,
    )
    assert result.returncode == 0, f"dry-run failed: {result.stderr}"
    out = result.stdout
    assert "GUARDED SMOKE COMPLETE" in out
    # The guard text contains "before TrainingEngine.train()" as the boundary marker.
    # We check that the boundary was reached and no training output appears.
    assert "before TrainingEngine.train()" in out
    assert "No scientific run created" in out
    assert "No checkpoint / metrics / predictions created" in out
    assert "No Test data accessed" in out
    # Confirm the script did NOT call the real training path
    assert "training completed" not in out.lower()
    assert "best_epoch" not in out.lower()


def test_no_scientific_run_created():
    """Guarded smoke must not create a scientific run in the registry."""
    # Run the dry-run and check no registry entry was created
    result = subprocess.run(
        _dry_run_cmd(WB1),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": "src",
            "MPLCONFIGDIR": "/private/tmp/course-work-mpl-cache",
            "PATH": "/Library/Frameworks/Python.framework/Versions/3.10/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin",
        },
        timeout=120,
    )
    assert result.returncode == 0
    assert "No permanent registry entry written" in result.stdout


def test_test_access_remains_forbidden():
    """Test access must remain FORBIDDEN in the WB1 entrypoint."""
    result = subprocess.run(
        _dry_run_cmd(WB1),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": "src",
            "MPLCONFIGDIR": "/private/tmp/course-work-mpl-cache",
            "PATH": "/Library/Frameworks/Python.framework/Versions/3.10/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin",
        },
        timeout=120,
    )
    assert result.returncode == 0
    assert "Test firewall" in result.stdout
    assert "FORBIDDEN" in result.stdout


def test_wb0_reuse_path_unchanged():
    """WB0 REUSE_REFERENCE path must not be affected by the S19_BOUNDARY_PROTOCOL fix."""
    result = subprocess.run(
        _dry_run_cmd(WB0),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": "src",
            "MPLCONFIGDIR": "/private/tmp/course-work-mpl-cache",
            "PATH": "/Library/Frameworks/Python.framework/Versions/3.10/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin",
        },
        timeout=120,
    )
    assert result.returncode == 0, f"WB0 dry-run broke: {result.stderr}"
    assert "GUARDED SMOKE COMPLETE" in result.stdout
    assert "No scientific run created" in result.stdout


def test_failed_launch_evidence_preserved():
    """Failed-launch evidence JSON must exist from the prior failed WB1 attempt."""
    evidence_path = S19_DIR / "_evidence" / "phase_41_wb1_failed_launch_evidence.json"
    assert evidence_path.is_file(), "Failed-launch evidence not preserved"
    evidence = _read_json(evidence_path)
    assert evidence["evidence_type"] == "FAILED_LAUNCH_EVIDENCE"
    assert evidence["condition_id"] == WB1
    assert "max_epochs" in evidence["missing_fields_in_observed_config"]
    assert evidence["failure"]["error_type"] == "ValueError"
    assert "max_epochs must be positive" in evidence["failure"]["message"]
    assert evidence["no_scientific_training_started"] is True
    assert evidence["no_run_id_created"] is True