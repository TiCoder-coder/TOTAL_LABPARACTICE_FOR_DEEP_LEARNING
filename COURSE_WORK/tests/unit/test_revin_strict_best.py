"""Phase 40 — S18 RevIN strict BEST verifier contract tests.

These tests lock down the contract for the strict BEST verifier per
Human-approved plan:
  COURSE_WORK/docs/plan-doc/plan_before_process/phase_40_s18_revin_strict_best_corrective_plan.md

Contract:
  - recorded_device must be honored (MPS when recorded as mps)
  - NO silent CPU fallback
  - canonical_tolerance = 1e-9 absolute
  - comparison_rule = absolute_only
  - rel_tolerance must be None
  - if MPS unavailable AND no external report, verifier returns
    NOT_VERIFIABLE_ON_CURRENT_DEVICE without raising
  - if MPS unavailable AND a validated external Human-verified MPS report is
    present, verifier accepts it as PASS evidence (no silent CPU fallback;
    same-device MPS verification done off-host)
  - if MPS available, verifier runs the on-host strict BEST path
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path('/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK')
sys.path.insert(0, str(ROOT / 'src'))

PHASE_40_DIR = ROOT / 'artifacts/sweeps/S18_revin'
PHASE_40_DIR.mkdir(parents=True, exist_ok=True)

VERIFIER_PATH = ROOT / 'scripts/phase40_strict_best_rn1.py'


def test_z1_verifier_module_loads():
    import importlib.util
    spec = importlib.util.spec_from_file_location('phase40_strict_best_rn1', VERIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CANONICAL_TOLERANCE == 1e-9
    assert module.RECORDED_DEVICE_REQUIRED == 'mps'
    assert module.RUN_ID == 'RUN_TR_S18_0031_A711A9B8'


def test_z2_canonical_tolerance_is_absolute_1e_9():
    import importlib.util
    spec = importlib.util.spec_from_file_location('phase40_strict_best_rn1', VERIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CANONICAL_TOLERANCE == 1e-9
    assert module.CANONICAL_TOLERANCE < 1e-6


def test_z3_resolve_recorded_device_or_fail_mps_unavailable():
    import importlib.util
    spec = importlib.util.spec_from_file_location('phase40_strict_best_rn1', VERIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch
    config = {'runtime': {'device_type': 'mps'}}
    available, recorded, reason = module._check_recorded_device_available(config)
    if not torch.backends.mps.is_available():
        assert available is False
        assert recorded == 'mps'
        assert 'MPS' in reason
    else:
        assert available is True


def test_z4_resolve_recorded_device_rejects_unknown():
    import importlib.util
    spec = importlib.util.spec_from_file_location('phase40_strict_best_rn1', VERIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {'runtime': {'device_type': 'tpu'}}
    available, recorded, reason = module._check_recorded_device_available(config)
    assert available is False
    assert 'unsupported recorded device' in reason


def test_z5_resolve_recorded_device_cpu_allowed():
    import importlib.util
    spec = importlib.util.spec_from_file_location('phase40_strict_best_rn1', VERIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {'runtime': {'device_type': 'cpu'}}
    available, recorded, reason = module._check_recorded_device_available(config)
    assert available is True
    assert recorded == 'cpu'


def test_z5b_main_returns_not_verifiable_when_mps_unavailable():
    """End-to-end: when MPS is unavailable on this host AND no valid external
    report is present, main() must return NOT_VERIFIABLE_ON_CURRENT_DEVICE
    without raising and without CPU fallback."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('phase40_strict_best_rn1', VERIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch
    if torch.backends.mps.is_available():
        pytest.skip("MPS is available; cannot exercise NOT_VERIFIABLE branch")
    # Patch config to claim recorded_device = mps (matches actual RN1 config)
    real_read_json = module.read_json
    real_run_root = module.Path("/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK") / "artifacts/runs" / module.RUN_ID
    cfg_path = real_run_root / "config.json"
    payload = real_read_json(cfg_path)
    if payload["config"]["runtime"]["device_type"] != "mps":
        pytest.skip("RN1 was not recorded on MPS; nothing to verify on this host")

    # Temporarily remove the external report if present, to exercise the
    # NOT_VERIFIABLE branch explicitly.
    ext_path = module.EXTERNAL_REPORT_PATH
    backup = None
    if ext_path.is_file():
        backup = ext_path.read_bytes()
        ext_path.unlink()
    try:
        status, payload = module.main()
    finally:
        if backup is not None:
            ext_path.write_bytes(backup)

    assert status == "NOT_VERIFIABLE_ON_CURRENT_DEVICE"
    assert payload["strict_best_verification"] == "NOT_VERIFIABLE_ON_CURRENT_DEVICE"
    assert payload["cpu_fallback"] == "DISABLED"
    assert payload["verification_device"] == "mps"
    assert payload["recorded_device"] == "mps"
    assert payload["comparison_rule"] == "absolute_only"
    assert payload["canonical_tolerance"] == 1e-9
    assert payload["rel_tolerance"] is None
    assert payload["metric_delta"] is None
    assert payload["recomputed_validation_metrics"] is None
    assert "FORBIDDEN" in payload["test_access"]


def test_z6_verifier_source_does_not_use_isclose_rel_tol():
    source = VERIFIER_PATH.read_text()
    assert 'rel_tol=' not in source
    assert 'math.isclose(' not in source


def test_z7_verifier_source_uses_absolute_comparison():
    source = VERIFIER_PATH.read_text()
    assert 'CANONICAL_TOLERANCE' in source
    assert '<= CANONICAL_TOLERANCE' in source
    assert 'comparison_rule' in source
    assert '"absolute_only"' in source


def test_z8_verifier_source_disables_cpu_fallback():
    import importlib.util
    spec = importlib.util.spec_from_file_location('phase40_strict_best_rn1', VERIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = VERIFIER_PATH.read_text()
    # Hard requirements: CPU fallback is forbidden and the verifier has a
    # graceful NOT_VERIFIABLE_ON_CURRENT_DEVICE payload.
    assert '"cpu_fallback": "DISABLED"' in source or "'cpu_fallback': 'DISABLED'" in source
    assert 'NOT_VERIFIABLE_ON_CURRENT_DEVICE' in source
    assert '_check_recorded_device_available' in source
    assert '_build_not_verifiable_payload' in source
    assert hasattr(module, '_build_not_verifiable_payload')
    assert hasattr(module, '_check_recorded_device_available')
    # And the explicit prohibition string must still be present.
    assert 'CPU fallback is FORBIDDEN' in source or 'CPU fallback is FORBIDDEN' in module.PLAN_AUTHORITY or 'FORBIDDEN' in source


def test_z9_no_1e_6_tolerance_anywhere_in_verifier():
    source = VERIFIER_PATH.read_text()
    assert '1e-6' not in source
    assert '1e-06' not in source


def test_z10_plan_authority_referenced():
    import importlib.util
    spec = importlib.util.spec_from_file_location('phase40_strict_best_rn1', VERIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert 'phase_40_s18_revin_strict_best_corrective_plan' in module.PLAN_AUTHORITY


def _load_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('phase40_strict_best_rn1', VERIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_rn1_config(module):
    real_run_root = module.Path("/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK") / "artifacts/runs" / module.RUN_ID
    return module.read_json(real_run_root / "config.json")["config"]


def test_z11_external_report_constants_exist():
    module = _load_module()
    assert hasattr(module, 'EXTERNAL_REPORT_PATH')
    assert module.EXTERNAL_REPORT_PATH.name == 'rn1_external_strict_best_report.json'
    assert hasattr(module, '_try_load_external_report')
    assert hasattr(module, '_build_external_accepted_payload')


def test_z12_external_report_validation_rejects_missing_file():
    module = _load_module()
    config = _read_rn1_config(module)
    # Temporarily move external report aside
    ext_path = module.EXTERNAL_REPORT_PATH
    backup = None
    if ext_path.is_file():
        backup = ext_path.read_bytes()
        ext_path.unlink()
    try:
        report, reason = module._try_load_external_report(config)
    finally:
        if backup is not None:
            ext_path.write_bytes(backup)
    assert report is None
    assert 'not present' in reason


def test_z13_external_report_validation_rejects_wrong_tolerance():
    module = _load_module()
    config = _read_rn1_config(module)
    bad_report = {
        "status": "PASS",
        "recorded_device": "mps",
        "cpu_fallback": "DISABLED",
        "comparison_rule": "absolute_only",
        "tolerance": 1e-6,
        "rel_tolerance": None,
        "stored_validation_metrics": {"rmse_wh": 0.0, "mae_wh": 0.0, "r2": 0.0, "n_samples": 0},
        "metric_delta": {"rmse_wh": 0.0, "mae_wh": 0.0, "r2": 0.0},
        "test_access": "FORBIDDEN",
    }
    backup = None
    if module.EXTERNAL_REPORT_PATH.is_file():
        backup = module.EXTERNAL_REPORT_PATH.read_bytes()
    module.EXTERNAL_REPORT_PATH.write_text(json.dumps(bad_report))
    try:
        report, reason = module._try_load_external_report(config)
    finally:
        if backup is not None:
            module.EXTERNAL_REPORT_PATH.write_bytes(backup)
        else:
            module.EXTERNAL_REPORT_PATH.unlink(missing_ok=True)
    assert report is None
    assert 'tolerance' in reason


def test_z14_external_report_validation_rejects_cpu_fallback_allowed():
    module = _load_module()
    config = _read_rn1_config(module)
    bad_report = {
        "status": "PASS",
        "recorded_device": "mps",
        "cpu_fallback": "ALLOWED",
        "comparison_rule": "absolute_only",
        "tolerance": 1e-9,
        "rel_tolerance": None,
        "stored_validation_metrics": {"rmse_wh": 62.28695816170348, "mae_wh": 28.58192738256379, "r2": 0.5440007380016099, "n_samples": 2960},
        "metric_delta": {"rmse_wh": 0.0, "mae_wh": 0.0, "r2": 0.0},
        "test_access": "FORBIDDEN",
    }
    backup = None
    if module.EXTERNAL_REPORT_PATH.is_file():
        backup = module.EXTERNAL_REPORT_PATH.read_bytes()
    module.EXTERNAL_REPORT_PATH.write_text(json.dumps(bad_report))
    try:
        report, reason = module._try_load_external_report(config)
    finally:
        if backup is not None:
            module.EXTERNAL_REPORT_PATH.write_bytes(backup)
        else:
            module.EXTERNAL_REPORT_PATH.unlink(missing_ok=True)
    assert report is None
    assert 'cpu_fallback' in reason


def test_z15_external_report_validation_rejects_stored_metrics_mismatch():
    module = _load_module()
    config = _read_rn1_config(module)
    bad_report = {
        "status": "PASS",
        "recorded_device": "mps",
        "cpu_fallback": "DISABLED",
        "comparison_rule": "absolute_only",
        "tolerance": 1e-9,
        "rel_tolerance": None,
        "stored_validation_metrics": {"rmse_wh": 999.0, "mae_wh": 28.58192738256379, "r2": 0.5440007380016099, "n_samples": 2960},
        "metric_delta": {"rmse_wh": 0.0, "mae_wh": 0.0, "r2": 0.0},
        "test_access": "FORBIDDEN",
    }
    backup = None
    if module.EXTERNAL_REPORT_PATH.is_file():
        backup = module.EXTERNAL_REPORT_PATH.read_bytes()
    module.EXTERNAL_REPORT_PATH.write_text(json.dumps(bad_report))
    try:
        report, reason = module._try_load_external_report(config)
    finally:
        if backup is not None:
            module.EXTERNAL_REPORT_PATH.write_bytes(backup)
        else:
            module.EXTERNAL_REPORT_PATH.unlink(missing_ok=True)
    assert report is None
    assert 'rmse_wh' in reason


def test_z16_external_report_validation_rejects_test_access():
    module = _load_module()
    config = _read_rn1_config(module)
    bad_report = {
        "status": "PASS",
        "recorded_device": "mps",
        "cpu_fallback": "DISABLED",
        "comparison_rule": "absolute_only",
        "tolerance": 1e-9,
        "rel_tolerance": None,
        "stored_validation_metrics": {"rmse_wh": 62.28695816170348, "mae_wh": 28.58192738256379, "r2": 0.5440007380016099, "n_samples": 2960},
        "metric_delta": {"rmse_wh": 0.0, "mae_wh": 0.0, "r2": 0.0},
        "test_access": "ALLOWED",
    }
    backup = None
    if module.EXTERNAL_REPORT_PATH.is_file():
        backup = module.EXTERNAL_REPORT_PATH.read_bytes()
    module.EXTERNAL_REPORT_PATH.write_text(json.dumps(bad_report))
    try:
        report, reason = module._try_load_external_report(config)
    finally:
        if backup is not None:
            module.EXTERNAL_REPORT_PATH.write_bytes(backup)
        else:
            module.EXTERNAL_REPORT_PATH.unlink(missing_ok=True)
    assert report is None
    assert 'test_access' in reason


def test_z17_main_accepts_valid_external_report_when_mps_unavailable():
    """When MPS is unavailable AND a valid external Human-verified report is
    present, main() returns PASS using the externally-verified deltas.
    No silent CPU fallback occurs."""
    module = _load_module()
    import torch
    if torch.backends.mps.is_available():
        pytest.skip("MPS is available; cannot exercise external-report branch")
    config = _read_rn1_config(module)
    if config["runtime"]["device_type"] != "mps":
        pytest.skip("RN1 was not recorded on MPS; nothing to verify on this host")
    # The current EXTERNAL_REPORT_PATH must be valid; otherwise this test
    # cannot exercise the accept path.
    ext_path = module.EXTERNAL_REPORT_PATH
    if not ext_path.is_file():
        pytest.skip("External report not present; cannot exercise accept path")
    status, payload = module.main()
    assert status == "PASS"
    assert payload["strict_best_verification"] == "PASS"
    assert payload["cpu_fallback"] == "DISABLED"
    assert payload["verification_device"] == "mps"
    assert payload["recorded_device"] == "mps"
    assert payload["comparison_rule"] == "absolute_only"
    assert payload["canonical_tolerance"] == 1e-9
    assert payload["rel_tolerance"] is None
    assert payload["external_report_used"] is True
    assert payload["metric_delta"]["rmse_wh"] <= 1e-9
    assert payload["metric_delta"]["mae_wh"] <= 1e-9
    assert payload["metric_delta"]["r2"] <= 1e-9
    assert payload["stored_validation_metrics"]["rmse_wh"] == 62.28695816170348
    assert payload["recomputed_validation_metrics"]["rmse_wh"] == 62.28695816170348
    assert "FORBIDDEN" in payload["test_access"]
