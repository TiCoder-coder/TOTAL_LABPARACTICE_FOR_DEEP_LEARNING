from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from course_work.experiments.phase_execution import (
    LEGACY_SWEEP_CONTRACTS,
    _validate_legacy_sweep_contract,
    inspect_phase_state,
)
from course_work.reporting.phase_summary import build_phase_resume_log
from course_work.utils.artifacts import get_project_root


SOURCE_ROOT = Path(get_project_root()).resolve()


def _copy_contract_fixture(tmp_path: Path, phase_id: int) -> dict:
    contract = LEGACY_SWEEP_CONTRACTS[phase_id]
    signoff_relative = (
        "artifacts/sweeps/S15_loss/phase_37_signoff.json"
        if phase_id == 37
        else "artifacts/sweeps/S16_epoch_cap/phase_38_signoff.json"
    )
    paths = [
        signoff_relative,
        *contract["historical_files"],
        *contract["core_artifacts"],
    ]
    for relative_path in paths:
        source = SOURCE_ROOT / relative_path
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    registry_record = None
    for line in (SOURCE_ROOT / "artifacts/experiments/experiment_registry.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        record = json.loads(line)
        if record.get("run_id") == contract["run_id"]:
            registry_record = record
            break
    assert registry_record is not None
    registry_path = tmp_path / "artifacts/experiments/experiment_registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry_record, sort_keys=True) + "\n", encoding="utf-8")
    return json.loads((tmp_path / signoff_relative).read_text(encoding="utf-8"))


@pytest.mark.parametrize("phase_id", [37, 38])
def test_exact_legacy_contract_and_registry_artifacts_are_accepted(
    tmp_path: Path, phase_id: int
) -> None:
    signoff = _copy_contract_fixture(tmp_path, phase_id)
    result = _validate_legacy_sweep_contract(tmp_path, phase_id, signoff)
    assert result["valid"] is True
    assert result["issues"] == []
    assert result["test_status"] == "NOT_ACCESSED"


@pytest.mark.parametrize("phase_id", [37, 38])
def test_altered_legacy_signoff_hash_fails(tmp_path: Path, phase_id: int) -> None:
    signoff = _copy_contract_fixture(tmp_path, phase_id)
    signoff["audit_tamper"] = True
    signoff_path = tmp_path / (
        "artifacts/sweeps/S15_loss/phase_37_signoff.json"
        if phase_id == 37
        else "artifacts/sweeps/S16_epoch_cap/phase_38_signoff.json"
    )
    signoff_path.write_text(json.dumps(signoff), encoding="utf-8")
    result = _validate_legacy_sweep_contract(tmp_path, phase_id, signoff)
    assert result["valid"] is False
    assert "LEGACY_SIGNOFF_CHECKSUM_MISMATCH" in {item["reason"] for item in result["issues"]}


@pytest.mark.parametrize("phase_id", [37, 38])
@pytest.mark.parametrize("artifact_kind", ["checkpoint", "predictions", "metrics"])
def test_altered_core_scientific_artifact_fails(
    tmp_path: Path, phase_id: int, artifact_kind: str
) -> None:
    signoff = _copy_contract_fixture(tmp_path, phase_id)
    relative_path = next(
        path
        for path in LEGACY_SWEEP_CONTRACTS[phase_id]["core_artifacts"]
        if artifact_kind in path
    )
    with (tmp_path / relative_path).open("ab") as handle:
        handle.write(b"tamper")
    result = _validate_legacy_sweep_contract(tmp_path, phase_id, signoff)
    assert result["valid"] is False
    assert {"path": relative_path, "reason": "CHECKSUM_MISMATCH"} in result["issues"]


@pytest.mark.parametrize("phase_id", [37, 38])
def test_missing_canonical_core_artifact_fails(tmp_path: Path, phase_id: int) -> None:
    signoff = _copy_contract_fixture(tmp_path, phase_id)
    relative_path = next(
        path
        for path in LEGACY_SWEEP_CONTRACTS[phase_id]["core_artifacts"]
        if "best_checkpoint.pt" in path
    )
    (tmp_path / relative_path).unlink()
    result = _validate_legacy_sweep_contract(tmp_path, phase_id, signoff)
    assert result["valid"] is False
    assert {"path": relative_path, "reason": "MISSING_CANONICAL_CORE_ARTIFACT"} in result["issues"]


def test_unrelated_phase_cannot_use_legacy_contract(tmp_path: Path) -> None:
    result = _validate_legacy_sweep_contract(
        tmp_path,
        42,
        {"phase": 42, "overall_status": "PASS"},
    )
    assert result["valid"] is False
    assert result["reporting_debt"] == []
    assert result["issues"][0]["reason"] == "LEGACY_CONTRACT_NOT_ALLOW_LISTED"


@pytest.mark.parametrize("phase_id", [37, 38])
def test_only_allow_listed_absent_reporting_files_are_debt(
    tmp_path: Path, phase_id: int
) -> None:
    signoff = _copy_contract_fixture(tmp_path, phase_id)
    result = _validate_legacy_sweep_contract(tmp_path, phase_id, signoff)
    assert result["valid"] is True
    assert set(result["reporting_debt"]) == set(LEGACY_SWEEP_CONTRACTS[phase_id]["reporting_debt"])

    protected_path = next(iter(LEGACY_SWEEP_CONTRACTS[phase_id]["historical_files"]))
    (tmp_path / protected_path).unlink()
    failed = _validate_legacy_sweep_contract(tmp_path, phase_id, signoff)
    assert failed["valid"] is False
    assert protected_path not in failed["reporting_debt"]


@pytest.mark.parametrize("phase_id", [37, 38])
def test_current_legacy_phase_is_reusable_and_rebuilt_log_is_traceable(phase_id: int) -> None:
    state = inspect_phase_state(phase_id, SOURCE_ROOT)
    assert state["state"] == "VALID_REUSABLE"
    assert state["action"] == "RENDER_ONLY"
    assert state["signoff"]["legacy_contract"]["valid"] is True

    log = build_phase_resume_log(phase_id, SOURCE_ROOT, allow_execution=False)
    assert log["status"] == "VALID_REUSABLE"
    assert log["discrepancies"] == []
    source_paths = {item["path"] for item in log["source_artifacts"]}
    contract = LEGACY_SWEEP_CONTRACTS[phase_id]
    assert set(contract["historical_files"]).issubset(source_paths)
    assert set(contract["core_artifacts"]).issubset(source_paths)
    assert "record" not in log["technical_details"]["inspection"]["processing_log"]
    phase_37_preflight = log["technical_details"]["phase_37_preflight"]
    if isinstance(phase_37_preflight, dict):
        assert "decision" not in phase_37_preflight
    debt_section = next(
        section for section in log["sections"] if section["title"] == "Historical reporting debt"
    )
    assert {row["Path"] for row in debt_section["rows"]} == set(contract["reporting_debt"])
