from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from course_work.experiments.phase_execution import (
    LEGACY_SWEEP_CONTRACTS,
    _validate_legacy_sweep_contract,
    inspect_phase_state,
    resolve_phase_conditions,
)
from course_work.reporting.phase_summary import build_phase_resume_log
from course_work.utils.artifacts import get_project_root


SOURCE_ROOT = Path(get_project_root()).resolve()
SIGNOFF_PATHS = {
    39: "artifacts/sweeps/S17_gradient_clipping/phase_39_signoff.json",
    40: "artifacts/sweeps/S18_revin/phase_40_signoff.json",
    41: "artifacts/sweeps/S19_boundary_protocol/phase_41_signoff.json",
}


def _copy_contract_fixture(tmp_path: Path, phase_id: int) -> dict:
    contract = LEGACY_SWEEP_CONTRACTS[phase_id]
    paths = [
        SIGNOFF_PATHS[phase_id],
        *contract["historical_files"],
        *contract["core_artifacts"],
    ]
    for relative_path in paths:
        source = SOURCE_ROOT / relative_path
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    run_ids = {item[0] for item in contract["canonical_runs"]}
    registry_records = []
    for line in (SOURCE_ROOT / "artifacts/experiments/experiment_registry.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        record = json.loads(line)
        if record.get("run_id") in run_ids:
            registry_records.append(record)
    assert {record["run_id"] for record in registry_records} == run_ids
    registry_path = tmp_path / "artifacts/experiments/experiment_registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in registry_records),
        encoding="utf-8",
    )
    return json.loads((tmp_path / SIGNOFF_PATHS[phase_id]).read_text(encoding="utf-8"))


@pytest.mark.parametrize("phase_id", [39, 40, 41])
def test_exact_contract_is_accepted(tmp_path: Path, phase_id: int) -> None:
    signoff = _copy_contract_fixture(tmp_path, phase_id)
    result = _validate_legacy_sweep_contract(tmp_path, phase_id, signoff)
    assert result["valid"] is True
    assert result["issues"] == []
    assert result["test_status"] == "NOT_ACCESSED"


@pytest.mark.parametrize("phase_id", [39, 40, 41])
def test_altered_signoff_fails_closed(tmp_path: Path, phase_id: int) -> None:
    signoff = _copy_contract_fixture(tmp_path, phase_id)
    signoff["audit_tamper"] = True
    (tmp_path / SIGNOFF_PATHS[phase_id]).write_text(json.dumps(signoff), encoding="utf-8")
    result = _validate_legacy_sweep_contract(tmp_path, phase_id, signoff)
    assert result["valid"] is False
    assert "LEGACY_SIGNOFF_CHECKSUM_MISMATCH" in {
        item["reason"] for item in result["issues"]
    }


@pytest.mark.parametrize("phase_id", [39, 40, 41])
@pytest.mark.parametrize("artifact_kind", ["checkpoint", "predictions", "metrics"])
def test_altered_scientific_core_fails_closed(
    tmp_path: Path, phase_id: int, artifact_kind: str
) -> None:
    signoff = _copy_contract_fixture(tmp_path, phase_id)
    relative_path = next(
        path
        for path in LEGACY_SWEEP_CONTRACTS[phase_id]["core_artifacts"]
        if LEGACY_SWEEP_CONTRACTS[phase_id]["run_id"] in path and artifact_kind in path
    )
    with (tmp_path / relative_path).open("ab") as handle:
        handle.write(b"tamper")
    result = _validate_legacy_sweep_contract(tmp_path, phase_id, signoff)
    assert result["valid"] is False
    assert {"path": relative_path, "reason": "CHECKSUM_MISMATCH"} in result["issues"]


@pytest.mark.parametrize("phase_id", [39, 40, 41])
def test_missing_scientific_core_fails_closed(tmp_path: Path, phase_id: int) -> None:
    signoff = _copy_contract_fixture(tmp_path, phase_id)
    relative_path = next(
        path
        for path in LEGACY_SWEEP_CONTRACTS[phase_id]["core_artifacts"]
        if LEGACY_SWEEP_CONTRACTS[phase_id]["run_id"] in path and "best_checkpoint.pt" in path
    )
    (tmp_path / relative_path).unlink()
    result = _validate_legacy_sweep_contract(tmp_path, phase_id, signoff)
    assert result["valid"] is False
    assert {"path": relative_path, "reason": "MISSING_CANONICAL_CORE_ARTIFACT"} in result[
        "issues"
    ]


@pytest.mark.parametrize("phase_id", [39, 40, 41])
@pytest.mark.parametrize(
    ("field", "value", "expected_reason"),
    [
        ("status", "FAILED", "CANONICAL_RUN_NOT_COMPLETED"),
        ("test_access_authorized", True, "TEST_FIREWALL_NOT_VERIFIED"),
    ],
)
def test_canonical_registry_status_and_test_firewall_fail_closed(
    tmp_path: Path,
    phase_id: int,
    field: str,
    value: object,
    expected_reason: str,
) -> None:
    signoff = _copy_contract_fixture(tmp_path, phase_id)
    registry_path = tmp_path / "artifacts/experiments/experiment_registry.jsonl"
    records = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines()]
    canonical_run_id = LEGACY_SWEEP_CONTRACTS[phase_id]["run_id"]
    for record in records:
        if record["run_id"] == canonical_run_id:
            record[field] = value
    registry_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    result = _validate_legacy_sweep_contract(tmp_path, phase_id, signoff)
    assert result["valid"] is False
    assert expected_reason in {item["reason"] for item in result["issues"]}


@pytest.mark.parametrize("phase_id", [39, 40, 41])
def test_legacy_conditions_use_only_locked_runs(phase_id: int) -> None:
    conditions = resolve_phase_conditions(phase_id, SOURCE_ROOT)
    contract = LEGACY_SWEEP_CONTRACTS[phase_id]
    assert conditions["complete"] is True
    assert conditions["missing_conditions"] == []
    assert {item["run_id"] for item in conditions["verified_conditions"]} == {
        item[2] for item in contract["condition_evidence"]
    }
    assert all(
        item["evidence_mode"] == "COMMIT_AND_REGISTRY_BOUND_LEGACY_EVIDENCE"
        for item in conditions["verified_conditions"]
    )


@pytest.mark.parametrize("phase_id", [39, 40, 41])
def test_current_phase_is_reusable_and_log_is_traceable(phase_id: int) -> None:
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
    debt_section = next(
        section for section in log["sections"] if section["title"] == "Historical reporting debt"
    )
    assert {row["Path"] for row in debt_section["rows"]} == set(contract["reporting_debt"])
