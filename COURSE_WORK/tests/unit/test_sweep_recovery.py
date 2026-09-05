import json
import hashlib
from pathlib import Path

import pytest

from course_work.experiments import sweep_recovery
from course_work.experiments.sweep_recovery import (
    inspect_environment_recovery,
    inspect_phase_22_recovery,
    inspect_recovery_chain,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def runtime_inventory(root: Path, accelerator: bool = True) -> dict:
    return {
        "python_version": "3.10.11",
        "python_executable": "/current/python",
        "project_root": str(root.resolve()),
        "kernel": {
            "kernel_name": "python3",
            "kernel_executable": "/current/python",
            "matches_interpreter": True,
        },
        "selected_device": "mps" if accelerator else "cpu",
        "cuda_available": False,
        "mps_available": accelerator,
    }


def test_environment_recovery_reports_identity_fields_separately(tmp_path: Path, monkeypatch) -> None:
    write_json(
        tmp_path / "artifacts/environment/environment_report.json",
        {
            "python_version": "3.10.21",
            "python_executable": "/signed/python",
            "project_root": "/signed/project",
        },
    )
    write_json(
        tmp_path / "artifacts/environment/phase_1_signoff.json",
        {"status": "PASS", "output_paths": [], "output_checksums": {}},
    )
    monkeypatch.setattr(sweep_recovery, "environment_inventory", lambda root: runtime_inventory(root))
    result = inspect_environment_recovery(tmp_path)
    fields = {item["field"]: item for item in result["identity_comparisons"]}
    assert fields["python_version"]["matches"] is False
    assert fields["python_executable"]["matches"] is False
    assert fields["project_root"]["matches"] is False
    assert result["signed_executable_exists"] is False
    assert result["ready_for_training"] is False


def test_environment_recovery_requires_accelerator(tmp_path: Path, monkeypatch) -> None:
    current = runtime_inventory(tmp_path, accelerator=False)
    executable = tmp_path / "python"
    executable.write_text("runtime", encoding="utf-8")
    current["python_executable"] = str(executable.resolve())
    current["kernel"]["kernel_executable"] = str(executable.resolve())
    write_json(
        tmp_path / "artifacts/environment/environment_report.json",
        {
            "python_version": current["python_version"],
            "python_executable": current["python_executable"],
            "project_root": current["project_root"],
        },
    )
    write_json(
        tmp_path / "artifacts/environment/phase_1_signoff.json",
        {"status": "PASS", "output_paths": [], "output_checksums": {}},
    )
    monkeypatch.setattr(sweep_recovery, "environment_inventory", lambda root: current)
    result = inspect_environment_recovery(tmp_path)
    assert result["ready_for_training"] is False
    assert "CUDA_OR_MPS_REQUIRED" in result["reasons"]


def test_phase_22_requires_summary_csv(tmp_path: Path) -> None:
    write_json(
        tmp_path / "artifacts/lstm_baseline/phase_20_signoff.json",
        {"status": "PASS", "output_paths": [], "output_checksums": {}},
    )
    write_json(
        tmp_path / "artifacts/transformer_b0/phase_21_signoff.json",
        {"status": "PASS", "output_paths": [], "output_checksums": {}},
    )
    write_json(
        tmp_path / "artifacts/learning_diagnostics/phase_22_signoff.json",
        {"status": "PASS", "output_paths": [], "output_checksums": {}},
    )
    result = inspect_phase_22_recovery(tmp_path)
    assert result["state"] == "CANONICAL_EVIDENCE_INVALID"
    assert result["required_action"] == "REPAIR_PHASE_22"
    assert result["issues"] == [
        {
            "path": "artifacts/learning_diagnostics/learning_diagnostics_summary.csv",
            "reason": "MISSING",
        }
    ]


def test_phase_22_accepts_existing_output_without_self_checksum(tmp_path: Path) -> None:
    for relative in (
        "artifacts/lstm_baseline/phase_20_signoff.json",
        "artifacts/transformer_b0/phase_21_signoff.json",
    ):
        write_json(
            tmp_path / relative,
            {"status": "PASS", "output_paths": [relative], "output_checksums": {}},
        )
    result = inspect_phase_22_recovery(tmp_path)
    assert all(item["scientifically_reusable"] for item in result["upstream_baselines"])
    assert result["required_action"] == "REPAIR_PHASE_22"


def test_phase_22_rejects_declared_checksum_mismatch(tmp_path: Path) -> None:
    for relative in (
        "artifacts/lstm_baseline/phase_20_signoff.json",
        "artifacts/transformer_b0/phase_21_signoff.json",
    ):
        path = tmp_path / relative
        write_json(path, {"status": "PASS", "output_paths": [], "output_checksums": {}})
    output = tmp_path / "artifacts/lstm_baseline/evidence.json"
    write_json(output, {"value": 1})
    path = tmp_path / "artifacts/lstm_baseline/phase_20_signoff.json"
    write_json(
        path,
        {
            "status": "PASS",
            "output_paths": ["artifacts/lstm_baseline/evidence.json"],
            "output_checksums": {"artifacts/lstm_baseline/evidence.json": hashlib.sha256(b"wrong").hexdigest()},
        },
    )
    result = inspect_phase_22_recovery(tmp_path)
    phase_20 = next(item for item in result["upstream_baselines"] if item["phase_id"] == 20)
    assert phase_20["scientifically_reusable"] is False
    assert phase_20["issues"] == [
        {"path": "artifacts/lstm_baseline/evidence.json", "reason": "CHECKSUM_MISMATCH"}
    ]


def test_recovery_chain_is_read_only_and_starts_at_first_invalid_baseline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sweep_recovery, "environment_inventory", lambda root: runtime_inventory(root, accelerator=False))
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = inspect_recovery_chain(30, tmp_path)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert result["earliest_invalid_phase"] == 20
    assert result["required_phases"] == list(range(20, 31))
    assert result["direct_target_allowed"] is False
    assert result["scientific_artifacts_written"] is False
    assert before == after


def test_recovery_chain_accepts_phase_34_target(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sweep_recovery, "environment_inventory", lambda root: runtime_inventory(root, accelerator=False))
    result = inspect_recovery_chain(34, tmp_path)
    assert result["target_phase"] == 34
    assert result["required_phases"] == list(range(20, 35))
    assert result["scientific_artifacts_written"] is False


def test_recovery_chain_rejects_phase_35_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Phase 23 and Phase 34"):
        inspect_recovery_chain(35, tmp_path)
