import json
from pathlib import Path

import pytest

from course_work.utils import environment
from course_work.utils.artifacts import read_json, sha256_file


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def inventory(root: Path, accelerator: bool) -> dict:
    return {
        "environment_id": "ENV-v1",
        "python_version": "3.10.11",
        "python_executable": "/current/python",
        "project_root": str(root.resolve()),
        "kernel": {"matches_interpreter": True},
        "default_dtype": "torch.float32",
        "cuda_available": accelerator,
        "mps_available": False,
        "selected_device": "cuda" if accelerator else "cpu",
        "development_seed": 42,
        "package_versions": {"torch": "test", "scikit_learn": "test"},
        "platform": "test",
    }


def create_existing_environment(root: Path) -> None:
    write_json(root / "artifacts/contracts/phase_0_signoff.json", {"status": "PASS"})
    report_path = root / "artifacts/environment/environment_report.json"
    freeze_path = root / "artifacts/environment/requirements_freeze.txt"
    smoke_path = root / "artifacts/environment/smoke_test_report.json"
    write_json(report_path, {"environment_id": "OLD"})
    freeze_path.write_text("old\n", encoding="utf-8")
    write_json(smoke_path, {"status": "PASS"})
    output_paths = [
        "artifacts/environment/environment_report.json",
        "artifacts/environment/requirements_freeze.txt",
        "artifacts/environment/smoke_test_report.json",
    ]
    write_json(
        root / "artifacts/environment/phase_1_signoff.json",
        {
            "phase_id": 1,
            "phase_version": "PHASE-1-v1",
            "status": "PASS",
            "output_paths": output_paths,
            "output_checksums": {
                relative_path: sha256_file(root / relative_path)
                for relative_path in output_paths
            },
        },
    )


def test_existing_environment_validation_does_not_require_runtime_accelerator(tmp_path: Path, monkeypatch) -> None:
    create_existing_environment(tmp_path)
    monkeypatch.setattr(environment, "materialize_phase_0", lambda root: {"status": "PASS"})
    monkeypatch.setattr(environment, "environment_inventory", lambda root: (_ for _ in ()).throw(AssertionError("runtime inventory called")))
    monkeypatch.setattr(environment, "device_smoke_test", lambda: (_ for _ in ()).throw(AssertionError("smoke test called")))
    signoff = environment.materialize_phase_1(tmp_path)
    assert signoff["status"] == "PASS"


def test_environment_recovery_preserves_historical_evidence(tmp_path: Path, monkeypatch) -> None:
    create_existing_environment(tmp_path)
    monkeypatch.setattr(environment, "environment_inventory", lambda root: inventory(root, True))
    monkeypatch.setattr(environment, "device_smoke_test", lambda: {"status": "PASS"})
    monkeypatch.setattr(environment, "dependency_freeze", lambda: "new\n")
    signoff = environment.recover_environment_revision(tmp_path)
    report = read_json(tmp_path / "artifacts/environment/environment_report.json")
    assert signoff["status"] == "PASS"
    assert report["environment_revision_id"] == signoff["environment_revision_id"]
    assert len(report["historical_environment_paths"]) == 4
    assert all((tmp_path / path).is_file() for path in report["historical_environment_paths"])
    for relative_path, expected in signoff["output_checksums"].items():
        assert sha256_file(tmp_path / relative_path) == expected


def test_environment_recovery_refuses_cpu_before_archiving(tmp_path: Path, monkeypatch) -> None:
    create_existing_environment(tmp_path)
    old_hash = sha256_file(tmp_path / "artifacts/environment/environment_report.json")
    monkeypatch.setattr(environment, "environment_inventory", lambda root: inventory(root, False))
    with pytest.raises(RuntimeError, match="requires CUDA or MPS"):
        environment.recover_environment_revision(tmp_path)
    assert sha256_file(tmp_path / "artifacts/environment/environment_report.json") == old_hash
    assert not (tmp_path / "artifacts/environment/_history").exists()
