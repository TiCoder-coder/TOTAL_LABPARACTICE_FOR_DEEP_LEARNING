import hashlib
import json
from pathlib import Path

import pytest

from scripts import recover_phase_13_to_21


def tree_state(root: Path, logical_root: str) -> dict:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        logical = (Path(logical_root) / path.relative_to(root)).as_posix()
        digest.update(logical.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\0")
    return {
        "file_count": len(files),
        "total_size_bytes": sum(path.stat().st_size for path in files),
        "tree_sha256": digest.hexdigest(),
    }


def create_project(root: Path) -> dict:
    roots = []
    for relative in recover_phase_13_to_21.ACTIVE_ROOTS:
        path = root / relative
        path.mkdir(parents=True)
        (path / "evidence.txt").write_text(relative, encoding="utf-8")
        roots.append({"path": relative, **tree_state(path, relative)})
    environment = root / "artifacts/environment"
    environment.mkdir(parents=True)
    (environment / "environment_report.json").write_text(
        json.dumps(
            {
                "environment_revision_id": "ENV-R-TEST",
                "python_version": "3.10.11",
                "selected_device": "mps",
                "mps_available": True,
                "cuda_available": False,
            }
        ),
        encoding="utf-8",
    )
    (environment / "phase_1_signoff.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    roots.append({"path": "artifacts/environment", **tree_state(environment, "artifacts/environment")})
    notebook = root / "notebook_course_work/CourseWork.ipynb"
    notebook.parent.mkdir(parents=True)
    notebook.write_text("{}", encoding="utf-8")
    manifest = {
        "artifact_version": "PHASE-13-21-RECOVERY-PRESERVATION-v1",
        "recovery_revision_id": "RECOVERY-TEST",
        "roots": roots,
        "notebook": {
            "path": "notebook_course_work/CourseWork.ipynb",
            "size_bytes": notebook.stat().st_size,
            "sha256": hashlib.sha256(notebook.read_bytes()).hexdigest(),
        },
        "status": "PASS",
    }
    path = root / recover_phase_13_to_21.PRESERVATION_MANIFEST
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_audit_only_preserves_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    create_project(tmp_path)
    monkeypatch.setattr(
        recover_phase_13_to_21,
        "_phase_state",
        lambda root, phase_id: {"phase_id": phase_id, "state": "INVALID"},
    )
    before = sorted(
        (path.relative_to(tmp_path).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    result = recover_phase_13_to_21.audit_recovery(tmp_path)
    after = sorted(
        (path.relative_to(tmp_path).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert result["preservation_state"] == "READY_TO_PRESERVE"
    assert result["scientific_artifacts_written"] is False
    assert before == after


def test_preserve_evidence_is_revisioned_and_reusable(tmp_path: Path) -> None:
    manifest = create_project(tmp_path)
    result = recover_phase_13_to_21.preserve_evidence(tmp_path)
    history = tmp_path / "artifacts/_history" / manifest["recovery_revision_id"]
    assert result["status"] == "PASS"
    for relative in recover_phase_13_to_21.ACTIVE_ROOTS:
        assert not (tmp_path / relative).exists()
        assert recover_phase_13_to_21._history_path(history, relative).is_dir()
    assert recover_phase_13_to_21.preserve_evidence(tmp_path) == result


def test_preservation_rejects_changed_evidence(tmp_path: Path) -> None:
    create_project(tmp_path)
    path = tmp_path / recover_phase_13_to_21.ACTIVE_ROOTS[0] / "evidence.txt"
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="preservation mismatch"):
        recover_phase_13_to_21.preserve_evidence(tmp_path)


def test_parser_requires_exactly_one_mode() -> None:
    with pytest.raises(SystemExit):
        recover_phase_13_to_21.build_parser().parse_args([])
    preserve = recover_phase_13_to_21.build_parser().parse_args(["--preserve-only"])
    assert preserve.preserve_only is True
    args = recover_phase_13_to_21.build_parser().parse_args(["--execute-through", "19"])
    assert args.execute_through == 19
