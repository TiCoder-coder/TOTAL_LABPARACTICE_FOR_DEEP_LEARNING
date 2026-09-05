from pathlib import Path

from scripts import run_all_pending


def preview(environment_ready: bool) -> dict:
    return {
        "target_phase": 30,
        "audit_only": True,
        "scientific_artifacts_written": False,
        "environment": {
            "ready_for_training": environment_ready,
            "reasons": [] if environment_ready else ["CUDA_OR_MPS_REQUIRED"],
        },
        "earliest_invalid_phase": 22,
        "required_phases": list(range(22, 31)),
        "direct_target_allowed": False,
        "execution_ready": False,
        "phases": [],
    }


def test_selected_phases_respects_dependency_mode() -> None:
    assert run_all_pending.selected_phases(30, False) == (30,)
    assert run_all_pending.selected_phases(30, True) == tuple(range(23, 31))
    assert run_all_pending.selected_phases(31, True) == (31,)
    assert run_all_pending.selected_phases(32, True) == (32,)
    assert run_all_pending.selected_phases(33, True) == (33,)
    assert run_all_pending.selected_phases(36, True) == (36,)


def test_audit_only_does_not_dispatch_or_write(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_all_pending, "ROOT", tmp_path)
    monkeypatch.setattr(run_all_pending, "recovery_preview", lambda target_phase: preview(False))
    monkeypatch.setattr(
        run_all_pending,
        "run_condition",
        lambda sweep_id, condition_id: (_ for _ in ()).throw(AssertionError("unexpected dispatch")),
    )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = run_all_pending.run_pending(30, True, audit_only=True)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert result["completed_conditions"] == []
    assert result["recovered_phases"] == []
    assert before == after


def test_execution_stops_before_phase_22_when_environment_is_invalid(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_all_pending, "ROOT", tmp_path)
    monkeypatch.setattr(run_all_pending, "recovery_preview", lambda target_phase: preview(False))
    monkeypatch.setattr(run_all_pending, "inspect_phase_22_recovery", lambda root: {"scientifically_reusable": True})
    result = run_all_pending.run_pending(30, True)
    assert result["blocked"] == [{"phase_id": 1, "reasons": ["CUDA_OR_MPS_REQUIRED"]}]
    assert result["failed"] == []


def test_phase_22_recovery_precedes_environment_training_gate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_all_pending, "ROOT", tmp_path)
    monkeypatch.setattr(run_all_pending, "recovery_preview", lambda target_phase: preview(False))
    states = iter(
        [
            {
                "scientifically_reusable": False,
                "upstream_baselines": [
                    {"phase_id": 20, "scientifically_reusable": True},
                    {"phase_id": 21, "scientifically_reusable": True},
                ],
            },
            {"scientifically_reusable": True},
        ]
    )
    monkeypatch.setattr(run_all_pending, "inspect_phase_22_recovery", lambda root: next(states))
    recovered = []
    monkeypatch.setattr(run_all_pending, "recover_phase_22", lambda root: recovered.append(22))
    monkeypatch.setattr(run_all_pending, "_refresh_log", lambda phase_id: f"phase_{phase_id}.json")
    result = run_all_pending.run_pending(30, True)
    assert recovered == [22]
    assert result["recovered_phases"] == [22]
    assert result["blocked"] == [{"phase_id": 1, "reasons": ["CUDA_OR_MPS_REQUIRED"]}]


def test_refresh_log_uses_standard_route_through_phase_30(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_all_pending, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(
        run_all_pending,
        "build_phase_processing_log",
        lambda phase_id, root: calls.append(("build_standard", phase_id, root)) or {"phase_id": phase_id},
    )
    monkeypatch.setattr(
        run_all_pending,
        "save_phase_processing_log",
        lambda log, root: calls.append(("save_standard", log["phase_id"], root))
        or root / "docs/save_log_in_processing/phase_30.json",
    )
    monkeypatch.setattr(
        run_all_pending,
        "materialize_phase_resume_log",
        lambda phase_id, root: (_ for _ in ()).throw(AssertionError("unexpected resume route")),
    )
    assert run_all_pending._refresh_log(30) == "docs/save_log_in_processing/phase_30.json"
    assert calls == [("build_standard", 30, tmp_path), ("save_standard", 30, tmp_path)]


def test_refresh_log_uses_selective_resume_route_for_phase_31(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_all_pending, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(
        run_all_pending,
        "materialize_phase_resume_log",
        lambda phase_id, root: (
            calls.append(("materialize_resume", phase_id, root)) or {"phase_id": phase_id},
            root / "docs/save_log_in_processing/phase_31.json",
        ),
    )
    monkeypatch.setattr(
        run_all_pending,
        "build_phase_processing_log",
        lambda phase_id, root: (_ for _ in ()).throw(AssertionError("unexpected standard route")),
    )
    assert run_all_pending._refresh_log(31) == "docs/save_log_in_processing/phase_31.json"
    assert calls == [("materialize_resume", 31, tmp_path)]
