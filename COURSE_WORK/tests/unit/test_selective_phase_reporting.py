from pathlib import Path

from course_work.reporting import phase_summary
from course_work.reporting.phase_summary import (
    build_phase_resume_log,
    materialize_phase_resume_log,
    render_all_logs_summary,
    render_phase_resume,
    render_phase_resume_log,
)
from course_work.utils.artifacts import read_json


def test_build_phase_resume_log_is_read_only(tmp_path: Path) -> None:
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    log = build_phase_resume_log(30, tmp_path)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert log["status"] == "BLOCKED"
    assert log["summary"]["State"] == "UPSTREAM_INVALID"
    assert log["summary"]["Effective action"] == "BLOCK"
    assert before == after


def test_render_phase_resume_log_is_static_persistent_html(tmp_path: Path) -> None:
    log = build_phase_resume_log(30, tmp_path)
    html = render_phase_resume_log(log).data
    assert "Phase 30 - S8 Learning-Rate Sweep" in html
    assert "LR1" in html
    assert "LR2" in html
    assert "LR3" in html
    assert "UPSTREAM_INVALID" in html
    assert "<script" not in html.lower()
    assert "jupyter.widget" not in html
    assert "http://" not in html
    assert "https://" not in html


def test_render_phase_resume_writes_only_derived_processing_log(tmp_path: Path) -> None:
    html = render_phase_resume(30, tmp_path)
    path = tmp_path / "docs/save_log_in_processing/phase_30_s8_learning_rate_log.json"
    assert path.is_file()
    assert html.data
    log = read_json(path)
    assert log["phase_id"] == 30
    assert log["status"] == "BLOCKED"
    assert log["technical_details"]["effective_action"] == "BLOCK"


def test_all_logs_summary_is_static_html(tmp_path: Path) -> None:
    render_phase_resume(30, tmp_path)
    html = render_all_logs_summary(tmp_path).data
    assert "Phase Processing Logs" in html
    assert "Phase 30" not in html
    assert "S8 Learning-Rate Sweep" in html
    assert "BLOCKED" in html
    assert "<script" not in html.lower()
    assert "jupyter.widget" not in html
    assert "http://" not in html
    assert "https://" not in html


def test_all_logs_summary_uses_valid_canonical_scientific_status(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "docs/save_log_in_processing/phase_31_s9_weight_decay_log.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"phase_id":31,"phase_name":"S9 Weight-Decay Sweep","status":"VALID_REUSABLE"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        phase_summary,
        "inspect_phase_state",
        lambda phase_id, root: {"signoff": {"valid": True, "record": {"status": "PASS"}}},
    )
    html = render_all_logs_summary(tmp_path).data
    assert "VALID_REUSABLE" not in html
    assert ">PASS<" in html
    assert "Needs attention: 0" in html


def test_phase_31_log_contains_static_s9_contract_and_exact_block_reasons(tmp_path: Path) -> None:
    log = build_phase_resume_log(31, tmp_path)
    html = render_phase_resume_log(log).data
    assert log["status"] == "BLOCKED"
    assert log["summary"]["Test access"] == "FORBIDDEN"
    assert "S9 frozen contract" in html
    assert "WD0 = 0.0; WD2 = 0.001" in html
    assert "s8_learning_rate_winner.json: MISSING" in html
    assert "<script" not in html.lower()
    assert "jupyter.widget" not in html


def test_render_phase_31_writes_only_derived_log(tmp_path: Path) -> None:
    render_phase_resume(31, tmp_path)
    paths = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file())
    assert paths == [Path("docs/save_log_in_processing/phase_31_s9_weight_decay_log.json")]
    log = read_json(tmp_path / paths[0])
    assert log["phase_id"] == 31
    assert log["technical_details"]["phase_31_preflight"]["ready"] is False


def test_phase_32_log_contains_static_s10_contract_and_exact_block_reasons(tmp_path: Path) -> None:
    log = build_phase_resume_log(32, tmp_path)
    html = render_phase_resume_log(log).data
    assert log["status"] == "BLOCKED"
    assert log["summary"]["Test access"] == "FORBIDDEN"
    assert "S10 frozen contract" in html
    assert "DR02 = 0.2; DR03 = 0.3" in html
    assert "s9_weight_decay_winner.json: MISSING" in html
    assert "jupyter.widget" not in html
    assert "<script" not in html.lower()


def test_render_phase_32_writes_only_derived_log(tmp_path: Path) -> None:
    render_phase_resume(32, tmp_path)
    paths = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file())
    assert paths == [Path("docs/save_log_in_processing/phase_32_s10_dropout_log.json")]
    log = read_json(tmp_path / paths[0])
    assert log["phase_id"] == 32
    assert log["technical_details"]["phase_32_preflight"]["ready"] is False


def test_phase_33_log_contains_static_s11_contract_and_exact_block_reasons(tmp_path: Path) -> None:
    log = build_phase_resume_log(33, tmp_path)
    html = render_phase_resume_log(log).data
    assert log["status"] == "BLOCKED"
    assert log["summary"]["Test access"] == "FORBIDDEN"
    assert "S11 frozen contract" in html
    assert "D32 = 32" in html
    assert "s10_dropout_winner.json: MISSING" in html
    assert "jupyter.widget" not in html
    assert "<script" not in html.lower()


def test_render_phase_33_writes_only_derived_log(tmp_path: Path) -> None:
    render_phase_resume(33, tmp_path)
    paths = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file())
    assert paths == [Path("docs/save_log_in_processing/phase_33_s11_d_model_log.json")]
    log = read_json(tmp_path / paths[0])
    assert log["phase_id"] == 33
    assert log["technical_details"]["phase_33_preflight"]["ready"] is False


def test_phase_34_log_contains_static_s12_contract_and_exact_block_reasons(tmp_path: Path) -> None:
    log = build_phase_resume_log(34, tmp_path)
    html = render_phase_resume_log(log).data
    assert log["status"] == "BLOCKED"
    assert log["summary"]["Test access"] == "FORBIDDEN"
    assert "S12 frozen contract" in html
    assert "H2 = 2" in html
    assert "s11_d_model_winner.json: MISSING" in html
    assert "jupyter.widget" not in html
    assert "<script" not in html.lower()


def test_render_phase_34_writes_only_derived_log(tmp_path: Path) -> None:
    render_phase_resume(34, tmp_path)
    paths = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file())
    assert paths == [Path("docs/save_log_in_processing/phase_34_s12_head_log.json")]
    log = read_json(tmp_path / paths[0])
    assert log["phase_id"] == 34
    assert log["technical_details"]["phase_34_preflight"]["ready"] is False


def test_materialize_phase_resume_log_rebuilds_stale_state_once(tmp_path: Path, monkeypatch) -> None:
    logs = iter(
        [
            {"phase_id": 31, "status": "LOG_STALE"},
            {"phase_id": 31, "status": "VALID_REUSABLE"},
        ]
    )
    saved = []
    monkeypatch.setattr(phase_summary, "build_phase_resume_log", lambda phase_id, root, allow_execution=False: next(logs))
    monkeypatch.setattr(
        phase_summary,
        "save_phase_resume_log",
        lambda log, root: saved.append(log["status"])
        or root / "docs/save_log_in_processing/phase_31_s9_weight_decay_log.json",
    )
    log, path = materialize_phase_resume_log(31, tmp_path)
    assert log["status"] == "VALID_REUSABLE"
    assert saved == ["LOG_STALE", "VALID_REUSABLE"]
    assert path.name == "phase_31_s9_weight_decay_log.json"


def test_materialize_phase_resume_log_keeps_blocked_state_single_pass(tmp_path: Path, monkeypatch) -> None:
    saved = []
    monkeypatch.setattr(
        phase_summary,
        "build_phase_resume_log",
        lambda phase_id, root, allow_execution=False: {"phase_id": phase_id, "status": "BLOCKED"},
    )
    monkeypatch.setattr(
        phase_summary,
        "save_phase_resume_log",
        lambda log, root: saved.append(log["status"])
        or root / "docs/save_log_in_processing/phase_31_s9_weight_decay_log.json",
    )
    log, _ = materialize_phase_resume_log(31, tmp_path)
    assert log["status"] == "BLOCKED"
    assert saved == ["BLOCKED"]
